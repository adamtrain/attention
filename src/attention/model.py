"""A tiny GPT: one transformer block, written out by hand in NumPy.

Every layer is a pair of functions. The forward function computes the layer's output. The
backward function takes how much the loss would change if each output number moved a little
(the gradient) and works out the same for the layer's inputs and weights. That is the chain
rule, applied one layer at a time, from the loss back to the embeddings: backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

type Array = np.ndarray

EPS = 1e-5


@dataclass(frozen=True, slots=True)
class Config:
    vocab: int  # how many different tokens there are
    context: int  # the longest sequence the model can read
    width: int = 16  # numbers per token vector
    heads: int = 2  # attention heads, side by side
    hidden: int = 64  # width of the MLP's middle layer

    @property
    def head_width(self) -> int:
        return self.width // self.heads


@dataclass(frozen=True, slots=True)
class Part:
    """What a group of weights is called, for people."""

    label: str
    layer: str


PARTS: dict[str, Part] = {
    "embed.token": Part("token embeddings", "embeddings"),
    "embed.position": Part("position embeddings", "embeddings"),
    "attn.norm": Part("attention norm", "attention"),
    "attn.query": Part("queries", "attention"),
    "attn.key": Part("keys", "attention"),
    "attn.value": Part("values", "attention"),
    "attn.out": Part("attention output", "attention"),
    "mlp.norm": Part("MLP norm", "mlp"),
    "mlp.up": Part("MLP up", "mlp"),
    "mlp.down": Part("MLP down", "mlp"),
    "head.norm": Part("final norm", "head"),
    "head.unembed": Part("unembedding", "head"),
}

LAYERS = ("embeddings", "attention", "mlp", "head")


def init_params(config: Config, rng: np.random.Generator) -> dict[str, Array]:
    """Random starting weights. Training will shape every one of them."""
    d, h, v = config.width, config.hidden, config.vocab

    def weights(rows: int, cols: int, scale: float = 1.0) -> Array:
        return rng.normal(0.0, scale / np.sqrt(rows), (rows, cols))

    return {
        "embed.token": rng.normal(0.0, 0.5, (v, d)),
        "embed.position": rng.normal(0.0, 0.5, (config.context, d)),
        "attn.norm": np.ones(d),
        "attn.query": weights(d, d),
        "attn.key": weights(d, d),
        "attn.value": weights(d, d),
        "attn.out": weights(d, d, 0.5),
        "mlp.norm": np.ones(d),
        "mlp.up": weights(d, h),
        "mlp.down": weights(h, d, 0.5),
        "head.norm": np.ones(d),
        # Small, so the untrained model starts out finding every token equally likely.
        "head.unembed": rng.normal(0.0, 0.02, (d, v)),
    }


# ── The pieces ────────────────────────────────────────────────────────────────


def softmax(x: Array, axis: int = -1) -> Array:
    """Turn any numbers into probabilities: exponentiate each, then divide by the total."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))  # subtracting the max avoids overflow
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(dp: Array, p: Array, axis: int = -1) -> Array:
    """Nudging one input of a softmax moves every output, because they all share the total."""
    return p * (dp - (dp * p).sum(axis=axis, keepdims=True))


def linear_backward(dy: Array, x: Array, w: Array) -> tuple[Array, Array]:
    """y = x @ w. Each weight's gradient is its input times the gradient that reached it."""
    dx = dy @ w.T
    dw = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    return dx, dw


def rmsnorm(x: Array, gain: Array) -> tuple[Array, Array]:
    """Rescale each vector to a typical size of 1, so no layer's numbers run away."""
    inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + EPS)
    return x * inv * gain, inv


def rmsnorm_backward(dy: Array, x: Array, inv: Array, gain: Array) -> tuple[Array, Array]:
    dgain = (dy * x * inv).reshape(-1, x.shape[-1]).sum(axis=0)
    dn = dy * gain
    dx = inv * dn - x * inv**3 * (dn * x).mean(axis=-1, keepdims=True)
    return dx, dgain


def causal_mask(t: int) -> Array:
    """True where a position may look: at itself and everything before it, never ahead."""
    return np.tril(np.ones((t, t), dtype=bool))


# ── The model ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Trace:
    """Everything one forward pass computed. The backward pass needs it, and it's fun to look at."""

    tokens: Array  # (batch, time) token ids
    x0: Array  # (batch, time, width) token embedding + position embedding
    a_in: Array  # the normalized input to attention
    a_inv: Array
    q: Array  # (batch, heads, time, head_width) queries, keys and values
    k: Array
    v: Array
    scores: Array  # (batch, heads, time, time) how well each query matches each key
    weights: Array  # the same after the causal mask and softmax: where each position looks
    mixed: Array  # (batch, time, width) what the heads gathered, side by side
    x1: Array  # after adding attention's output back in
    m_in: Array
    m_inv: Array
    pre: Array  # (batch, time, hidden) the MLP's middle layer, before ReLU
    x2: Array  # after adding the MLP's output back in
    f_in: Array
    f_inv: Array
    logits: Array  # (batch, time, vocab) a score for every possible next token

    @property
    def probs(self) -> Array:
        return softmax(self.logits)


class Transformer:
    def __init__(self, config: Config, params: dict[str, Array]):
        self.config = config
        self.params = params

    @classmethod
    def create(cls, config: Config, rng: np.random.Generator) -> Transformer:
        return cls(config, init_params(config, rng))

    @property
    def size(self) -> int:
        return sum(p.size for p in self.params.values())

    def copy(self) -> Transformer:
        return Transformer(self.config, {k: v.copy() for k, v in self.params.items()})

    def forward(self, tokens: Array) -> Trace:
        """Read token ids, shape (batch, time), and score every possible next token."""
        p, c = self.params, self.config
        tokens = np.atleast_2d(tokens)
        t = tokens.shape[1]
        if t > c.context:
            raise ValueError(f"at most {c.context} tokens fit in the context, got {t}")

        # 1. Look up a vector for each token and add one for its position.
        x0 = p["embed.token"][tokens] + p["embed.position"][:t]

        # 2. Attention: every position gathers information from the ones before it.
        a_in, a_inv = rmsnorm(x0, p["attn.norm"])
        q = split_heads(a_in @ p["attn.query"], c.heads)
        k = split_heads(a_in @ p["attn.key"], c.heads)
        v = split_heads(a_in @ p["attn.value"], c.heads)
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(c.head_width)
        scores = np.where(causal_mask(t), scores, -np.inf)
        weights = softmax(scores)
        mixed = merge_heads(weights @ v)
        x1 = x0 + mixed @ p["attn.out"]

        # 3. MLP: every position thinks about what it gathered, on its own.
        m_in, m_inv = rmsnorm(x1, p["mlp.norm"])
        pre = m_in @ p["mlp.up"]
        x2 = x1 + np.maximum(pre, 0.0) @ p["mlp.down"]

        # 4. Score every token in the vocabulary as the next one.
        f_in, f_inv = rmsnorm(x2, p["head.norm"])
        logits = f_in @ p["head.unembed"]

        return Trace(
            tokens, x0, a_in, a_inv, q, k, v, scores, weights, mixed, x1,
            m_in, m_inv, pre, x2, f_in, f_inv, logits,
        )  # fmt: skip

    def backward(self, tr: Trace, dlogits: Array) -> tuple[dict[str, Array], Array]:
        """Push the gradient from the logits back through every layer.

        Returns the gradient for every weight, plus the gradient for the input vectors (x0).
        """
        p, c = self.params, self.config
        g: dict[str, Array] = {}

        # 4. The head.
        df_in, g["head.unembed"] = linear_backward(dlogits, tr.f_in, p["head.unembed"])
        dx2, g["head.norm"] = rmsnorm_backward(df_in, tr.x2, tr.f_inv, p["head.norm"])

        # 3. The MLP. The residual connection passes dx2 straight through, too.
        relu = np.maximum(tr.pre, 0.0)
        drelu, g["mlp.down"] = linear_backward(dx2, relu, p["mlp.down"])
        dpre = drelu * (tr.pre > 0)
        dm_in, g["mlp.up"] = linear_backward(dpre, tr.m_in, p["mlp.up"])
        dx, g["mlp.norm"] = rmsnorm_backward(dm_in, tr.x1, tr.m_inv, p["mlp.norm"])
        dx1 = dx2 + dx

        # 2. Attention.
        dmixed, g["attn.out"] = linear_backward(dx1, tr.mixed, p["attn.out"])
        dout = split_heads(dmixed, c.heads)
        dweights = dout @ tr.v.swapaxes(-1, -2)
        dv = tr.weights.swapaxes(-1, -2) @ dout
        dscores = softmax_backward(dweights, tr.weights) / np.sqrt(c.head_width)
        dq = dscores @ tr.k
        dk = dscores.swapaxes(-1, -2) @ tr.q
        da_in = np.zeros_like(tr.a_in)
        for name, dh in (("query", dq), ("key", dk), ("value", dv)):
            d, g[f"attn.{name}"] = linear_backward(merge_heads(dh), tr.a_in, p[f"attn.{name}"])
            da_in += d
        dx, g["attn.norm"] = rmsnorm_backward(da_in, tr.x0, tr.a_inv, p["attn.norm"])
        dx0 = dx1 + dx

        # 1. The embeddings. Only the rows for tokens that actually appeared get a gradient.
        g["embed.token"] = np.zeros_like(p["embed.token"])
        np.add.at(g["embed.token"], tr.tokens, dx0)
        g["embed.position"] = np.zeros_like(p["embed.position"])
        g["embed.position"][: tr.tokens.shape[1]] = dx0.sum(axis=0)

        return {name: g[name] for name in p}, dx0


def split_heads(x: Array, heads: int) -> Array:
    """(batch, time, width) → (batch, heads, time, head_width)."""
    b, t, d = x.shape
    return x.reshape(b, t, heads, d // heads).swapaxes(1, 2)


def merge_heads(x: Array) -> Array:
    """(batch, heads, time, head_width) → (batch, time, width)."""
    b, h, t, hw = x.shape
    return x.swapaxes(1, 2).reshape(b, t, h * hw)


def cross_entropy(logits: Array, targets: Array) -> tuple[float, Array]:
    """How surprised the model was by the right answers, on average, and the gradient of that.

    The loss for one prediction is -log(probability given to the right answer). Its gradient
    with respect to the logits is beautifully simple: the probabilities, minus 1 at the right
    answer. Targets of -1 are padding and don't count.
    """
    probs = softmax(logits)
    flat = probs.reshape(-1, probs.shape[-1])
    t = targets.reshape(-1)
    rows = np.flatnonzero(t >= 0)
    loss = float(-np.log(flat[rows, t[rows]]).mean())
    grad = flat.copy()
    grad[rows, t[rows]] -= 1.0
    grad[t < 0] = 0.0
    return loss, (grad / len(rows)).reshape(probs.shape)
