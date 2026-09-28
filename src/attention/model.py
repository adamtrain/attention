"""A small GPT, written out by hand in NumPy, built the way today's open models are built.

It has the same parts as Llama, Mistral or Qwen, just far fewer of them: token embeddings, a stack
of identical layers (attention, then an MLP, each adding onto a shared residual stream), rotary
position embeddings, grouped-query attention, a gated MLP, and RMSNorm before every block.

Every layer is a pair of functions. The forward function computes the layer's output. The
backward function takes how much the loss would change if each output number moved a little
(the gradient) and works out the same for the layer's inputs and weights. That is the chain
rule, applied one layer at a time, from the loss back to the embeddings: backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

type Array = np.ndarray

EPS = 1e-5


@dataclass(frozen=True, slots=True)
class Config:
    vocab: int  # how many different tokens there are
    context: int  # the most tokens the model can read at once
    width: int = 64  # numbers per token in the residual stream
    layers: int = 4  # blocks, stacked
    heads: int = 4  # attention heads per layer, each asking its own question
    kv_heads: int = 2  # key and value heads per layer, each shared by a group of query heads
    hidden: int = 192  # width of the MLP's middle
    rope_base: float = 10_000.0  # how slowly the slowest position rotation turns

    @property
    def head_width(self) -> int:
        return self.width // self.heads

    @property
    def group(self) -> int:
        """How many query heads share each key and value head."""
        return self.heads // self.kv_heads

    def hugging_face(self) -> dict[str, object]:
        """The same settings, named the way a Llama model's config.json names them."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": self.vocab,
            "max_position_embeddings": self.context,
            "hidden_size": self.width,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_width,
            "intermediate_size": self.hidden,
            "hidden_act": "silu",
            "rope_theta": self.rope_base,
            "rms_norm_eps": EPS,
            "tie_word_embeddings": True,
        }


# ── The weights, and what they're called ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Part:
    """What a group of weights is called, for people."""

    label: str  # "queries"
    group: str  # "layer 2"
    hugging_face: str  # the same weights' name in a Llama checkpoint


BLOCK = {
    "attn.norm": ("attention norm", "input_layernorm.weight"),
    "attn.query": ("queries", "self_attn.q_proj.weight"),
    "attn.key": ("keys", "self_attn.k_proj.weight"),
    "attn.value": ("values", "self_attn.v_proj.weight"),
    "attn.out": ("attention output", "self_attn.o_proj.weight"),
    "mlp.norm": ("MLP norm", "post_attention_layernorm.weight"),
    "mlp.gate": ("MLP gate", "mlp.gate_proj.weight"),
    "mlp.up": ("MLP up", "mlp.up_proj.weight"),
    "mlp.down": ("MLP down", "mlp.down_proj.weight"),
}


def part(name: str) -> Part:
    if name == "embed":
        return Part("token embeddings", "embeddings", "model.embed_tokens.weight")
    if name == "norm":
        return Part("final norm", "final norm", "model.norm.weight")
    _, i, rest = name.split(".", 2)
    label, hf = BLOCK[rest]
    return Part(label, f"layer {int(i) + 1}", f"model.layers.{i}.{hf}")


def groups(config: Config) -> list[str]:
    """The model's parts from input to output, for charts."""
    return ["embeddings", *(f"layer {i + 1}" for i in range(config.layers)), "final norm"]


def init_params(config: Config, rng: np.random.Generator) -> dict[str, Array]:
    """Random starting weights. Training will shape every one of them."""
    c = config
    d, f, hd = c.width, c.hidden, c.head_width
    # Each block adds its output onto the stream; starting those small keeps a deep stack calm.
    out_scale = 1 / np.sqrt(2 * c.layers)

    def weights(rows: int, cols: int, scale: float = 1.0) -> Array:
        return rng.normal(0.0, scale / np.sqrt(rows), (rows, cols))

    # Small, so the untrained model starts out finding every token about equally likely.
    params = {"embed": rng.normal(0.0, 0.02, (c.vocab, d))}
    for i in range(c.layers):
        params |= {
            f"layers.{i}.attn.norm": np.ones(d),
            f"layers.{i}.attn.query": weights(d, c.heads * hd),
            f"layers.{i}.attn.key": weights(d, c.kv_heads * hd),
            f"layers.{i}.attn.value": weights(d, c.kv_heads * hd),
            f"layers.{i}.attn.out": weights(c.heads * hd, d, out_scale),
            f"layers.{i}.mlp.norm": np.ones(d),
            f"layers.{i}.mlp.gate": weights(d, f),
            f"layers.{i}.mlp.up": weights(d, f),
            f"layers.{i}.mlp.down": weights(f, d, out_scale),
        }
    params["norm"] = np.ones(d)
    return params


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


def sigmoid(x: Array) -> Array:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # the same as 1 / (1 + e^-x), without overflowing


def silu(x: Array) -> Array:
    """A smooth switch: about 0 for negative numbers, about x for positive ones."""
    return x * sigmoid(x)


def silu_backward(dy: Array, x: Array) -> Array:
    s = sigmoid(x)
    return dy * s * (1 + x * (1 - s))


def angles(positions: Array, head_width: int, base: float) -> Array:
    """How far RoPE turns each pair of numbers, at each position: shape (time, head_width / 2).

    The first pair turns fast, a radian per position; each pair after it turns more slowly,
    down to one that barely turns at all.
    """
    speeds = base ** (-np.arange(0, head_width, 2) / head_width)  # radians per position
    return np.outer(positions, speeds)


def rope(x: Array, angle: Array) -> Array:
    """Rotary position embedding: treat each pair of numbers as a point, and turn it by `angle`.

    Only queries and keys are turned. When a query at one position meets a key at another, the
    turns partly cancel, so their match depends on how far apart they are, not where they sit.
    Turning by -angle undoes it, which is also its backward pass.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out


def causal_mask(t: int, s: int | None = None) -> Array:
    """True where a position may look: at itself and everything before it, never ahead.

    The last t of s positions are the ones looking (with a KV cache, the rest came earlier).
    """
    s = t if s is None else s
    return np.arange(s)[None, :] <= np.arange(s - t, s)[:, None]


def split_heads(x: Array, heads: int) -> Array:
    """(batch, time, heads × head_width) → (batch, heads, time, head_width)."""
    b, t, d = x.shape
    return x.reshape(b, t, heads, d // heads).swapaxes(1, 2)


def merge_heads(x: Array) -> Array:
    """(batch, heads, time, head_width) → (batch, time, heads × head_width)."""
    b, h, t, hw = x.shape
    return x.swapaxes(1, 2).reshape(b, t, h * hw)


def share(kv: Array, group: int) -> Array:
    """Grouped-query attention: each key or value head serves `group` query heads in a row."""
    return kv if group == 1 else np.repeat(kv, group, axis=1)


def unshare(d: Array, group: int) -> Array:
    """A shared head's gradient is the sum of what each query head it served sent back."""
    if group == 1:
        return d
    b, h, s, hw = d.shape
    return d.reshape(b, h // group, group, s, hw).sum(axis=2)


# ── The KV cache ──────────────────────────────────────────────────────────────


@dataclass
class Cache:
    """Every layer's keys and values for the positions read so far: the KV cache.

    When the model writes, each new token needs its own query, but it compares that query with
    the keys of every earlier position and blends their values. Those never change once they're
    made (the causal mask sees to that), so they're kept here instead of being made again.
    """

    # For each layer: (batch, kv_heads, positions, head_width).
    keys: list[Array] = field(default_factory=list)
    values: list[Array] = field(default_factory=list)

    @property
    def length(self) -> int:
        """How many positions it holds."""
        return self.keys[0].shape[2] if self.keys else 0

    @property
    def size(self) -> int:
        """How many numbers it holds."""
        return sum(k.size + v.size for k, v in zip(self.keys, self.values, strict=True))

    def add(self, layer: int, k: Array, v: Array) -> tuple[Array, Array]:
        """Append new keys and values for one layer. Returns all of that layer's so far."""
        if layer == len(self.keys):
            self.keys.append(k)
            self.values.append(v)
        else:
            self.keys[layer] = np.concatenate([self.keys[layer], k], axis=2)
            self.values[layer] = np.concatenate([self.values[layer], v], axis=2)
        return self.keys[layer], self.values[layer]


# ── The model ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Block:
    """Everything one layer computed."""

    x: Array  # (batch, time, width) the residual stream coming in
    a_in: Array  # the normalized input to attention
    a_inv: Array
    q: Array  # (batch, heads, time, head_width) queries, after RoPE
    k: Array  # (batch, kv_heads, positions, head_width) keys, after RoPE, cached ones included
    v: Array  # (batch, kv_heads, positions, head_width) values
    weights: Array  # (batch, heads, time, positions) where each position looked
    mixed: Array  # (batch, time, width) what the heads gathered, side by side
    mid: Array  # the stream after attention added its output
    m_in: Array  # the normalized input to the MLP
    m_inv: Array
    gate: Array  # (batch, time, hidden) the MLP's two middle layers
    up: Array
    out: Array  # the stream after the MLP added its output: the next layer's x


@dataclass(slots=True)
class Trace:
    """Everything one forward pass computed. The backward pass needs it, and it's fun to look at."""

    tokens: Array  # (batch, time) token ids
    start: int  # the position of the first token (after any cached ones)
    blocks: list[Block]
    f_in: Array  # the stream after the final norm
    f_inv: Array
    logits: Array  # (batch, time, vocab) a score for every possible next token

    @property
    def probs(self) -> Array:
        return softmax(self.logits)

    @property
    def weights(self) -> Array:
        """Where every head in every layer looked: (layers, batch, heads, time, positions)."""
        return np.stack([b.weights for b in self.blocks])

    def stream(self, mid: bool = False) -> list[Array]:
        """The residual stream after the embeddings, then after each layer.

        With `mid`, after each attention block and each MLP separately.
        """
        out = [self.blocks[0].x]
        for b in self.blocks:
            out += [b.mid, b.out] if mid else [b.out]
        return out


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

    def layer(self, i: int) -> dict[str, Array]:
        """One layer's weights, by their short names."""
        return {name: self.params[f"layers.{i}.{name}"] for name in BLOCK}

    def forward(self, tokens: Array, cache: Cache | None = None) -> Trace:
        """Read token ids, shape (batch, time), and score every possible next token.

        With a cache, the tokens carry on from the ones already in it: their keys and values
        are added to the cache, and they can look back at everything there.
        """
        p, c = self.params, self.config
        tokens = np.atleast_2d(tokens)
        start = cache.length if cache else 0
        end = start + tokens.shape[1]
        if end > c.context:
            raise ValueError(f"at most {c.context} tokens fit in the context, got {end}")
        turn = angles(np.arange(start, end), c.head_width, c.rope_base)

        # 1. Look up a vector for each token. This starts the residual stream.
        x = p["embed"][tokens]

        # 2. Each layer reads the stream and adds to it: attention, then the MLP.
        blocks = []
        for i in range(c.layers):
            blocks.append(self.block(i, x, turn, cache))
            x = blocks[-1].out

        # 3. Score every token as the next one, using the embedding table from step 1 again.
        f_in, f_inv = rmsnorm(x, p["norm"])
        logits = f_in @ p["embed"].T

        return Trace(tokens, start, blocks, f_in, f_inv, logits)

    def block(self, i: int, x: Array, turn: Array, cache: Cache | None = None) -> Block:
        """One layer. Attention moves information between positions; the MLP works on each."""
        c, w = self.config, self.layer(i)

        # Attention: every position gathers information from the ones before it.
        a_in, a_inv = rmsnorm(x, w["attn.norm"])
        q = rope(split_heads(a_in @ w["attn.query"], c.heads), turn)  # what am I looking for?
        k = rope(split_heads(a_in @ w["attn.key"], c.kv_heads), turn)  # what do I have?
        v = split_heads(a_in @ w["attn.value"], c.kv_heads)  # what I'll pass on if you look
        if cache is not None:
            k, v = cache.add(i, k, v)  # every position's keys and values so far
        scores = q @ share(k, c.group).swapaxes(-1, -2) / np.sqrt(c.head_width)
        scores = np.where(causal_mask(q.shape[2], k.shape[2]), scores, -np.inf)
        weights = softmax(scores)
        mixed = merge_heads(weights @ share(v, c.group))
        mid = x + mixed @ w["attn.out"]

        # MLP: every position works on what it gathered, on its own.
        m_in, m_inv = rmsnorm(mid, w["mlp.norm"])
        gate, up = m_in @ w["mlp.gate"], m_in @ w["mlp.up"]
        out = mid + (silu(gate) * up) @ w["mlp.down"]

        return Block(x, a_in, a_inv, q, k, v, weights, mixed, mid, m_in, m_inv, gate, up, out)

    def backward(self, tr: Trace, dlogits: Array) -> tuple[dict[str, Array], Array]:
        """Push the gradient from the logits back through every layer.

        Returns the gradient for every weight, plus the gradient for the token vectors that went
        in (the start of the residual stream).
        """
        p, c = self.params, self.config
        g: dict[str, Array] = {}

        # 3. The scores. The embedding table was used here too, so it gets a gradient here too.
        g["embed"] = dlogits.reshape(-1, c.vocab).T @ tr.f_in.reshape(-1, c.width)
        df = dlogits @ p["embed"]
        dx, g["norm"] = rmsnorm_backward(df, tr.blocks[-1].out, tr.f_inv, p["norm"])

        # 2. The layers, last to first. Each hands back the gradient for the stream it read.
        turn = angles(np.arange(tr.start, tr.start + tr.tokens.shape[1]), c.head_width, c.rope_base)
        for i in reversed(range(c.layers)):
            layer_grads, dx = self.block_backward(i, tr.blocks[i], dx, turn)
            g |= {f"layers.{i}.{name}": grad for name, grad in layer_grads.items()}

        # 1. The embeddings. Only the rows for tokens that actually appeared get a gradient.
        np.add.at(g["embed"], tr.tokens, dx)

        return {name: g[name] for name in p}, dx

    def block_backward(
        self, i: int, b: Block, dout: Array, turn: Array
    ) -> tuple[dict[str, Array], Array]:
        """One layer, backward. The residual connections pass dout straight through, too."""
        c, w = self.config, self.layer(i)
        g: dict[str, Array] = {}

        # The MLP: out = mid + (silu(gate) × up) @ down.
        act = silu(b.gate)
        dh, g["mlp.down"] = linear_backward(dout, act * b.up, w["mlp.down"])
        dgate = silu_backward(dh * b.up, b.gate)
        dm_in, g["mlp.gate"] = linear_backward(dgate, b.m_in, w["mlp.gate"])
        dm_up, g["mlp.up"] = linear_backward(dh * act, b.m_in, w["mlp.up"])
        dx, g["mlp.norm"] = rmsnorm_backward(dm_in + dm_up, b.mid, b.m_inv, w["mlp.norm"])
        dmid = dout + dx

        # Attention.
        dmixed, g["attn.out"] = linear_backward(dmid, b.mixed, w["attn.out"])
        dmix = split_heads(dmixed, c.heads)
        dweights = dmix @ share(b.v, c.group).swapaxes(-1, -2)
        dv = unshare(b.weights.swapaxes(-1, -2) @ dmix, c.group)
        dscores = softmax_backward(dweights, b.weights) / np.sqrt(c.head_width)
        dq = rope(dscores @ share(b.k, c.group), -turn)
        dk = rope(unshare(dscores.swapaxes(-1, -2) @ b.q, c.group), -turn)
        da_in = np.zeros_like(b.a_in)
        for name, dh in (("query", dq), ("key", dk), ("value", dv)):
            d, g[f"attn.{name}"] = linear_backward(merge_heads(dh), b.a_in, w[f"attn.{name}"])
            da_in += d
        dx, g["attn.norm"] = rmsnorm_backward(da_in, b.x, b.a_inv, w["attn.norm"])

        return g, dmid + dx

    def lens(self, trace: Trace, mid: bool = False) -> Array:
        """The logit lens: what the model would predict if it stopped after each layer.

        The residual stream at each depth, put through the final norm and the unembedding, as
        probabilities: shape (depths, batch, time, vocab). The last depth is the real answer.
        """
        p = self.params
        guesses = [softmax(rmsnorm(x, p["norm"])[0] @ p["embed"].T) for x in trace.stream(mid)]
        return np.stack(guesses)


def cross_entropy(logits: Array, targets: Array) -> tuple[float, Array]:
    """How surprised the model was by the right answers, on average, and the gradient of that.

    The loss for one prediction is -log(probability given to the right answer). Its gradient
    with respect to the logits is beautifully simple: the probabilities, minus 1 at the right
    answer. Targets of -1 don't count (in chat fine-tuning, those are the user's words).
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
