"""After pretraining: fine-tuning, LoRA, a chat model, and learning from feedback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .corpus import Corpus, Dataset, Niche, stream
from .generate import picks
from .model import Array, Transformer, cross_entropy
from .tokenizer import ASSISTANT, END, USER, Tokenizer
from .train import Adam, Trainer, clip

STEPS = 150
LEARNING_RATE = 0.002  # gentler than pretraining: adjust what it knows, don't overwrite it


# ── Fine-tuning on a niche ────────────────────────────────────────────────────


def narrow(data: Dataset, niche: Niche) -> Dataset:
    """The same data, keeping only the training documents in the niche."""
    docs = tuple(d for d in data.train_docs if niche.has(d))
    return Dataset(data.tokenizer, docs, data.val_docs, stream(data.tokenizer, docs), data.val)


def niche_trainer(
    model: Transformer, data: Dataset, niche: Niche, rng: np.random.Generator, steps: int = STEPS
) -> Trainer:
    """A trainer that shows a copy of the model only the documents in one niche."""
    return Trainer(
        model.copy(),
        narrow(data, niche),
        rng,
        steps=steps,
        lr=LEARNING_RATE,
        decay=0.0,  # pulling a pretrained model's weights toward zero would only make it forget
        warmup=10,
    )


def drift(model: Transformer, start: Transformer) -> float:
    """How far a model's weights have moved from where they started, relative to their size."""
    moved = sum(float(((model.params[k] - start.params[k]) ** 2).sum()) for k in start.params)
    size = sum(float((start.params[k] ** 2).sum()) for k in start.params)
    return (moved / size) ** 0.5


def share(texts: Sequence[str], niche: Niche) -> float:
    return sum(niche.has(t) for t in texts) / max(len(texts), 1)


# ── LoRA ──────────────────────────────────────────────────────────────────────

RANK = 8  # at 4, on some models it learned the niche far less well than full fine-tuning
TARGETS = ("attn.query", "attn.key", "attn.value", "attn.out", "mlp.gate", "mlp.up", "mlp.down")


@dataclass
class LoRA:
    """Low-rank adaptation: freeze the model, and learn a small correction to each grid.

    For a grid W with m rows and n columns, LoRA learns two thin grids, A (m × r) and B (r × n),
    and uses W + A @ B in W's place. With r = 8, that's 8 × (m + n) numbers instead of m × n.
    B starts at zero, so to begin with the model is exactly the one it started from.
    """

    base: Transformer
    rank: int = RANK
    a: dict[str, Array] = field(default_factory=dict)
    b: dict[str, Array] = field(default_factory=dict)

    @classmethod
    def create(
        cls, base: Transformer, rng: np.random.Generator, rank: int = RANK, targets=TARGETS
    ) -> LoRA:
        lora = cls(base, rank)
        for name, w in base.params.items():
            if name.endswith(targets):
                lora.a[name] = rng.normal(0.0, 1 / np.sqrt(w.shape[0]), (w.shape[0], rank))
                lora.b[name] = np.zeros((rank, w.shape[1]))
        return lora

    @property
    def size(self) -> int:
        """How many numbers it learns."""
        return sum(a.size + self.b[name].size for name, a in self.a.items())

    @property
    def params(self) -> dict[str, Array]:
        """The thin grids, by name, for the optimizer to move."""
        return {
            f"{n}.{side}": g for n in self.a for side, g in (("a", self.a[n]), ("b", self.b[n]))
        }

    def merged(self) -> Transformer:
        """The model with each corrected grid replaced by W + A @ B. The rest are shared."""
        params = dict(self.base.params)
        for name, a in self.a.items():
            params[name] = self.base.params[name] + a @ self.b[name]
        return Transformer(self.base.config, params)

    def gradients(self, grads: dict[str, Array]) -> dict[str, Array]:
        """From the gradient of each merged grid, the gradients of its A and B.

        The chain rule once more: W + A @ B moves by dA @ B when A moves by dA, so A's gradient
        is the grid's gradient times B, turned around. The same goes for B, with A.
        """
        out = {}
        for name, a in self.a.items():
            out[f"{name}.a"] = grads[name] @ self.b[name].T
            out[f"{name}.b"] = a.T @ grads[name]
        return out


class AdapterTrainer:
    """Like a Trainer, but only the LoRA grids learn: the model underneath stays frozen."""

    def __init__(self, lora: LoRA, batches, rng: np.random.Generator, steps: int, lr: float):
        self.lora, self.batches, self.rng = lora, batches, rng
        self.steps, self.lr = steps, lr
        self.step_number = 0
        self.losses: list[float] = []
        self.optimizer = Adam(lora.params)

    @property
    def done(self) -> bool:
        return self.step_number >= self.steps

    @property
    def model(self) -> Transformer:
        return self.lora.merged()

    def step(self) -> float:
        model = self.lora.merged()
        inputs, targets = self.batches(self.rng)
        trace = model.forward(inputs)
        loss, dlogits = cross_entropy(trace.logits, targets)
        grads, _ = model.backward(trace, dlogits)
        adapter, _ = clip(self.lora.gradients(grads), 1.0)
        self.optimizer.step(self.lora.params, adapter, self.lr)
        self.step_number += 1
        self.losses.append(loss)
        return loss


# ── A chat model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Exchange:
    request: str
    reply: str


def exchanges(corpus: Corpus, docs: Sequence[str]) -> list[Exchange]:
    """Requests and replies made from the corpus, to teach the chat format with."""
    out = []
    for doc in docs:
        if corpus.play:  # someone says something; the next speaker answers, in character
            for said, answer in pairwise(doc.split("\n\n")):
                out.append(Exchange(said.split("\n", 1)[-1], answer))
        elif corpus.titled and "\n\n" in doc:  # ask for a story by name; it's told, title first
            title = doc.split("\n\n", 1)[0]
            named = title[0].lower() + title[1:] if title.startswith("The ") else title
            out.append(Exchange(f"Tell me the {corpus.noun} of {named}.", doc))
        elif ". " in doc:  # the start of a passage, and the rest of it
            first, rest = doc.split(". ", 1)
            out.append(Exchange(f"Go on: {first}.", rest))
    return out


def chat_ids(
    tokenizer: Tokenizer, request: str, reply: str | None = None, most: int | None = None
) -> list[int]:
    """A conversation, as the model reads it: the chat template, with its special tokens.

    A request longer than `most` tokens keeps only its end, the part nearest the reply.
    """
    asked = tokenizer.encode(request)
    ids = [END, USER, *(asked[-most:] if most else asked), ASSISTANT]
    if reply is not None:
        ids += [*tokenizer.encode(reply), END]
    return ids


def chat_batch(
    tokenizer: Tokenizer,
    examples: Sequence[Exchange],
    rng: np.random.Generator,
    count: int,
    context: int,
) -> tuple[Array, Array]:
    """Some conversations, picked at random. Only the replies count toward the loss."""
    picked = [examples[int(i)] for i in rng.choice(len(examples), count)]
    return conversations(tokenizer, picked, context)


def conversations(
    tokenizer: Tokenizer, examples: Sequence[Exchange], context: int
) -> tuple[Array, Array]:
    """Conversations, padded to the same length, with the loss counting only the replies.

    Every target up to and including <|assistant|> is -1, so the model is never graded on
    writing requests, only on answering them.
    """
    rows = []
    for ex in examples:
        ids = chat_ids(tokenizer, ex.request, ex.reply, context // 2)[: context + 1]
        start = ids.index(ASSISTANT)
        rows.append((ids[:-1], [t if j >= start else -1 for j, t in enumerate(ids[1:])]))
    return padded(rows)


def padded(rows: list[tuple[list[int], list[int]]]) -> tuple[Array, Array]:
    width = max(len(r[0]) for r in rows)
    inputs = np.full((len(rows), width), END, dtype=np.int64)
    targets = np.full((len(rows), width), -1, dtype=np.int64)
    for row, (ids, tgt) in enumerate(rows):
        inputs[row, : len(ids)] = ids
        targets[row, : len(tgt)] = tgt
    return inputs, targets


CHAT_STEPS = 500


def chat_trainer(
    model: Transformer,
    data: Dataset,
    examples: Sequence[Exchange],
    rng: np.random.Generator,
    steps: int = CHAT_STEPS,
) -> Trainer:
    """Fine-tuning on conversations: a copy of the model, learning to answer in the template."""
    context = model.config.context
    return Trainer(
        model.copy(),
        data,
        rng,
        steps=steps,
        lr=LEARNING_RATE,
        decay=0.0,
        warmup=20,
        batches=lambda r: chat_batch(data.tokenizer, examples, r, 16, context),
    )


def answer(
    model: Transformer,
    tokenizer: Tokenizer,
    request: str,
    rng: np.random.Generator,
    temperature: float = 0.3,
    limit: int = 100,
) -> str:
    """The model's reply to a request, written after <|assistant|> until it says it's done."""
    ids = chat_ids(tokenizer, request, most=model.config.context // 2)
    room = min(limit, model.config.context - len(ids))
    if room <= 0:
        return ""
    written = [p.token for p in picks(model, ids, rng, temperature, limit=room)]
    return tokenizer.decode(t for t in written if not tokenizer.is_special(t))


# ── Feedback ──────────────────────────────────────────────────────────────────


def examples_batch(
    tokenizer: Tokenizer, texts: Sequence[str], context: int, request: str | None = None
) -> tuple[Array, Array]:
    """Texts as training examples: whole documents, or replies to a request."""
    if request is not None:
        return conversations(tokenizer, [Exchange(request, t) for t in texts], context)
    seqs = [[END, *tokenizer.encode(t), END][: context + 1] for t in texts]
    return padded([(s[:-1], s[1:]) for s in seqs])


def per_token(
    model: Transformer, tokenizer: Tokenizer, text: str, request: str | None = None
) -> float:
    """The average log-probability of each token of this text, the ending included."""
    inputs, targets = examples_batch(tokenizer, [text], model.config.context, request)
    probs = model.forward(inputs).probs[0]
    rows = np.flatnonzero(targets[0] >= 0)
    return float(np.log(probs[rows, targets[0, rows]]).mean())


@dataclass(frozen=True, slots=True)
class Feedback:
    """One round of learning from what someone liked, and how each text's odds changed."""

    model: Transformer
    before: dict[str, float]  # each text's average log-probability per token, before and after
    after: dict[str, float]

    def chance(self, text: str) -> tuple[float, float]:
        """The typical chance it gave each token of the text, before and after."""
        return float(np.exp(self.before[text])), float(np.exp(self.after[text]))

    def change(self, text: str) -> float:
        """How many times likelier each token became, typically (below 1: less likely)."""
        return float(np.exp(self.after[text] - self.before[text]))


def learn_from(
    model: Transformer,
    tokenizer: Tokenizer,
    liked: list[str],
    disliked: list[str],
    steps: int = 2,
    lr: float = 0.001,
    push_away: float = 0.2,
    request: str | None = None,
) -> Feedback:
    """Make the liked texts likelier and the rest a little less likely, on a copy.

    Liked texts are trained on like any example. Disliked ones get the opposite push, but
    gently: pushing hard away from things quickly breaks a model. With a request, the texts
    are replies to it, and only the replies are learned.
    """
    copy = model.copy()
    texts = liked + disliked
    before = {t: per_token(copy, tokenizer, t, request) for t in texts}
    adam = Adam(copy.params)
    for _ in range(steps):
        total = {k: np.zeros_like(v) for k, v in copy.params.items()}
        for group, sign in ((liked, 1.0), (disliked, -push_away)):
            if not group:
                continue
            inputs, targets = examples_batch(tokenizer, group, copy.config.context, request)
            trace = copy.forward(inputs)
            _, dlogits = cross_entropy(trace.logits, targets)
            grads, _ = copy.backward(trace, dlogits)
            for name in total:
                total[name] += sign * grads[name]
        adam.step(copy.params, total, lr)
    after = {t: per_token(copy, tokenizer, t, request) for t in texts}
    return Feedback(copy, before, after)
