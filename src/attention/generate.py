"""Inference: run the model forward, roll a die weighted by its probabilities, repeat."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .model import Array, Cache, Transformer, softmax
from .tokenizer import END, Tokenizer

LIMIT = 120  # tokens to write, at most, unless asked for more


@dataclass(frozen=True, slots=True)
class Pick:
    """One step of writing: what the model had read, what it thought, and what it chose."""

    context: tuple[int, ...]  # every token so far, starting with <|endoftext|>
    probs: Array  # (vocab,) how likely each next token is, after temperature, top-k and top-p
    weights: Array  # (layers, heads, positions) where the newest position looked, in every head
    roll: float  # a random number between 0 and 1
    token: int  # the token the roll landed on

    @property
    def done(self) -> bool:
        return self.token == END


def choose(probs: Array, roll: float) -> int:
    """Line the probabilities up end to end from 0 to 1 and see which one the roll lands in."""
    return min(int(np.searchsorted(np.cumsum(probs), roll, side="right")), len(probs) - 1)


def nucleus(probs: Array, top_p: float) -> Array:
    """Top-p: keep the likeliest tokens until they add up to `top_p`, drop the long tail."""
    if top_p >= 1.0:
        return probs
    order = np.argsort(-probs, kind="stable")
    keep = order[: int(np.searchsorted(np.cumsum(probs[order]), top_p)) + 1]
    kept = np.zeros_like(probs)
    kept[keep] = probs[keep]
    return kept / kept.sum()


def top_k(probs: Array, k: int) -> Array:
    """Top-k: keep only the k likeliest tokens."""
    if k <= 0 or k >= len(probs):
        return probs
    keep = np.argsort(-probs, kind="stable")[:k]
    kept = np.zeros_like(probs)
    kept[keep] = probs[keep]
    return kept / kept.sum()


def dials(logits: Array, temperature: float = 1.0, k: int = 0, top_p: float = 1.0) -> Array:
    """Scores to probabilities, the way a chatbot's settings shape them."""
    return nucleus(top_k(softmax(logits / max(temperature, 1e-3)), k), top_p)


def prompt(tokenizer: Tokenizer, text: str = "") -> list[int]:
    """A new document, starting with `text`: <|endoftext|> first, as in training."""
    return [END, *tokenizer.encode(text)]


def picks(
    model: Transformer,
    ids: Sequence[int],
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_p: float = 1.0,
    k: int = 0,
    limit: int = LIMIT,
    stop: Sequence[int] = (END,),
) -> Iterator[Pick]:
    """Write after `ids`, one token at a time, until a stop token, the limit or a full context.

    First the prompt is read in one pass (the prefill), which fills the KV cache. After that,
    each step runs the model on just the newest token (a decode step): everything earlier is
    already in the cache.
    """
    ids = list(ids)
    cache = Cache()
    trace = model.forward(np.array([ids]), cache)
    for _ in range(limit):
        probs = dials(trace.logits[0, -1], temperature, k, top_p)
        roll = float(rng.random())
        token = choose(probs, roll)
        yield Pick(tuple(ids), probs, trace.weights[:, 0, :, -1], roll, token)
        ids.append(token)
        if token in stop or len(ids) >= model.config.context:
            return
        trace = model.forward(np.array([[token]]), cache)


def write(
    model: Transformer,
    tokenizer: Tokenizer,
    rng: np.random.Generator,
    text: str = "",
    temperature: float = 1.0,
    top_p: float = 1.0,
    k: int = 0,
    limit: int = LIMIT,
) -> str:
    """Carry on from `text` (or start a new document) and return what the model wrote."""
    written = [
        p.token for p in picks(model, prompt(tokenizer, text), rng, temperature, top_p, k, limit)
    ]
    return tokenizer.decode(t for t in written if t != END)


def samples(
    model: Transformer,
    tokenizer: Tokenizer,
    rng: np.random.Generator,
    count: int,
    text: str = "",
    temperature: float = 1.0,
    top_p: float = 1.0,
    k: int = 0,
    limit: int = LIMIT,
) -> list[str]:
    return [write(model, tokenizer, rng, text, temperature, top_p, k, limit) for _ in range(count)]


def log_probs(model: Transformer, tokenizer: Tokenizer, text: str) -> Array:
    """How likely the model found each token of `text`, in turn, as a new document."""
    ids = prompt(tokenizer, text)[: model.config.context + 1]
    probs = model.forward(np.array([ids[:-1]])).probs[0]
    return np.log(probs[np.arange(len(ids) - 1), ids[1:]])


# ── Copying ───────────────────────────────────────────────────────────────────

RUN = 12  # this many tokens in a row, word for word from the training text, counts as copying
PRIME = np.uint64(0x100000001B3)


def fingerprints(ids: Array, n: int) -> Array:
    """A number for every run of n tokens, the same whenever the tokens are."""
    ids = np.asarray(ids, dtype=np.uint64)
    if len(ids) < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(ids) - n + 1, dtype=np.uint64)
    for j in range(n):  # wraps around at 2^64, which is fine for a fingerprint
        h = h * PRIME + ids[j : len(ids) - n + 1 + j] + np.uint64(1)
    return h


class Memory:
    """Every run of a few tokens in the training text, to tell when a model copies it."""

    def __init__(self, stream: Array, n: int = RUN):
        self.n = n
        self.seen = np.unique(fingerprints(stream, n))

    def copied(self, ids: Sequence[int] | Array) -> Array:
        """For each token: is it part of a run that appears, word for word, in the training text?"""
        ids = np.asarray(ids)
        hits = np.isin(fingerprints(ids, self.n), self.seen)
        marks = np.zeros(len(ids), dtype=bool)
        for i in np.flatnonzero(hits):
            marks[i : i + self.n] = True
        return marks

    def longest(self, ids: Sequence[int] | Array) -> int:
        """The longest stretch of tokens copied word for word."""
        best = run = 0
        for mark in self.copied(ids):
            run = run + 1 if mark else 0
            best = max(best, run)
        return best
