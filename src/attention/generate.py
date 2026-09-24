"""Inference: run the model forward, roll a die weighted by its probabilities, repeat."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .corpus import BOUNDARY, Vocab
from .model import Array, Transformer, softmax


@dataclass(frozen=True, slots=True)
class Pick:
    """One step of generating: what the model saw, what it thought, and what it chose."""

    context: str  # everything so far, starting with "."
    probs: Array  # (vocab,) how likely each next character is, after temperature
    weights: Array  # (heads, time) where the last position looked
    roll: float  # a random number between 0 and 1
    token: int  # the character the roll landed on

    @property
    def done(self) -> bool:
        return self.token == 0


def choose(probs: Array, roll: float) -> int:
    """Line the probabilities up end to end from 0 to 1 and see which one the roll lands in."""
    return min(int(np.searchsorted(np.cumsum(probs), roll, side="right")), len(probs) - 1)


def nucleus(probs: Array, top_p: float) -> Array:
    """Top-p: keep the likeliest tokens until they add up to `top_p`, drop the long tail."""
    if top_p >= 1.0:
        return probs
    order = np.argsort(-probs)
    keep = order[: int(np.searchsorted(np.cumsum(probs[order]), top_p)) + 1]
    kept = np.zeros_like(probs)
    kept[keep] = probs[keep]
    return kept / kept.sum()


def picks(
    model: Transformer,
    vocab: Vocab,
    rng: np.random.Generator,
    temperature: float = 1.0,
    prefix: str = "",
    top_p: float = 1.0,
) -> Iterator[Pick]:
    ids = vocab.encode(BOUNDARY + prefix)
    while len(ids) < model.config.context:
        trace = model.forward(np.array([ids]))
        probs = nucleus(softmax(trace.logits[0, -1] / max(temperature, 1e-3)), top_p)
        roll = float(rng.random())
        token = choose(probs, roll)
        yield Pick(vocab.decode(ids), probs, trace.weights[0, :, -1], roll, token)
        if token == 0:
            return
        ids.append(token)


def sample(
    model: Transformer,
    vocab: Vocab,
    rng: np.random.Generator,
    temperature: float = 1.0,
    prefix: str = "",
    top_p: float = 1.0,
) -> str:
    word = prefix
    for pick in picks(model, vocab, rng, temperature, prefix, top_p):
        if not pick.done:
            word = pick.context[1:] + vocab.chars[pick.token]
    return word


def invent(
    model: Transformer,
    vocab: Vocab,
    rng: np.random.Generator,
    count: int,
    temperature: float = 1.0,
    prefix: str = "",
    top_p: float = 1.0,
) -> list[str]:
    """Sample a few different words."""
    words: list[str] = []
    for _ in range(count * 4):
        word = sample(model, vocab, rng, temperature, prefix, top_p)
        if word and word not in words:
            words.append(word)
        if len(words) == count:
            break
    return words


def log_prob(model: Transformer, vocab: Vocab, word: str) -> Array:
    """How likely the model found each letter of a word (and its ending), in turn."""
    ids = vocab.sequence(word)
    probs = model.forward(np.array([ids[:-1]])).probs[0]
    return np.log(probs[np.arange(len(ids) - 1), ids[1:]])
