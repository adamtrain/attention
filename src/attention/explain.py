"""Backprop after training: not to learn, but to look inside one prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import Array, Trace, Transformer, softmax
from .tokenizer import END


@dataclass(frozen=True, slots=True)
class Influence:
    ids: tuple[int, ...]  # what the model read, <|endoftext|> first
    trace: Trace
    probs: Array  # what comes next
    sizes: Array  # how strongly each token read could sway the top prediction

    @property
    def top(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def runner_up(self) -> int:
        """The likeliest token after the top one (not the end of the document)."""
        order = [int(i) for i in np.argsort(-self.probs) if int(i) not in (self.top, END)]
        return order[0]


def surprise_gradient(trace: Trace, probs: Array, target: int) -> Array:
    """The gradient of -log p(target) at the last position: probabilities minus the one-hot."""
    dlogits = np.zeros_like(trace.logits)
    dlogits[0, -1] = probs
    dlogits[0, -1, target] -= 1.0
    return dlogits


def influence(model: Transformer, ids: Sequence[int], target: int | None = None) -> Influence:
    """Backpropagate from one prediction to the tokens that went in, leaving the weights alone."""
    trace = model.forward(np.array([ids]))
    probs = softmax(trace.logits[0, -1])
    target = int(np.argmax(probs)) if target is None else target
    _, dx = model.backward(trace, surprise_gradient(trace, probs, target))
    return Influence(tuple(ids), trace, probs, np.linalg.norm(dx[0], axis=-1))


@dataclass(frozen=True, slots=True)
class Nudge:
    target: int
    lr: float
    before: Array
    after: Array


def nudge(model: Transformer, ids: Sequence[int], target: int) -> Nudge:
    """One step of plain gradient descent toward `target`, on a copy of the model.

    Uses the smallest learning rate from a short ladder that makes the change easy to see.
    """
    batch = np.array([ids])
    trace = model.forward(batch)
    before = softmax(trace.logits[0, -1])
    after, lr = before, 0.0
    for lr in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
        copy = model.copy()
        grads, _ = copy.backward(trace, surprise_gradient(trace, before, target))
        for name, w in copy.params.items():
            w -= lr * grads[name]
        after = softmax(copy.forward(batch).logits[0, -1])
        if after[target] >= 0.2:
            break
    return Nudge(target, lr, before, after)
