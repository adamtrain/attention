"""Backprop after training: not to learn, but to look inside one prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import BOUNDARY, Vocab
from .model import Array, Trace, Transformer, softmax


@dataclass(frozen=True, slots=True)
class Influence:
    prefix: str
    trace: Trace
    probs: Array  # what comes next
    sizes: Array  # how strongly each input position could sway the top prediction

    @property
    def top(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def runner_up(self) -> int:
        """The likeliest letter after the top one (not the end of the word)."""
        order = [int(i) for i in np.argsort(-self.probs) if int(i) not in (self.top, 0)]
        return order[0]


def surprise_gradient(trace: Trace, probs: Array, target: int) -> Array:
    """The gradient of -log p(target) at the last position: probabilities minus the one-hot."""
    dlogits = np.zeros_like(trace.logits)
    dlogits[0, -1] = probs
    dlogits[0, -1, target] -= 1.0
    return dlogits


def influence(
    model: Transformer, vocab: Vocab, prefix: str, target: int | None = None
) -> Influence:
    """Backpropagate from one prediction to the input vectors, leaving the weights alone."""
    ids = np.array([vocab.encode(BOUNDARY + prefix)])
    trace = model.forward(ids)
    probs = softmax(trace.logits[0, -1])
    target = int(np.argmax(probs)) if target is None else target
    _, dx0 = model.backward(trace, surprise_gradient(trace, probs, target))
    return Influence(prefix, trace, probs, np.linalg.norm(dx0[0], axis=-1))


@dataclass(frozen=True, slots=True)
class Nudge:
    target: int
    lr: float
    before: Array
    after: Array


def nudge(model: Transformer, vocab: Vocab, prefix: str, target: int) -> Nudge:
    """One step of plain gradient descent toward `target`, on a copy of the model.

    Uses the smallest learning rate from a short ladder that makes the change easy to see.
    """
    ids = np.array([vocab.encode(BOUNDARY + prefix)])
    trace = model.forward(ids)
    before = softmax(trace.logits[0, -1])
    after, lr = before, 0.0
    for lr in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
        copy = model.copy()
        grads, _ = copy.backward(trace, surprise_gradient(trace, before, target))
        for name, w in copy.params.items():
            w -= lr * grads[name]
        after = softmax(copy.forward(ids).logits[0, -1])
        if after[target] >= 0.2:
            break
    return Nudge(target, lr, before, after)
