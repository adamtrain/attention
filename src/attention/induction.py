"""Why stack layers? An experiment in which two layers can do something one layer can't.

The task: random tokens, with one stretch of them repeated later on, at a random distance. The
models are graded only inside the repeat. To get those right, a position has to find where its
own token turned up before, and pass on the token that came *after* it there. That takes two
steps, one per layer, and the pair of heads that learn to do it is called an induction head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Array, Config, Transformer, cross_entropy
from .train import Adam, clip

LENGTH = 32  # tokens per sequence
STEPS = 2000
BATCH = 32
LEARNING_RATE = 0.01
EVERY = 25  # steps between check-ups


@dataclass(frozen=True, slots=True)
class Batch:
    inputs: Array  # (count, LENGTH) tokens; 0 starts each row
    targets: Array  # the next token inside each repeat, and -1 everywhere else
    gaps: Array  # (count,) how far back each row's first copy is


def sequences(rng: np.random.Generator, count: int, vocab: int) -> Batch:
    """Random tokens, with a stretch of 6 to 10 of them repeated further on.

    Only the repeat is graded, after its first token: the only predictions that can be worked
    out from what came before.
    """
    x = rng.integers(1, vocab, (count, LENGTH + 1))
    x[:, 0] = 0
    y = np.full((count, LENGTH), -1, dtype=np.int64)
    gaps = np.zeros(count, dtype=np.int64)
    for row in range(count):
        n = int(rng.integers(6, 11))
        first = int(rng.integers(1, LENGTH // 2 - n + 1))
        again = int(rng.integers(first + n + 1, LENGTH - n + 2))
        x[row, again : again + n] = x[row, first : first + n]
        y[row, again : again + n - 1] = x[row, again + 1 : again + n]
        gaps[row] = again - first
    return Batch(x[:, :-1], y, gaps)


def small(vocab: int, layers: int) -> Config:
    """The same design as your model, much smaller: two heads per layer, 32 numbers per token."""
    return Config(vocab, LENGTH, width=32, layers=layers, heads=2, kv_heads=2, hidden=64)


@dataclass
class Learner:
    model: Transformer
    optimizer: Adam
    history: list[tuple[int, float]] = field(default_factory=list)  # (step, loss on the repeats)

    @property
    def loss(self) -> float:
        return self.history[-1][1]


@dataclass
class Race:
    """A one-layer model and a two-layer model, trained side by side on the same batches."""

    vocab: int
    rng: np.random.Generator
    steps: int = STEPS
    learners: list[Learner] = field(default_factory=list)
    test: Batch = field(init=False)
    step_number: int = 0

    def __post_init__(self) -> None:
        self.test = sequences(np.random.default_rng(99), 128, self.vocab)
        for layers in (1, 2):
            model = Transformer.create(small(self.vocab, layers), self.rng)
            self.learners.append(Learner(model, Adam(model.params)))
        self.check()

    @property
    def done(self) -> bool:
        return self.step_number >= self.steps

    def step(self) -> None:
        batch = sequences(self.rng, BATCH, self.vocab)
        lr = LEARNING_RATE * min(1.0, (self.step_number + 1) / 50)
        for learner in self.learners:
            trace = learner.model.forward(batch.inputs)
            _, dlogits = cross_entropy(trace.logits, batch.targets)
            grads, _ = learner.model.backward(trace, dlogits)
            learner.optimizer.step(learner.model.params, clip(grads, 1.0)[0], lr)
        self.step_number += 1
        if self.step_number % EVERY == 0 or self.done:
            self.check()

    def check(self) -> None:
        for learner in self.learners:
            logits = learner.model.forward(self.test.inputs).logits
            loss, _ = cross_entropy(logits, self.test.targets)
            learner.history.append((self.step_number, loss))


# ── Finding the circuit ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Habits:
    """Where each head looks from inside a repeat, averaged: each array is (layers, heads)."""

    back: Array  # (3, layers, heads): at the token 1, 2 and 3 places back
    start: Array  # at the 0 that starts the row
    # (4, layers, heads): at the answer (the token after the first copy), then 1 to 3 past it.
    past: Array

    def previous(self) -> tuple[int, float]:
        """Layer 1's most previous-token head, and how much it looks one back."""
        h = int(np.argmax(self.back[0, 0]))
        return h, float(self.back[0, 0, h])

    def induction(self) -> tuple[int, float]:
        """The last layer's head that looks most at the answer, and how much."""
        h = int(np.argmax(self.past[0, -1]))
        return h, float(self.past[0, -1, h])

    @property
    def textbook(self) -> bool:
        """A previous-token head in layer 1 and a head in layer 2 that looks at the answer."""
        return self.previous()[1] > 0.4 and self.induction()[1] > 0.5


def habits(model: Transformer, batch: Batch) -> Habits:
    weights = model.forward(batch.inputs).weights  # (layers, batch, heads, time, time)
    rows, cols = np.nonzero(batch.targets >= 0)
    answer = cols - batch.gaps[rows] + 1  # just after the same token, in the first copy

    def at(where: Array) -> Array:
        return weights[:, rows, :, cols, where].mean(axis=0)  # (layers, heads)

    back = np.stack([at(cols - k) for k in (1, 2, 3)])
    past = np.stack([at(np.minimum(answer + d, cols)) for d in (0, 1, 2, 3)])
    return Habits(back, at(np.zeros_like(cols)), past)
