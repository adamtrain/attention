"""Pretraining: show the model words, measure its surprise, and nudge every weight downhill."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .corpus import Corpus, Dataset
from .model import LAYERS, PARTS, Array, Config, Transformer, cross_entropy

STEPS = 400
BATCH = 32
LEARNING_RATE = 0.01


class Adam:
    """Gradient descent with two refinements that make it much faster in practice.

    Momentum: each weight keeps a running average of its recent gradients, so noise from one
    batch to the next cancels out. Scale: each weight also tracks how big its gradients tend to
    be, and steps by roughly the same amount whether they're large or tiny.
    """

    def __init__(self, params: dict[str, Array], betas: tuple[float, float] = (0.9, 0.99)):
        self.b1, self.b2 = betas
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(
        self, params: dict[str, Array], grads: dict[str, Array], lr: float
    ) -> dict[str, Array]:
        """Move every weight a little way downhill. Returns how much each one moved."""
        self.t += 1
        moved = {}
        for name, w in params.items():
            g = grads[name]
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * g * g
            m = self.m[name] / (1 - self.b1**self.t)
            v = self.v[name] / (1 - self.b2**self.t)
            moved[name] = -lr * m / (np.sqrt(v) + 1e-8)
            w += moved[name]
        return moved


@dataclass(slots=True)
class Step:
    """What happened in one step of training."""

    number: int
    loss: float
    lr: float
    grads: dict[str, Array]
    moved: dict[str, Array]

    def layer_norms(self) -> dict[str, float]:
        """How big the gradient was for each layer: how hard backprop pushed on it."""
        totals = dict.fromkeys(LAYERS, 0.0)
        for name, g in self.grads.items():
            totals[PARTS[name].layer] += float((g * g).sum())
        return {layer: total**0.5 for layer, total in totals.items()}


@dataclass(slots=True)
class Trainer:
    model: Transformer
    data: Dataset
    rng: np.random.Generator
    steps: int = STEPS
    batch_size: int = BATCH
    lr: float = LEARNING_RATE
    step_number: int = 0
    losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    optimizer: Adam = field(init=False)

    def __post_init__(self) -> None:
        self.optimizer = Adam(self.model.params)

    @property
    def done(self) -> bool:
        return self.step_number >= self.steps

    def learning_rate(self) -> float:
        """Big steps at first, smaller ones as the model settles in."""
        warmup = 20
        if self.step_number < warmup:
            return self.lr * (self.step_number + 1) / warmup
        return self.lr * (1 - 0.9 * self.step_number / self.steps)

    def step(self) -> Step:
        picks = self.rng.choice(len(self.data.train), size=self.batch_size, replace=False)
        inputs, targets = self.data.batch([self.data.train[i] for i in picks])
        trace = self.model.forward(inputs)
        loss, dlogits = cross_entropy(trace.logits, targets)
        grads, _ = self.model.backward(trace, dlogits)
        lr = self.learning_rate()
        moved = self.optimizer.step(self.model.params, grads, lr)
        self.step_number += 1
        self.losses.append(loss)
        return Step(self.step_number, loss, lr, grads, moved)

    def evaluate(self) -> float:
        """Loss on the held-back words, which the model never trains on."""
        loss = evaluate(self.model, self.data, self.data.val)
        self.val_losses.append((self.step_number, loss))
        return loss


def prepare(corpus: Corpus, seed: int, steps: int = STEPS) -> Trainer:
    """A fresh, untrained model and everything needed to train it.

    The seed decides the held-back words, the starting weights and the order of the batches,
    so the same seed and corpus always make the same model.
    """
    data = Dataset.split(corpus, np.random.default_rng([seed, 0]))
    config = Config(vocab=len(data.vocab), context=data.context)
    model = Transformer.create(config, np.random.default_rng([seed, 1]))
    return Trainer(model, data, np.random.default_rng([seed, 2]), steps=steps)


def evaluate(model: Transformer, data: Dataset, words: tuple[str, ...]) -> float:
    inputs, targets = data.batch(words)
    return cross_entropy(model.forward(inputs).logits, targets)[0]


def smooth(values: list[float], alpha: float = 0.05) -> list[float]:
    """An exponential moving average, to see the trend through the batch-to-batch noise."""
    out, avg = [], values[0] if values else 0.0
    for v in values:
        avg = alpha * v + (1 - alpha) * avg
        out.append(avg)
    return out


# ── Baselines ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Baselines:
    """The loss you'd get without a neural network at all, for comparison."""

    uniform: float  # every character equally likely
    letters: float  # knowing how common each character is
    pairs: float  # knowing which character tends to follow which


def baselines(data: Dataset) -> Baselines:
    vocab = data.vocab
    seqs = [vocab.sequence(w) for w in data.train]
    held = [vocab.sequence(w) for w in data.val]
    n = len(vocab)

    single = Counter(t for s in seqs for t in s[1:])
    probs = np.array([single[i] + 1 for i in range(n)], dtype=float)
    probs /= probs.sum()
    letters = -np.mean([np.log(probs[t]) for s in held for t in s[1:]])

    pairs = np.ones((n, n))
    for s in seqs:
        for a, b in pairwise(s):
            pairs[a, b] += 1
    pairs /= pairs.sum(axis=1, keepdims=True)
    pair_loss = -np.mean([np.log(pairs[a, b]) for s in held for a, b in pairwise(s)])

    return Baselines(float(np.log(n)), float(letters), float(pair_loss))
