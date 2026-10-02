"""Pretraining: show the model text, measure its surprise, and nudge every weight downhill."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .corpus import VOCAB, Corpus, Dataset
from .model import Array, Config, Transformer, cross_entropy, groups, part

PASSES = 16  # how many times, on average, pretraining reads each token of its text
STEPS = (600, 3600)  # the fewest and most steps it will take: the most, about 4 minutes
BATCH = 16  # stretches of text per step
SHARE = 4  # stretches each core takes at a time, when a step's batch is split between cores
CONTEXT = 128  # tokens per stretch, and the most the model can read at once
LEARNING_RATE = 0.01
WARMUP = 100  # steps to ramp the learning rate up from almost nothing
WEIGHT_DECAY = 0.1
CLIP = 1.0  # the largest a step's whole gradient may be, measured as one long vector
EVAL = 48  # held-back stretches to measure progress on

type Batches = Callable[[np.random.Generator], tuple[Array, Array]]

CORES = ThreadPoolExecutor(os.cpu_count())  # its threads start only when there's work for them


class Adam:
    """Gradient descent with two refinements that make it much faster in practice.

    Momentum: each weight keeps a running average of its recent gradients, so noise from one
    batch to the next cancels out. Scale: each weight also tracks how big its gradients tend to
    be, and steps by roughly the same amount whether they're large or tiny.

    With `decay`, every step also shrinks each weight a little toward zero, so only weights
    the gradient keeps pushing on stay big. That's weight decay, and Adam with it is AdamW.
    The norms' weights are left alone: they're scales that start at 1, not patterns.
    """

    def __init__(
        self,
        params: dict[str, Array],
        betas: tuple[float, float] = (0.9, 0.95),
        decay: float = 0.0,
    ):
        self.b1, self.b2 = betas
        self.decay = decay
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
            if self.decay and not name.endswith("norm"):
                moved[name] -= lr * self.decay * w
            w += moved[name]
        return moved


def gradient_size(grads: dict[str, Array]) -> float:
    """Every gradient as one long vector: how long is it?"""
    return float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))


def clip(grads: dict[str, Array], most: float) -> tuple[dict[str, Array], float]:
    """Shrink the whole gradient if it's longer than `most`, keeping its direction.

    Now and then one batch produces a huge gradient, and one huge step can undo hours of
    training. Clipping caps the step's size without changing which way it goes.
    """
    size = gradient_size(grads)
    if size <= most:
        return grads, size
    return {k: g * (most / size) for k, g in grads.items()}, size


def gradients(model: Transformer, inputs: Array, targets: Array) -> tuple[float, dict[str, Array]]:
    """A batch's loss and the gradient for every weight, worked out on several cores at once.

    Nothing passes between one stretch of text and another on the way through the model, so
    the batch can be dealt out, SHARE stretches at a time. Each core runs its share forward
    and backward (NumPy's arithmetic doesn't hold Python's lock, so they really do run at the
    same time), and the gradients are added up. A share counts for as much as it has targets,
    which makes the total the gradient of the whole batch's average loss.

    The batch is dealt the same way however many cores there are, and added up in the same
    order, so the same seed makes the same model on any computer.
    """
    total = (targets >= 0).sum()
    cuts = [slice(i, i + SHARE) for i in range(0, len(inputs), SHARE)]

    def share(cut: slice) -> tuple[float, dict[str, Array]]:
        weight = float((targets[cut] >= 0).sum() / total)
        trace = model.forward(inputs[cut])
        loss, dlogits = cross_entropy(trace.logits, targets[cut])
        grads, _ = model.backward(trace, dlogits * weight)
        return loss * weight, grads

    # A share with nothing to learn from has no average loss to take.
    shares = list(CORES.map(share, [cut for cut in cuts if (targets[cut] >= 0).any()]))
    loss, grads = shares[0]
    for more, extra in shares[1:]:
        loss += more
        for name, g in extra.items():
            grads[name] += g
    return loss, grads


@dataclass(slots=True)
class Step:
    """What happened in one step of training."""

    number: int
    loss: float
    lr: float
    size: float  # how long the gradient was, before clipping
    grads: dict[str, Array]
    moved: dict[str, Array]

    def group_sizes(self, config: Config) -> dict[str, float]:
        """How big the gradient was for each part of the model: how hard backprop pushed on it."""
        totals = dict.fromkeys(groups(config), 0.0)
        for name, g in self.grads.items():
            totals[part(name).group] += float((g * g).sum())
        return {group: total**0.5 for group, total in totals.items()}


def steps_for(data: Dataset, batch: int = BATCH, context: int = CONTEXT) -> int:
    """Enough steps to read the training text about PASSES times, to the nearest hundred.

    More text means more steps. Fewer, and the model would still be learning when it stopped;
    many more, and it would start memorizing the text instead of learning from it. A big text
    hits the ceiling first and is read fewer times, the way big models read most of theirs once.
    """
    steps = round(PASSES * len(data.train) / (batch * context), -2)
    return int(min(max(steps, STEPS[0]), STEPS[1]))


@dataclass(slots=True)
class Trainer:
    model: Transformer
    data: Dataset
    rng: np.random.Generator
    steps: int = STEPS[1]
    batch_size: int = BATCH
    lr: float = LEARNING_RATE
    decay: float = WEIGHT_DECAY
    warmup: int = WARMUP
    most: float = CLIP
    batches: Batches | None = None  # where each step's examples come from (default: the text)
    step_number: int = 0
    losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)  # each step's gradient size
    optimizer: Adam = field(init=False)
    held: tuple[Array, Array] = field(init=False)

    def __post_init__(self) -> None:
        self.optimizer = Adam(self.model.params, decay=self.decay)
        self.held = self.data.fixed(EVAL, self.model.config.context)

    @property
    def done(self) -> bool:
        return self.step_number >= self.steps

    def learning_rate(self) -> float:
        """Ramp up gently, then ease down along a cosine curve to a tenth of the peak."""
        if self.step_number < self.warmup:
            return self.lr * (self.step_number + 1) / self.warmup
        progress = (self.step_number - self.warmup) / max(1, self.steps - self.warmup)
        return self.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0))))

    def batch(self) -> tuple[Array, Array]:
        if self.batches is not None:
            return self.batches(self.rng)
        return self.data.windows(self.rng, self.batch_size, self.model.config.context)

    def step(self) -> Step:
        inputs, targets = self.batch()
        loss, grads = gradients(self.model, inputs, targets)
        clipped, size = clip(grads, self.most)
        lr = self.learning_rate()
        moved = self.optimizer.step(self.model.params, clipped, lr)
        self.step_number += 1
        self.losses.append(loss)
        self.sizes.append(size)
        return Step(self.step_number, loss, lr, size, grads, moved)

    def evaluate(self) -> float:
        """Loss on held-back text, which the model never trains on."""
        loss = evaluate(self.model, *self.held)
        self.val_losses.append((self.step_number, loss))
        return loss


def prepare(
    corpus: Corpus,
    seed: int,
    steps: int | None = None,
    config: Config | None = None,
    vocab: int = VOCAB,
) -> Trainer:
    """A fresh, untrained model and everything needed to train it.

    The seed decides the held-back documents, the starting weights and the order of the
    batches, so the same seed and corpus always make the same model.
    """
    data = Dataset.split(corpus, np.random.default_rng([seed, 0]), vocab=vocab)
    config = config or Config(vocab=len(data.tokenizer), context=CONTEXT)
    model = Transformer.create(config, np.random.default_rng([seed, 1]))
    steps = steps or steps_for(data, context=config.context)
    return Trainer(model, data, np.random.default_rng([seed, 2]), steps=steps)


def evaluate(model: Transformer, inputs: Array, targets: Array, batch: int = SHARE) -> float:
    """The average loss over some text, a few stretches at a time, on several cores at once."""

    def some(i: int) -> tuple[float, int]:
        logits = model.forward(inputs[i : i + batch]).logits
        n = int((targets[i : i + batch] >= 0).sum())
        return cross_entropy(logits, targets[i : i + batch])[0] * n if n else 0.0, n

    totals = list(CORES.map(some, range(0, len(inputs), batch)))
    return sum(loss for loss, _ in totals) / max(sum(n for _, n in totals), 1)


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

    uniform: float  # every token equally likely
    tokens: float  # knowing how common each token is
    pairs: float  # knowing which token tends to follow which


def baselines(data: Dataset) -> Baselines:
    n = len(data.tokenizer)
    train, held = data.train, data.val
    counts = np.bincount(train[1:], minlength=n) + 1.0
    single = -np.log(counts[held[1:]] / counts.sum()).mean()
    pairs = np.ones((n, n))
    np.add.at(pairs, (train[:-1], train[1:]), 1.0)
    pairs /= pairs.sum(axis=1, keepdims=True)
    pair_loss = -np.log(pairs[held[:-1], held[1:]]).mean()
    return Baselines(float(np.log(n)), float(single), float(pair_loss))
