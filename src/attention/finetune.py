"""After pretraining: fine-tuning on a few examples, and nudging a model with feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import Dataset, Niche
from .generate import log_prob
from .model import Array, Transformer, cross_entropy
from .train import Adam, Trainer

STEPS = 60
LEARNING_RATE = 0.003  # gentler than pretraining: we only want to adjust what it already knows


def niche_trainer(
    model: Transformer, data: Dataset, niche: Niche, rng: np.random.Generator
) -> Trainer:
    """A trainer that shows a copy of the model only the words in one niche."""
    words = tuple(w for w in data.train if niche.has(w))
    narrow = Dataset(data.vocab, words, data.val)
    return Trainer(
        model.copy(),
        narrow,
        rng,
        steps=STEPS,
        batch_size=min(16, len(words)),
        lr=LEARNING_RATE,
    )


def drift(model: Transformer, start: Transformer) -> float:
    """How far a model's weights have moved from where they started, relative to their size."""
    moved = sum(float(((model.params[k] - start.params[k]) ** 2).sum()) for k in start.params)
    size = sum(float((start.params[k] ** 2).sum()) for k in start.params)
    return (moved / size) ** 0.5


def share(words: list[str], niche: Niche) -> float:
    return sum(niche.has(w) for w in words) / max(len(words), 1)


@dataclass(frozen=True, slots=True)
class Feedback:
    """One round of learning from what someone liked, and how each word's odds changed."""

    model: Transformer
    before: dict[str, float]  # log-probability of each whole word, before and after
    after: dict[str, float]

    def odds(self, word: str) -> tuple[float, float]:
        """A word's chance of being written, as 1 in N, before and after."""
        return float(np.exp(-self.before[word])), float(np.exp(-self.after[word]))


def learn_from(
    model: Transformer,
    data: Dataset,
    liked: list[str],
    disliked: list[str],
    steps: int = 3,
    lr: float = 0.002,
    push_away: float = 0.2,
) -> Feedback:
    """Make the liked words likelier and the rest a little less likely, on a copy.

    Liked words are trained on like any example. Disliked ones get the opposite push, but
    gently: pushing hard away from things quickly breaks a model.
    """
    copy = model.copy()
    vocab = data.vocab
    words = liked + disliked
    before = {w: float(log_prob(copy, vocab, w).sum()) for w in words}
    adam = Adam(copy.params)
    for _ in range(steps):
        total: dict[str, Array] = {k: np.zeros_like(v) for k, v in copy.params.items()}
        for group, sign in ((liked, 1.0), (disliked, -push_away)):
            if not group:
                continue
            inputs, targets = data.batch(group)
            trace = copy.forward(inputs)
            _, dlogits = cross_entropy(trace.logits, targets)
            grads, _ = copy.backward(trace, dlogits)
            for name in total:
                total[name] += sign * grads[name]
        adam.step(copy.params, total, lr)
    after = {w: float(log_prob(copy, vocab, w).sum()) for w in words}
    return Feedback(copy, before, after)
