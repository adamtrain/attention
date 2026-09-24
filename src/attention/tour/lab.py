"""Everything the tour builds up as it goes: the corpus, the model, how training went."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..corpus import Corpus, Dataset, Vocab
from ..model import Transformer
from ..store import Card, Saved, default_path, pick_name, save
from ..train import STEPS, Baselines, Trainer, baselines, evaluate, prepare


@dataclass
class Lab:
    corpus: Corpus
    seed: int
    trainer: Trainer
    baselines: Baselines
    initial: Transformer  # the untrained model, kept for before-and-after comparisons
    rng: np.random.Generator  # for the tour's own dice rolls, separate from training's
    model_path: Path = field(default_factory=default_path)
    name: str = ""
    seconds: float = 0.0
    saved_to: Path | None = None
    rolls: dict[int, int] = field(default_factory=dict)  # how often each dice stream was used

    @classmethod
    def create(
        cls, corpus: Corpus, seed: int, model_path: Path | None = None, steps: int = STEPS
    ) -> Lab:
        trainer = prepare(corpus, seed, steps)
        return cls(
            corpus=corpus,
            seed=seed,
            trainer=trainer,
            baselines=baselines(trainer.data),
            initial=trainer.model.copy(),
            rng=np.random.default_rng([seed, 3]),
            model_path=model_path or default_path(),
        )

    def dice(self, stream: int) -> np.random.Generator:
        """Random numbers for one job: the same on a seed's first roll, different on a replay."""
        n = self.rolls.get(stream, 0)
        self.rolls[stream] = n + 1
        return np.random.default_rng([self.seed, stream, n] if n else [self.seed, stream])

    def reseed(self) -> None:
        """Start over with a fresh, untrained model, from new random numbers."""
        seed = self.seed
        while seed == self.seed:
            seed = int(np.random.default_rng().integers(1, 10_000))
        self.seed = seed
        self.trainer = prepare(self.corpus, seed, self.trainer.steps)
        self.baselines = baselines(self.trainer.data)
        self.initial = self.trainer.model.copy()
        self.rolls.clear()

    @property
    def model(self) -> Transformer:
        return self.trainer.model

    @property
    def data(self) -> Dataset:
        return self.trainer.data

    @property
    def vocab(self) -> Vocab:
        return self.trainer.data.vocab

    @property
    def trained(self) -> bool:
        return self.trainer.done

    def example(self) -> tuple[np.ndarray, np.ndarray]:
        """The example word as input and target ids, each shaped (1, time)."""
        return self.data.batch([self.corpus.example])

    def probe(self) -> np.ndarray:
        """The probe prefix as input ids, starting with `.`, shaped (1, time)."""
        return np.array([self.vocab.encode("." + self.corpus.probe)])

    def train_quietly(self) -> None:
        start = time.perf_counter()
        while not self.trainer.done:
            self.trainer.step()
            if self.trainer.step_number % 50 == 0 or self.trainer.done:
                self.trainer.evaluate()
        self.seconds = time.perf_counter() - start
        self.finish()

    def finish(self) -> Path:
        """Name the trained model and save it."""
        # Its own dice, so the same seed always gives the same name.
        namer = np.random.default_rng([self.seed, 4])
        self.name = pick_name(self.model, self.vocab, namer, self.corpus)
        val = self.trainer.val_losses[-1][1] if self.trainer.val_losses else float("nan")
        card = Card(
            name=self.name,
            corpus=self.corpus.key,
            title=self.corpus.title,
            noun=self.corpus.noun,
            seed=self.seed,
            steps=self.trainer.step_number,
            loss=evaluate(self.model, self.data, self.data.train),
            val_loss=val,
            pair_loss=self.baselines.pairs,
            trained_at=Card.now(),
            probe=self.corpus.probe,
        )
        self.saved_to = save(
            self.model_path, Saved(self.model, self.vocab, card, self.corpus.words)
        )
        return self.saved_to
