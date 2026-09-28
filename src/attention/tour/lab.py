"""Everything the tour builds up as it goes: the corpus, the model, how training went."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .. import finetune, induction
from ..corpus import VOCAB, Corpus, Dataset
from ..generate import Memory, prompt
from ..model import Config, Transformer
from ..store import Card, Outdated, Saved, default_path, load, pick_name, save
from ..tokenizer import Tokenizer
from ..train import Baselines, Trainer, baselines, evaluate, prepare


@dataclass(frozen=True, slots=True)
class Budget:
    """How much training the tour does. The tests use a tiny one."""

    config: Config | None = None  # None: the real model's size
    steps: int | None = None  # None: enough for the text (see train.steps_for)
    memorizing: int = 1200
    race: int = induction.STEPS
    tuning: int = finetune.STEPS
    chat: int = finetune.CHAT_STEPS
    vocab: int = VOCAB  # tokens for the tokenizer to learn, and the model to score

    @classmethod
    def tiny(cls, vocab: int = 512) -> Budget:
        config = Config(vocab, 64, width=32, layers=2, heads=4, kv_heads=2, hidden=64)
        return cls(config, steps=40, memorizing=40, race=30, tuning=10, chat=10, vocab=vocab)


@dataclass
class Lab:
    corpus: Corpus
    seed: int
    trainer: Trainer
    baselines: Baselines
    initial: Transformer  # the untrained model, kept for before-and-after comparisons
    rng: np.random.Generator  # for the tour's own dice rolls, separate from training's
    budget: Budget = field(default_factory=Budget)
    model_path: Path = field(default_factory=default_path)
    name: str = ""
    seconds: float = 0.0
    saved_to: Path | None = None
    rolls: dict[int, int] = field(default_factory=dict)  # how often each dice stream was used

    @classmethod
    def create(
        cls, corpus: Corpus, seed: int, model_path: Path | None = None, budget: Budget | None = None
    ) -> Lab:
        budget = budget or Budget()
        trainer = prepare(corpus, seed, budget.steps, budget.config, budget.vocab)
        return cls(
            corpus=corpus,
            seed=seed,
            trainer=trainer,
            baselines=baselines(trainer.data),
            initial=trainer.model.copy(),
            rng=np.random.default_rng([seed, 3]),
            budget=budget,
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
        self.trainer = prepare(
            self.corpus, seed, self.budget.steps, self.budget.config, self.budget.vocab
        )
        self.baselines = baselines(self.trainer.data)
        self.initial = self.trainer.model.copy()
        self.rolls.clear()
        self.__dict__.pop("memory", None)

    @property
    def model(self) -> Transformer:
        return self.trainer.model

    @property
    def data(self) -> Dataset:
        return self.trainer.data

    @property
    def tokenizer(self) -> Tokenizer:
        return self.trainer.data.tokenizer

    @property
    def trained(self) -> bool:
        return self.trainer.done

    @cached_property
    def memory(self) -> Memory:
        """The training text's runs of tokens, to spot copying."""
        return Memory(self.data.train)

    def example(self) -> tuple[np.ndarray, np.ndarray]:
        """The example passage as input and target ids, each shaped (1, time)."""
        ids = prompt(self.tokenizer, self.corpus.example)
        return np.array([ids[:-1]]), np.array([ids[1:]])

    def probe(self) -> list[int]:
        """The probe's ids, starting with <|endoftext|>."""
        return prompt(self.tokenizer, self.corpus.probe)

    def train_quietly(self, progress: Callable[[int, int], None] | None = None) -> None:
        start = time.perf_counter()
        while not self.trainer.done:
            self.trainer.step()
            if self.trainer.step_number % 100 == 0 or self.trainer.done:
                self.trainer.evaluate()
            if progress:
                progress(self.trainer.step_number, self.trainer.steps)
        self.seconds = time.perf_counter() - start
        self.finish()

    def restore(self) -> bool:
        """Pick up the saved model instead of training again, if it's this corpus and seed.

        The same seed and corpus always make the same model, so there's no need to wait for it.
        """
        try:
            saved = load(self.model_path)
        except (OSError, ValueError, KeyError, TypeError, Outdated):
            return False
        card = saved.card
        same = (card.corpus, card.seed, card.steps) == (
            self.corpus.key,
            self.seed,
            self.trainer.steps,
        )
        if not same or saved.tokenizer.pieces != self.tokenizer.pieces:
            return False
        if saved.model.config != self.model.config:
            return False
        self.trainer.model.params = saved.model.params
        self.trainer.step_number = self.trainer.steps
        self.trainer.val_losses.append((self.trainer.steps, card.val_loss))
        self.name = card.name
        self.saved_to = self.model_path
        return True

    def finish(self) -> Path:
        """Name the trained model and save it."""
        # Its own dice, so the same seed always gives the same name.
        namer = np.random.default_rng([self.seed, 4])
        self.name = pick_name(self.model, self.tokenizer, namer, self.corpus)
        val = self.trainer.val_losses[-1][1] if self.trainer.val_losses else float("nan")
        card = Card(
            name=self.name,
            corpus=self.corpus.key,
            title=self.corpus.title,
            noun=self.corpus.noun,
            plural=self.corpus.plural,
            seed=self.seed,
            steps=self.trainer.step_number,
            loss=evaluate(self.model, *self.data.fixed(48, self.model.config.context, False)),
            val_loss=val,
            pair_loss=self.baselines.pairs,
            trained_at=Card.now(),
            probe=self.corpus.probe,
        )
        saved = Saved(self.model, self.tokenizer, card, self.data.train_docs)
        self.saved_to = save(self.model_path, saved)
        return self.saved_to
