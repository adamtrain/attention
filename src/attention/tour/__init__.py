"""The guided tour: build a transformer, train it, and take it home."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import corpus as corpora
from .. import viz
from . import (
    attention,
    backprop,
    descent,
    embeddings,
    finale,
    finetuning,
    forward,
    inference,
    learned,
    loss,
    memorizing,
    microscope,
    pretraining,
    softmax,
    tokens,
    welcome,
)
from .lab import Lab
from .stage import Quit, Stage


@dataclass(frozen=True, slots=True)
class Chapter:
    title: str
    subtitle: str
    run: Callable[[Stage, Lab], None]
    needs_training: bool = False


CHAPTERS = [
    Chapter("Tokens", "Text becomes numbers", tokens.run),
    Chapter("Embeddings", "Numbers become vectors", embeddings.run),
    Chapter("Attention", "Looking back at what came before", attention.run),
    Chapter("Softmax", "Scores become probabilities", softmax.run),
    Chapter("The whole model", "One forward pass, top to bottom", forward.run),
    Chapter("Loss", "How wrong was it?", loss.run),
    Chapter("Backpropagation", "Which way should each weight move?", backprop.run),
    Chapter("Gradient descent", "Walking downhill", descent.run),
    Chapter("Pretraining", "Watch it learn", pretraining.run),
    Chapter("Memorizing", "Too big, too long", memorizing.run, needs_training=True),
    Chapter("What it learned", "Looking inside", learned.run, needs_training=True),
    Chapter("Inference", "Writing, one letter at a time", inference.run, needs_training=True),
    Chapter("Under the microscope", "Backprop after training", microscope.run, needs_training=True),
    Chapter(
        "Fine-tuning",
        "From a pretrained model to an assistant",
        finetuning.run,
        needs_training=True,
    ),
    Chapter("Your model", "Take it home", finale.run, needs_training=True),
]


@dataclass(slots=True)
class Outcome:
    lab: Lab | None
    reached: int  # the chapter the viewer got to
    finished: bool


def run(
    stage: Stage,
    *,
    corpus: str | None = None,
    seed: int | None = None,
    start: int = 1,
    model_path: Path | None = None,
) -> Outcome:
    """Play the tour from chapter `start`, until the end or until the viewer presses q."""
    seed = int(np.random.default_rng().integers(1, 10_000)) if seed is None else seed
    lab: Lab | None = None
    number = start
    try:
        if start <= 1:
            welcome.title(stage)
        if corpus:
            chosen = corpora.load(corpus)
        elif start <= 1:
            chosen = welcome.choose(stage, np.random.default_rng([seed, 9]))
        else:
            chosen = corpora.built_in(list(corpora.BUILT_IN)[seed % len(corpora.BUILT_IN)])
        lab = Lab.create(chosen, seed, model_path)
        if start <= 1:
            welcome.chosen(stage, chosen, seed)
            stage.wait("begin")
        for number, chapter in enumerate(CHAPTERS, 1):
            if number < start:
                continue
            if chapter.needs_training and not lab.trained:
                with stage.console.status("Training a model to look at…", spinner_style=viz.ACCENT):
                    lab.train_quietly()
            stage.clear()
            stage.header(number, len(CHAPTERS), chapter.title, chapter.subtitle)
            chapter.run(stage, lab)
            if number < len(CHAPTERS):
                stage.wait(f"continue to {CHAPTERS[number].title}")
        return Outcome(lab, len(CHAPTERS), True)
    except Quit:
        return Outcome(lab, number, False)
