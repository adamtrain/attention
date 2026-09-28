"""The guided tour: build a transformer, train it, look inside it, and take it home."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from .. import corpus as corpora
from .. import viz
from . import (
    attention,
    backprop,
    cache,
    chat,
    descent,
    embeddings,
    finale,
    finetuning,
    inference,
    layers,
    learned,
    loss,
    memorizing,
    microscope,
    position,
    pretraining,
    quantization,
    softmax,
    stack,
    tokens,
    welcome,
)
from .lab import Budget, Lab
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
    Chapter("Attention", "Queries, keys and values", attention.run),
    Chapter("Position", "Where am I?", position.run),
    Chapter("Softmax", "Scores become probabilities", softmax.run),
    Chapter("The stack", "Layer after layer", stack.run),
    Chapter("Loss", "How wrong was it?", loss.run),
    Chapter("Backpropagation", "Which way should each weight move?", backprop.run),
    Chapter("Gradient descent", "Walking downhill", descent.run),
    Chapter("Pretraining", "Watch it learn", pretraining.run),
    Chapter("Memorizing", "Too little to read, for too long", memorizing.run, needs_training=True),
    Chapter("What it learned", "Looking inside", learned.run, needs_training=True),
    Chapter("Why layers stack", "Two can do what one can't", layers.run, needs_training=True),
    Chapter("Inference", "Writing, one token at a time", inference.run, needs_training=True),
    Chapter("The KV cache", "Remembering instead of rereading", cache.run, needs_training=True),
    Chapter("Under the microscope", "Backprop after training", microscope.run, needs_training=True),
    Chapter("Quantization", "Fewer bits per weight", quantization.run, needs_training=True),
    Chapter("Fine-tuning", "Teaching it a speciality", finetuning.run, needs_training=True),
    Chapter("Chat", "From predicting text to answering", chat.run, needs_training=True),
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
    budget: Budget | None = None,
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
        lab = Lab.create(chosen, seed, model_path, budget)
        if start <= 1:
            welcome.chosen(stage, lab)
            stage.wait("begin")
        for number, chapter in enumerate(CHAPTERS, 1):
            if number < start:
                continue
            if chapter.needs_training and not lab.trained and not lab.restore():
                catch_up(stage, lab)
            stage.clear()
            stage.header(number, len(CHAPTERS), chapter.title, chapter.subtitle)
            chapter.run(stage, lab)
            if number < len(CHAPTERS):
                stage.wait(f"continue to {CHAPTERS[number].title}")
        return Outcome(lab, len(CHAPTERS), True)
    except Quit:
        return Outcome(lab, number, False)


def catch_up(stage: Stage, lab: Lab) -> None:
    """Train the model the later chapters need, with a progress bar, when starting part way."""
    columns = (
        TextColumn("  Training a model to look at", style=viz.FAINT),
        BarColumn(complete_style=viz.ACCENT, finished_style=viz.GREEN),
        TextColumn("{task.completed}/{task.total} steps", style=viz.FAINT),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=stage.console, transient=True) as progress:
        task = progress.add_task("training", total=lab.trainer.steps)
        lab.train_quietly(lambda done, _: progress.update(task, completed=done))
