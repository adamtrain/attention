"""The title screen, and choosing what the model will learn."""

from __future__ import annotations

import numpy as np
from rich.table import Table
from rich.text import Text

from .. import corpus as corpora
from .. import viz
from ..corpus import Corpus
from ..model import Config, Transformer
from ..train import CONTEXT
from .lab import Lab
from .stage import Stage, typing

TAGLINE = "a small language model, built and trained in front of you"


def title(stage: Stage) -> None:
    stage.clear()
    stage.console.print()
    stage.show(viz.wordmark(), gap=False)
    stage.play(lambda: typing(TAGLINE, f"italic {viz.FAINT}", 0.02), start=None, then=None)
    stage.say(
        "Large language models like ChatGPT and Claude write by doing one thing over and over: "
        "guessing what comes next. In this tour you'll build a small one, the way the real ones "
        "are built: the same kind of tokens, the same stack of layers, the same attention, the "
        "same training. You'll watch it learn, look inside it, and keep it."
    )
    size = Transformer.create(Config(corpora.VOCAB, CONTEXT), np.random.default_rng(0)).size
    stage.say(
        f"Yours will have about {round(size, -4):,} [b]parameters[/b], the numbers a model "
        "learns. GPT-3 has 175 billion, and today's largest models have more than a trillion. "
        "Yours will train "
        "for a few minutes, right here on this computer, and you'll see every number "
        "that matters along the way."
    )


def choose(stage: Stage, rng: np.random.Generator) -> Corpus:
    options = [corpora.built_in(key) for key in corpora.BUILT_IN]
    stage.say("What should it learn to write?", gap=False)
    stage.console.print()
    menu = Table.grid(padding=(0, 2))
    menu.add_column(style=f"bold {viz.ACCENT}", justify="right")
    menu.add_column(style="bold")
    menu.add_column(style=viz.FAINT)
    for i, c in enumerate(options, 1):
        menu.add_row(str(i), c.title, c.blurb)
    menu.add_row("␣", "Surprise me", "")
    stage.show(menu)

    picked = None
    if stage.keys and stage.auto is None:
        while picked is None:
            key = stage.wait("be surprised")
            if key and key.isdigit() and 1 <= int(key) <= len(options):
                picked = options[int(key) - 1]
            elif key in ("space", "enter", "right"):
                break
    return picked or options[int(rng.integers(len(options)))]


def chosen(stage: Stage, lab: Lab) -> None:
    corpus = lab.corpus
    stage.show(
        Text.assemble(
            ("◇ ", viz.ACCENT),
            (corpus.title, f"bold {viz.ACCENT}"),
            (f" · {corpus.blurb} · {corpus.characters:,} characters · seed {lab.seed}", viz.FAINT),
        )
    )
    stage.say(
        f"Good choice. Every run starts from different random numbers, so your {corpus.noun} "
        "model will be one of a kind."
    )
