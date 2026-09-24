"""The title screen, and choosing what the model will learn."""

from __future__ import annotations

import numpy as np
from rich.table import Table
from rich.text import Text

from .. import corpus as corpora
from .. import viz
from ..corpus import Corpus
from .stage import Stage, typing

TAGLINE = "a tiny transformer, built and trained in front of you"


def title(stage: Stage) -> None:
    stage.clear()
    stage.console.print()
    stage.show(viz.wordmark(), gap=False)
    stage.play(typing(TAGLINE, f"italic {viz.FAINT}", 0.02))
    stage.say(
        "Large language models like ChatGPT and Claude write by doing one thing over and over: "
        "guessing what comes next. In the next few minutes you'll build a very small one, "
        "watch it learn, and keep it."
    )
    stage.say(
        "Yours will have a little over 4,000 [b]parameters[/b], the numbers a model learns. "
        "GPT-3 has 175 billion, some 40 million times as many. Yours will train in about a "
        "second, right here on this computer, though we'll slow it down so you can watch."
    )


def choose(stage: Stage, rng: np.random.Generator) -> Corpus:
    options = [corpora.built_in(key) for key in corpora.BUILT_IN]
    stage.say("What should it learn to invent?", gap=False)
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


def chosen(stage: Stage, corpus: Corpus, seed: int) -> None:
    stage.show(
        Text.assemble(
            ("◇ ", viz.ACCENT),
            (corpus.title, f"bold {viz.ACCENT}"),
            (f" · {corpus.blurb} · seed {seed}", viz.FAINT),
        )
    )
    stage.say(
        f"Good choice. Every run starts from different random numbers, so your {corpus.noun} "
        "model will be one of a kind."
    )
