"""Chapter: softmax, and temperature."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import softmax
from ..views import token_chip
from .code import excerpt
from .lab import Lab
from .stage import Frame, Stage, hold

LETTERS = "aeoiu"
SCORES = np.array([2.0, 1.0, 0.2, -0.5, -1.5])


def run(stage: Stage, lab: Lab) -> None:
    stage.say(
        "Attention used softmax to turn scores into weights, and the model's final guess will "
        "use it too, so it's worth a closer look. [b]Softmax[/b] turns any list of numbers "
        "into probabilities, in two moves:"
    )
    stage.say(
        "[accent]1.[/] Raise [i]e[/i] (2.718…) to the power of each number. That makes "
        "everything positive, and stretches the big ones much further apart.\n"
        "[accent]2.[/] Divide each result by the total, so they add up to 100%.",
    )
    stage.say("Say the model scored five letters like this:", gap=False)
    stage.console.print()
    stage.play(steps(lab), fps=1, start="work it out")
    stage.wait()

    stage.say(
        "One more knob. Dividing the scores by a [b]temperature[/b] before softmax changes how "
        "bold the choices are. Below 1, the favorite gets even more likely. Above 1, the "
        "long shots catch up. Watch:"
    )
    stage.play(temperatures(lab), fps=20, start="turn the dial")
    stage.say(
        "You'll get to play with the temperature once your model is trained. Here's softmax "
        "itself, as your model runs it:"
    )
    stage.show(excerpt(softmax))


def ids(lab: Lab) -> list[int]:
    return [
        lab.vocab.chars.index(ch) if ch in lab.vocab.chars else i + 1
        for i, ch in enumerate(LETTERS)
    ]


def table(lab: Lab, shown: int) -> Table:
    exp = np.exp(SCORES)
    probs = exp / exp.sum()
    grid = Table.grid(padding=(0, 2))
    for _ in range(4):
        grid.add_column(no_wrap=True)
    heads = ["", "score", "e^score", "probability"]
    grid.add_row(*(Text(h, style=viz.FAINT) for h in heads[: shown + 1]))
    for i, token in enumerate(ids(lab)):
        cols = [
            token_chip(lab.vocab, token),
            Text.assemble(
                (f"{SCORES[i]:>5.1f} ", "bold"),
                viz.signed_bar(SCORES[i], 2.0, 6, viz.BLUE, viz.AMBER),
            ),
            Text.assemble(
                (f"{exp[i]:>5.2f} ", "bold"),
                viz.bar(exp[i] / exp.max(), 14, viz.PURPLE),
            ),
            Text.assemble((f"{probs[i]:>4.0%} ", "bold"), viz.bar(probs[i], 20, viz.ACCENT)),
        ]
        grid.add_row(*cols[: shown + 1])
    if shown >= 2:
        total = Text.assemble(("total ", viz.FAINT), (f"{exp.sum():.2f}", "bold"))
        grid.add_row(
            "",
            "",
            total,
            *([Text("total 100%", style=viz.FAINT)] if shown >= 3 else []),
        )
    return grid


def steps(lab: Lab) -> Iterator[Frame]:
    yield hold(table(lab, 1), 1.2)
    yield hold(table(lab, 2), 1.6)
    yield hold(table(lab, 3), 0.1)


def temperatures(lab: Lab) -> Iterator[Frame]:
    path = np.concatenate(
        [np.linspace(1.0, 0.25, 40), [0.25] * 20, np.linspace(0.25, 3.0, 70), [3.0] * 20,
         np.linspace(3.0, 1.0, 40), [1.0] * 5]
    )  # fmt: skip
    for t in path:
        yield warmth(lab, float(t))


def warmth(lab: Lab, t: float) -> Group:
    probs = softmax(SCORES / t)
    lo, hi = 0.25, 3.0
    pos = (np.log(t) - np.log(lo)) / (np.log(hi) - np.log(lo))
    slider = Text(no_wrap=True)
    slider.append("temperature ", style=viz.FAINT)
    slider.append(f"{t:4.2f} ", style=f"bold {viz.AMBER}")
    cells = 30
    at = round(pos * (cells - 1))
    for i in range(cells):
        slider.append(
            "●" if i == at else "━",
            style=viz.AMBER if i == at else viz.mix(viz.BLUE, viz.RED, i / cells),
        )
    slider.append("  cold → hot", style=viz.FAINT)
    grid = Table.grid(padding=(0, 1))
    for _ in range(3):
        grid.add_column(no_wrap=True)
    for i, token in enumerate(ids(lab)):
        grid.add_row(
            token_chip(lab.vocab, token),
            viz.bar(float(probs[i]), 40, viz.ACCENT),
            Text(f"{probs[i]:>4.0%}"),
        )
    mood = "plays it safe" if t < 0.7 else "takes chances" if t > 1.5 else "as trained"
    return Group(slider, Text(""), grid, Text(""), Text(f"  {mood}", style=f"italic {viz.FAINT}"))
