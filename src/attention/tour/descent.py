"""Chapter: gradient descent, on a landscape and on the real model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import cross_entropy
from .lab import Lab
from .stage import Frame, Stage, hold

# A long, narrow valley: steep across (b), gentle along (a).
CENTER = (1.3, -0.25)
START = (-2.8, 1.1)
A_RANGE, B_RANGE = (-3.3, 3.3), (-1.5, 1.5)
RATES = (
    (0.05, "too timid", "crawls"),
    (0.25, "about right", "gets there"),
    (0.49, "too bold", "zigzags"),
)
LAND = viz.Ramp("#12142a", "#232769", "#4a4fb0", viz.PURPLE, "#d9a6ff")


def loss(a: float, b: float) -> float:
    return 0.25 * (a - CENTER[0]) ** 2 + 2.0 * (b - CENTER[1]) ** 2


def slope(a: float, b: float) -> tuple[float, float]:
    return 0.5 * (a - CENTER[0]), 4.0 * (b - CENTER[1])


def run(stage: Stage, lab: Lab) -> None:
    stage.say(
        "Picture the loss as a landscape, where every direction is one of the weights. "
        "Training is a walk downhill in the dark. The gradient says which way is uphill, "
        "right where you stand, so you step the other way:"
    )
    stage.show(Text.assemble(
        ("weight ← weight − ", "bold"), ("learning rate", f"bold {viz.AMBER}"), (" × gradient", "bold")
    ))  # fmt: skip
    stage.say(
        "Here's a landscape with just two weights, [b]a[/b] and [b]b[/b]. Darker is lower. "
        "Three walkers set off from the same spot with different learning rates:"
    )
    stage.play(lambda: race(stage.width), fps=12, start="let them walk")
    stage.say(
        "Too small and training takes forever. Too big and it overshoots, bouncing from one "
        "side of the valley to the other (or flying off completely). Choosing the learning "
        f"rate is one of the fiddliest parts of training. Your model's landscape has "
        f"{lab.model.size:,} directions instead of two, but the walk is the same."
    )
    stage.wait("take one real step")

    stage.say(
        f"Now for real. Take 32 {lab.corpus.plural}, run them forward, backpropagate, and move "
        "every weight in your model one step downhill (learning rate 1). Here's the token "
        "embedding table before the step, the step itself, and after:"
    )
    step = real_step(lab)
    stage.show(step.picture(lab, stage.width))
    stage.say(
        f"The loss on those {lab.corpus.plural} went from [b]{step.before:.2f}[/b] to "
        f"[b][green]{step.after:.2f}[/green][/b]. One small step. Pretraining is this, "
        "over and over."
    )
    stage.note(
        "Your model will actually train with Adam, a refinement of gradient descent that gives "
        "each weight momentum (it keeps rolling the way it has been going) and its own step "
        "size. It's what nearly every language model trains with."
    )


# ── The race ──────────────────────────────────────────────────────────────────


def race(width: int, steps: int = 30) -> Iterator[Frame]:
    cols = max(18, (width - 4) // 3)
    rows = 9
    base = [terrain(cols, rows) for _ in RATES]
    paths = []
    for lr, _, _ in RATES:
        a, b = START
        path = [(a, b)]
        for _ in range(steps):
            ga, gb = slope(a, b)
            a, b = a - lr * ga, b - lr * gb
            path.append((a, b))
        paths.append(path)
    for i in range(steps + 1):
        panels = [
            panel(
                base[k],
                paths[k][: i + 1],
                cols,
                rows,
                lr,
                verdict if i == steps else "",
                i,
            )
            for k, (lr, verdict, _) in enumerate(RATES)
        ]
        grid = Table.grid(padding=(0, 2))
        for _ in RATES:
            grid.add_column(no_wrap=True)
        grid.add_row(*panels)
        yield (grid, 1.2) if i == 0 else (grid, 0.12) if i < steps else hold(grid, 0.3)


def terrain(cols: int, rows: int) -> list[list[str]]:
    """The color of every pixel: `rows` lines of two pixels each."""
    top = loss(A_RANGE[0], B_RANGE[1])
    pixels = []
    for j in range(rows * 2):
        b = B_RANGE[1] - (j + 0.5) / (rows * 2) * (B_RANGE[1] - B_RANGE[0])
        line = []
        for i in range(cols):
            a = A_RANGE[0] + (i + 0.5) / cols * (A_RANGE[1] - A_RANGE[0])
            level = np.floor(np.sqrt(loss(a, b) / top) * 10) / 10  # contour bands
            line.append(LAND.color(float(level)))
        pixels.append(line)
    return pixels


def cell(a: float, b: float, cols: int, rows: int) -> tuple[int, int] | None:
    i = int((a - A_RANGE[0]) / (A_RANGE[1] - A_RANGE[0]) * cols)
    j = int((B_RANGE[1] - b) / (B_RANGE[1] - B_RANGE[0]) * rows)
    return (i, j) if 0 <= i < cols and 0 <= j < rows else None


def panel(pixels, path, cols: int, rows: int, lr: float, verdict: str, step: int) -> Group:
    marks: dict[tuple[int, int], tuple[str, str]] = {}
    if goal := cell(*CENTER, cols, rows):
        marks[goal] = ("+", f"bold {viz.GREEN}")
    for a, b in path[:-1]:
        if c := cell(a, b, cols, rows):
            marks[c] = ("•", viz.AMBER)
    a, b = path[-1]
    here = cell(a, b, cols, rows)
    if here:
        marks[here] = ("●", f"bold {viz.AMBER}")
    lines = []
    for r in range(rows):
        line = Text(no_wrap=True)
        for c in range(cols):
            upper, lower = pixels[2 * r][c], pixels[2 * r + 1][c]
            if (c, r) in marks:
                ch, st = marks[(c, r)]
                line.append(
                    ch,
                    viz.style(st.split()[-1], viz.mix(upper, lower, 0.5), bold="bold" in st),
                )
            else:
                line.append("▀", viz.style(upper, lower))
        lines.append(line)
    head = Text.assemble(("learning rate ", viz.FAINT), (f"{lr}", f"bold {viz.AMBER}"))
    where = "flew off" if here is None else f"loss {loss(a, b):.2f}"
    foot = Text.assemble((f"step {step:2d}  ", viz.FAINT), (where, "bold"))
    tag = Text(verdict, style=f"italic {viz.GREEN if verdict == 'about right' else viz.RED}")
    return Group(head, *lines, foot, tag)


# ── One real step ─────────────────────────────────────────────────────────────


@dataclass
class RealStep:
    before: float
    after: float
    old: np.ndarray
    new: np.ndarray

    def picture(self, lab: Lab, width: int) -> Table:
        scale = viz.scale_of(self.old)
        maps = [
            ("before", viz.signed_blocks(self.old.T, scale)),
            ("the step", viz.nudge_blocks((self.new - self.old).T)),
            ("after", viz.signed_blocks(self.new.T, scale)),
        ]
        letters = Text(lab.vocab.chars, style=viz.FAINT, no_wrap=True)
        side_by_side = width >= 3 * len(lab.vocab) + 8
        grid = Table.grid(padding=(0, 3))
        if side_by_side:
            for _ in maps:
                grid.add_column(no_wrap=True)
            grid.add_row(*(Text(name, style="bold") for name, _ in maps))
            grid.add_row(*(Group(*lines, letters) for _, lines in maps))
        else:
            grid.add_column(no_wrap=True)
            for name, lines in maps:
                grid.add_row(Group(Text(name, style="bold"), *lines, letters, Text("")))
        return grid


def real_step(lab: Lab, lr: float = 1.0) -> RealStep:
    words = [lab.data.train[int(i)] for i in lab.rng.choice(len(lab.data.train), 32, replace=False)]
    inputs, targets = lab.data.batch(words)
    model = lab.model.copy()  # practice on a copy: the real training starts from scratch
    tr = model.forward(inputs)
    before, dlogits = cross_entropy(tr.logits, targets)
    grads, _ = model.backward(tr, dlogits)
    old = model.params["embed.token"].copy()
    for name, w in model.params.items():
        w -= lr * grads[name]
    after, _ = cross_entropy(model.forward(inputs).logits, targets)
    return RealStep(before, after, old, model.params["embed.token"].copy())
