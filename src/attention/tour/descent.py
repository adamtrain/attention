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

    trainer = lab.trainer
    stage.say(
        f"Now for real. Take {trainer.batch_size} stretches of your {lab.corpus.plural}, "
        f"{lab.model.config.context} tokens each, run them forward, backpropagate, and move "
        "every weight in your model one step downhill. Here's a corner of one of its grids, "
        "layer 1's queries, before the step, the step itself, and after:"
    )
    step = real_step(lab)
    stage.show(step.picture(stage.width))
    stage.say(
        f"The loss on those stretches went from [b]{step.before:.2f}[/b] to "
        f"[b][green]{step.after:.2f}[/green][/b], with a learning rate of {step.lr:g}. One small "
        "step. Pretraining is this, over and over."
    )
    stage.wait("see the real recipe")

    stage.say(
        "Real training adds three refinements, and your model uses all of them. The first is "
        "[b]Adam[/b]: instead of stepping straight down the gradient, each weight keeps "
        "momentum (it keeps rolling the way it has been going, so one noisy batch can't knock "
        "it off course) and its own step size (weights whose gradients are always tiny still "
        "move). With weight decay, which you'll meet in the chapter on memorizing, it's "
        "called [b]AdamW[/b], and nearly every language model trains with it."
    )
    stage.say(
        "The second is a [b]schedule[/b] for the learning rate. It starts near zero and ramps "
        f"up over the first {trainer.warmup} steps (the warmup: big steps from random weights "
        "can do damage), then eases down along a cosine curve, so the last steps are small "
        "and careful. Here's your model's:"
    )
    stage.show(schedule(lab, min(56, stage.width - 12)))
    stage.say(
        "The third is [b]gradient clipping[/b]. Now and then one unlucky batch produces a "
        "huge gradient, and one huge step can undo a lot of work. So before each step, the "
        "whole gradient, every weight's, is measured as one long vector, and if it's longer "
        f"than {trainer.most:g} it's shrunk to that length. It still points the same way; it "
        "just can't leap. You'll see its length on the training dashboard."
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
    lr: float
    old: np.ndarray
    new: np.ndarray

    def picture(self, width: int) -> Group:
        scale = viz.scale_of(self.old)
        maps = [
            ("before", viz.signed_blocks(self.old, scale, width=2)),
            ("the step", viz.nudge_blocks(self.new - self.old, width=2)),
            ("after", viz.signed_blocks(self.new, scale, width=2)),
        ]
        gap = max(1, min(3, (width - len(maps) * 2 * self.old.shape[1]) // (len(maps) - 1)))
        grid = Table.grid(padding=(0, gap))
        for _ in maps:
            grid.add_column(no_wrap=True)
        grid.add_row(*(Text(name, style="bold") for name, _ in maps))
        grid.add_row(*(Group(*lines) for _, lines in maps))
        legend = Text.assemble(
            viz.legend(viz.SIGNED, "−", "+"), ("     ", ""), viz.legend(viz.NUDGE, "down", "up")
        )
        corner = Text(f"a {len(self.old)} × {self.old.shape[1]} corner of it", style=viz.FAINT)
        return Group(grid, corner, Text(""), legend)


def real_step(lab: Lab, corner: tuple[int, int] = (16, 12)) -> RealStep:
    """One plain step of gradient descent, on a copy: the real training starts from scratch."""
    inputs, targets = lab.data.windows(lab.rng, lab.trainer.batch_size, lab.model.config.context)
    tr = lab.model.forward(inputs)
    before, dlogits = cross_entropy(tr.logits, targets)
    grads, _ = lab.model.backward(tr, dlogits)
    name = "layers.0.attn.query"
    rows, cols = corner
    for lr in (1.0, 0.5, 0.2, 0.1):  # the biggest step that still goes downhill
        model = lab.model.copy()
        for key, w in model.params.items():
            w -= lr * grads[key]
        after, _ = cross_entropy(model.forward(inputs).logits, targets)
        if after < before:
            break
    old = lab.model.params[name][:rows, :cols].copy()
    return RealStep(before, after, lr, old, model.params[name][:rows, :cols].copy())


def schedule(lab: Lab, width: int) -> Group:
    trainer = lab.trainer
    saved = trainer.step_number
    rates = []
    for n in range(trainer.steps):
        trainer.step_number = n
        rates.append(trainer.learning_rate())
    trainer.step_number = saved
    plot = viz.Plot(width, 6, x_max=trainer.steps, lo=0.0, hi=trainer.lr * 1.05)
    plot.line(list(enumerate(rates)), viz.AMBER)
    axis = Text.assemble(
        ("       step 0", viz.FAINT), (" " * (width - 14)), (f"{trainer.steps:,}", viz.FAINT)
    )
    return Group(Text("  learning rate", style=viz.FAINT), *plot.render(fmt="{:.3f}"), axis)
