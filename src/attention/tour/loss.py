"""Chapter: measuring how wrong the model is."""

from __future__ import annotations

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import Transformer, cross_entropy
from ..views import token_chip
from .lab import Lab
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    v = lab.model.config.vocab
    stage.say(
        "To learn, the model needs a score for how wrong it was: one number it can try to make "
        "smaller. The usual one looks only at the probability the model gave the right "
        "answer, and takes minus its logarithm (−log). That's the [b]loss[/b]. Here's how "
        "the loss depends on that probability:"
    )
    stage.show(curve(v))
    stage.say(
        "Read it from right to left. If the model gave the right answer 90%, the loss is only "
        f"{-np.log(0.9):.2f}: it was confident and right, so there's hardly anything to fix. "
        f"At 50% the loss is {np.log(2):.2f}. If it had no idea and spread its bets evenly, "
        f"giving each of the {v} tokens the same 1 in {v}, the loss is {np.log(v):.2f}. "
        f"Below that the curve shoots up: 1% costs {-np.log(0.01):.2f}, 0.1% costs "
        f"{-np.log(0.001):.2f}, and as the probability heads toward zero, the loss heads "
        "toward infinity."
    )
    stage.say(
        "That steep climb is the point. Being confident about the [i]wrong[/i] answer "
        "leaves almost no probability for the right one, and that gets punished far harder "
        "than admitting it doesn't know. So the way to score well on average is to be "
        "confident only when there's good reason to be, and to spread the bets when there "
        "isn't."
    )
    stage.wait()

    word = lab.corpus.example
    stage.say(f"Here's the untrained model on every quiz in {word.capitalize()}:")
    stage.show(quiz_losses(lab, lab.model))
    inputs, targets = lab.example()
    loss, _ = cross_entropy(lab.model.forward(inputs).logits, targets)
    stage.say(
        f"Untrained, it's close to the {np.log(v):.2f} of an even spread on almost every quiz: "
        f"it's shrugging at everything. The average, {loss:.2f}, is the model's "
        "[b]cross-entropy loss[/b] on this word. Training means exactly one thing: make that "
        "number smaller, averaged over every quiz in every word."
    )


def curve(vocab: int) -> Group:
    width, height = 44, 9
    plot = viz.Plot(width, height, x_max=1.0, lo=0.0, hi=4.6)
    ps = np.linspace(0.01, 1.0, 200)
    plot.line(list(zip(ps, -np.log(ps), strict=True)), viz.ACCENT)
    marks = [(1 / vocab, viz.RED), (0.5, viz.AMBER), (0.9, viz.GREEN)]
    for p, color in marks:
        plot.mark(p, -np.log(p), "●", color)
    lines = plot.render(fmt="{:.0f}")
    axis = Text("  └" + "─" * width, style=viz.FAINT)
    ticks = Text.assemble(
        ("   0%", viz.FAINT), (" " * (width // 2 - 5)), ("50%", viz.FAINT), (" " * (width // 2 - 5)),
        ("100%", viz.FAINT),
    )  # fmt: skip
    caption = Text("   probability given to the right answer", style=viz.FAINT)
    key = Text(no_wrap=True)
    for i, (p, color) in enumerate(marks):
        if i:
            key.append("    ")
        key.append("● ", style=color)
        key.append(f"{viz.percent(p)} → ", style=viz.FAINT)
        key.append(f"{-np.log(p):.2f}", style="bold")
    return Group(Text("  loss", style=viz.FAINT), *lines, axis, ticks, caption, Text(""), key)


def quiz_losses(lab: Lab, model: Transformer, bar_width: int = 24, compact: bool = False) -> Table:
    """The loss on every quiz in the example word. Compact leaves out the quizzes themselves."""
    inputs, targets = lab.example()
    probs = model.forward(inputs).probs[0]
    t = inputs.shape[1]
    grid = Table.grid(padding=(0, 1))
    for _ in range(2 if compact else 5):
        grid.add_column(no_wrap=True)
    heads = [Text("so far", style=viz.FAINT), Text(""), Text("answer", style=viz.FAINT)]
    grid.add_row(
        *([] if compact else heads),
        Text("  chance", style=viz.FAINT),
        Text("loss", style=viz.FAINT),
    )
    top = np.log(lab.model.config.vocab) * 1.4
    for i in range(t):
        right = int(targets[0, i])
        p = float(probs[i, right])
        loss = -np.log(p)
        color = viz.GREEN if loss < 1 else viz.AMBER if loss < 2.5 else viz.RED
        quiz = [
            Text(lab.vocab.decode(inputs[0, : i + 1]), style="bold"),
            Text("→", style=viz.FAINT),
            token_chip(lab.vocab, right),
        ]
        grid.add_row(
            *([] if compact else quiz),
            Text(f"{viz.percent(p):>8}"),
            Text.assemble(
                (f"{loss:5.2f} ", "bold"),
                viz.bar(loss / top, bar_width, color, track=False),
            ),
        )
    return grid
