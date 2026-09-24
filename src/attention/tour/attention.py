"""Chapter: attention, the part that looks back."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import Transformer
from ..views import attention_grid, labels, looks_at
from .code import excerpt
from .lab import Lab
from .stage import Frame, Stage, hold


def run(stage: Stage, lab: Lab) -> None:
    word = lab.corpus.example
    inputs, _ = lab.example()
    tr = lab.model.forward(inputs)
    chars = labels(lab.vocab, inputs[0])
    focus = focus_row(word)
    seen = "".join(chars[: focus + 1])

    stage.say(
        f"To guess what comes after `{seen}`, the model has to look back at the letters so "
        "far and work out which ones matter. That's what [b]attention[/b] does."
    )
    stage.say(
        "Every position turns its vector into three new ones, each by multiplying it by a "
        "grid of weights:",
        gap=False,
    )
    roles = Table.grid(padding=(0, 2))
    roles.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    roles.add_column()
    roles.add_row("query", "what am I looking for?")
    roles.add_row("key", "what do I have?")
    roles.add_row("value", "what I'll pass along if you pick me")
    stage.console.print()
    stage.show(roles)
    stage.say(
        "Then each position compares its query with the keys of every position, by "
        "multiplying them number by number and adding up (a [b]dot product[/b]). A big result "
        "means a good match: look there."
    )
    stage.wait("see it happen")

    raw = (tr.q[0, 0] @ tr.k[0, 0].T) / np.sqrt(lab.model.config.head_width)
    stage.play(scoring(raw, tr.weights[0, 0], chars), fps=14)
    stage.say(
        "Each row is one position, and each column is where it might look. The dots are the "
        "[b]causal mask[/b]: a position can't look ahead at letters that haven't been written "
        "yet. [b]Softmax[/b] (more on it next) turns what's left of each row into weights that "
        "add up to 100%."
    )
    stage.wait()

    stage.say(
        f"Take the `{chars[focus]}` row: the position that has read `{seen}` so far. It looks at:"
    )
    stage.show(looks_at(tr.weights[0, 0, focus, : focus + 1], chars[: focus + 1], width=24))
    stage.say(
        "It blends the [i]values[/i] of those positions in those proportions, and adds the "
        "blend to its own vector. That's how information from earlier letters reaches the "
        "prediction. For now the weights are random, so where it looks is arbitrary. Training "
        "will change that."
    )
    stage.wait()

    stage.say(
        f"Your model has {lab.model.config.heads} [b]heads[/b] doing this side by side, each "
        "with its own weights, so each can learn to look for something different:"
    )
    stage.show(heads(lab.model, tr.weights[0], chars))
    stage.say("And this is the code that does it, from your model:")
    stage.show(excerpt(Transformer.forward, "# 2. Attention", "# 3. MLP"))


def focus_row(word: str) -> int:
    return max(1, min(len(word) - 1, round(len(word) * 0.65)))


def scoring(raw: np.ndarray, weights: np.ndarray, chars: list[str]) -> Iterator[Frame]:
    t = len(chars)
    captions = [
        ("1", "scores", "each query · each key"),
        ("2", "mask", "no looking ahead"),
        ("3", "softmax", "each row becomes weights that add up to 100%"),
    ]

    def frame(step: int, masked: int, soft: int) -> Group:
        n, name, what = captions[step]
        caption = Text.assemble(
            (f"{n} ", f"bold {viz.ACCENT}"), (name, "bold"), (f"  {what}", viz.FAINT)
        )
        return Group(caption, Text(""), *score_rows(raw, weights, chars, masked, soft))

    yield hold(frame(0, 0, 0), 2.2)
    for r in range(t):
        yield frame(1, r + 1, 0), 0.07
    yield hold(frame(1, t, 0), 1.2)
    for r in range(t):
        yield frame(2, t, r + 1), 0.12
    yield hold(frame(2, t, t), 0.5)


def score_rows(
    raw: np.ndarray, weights: np.ndarray, chars: list[str], masked: int, soft: int
) -> list[Text]:
    t = len(chars)
    cell = 5
    scale = viz.scale_of(raw)
    lines = []
    for r in range(t):
        line = Text(no_wrap=True)
        line.append(f" {chars[r]} ", style=f"bold {viz.ACCENT}")
        for c in range(t):
            if c > r and r < masked:
                line.append(f"{'·':^{cell}}", style=viz.FAINT)
            elif r < soft:
                bg = viz.HEAT.color(float(weights[r, c]) ** 0.6)
                body = f"{weights[r, c]:.2f}".replace("0.", ".", 1)
                line.append(f"{body:^{cell}}", viz.style(viz.ink_for(bg), bg))
            else:
                bg = viz.SIGNED.diverging(float(raw[r, c]), scale)
                body = f"{raw[r, c]:.1f}".replace("0.", ".", 1)
                line.append(f"{body:^{cell}}", viz.style(viz.ink_for(bg), bg))
        lines.append(line)
    header = Text("   ", no_wrap=True)
    for ch in chars:
        header.append(f"{ch:^{cell}}", style=viz.FAINT)
    lines.append(header)
    return lines


def heads(model: Transformer, weights: np.ndarray, chars: list[str]) -> Table:
    grid = Table.grid(padding=(0, 4))
    for _ in range(len(weights)):
        grid.add_column(no_wrap=True)
    grid.add_row(*(Text(f"head {h + 1}", style="bold") for h in range(len(weights))))
    grid.add_row(*(Group(*attention_grid(w, chars, numbers=False)) for w in weights))
    return grid
