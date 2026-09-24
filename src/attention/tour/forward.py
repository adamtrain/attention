"""Chapter: the whole model, end to end."""

from __future__ import annotations

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import Transformer
from ..views import prob_bars
from .code import excerpt
from .lab import Lab
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    c = lab.model.config
    inputs, targets = lab.example()
    t = inputs.shape[1]
    stage.say(
        "Here's the whole model, top to bottom. The purple numbers on the right are the shape "
        f"of the grid flowing through it for {t} tokens: rows, times numbers per row."
    )
    stage.show(diagram(lab, t))
    focus = max(1, round(len(lab.corpus.example) * 0.65))
    seen = lab.vocab.decode(inputs[0, : focus + 1])
    stage.say(
        "You've met embeddings and attention. The new box is the [b]MLP[/b] (short for "
        "multi-layer perceptron, an old name for the simplest kind of neural network). "
        "Attention moves information [i]between[/i] positions. The MLP then works on what each "
        "position has gathered, one position at a time, without looking at the others."
    )
    stage.say(
        f"It's two grids of weights with a switch in between. The first grid turns a "
        f"position's {c.width} numbers into {c.hidden}: {c.hidden} different weighted sums, "
        "each one a little detector for some pattern in the vector. Then the switch, called "
        "[b]ReLU[/b], sets every negative number to 0 and lets positive ones through, so each "
        "detector either fires or stays quiet. The second grid mixes whichever detectors "
        f"fired back down to {c.width} numbers. Here it is at the `{seen[-1]}` in `{seen}`:"
    )
    tr = lab.model.forward(inputs)
    stage.show(mlp(tr.m_in[0, focus], tr.pre[0, focus], tr.x2[0, focus] - tr.x1[0, focus]))
    stage.say(
        "The switch is what makes it more than arithmetic. Two grids in a row with nothing "
        "between them can only ever do what one grid could; the on-or-off step lets the "
        "network react to combinations, like “a vowel, right after a q”. Research suggests "
        "much of what a big model knows, facts included, is stored in its MLPs."
    )
    stage.wait()

    stage.say(
        "The [b]⊕[/b] is a [b]residual connection[/b]: each block's output is [i]added[/i] "
        "onto its input instead of replacing it. Picture the grid running down the left side "
        "as a shared notebook. Each block reads it, and adds its notes back in. Whatever a "
        "block doesn't change flows straight past it. That also gives backprop a clear "
        "path back down, which is a big part of why stacks of dozens of blocks can be trained "
        "at all."
    )
    stage.say(
        "Before each block, a [b]norm[/b] rescales each position's vector to a steady size "
        "without changing its direction, so no block gets thrown by numbers that happen to be "
        "huge or tiny. And at the bottom, [b]unembed[/b] is one last grid: it turns each "
        f"position's {c.width} numbers into a score for each of the {c.vocab} tokens, which "
        "softmax turns into probabilities for the next one."
    )
    stage.note(
        "Big models stack this attention-and-MLP block dozens of times; GPT-3 has 96 of them. "
        "Yours has one."
    )
    stage.wait()

    stage.say(f"Where its {lab.model.size:,} parameters live:")
    stage.show(breakdown(lab, stage.width))
    stage.wait()

    stage.say(
        "And here's the entire forward pass, from your model's source code. That's a transformer:"
    )
    stage.show(excerpt(Transformer.forward, "# 1. Look up", "return Trace"))
    stage.wait()

    right = int(targets[0, focus])
    probs = tr.probs[0, focus]
    stage.say(f"Let's run it. After `{seen}`, the untrained model thinks the next letter is:")
    stage.show(prob_bars(probs, lab.vocab, top=8, width=30, right=right))
    stage.say(
        f"Every token gets about 1 in {c.vocab}, around {1 / c.vocab:.1%}. It hasn't learned a "
        "thing yet. Its weights are random, so its guesses are too."
    )


def mlp(x: np.ndarray, pre: np.ndarray, out: np.ndarray) -> Group:
    """One position through the MLP: widen, switch, narrow."""
    fired = np.maximum(pre, 0.0)
    scale = viz.scale_of(pre)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row("in", viz.signed_cells(x, viz.scale_of(x)))
    grid.add_row("up", viz.signed_cells(pre, scale, 1))
    grid.add_row("ReLU", viz.signed_cells(fired, scale, 1))
    grid.add_row("down", viz.signed_cells(out, viz.scale_of(out)))
    n = int((pre > 0).sum())
    caption = Text.assemble(
        (f"{len(x)} numbers in, {len(pre)} detectors, ", viz.FAINT), (f"{n} fired", "bold"),
        (f", {len(out)} numbers out, added onto the vector", viz.FAINT),
    )  # fmt: skip
    return Group(grid, Text(""), caption)


def diagram(lab: Lab, t: int) -> Table:
    c = lab.model.config
    box, wire, faint = viz.ACCENT, viz.FAINT, viz.FAINT
    rows: list[tuple[Text, str, str]] = []

    def line(*parts: tuple[str, str]) -> Text:
        return Text.assemble(*parts, no_wrap=True)

    def block(name: str, indent: str, what: str, shape: str) -> None:
        rows.append((line((indent, wire), ("╭" + "─" * 16 + "╮", box)), "", ""))
        rows.append(
            (
                line((indent, wire), ("│", box), (f"{name:^16}", "bold"), ("│", box)),
                what,
                shape,
            )
        )
        rows.append((line((indent, wire), ("╰" + "─" * 7 + "┬" + "─" * 8 + "╯", box)), "", ""))

    rows.append((line(("      tokens", "bold")), "", f"{t}"))
    rows.append((line(("        │", wire)), "", ""))
    block("embeddings", "", "token + position", f"{t} × {c.width}")
    for name, what in (
        ("attention", "look back at earlier tokens"),
        ("MLP", "think it over"),
    ):
        rows.append((line(("        ├" + "─" * 10 + "╮", wire)), "", ""))
        rows.append(
            (
                line(("        │  ", wire), ("╭" + "─" * 7 + "┴" + "─" * 8 + "╮", box)),
                "",
                "",
            )
        )
        rows.append(
            (
                line(
                    ("        │  ", wire),
                    ("│", box),
                    (f"{name:^16}", "bold"),
                    ("│", box),
                ),
                what,
                f"{t} × {c.width}",
            )
        )
        rows.append(
            (
                line(("        │  ", wire), ("╰" + "─" * 7 + "┬" + "─" * 8 + "╯", box)),
                "",
                "",
            )
        )
        rows.append(
            (
                line(("        ⊕", f"bold {viz.AMBER}"), ("◂" + "─" * 9 + "╯", wire)),
                "add",
                "",
            )
        )
    rows.append((line(("        │", wire)), "", ""))
    block("unembed", "", "a score for every token", f"{t} × {c.vocab}")
    rows.append((line(("        │", wire)), "", ""))
    rows.append(
        (
            line(("     softmax", "bold")),
            "probabilities for the next token",
            f"{t} × {c.vocab}",
        )
    )

    grid = Table.grid(padding=(0, 3))
    grid.add_column(no_wrap=True)
    grid.add_column(style=faint, no_wrap=True)
    grid.add_column(style=viz.PURPLE, no_wrap=True, justify="right")
    for graphic, what, shape in rows:
        grid.add_row(graphic, what, shape)
    return grid


def breakdown(lab: Lab, width: int) -> Table:
    p, c = lab.model.params, lab.model.config
    groups = [
        ("embeddings", f"{c.vocab}×{c.width} tokens, {c.context}×{c.width} positions", ["embed.token", "embed.position"]),
        ("attention", f"query, key, value, output: 4 × {c.width}×{c.width}", ["attn.query", "attn.key", "attn.value", "attn.out"]),
        ("MLP", f"up {c.width}×{c.hidden}, down {c.hidden}×{c.width}", ["mlp.up", "mlp.down"]),
        ("unembed", f"{c.width}×{c.vocab}", ["head.unembed"]),
        ("norms", f"3 × {c.width}", ["attn.norm", "mlp.norm", "head.norm"]),
    ]  # fmt: skip
    counts = [sum(p[n].size for n in names) for _, _, names in groups]
    biggest = max(counts)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    for (name, detail, _), n in zip(groups, counts, strict=True):
        grid.add_row(name, detail, f"{n:,}", viz.bar(n / biggest, 20, viz.PURPLE, track=False))
    grid.add_row("", Text("total", style="bold"), Text(f"{sum(counts):,}", style="bold"), "")
    return grid
