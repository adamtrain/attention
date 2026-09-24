"""Chapter: the whole model, end to end."""

from __future__ import annotations

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
        "Now you've seen the main parts. Here's the whole model, top to bottom, with the shape "
        f"of the grid of numbers flowing through it for {t} tokens:"
    )
    stage.show(diagram(lab, t))
    stage.say(
        "The [b]MLP[/b] (multi-layer perceptron) is a small two-layer network that each "
        f"position runs on its own, widening its {c.width} numbers to {c.hidden} and back. "
        "Attention moves information between positions; the MLP works on it. The [b]⊕[/b] "
        "adds each block's output back onto its input, so information can also flow straight "
        "past. Before each block, a [b]norm[/b] rescales every vector to a steady size."
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

    focus = max(1, round(len(lab.corpus.example) * 0.65))
    seen = lab.vocab.decode(inputs[0, : focus + 1])
    right = int(targets[0, focus])
    probs = lab.model.forward(inputs).probs[0, focus]
    stage.say(f"Let's run it. After `{seen}`, the untrained model thinks the next letter is:")
    stage.show(prob_bars(probs, lab.vocab, top=8, width=30, right=right))
    stage.say(
        f"Every token gets about 1 in {c.vocab}, around {1 / c.vocab:.1%}. It hasn't learned a "
        "thing yet. Its weights are random, so its guesses are too."
    )


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
