"""Chapter: quantization. Storing each weight in fewer bits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..generate import write
from ..quantize import BLOCK, LEVELS, quantize, shrink
from ..train import evaluate
from ..views import display
from .lab import Lab
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    model = lab.model
    size = model.size * 8
    stage.say(
        f"Your model's {model.size:,} numbers are stored the way NumPy stores numbers by "
        f"default, in 64 bits each: {size / 1e6:.1f} MB. Real models are trained with 16-bit "
        "numbers, and often run with far fewer: 8 bits per weight, 4, sometimes less. Llama 3 "
        "70B takes 140 GB in 16 bits and 35 GB in 4, which is the difference between a rack "
        "of GPUs and a big laptop. That's [b]quantization[/b]."
    )
    w = model.params["layers.1.mlp.up"]
    block = w[:BLOCK, 0]
    stage.say(
        f"The trick: cut each grid of weights into blocks of {BLOCK} numbers. In each block, "
        "find the biggest number, and use it to set a scale: in 4 bits, every number becomes a "
        "whole number of steps from −7 to 7, where a step is the biggest number ÷ 7. Store the "
        "whole numbers (4 bits each) and the scale (16 bits, once per block), and multiply "
        "them back when you use them. Here's one real block from your model:"
    )
    stage.show(block_view(block, stage.width))
    stage.wait("shrink the whole model")

    base = evaluate(model, *lab.trainer.held)
    results = [shrink(model, bits) for bits in LEVELS]
    losses = [evaluate(r.model, *lab.trainer.held) for r in results]
    stage.say(
        f"Now every grid in the model, rounded the same way. Here's how big it gets, and its "
        f"loss on the held-back {lab.corpus.plural}:"
    )
    stage.show(results_view(model.size * 8, base, results, losses))
    fine = [r for r, loss in zip(results, losses, strict=True) if loss - base < 0.1]
    lowest = min((r.bits for r in fine), default=16)
    stage.say(
        f"Down to {lowest} bits, it barely notices. A trained network is surprisingly "
        "tolerant of small nudges to its weights: each one is off by a little, in a random "
        "direction, and the errors mostly cancel out. Below that, the steps get too coarse and "
        "it falls apart. Here's what it writes at each size, from the same dice:"
    )
    stage.show(samples(lab, results, stage.width))
    stage.say(
        "When you download a model to run on your own computer, with llama.cpp or Ollama or "
        "LM Studio, file names like Q8_0 and Q4_K_M are exactly this: Q8 and Q4 are the bits "
        "per weight, and the letters say how the blocks and scales are arranged (the K-quants "
        "give blocks their own smaller scales, inside bigger blocks with scales of their own). "
        "The same idea shrinks the KV cache, and some models are even trained knowing they'll "
        "be run in 4 bits, so they learn weights that round well."
    )


def block_view(block: np.ndarray, width: int) -> Group:
    """One block rounded to 4 bits: all of it as colors, then a few of its numbers worked out."""
    top = 7
    biggest = int(np.argmax(np.abs(block)))
    scale = float(abs(block[biggest])) / top
    ints = np.clip(np.round(block / scale), -top, top).astype(int)
    rebuilt = quantize(block[None, :], 4, len(block))[0] + 0.0  # + 0.0 turns -0.0 into 0.0
    shade = viz.scale_of(block)

    strips = Table.grid(padding=(0, 2))
    strips.add_column(style=viz.FAINT, no_wrap=True, justify="right")
    strips.add_column(no_wrap=True)
    cell = 2 if 2 * len(block) + 9 <= width else 1
    strips.add_row("weights", viz.signed_cells(block, shade, cell))
    strips.add_row("rounded", viz.signed_cells(rebuilt, shade, cell))
    strips.add_row("", Text(" " * biggest * cell + "↑", style=f"bold {viz.ACCENT}"))

    # As many as fit, and always the biggest, after a gap if it's further along.
    count = max(3, min(8, (width - 6) // 8))
    shown: list[int | None] = list(range(count))
    if biggest >= count:
        shown = [*range(count - 2), None, biggest]
    work = Table.grid(padding=(0, 0, 0, 2))
    work.add_column(style=viz.FAINT, no_wrap=True)
    for _ in shown:
        work.add_column(justify="right", no_wrap=True)

    def row(label: str, cell: Callable[[int], Text]) -> None:
        work.add_row(label, *(Text("…", style=viz.FAINT) if i is None else cell(i) for i in shown))

    def signed(n: int) -> Text:
        tint = viz.AMBER if n > 0 else viz.BLUE if n < 0 else viz.FAINT  # as in the strips
        return Text(f"{n:+d}" if n else "0", style=f"bold {tint}")

    row("", lambda i: Text(f"#{i + 1}", style=f"bold {viz.ACCENT}" if i == biggest else viz.FAINT))
    row("weight", lambda i: Text(f"{block[i]:+.3f}"))
    row("÷ step", lambda i: Text(f"{block[i] / scale:+.2f}", style=viz.FAINT))
    row("stored", lambda i: signed(int(ints[i])))
    row("used", lambda i: Text(f"{rebuilt[i]:+.3f}"))

    step = Text.assemble(
        ("The biggest is ", viz.FAINT), (f"#{biggest + 1}", f"bold {viz.ACCENT}"),
        (", so a step is ", viz.FAINT), (f"{abs(block[biggest]):.3f} ÷ 7 = {scale:.4f}", "bold"),
        (". Worked out:", viz.FAINT),
    )  # fmt: skip
    err = float(np.abs(rebuilt - block).max())
    bits = len(block) * 4 + 16
    return Group(
        strips,
        Text(""),
        step,
        Text(""),
        work,
        Text(""),
        Text(f"Off by at most {err:.4f} anywhere in the block: half a step.", style=viz.FAINT),
        Text(
            f"Kept as {len(block)} × 4 bits, plus 16 for the step: {bits} bits, "
            f"{bits / len(block):.1f} bits a weight.",
            style=viz.FAINT,
        ),
    )


def results_view(original: int, base: float, results, losses: list[float]) -> Table:
    grid = Table.grid(padding=(0, 2))
    for justify in ("left", "right", "left", "right", "left"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(*(Text(h, style=viz.FAINT) for h in ("", "size", "", "held-back loss", "")))
    grid.add_row(
        Text("64-bit (yours)", style="bold"),
        f"{original / 1024:,.0f} KB",
        viz.bar(1.0, 20, viz.PURPLE),
        f"{base:.3f}",
        "",
    )
    for r, loss in zip(results, losses, strict=True):
        worse = loss - base
        tint = viz.GREEN if worse < 0.1 else viz.AMBER if worse < 0.5 else viz.RED
        grid.add_row(
            Text(r.label, style="bold"),
            f"{r.size / 1024:,.0f} KB",
            viz.bar(r.size / original, 20, viz.PURPLE),
            Text(f"{loss:.3f}", style=f"bold {tint}"),
            Text(f"{worse:+.3f}", style=tint),
        )
    return grid


def samples(lab: Lab, results, width: int) -> Table:
    shown = [r for r in results if r.bits in (8, 4, 2)]
    most = min(70, width - max(len(r.label) for r in shown) - 2)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    for r in shown:
        text = write(
            r.model, lab.tokenizer, np.random.default_rng([lab.seed, 15]), "", 0.7, limit=24
        )
        grid.add_row(r.label, Text(display(text, most), style="italic"))
    return grid
