"""Chapter: what the model learned."""

from __future__ import annotations

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..views import attention_grid, labels
from .lab import Lab
from .loss import quiz_losses
from .stage import Stage

VOWELS = set("aeiou")


def run(stage: Stage, lab: Lab) -> None:
    word = lab.corpus.example.capitalize()
    stage.say(f"Remember the quizzes in {word}? Before training, and now:")
    stage.show(before_after(lab, stage.width))
    stage.say(
        "Where it used to shrug at every letter, it now knows what's coming, most of the time. "
        "The biggest losses left are the genuinely hard calls, usually the first letter or two, "
        "where anything goes."
    )
    stage.wait()

    stage.say(
        "What did the embeddings learn? Squash each letter's 16 numbers down to the 2 that vary "
        "most, and plot them. Letters the model uses in similar ways drift toward each other. "
        "Look for the [amber]vowels[/amber]: they usually end up on the same side."
    )
    stage.show(scatter(lab, min(stage.width, 64), 15))
    stage.note("Sixteen numbers flattened to two loses a lot, so treat it as a rough map.")
    stage.wait()

    inputs, _ = lab.example()
    chars = labels(lab.vocab, inputs[0])
    before, after = (
        lab.initial.forward(inputs).weights[0],
        lab.model.forward(inputs).weights[0],
    )
    stage.say(
        "And attention? Here's where each head looks as it reads "
        f"{word}, before and after training:"
    )
    stage.show(heads_before_after(before, after, chars, stage.width))
    stage.say(habits(lab))


def before_after(lab: Lab, width: int) -> Table:
    grid = Table.grid(padding=(0, 4))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(Text("untrained", style=f"bold {viz.FAINT}"), Text("trained", style="bold"))
    narrow = width < 110
    grid.add_row(
        quiz_losses(lab, lab.initial, bar_width=8 if narrow else 16),
        quiz_losses(lab, lab.model, bar_width=8 if narrow else 16, compact=True),
    )
    return grid


def scatter(lab: Lab, width: int, height: int) -> Group:
    table = lab.model.params["embed.token"]
    centered = table - table.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T
    lo, hi = points.min(axis=0), points.max(axis=0)
    canvas = viz.Canvas(width, height)
    for x in range(width):
        canvas.put(x, height - 1, "─", viz.FAINT)
    for y in range(height - 1):
        canvas.put(0, y, "│", viz.FAINT)
    canvas.put(0, height - 1, "└", viz.FAINT)
    for i, ch in enumerate(lab.vocab.chars):
        x = 2 + round((points[i, 0] - lo[0]) / (hi[0] - lo[0] + 1e-9) * (width - 4))
        y = round((1 - (points[i, 1] - lo[1]) / (hi[1] - lo[1] + 1e-9)) * (height - 3))
        color = viz.AMBER if ch in VOWELS else viz.GREEN if ch == "." else viz.ACCENT
        while x < width - 2 and (canvas.chars[y][x] != " " or canvas.chars[y][x - 1] != " "):
            x += 1  # don't let letters cover or touch each other
        canvas.put(x, y, ch, f"bold {color}")
    key = Text.assemble(
        ("● ", viz.AMBER), ("vowels   ", viz.FAINT), ("● ", viz.ACCENT), ("consonants   ", viz.FAINT),
        ("● ", viz.GREEN), (". (start and end)", viz.FAINT),
    )  # fmt: skip
    return Group(canvas.text(), key)


def heads_before_after(
    before: np.ndarray, after: np.ndarray, chars: list[str], width: int
) -> Table:
    """Each head before and after training, side by side (two heads to a row if they fit)."""
    pair_width = 2 * (2 * len(chars) + 2) + 3
    per_row = max(1, (width + 5) // (pair_width + 5))
    grid = Table.grid(padding=(0, 5))
    for _ in range(min(per_row, len(before))):
        grid.add_column(no_wrap=True)

    def pair(h: int) -> Table:
        inner = Table.grid(padding=(0, 3))
        inner.add_column(no_wrap=True)
        inner.add_column(no_wrap=True)
        inner.add_row(
            Group(Text.assemble((f"head {h + 1} ", "bold"), ("before", viz.FAINT)), *attention_grid(before[h], chars, numbers=False)),
            Group(Text.assemble(("after", viz.GREEN)), *attention_grid(after[h], chars, numbers=False)),
        )  # fmt: skip
        return inner

    heads = list(range(len(before)))
    for start in range(0, len(heads), per_row):
        grid.add_row(*(pair(h) for h in heads[start : start + per_row]))
    return grid


def habits(lab: Lab) -> str:
    """Say, in numbers, where each head tends to look on the held-back words."""
    inputs, targets = lab.data.batch(lab.data.val)
    weights = lab.model.forward(inputs).weights  # (batch, heads, time, time)
    valid = targets >= 0
    valid[:, 0] = False  # the first position has nowhere to look but itself
    rows, cols = np.nonzero(valid)
    parts = []
    for h in range(weights.shape[1]):
        w = weights[rows, h, cols]  # (n, time)
        idx = np.arange(len(cols))
        mine = w[idx, cols].mean()
        prev = w[idx, cols - 1].mean()
        start = w[:, 0].mean()
        options = {
            "itself": mine,
            "the letter just before": prev,
            "the starting `.`": start,
        }
        where, share = max(options.items(), key=lambda kv: kv[1])
        parts.append(f"head {h + 1} spends {share:.0%} of its attention on {where}")
    return (
        "Averaged over the held-back words, "
        + " and ".join(parts)
        + ". Nobody told them to. Those habits are simply what lowered the loss."
    )
