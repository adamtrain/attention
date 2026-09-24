"""Chapter: tokens become vectors."""

from __future__ import annotations

import numpy as np
from rich.table import Table
from rich.text import Text

from .. import viz
from ..views import display, token_chip
from .lab import Lab
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    p, vocab, width = lab.model.params, lab.vocab, lab.model.config.width
    stage.say(
        "A token ID is only a name tag: 19 isn't more of anything than 5. So each token "
        f"looks up its own list of {width} numbers, called its [b]embedding[/b], in a table:"
    )
    table = p["embed.token"]
    scale = viz.scale_of(table)
    stage.show(embedding_table(lab, table.T, scale))
    stage.say(
        f"Each column is one token's {width} numbers. That's room to describe a character: "
        "Is it a vowel? Does it tend to end words? Nobody tells the model what the numbers "
        "should mean. They start out random, as they are here, and training shapes them."
    )
    stage.wait()

    word = lab.corpus.example
    ids = vocab.sequence(word)[:-1]
    pos = min(3, len(ids) - 1)
    ch = vocab.chars[ids[pos]]
    stage.say(
        f"One problem: an `{ch}` at the start of a word and an `{ch}` at the end would get "
        "identical vectors. So every position has an embedding too, and the model adds the two:"
    )
    tok, where = table[ids[pos]], p["embed.position"][pos]
    stage.show(sum_rows(lab, ids[pos], pos, tok, where, scale))
    stage.wait()

    stage.say(
        f"Do that at every position and {display(word)} becomes a grid of numbers, one row "
        "per token. Everything from here on is arithmetic on grids like this one."
    )
    tr = lab.model.forward(np.array([ids]))
    stage.show(input_grid(lab, ids, tr.x0[0], scale))


def embedding_table(lab: Lab, matrix: np.ndarray, scale: float) -> Table:
    vocab = lab.vocab
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    lines = viz.signed_blocks(matrix, scale, width=2)
    for i, line in enumerate(lines):
        dims = f"{i * 2 + 1}–{i * 2 + 2}" if i in (0, len(lines) - 1) else ""
        grid.add_row(Text(dims, style=viz.FAINT), line)
    letters = Text(no_wrap=True)
    for ch in vocab.chars:
        letters.append(f"{ch:<2}", style="bold")
    grid.add_row(Text("token", style=viz.FAINT), letters)
    grid.add_row("", Text(""))
    grid.add_row("", viz.legend(viz.SIGNED, "negative", "positive"))
    return grid


def sum_rows(lab: Lab, token: int, pos: int, tok, where, scale: float) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(
        Text(" "),
        Text.assemble("token ", token_chip(lab.vocab, token)),
        viz.signed_cells(tok, scale),
    )
    grid.add_row(Text("+", style="bold"), Text(f"position {pos}"), viz.signed_cells(where, scale))
    grid.add_row(
        Text("=", style="bold"),
        Text("input", style="bold"),
        viz.signed_cells(tok + where, scale),
    )
    return grid


def input_grid(lab: Lab, ids: list[int], x0: np.ndarray, scale: float) -> Table:
    p = lab.model.params
    grid = Table.grid(padding=(0, 1))
    for _ in range(7):
        grid.add_column(no_wrap=True)
    grid.add_row(
        "", Text("token", style=viz.FAINT), "", Text("position", style=viz.FAINT), "",
        Text("input to the model", style=viz.FAINT), "",
    )  # fmt: skip
    for t, token in enumerate(ids):
        grid.add_row(
            token_chip(lab.vocab, token),
            viz.signed_cells(p["embed.token"][token], scale, 1),
            Text("+", style=viz.FAINT),
            viz.signed_cells(p["embed.position"][t], scale, 1),
            Text("=", style=viz.FAINT),
            viz.signed_cells(x0[t], scale, 2),
            Text(f"{t}", style=viz.FAINT),
        )
    return grid
