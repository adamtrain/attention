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
    p, vocab, c = lab.model.params, lab.vocab, lab.model.config
    width = c.width
    stage.say(
        "A token ID is only a name tag. Token 19 isn't bigger than token 5, or more like token "
        "18 than token 2, so adding or multiplying IDs means nothing. The model needs a "
        "description of each token that it [i]can[/i] do arithmetic on, and where tokens that "
        "behave alike can look alike."
    )
    stage.say(
        f"So each token gets its own list of {width} numbers, called its [b]embedding[/b]. (A "
        "list of numbers like this is a [b]vector[/b]; you'll see that word a lot.) The "
        "embeddings live in a table with one column per token, and turning a token into its "
        "vector is just a lookup: find its column, read off the numbers. Here's your model's "
        "table:"
    )
    table = p["embed.token"]
    scale = viz.scale_of(table)
    stage.show(embedding_table(lab, table.T, scale))
    stage.say(
        f"Each column is one token and each row is one of the {width} numbers. The color "
        "shows each number's sign and size. Think of the numbers as dials that together "
        "describe a token. One dial might end up meaning “is this a vowel?”, another “does "
        "this tend to end a word?”. Tokens that behave alike get similar settings, so "
        "whatever the model learns about one of them partly carries over to the rest."
    )
    stage.say(
        "Nobody decides what the dials mean. The table is weights, like every other part of "
        "the model: it starts out random, as it is here, and training adjusts every number in "
        "it. In practice the meanings don't line up neatly with one dial each; they end up "
        "smeared across many of them. Big models use the same trick with far more numbers "
        "per token: GPT-3 used 12,288."
    )
    stage.wait()

    word = lab.corpus.example
    ids = vocab.sequence(word)[:-1]
    pos = min(3, len(ids) - 1)
    ch = vocab.chars[ids[pos]]
    stage.say(
        f"One problem: an `{ch}` at the start of a word and an `{ch}` at the end would get "
        "identical vectors, but order matters (“tops” and “stop” use the same letters). So "
        f"there's a second table, with a vector for each of the {c.context} places a letter "
        "can sit in a word, and it's learned too. The model adds the token's "
        "vector and the position's vector, number by number, so the result says both "
        "[i]which[/i] letter this is and [i]where[/i] it is:"
    )
    tok, where = table[ids[pos]], p["embed.position"][pos]
    stage.show(sum_rows(lab, ids[pos], pos, tok, where, scale))
    stage.wait()

    stage.say(
        f"Do that at every position and {display(word)} becomes a grid of numbers: one row "
        f"per token, {width} numbers per row. That grid is what flows into the rest of the "
        "model. Everything from here on is arithmetic on grids like this one, and each step "
        f"hands the next one a grid of the same shape, still {width} numbers per position."
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
