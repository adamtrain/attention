"""Chapter: tokens become vectors."""

from __future__ import annotations

import numpy as np
from rich.table import Table
from rich.text import Text

from .. import viz
from ..views import chip, label
from .lab import Lab
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    p, tok, c = lab.model.params, lab.tokenizer, lab.model.config
    width = c.width
    stage.say(
        "A token ID is only a name tag. Token 300 isn't bigger than token 5, or more like token "
        "299 than token 2, so adding or multiplying IDs means nothing. The model needs a "
        "description of each token that it [i]can[/i] do arithmetic on, and where tokens that "
        "behave alike can look alike."
    )
    stage.say(
        f"So each token gets its own list of {width} numbers, called its [b]embedding[/b]. (A "
        "list of numbers like this is a [b]vector[/b]; you'll see that word a lot.) The "
        f"embeddings live in a table with one row per token: {len(tok)} rows of {width} "
        "numbers. Turning a token into its vector is just a lookup: find its row, read off the "
        "numbers. Here are the rows for the tokens of your example passage:"
    )
    inputs, _ = lab.example()
    ids = [int(t) for t in inputs[0, 1:13]]
    table = p["embed"]
    scale = viz.scale_of(table)
    stage.show(rows(lab, ids, scale, stage.width))
    stage.say(
        f"Each row is one token, each column one of its {width} numbers, and the color shows "
        "each number's sign and size. Think of the numbers as dials that together describe a "
        "token: one might end up meaning “is this a name?”, another “does this end a "
        "sentence?”. Tokens that behave alike get similar settings, so whatever the model "
        "learns about one of them partly carries over to the rest."
    )
    stage.say(
        "Nobody decides what the dials mean. The table is weights, like every other part of the "
        "model: it starts out random, as it is here, and training adjusts every number in it. "
        "In practice the meanings don't line up neatly with one dial each; they're smeared "
        f"across many. The table holds {table.size:,} of your model's {lab.model.size:,} "
        "parameters. GPT-3's holds 617 million: 50,257 tokens with 12,288 numbers each."
    )
    stage.wait()

    stage.say(
        "What about order? “The Fox ate the Hen” and “The Hen ate the Fox” have the same "
        "tokens. Older models like GPT-2 and GPT-3 had a second table with a vector for every "
        "position, added to each token's vector. Today's models don't: position is brought in "
        "later, inside attention, by rotating vectors. That's two chapters from now. For now, "
        "each token is just its row of the table."
    )
    tr = lab.model.forward(inputs)
    t = inputs.shape[1]
    stage.say(
        f"Stack the rows up and your example becomes a grid of numbers: {t} tokens, {width} "
        "numbers each. This grid is the start of what's called the [b]residual stream[/b]. "
        "Every layer of the model will read it and add its own numbers to it, and at the end "
        "it's turned back into a guess about the next token, using the very same table again, "
        "read the other way. The grid keeps its shape the whole way through: "
        f"{t} × {width}."
    )
    stage.show(stream_shape(lab, tr.blocks[0].x[0], ids, scale, stage.width))


def rows(lab: Lab, ids: list[int], scale: float, width: int) -> Table:
    tok = lab.tokenizer
    tiles = max(len(label(tok, t)) + 2 for t in ids)
    numbers = tiles + 1 + len(str(max(ids))) + 1 + lab.model.config.width <= width  # ids fit?
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    seen = []
    for token in ids:
        if token in seen:
            continue
        seen.append(token)
        grid.add_row(
            chip(tok, token),
            Text(str(token) if numbers else "", style=viz.FAINT),
            viz.signed_cells(lab.model.params["embed"][token], scale, 1),
        )
    grid.add_row("", "", "")
    grid.add_row("", "", viz.legend(viz.SIGNED, "negative", "positive"))
    return grid


def stream_shape(lab: Lab, x: np.ndarray, ids: list[int], scale: float, width: int) -> Table:
    tok = lab.tokenizer
    tiles = max(len(label(tok, t)) + 2 for t in ids[:8])
    numbers = tiles + 1 + x.shape[1] + 2 <= width  # room for the row numbers?
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, style=viz.FAINT)
    for i, token in enumerate(ids[:8]):
        grid.add_row(
            chip(tok, token), viz.signed_cells(x[i + 1], scale, 1), f"{i + 1}" if numbers else ""
        )
    grid.add_row(Text("…", style=viz.FAINT), Text(f"{len(x)} rows in all", style=viz.FAINT), "")
    return grid
