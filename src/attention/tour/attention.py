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
    c = lab.model.config
    chars = labels(lab.vocab, inputs[0])
    focus = focus_row(word)
    seen = "".join(chars[: focus + 1])
    me = chars[focus]

    stage.say(
        f"To guess what comes after `{seen}`, the model has to look back at the letters so "
        "far and work out which ones matter. That's what [b]attention[/b] does. Think of it as a "
        "search: every position asks a question, every position advertises what it has, and "
        "each one listens most to the positions that answer its question best."
    )
    stage.say(
        "It's built out of the most common move in any neural network: multiplying a vector by "
        "a grid of weights, called a [b]matrix[/b]. Each number in the result comes from one "
        f"column of the grid. Take the vector's {c.width} numbers and that column's "
        f"{c.width} numbers, multiply them in pairs (first with first, second with second, and "
        f"so on), and add up the {c.width} products. So every column is a recipe: how much of "
        "each input number to mix into one output number. Here's the vector at the "
        f"`{me}` position going through one of those grids:"
    )
    stage.show(multiply(tr.a_in[0, focus], lab.model.params["attn.query"], me))
    stage.say(
        "Every position does that three times, with three different grids, to make three new "
        "vectors:",
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
        "Nobody tells the model what to look for. The three grids are weights, random for now, "
        "and training will shape them into whatever questions and answers make its guesses "
        "better."
    )
    stage.wait()

    scores = tr.q[0, 0, focus] @ tr.k[0, 0, : focus + 1].T
    other = int(np.argmax(scores))
    them = chars[other]
    stage.say(
        "To compare a query with a key, the model uses the same move again: multiply the pairs "
        "and add them up. That's called a [b]dot product[/b]. When the two vectors have big "
        "numbers in the same places, with the same signs, the products are big and positive, "
        "and so is the total: a good match. Where they disagree, products come out negative "
        f"and the total shrinks. Here's the `{me}` position's query against the key of the "
        f"`{them}` at position {other}, in the first of your model's {c.heads} heads (each head "
        f"works with {c.head_width} of the {c.width} numbers; more on heads below):"
    )
    stage.show(dot_product(tr.q[0, 0, focus], tr.k[0, 0, other], me, them))
    stage.say(
        f"(Dividing by √{c.head_width} keeps the scores in a sensible range however many "
        "numbers go into them.) Do that for every query against every key and you get a grid "
        "of scores: a row for each position doing the looking, a column for each position it "
        "might look at. That's one more matrix multiplication, all the queries times all the "
        "keys at once, which is why this runs so fast on a GPU. Here are the scores for "
        f"{word.capitalize()}:"
    )

    raw = (tr.q[0, 0] @ tr.k[0, 0].T) / np.sqrt(c.head_width)
    stage.play(
        lambda: scoring(raw, tr.weights[0, 0], chars), fps=14, start="mask them and softmax them"
    )
    stage.say(
        "Each row is one position, and each column is where it might look. The dots are the "
        "[b]causal mask[/b]: a position can't look ahead at letters that haven't been written "
        "yet. [b]Softmax[/b] (more on it next) turns what's left of each row into weights that "
        "add up to 100%."
    )
    stage.wait()

    stage.say(f"Take the `{me}` row: the position that has read `{seen}` so far. It looks at:")
    stage.show(looks_at(tr.weights[0, 0, focus, : focus + 1], chars[: focus + 1], width=24))
    stage.say(
        "Now the values come in. The position takes each earlier position's value vector, "
        "scales it by that position's weight, and adds them all up: mostly the value from the "
        "biggest bar, a little of the others. That blend is a summary of what it found by "
        "looking back. One last grid of weights turns it into a vector the same size as the "
        "original, and it's added onto the position's own vector. That's how information from "
        "earlier letters reaches the prediction."
    )
    stage.say(
        "For now the weights are random, so where it looks is arbitrary. Training will change that."
    )
    stage.wait()

    stage.say(
        f"Your model has {c.heads} [b]heads[/b] doing all of this side by side. The query, key "
        f"and value vectors are each cut into {c.heads} pieces of {c.head_width} numbers, and "
        "each head does its own matching with its own piece, so each can learn to look for "
        "something different: one might follow the letter just before, another the start of "
        f"the word. Their {c.heads} blends are glued back together at the end:"
    )
    stage.show(heads(lab.model, tr.weights[0], chars))
    stage.say("And this is the code that does it, from your model:")
    stage.show(excerpt(Transformer.forward, "# 2. Attention", "# 3. MLP"))


def multiply(x: np.ndarray, weights: np.ndarray, token: str) -> Table:
    """A vector times a matrix: each column of the grid makes the number under it."""
    out = x @ weights
    strip = viz.signed_blocks(x[:, None], width=2)
    grid = viz.signed_blocks(weights, width=2)
    middle = len(grid) // 2
    table = Table.grid(padding=(0, 2))
    for _ in range(3):
        table.add_column(no_wrap=True)
    n = len(x)
    table.add_row(
        Text(f" {token} ", style=f"bold {viz.ACCENT}"),
        "",
        Text(f"query weights: {n} × {n}, one column per output number", style=viz.FAINT),
    )
    for i, (left, right) in enumerate(zip(strip, grid, strict=True)):
        table.add_row(left, Text("×" if i == middle else "", style="bold"), right)
    table.add_row("", "", Text("↓ " * n, style=viz.FAINT))
    table.add_row(
        Text("query", style="bold"),
        Text("=", style="bold"),
        viz.signed_cells(out, viz.scale_of(out)),
    )
    table.add_row("", "", "")
    table.add_row("", "", viz.legend(viz.SIGNED, "negative", "positive"))
    return table


def dot_product(q: np.ndarray, k: np.ndarray, me: str, them: str) -> Group:
    """One query against one key, pair by pair."""
    products = q * k
    rows = np.stack([q, k, products])
    scale = viz.scale_of(rows)
    lines = viz.matrix_grid(
        rows,
        [f"query of {me}", f"key of {them}", "multiplied"],
        [str(i + 1) for i in range(len(q))],
        lambda v: viz.SIGNED.diverging(v, scale),
        fmt=".2f",
        cell=6,
    )
    total = float(products.sum())
    width = len(q)
    score = total / np.sqrt(width)
    sum_line = Text.assemble(
        ("added up ", viz.FAINT), (f"{total:+.2f}", "bold"),
        ("   ÷ √", viz.FAINT), (str(width), viz.FAINT), ("  =  score ", viz.FAINT),
        (f"{score:+.2f}", f"bold {viz.GREEN if score > 0 else viz.RED}"),
    )  # fmt: skip
    return Group(*lines, Text(""), sum_line)


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

    yield frame(0, 0, 0)
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
