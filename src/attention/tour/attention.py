"""Chapter: attention. Queries, keys and values, and what each is for."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import Transformer, share, softmax
from ..views import attention_grid, label, looks_at
from .code import excerpt
from .lab import Lab
from .stage import Frame, Stage, hold

SHOWN = 16  # tokens of the example in the attention grids

# A made-up lookup, in two numbers per vector, to see the arithmetic before the real thing.
TOY_KEYS = np.array([[2.0, 0.1], [0.2, 1.8], [-0.5, -0.3]])
TOY_VALUES = np.array([[0.9, -0.3], [-0.4, 0.8], [0.1, 0.1]])
TOY_QUERY = np.array([1.9, 0.4])


def run(stage: Stage, lab: Lab) -> None:
    tok, c = lab.tokenizer, lab.model.config
    inputs, _ = lab.example()
    ids = [int(t) for t in inputs[0, :SHOWN]]
    names = [label(tok, t) for t in ids]
    tr = lab.model.forward(np.array([ids]))
    block = tr.blocks[0]
    focus = len(ids) - 1
    me = names[focus]

    stage.say(
        "Look at the example so far: "
        + " ".join(f"`{n}`" for n in names[1:])
        + ". To guess what comes next, the model needs more than the last token. It needs to "
        "look back at the ones before it and work out which of them matter. That's what "
        "[b]attention[/b] does, and it's the heart of a transformer."
    )
    stage.say(
        "Think of it as a fuzzy dictionary lookup. A Python dictionary finds an entry whose key "
        "exactly matches what you ask for. Attention compares what you ask for with [i]every[/i] "
        "key, and takes a blend of all the entries, mostly from the best matches. Three kinds "
        "of vector are involved:",
        gap=False,
    )
    stage.console.print()
    stage.show(roles())
    stage.say(
        "Here it is with made-up numbers, two per vector. The position doing the looking asks "
        "a question (its query); three earlier positions each have a key and a value:"
    )
    stage.play(toy, fps=1, start="look it up")
    stage.say(
        "The query matched the first key best, so the answer is mostly the first value, with a "
        "little of the others mixed in. That's the whole idea. The rest of this chapter is "
        "where the queries, keys and values come from, and doing this for every position at "
        "once."
    )
    stage.wait("see where they come from")

    stage.say(
        "Every position makes all three from its own vector (the residual stream from the last "
        "chapter, rescaled by a norm). It multiplies the vector by a grid of weights, a "
        "[b]matrix[/b], once for each. Multiplying a vector by a grid is the most common move "
        "in any neural network. Each number in the result comes from one column of the grid: "
        "multiply the vector's numbers by the column's numbers in pairs, and add up the "
        f"products. Here's the first number of the query at `{me}`, from the first column of "
        f"the query grid in layer 1. Each is {c.width} numbers, drawn as colors:"
    )
    query = lab.model.params["layers.0.attn.query"][:, 0]
    stage.show(column(block.a_in[0, focus], query, me, stage.width))
    stage.say(
        f"Do that for all {c.heads * c.head_width} columns and you have the query. The key and "
        "value grids work the same way. So at every position, in every layer:"
    )
    stage.show(shapes(lab))
    stage.wait()

    stage.say(
        "Why three different vectors? Why not just compare the positions' own vectors with each "
        "other? Because asking, being found, and telling are different jobs, and three separate "
        "grids let training make each one whatever is most useful:"
    )
    stage.show(jobs())
    stage.say(
        "In a fable that begins “The Fox and the Grapes”, a position later on might ask “which "
        "animal is this about?”. The title's `Fox` should be easy to find for that question, "
        "so its [b]key[/b] says something like “I'm an animal named in the title”. But what it "
        "should [i]hand over[/i] is which animal: its [b]value[/b] carries “Fox”. The key is "
        "how a position gets found; the value is what it tells you once you've found it."
    )
    stage.say(
        "So queries and keys only ever meet each other, in the comparisons that decide where "
        "to look. Values only ever get blended by the results. Researchers call these two "
        "halves the [b]QK circuit[/b] (where to look) and the [b]OV circuit[/b] (what gets "
        "moved; O is a last grid you'll meet below). You'll see a real, trained example of the "
        "difference in the chapter on why layers stack."
    )
    stage.wait("compare them")

    head_q, head_k = block.q[0, 0], share(block.k, c.group)[0, 0]
    scores = head_q[focus] @ head_k[: focus + 1].T
    other = int(np.argmax(scores[:-1]))
    them = names[other]
    stage.say(
        "To compare a query with a key, the model uses the same move again: multiply the pairs "
        "and add them up. That's called a [b]dot product[/b]. When the two vectors have big "
        "numbers in the same places, with the same signs, the total is big and positive: a good "
        "match. Where they disagree, the products are negative and the total shrinks. Here's "
        f"the query at `{me}` against the key of `{them}`, in the first of your model's "
        f"{c.heads} heads (each head works with its own {c.head_width} numbers; more on heads "
        "below):"
    )
    stage.show(dot_product(head_q[focus], head_k[other], me, them, stage.width))
    stage.say(
        f"(Dividing by √{c.head_width} keeps the scores in a sensible range however many "
        "numbers go into them.) Do that for every query against every key and you get a grid "
        "of scores: a row for each position doing the looking, a column for each position it "
        "might look at. That's one more matrix multiplication, all the queries times all the "
        "keys at once, which is why this runs so fast on a GPU. Here's the grid for the "
        "example, in layer 1's first head:"
    )
    raw = head_q @ head_k.T / np.sqrt(c.head_width)
    stage.play(
        lambda: scoring(raw, block.weights[0, 0], names), fps=14, start="mask them and softmax them"
    )
    stage.say(
        "The dots are the [b]causal mask[/b]: a position can't look ahead at tokens that haven't "
        "been written yet. [b]Softmax[/b] (more on it soon) turns what's left of each row into "
        "weights that add up to 100%."
    )
    stage.wait()

    stage.say(f"Take the `{me}` row, the last one. It looks at:")
    stage.show(looks_at(block.weights[0, 0, focus], names, width=24))
    stage.say(
        "Now the values come in. The position takes each earlier position's value, scales it by "
        "that position's weight, and adds them up: mostly the value from the biggest bar, a "
        "little of the others. That blend is what it found by looking back. A last grid, the "
        f"[b]output[/b] grid (the O), turns the blends from all {c.heads} heads into "
        f"{c.width} numbers, and they're [i]added[/i] onto the position's vector in the "
        "residual stream. That's how information from earlier tokens reaches the prediction."
    )
    stage.say(
        "For now the weights are random, so where it looks is arbitrary. Training will change that."
    )
    stage.wait("meet the heads")

    stage.say(
        f"Your model has {c.heads} [b]heads[/b] doing all of this side by side in each layer. "
        f"Each head has its own {c.head_width}-number queries, so each can ask a different "
        "question: one might follow the token just before, another the start of the sentence. "
        f"Their {c.heads} blends are glued back together before the output grid:"
    )
    stage.show(heads(lab.model, tr.weights[0, 0], names, stage.width))
    stage.say(
        f"Look closely and you'll see heads come in pairs. Your model has {c.heads} query heads "
        f"but only {c.kv_heads} key heads and {c.kv_heads} value heads: each key and value head "
        f"is shared by {c.group} query heads. That's called [b]grouped-query attention[/b] "
        "(GQA). The queries still ask different questions, but they're asked of the same keys, "
        "and answered from the same values. It saves memory when the model writes, for reasons "
        "you'll see in the KV cache chapter, and costs little. Llama 3 8B has 32 query heads "
        "sharing 8 key/value heads."
    )
    stage.show(grouping(c.heads, c.kv_heads))
    stage.say("And this is the code that does it all, from your model:")
    stage.show(excerpt(Transformer.block, "# Attention", "# MLP"))


# ── The idea, with made-up numbers ────────────────────────────────────────────


def roles() -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column()
    grid.add_row("query", "what am I looking for?")
    grid.add_row("key", "what do I have? (how others find me)")
    grid.add_row("value", "what I'll hand over if you pick me")
    return grid


def toy() -> Iterator[Frame]:
    scores = TOY_KEYS @ TOY_QUERY
    weights = softmax(scores)
    blend = weights @ TOY_VALUES
    names = ["position 1", "position 2", "position 3"]

    def vec(v: np.ndarray, style: str = "") -> Text:
        return Text(f"[{v[0]:5.2f}, {v[1]:5.2f}]", style=style)

    def frame(stage: int) -> Group:
        grid = Table.grid(padding=(0, 3))
        for justify in ("left", "left", "right", "right", "left"):
            grid.add_column(justify=justify, no_wrap=True)
        grid.add_row(
            "", Text("key", style=viz.FAINT), Text("score", style=viz.FAINT) if stage >= 1 else "",
            Text("weight", style=viz.FAINT) if stage >= 2 else "", Text("value", style=viz.FAINT),
        )  # fmt: skip
        for i, name in enumerate(names):
            grid.add_row(
                Text(name, style="bold"),
                vec(TOY_KEYS[i]),
                Text(f"{scores[i]:+.2f}", style="bold") if stage >= 1 else "",
                Text(f"{weights[i]:.0%}", style=f"bold {viz.ACCENT}") if stage >= 2 else "",
                vec(TOY_VALUES[i], viz.FAINT if stage < 3 else ""),
            )
        lines = [
            Text.assemble(("query ", viz.FAINT), vec(TOY_QUERY, f"bold {viz.AMBER}")),
            Text(""),
            grid,
            Text(""),
        ]
        steps = [
            "Compare the query with each key: multiply the pairs and add them up.",
            "Higher score, better match. (Query · key 1 = 1.9 × 2.0 + 0.4 × 0.1 = 3.84.)",
            "Softmax turns the scores into weights that add up to 100%.",
            "The answer: each value, times its weight, all added up.",
        ]
        lines.append(Text(steps[stage], style=viz.FAINT))
        if stage >= 3:
            lines.append(Text.assemble(("answer ", viz.FAINT), vec(blend, f"bold {viz.GREEN}")))
        return Group(*lines)

    for stage in range(4):
        yield hold(frame(stage), 2.2)


# ── Where the vectors come from ───────────────────────────────────────────────


def column(x: np.ndarray, col: np.ndarray, token: str, width: int) -> Table:
    """One output number: a vector and one column of a grid, multiplied in pairs, added up."""
    products = x * col
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, justify="right", style=viz.FAINT)
    grid.add_column(no_wrap=True)
    at = f"at {token}" if len(token) <= 7 else "the vector"  # no wider than "= products"
    room = width - len("= products") - 1
    grid.add_row(at, viz.signed_fitted(x, viz.scale_of(x), room))
    grid.add_row("× column 1", viz.signed_fitted(col, viz.scale_of(col), room))
    grid.add_row("= products", viz.signed_fitted(products, viz.scale_of(products), room))
    grid.add_row(
        "",
        Text.assemble(
            ("add them all up: ", viz.FAINT), (f"{products.sum():+.3f}", "bold"),
            ("  the query's first number", viz.FAINT),
        ),
    )  # fmt: skip
    grid.add_row("", Text(""))
    grid.add_row("", viz.legend(viz.SIGNED, "negative", "positive"))
    return grid


def shapes(lab: Lab) -> Table:
    c = lab.model.config
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column(no_wrap=True, style=viz.FAINT)
    grid.add_column(no_wrap=True)
    rows = [
        ("query", f"{c.width} × {c.heads * c.head_width} grid", f"{c.heads} heads × {c.head_width} numbers"),
        ("key", f"{c.width} × {c.kv_heads * c.head_width} grid", f"{c.kv_heads} heads × {c.head_width} numbers"),
        ("value", f"{c.width} × {c.kv_heads * c.head_width} grid", f"{c.kv_heads} heads × {c.head_width} numbers"),
    ]  # fmt: skip
    for name, size, result in rows:
        grid.add_row(name, size, result)
    return grid


def jobs() -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column()
    grid.add_row("query", "shaped for asking: what would help me guess what comes next?")
    grid.add_row("key", "shaped for being found: what kind of question am I an answer to?")
    grid.add_row("value", "shaped for telling: once found, what's worth passing on?")
    return grid


# ── Real scores ───────────────────────────────────────────────────────────────


def dot_product(q: np.ndarray, k: np.ndarray, me: str, them: str, width: int) -> Group:
    """One query against one key, pair by pair."""
    products = q * k
    rows = np.stack([q, k, products])
    scale = viz.scale_of(rows)
    names = [f"query of {me}", f"key of {them}", "multiplied"]
    cell = 5 if max(map(len, names)) + 2 + 5 * len(q) <= width else 3  # room for the numbers?
    lines = viz.matrix_grid(
        rows,
        names,
        [str(i + 1) for i in range(len(q))],
        lambda v: viz.SIGNED.diverging(v, scale),
        fmt=".1f" if cell >= 5 else None,
        cell=cell,
    )
    total = float(products.sum())
    score = total / np.sqrt(len(q))
    sum_line = Text.assemble(
        ("added up ", viz.FAINT), (f"{total:+.2f}", "bold"),
        ("   ÷ √", viz.FAINT), (str(len(q)), viz.FAINT), ("  =  score ", viz.FAINT),
        (f"{score:+.2f}", f"bold {viz.GREEN if score > 0 else viz.RED}"),
    )  # fmt: skip
    return Group(*lines, Text(""), sum_line)


def scoring(raw: np.ndarray, weights: np.ndarray, names: list[str]) -> Iterator[Frame]:
    t = len(names)
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
        return Group(caption, Text(""), *score_rows(raw, weights, names, masked, soft))

    yield frame(0, 0, 0)
    for r in range(t):
        yield frame(1, r + 1, 0), 0.07
    yield hold(frame(1, t, 0), 1.2)
    for r in range(t):
        yield frame(2, t, r + 1), 0.12
    yield hold(frame(2, t, t), 0.5)


def score_rows(
    raw: np.ndarray, weights: np.ndarray, names: list[str], masked: int, soft: int
) -> list[Text]:
    t = len(names)
    cell = 4
    scale = viz.scale_of(raw)
    pad = max(len(n) for n in names)
    lines = []
    for r in range(t):
        line = Text(no_wrap=True)
        line.append(f"{names[r]:>{pad}} ", style=f"bold {viz.ACCENT}")
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
                line.append(f"{body[:cell]:^{cell}}", viz.style(viz.ink_for(bg), bg))
        lines.append(line)
    return lines


# ── Heads ─────────────────────────────────────────────────────────────────────


def heads(model: Transformer, weights: np.ndarray, names: list[str], width: int) -> Table:
    """Every head in layer 1, side by side (as many to a row as fit)."""
    each = 2 * len(names) + max(len(n) for n in names) + 3
    per_row = max(1, min(len(weights), width // each))
    grid = Table.grid(padding=(0, 3))
    for _ in range(per_row):
        grid.add_column(no_wrap=True)
    cells = []
    for h, w in enumerate(weights):
        title = Text.assemble(
            (f"head {h + 1}", "bold"),
            (f"  keys and values from pair {h // model.config.group + 1}", viz.FAINT),
        )
        cells.append(Group(title, *attention_grid(w, names, numbers=False)))
    for start in range(0, len(cells), per_row):
        row = cells[start : start + per_row]
        grid.add_row(*row, *([""] * (per_row - len(row))))
    return grid


def grouping(heads: int, kv_heads: int) -> Text:
    group = heads // kv_heads
    top = Text("query heads      ", style=viz.FAINT, no_wrap=True)
    for h in range(heads):
        top.append(f" {h + 1} ", style=f"bold {viz.ACCENT}")
        top.append(" ")
    mid = Text(" " * 17, no_wrap=True)
    bottom = Text("key/value heads  ", style=viz.FAINT, no_wrap=True)
    for g in range(kv_heads):
        mid.append(("╰─" + "─" * (4 * group - 5) + "╯").center(4 * group), style=viz.FAINT)
        bottom.append(f" {g + 1} ".center(4 * group), style=f"bold {viz.AMBER}")
    return Text("\n").join([top, mid, bottom])
