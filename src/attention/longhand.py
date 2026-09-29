"""The whole forward pass, worked out longhand, on a model small enough to print.

`attention math` shrinks your model's architecture until every number fits on the screen: 8
numbers per token, 2 layers, 2 query heads sharing 1 key/value head, an MLP 12 wide, and a
vocabulary of 12 whole words from fable titles. It trains for a moment on titles like The Fox
and the Crow, then reads one sentence, and every operation is animated as it happens: which
numbers go in, what's done with them, and where each result lands. It runs the model's own
functions, and checks its answer against the model's own forward pass.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from rich.console import Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import viz
from .corpus import VOCAB, built_in
from .model import (
    EPS,
    Array,
    Config,
    Transformer,
    angles,
    causal_mask,
    cross_entropy,
    merge_heads,
    rmsnorm,
    rope,
    share,
    sigmoid,
    silu,
    softmax,
    split_heads,
)
from .tokenizer import END
from .tour.stage import Frame, Stage, hold
from .train import Adam, clip

ANIMALS = 8  # the commonest in titles like The Fox and the Crow
CONTEXT = 8
ROWS = 26  # the tallest picture, a blank line and the prompt: the terminal needs this many
READ = viz.style("#ffffff", viz.ACCENT, bold=True)  # numbers being read
WRITE = viz.style(viz.DARK, viz.GREEN, bold=True)  # the number being written


# ── A tiny model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Words:
    """A vocabulary of whole words: the end marker, The, and, the, and some animals."""

    pieces: tuple[str, ...]

    def label(self, token: int) -> str:
        return "‹end›" if token == END else self.pieces[token].replace(" ", "·")

    def encode(self, text: str) -> list[int]:
        """The end marker, then each word: the first as it is, the rest with a space before.

        Capitals don't matter.
        """
        ids = [END]
        for i, word in enumerate(text.split()):
            same = [p for p in self.pieces[1:] if p.strip().lower() == word.lower()]
            if not same:
                known: dict[str, str] = {}
                for p in self.pieces[1:]:
                    known.setdefault(p.strip().lower(), p.strip())
                raise ValueError(
                    f"the little model doesn't know “{word}”. It knows: {', '.join(known.values())}"
                )
            spaced = [p for p in same if p.startswith(" ") == (i > 0)]  # The to start, then ·the
            ids.append(self.pieces.index((spaced or same)[0]))
        if len(ids) > CONTEXT:
            raise ValueError(f"the little model reads at most {CONTEXT - 1} words")
        return ids


def titles() -> list[tuple[str, str]]:
    """Every fable title of the form The X and the Y, as (X, Y)."""
    found = []
    for doc in built_in("fables").documents:
        if m := re.fullmatch(r"The (\w+) and the (\w+)", doc.split("\n", 1)[0]):
            found.append((m.group(1), m.group(2)))
    return found


def toy(seed: int = 1, steps: int = 600) -> tuple[Transformer, Words, int]:
    """The little model, trained on titles like The Fox and the Crow. Also: how many titles."""
    pairs = titles()
    animals = [w for w, _ in Counter(w for p in pairs for w in p).most_common(ANIMALS)]
    words = Words(("<|endoftext|>", "The", " and", " the", *(" " + a for a in animals)))
    ids = {p: i for i, p in enumerate(words.pieces)}
    kept = [(a, b) for a, b in pairs if a in animals and b in animals]
    stream = [t for a, b in kept for t in (END, 1, ids[" " + a], 2, 3, ids[" " + b])] + [END]
    data = np.array(stream)
    config = Config(len(words.pieces), CONTEXT, width=8, layers=2, heads=2, kv_heads=1, hidden=12)
    model = Transformer.create(config, np.random.default_rng([seed, 1]))
    rng, optimizer = np.random.default_rng([seed, 2]), Adam(model.params)
    for step in range(steps):
        starts = rng.integers(0, len(data) - CONTEXT, 32)
        inputs = np.stack([data[s : s + CONTEXT - 1] for s in starts])
        targets = np.stack([data[s + 1 : s + CONTEXT] for s in starts])
        trace = model.forward(inputs)
        _, dlogits = cross_entropy(trace.logits, targets)
        grads, _ = model.backward(trace, dlogits)
        lr = 0.02 * (0.1 + 0.9 * (1 - step / steps))
        optimizer.step(model.params, clip(grads, 1.0)[0], lr)
    return model, words, len(kept)


# ── Every number, one operation at a time ─────────────────────────────────────


@dataclass
class Layer:
    x: Array  # (time, width): the stream coming in
    a_inv: Array  # (time, 1): 1 / the root mean square of each row
    a_in: Array  # the stream, normalized, for attention
    q0: Array  # (heads, time, head width): queries, before RoPE
    k0: Array  # (kv heads, time, head width): keys, before RoPE
    v: Array  # (kv heads, time, head width): values
    q: Array  # queries, turned by RoPE
    k: Array  # keys, turned by RoPE
    scores: Array  # (heads, time, time): queries · keys ÷ √(head width), before the mask
    weights: Array  # (heads, time, time): after the mask and softmax
    heads: Array  # (heads, time, head width): each head's weighted mix of values
    mixed: Array  # (time, width): the heads side by side
    attn: Array  # (time, width): mixed · W_o, what attention adds
    mid: Array  # the stream after attention
    m_inv: Array
    m_in: Array  # the stream, normalized, for the MLP
    gate: Array  # (time, hidden)
    up: Array
    act: Array  # SiLU(gate)
    hidden: Array  # SiLU(gate) × up
    mlp: Array  # (time, width): hidden · W_down, what the MLP adds
    out: Array  # the stream after the layer


@dataclass
class Walk:
    model: Transformer
    words: Words
    ids: list[int]
    turn: Array  # (time, head width / 2): RoPE's angle for each position and pair
    x: Array  # the embeddings: the stream's start
    layers: list[Layer]
    f_inv: Array
    f_in: Array  # the stream at the end, normalized
    logits: Array  # (time, vocab)
    probs: Array
    error: float  # how far the longhand logits are from the model's own forward pass

    @property
    def labels(self) -> list[str]:
        return [self.words.label(t) for t in self.ids]


def walk(model: Transformer, words: Words, ids: list[int]) -> Walk:
    """The forward pass, for one sentence, keeping every number along the way.

    The same functions as Transformer.forward, called one at a time instead of in a loop.
    """
    c, p = model.config, model.params
    t = len(ids)
    turn = angles(np.arange(t), c.head_width, c.rope_base)
    x = p["embed"][ids]
    start, layers = x, []
    for i in range(c.layers):
        w = model.layer(i)
        a_in, a_inv = rmsnorm(x, w["attn.norm"])
        q0 = split_heads((a_in @ w["attn.query"])[None], c.heads)[0]
        k0 = split_heads((a_in @ w["attn.key"])[None], c.kv_heads)[0]
        v = split_heads((a_in @ w["attn.value"])[None], c.kv_heads)[0]
        q, k = rope(q0, turn), rope(k0, turn)
        scores = q @ share(k[None], c.group)[0].swapaxes(-1, -2) / np.sqrt(c.head_width)
        weights = softmax(np.where(causal_mask(t), scores, -np.inf))
        heads = weights @ share(v[None], c.group)[0]
        mixed = merge_heads(heads[None])[0]
        attn = mixed @ w["attn.out"]
        mid = x + attn
        m_in, m_inv = rmsnorm(mid, w["mlp.norm"])
        gate, up = m_in @ w["mlp.gate"], m_in @ w["mlp.up"]
        act = silu(gate)
        hidden = act * up
        mlp = hidden @ w["mlp.down"]
        out = mid + mlp
        layers.append(
            Layer(
                x,
                a_inv,
                a_in,
                q0,
                k0,
                v,
                q,
                k,
                scores,
                weights,
                heads,
                mixed,
                attn,
                mid,
                m_inv,
                m_in,
                gate,
                up,
                act,
                hidden,
                mlp,
                out,
            )
        )
        x = out
    f_in, f_inv = rmsnorm(x, p["norm"])
    logits = f_in @ p["embed"].T
    error = float(np.abs(logits - model.forward(np.array([ids])).logits[0]).max())
    return Walk(model, words, ids, turn, start, layers, f_inv, f_in, logits, softmax(logits), error)


# ── Drawing numbers ───────────────────────────────────────────────────────────


def number(v: float, digits: int) -> str:
    """Signed, with `digits` decimals: one fewer from 10 up, to keep every number as wide."""
    if abs(round(float(v), digits)) >= 10:
        digits -= 1  # +10.2 beside +1.23, +10 beside +1.2
    return f"{v:+.{max(digits, 0)}f}"


def digits_for(values: Array, columns: int, label: int, room: int) -> int:
    """Two decimals if a row of them fits in `room`, or one."""
    finite = values[np.isfinite(values)]
    width = max((len(number(v, 2)) for v in finite), default=5)
    return 2 if label + (width + 1) * columns <= room else 1


@dataclass
class Grid:
    """A matrix to draw: its numbers, its name, and what its rows and columns stand for."""

    name: str
    values: Array
    rows: Sequence[str]
    cols: Sequence[str]
    note: str = ""

    @property
    def shape(self) -> str:
        r, c = self.values.shape
        return f"{r} × {c}"


def draw(
    g: Grid,
    room: int,
    *,
    row: int | None = None,
    col: int | None = None,
    cell: tuple[int, int] | None = None,
    done: Array | None = None,
    masked: Array | None = None,
    read: Style = READ,
) -> Group:
    """A matrix as numbers on colors for their sign and size.

    `row` and `col` are being read (purple), `cell` is being written (green), cells where
    `done` is False aren't worked out yet, and `masked` ones are −∞.
    """
    values = g.values
    label = max(len(r) for r in g.rows)
    digits = digits_for(values, values.shape[1], label, room)
    numbers = max((len(number(v, digits)) for v in values[np.isfinite(values)].ravel()), default=5)
    width = max(numbers, *(len(c) for c in g.cols))
    staggered = label + (width + 1) * len(g.cols) > room  # headings too wide to sit in a row
    if staggered:
        width = numbers
    scale = viz.scale_of(values[np.isfinite(values)]) if np.isfinite(values).any() else 1.0
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", no_wrap=True)
    for _ in g.cols:
        table.add_column(justify="right", no_wrap=True)
    if not staggered:
        table.add_row("", *(heading(c, j == col) for j, c in enumerate(g.cols)))
    for i, name in enumerate(g.rows):
        cells = []
        for j, v in enumerate(values[i]):
            if masked is not None and masked[i, j]:
                cells.append(Text("−∞".rjust(width), style=viz.FAINT))
            elif done is not None and not done[i, j]:
                cells.append(Text("·".rjust(width), style=viz.FAINT))
            elif cell == (i, j):
                cells.append(Text(number(v, digits).rjust(width), style=WRITE))
            elif i == row or j == col:
                cells.append(Text(number(v, digits).rjust(width), style=read))
            else:
                bg = viz.SIGNED.diverging(float(v), scale)
                cells.append(
                    Text(number(v, digits).rjust(width), style=viz.style(viz.ink_for(bg), bg))
                )
        lit = i == row or (cell is not None and cell[0] == i)
        mark = viz.GREEN if read is WRITE or (cell is not None and i != row) else viz.ACCENT
        table.add_row(Text(name, style=f"bold {mark}" if lit else viz.FAINT), *cells)
    head = Text.assemble((g.name, "bold"), (f"  {g.shape}", viz.FAINT), (f"   {g.note}", viz.FAINT))
    if staggered:
        return Group(head, *stagger(g.cols, label, width, col), table)
    return Group(head, table)


def heading(text: str, lit: bool) -> Text:
    return Text(text, style=f"bold {viz.ACCENT}" if lit else viz.FAINT)


def stagger(heads: Sequence[str], label: int, width: int, col: int | None) -> list[Text]:
    """Column headings on two lines, taking turns, over columns narrower than they are."""
    lines = [Text(no_wrap=True), Text(no_wrap=True)]
    for j, text in enumerate(heads):
        line = lines[j % 2]
        end = label + (j + 1) * (width + 1)  # where column j ends, after the gaps before it
        line.append(" " * (end - len(text) - len(line)))
        line.append_text(heading(text, j == col))
    return lines


def strip(
    lines: Sequence[tuple[str, Array, Style | None]],
    room: int,
    heads: Sequence[str] | None = None,
) -> RenderableType:
    """Rows of numbers lined up, one column per term: what gets combined, and how.

    With `heads`, the columns get headings, on two staggered lines if they don't fit on one.
    """
    label = max(len(name) for name, *_ in lines)
    n = max(len(v) for _, v, _ in lines)
    digits = digits_for(np.concatenate([v for _, v, _ in lines]), n, label, room)
    width = max(len(number(v, digits)) for _, values, _ in lines for v in values)
    widest = max([width, *(len(h) for h in heads or ())])
    staggered = heads is not None and label + (widest + 1) * n > room
    if not staggered:
        width = widest
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", no_wrap=True)
    for _ in range(n):
        table.add_column(justify="right", no_wrap=True)
    if heads is not None and not staggered:
        table.add_row("", *(heading(h, False) for h in heads))
    for name, values, style in lines:
        scale = viz.scale_of(values)
        cells = []
        for v in values:
            text = number(v, digits).rjust(width)
            if style is None:
                bg = viz.SIGNED.diverging(float(v), scale)
                cells.append(Text(text, style=viz.style(viz.ink_for(bg), bg)))
            else:
                cells.append(Text(text, style=style))
        table.add_row(Text(name, style=viz.FAINT), *cells)
    if heads is not None and staggered:
        return Group(*stagger(heads, label, width, None), table)
    return table


def result(*parts: tuple[str, str]) -> Text:
    return Text.assemble(*parts)


def dims(n: int) -> list[str]:
    return [str(i + 1) for i in range(n)]


def timing(done: int, total: int, per_row: int) -> float:
    """Slow for the first two numbers, quick for the rest of the first row, then a row a frame."""
    return 1.6 if done < 2 else 0.2 if done < per_row else 0.6


# ── Animations ────────────────────────────────────────────────────────────────


def lookup(w: Walk, room: int) -> Iterator[Frame]:
    """Each token picks a row of the embedding table, and the rows stack up into the stream."""
    table = w.model.params["embed"]
    words = [w.words.label(i) for i in range(len(w.words.pieces))]
    e = Grid("E", table, words, dims(table.shape[1]), "the embedding table: a row for each word")
    x = Grid("X", w.x, w.labels, dims(w.x.shape[1]), "the stream: a row for each token read")
    for t in range(len(w.ids) + 1):
        done = np.repeat(np.arange(len(w.ids))[:, None] <= t, w.x.shape[1], axis=1)
        now = t < len(w.ids)
        say = (
            result(
                (w.labels[t], "bold"),
                (f" is word {w.ids[t]}: its row of E is copied into X", viz.FAINT),
            )
            if now
            else result(("Each row of X is its token's row of E.", viz.FAINT))
        )
        yield (
            Group(
                draw(e, room, row=w.ids[t] if now else None),
                Text(""),
                draw(x, room, row=t if now else None, done=done, read=WRITE),
                Text(""),
                say,
            ),
            (1.4 if t == 0 else 0.7),
        )


def normalize(
    x: Array,
    inv: Array,
    gain: Array,
    out: Array,
    names: tuple[str, str],
    labels: list[str],
    room: int,
) -> Iterator[Frame]:
    """RMSNorm, row by row: the root of the mean square, divide by it, times the gains."""
    t, d = x.shape
    source = Grid(names[0], x, labels, dims(d), "coming in")
    gains = Grid("g", gain[None, :], ["gain"], dims(d), "learned: one per column")
    target = Grid(names[1], out, labels, dims(d), "each row rescaled to a typical size of 1")
    for i in range(t + 1):
        now = i < t
        done = np.repeat(np.arange(t)[:, None] <= i, d, axis=1)
        parts: list[RenderableType] = [draw(source, room, row=i if now else None), Text("")]
        if now:
            squares = x[i] ** 2
            rms = 1 / float(inv[i, 0])
            parts += [
                strip([(f"{names[0]}[{labels[i]}]", x[i], READ), ("squared", squares, None)], room),
                result(
                    ("mean of the squares ", viz.FAINT),
                    (f"{squares.mean():.3f}", "bold"),
                    ("   root ", viz.FAINT),
                    (f"√({squares.mean():.3f} + {EPS:g}) = {rms:.3f}", "bold"),
                    ("   then each number ÷ ", viz.FAINT),
                    (f"{rms:.3f}", "bold"),
                    (" × its gain", viz.FAINT),
                ),
                Text(""),
            ]
        parts += [
            draw(gains, room),
            Text(""),
            draw(target, room, row=i if now else None, done=done, read=WRITE),
        ]
        yield Group(*parts), (2.4 if i == 0 else 0.9)


def matmul(a: Grid, b: Grid, c: Grid, room: int, quick: bool = False) -> Iterator[Frame]:
    """C = A · B, one number at a time: a row of A, a column of B, multiplied pair by pair and
    added up. The first few slowly; then faster, and a row at a time."""
    rows, cols = c.values.shape
    order = [(i, j) for i in range(rows) for j in range(cols)]
    done = np.zeros((rows, cols), dtype=bool)
    k = 0
    while k < len(order):
        take = 1 if k < cols and not quick else cols - order[k][1]  # then the rest of a row
        for i2, j2 in order[k : k + take]:
            done[i2, j2] = True
        last = order[k + take - 1]
        i, j = last
        products = a.values[i] * b.values[:, j]
        yield (
            Group(
                draw(a, room, row=i),
                Text(""),
                strip(
                    [
                        (f"{a.name}[{a.rows[i]}]", a.values[i], READ),
                        (f"{b.name}[:,{b.cols[j]}]", b.values[:, j], READ),
                        ("products", products, None),
                    ],
                    room,
                ),
                result(
                    ("added up: ", viz.FAINT),
                    (f"{products.sum():+.3f}", f"bold {viz.GREEN}"),
                    (f"  → {c.name}[{c.rows[i]}, {c.cols[j]}]", viz.FAINT),
                ),
                Text(""),
                draw(c, room, cell=last, done=done.copy()),
            ),
            (0.5 if quick else timing(k, len(order), cols)),
        )
        k += take
    yield hold(Group(draw(a, room), Text(""), draw(c, room)), 0.3)


def turning(before: Grid, after: Grid, turn: Array, room: int) -> Iterator[Frame]:
    """RoPE, position by position: each pair of numbers turned by its angle."""
    t, hw = before.values.shape
    pairs = hw // 2
    angle = Grid(
        "θ",
        turn,
        before.rows,
        [f"pair {p + 1}" for p in range(pairs)],
        "radians: position × (1, 0.01, …) for pairs (1, 2, …)",
    )
    for i in range(t + 1):
        now = i < t
        done = np.repeat(np.arange(t)[:, None] <= i, hw, axis=1)
        side = Table.grid(padding=(0, 4))
        side.add_row(
            draw(before, room // 2, row=i if now else None),
            draw(after, room // 2, row=i if now else None, done=done, read=WRITE),
        )
        parts: list[RenderableType] = [draw(angle, room, row=i if now else None), Text(""), side]
        if now:
            lines = []
            for p in range(pairs):
                a, b = before.values[i, 2 * p], before.values[i, 2 * p + 1]
                th = float(turn[i, p])
                x, y = a * np.cos(th) - b * np.sin(th), a * np.sin(th) + b * np.cos(th)
                lines.append(
                    result(
                        (f"pair {p + 1}: ", viz.FAINT),
                        (f"({a:+.2f}, {b:+.2f})", "bold"),
                        (f" turned by θ = {th:.2f} {arrow(th)}: ", viz.FAINT),
                        ("(a·cos θ − b·sin θ, a·sin θ + b·cos θ) = ", viz.FAINT),
                        (f"({x:+.2f}, {y:+.2f})", f"bold {viz.GREEN}"),
                    )
                )
            parts += [Text(""), *lines]
        yield Group(*parts), (2.4 if i in (0, 1) else 0.8)


def arrow(theta: float) -> str:
    """Which way an angle points, as an arrow: 0 is →, and it turns counterclockwise."""
    return "→↗↑↖←↙↓↘"[round((theta % (2 * np.pi)) / (np.pi / 4)) % 8]


def attend(w: Walk, layer: Layer, h: int, room: int) -> Iterator[Frame]:
    """One head's scores: its queries against the keys, every pair, ÷ √(head width)."""
    c = w.model.config
    kv = h // c.group
    hw = c.head_width
    q = Grid(f"Q{h + 1}", layer.q[h], w.labels, dims(hw), f"head {h + 1}'s queries")
    k = Grid(
        f"K{kv + 1}", layer.k[kv], w.labels, dims(hw), "keys" + (" (shared)" if c.group > 1 else "")
    )
    s = Grid(
        f"S{h + 1}", layer.scores[h], w.labels, w.labels, "rows: who's looking; columns: at whom"
    )
    t = len(w.ids)
    done = np.zeros((t, t), dtype=bool)
    order = [(i, j) for i in range(t) for j in range(t)]
    k_i = 0
    while k_i < len(order):
        i, j = order[k_i]
        take = 1 if k_i < t else t
        for i2, j2 in order[k_i : k_i + take]:
            done[i2, j2] = True
        i, j = order[k_i + take - 1]
        products = layer.q[h][i] * layer.k[kv][j]
        side = Table.grid(padding=(0, 4))
        side.add_row(draw(q, room // 2, row=i), draw(k, room // 2, row=j))
        yield (
            Group(
                side,
                Text(""),
                strip(
                    [
                        (f"Q{h + 1}[{w.labels[i]}]", layer.q[h][i], READ),
                        (f"K{kv + 1}[{w.labels[j]}]", layer.k[kv][j], READ),
                        ("products", products, None),
                    ],
                    room,
                ),
                result(
                    ("added up ", viz.FAINT),
                    (f"{products.sum():+.3f}", "bold"),
                    (f", ÷ √{hw} = ", viz.FAINT),
                    (f"{products.sum() / np.sqrt(hw):+.3f}", f"bold {viz.GREEN}"),
                ),
                Text(""),
                draw(s, room, cell=(i, j), done=done.copy()),
            ),
            timing(k_i, len(order), t),
        )
        k_i += take


def weigh(w: Walk, layer: Layer, h: int, room: int) -> Iterator[Frame]:
    """The causal mask, then softmax, row by row: e to the power of each score, ÷ their total."""
    t = len(w.ids)
    future = ~causal_mask(t)
    s = Grid(f"S{h + 1}", layer.scores[h], w.labels, w.labels, "scores")
    a = Grid(
        f"A{h + 1}",
        layer.weights[h],
        w.labels,
        w.labels,
        "attention weights: each row adds up to 1",
    )
    yield (
        Group(draw(s, room), Text(""), result(("First the mask: no looking ahead.", viz.FAINT))),
        1.6,
    )
    for i in range(t + 1):
        now = i < t
        done = np.repeat(np.arange(t)[:, None] <= i, t, axis=1)
        parts: list[RenderableType] = [
            draw(s, room, row=i if now else None, masked=future),
            Text(""),
        ]
        if now:
            row = layer.scores[h][i, : i + 1]
            e = np.exp(row - row.max())
            parts += [
                strip(
                    [
                        (f"S{h + 1}[{w.labels[i]}]", row, READ),
                        ("e^(x − max)", e, None),
                        ("÷ total", e / e.sum(), None),
                    ],
                    room,
                ),
                result(
                    ("the total: ", viz.FAINT),
                    (f"{e.sum():.3f}", "bold"),
                    ("   (−∞ becomes 0: the future gets no attention)", viz.FAINT),
                ),
                Text(""),
            ]
        parts.append(draw(a, room, row=i if now else None, done=done, read=WRITE))
        yield Group(*parts), (2.4 if i in (0, 1) else 0.9)


def mix(w: Walk, layer: Layer, h: int, room: int) -> Iterator[Frame]:
    """One head's output: its weights times the values. Each row, a weighted average of V."""
    c = w.model.config
    kv, hw, t = h // c.group, c.head_width, len(w.ids)
    a = Grid(f"A{h + 1}", layer.weights[h], w.labels, w.labels, "weights")
    v = Grid(
        f"V{kv + 1}",
        layer.v[kv],
        w.labels,
        dims(hw),
        "values" + (" (shared)" if c.group > 1 else ""),
    )
    o = Grid(f"O{h + 1}", layer.heads[h], w.labels, dims(hw), f"head {h + 1}'s output")
    done = np.zeros((t, hw), dtype=bool)
    order = [(i, j) for i in range(t) for j in range(hw)]
    k = 0
    while k < len(order):
        take = 1 if k < hw else hw
        for i2, j2 in order[k : k + take]:
            done[i2, j2] = True
        i, j = order[k + take - 1]
        products = layer.weights[h][i] * layer.v[kv][:, j]
        side = Table.grid(padding=(0, 4))
        side.add_row(draw(a, room // 2 + 6, row=i), draw(v, room // 2 - 6, col=j))
        yield (
            Group(
                side,
                Text(""),
                strip(
                    [
                        (f"A{h + 1}[{w.labels[i]}]", layer.weights[h][i], READ),
                        (f"V{kv + 1}[:,{j + 1}]", layer.v[kv][:, j], READ),
                        ("products", products, None),
                    ],
                    room,
                ),
                result(
                    ("added up: ", viz.FAINT),
                    (f"{products.sum():+.3f}", f"bold {viz.GREEN}"),
                    ("   mostly the values it looked at hardest", viz.FAINT),
                ),
                Text(""),
                draw(o, room, cell=(i, j), done=done.copy()),
            ),
            timing(k, len(order), hw),
        )
        k += take


def side_by_side(w: Walk, layer: Layer, room: int) -> Iterator[Frame]:
    """The heads' outputs, next to each other: one row of 8 numbers per position again."""
    c = w.model.config
    t, hw = len(w.ids), c.head_width
    m = Grid("O", layer.mixed, w.labels, dims(c.width), "the heads' outputs, side by side")
    for h in range(c.heads + 1):
        done = np.repeat(np.arange(c.width)[None, :] < h * hw, t, axis=0)
        yield Group(draw(m, room, done=done)), 0.8


def add(x: Grid, y: Grid, z: Grid, room: int) -> Iterator[Frame]:
    """The residual connection: add, number by number, row by row."""
    t, d = z.values.shape
    for i in range(t + 1):
        now = i < t
        done = np.repeat(np.arange(t)[:, None] <= i, d, axis=1)
        parts: list[RenderableType] = [draw(x, room, row=i if now else None), Text("")]
        if now:
            parts += [
                strip(
                    [
                        (f"{x.name}[{x.rows[i]}]", x.values[i], READ),
                        (f"+ {y.name}[{y.rows[i]}]", y.values[i], READ),
                        ("=", z.values[i], None),
                    ],
                    room,
                ),
                Text(""),
            ]
        parts.append(draw(z, room, row=i if now else None, done=done, read=WRITE))
        yield Group(*parts), (2.0 if i == 0 else 0.6)


def switch(g: Grid, u: Grid, act: Array, h: Grid, room: int) -> Iterator[Frame]:
    """SiLU on the gate, g × σ(g), then times up: row by row."""
    t, n = h.values.shape
    for i in range(t + 1):
        now = i < t
        done = np.repeat(np.arange(t)[:, None] <= i, n, axis=1)
        parts: list[RenderableType] = [draw(g, room, row=i if now else None), Text("")]
        if now:
            gi = g.values[i]
            parts += [
                strip(
                    [
                        (f"G[{g.rows[i]}]", gi, READ),
                        ("σ(G)", sigmoid(gi), None),
                        ("SiLU(G) = G·σ", act[i], None),
                        (f"U[{u.rows[i]}]", u.values[i], READ),
                        ("SiLU(G) × U", h.values[i], None),
                    ],
                    room,
                ),
                result(
                    (
                        "σ(x) = 1 / (1 + e^−x): about 0 for negative x, about 1 for positive x, "
                        "so SiLU shuts negative gates and passes positive ones",
                        viz.FAINT,
                    )
                ),
                Text(""),
            ]
        parts.append(draw(h, room, row=i if now else None, done=done, read=WRITE))
        yield Group(*parts), (3.0 if i == 0 else 0.8)


def guess(w: Walk, room: int) -> Iterator[Frame]:
    """Softmax on the last row of logits: the probabilities for the next word."""
    t = len(w.ids) - 1
    words = [w.words.label(i) for i in range(len(w.words.pieces))]
    row = w.logits[t]
    e = np.exp(row - row.max())
    probs = e / e.sum()
    yield (
        Group(
            strip(
                [
                    (f"logits[{w.labels[t]}]", row, READ),
                    ("e^(x − max)", e, None),
                    ("÷ total", probs, None),
                ],
                room,
                words,
            ),
            result(("the total: ", viz.FAINT), (f"{e.sum():.3f}", "bold")),
        ),
        2.0,
    )
    bars = Table.grid(padding=(0, 2))
    for _ in range(3):
        bars.add_column(no_wrap=True)
    for i in np.argsort(-probs):
        bars.add_row(
            Text(words[i], style="bold" if i == np.argmax(probs) else ""),
            viz.bar(float(probs[i]), 30, viz.AMBER if i == np.argmax(probs) else viz.ACCENT),
            Text(viz.percent(float(probs[i])), style=viz.FAINT),
        )
    yield hold(
        Group(
            strip(
                [
                    (f"logits[{w.labels[t]}]", row, READ),
                    ("e^(x − max)", e, None),
                    ("÷ total", probs, None),
                ],
                room,
                words,
            ),
            result(("the total: ", viz.FAINT), (f"{e.sum():.3f}", "bold")),
            Text(""),
            bars,
        ),
        0.5,
    )


# ── The walk ──────────────────────────────────────────────────────────────────


def run(stage: Stage, text: str = "The Fox and the", seed: int = 1) -> Walk:
    """Every step of reading `text`, animated."""
    model, words, count = toy(seed)
    w = walk(model, words, words.encode(text))
    c, p, room = model.config, model.params, stage.width
    big = Config(VOCAB, 0)  # the model the tour trains
    t, labels = len(w.ids), w.labels
    steps = 2 + 14 * c.layers + 3
    number_of = iter(range(1, steps + 1))

    def step(title: str, subtitle: str, *explanation: str) -> None:
        stage.header(next(number_of), steps, title, subtitle)
        for part in explanation:
            stage.say(part)

    def play(frames) -> None:
        stage.play(frames, start=None, then="continue", again="watch it again")

    step(
        "The little model",
        "your model's architecture, small enough to print",
        f"This is your model's architecture, shrunk until every number fits on the screen: "
        f"{c.width} numbers per token (yours has {big.width}), {c.layers} layers (yours has "
        f"{big.layers}), {c.heads} query heads sharing {c.kv_heads} key/value head, an MLP "
        f"{c.hidden} wide, and a vocabulary of {c.vocab} whole words from fable titles instead "
        f"of {big.vocab:,} tokens. It "
        f"has just trained for a moment on {count} titles like The Fox and the Crow. Everything "
        "else is your model's own code, run one operation at a time.",
        "Purple is what's being read; green is what's being written. Every other number sits "
        "on a color for its sign (blue below zero, amber above) and size. The sentence, as "
        "token ids:",
    )
    stage.show(
        Text.assemble(
            *(
                item
                for i, tok in zip(labels, w.ids, strict=True)
                for item in ((i, "bold"), (f" {tok}   ", viz.FAINT))
            )
        )
    )

    step(
        "Embedding",
        "X = E[tokens]",
        f"Each token id picks its row of the embedding table E ({c.vocab} × {c.width}). Stacked "
        f"up, the rows make X ({t} × {c.width}): the residual stream. Every layer will read X "
        "and add to it; nothing is ever replaced.",
    )
    play(lambda: lookup(w, room))

    stream = "X"
    for n, layer in enumerate(w.layers, 1):
        wts = w.model.layer(n - 1)
        where = f"Layer {n}"
        step(
            f"{where} · normalize",
            "X̂ = X ÷ rms(X) × g",
            "Before attention, each row is rescaled to a typical size of 1 (RMSNorm): divide it "
            "by the root of its mean square, then multiply each number by a learned gain.",
        )
        play(
            lambda layer=layer, wts=wts: normalize(
                layer.x, layer.a_inv, wts["attn.norm"], layer.a_in, (stream, "X̂"), labels, room
            )
        )

        a_in = Grid("X̂", layer.a_in, labels, dims(c.width), "the stream, normalized")
        hw = c.head_width
        step(
            f"{where} · queries",
            f"Q = X̂ · W_q    ({t} × {c.width}) · ({c.width} × {c.width})",
            "A matrix times a matrix is many dot products: each number of Q is a row of X̂ times "
            "a column of W_q, multiplied pair by pair and added up. Every position's row goes "
            f"through the same grid. Columns 1–{hw} of Q are head 1's query; {hw + 1}–{2 * hw}, "
            "head 2's.",
        )
        wq = Grid("W_q", wts["attn.query"], dims(c.width), dims(c.width), "learned")
        stage.show(draw(wq, room))
        q_all = Grid("Q", merge_heads(layer.q0[None])[0], labels, dims(c.heads * hw), "queries")
        play(lambda a_in=a_in, wq=wq, q_all=q_all: matmul(a_in, wq, q_all, room))

        step(
            f"{where} · keys and values",
            f"K = X̂ · W_k,  V = X̂ · W_v    ({c.width} × {hw} each)",
            f"The same, with W_k and W_v. They're only {hw} columns wide: there's one key/value "
            "head, and both query heads will use it. That's grouped-query attention: fewer keys "
            "and values to keep.",
        )
        wk = Grid("W_k", wts["attn.key"], dims(c.width), dims(hw), "learned")
        wv = Grid("W_v", wts["attn.value"], dims(c.width), dims(hw), "learned")
        both = Table.grid(padding=(0, 4))
        both.add_row(draw(wk, room // 2), draw(wv, room // 2))
        stage.show(both)
        k0 = Grid("K", layer.k0[0], labels, dims(hw), "keys")
        v0 = Grid("V", layer.v[0], labels, dims(hw), "values")
        play(lambda a_in=a_in, wk=wk, k0=k0: matmul(a_in, wk, k0, room))
        play(lambda a_in=a_in, wv=wv, v0=v0: matmul(a_in, wv, v0, room, quick=True))

        step(
            f"{where} · RoPE",
            "turn each pair of numbers by position × speed",
            "Queries and keys learn where each token sits by being turned: each pair of numbers "
            "is a point, and position m turns it by m × (the pair's speed). Pair 1 turns 1 "
            "radian a position; pair 2, 0.01. Values aren't turned.",
        )

        def rotations(layer=layer, hw=hw) -> Iterator[Frame]:
            for h in range(c.heads):
                yield from turning(
                    Grid(f"Q{h + 1}", layer.q0[h], labels, dims(hw), "before"),
                    Grid(f"Q{h + 1}", layer.q[h], labels, dims(hw), "after"),
                    w.turn,
                    room,
                )
            yield from turning(
                Grid("K", layer.k0[0], labels, dims(hw), "before"),
                Grid("K", layer.k[0], labels, dims(hw), "after"),
                w.turn,
                room,
            )

        play(rotations)

        step(
            f"{where} · scores",
            f"S = Q · Kᵀ ÷ √{hw}",
            "How well each position's question matches each position's answer: every query "
            f"row times every key row, ÷ √{hw} to keep the numbers calm. Head by head; both "
            "heads use the same keys.",
        )
        for h in range(c.heads):
            play(lambda layer=layer, h=h: attend(w, layer, h, room))

        step(
            f"{where} · mask and softmax",
            "A = softmax(S, with the future at −∞)",
            "A position can't look at tokens that come after it: those scores become −∞. Then "
            "each row becomes weights that add up to 1: e to the power of each score, divided "
            "by the row's total. (Subtracting the row's biggest score first changes nothing, "
            "and keeps e from overflowing.)",
        )
        for h in range(c.heads):
            play(lambda layer=layer, h=h: weigh(w, layer, h, room))

        step(
            f"{where} · mixing values",
            "O = A · V",
            "Each head's output: its weights times the values. A row of O is a weighted average "
            "of the value rows: mostly the ones its position attended to hardest.",
        )
        for h in range(c.heads):
            play(lambda layer=layer, h=h: mix(w, layer, h, room))
        play(lambda layer=layer: side_by_side(w, layer, room))

        step(
            f"{where} · output",
            f"attention = O · W_o    ({t} × {c.width}) · ({c.width} × {c.width})",
            "One more grid mixes the heads' outputs together, into what attention will add to "
            "the stream.",
        )
        wo = Grid("W_o", wts["attn.out"], dims(c.width), dims(c.width), "learned")
        stage.show(draw(wo, room))
        o = Grid("O", layer.mixed, labels, dims(c.width), "the heads' outputs")
        attn = Grid("attention", layer.attn, labels, dims(c.width), "what attention adds")
        play(lambda o=o, wo=wo, attn=attn: matmul(o, wo, attn, room))

        step(
            f"{where} · add",
            f"{stream} ← {stream} + attention",
            "The residual connection: attention's result is added onto the stream, number by "
            "number. The stream keeps everything it had.",
        )
        x = Grid(stream, layer.x, labels, dims(c.width), "the stream")
        mid = Grid(stream, layer.mid, labels, dims(c.width), "the stream, with attention added")
        play(lambda x=x, attn=attn, mid=mid: add(x, attn, mid, room))

        step(
            f"{where} · normalize again",
            "X̂ = X ÷ rms(X) × g",
            "The same kind of norm before the MLP, with its own gains.",
        )
        play(
            lambda layer=layer, wts=wts: normalize(
                layer.mid, layer.m_inv, wts["mlp.norm"], layer.m_in, (stream, "X̂"), labels, room
            )
        )

        m_in = Grid("X̂", layer.m_in, labels, dims(c.width), "the stream, normalized")
        step(
            f"{where} · gate and up",
            f"G = X̂ · W_gate,  U = X̂ · W_up    ({c.width} × {c.hidden} each)",
            f"The MLP widens each row from {c.width} numbers to {c.hidden}, twice. Each of the "
            f"{c.hidden} is a detector: how far the row points along one learned direction.",
        )
        wg = Grid("W_gate", wts["mlp.gate"], dims(c.width), dims(c.hidden), "learned")
        wu = Grid("W_up", wts["mlp.up"], dims(c.width), dims(c.hidden), "learned")
        stage.show(draw(wg, room))
        g = Grid("G", layer.gate, labels, dims(c.hidden), "gates")
        u = Grid("U", layer.up, labels, dims(c.hidden), "ups")
        play(lambda m_in=m_in, wg=wg, g=g: matmul(m_in, wg, g, room))
        stage.show(draw(wu, room))
        play(lambda m_in=m_in, wu=wu, u=u: matmul(m_in, wu, u, room, quick=True))

        step(
            f"{where} · switch",
            "H = SiLU(G) × U",
            "The switch: SiLU(g) = g × σ(g) shuts the gates that came out negative and opens "
            "the positive ones. Then each gate's opening multiplies its up: that's the gated "
            "part of SwiGLU.",
        )
        h_grid = Grid("H", layer.hidden, labels, dims(c.hidden), "switched and gated")
        play(lambda g=g, u=u, layer=layer, h_grid=h_grid: switch(g, u, layer.act, h_grid, room))

        step(
            f"{where} · down",
            f"MLP = H · W_down    ({t} × {c.hidden}) · ({c.hidden} × {c.width})",
            f"Back from {c.hidden} numbers to {c.width}. Each row of W_down is what one detector "
            "writes when it fires, so the result is those rows, each scaled by how open its "
            "gate was, added up.",
        )
        wd = Grid("W_down", wts["mlp.down"], dims(c.hidden), dims(c.width), "learned")
        stage.show(draw(wd, room))
        mlp = Grid("MLP", layer.mlp, labels, dims(c.width), "what the MLP adds")
        play(lambda h_grid=h_grid, wd=wd, mlp=mlp: matmul(h_grid, wd, mlp, room))

        step(
            f"{where} · add",
            f"{stream} ← {stream} + MLP",
            "Added onto the stream, and the layer is done."
            + (" The next layer starts from this." if n < len(w.layers) else ""),
        )
        mid_grid = Grid(stream, layer.mid, labels, dims(c.width), "the stream")
        out = Grid(stream, layer.out, labels, dims(c.width), f"the stream after layer {n}")
        play(lambda mid_grid=mid_grid, mlp=mlp, out=out: add(mid_grid, mlp, out, room))

    step("Final norm", "X̂ = X ÷ rms(X) × g", "One last norm, with its own gains.")
    last = w.layers[-1]
    play(lambda: normalize(last.out, w.f_inv, p["norm"], w.f_in, (stream, "X̂"), labels, room))

    words = [w.words.label(i) for i in range(c.vocab)]
    step(
        "Scores for every word",
        f"logits = X̂ · Eᵀ    ({t} × {c.width}) · ({c.width} × {c.vocab})",
        "Every row scores every word as the one that comes next, using the embedding table "
        "from the start again, on its side: a row's score for a word is how closely it points "
        "along that word's embedding.",
    )
    et = Grid("Eᵀ", p["embed"].T, dims(c.width), words, "the embedding table, turned")
    stage.show(draw(et, room))
    f_in = Grid("X̂", w.f_in, labels, dims(c.width), "the stream, normalized")
    logits = Grid("logits", w.logits, labels, words, "each row's score for each word")
    play(lambda: matmul(f_in, et, logits, room, quick=True))

    step(
        "The next word",
        "softmax(the last row)",
        f"Softmax turns the last row's scores into probabilities: its guess at the word after "
        f"{labels[-1]}. Every other row made its own guess too, for the word after it; in "
        "training, all of them are graded at once.",
    )
    play(lambda: guess(w, room))
    rows = Table.grid(padding=(0, 2))
    for i in range(t):
        best = int(np.argmax(w.probs[i]))
        rows.add_row(
            Text(f"after {labels[i]}", style=viz.FAINT),
            Text(words[best], style="bold"),
            Text(viz.percent(float(w.probs[i, best])), style=viz.FAINT),
        )
    stage.show(rows)
    same = "exactly" if w.error == 0 else f"to within {w.error:.0e}, rounding error"
    stage.say(
        "That's the whole forward pass: look up rows; in each layer, normalize, attend, add, "
        "normalize, widen, switch, narrow, add; normalize once more, and score every word. The "
        f"longhand logits match the model's own forward pass {same}. Your model does the very "
        f"same with {big.width} numbers per token, {big.layers} layers, "
        f"{big.heads} query heads sharing {big.kv_heads} key/value heads, an MLP {big.hidden} "
        f"wide, and {big.vocab:,} tokens."
    )
    return w
