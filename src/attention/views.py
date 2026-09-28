"""Pictures of the model, shared by the tour, the dashboard and the commands."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from rich.console import Group
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import viz
from .explain import Influence, Nudge
from .generate import Pick
from .model import Array
from .tokenizer import SPECIAL, Tokenizer

GOLDEN = 0.618033988749895
SPECIAL_LABELS = {0: "‹end›", 1: "‹user›", 2: "‹assistant›"}
SHADES = ("#2b2e4a", "#3d4270")  # alternate token backgrounds in running text
COPIED = ("#5c2231", "#7a2d40")


# ── Tokens ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=2048)
def token_color(token: int) -> str:
    """Each token gets its own color, the same everywhere in the tour."""
    if token < len(SPECIAL):
        return "#8a8fa3"
    r, g, b = colorsys.hsv_to_rgb((0.62 + token * GOLDEN) % 1.0, 0.42, 0.93)
    return viz.hexcolor((round(r * 255), round(g * 255), round(b * 255)))


def visible(piece: str) -> str:
    """A token's text with its spaces and line breaks made visible."""
    return piece.replace(" ", "·").replace("\n", "↵")


def label(tokenizer: Tokenizer, token: int) -> str:
    return SPECIAL_LABELS.get(token) or visible(tokenizer.pieces[token])


def chip(tokenizer: Tokenizer, token: int, width: int | None = None) -> Text:
    """A token as a little colored tile."""
    text = label(tokenizer, token)
    return viz.chip(text, token_color(token), width or len(text) + 2)


def chips(tokenizer: Tokenizer, ids: Sequence[int], gap: str = "") -> Text:
    text = Text(no_wrap=True)
    for i, token in enumerate(ids):
        if i and gap:
            text.append(gap)
        text.append_text(chip(tokenizer, int(token)))
    return text


def tiles(tokenizer: Tokenizer, ids: Sequence[int], width: int, numbers: bool = True) -> list[Text]:
    """Tokens as tiles with their ids underneath, wrapped to fit."""
    lines: list[Text] = []
    top, bottom, used = Text(no_wrap=True), Text(no_wrap=True), 0
    for token in ids:
        tile = chip(tokenizer, int(token))
        cell = max(len(tile), len(str(token))) + 1
        if used and used + cell > width:
            lines += [top, bottom] if numbers else [top]
            top, bottom, used = Text(no_wrap=True), Text(no_wrap=True), 0
        top.append_text(tile)
        top.append(" " * (cell - len(tile)))
        bottom.append(f"{token:^{cell - 1}} ", style=viz.FAINT)
        used += cell
    lines += [top, bottom] if numbers else [top]
    return lines


def running(
    tokenizer: Tokenizer,
    ids: Sequence[int],
    width: int,
    copied: Sequence[bool] | Array | None = None,
    start: int = 0,
    newest: int | None = None,
) -> list[Text]:
    """Text as it reads, each token shaded so you can see where one ends and the next begins.

    Tokens marked in `copied` are shaded red: word for word from the training text. Tokens
    before `start` are shown plainly, and the token at `newest` in amber.
    """
    lines = [Text(no_wrap=True)]
    for i, token in enumerate(ids):
        token = int(token)
        if token < len(SPECIAL):
            continue
        shades = COPIED if copied is not None and copied[i] else SHADES
        if i == newest:
            style = viz.style(viz.DARK, viz.AMBER, bold=True)
        else:
            style = "" if i < start else viz.style("#f4f4ff", shades[i % 2])
        for j, part in enumerate(tokenizer.pieces[token].split("\n")):
            if j:
                lines.append(Text(no_wrap=True))
            if part:
                wrap(lines, part, style, width)
    return lines


def wrap(lines: list[Text], part: str, style, width: int) -> None:
    """Add a piece of text to the last line, carrying the word it's part of over if it won't fit."""
    line = lines[-1]
    if len(line) + len(part) <= width or not line.plain.strip():
        line.append(part, style)
        return
    if part.startswith(" "):
        lines.append(Text(part.lstrip(" "), style, no_wrap=True))
        return
    cut = line.plain.rfind(" ") + 1  # where the word this piece belongs to begins
    if cut <= 0 or len(line) - cut + len(part) > width:
        lines.append(Text(part, style, no_wrap=True))
        return
    lines[-1] = line[:cut]
    lines[-1].rstrip()
    lines.append(line[cut:])
    lines[-1].append(part, style)


def display(text: str, most: int = 60) -> str:
    """Text on one line, shortened if need be."""
    flat = " ".join(text.split())
    return flat if len(flat) <= most else flat[: most - 1].rstrip() + "…"


# ── Probabilities ─────────────────────────────────────────────────────────────


def prob_bars(
    probs: Array,
    tokenizer: Tokenizer,
    *,
    top: int = 8,
    width: int = 30,
    right: int | None = None,
    color: str = viz.ACCENT,
) -> Table:
    """The most likely next tokens, as bars. `right` marks the right answer, if there is one."""
    order = [int(i) for i in np.argsort(-probs, kind="stable")[:top]]
    if right is not None and right not in order:
        order = [*order[:-1], right]
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    scale = max(float(probs.max()), 1e-9)
    for i in order:
        is_right = right is not None and i == right
        tint = viz.GREEN if is_right else color
        grid.add_row(
            chip(tokenizer, i),
            viz.bar(float(probs[i]) / scale, width, tint),
            Text(viz.percent(float(probs[i])), style="bold" if is_right else ""),
            Text("← right answer", style=viz.GREEN) if is_right else Text(""),
        )
    return grid


def spectrum(
    probs: Array,
    tokenizer: Tokenizer,
    width: int,
    roll: float | None = None,
    faded: Array | None = None,
) -> Group:
    """All the probabilities laid end to end, 0 to 1, with where a random roll landed.

    Tokens marked in `faded` are drawn washed out, as if about to be thrown away.
    """
    edges = np.concatenate([[0.0], np.cumsum(probs)])
    band = Text(no_wrap=True)
    for cell in range(width):
        mid = (cell + 0.5) / width
        token = min(int(np.searchsorted(edges, mid, side="right")) - 1, len(probs) - 1)
        if faded is not None and faded[token]:
            band.append("░", style=viz.FAINT)
        else:
            band.append("█", style=token_color(token))
    # Label the tokens that are wide enough to hold their label.
    marks = [" "] * width
    for token in np.argsort(-probs)[:40]:
        a, b = edges[token] * width, edges[token + 1] * width
        name = label(tokenizer, int(token))
        if b - a >= len(name) + 1 and not (faded is not None and faded[token]):
            at = int((a + b) / 2 - len(name) / 2)
            if all(ch == " " for ch in marks[max(at - 1, 0) : at + len(name) + 1]):
                marks[at : at + len(name)] = list(name)
    under = Text("".join(marks)[:width], style=viz.FAINT, no_wrap=True)
    top = Text(no_wrap=True)
    if roll is not None:
        pos = min(int(roll * width), width - 1)
        top.append(" " * pos)
        top.append("▼", style=f"bold {viz.AMBER}")
        top.append(f" {roll:.2f}", style=viz.AMBER)
        top.truncate(width + 6)
    ruler = Text.assemble(("0", viz.FAINT), (" " * (width - 2)), ("1", viz.FAINT))
    return Group(top, band, under, ruler)


# ── Attention ─────────────────────────────────────────────────────────────────


def short(tokenizer: Tokenizer, token: int, most: int = 7) -> str:
    text = label(tokenizer, token)
    return text if len(text) <= most else text[: most - 1] + "…"


def attention_grid(
    weights: Array,
    names: Sequence[str],
    *,
    numbers: bool = True,
    highlight: int | None = None,
    cell: int | None = None,
) -> list[Text]:
    """One head's attention: each row is a position, each column where it looked."""
    t = len(names)
    hidden = ~np.tril(np.ones((t, t), dtype=bool))
    # Columns are the same tokens as the rows; only the grids with numbers have room to say so.
    cols = [n.lstrip("·")[:2] or n[:2] for n in names] if numbers else [""] * t
    return viz.matrix_grid(
        weights,
        names,
        cols,
        lambda v: viz.HEAT.color(v**0.6),
        fmt=".2f" if numbers else None,
        cell=cell or (4 if numbers else 2),
        hidden=hidden,
        highlight_row=highlight,
    )


def looks_at(weights: Array, names: Sequence[str], width: int = 10) -> Table:
    """Where one position's attention went, as bars."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, justify="right")
    for name, w in zip(names, weights, strict=False):
        grid.add_row(
            Text(name, style=f"bold {viz.ACCENT}"),
            viz.bar(float(w), width, viz.ACCENT),
            Text(viz.percent(float(w)), style=viz.FAINT),
        )
    return grid


def heat_tokens(tokenizer: Tokenizer, ids: Sequence[int], weights: Array) -> Text:
    """The text read so far, each token lit up by how much attention it got."""
    row = Text(no_wrap=True)
    for token, w in zip(ids, weights, strict=True):
        row.append(heat_piece(tokenizer, token), heat_style(w))
    return row


def heat_lines(tokenizer: Tokenizer, ids: Sequence[int], weights: Array, width: int) -> list[Text]:
    """heat_tokens, wrapped to fit: lines break between words, the way running text does."""
    lines = [Text(no_wrap=True)]
    for token, w in zip(ids, weights, strict=True):
        wrap(lines, heat_piece(tokenizer, token), heat_style(w), width)
    return lines


def heat_piece(tokenizer: Tokenizer, token: int) -> str:
    return SPECIAL_LABELS.get(int(token)) or tokenizer.pieces[int(token)].replace("\n", "↵")


def heat_style(weight: float) -> Style:
    bg = viz.HEAT.color(float(weight) ** 0.6)
    return viz.style(viz.ink_for(bg), bg, bold=True)


# ── Writing ───────────────────────────────────────────────────────────────────


def looking(tokenizer: Tokenizer, pick: Pick, width: int, top: int = 5, recent: int = 12) -> Table:
    """Where the newest position looked in each layer (its heads averaged), and what comes next."""
    ids = pick.context[-recent:]
    while len(ids) > 1 and sum(len(heat_piece(tokenizer, t)) for t in ids) > max(10, width - 44):
        ids = ids[1:]  # keep the newest tokens: those are the ones it looks at most
    looked = Table.grid(padding=(0, 1))
    looked.add_column(style=viz.FAINT, no_wrap=True)
    looked.add_column(no_wrap=True)
    for layer, weights in enumerate(pick.weights):
        shown = weights.mean(axis=0)[-len(ids) :]
        row = heat_tokens(tokenizer, ids, shown / max(float(shown.max()), 1e-9))
        looked.add_row(f"layer {layer + 1}", row)
    grid = Table.grid(padding=(0, 3))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(
        Group(Text("where it looked", style="bold"), looked),
        Group(
            Text("next token", style="bold"), prob_bars(pick.probs, tokenizer, top=top, width=12)
        ),
    )
    return grid


def step_view(tokenizer: Tokenizer, pick: Pick, width: int, roll: float | None) -> Group:
    """One step of writing: what it has written, where it looked, and where the dart landed."""
    landed = roll is not None and not pick.done
    ids = [*pick.context, pick.token] if landed else list(pick.context)
    so_far = running(tokenizer, ids, width - 2, newest=len(ids) - 1 if landed else None)
    band = spectrum(pick.probs, tokenizer, min(60, width - 4), roll)
    verdict = Text("")
    if roll is not None:
        verdict = Text.assemble(
            ("the dart landed on ", viz.FAINT), chip(tokenizer, pick.token),
            ("  (the end of the document)" if pick.done else "", viz.FAINT),
        )  # fmt: skip
    return Group(
        *so_far[-6:],
        Text(""),
        looking(tokenizer, pick, width),
        Text(""),
        Text("throw a dart", style="bold"),
        band,
        verdict,
    )


# ── Backprop as a microscope ──────────────────────────────────────────────────


def influence_view(tokenizer: Tokenizer, report: Influence, recent: int = 14) -> Table:
    """How strongly each token read could sway a prediction, from backprop."""
    ids = report.ids[-recent:]
    sizes = report.sizes[-recent:]
    top = max(float(sizes.max()), 1e-12)
    total = float(report.sizes.sum())
    grid = Table.grid(padding=(0, 1))
    for _ in ids:
        grid.add_column(justify="center", no_wrap=True)
    cells, bars, pct = [], [], []
    for token, size in zip(ids, sizes, strict=True):
        share = float(size) / top
        bg = viz.mix("#2a2d3a", viz.AMBER, share)
        cells.append(
            Text(label(tokenizer, int(token)), style=viz.style(viz.ink_for(bg), bg, bold=True))
        )
        bars.append(Text(viz.SPARKS[min(7, int(share * 7.99))], style=viz.AMBER))
        pct.append(Text(f"{float(size) / total:.0%}", style=viz.FAINT))
    grid.add_row(*cells)
    grid.add_row(*bars)
    grid.add_row(*pct)
    return grid


def nudge_view(tokenizer: Tokenizer, change: Nudge, top: int = 5) -> Table:
    """The likeliest next tokens before and after one step of learning toward a target."""
    rows = {int(i) for i in np.argsort(-change.before)[:top]} | {change.target}
    rows |= {int(i) for i in np.argsort(-change.after)[:2]}
    grid = Table.grid(padding=(0, 1))
    for justify in ("right", "left", "right", "center", "left", "right", "left"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(
        "", Text("before", style="bold"), "", "", Text("after one step", style="bold"), "",
        Text(f"learning rate {change.lr:g}", style=viz.FAINT),
    )  # fmt: skip
    for i in sorted(rows, key=lambda i: -change.before[i]):
        taught = i == change.target
        tint = viz.GREEN if taught else viz.ACCENT
        grid.add_row(
            chip(tokenizer, i),
            viz.bar(float(change.before[i]), 14, tint),
            Text(viz.percent(float(change.before[i]))),
            Text("→", style=viz.FAINT),
            viz.bar(float(change.after[i]), 14, tint),
            Text(viz.percent(float(change.after[i])), style="bold" if taught else ""),
            Text("← taught" if taught else "", style=viz.GREEN),
        )
    return grid


# ── Commands ──────────────────────────────────────────────────────────────────


def copied_note(copied: Array) -> Text:
    n = int(copied.sum())
    if not n:
        return Text("✦ all new: no run of it is in the training text", style=viz.GREEN)
    return Text.assemble(
        ("■ ", viz.RED),
        (f"{n} of {len(copied)} tokens are part of a run copied word for word", viz.FAINT),
    )
