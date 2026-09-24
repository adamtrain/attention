"""Pictures of the model, shared by the tour and the commands."""

from __future__ import annotations

import colorsys
import zlib
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from . import viz
from .corpus import BOUNDARY, Vocab
from .explain import Influence, Nudge
from .generate import Pick
from .model import Array

GOLDEN = 0.618033988749895


@lru_cache(maxsize=128)
def token_color(token: int) -> str:
    """Each token gets its own color, the same everywhere in the tour."""
    if token == 0:
        return "#8a8fa3"
    r, g, b = colorsys.hsv_to_rgb((0.62 + token * GOLDEN) % 1.0, 0.42, 0.93)
    return viz.hexcolor((round(r * 255), round(g * 255), round(b * 255)))


def token_chip(vocab: Vocab, token: int, width: int = 3) -> Text:
    return viz.chip(vocab.chars[token], token_color(token), width)


@lru_cache(maxsize=1024)
def piece_color(piece: str) -> str:
    """A color for a chunk of text several letters long."""
    r, g, b = colorsys.hsv_to_rgb((zlib.crc32(piece.encode()) % 997) / 997, 0.5, 0.95)
    return viz.hexcolor((round(r * 255), round(g * 255), round(b * 255)))


def piece_chips(pieces: Sequence[str], vocab: Vocab) -> Text:
    """Tokens that may be several letters long, as tiles. Single letters keep their colors."""
    text = Text(no_wrap=True)
    for i, piece in enumerate(pieces):
        if i:
            text.append(" ")
        single = len(piece) == 1 and piece in vocab.chars
        color = token_color(vocab.chars.index(piece)) if single else piece_color(piece)
        text.append_text(viz.chip(piece, color, len(piece) + 2))
    return text


def tiles(vocab: Vocab, ids: Sequence[int], width: int, ids_below: bool = True) -> list[Text]:
    """Tokens as colored tiles with their ids underneath, wrapped to fit."""
    per_row = max(1, width // 4)
    lines: list[Text] = []
    for start in range(0, len(ids), per_row):
        top, bottom = Text(no_wrap=True), Text(no_wrap=True)
        for token in ids[start : start + per_row]:
            top.append_text(token_chip(vocab, token))
            top.append(" ")
            bottom.append(f"{token:^3} ", style=viz.FAINT)
        lines.append(top)
        if ids_below:
            lines.append(bottom)
    return lines


def chips(vocab: Vocab, ids: Sequence[int], gap: str = "") -> Text:
    text = Text(no_wrap=True)
    for i, token in enumerate(ids):
        if i and gap:
            text.append(gap)
        text.append_text(token_chip(vocab, token))
    return text


def display(word: str) -> str:
    return word.capitalize()


def labels(vocab: Vocab, ids: Sequence[int] | Array) -> list[str]:
    return [vocab.chars[int(i)] for i in ids]


# ── Probabilities ─────────────────────────────────────────────────────────────


def prob_bars(
    probs: Array,
    vocab: Vocab,
    *,
    top: int = 8,
    width: int = 30,
    right: int | None = None,
    color: str = viz.ACCENT,
) -> Table:
    """The most likely next tokens, as bars. `right` marks the right answer, if there is one."""
    order = np.argsort(-probs)[:top]
    if right is not None and right not in order:
        order = np.append(order[:-1], right)
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    scale = max(float(probs.max()), 1e-9)
    for i in order:
        is_right = right is not None and int(i) == right
        tint = viz.GREEN if is_right else color
        grid.add_row(
            token_chip(vocab, int(i)),
            viz.bar(float(probs[i]) / scale, width, tint),
            Text(viz.percent(float(probs[i])), style="bold" if is_right else ""),
            Text("← right answer", style=viz.GREEN) if is_right else Text(""),
        )
    return grid


def spectrum(
    probs: Array,
    vocab: Vocab,
    width: int,
    roll: float | None = None,
    faded: Array | None = None,
) -> Group:
    """All the probabilities laid end to end, 0 to 1, with where a random roll landed.

    Tokens marked in `faded` are drawn washed out, as if about to be thrown away.
    """
    edges = np.concatenate([[0.0], np.cumsum(probs)])
    top = Text(no_wrap=True)
    band = Text(no_wrap=True)
    for cell in range(width):
        mid = (cell + 0.5) / width
        token = min(int(np.searchsorted(edges, mid, side="right")) - 1, len(probs) - 1)
        if faded is not None and faded[token]:
            band.append("░", style=viz.FAINT)
        else:
            band.append("█", style=token_color(token))
    # Label the tokens that are wide enough to hold one.
    marks = [" "] * width
    for token in range(len(probs)):
        a, b = edges[token] * width, edges[token + 1] * width
        if b - a >= 2 and not (faded is not None and faded[token]):
            marks[int((a + b) / 2)] = vocab.chars[token]
    under = Text("".join(marks), style=viz.FAINT, no_wrap=True)
    if roll is not None:
        pos = min(int(roll * width), width - 1)
        top.append(" " * pos)
        top.append("▼", style=f"bold {viz.AMBER}")
        top.append(f" {roll:.2f}", style=viz.AMBER)
        top.truncate(width + 6)
    else:
        top.append("")
    ruler = Text.assemble(("0", viz.FAINT), (" " * (width - 2)), ("1", viz.FAINT))
    return Group(top, band, under, ruler)


# ── Attention ─────────────────────────────────────────────────────────────────


def attention_grid(
    weights: Array,
    chars: Sequence[str],
    *,
    numbers: bool = True,
    highlight: int | None = None,
) -> list[Text]:
    """One head's attention: each row is a position, each column where it looked."""
    t = len(chars)
    hidden = ~np.tril(np.ones((t, t), dtype=bool))
    return viz.matrix_grid(
        weights,
        chars,
        chars,
        lambda v: viz.HEAT.color(v**0.6),
        fmt=".2f" if numbers else None,
        cell=4 if numbers else 2,
        hidden=hidden,
        highlight_row=highlight,
    )


def looks_at(weights: Array, chars: Sequence[str], width: int = 10) -> Table:
    """Where one position's attention went, as bars."""
    grid = Table.grid(padding=(0, 1))
    for _ in range(3):
        grid.add_column(no_wrap=True)
    for ch, w in zip(chars, weights, strict=False):
        grid.add_row(
            Text(f" {ch} ", style=f"bold {viz.ACCENT}" if ch != BOUNDARY else viz.FAINT),
            viz.bar(float(w), width, viz.ACCENT),
            Text(viz.percent(float(w)), style=viz.FAINT),
        )
    return grid


# ── Inference ─────────────────────────────────────────────────────────────────


def step_view(vocab: Vocab, pick: Pick, width: int, roll: float | None) -> Group:
    """One step of writing: what it saw, where it looked, what it expected, where it landed."""
    chars = list(pick.context)
    so_far = Text.assemble(("so far  ", viz.FAINT))
    for ch in chars:
        so_far.append_text(token_chip(vocab, vocab.chars.index(ch)))
    if roll is not None:
        so_far.append_text(token_chip(vocab, pick.token))
        so_far.append(" new", style=viz.AMBER)
    band = spectrum(pick.probs, vocab, min(60, width - 4), roll)
    verdict = Text("")
    if roll is not None:
        verdict = Text.assemble(
            ("the dart landed on ", viz.FAINT), token_chip(vocab, pick.token),
            ("  (the end of the word)" if pick.done else "", viz.FAINT),
        )  # fmt: skip
    return Group(
        so_far,
        Text(""),
        looking(vocab, pick),
        Text(""),
        Text("throw a dart", style="bold"),
        band,
        verdict,
    )


def looking(vocab: Vocab, pick: Pick, top: int = 5) -> Table:
    """Where the last position looked, per head, and what it expects next."""
    chars = list(pick.context)
    looked = Table.grid(padding=(0, 1))
    looked.add_column(style=viz.FAINT, no_wrap=True)
    looked.add_column(no_wrap=True)
    for h, weights in enumerate(pick.weights):
        row = Text(no_wrap=True)
        for ch, w in zip(chars, weights, strict=True):
            bg = viz.HEAT.color(float(w) ** 0.6)
            row.append(f" {ch} ", viz.style(viz.ink_for(bg), bg, bold=True))
        looked.add_row(f"head {h + 1}", row)

    grid = Table.grid(padding=(0, 4))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(
        Group(Text("where it looked", style="bold"), looked),
        Group(Text("next letter", style="bold"), prob_bars(pick.probs, vocab, top=top, width=16)),
    )
    return grid


# ── Backprop as a microscope ──────────────────────────────────────────────────


def influence_view(report: Influence) -> Table:
    """How strongly each input letter could sway a prediction, from backprop."""
    chars = "." + report.prefix
    top = max(float(report.sizes.max()), 1e-12)
    grid = Table.grid(padding=(0, 1))
    for _ in chars:
        grid.add_column(justify="center", no_wrap=True)
    cells, bars, pct = [], [], []
    for ch, size in zip(chars, report.sizes, strict=True):
        share = float(size) / top
        bg = viz.mix("#2a2d3a", viz.AMBER, share)
        cells.append(Text(f" {ch} ", style=viz.style(viz.ink_for(bg), bg, bold=True)))
        bars.append(Text(viz.SPARKS[min(7, int(share * 7.99))], style=viz.AMBER))
        pct.append(Text(f"{float(size) / float(report.sizes.sum()):.0%}", style=viz.FAINT))
    grid.add_row(*cells)
    grid.add_row(*bars)
    grid.add_row(*(Text(f"{p.plain:^3}", style=viz.FAINT) for p in pct))
    return grid


def nudge_view(vocab: Vocab, change: Nudge, top: int = 5) -> Table:
    """The likeliest next letters before and after one step of learning toward a target."""
    rows = {int(i) for i in np.argsort(-change.before)[:top]} | {change.target}
    rows |= {int(i) for i in np.argsort(-change.after)[:2]}
    grid = Table.grid(padding=(0, 1))
    for justify in ("left", "left", "right", "center", "left", "right", "left"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(
        "", Text("before", style="bold"), "", "", Text("after one step", style="bold"), "",
        Text(f"learning rate {change.lr:g}", style=viz.FAINT),
    )  # fmt: skip
    for i in sorted(rows, key=lambda i: -change.before[i]):
        taught = i == change.target
        tint = viz.GREEN if taught else viz.ACCENT
        grid.add_row(
            token_chip(vocab, i),
            viz.bar(float(change.before[i]), 14, tint),
            Text(viz.percent(float(change.before[i]))),
            Text("→", style=viz.FAINT),
            viz.bar(float(change.after[i]), 14, tint),
            Text(viz.percent(float(change.after[i])), style="bold" if taught else ""),
            Text("← taught" if taught else "", style=viz.GREEN),
        )
    return grid


# ── Commands ──────────────────────────────────────────────────────────────────


def word_row(word: str, known: set[str]) -> Text:
    new = word not in known
    return Text.assemble(
        ("✦ " if new else "· ", viz.GREEN if new else viz.FAINT),
        (display(word), "bold" if new else viz.FAINT),
        ("" if new else "  from the training data", viz.FAINT),
    )


def tally(words: list[str], known: set[str]) -> Text:
    new = sum(w not in known for w in words)
    return Text.assemble(
        ("✦ ", viz.GREEN), (f"{new} invented", viz.FAINT), ("   · ", viz.FAINT),
        (f"{len(words) - new} straight from the training data", viz.FAINT),
    )  # fmt: skip


def reading(vocab: Vocab, pick: Pick, name: str) -> Group:
    """The explain view's opening: one forward pass on a prefix, laid open."""
    title = Text.assemble(("◆ ", viz.PURPLE), (name, f"bold {viz.PURPLE}"), (" reads ", viz.FAINT),
                          (pick.context, f"bold {viz.ACCENT}"))  # fmt: skip
    return Group(title, Text(""), looking(vocab, pick, top=6))


def model_card(card, size: int, config, vocab: Vocab, words: int, path: str, kb: float) -> Table:
    """Everything about a saved model, as a small table."""
    from datetime import datetime

    when = datetime.fromisoformat(card.trained_at).astimezone()
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column()
    grid.add_row("made from", f"{words:,} {card.title.lower()} (a tenth held back)")
    grid.add_row(
        "trained",
        f"{when.day} {when:%b %Y} at {when:%H:%M} · {card.steps} steps · seed {card.seed}",
    )
    grid.add_row(
        "parameters", f"{size:,}  (1 layer, {config.heads} heads, {config.width} numbers per token)"
    )
    grid.add_row("vocabulary", Text.assemble(f"{config.vocab} tokens  ", (vocab.chars, viz.ACCENT)))
    grid.add_row("context", f"{config.context} characters")
    grid.add_row(
        "loss",
        Text.assemble(
            (f"{card.loss:.2f}", "bold"), (" on its training words, ", viz.FAINT),
            (f"{card.val_loss:.2f}", "bold"), (" on held-back ones", viz.FAINT),
        ),
    )  # fmt: skip
    grid.add_row(
        "", Text(f"(counting letter pairs alone scores {card.pair_loss:.2f})", style=viz.FAINT)
    )
    grid.add_row("file", f"{path}  ({kb:.0f} KB)")
    return grid
