"""Drawing in the terminal: colors, heatmaps, bars, plots and a little canvas for diagrams."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from itertools import pairwise

import numpy as np
from rich.style import Style
from rich.text import Text

ACCENT = "#7c83f7"
GREEN = "#1fbf8f"
AMBER = "#eb9a12"
RED = "#f2506e"
PURPLE = "#a871f7"
BLUE = "#4aa8e0"
FAINT = "grey42"
DARK = "#111111"

type RGB = tuple[int, int, int]


def rgb(color: str) -> RGB:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def hexcolor(c: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)


def mix(a: str, b: str, t: float) -> str:
    """The color `t` of the way from a to b."""
    (r1, g1, b1), (r2, g2, b2) = rgb(a), rgb(b)
    return hexcolor(
        (round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t))
    )


class Ramp:
    """A color scale. Call it with a number from 0 to 1 to get a color."""

    LEVELS = 48

    def __init__(self, *stops: str):
        self.stops = stops

    def color(self, t: float) -> str:
        t = min(max(t, 0.0), 1.0)
        return self._color(round(t * self.LEVELS))

    @lru_cache(maxsize=256)  # noqa: B019 (ramps live for the whole program)
    def _color(self, level: int) -> str:
        pos = level / self.LEVELS * (len(self.stops) - 1)
        i = min(int(pos), len(self.stops) - 2)
        return mix(self.stops[i], self.stops[i + 1], pos - i)

    def diverging(self, value: float, scale: float) -> str:
        """Color for a value from -scale to +scale, with 0 in the middle."""
        return self.color(0.5 + 0.5 * value / scale if scale else 0.5)


# Weights and activations: blue below zero, amber above, dark near zero.
SIGNED = Ramp("#7fd0ff", BLUE, "#1f4a6b", "#2a2d3a", "#6b4a12", AMBER, "#ffd27f")
# Nudges: red where a weight went down, green where it went up.
NUDGE = Ramp("#ff9aae", RED, "#6b2533", "#2a2d3a", "#1d5a47", GREEN, "#8ff0cf")
# Probabilities and attention: dark for nothing, bright for everything.
HEAT = Ramp("#1c1f2b", "#34387a", ACCENT, "#b4b8ff", "#f4f4ff")
TITLE = Ramp(ACCENT, PURPLE, RED, AMBER)


@lru_cache(maxsize=4096)
def style(fg: str | None = None, bg: str | None = None, bold: bool = False) -> Style:
    return Style(color=fg, bgcolor=bg, bold=bold)


def ink_for(bg: str) -> str:
    """Black or white text, whichever reads better on this background."""
    r, g, b = rgb(bg)
    return DARK if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#f4f4ff"


def scale_of(values: np.ndarray, floor: float = 1e-9) -> float:
    """A robust "biggest value" for color scales, so one outlier doesn't wash out the rest."""
    a = np.abs(values[np.isfinite(values)])
    return max(float(np.percentile(a, 98)) if a.size else 0.0, floor)


# ── Heatmaps ──────────────────────────────────────────────────────────────────


def cells(values: Iterable[float], colors: Iterable[str], width: int = 2) -> Text:
    """A row of colored blocks."""
    text = Text(no_wrap=True)
    for _, color in zip(values, colors, strict=False):
        text.append(" " * width, style(bg=color))
    return text


def signed_cells(values: np.ndarray, scale: float, width: int = 2) -> Text:
    return cells(values, (SIGNED.diverging(v, scale) for v in values), width)


def blocks(matrix: np.ndarray, color, width: int = 1) -> list[Text]:
    """A dense heatmap, two rows of the matrix per line of text (using ▀ half blocks).

    `color` maps a value to a color.
    """
    lines = []
    rows = len(matrix)
    for top in range(0, rows, 2):
        line = Text(no_wrap=True)
        for c in range(matrix.shape[1]):
            upper = color(matrix[top, c])
            lower = color(matrix[top + 1, c]) if top + 1 < rows else None
            line.append("▀" * width, style(upper, lower))
        lines.append(line)
    return lines


def signed_blocks(matrix: np.ndarray, scale: float | None = None, width: int = 1) -> list[Text]:
    s = scale or scale_of(matrix)
    return blocks(matrix, lambda v: SIGNED.diverging(v, s), width)


def nudge_blocks(matrix: np.ndarray, scale: float | None = None, width: int = 1) -> list[Text]:
    s = scale or scale_of(matrix)
    return blocks(matrix, lambda v: NUDGE.diverging(v, s), width)


def matrix_grid(
    matrix: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str],
    color,
    *,
    fmt: str | None = None,
    cell: int = 3,
    hidden: np.ndarray | None = None,
    highlight_row: int | None = None,
) -> list[Text]:
    """A labeled heatmap, one cell per number, optionally with the numbers printed in it.

    Cells where `hidden` is True show a faint dot instead (for the causal mask).
    """
    label_w = max((len(r) for r in rows), default=0)
    lines = []
    for r, label in enumerate(rows):
        line = Text(no_wrap=True)
        line.append(
            f"{label:>{label_w}} ",
            style=f"bold {ACCENT}" if r == highlight_row else FAINT,
        )
        for c in range(len(cols)):
            if hidden is not None and hidden[r, c]:
                line.append(f"{'·':^{cell}}", style=FAINT)
                continue
            value = float(matrix[r, c])
            bg = color(value)
            body = format(value, fmt).replace("0.", ".", 1) if fmt else ""
            line.append(f"{body:^{cell}}", style(ink_for(bg), bg))
        lines.append(line)
    header = Text(" " * (label_w + 1), no_wrap=True)
    for c in cols:
        header.append(f"{c:^{cell}}", style=FAINT)
    lines.append(header)
    return lines


def legend(ramp: Ramp, left: str, right: str, middle: str | None = None, cells_n: int = 7) -> Text:
    text = Text(no_wrap=True)
    text.append(f"{left} ", style=FAINT)
    for i in range(cells_n):
        text.append("  ", style(bg=ramp.color(i / (cells_n - 1))))
    text.append(f" {right}", style=FAINT)
    if middle:
        text.append(f"   (middle: {middle})", style=FAINT)
    return text


# ── Bars ──────────────────────────────────────────────────────────────────────


def bar(fraction: float, width: int, color: str, track: bool = True) -> Text:
    """━━━━━╸──── with half-cell precision."""
    halves = round(min(max(fraction, 0.0), 1.0) * width * 2)
    full, half = divmod(halves, 2)
    text = Text("━" * full, style=color, no_wrap=True)
    if half:
        text.append("╸", style=color)
    rest = width - full - half
    if rest > 0:
        text.append(("─" if track else " ") * rest, style=FAINT if track else "")
    return text


def signed_bar(value: float, scale: float, half: int, neg: str = RED, pos: str = GREEN) -> Text:
    """A bar growing left (negative) or right (positive) from a center line."""
    n = round(min(abs(value) / scale, 1.0) * half) if scale else 0
    text = Text(no_wrap=True)
    if value < 0:
        text.append(" " * (half - n))
        text.append("━" * n, style=neg)
    else:
        text.append(" " * half)
    text.append("┃", style=FAINT)
    if value > 0:
        text.append("━" * n, style=pos)
        text.append(" " * (half - n))
    else:
        text.append(" " * half)
    return text


SPARKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], lo: float | None = None, hi: float | None = None) -> str:
    if not values:
        return ""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = (hi - lo) or 1.0
    return "".join(SPARKS[min(int((v - lo) / span * 8), 7)] for v in values)


def percent(p: float) -> str:
    if p >= 0.995:
        return "100%"
    if p >= 0.1:
        return f"{p:.0%}"
    if p >= 0.0005:
        return f"{p * 100:.1f}%"
    return "<0.1%"


# ── Braille plots ─────────────────────────────────────────────────────────────

BRAILLE = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


class Plot:
    """A line chart drawn with braille dots: each character cell holds 2×4 of them."""

    def __init__(self, width: int, height: int, x_max: float, lo: float, hi: float):
        self.width, self.height = width, height
        self.x_max, self.lo, self.hi = max(x_max, 1e-9), lo, hi
        self.dots = np.zeros((height * 4, width * 2), dtype=bool)
        self.colors: list[list[str | None]] = [[None] * width for _ in range(height)]
        self.guides: list[tuple[int, str, str]] = []
        self.marks: list[tuple[int, int, str, str]] = []

    def _xy(self, x: float, y: float) -> tuple[int, int]:
        px = round(x / self.x_max * (self.width * 2 - 1))
        frac = (y - self.lo) / (self.hi - self.lo)
        py = round((1 - frac) * (self.height * 4 - 1))
        return px, py

    def _dot(self, px: int, py: int, color: str) -> None:
        if 0 <= px < self.width * 2 and 0 <= py < self.height * 4:
            self.dots[py, px] = True
            self.colors[py // 4][px // 2] = color

    def line(self, points: Sequence[tuple[float, float]], color: str) -> None:
        pts = [self._xy(x, y) for x, y in points]
        if len(pts) == 1:
            self._dot(*pts[0], color)
        for (x0, y0), (x1, y1) in pairwise(pts):
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            for i in range(n + 1):
                self._dot(round(x0 + (x1 - x0) * i / n), round(y0 + (y1 - y0) * i / n), color)

    def guide(self, y: float, color: str, label: str = "") -> None:
        """A dashed horizontal line, for comparison."""
        row = self._xy(0, y)[1] // 4
        if 0 <= row < self.height:
            self.guides.append((row, color, label))

    def mark(self, x: float, y: float, char: str, color: str) -> None:
        px, py = self._xy(x, y)
        self.marks.append((py // 4, px // 2, char, color))

    def render(self, labels: bool = True, fmt: str = "{:.1f}") -> list[Text]:
        top, bottom = fmt.format(self.hi), fmt.format(self.lo)
        pad = max(len(top), len(bottom)) if labels else 0
        guide_rows = {row: (color, label) for row, color, label in self.guides}
        lines = []
        for row in range(self.height):
            line = Text(no_wrap=True)
            if labels:
                tag = top if row == 0 else bottom if row == self.height - 1 else ""
                line.append(f"{tag:>{pad}} ", style=FAINT)
                line.append("┤" if row in (0, self.height - 1) else "│", style=FAINT)
            guide = guide_rows.get(row)
            chars: list[tuple[str, str | None]] = []
            for col in range(self.width):
                block = self.dots[row * 4 : row * 4 + 4, col * 2 : col * 2 + 2]
                bits = sum(BRAILLE[r][c] for r in range(4) for c in range(2) if block[r, c])
                if bits:
                    chars.append((chr(0x2800 + bits), self.colors[row][col]))
                elif guide:
                    chars.append(("╌", guide[0]))
                else:
                    chars.append((" ", None))
            for r, c, char, color in self.marks:
                if r == row and 0 <= c < self.width:
                    chars[c] = (char, color)
            if guide and guide[1]:
                label = f" {guide[1]}"
                start = self.width - len(label)
                if start > 0:
                    for i, ch in enumerate(label):
                        if chars[start + i][0] in (" ", "╌"):
                            chars[start + i] = (ch, guide[0])
            for ch, color in chars:
                line.append(ch, style=color or "")
            lines.append(line)
        return lines


# ── A canvas for diagrams ─────────────────────────────────────────────────────


class Canvas:
    """A grid of characters you can write anywhere on, then turn into Text."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.chars = [[" "] * width for _ in range(height)]
        self.styles: list[list[Style | str]] = [[""] * width for _ in range(height)]

    def put(self, x: int, y: int, text: str | Text, style: Style | str = "") -> None:
        if not 0 <= y < self.height:
            return
        if isinstance(text, Text):
            for i, ch in enumerate(text.plain):
                spans = [s.style for s in text.spans if s.start <= i < s.end]
                self.put(x + i, y, ch, spans[-1] if spans else text.style)
            return
        for i, ch in enumerate(text):
            if 0 <= x + i < self.width:
                self.chars[y][x + i] = ch
                self.styles[y][x + i] = style

    def restyle(self, x: int, y: int, width: int, style: Style | str) -> None:
        for i in range(width):
            if 0 <= y < self.height and 0 <= x + i < self.width:
                self.styles[y][x + i] = style

    def lines(self) -> list[Text]:
        out = []
        for chars, styles in zip(self.chars, self.styles, strict=True):
            line = Text(no_wrap=True)
            for ch, st in zip(chars, styles, strict=True):
                line.append(ch, style=st)
            line.rstrip()
            out.append(line)
        return out

    def text(self) -> Text:
        return Text("\n", no_wrap=True).join(self.lines())


# ── Bits and pieces ───────────────────────────────────────────────────────────


FONT = {
    "a": ("▄▀█", "█▀█"),
    "t": ("▀█▀", " █ "),
    "e": ("█▀▀", "██▄"),
    "n": ("█▄ █", "█ ▀█"),
    "i": ("█", "█"),
    "o": ("█▀█", "█▄█"),
}


def wordmark(word: str = "attention", shift: float = 0.0) -> Text:
    """The name in big letters, shaded left to right."""
    rows = [" ".join(FONT[ch][r] for ch in word) for r in range(2)]
    width = len(rows[0])
    text = Text(no_wrap=True)
    for r, row in enumerate(rows):
        for i, ch in enumerate(row):
            t = (i / max(width - 1, 1) + shift) % 1.0 if shift else i / max(width - 1, 1)
            text.append(ch, style(TITLE.color(t), bold=True))
        if r == 0:
            text.append("\n")
    return text


def chip(label: str, color: str, width: int = 3) -> Text:
    """A token: a little colored tile."""
    return Text(f"{label:^{width}}", style=style(ink_for(color), color, bold=True))


def show_char(ch: str) -> str:
    """How to display a token, so the word boundary stands out."""
    return "·" if ch == " " else ch


def plural(n: int, word: str, words: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (words or word + 's')}"
