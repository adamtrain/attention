"""Pacing for the tour: paragraphs, pictures, animations, and waiting for a keypress."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .. import viz
from ..keys import Keys

MAX_WIDTH = 100
PROSE_WIDTH = 78
MARGIN = 2

THEME = Theme(
    {
        "accent": viz.ACCENT,
        "faint": viz.FAINT,
        "green": viz.GREEN,
        "amber": viz.AMBER,
        "red": viz.RED,
        "purple": viz.PURPLE,
        "blue": viz.BLUE,
        "code": f"bold {viz.ACCENT}",
    }
)


class Quit(Exception):
    """The viewer pressed q."""


type Frame = RenderableType | tuple[RenderableType, float]


def markup(text: str) -> Text:
    """Rich markup, plus `backticks` for code."""
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    text = re.sub(r"`([^`]+)`", r"[code]\1[/code]", text)
    return Text.from_markup(text)


@dataclass
class Stage:
    console: Console
    keys: Keys | None = None
    animate: bool = True  # False: skip straight to the end of every animation
    auto: float | None = None  # advance on a timer instead of waiting for a key
    speed: float = 1.0

    @property
    def width(self) -> int:
        """How wide pictures can be, inside the margin."""
        return min(self.console.width, MAX_WIDTH) - MARGIN * 2

    @property
    def prose_width(self) -> int:
        return min(self.width, PROSE_WIDTH)

    @property
    def height(self) -> int:
        return self.console.height

    # ── Output ────────────────────────────────────────────────────────────────

    def pad(self, renderable: RenderableType) -> Padding:
        return Padding(renderable, (0, 0, 0, MARGIN))

    def show(self, *items: RenderableType | list[Text], gap: bool = True) -> None:
        for item in items:
            if isinstance(item, list):
                item = Group(*item)
            self.console.print(self.pad(item), width=self.width + MARGIN)
        if gap:
            self.console.print()

    def say(self, text: str, gap: bool = True) -> None:
        """A paragraph of prose."""
        self.console.print(self.pad(markup(text)), width=self.prose_width + MARGIN)
        if gap:
            self.console.print()

    def note(self, text: str, gap: bool = True) -> None:
        """A quieter aside."""
        body = markup(text)
        body.stylize(viz.FAINT)
        self.console.print(self.pad(body), width=self.prose_width + MARGIN)
        if gap:
            self.console.print()

    def clear(self) -> None:
        if self.console.is_terminal:
            self.console.clear()

    def header(self, number: int, total: int, title: str, subtitle: str) -> None:
        dots = Text(no_wrap=True)
        for i in range(1, total + 1):
            dots.append(
                "●" if i <= number else "○",
                style=viz.ACCENT if i <= number else viz.FAINT,
            )
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_row(
            Text.assemble((f"{number:02d}  ", f"bold {viz.ACCENT}"), (title, "bold")),
            dots,
        )
        grid.add_row(Text.assemble(("    ", ""), (subtitle, f"italic {viz.FAINT}")), "")
        self.console.print()
        self.show(grid, gap=False)
        self.show(Text("─" * self.width, style=viz.FAINT))

    # ── Time ──────────────────────────────────────────────────────────────────

    def _key(self, timeout: float | None) -> str | None:
        if self.keys is None:
            if timeout:
                time.sleep(timeout)
            return None
        key = self.keys.read(timeout)
        if key in ("q", "escape"):
            raise Quit
        return key

    def wait(self, hint: str = "continue") -> str | None:
        """Pause until the viewer presses a key (or, on autopilot, for a moment)."""
        if self.keys is None and self.auto is None:
            return None
        prompt = Text.assemble(
            ("  › ", viz.ACCENT), ("space", "bold"), (f" to {hint}  ·  ", viz.FAINT), ("q", "bold"),
            (" to quit", viz.FAINT),
        )  # fmt: skip
        with Live(prompt, console=self.console, auto_refresh=False, transient=True):
            if self.keys:
                self.keys.drain()
            return self._key(self.auto / self.speed if self.auto is not None else None)

    def sleep(self, seconds: float) -> bool:
        """Wait a moment. Returns True if a key cut it short."""
        if not self.animate:
            return False
        return self._key(seconds / self.speed) is not None

    def play(self, frames: Iterable[Frame], fps: float = 12.0) -> None:
        """Show an animation. Any key skips to the end; the last frame stays on screen.

        Frames can be a renderable (shown for 1/fps seconds) or (renderable, seconds).
        """
        last: RenderableType | None = None
        if not self.animate:
            for frame in frames:
                last = frame[0] if isinstance(frame, tuple) else frame
            if last is not None:
                self.show(last)
            return
        skipping = False
        with Live(console=self.console, auto_refresh=False, transient=False) as live:
            for frame in frames:
                last, seconds = frame if isinstance(frame, tuple) else (frame, 1 / fps)
                if skipping:
                    continue
                live.update(self.pad(last), refresh=True)
                skipping = self._key(seconds / self.speed) is not None
            if last is not None:
                live.update(self.pad(last), refresh=True)
        self.console.print()

    @contextmanager
    def live(self) -> Iterator[Live]:
        """For hand-rolled animations: update it yourself, poll `stage.pressed()`."""
        with Live(console=self.console, auto_refresh=False, transient=False) as live:
            yield live
        self.console.print()

    def key(self, timeout: float | None = None) -> str | None:
        """The next key the viewer presses (q still quits)."""
        return self._key(timeout)

    def pressed(self, within: float = 0.0) -> bool:
        """Wait up to `within` seconds for a key. Was one pressed?"""
        return self._key(within / self.speed) is not None


def hold(frame: RenderableType, seconds: float) -> tuple[RenderableType, float]:
    return frame, seconds


def typing(text: str, style: str = "", per_char: float = 0.03) -> Iterator[Frame]:
    """Type a line out, one character at a time."""
    for i in range(1, len(text) + 1):
        yield Text(text[:i], style=style), per_char
