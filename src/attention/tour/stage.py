"""Pacing for the tour: paragraphs, pictures, animations, and waiting for a keypress."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Iterator
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
type Frames = Callable[[], Iterable[Frame]]  # called again for every replay

PAUSE = ("space", "p")
SKIP = ("right", "enter", "s")
RESTART = ("r",)
GLIMPSE = 0.1  # while skipping to the end, show a frame this often (seconds)
AGAIN = 0.8  # on a replay, hold the first frame at least this long, to see it start over


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

    @property
    def patient(self) -> bool:
        """Is there someone to wait for (or an autopilot standing in for them)?"""
        return self.keys is not None or self.auto is not None

    def prompt(self, hint: str, again: str | None = None) -> Text:
        """The line that says which keys do what."""
        return Text.assemble(
            ("› ", viz.ACCENT), ("space", "bold"), (f" to {hint}  ·  ", viz.FAINT),
            *((("r", "bold"), (f" to {again}  ·  ", viz.FAINT)) if again else ()),
            ("q", "bold"), (" to quit", viz.FAINT),
        )  # fmt: skip

    def controls(self, paused: bool) -> Text:
        """The line under a playing animation."""
        state = ("❚❚ paused", f"bold {viz.AMBER}") if paused else ("▶ playing", viz.ACCENT)
        return Text.assemble(
            ("  ", ""), state, ("   ", ""),
            ("space", "bold"), (" resume  ·  " if paused else " pause  ·  ", viz.FAINT),
            ("→", "bold"), (" skip  ·  ", viz.FAINT), ("r", "bold"), (" restart  ·  ", viz.FAINT),
            ("q", "bold"), (" quit", viz.FAINT),
        )  # fmt: skip

    def _await(self) -> str | None:
        if self.keys:
            self.keys.drain()
        return self._key(self.auto / self.speed if self.auto is not None else None)

    def wait(self, hint: str = "continue") -> str | None:
        """Pause until the viewer presses a key (or, on autopilot, for a moment)."""
        if not self.patient:
            return None
        with Live(
            self.pad(self.prompt(hint)), console=self.console, auto_refresh=False, transient=True
        ):
            return self._await()

    def ready(
        self, live: Live, frame: RenderableType, hint: str, again: str | None = None
    ) -> str | None:
        """Show a frame with a prompt under it, and wait for the go-ahead. Returns the key.

        Before an animation, that lets the viewer finish reading and make sense of the picture
        before it moves. After one, it's the chance to watch it again.
        """
        key = None
        if self.patient:
            live.update(self.pad(Group(frame, Text(""), self.prompt(hint, again))), refresh=True)
            key = self._await()
        live.update(self.pad(frame), refresh=True)
        return key

    def sleep(self, seconds: float) -> bool:
        """Wait a moment. Returns True if a key cut it short."""
        if not self.animate:
            return False
        return self._key(seconds / self.speed) is not None

    def play(
        self,
        frames: Frames,
        fps: float = 12.0,
        start: str | None = "play it",
        then: str | None = "continue",
        again: str = "replay",
    ) -> None:
        """Show an animation. The last frame stays on screen.

        `frames()` makes the frames: a renderable (shown for 1/fps seconds) or (renderable,
        seconds). While it plays, space pauses, → skips to the end and r starts it over. With
        `start`, the opening frame waits for a key before the rest plays, and `start` is the
        prompt ("space to …"). With `then`, the last frame waits too, and offers to play it
        `again`. Every replay calls `frames()` again, so anything random comes out differently.
        """
        if not self.animate:
            last = None
            for frame in frames():
                last = frame[0] if isinstance(frame, tuple) else frame
            if last is not None:
                self.show(last)
            return
        with Live(console=self.console, auto_refresh=False, transient=False) as live:
            last, replay = self._run(live, frames(), fps, start)
            while replay or (
                last is not None and then and self.ready(live, last, then, again) in RESTART
            ):
                last, replay = self._run(live, frames(), fps, None, AGAIN)
            if last is not None:
                live.update(self.pad(last), refresh=True)
        self.console.print()

    def _run(
        self,
        live: Live,
        frames: Iterable[Frame],
        fps: float,
        start: str | None,
        first: float = 0.0,
    ) -> tuple[RenderableType | None, bool]:
        """Play frames once. Returns the last one, and whether the viewer asked to start over.

        The first frame stays up for at least `first` seconds.
        """
        last: RenderableType | None = None
        controls: bool | None = None  # decided on the first frame: only if they fit
        skipping = False
        glimpsed = due = time.monotonic()
        for i, frame in enumerate(frames):
            last, seconds = frame if isinstance(frame, tuple) else (frame, 1 / fps)
            if i == 0:
                seconds = max(seconds, first)
            if skipping:
                if time.monotonic() - glimpsed >= GLIMPSE:
                    live.update(self.pad(last), refresh=True)
                    glimpsed = time.monotonic()
                continue
            if controls is None:
                controls = self.keys is not None and self._fits(last, extra=2)
            if i == 0 and start:
                # They've had a good look at this one already. → goes straight to the end.
                skipping = self.ready(live, last, start) == "right"
                glimpsed = due = time.monotonic()
                continue
            self._draw(live, last, controls, paused=False)
            # Making this frame ate into the last one's time; if it took longer, don't catch up.
            due = max(due + seconds / self.speed, time.monotonic())
            action, due = self._hold(live, last, controls, due)
            if action == "restart":
                return last, True
            if action == "skip":
                skipping, glimpsed = True, time.monotonic()
        return last, False

    def _hold(
        self, live: Live, frame: RenderableType, controls: bool, due: float
    ) -> tuple[str | None, float]:
        """Keep a frame up until `due`, pausing if asked. Returns what to do next, and when."""
        while True:
            key = self._key(max(0.0, due - time.monotonic()))
            if key is None:
                return None, due
            if key in SKIP:
                return "skip", due
            if key in RESTART:
                return "restart", due
            if key not in PAUSE:
                continue
            paused = time.monotonic()
            self._draw(live, frame, controls, paused=True)
            while (key := self._key(None)) is not None and key not in PAUSE:
                if key in SKIP:
                    return "skip", due
                if key in RESTART:
                    return "restart", due
            due += time.monotonic() - paused  # the pause doesn't use up the frame's time
            self._draw(live, frame, controls, paused=False)

    def _draw(self, live: Live, frame: RenderableType, controls: bool, paused: bool) -> None:
        shown = Group(frame, Text(""), self.controls(paused)) if controls else frame
        live.update(self.pad(shown), refresh=True)

    def _fits(self, frame: RenderableType, extra: int) -> bool:
        """Is there room under this frame for `extra` more lines?"""
        lines = self.console.render_lines(self.pad(frame), pad=False)
        return len(lines) + extra < self.height

    @contextmanager
    def live(self) -> Iterator[Live]:
        """For hand-rolled interactive pictures: update it yourself, read `stage.key()`."""
        with Live(console=self.console, auto_refresh=False, transient=False) as live:
            yield live
        self.console.print()

    def key(self, timeout: float | None = None) -> str | None:
        """The next key the viewer presses (q still quits)."""
        return self._key(timeout)


def hold(frame: RenderableType, seconds: float) -> tuple[RenderableType, float]:
    return frame, seconds


def typing(text: str, style: str = "", per_char: float = 0.03) -> Iterator[Frame]:
    """Type a line out, one character at a time."""
    for i in range(1, len(text) + 1):
        yield Text(text[:i], style=style), per_char
