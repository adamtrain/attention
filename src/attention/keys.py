"""Single keypresses from the terminal, without waiting for Enter."""

from __future__ import annotations

import os
import sys
import time
from types import TracebackType
from typing import Self

NAMES = {
    " ": "space",
    "\r": "enter",
    "\n": "enter",
    "\x1b": "escape",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1bOC": "right",
    "\x1bOD": "left",
    "\x7f": "backspace",
}


class Keys:
    """Read keys one at a time. Use as a context manager around everything that reads keys."""

    def __init__(self) -> None:
        self.saved = None
        self.fd = -1

    @staticmethod
    def available() -> bool:
        return sys.stdin.isatty() and sys.stdout.isatty()

    def __enter__(self) -> Self:
        if os.name != "nt":
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)  # keys arrive one at a time, unechoed; ctrl-c still works
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.saved is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read(self, timeout: float | None = None) -> str | None:
        """The next key, or None if none arrives within `timeout` seconds."""
        raw = _read_windows(timeout) if os.name == "nt" else self._read_posix(timeout)
        if raw is None:
            return None
        return NAMES.get(raw, raw.lower() if len(raw) == 1 else raw)

    def _read_posix(self, timeout: float | None) -> str | None:
        import select

        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self.fd, 1)
        if data == b"\x1b":  # maybe the start of an arrow key: collect the rest if it's there
            while select.select([self.fd], [], [], 0.02)[0]:
                data += os.read(self.fd, 1)
                if len(data) >= 3:
                    break
        return data.decode("utf-8", errors="replace")

    def drain(self) -> None:
        """Forget keys pressed while we weren't listening."""
        while self.read(0) is not None:
            pass


def _read_windows(timeout: float | None) -> str | None:
    import msvcrt

    end = None if timeout is None else time.monotonic() + timeout
    while not msvcrt.kbhit():  # ty: ignore[unresolved-attribute]
        if end is not None and time.monotonic() >= end:
            return None
        time.sleep(0.01)
    ch = msvcrt.getwch()  # ty: ignore[unresolved-attribute]
    if ch in ("\x00", "\xe0"):
        return {"M": "\x1b[C", "K": "\x1b[D", "H": "\x1b[A", "P": "\x1b[B"}.get(msvcrt.getwch(), "")  # ty: ignore[unresolved-attribute]
    return ch
