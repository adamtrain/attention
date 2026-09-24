"""Showing the model's own source code in the tour."""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable

from rich.syntax import Syntax


def excerpt(func: Callable, start: str | None = None, end: str | None = None) -> Syntax:
    """The lines of a function from the one containing `start` up to (not including) `end`."""
    lines = inspect.getsource(func).splitlines()
    first = next((i for i, line in enumerate(lines) if start and start in line), 0)
    last = next((i for i, line in enumerate(lines) if end and end in line and i > first), None)
    chunk = textwrap.dedent("\n".join(lines[first:last])).strip("\n")
    return Syntax(chunk, "python", theme="ansi_dark", background_color="default", word_wrap=False)
