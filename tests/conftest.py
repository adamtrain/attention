import io

import pytest
from rich.console import Console

from attention.tour.stage import THEME, Stage


@pytest.fixture(autouse=True)
def model_home(tmp_path, monkeypatch):
    """Keep every test's saved models out of the real data directory."""
    home = tmp_path / "home"
    monkeypatch.setenv("ATTENTION_HOME", str(home))
    return home


def recording(width: int = 100) -> Console:
    return Console(
        record=True,
        width=width,
        height=50,
        force_terminal=True,
        color_system="truecolor",
        file=io.StringIO(),
        theme=THEME,
    )


@pytest.fixture
def stage() -> Stage:
    """A stage that draws the final frame of every animation and never waits."""
    return Stage(recording(), animate=False)
