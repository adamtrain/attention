import pytest
from rich.text import Text

from attention.tour.stage import Quit, Stage

from .conftest import recording


class FakeKeys:
    """Keys pressed in advance, and a record of how long each read was willing to wait."""

    def __init__(self, *keys: str):
        self.keys = list(keys)
        self.waits: list[float | None] = []

    def read(self, timeout: float | None = None) -> str | None:
        self.waits.append(timeout)
        return self.keys.pop(0) if self.keys else None

    def drain(self) -> None:
        pass


def frames():
    return [Text(f"frame {i}") for i in range(3)]


def test_an_animation_waits_on_its_first_frame_until_asked_to_start():
    keys = FakeKeys("space")
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(frames(), fps=1000, start="watch it")
    assert keys.waits[0] is None  # the first read blocks: nothing moves until a key
    assert all(w is not None for w in keys.waits[1:])
    text = stage.console.export_text()
    assert "space to watch it" in text and "frame 2" in text


def test_animations_without_a_prompt_start_straight_away():
    keys = FakeKeys()
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(frames(), fps=1000, start=None)
    assert None not in keys.waits


def test_q_at_the_start_prompt_quits():
    stage = Stage(recording(), keys=FakeKeys("q"))  # ty: ignore[invalid-argument-type]
    with pytest.raises(Quit):
        stage.play(frames(), fps=1000, start="watch it")


def test_without_a_keyboard_nothing_waits():
    stage = Stage(recording(), animate=True)
    stage.play(frames(), fps=1000, start="watch it")
    assert "space to" not in stage.console.export_text()
