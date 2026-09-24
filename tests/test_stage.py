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


class Counting:
    """Frames that count how often they've been asked for."""

    def __init__(self) -> None:
        self.runs = 0

    def __call__(self):
        self.runs += 1
        return [Text(f"run {self.runs} frame {i}") for i in range(3)]


def test_an_animation_waits_on_its_first_frame_until_asked_to_start():
    keys = FakeKeys("space")
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(frames, fps=1000, start="watch it", then=None)
    assert keys.waits[0] is None  # the first read blocks: nothing moves until a key
    assert all(w is not None for w in keys.waits[1:])
    text = stage.console.export_text()
    assert "space to watch it" in text and "frame 2" in text


def test_animations_without_a_prompt_start_straight_away():
    keys = FakeKeys()
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(frames, fps=1000, start=None, then=None)
    assert None not in keys.waits


def test_q_at_the_start_prompt_quits():
    stage = Stage(recording(), keys=FakeKeys("q"))  # ty: ignore[invalid-argument-type]
    with pytest.raises(Quit):
        stage.play(frames, fps=1000, start="watch it")


def test_without_a_keyboard_nothing_waits():
    stage = Stage(recording(), animate=True)
    stage.play(frames, fps=1000, start="watch it")
    assert "space to" not in stage.console.export_text()


def test_an_animation_ends_by_offering_a_replay():
    made = Counting()
    stage = Stage(recording(), keys=FakeKeys("space", "space"))  # ty: ignore[invalid-argument-type]
    stage.play(made, fps=1000, start="watch it")
    assert made.runs == 1
    assert "r to replay" in stage.console.export_text()


def test_r_at_the_end_plays_it_again_from_new_frames():
    made = Counting()
    keys = FakeKeys("space", "r", "space")
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(made, fps=1000, start="watch it", again="write another")
    assert made.runs == 2
    text = stage.console.export_text()
    assert "r to write another" in text
    assert "run 2 frame 2" in text


def test_r_while_playing_starts_over():
    made = Counting()
    stage = Stage(recording(), keys=FakeKeys("r"))  # ty: ignore[invalid-argument-type]
    stage.play(made, fps=1000, start=None, then=None)
    assert made.runs == 2


def test_space_pauses_until_pressed_again():
    keys = FakeKeys("space", "x", "space")
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(frames, fps=1000, start=None, then=None)
    assert keys.waits[1:3] == [None, None]  # paused: waiting for as long as it takes
    assert "paused" in stage.console.export_text()


def test_skipping_goes_straight_to_the_last_frame():
    keys = FakeKeys("right")
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(lambda: (Text(f"frame {i}") for i in range(500)), fps=10, start=None, then=None)
    assert len(keys.waits) == 1  # it stopped listening, and stopped waiting
    assert "frame 499" in stage.console.export_text()


def test_right_at_the_start_prompt_skips_the_animation():
    keys = FakeKeys("right")
    stage = Stage(recording(), keys=keys)  # ty: ignore[invalid-argument-type]
    stage.play(lambda: (Text(f"frame {i}") for i in range(500)), fps=10, then=None)
    assert len(keys.waits) == 1


def test_the_controls_show_while_it_plays():
    stage = Stage(recording(), keys=FakeKeys())  # ty: ignore[invalid-argument-type]
    stage.play(frames, fps=1000, start=None, then=None)
    assert "space pause" in stage.console.export_text()
