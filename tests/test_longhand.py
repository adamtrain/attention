import numpy as np
import pytest

from attention import longhand

from .conftest import recording
from .test_tour import Measuring, loses_ink


@pytest.fixture(scope="module")
def little():
    return longhand.toy(1)


def test_the_little_model_learns_its_titles(little):
    model, words, _ = little
    w = longhand.walk(model, words, words.encode("The Fox and the"))
    best = [words.label(int(i)) for i in w.probs.argmax(axis=1)]
    assert best[2:4] == ["·and", "·the"]
    partners = {second for first, second in longhand.titles() if first == "Fox"}
    assert best[4].removeprefix("·") in partners


@pytest.mark.parametrize("text", ["The Fox and the", "the wolf and the lion", "Dog", ""])
def test_the_longhand_is_the_model_itself(little, text):
    model, words, _ = little
    w = longhand.walk(model, words, words.encode(text))
    assert w.error < 1e-12
    np.testing.assert_allclose(w.probs.sum(axis=1), 1.0)


def test_words_are_whole_tokens(little):
    _, words, _ = little
    assert words.encode("the fox AND The") == words.encode("The Fox and the")
    assert [words.label(i) for i in words.encode("The Fox")] == ["‹end›", "The", "·Fox"]
    with pytest.raises(ValueError, match="doesn't know “Unicorn”"):
        words.encode("The Fox and the Unicorn")
    with pytest.raises(ValueError, match="at most"):
        words.encode("The Fox and the Crow and the Dog")


class EveryFrame(Measuring):
    """A stage that checks every frame of every animation, not only the last."""

    def play(self, frames, *args, **kwargs) -> None:
        for frame in frames():
            picture = frame[0] if isinstance(frame, tuple) else frame
            if loses_ink(self.console, picture, self.width):
                self.cut.append(f"{self.chapter}: a frame")
        super().play(frames, *args, **kwargs)


@pytest.mark.parametrize("width", [80, 100])
def test_no_frame_is_cut_off_at_the_edge_of_the_terminal(width):
    stage = EveryFrame(recording(width), animate=False)
    longhand.run(stage, "The Wolf and the")
    assert stage.cut == []
    text = stage.console.export_text()
    for title in ("Embedding", "Layer 1 · RoPE", "Layer 2 · switch", "The next word"):
        assert title in text
