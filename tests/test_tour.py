import pytest

from attention import tour
from attention.store import default_path, load


@pytest.mark.parametrize("corpus", ["dinosaurs", "names", "towns"])
def test_the_whole_tour_plays_through(stage, corpus):
    outcome = tour.run(stage, corpus=corpus, seed=12)
    assert outcome.finished and outcome.lab is not None
    text = stage.console.export_text()
    for title in (
        "Tokens",
        "Attention",
        "Backpropagation",
        "Pretraining",
        "Inference",
        "Your model",
    ):
        assert title in text
    assert "It named itself" in text
    saved = load(default_path())
    assert saved.card.name == outcome.lab.name
    assert saved.card.corpus == corpus


def test_starting_late_trains_a_model_first(stage):
    start = next(i for i, c in enumerate(tour.CHAPTERS, 1) if c.title == "Inference")
    outcome = tour.run(stage, corpus="names", seed=3, start=start)
    assert outcome.finished
    assert outcome.lab is not None and outcome.lab.trained
    text = stage.console.export_text()
    assert "Inference" in text and "Tokens" not in text


def test_narrow_terminals_still_work():
    from rich.console import Console

    from attention.tour.stage import THEME, Stage

    console = Console(record=True, width=80, height=30, force_terminal=True, theme=THEME)
    narrow = Stage(console, animate=False)
    assert tour.run(narrow, corpus="dinosaurs", seed=1).finished
