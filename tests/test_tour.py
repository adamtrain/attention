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


def test_dice_repeat_for_a_seed_but_not_for_a_replay():
    import numpy as np

    from attention.corpus import built_in
    from attention.tour.lab import Lab

    first, second = Lab.create(built_in("names"), 4), Lab.create(built_in("names"), 4)
    roll = first.dice(7).random()
    assert roll == second.dice(7).random() == np.random.default_rng([4, 7]).random()
    assert first.dice(7).random() != roll  # the replay


def test_reseeding_starts_over_with_a_fresh_model():
    from attention.corpus import built_in
    from attention.tour.lab import Lab

    lab = Lab.create(built_in("towns"), 4)
    lab.train_quietly()
    lab.dice(7)
    lab.reseed()
    assert lab.seed != 4
    assert not lab.trained and lab.trainer.step_number == 0
    assert lab.rolls == {}
    assert all((lab.initial.params[k] == lab.model.params[k]).all() for k in lab.model.params)


def test_the_dashboard_trains_to_the_end_a_frame_at_a_time():
    from attention import dashboard
    from attention.corpus import built_in
    from attention.tour.lab import Lab

    lab = Lab.create(built_in("dinosaurs"), 2, steps=60)
    watch = dashboard.Watch(lab.trainer, lab.baselines, lab.corpus, lab.seed, lab.rng)
    frames = list(dashboard.frames(watch, 96, seconds=2))
    assert lab.trained
    assert 1 < len(frames) <= 61  # at least one step per frame, after the untrained one
