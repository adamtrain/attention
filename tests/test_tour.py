from dataclasses import dataclass, field

import numpy as np
import pytest
from rich.console import Console, Group, RenderableType
from rich.text import Text

from attention import dashboard, tour
from attention.corpus import built_in
from attention.store import chat_path, default_path, load
from attention.tour.lab import Budget, Lab
from attention.tour.stage import Stage

from .conftest import recording


@pytest.mark.parametrize("corpus", ["fables", "fairytales", "shakespeare"])
def test_the_whole_tour_plays_through(stage, corpus):
    outcome = tour.run(stage, corpus=corpus, seed=12, budget=Budget.tiny())
    assert outcome.finished and outcome.lab is not None
    text = stage.console.export_text()
    for chapter in tour.CHAPTERS:
        assert chapter.title in text
    assert "It named itself" in text
    saved = load(default_path())
    assert saved.card.name == outcome.lab.name
    assert saved.card.corpus == corpus
    assert chat_path(default_path()).exists()


def test_starting_late_trains_a_model_first(stage):
    start = next(i for i, c in enumerate(tour.CHAPTERS, 1) if c.title == "Inference")
    outcome = tour.run(stage, corpus="fables", seed=3, start=start, budget=Budget.tiny())
    assert outcome.finished
    assert outcome.lab is not None and outcome.lab.trained
    text = stage.console.export_text()
    assert "Inference" in text and "Tokens" not in text


def test_starting_late_picks_up_the_saved_model():
    lab = Lab.create(built_in("fables"), 5, budget=Budget.tiny())
    lab.train_quietly()
    again = Lab.create(built_in("fables"), 5, budget=Budget.tiny())
    assert again.restore() and again.trained and again.name == lab.name
    for name, w in lab.model.params.items():
        np.testing.assert_array_equal(again.model.params[name], w)
    assert not Lab.create(built_in("fables"), 6, budget=Budget.tiny()).restore()


@dataclass
class Measuring(Stage):
    """A stage that notes every picture the edge of the terminal cuts into."""

    chapter: str = ""
    cut: list[str] = field(default_factory=list)

    def header(self, number: int, total: int, title: str, subtitle: str) -> None:
        self.chapter = title
        super().header(number, total, title, subtitle)

    def show(self, *items: RenderableType | list[Text], gap: bool = True) -> None:
        for item in items:
            picture = Group(*item) if isinstance(item, list) else item
            if loses_ink(self.console, picture, self.width):
                self.cut.append(f"{self.chapter}: {type(picture).__name__}")
        super().show(*items, gap=gap)


def loses_ink(console: Console, picture: RenderableType, width: int) -> bool:
    """Does drawing it this wide lose a character, or colored cells that wrapping can't explain?

    Where a line of colored text wraps, the space at the break goes, which is fine. Anything
    else that goes missing was cut off by the edge, or squeezed out of a table.
    """

    def ink(wide: int) -> tuple[int, int, int]:
        lines = console.render_lines(picture, console.options.update(width=wide), pad=False)
        chars = cells = 0
        for line in lines:
            for segment in line:
                chars += len(segment.text.replace(" ", ""))
                if segment.style and segment.style.bgcolor:
                    cells += segment.text.count(" ")
        return chars, cells, len(lines)

    chars, cells, lines = ink(width)
    all_chars, all_cells, all_lines = ink(1000)
    return chars != all_chars or all_cells - cells > lines - all_lines


@pytest.mark.parametrize(("width", "corpus"), [(80, "shakespeare"), (100, "fables")])
def test_no_picture_is_cut_off_at_the_edge_of_the_terminal(width, corpus):
    stage = Measuring(recording(width), animate=False)
    few = Budget(steps=10, memorizing=10, race=10, tuning=5, chat=5)  # the real model's shape
    assert tour.run(stage, corpus=corpus, seed=1, budget=few).finished
    assert stage.cut == []


def test_dice_repeat_for_a_seed_but_not_for_a_replay():
    first = Lab.create(built_in("fables"), 4, budget=Budget.tiny())
    second = Lab.create(built_in("fables"), 4, budget=Budget.tiny())
    roll = first.dice(7).random()
    assert roll == second.dice(7).random() == np.random.default_rng([4, 7]).random()
    assert first.dice(7).random() != roll  # the replay


def test_reseeding_starts_over_with_a_fresh_model():
    lab = Lab.create(built_in("fairytales"), 4, budget=Budget.tiny())
    lab.train_quietly()
    lab.dice(7)
    lab.reseed()
    assert lab.seed != 4
    assert not lab.trained and lab.trainer.step_number == 0
    assert lab.rolls == {}
    assert all((lab.initial.params[k] == lab.model.params[k]).all() for k in lab.model.params)


def test_the_dashboard_trains_to_the_end_a_frame_at_a_time():
    lab = Lab.create(built_in("fables"), 2, budget=Budget.tiny())
    watch = dashboard.Watch(lab.trainer, lab.baselines, lab.corpus, lab.seed, lab.rng)
    frames = list(dashboard.frames(watch, 96, seconds=2))
    assert lab.trained
    assert 1 < len(frames) <= lab.trainer.steps + 1  # at least one step per frame, after the first
