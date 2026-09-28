import io

import numpy as np
import pytest
from rich.console import Console

from attention.corpus import built_in
from attention.model import Config
from attention.train import Trainer, prepare

# A model small enough to train in a few seconds, with every part of the real one in it.
SMALL = {"width": 32, "layers": 2, "heads": 4, "kv_heads": 2, "hidden": 64}
SMALL_VOCAB = 512  # fewer tokens than the real model's, to keep it quick


@pytest.fixture(autouse=True)
def model_home(tmp_path, monkeypatch):
    """Keep every test's saved models out of the real data directory."""
    home = tmp_path / "home"
    monkeypatch.setenv("ATTENTION_HOME", str(home))
    return home


def recording(width: int = 100) -> Console:
    from attention.tour.stage import THEME

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
def stage():
    """A stage that draws the final frame of every animation and never waits."""
    from attention.tour.stage import Stage

    return Stage(recording(), animate=False)


def small_trainer(key: str = "fables", seed: int = 3, steps: int = 300) -> Trainer:
    corpus = built_in(key)
    config = Config(vocab=SMALL_VOCAB, context=64, **SMALL)
    return prepare(corpus, seed, steps=steps, config=config, vocab=SMALL_VOCAB)


@pytest.fixture(scope="session")
def trained() -> Trainer:
    """A small model, trained for a few hundred steps on the fables. Don't change it."""
    trainer = small_trainer()
    while not trainer.done:
        trainer.step()
    trainer.evaluate()
    return trainer


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
