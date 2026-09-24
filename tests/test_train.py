import numpy as np

from attention.corpus import built_in
from attention.train import Adam, baselines, evaluate, prepare, smooth


def test_training_beats_counting_letter_pairs():
    trainer = prepare(built_in("dinosaurs"), seed=3)
    b = baselines(trainer.data)
    start = evaluate(trainer.model, trainer.data, trainer.data.val)
    assert abs(start - b.uniform) < 0.1  # untrained: every token about equally likely
    while not trainer.done:
        trainer.step()
    held_back = evaluate(trainer.model, trainer.data, trainer.data.val)
    assert held_back < b.pairs
    assert b.uniform > b.letters > b.pairs


def test_the_same_seed_makes_the_same_model():
    runs = []
    for _ in range(2):
        trainer = prepare(built_in("names"), seed=42, steps=30)
        while not trainer.done:
            trainer.step()
        runs.append(trainer.model.params["mlp.up"])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_steps_report_what_moved():
    trainer = prepare(built_in("towns"), seed=1, steps=5)
    before = trainer.model.params["embed.token"].copy()
    step = trainer.step()
    np.testing.assert_allclose(
        trainer.model.params["embed.token"], before + step.moved["embed.token"]
    )
    norms = step.layer_norms()
    assert set(norms) == {"embeddings", "attention", "mlp", "head"}
    assert all(n > 0 for n in norms.values())


def test_adam_moves_against_the_gradient():
    w = {"w": np.array([1.0, -1.0])}
    moved = Adam(w).step(w, {"w": np.array([2.0, -3.0])}, lr=0.1)
    assert moved["w"][0] < 0 < moved["w"][1]
    np.testing.assert_allclose(np.abs(moved["w"]), 0.1, rtol=1e-6)  # the first step is ±lr


def test_learning_rate_warms_up_then_decays():
    trainer = prepare(built_in("names"), seed=0, steps=100)
    first = trainer.learning_rate()
    trainer.step_number = 20
    peak = trainer.learning_rate()
    trainer.step_number = 99
    assert first < peak and trainer.learning_rate() < peak


def test_smoothing_follows_the_trend():
    assert smooth([]) == []
    out = smooth([4.0, 0.0, 0.0, 0.0], alpha=0.5)
    assert out == [4.0, 2.0, 1.0, 0.5]
