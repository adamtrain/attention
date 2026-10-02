import numpy as np
import pytest

from attention.corpus import Dataset, built_in
from attention.model import cross_entropy
from attention.train import (
    CONTEXT,
    STEPS,
    Adam,
    baselines,
    clip,
    evaluate,
    gradients,
    prepare,
    smooth,
    steps_for,
)

from .conftest import small_trainer


def test_training_beats_counting_tokens(trained):
    b = baselines(trained.data)
    assert b.uniform > b.tokens > b.pairs
    first = trained.val_losses[0][1] if len(trained.val_losses) > 1 else b.uniform
    assert trained.val_losses[-1][1] < b.tokens
    assert trained.val_losses[-1][1] <= first
    assert smooth(trained.losses, 0.1)[-1] < smooth(trained.losses, 0.1)[20]


def test_an_untrained_model_is_about_as_good_as_guessing():
    trainer = small_trainer(steps=1)
    assert abs(evaluate(trainer.model, *trainer.held) - baselines(trainer.data).uniform) < 0.2


def test_the_same_seed_makes_the_same_model():
    runs = []
    for _ in range(2):
        trainer = small_trainer("shakespeare", seed=42, steps=10)
        while not trainer.done:
            trainer.step()
        runs.append(trainer.model.params["layers.1.mlp.up"])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_steps_report_what_moved_and_how_big_the_gradient_was():
    trainer = small_trainer("fairytales", steps=5)
    before = trainer.model.params["embed"].copy()
    step = trainer.step()
    np.testing.assert_allclose(trainer.model.params["embed"], before + step.moved["embed"])
    sizes = step.group_sizes(trainer.model.config)
    assert list(sizes) == ["embeddings", "layer 1", "layer 2", "final norm"]
    assert all(n > 0 for n in sizes.values())
    assert step.size == pytest.approx(np.sqrt(sum(s * s for s in sizes.values())))


def test_a_batch_split_between_cores_has_the_same_gradient():
    trainer = small_trainer(steps=1)
    inputs, targets = trainer.batch()
    targets = targets.copy()
    targets[:5, :40] = -1  # shares with different numbers of targets, as in chat fine-tuning
    targets[8:12] = -1  # and one with none at all
    trace = trainer.model.forward(inputs)
    loss, dlogits = cross_entropy(trace.logits, targets)
    whole, _ = trainer.model.backward(trace, dlogits)
    split_loss, split = gradients(trainer.model, inputs, targets)
    assert split_loss == pytest.approx(loss)
    assert list(split) == list(whole)
    for name, grad in whole.items():
        np.testing.assert_allclose(split[name], grad, atol=1e-12)


def test_clipping_shortens_a_long_gradient_without_turning_it():
    grads = {"a": np.array([3.0, 0.0]), "b": np.array([4.0])}
    clipped, size = clip(grads, 1.0)
    assert size == pytest.approx(5.0)
    np.testing.assert_allclose(clipped["a"], [0.6, 0.0])
    assert clip(grads, 10.0)[0] is grads


def test_adam_moves_against_the_gradient():
    w = {"w": np.array([1.0, -1.0])}
    moved = Adam(w).step(w, {"w": np.array([2.0, -3.0])}, lr=0.1)
    assert moved["w"][0] < 0 < moved["w"][1]
    np.testing.assert_allclose(np.abs(moved["w"]), 0.1, rtol=1e-6)  # the first step is ±lr


def test_weight_decay_shrinks_weights_but_not_the_norms():
    params = {"layers.0.mlp.up": np.ones((2, 2)), "layers.0.mlp.norm": np.ones(2)}
    adam = Adam(params, decay=2.0)
    adam.step(params, {k: np.zeros_like(v) for k, v in params.items()}, lr=0.01)
    assert np.allclose(params["layers.0.mlp.up"], 0.98)  # no gradient at all, and it still shrank
    assert np.allclose(params["layers.0.mlp.norm"], 1.0)


def test_learning_rate_warms_up_then_eases_down():
    trainer = small_trainer(steps=1000)
    rates = []
    for n in (0, 50, 100, 550, 999):
        trainer.step_number = n
        rates.append(trainer.learning_rate())
    assert rates[0] < rates[1] < rates[2]
    assert rates[2] > rates[3] > rates[4] == pytest.approx(trainer.lr * 0.1, rel=0.01)


def test_more_text_means_more_steps():
    small = Dataset.split(built_in("fables"), np.random.default_rng([1, 0]))  # as seed 1 splits it
    big = Dataset.split(built_in("fairytales"), np.random.default_rng([1, 0]))
    assert STEPS[0] <= steps_for(small) < steps_for(big) <= STEPS[1]
    assert prepare(built_in("fables"), 1).steps == steps_for(small, context=CONTEXT)


def test_smoothing_follows_the_trend():
    assert smooth([]) == []
    assert smooth([4.0, 0.0, 0.0, 0.0], alpha=0.5) == [4.0, 2.0, 1.0, 0.5]
