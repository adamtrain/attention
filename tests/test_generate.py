import numpy as np
import pytest

from attention.explain import influence, nudge
from attention.generate import (
    Memory,
    choose,
    dials,
    log_probs,
    nucleus,
    picks,
    prompt,
    samples,
    top_k,
    write,
)
from attention.tokenizer import END


def test_choose_lines_probabilities_up_from_zero_to_one():
    probs = np.array([0.5, 0.25, 0.25])
    assert [choose(probs, r) for r in (0.0, 0.49, 0.5, 0.74, 0.75, 0.999)] == [0, 0, 1, 1, 2, 2]


def test_top_p_keeps_the_fewest_tokens_that_reach_p():
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    np.testing.assert_allclose(nucleus(probs, 0.75), [0.5 / 0.8, 0.3 / 0.8, 0, 0])
    np.testing.assert_allclose(nucleus(probs, 0.5), [1, 0, 0, 0])
    np.testing.assert_array_equal(nucleus(probs, 1.0), probs)


def test_top_k_keeps_the_k_likeliest():
    probs = np.array([0.1, 0.5, 0.15, 0.25])
    np.testing.assert_allclose(top_k(probs, 2), [0, 0.5 / 0.75, 0, 0.25 / 0.75])
    np.testing.assert_array_equal(top_k(probs, 0), probs)
    cold = dials(np.array([2.0, 1.0, 0.0]), temperature=0.1)
    assert cold[0] > 0.99


def test_writing_carries_on_from_the_prompt_until_the_end(trained, rng):
    model, tok = trained.model, trained.data.tokenizer
    steps = list(picks(model, prompt(tok, "The Fox and the"), rng, limit=40))
    assert tok.decode(steps[0].context) == "The Fox and the"
    assert steps[0].context[0] == END
    assert all(abs(p.probs.sum() - 1) < 1e-9 for p in steps)
    assert steps[-1].done or len(steps) == 40
    assert steps[1].weights.shape == (2, 4, len(steps[1].context))  # layers, heads, positions
    assert isinstance(write(model, tok, rng, "The Fox", limit=10), str)
    assert len(samples(model, tok, rng, 3, limit=5)) == 3


def test_writing_with_the_cache_matches_rereading_everything(trained):
    model, tok = trained.model, trained.data.tokenizer
    steps = list(picks(model, prompt(tok, "A Wolf"), np.random.default_rng(1), limit=12))
    for pick in steps:
        again = model.forward(np.array([pick.context])).probs[0, -1]
        np.testing.assert_allclose(pick.probs, again, atol=1e-10)


def test_the_context_window_stops_the_writing(trained, rng):
    ids = [END] * (trained.model.config.context - 3)
    assert len(list(picks(trained.model, ids, rng, stop=(), limit=100))) == 3


def test_copies_of_the_training_text_are_spotted(trained):
    memory = Memory(trained.data.train)
    copied = trained.data.train[1000:1030]
    invented = np.random.default_rng(0).integers(100, 500, 30)
    assert memory.copied(copied).all() and memory.longest(copied) == 30
    assert not memory.copied(invented).any()
    mixed = np.concatenate([invented[:10], copied[:15], invented[10:20]])
    assert list(np.flatnonzero(memory.copied(mixed))) == list(range(10, 25))


def test_log_probs_score_every_token_and_the_ending(trained):
    tok = trained.data.tokenizer
    lp = log_probs(trained.model, tok, "The Fox and the Grapes")
    assert lp.shape == (len(tok.encode("The Fox and the Grapes")),)
    assert (lp <= 0).all()


def test_backprop_to_the_inputs_leaves_the_weights_alone(trained):
    model, tok = trained.model, trained.data.tokenizer
    before = {k: v.copy() for k, v in model.params.items()}
    report = influence(model, prompt(tok, "The Fox and the"))
    assert report.sizes.shape == (len(report.ids),)
    assert (report.sizes > 0).all()
    assert report.runner_up not in (report.top, END)
    for k, v in model.params.items():
        np.testing.assert_array_equal(v, before[k])


def test_one_step_toward_a_token_makes_it_likelier(trained):
    model, tok = trained.model, trained.data.tokenizer
    target = tok.encode(" Crane")[0]
    change = nudge(model, prompt(tok, "The Wolf and the"), target)
    assert change.after[target] > change.before[target]
    assert change.after[target] >= 0.2 or change.lr == pytest.approx(4.0)
