import numpy as np

from attention.corpus import built_in
from attention.explain import influence, nudge
from attention.generate import choose, invent, log_prob, nucleus, picks, sample
from attention.train import prepare


def trained(key="dinosaurs", seed=2):
    trainer = prepare(built_in(key), seed=seed)
    while not trainer.done:
        trainer.step()
    return trainer.model, trainer.data.vocab


def test_choose_lines_probabilities_up_from_zero_to_one():
    probs = np.array([0.5, 0.25, 0.25])
    assert [choose(probs, r) for r in (0.0, 0.49, 0.5, 0.74, 0.75, 0.999)] == [0, 0, 1, 1, 2, 2]


def test_writing_stops_at_the_boundary_and_respects_the_prefix():
    model, vocab = trained()
    rng = np.random.default_rng(0)
    steps = list(picks(model, vocab, rng, prefix="steg"))
    assert steps[0].context == ".steg"
    assert all(abs(p.probs.sum() - 1) < 1e-9 for p in steps)
    assert steps[-1].done or len(steps[-1].context) == model.config.context
    assert sample(model, vocab, rng, prefix="steg").startswith("steg")


def test_cold_sampling_is_less_varied_than_hot():
    model, vocab = trained("names")
    rng = np.random.default_rng(1)
    cold = {sample(model, vocab, rng, temperature=0.2) for _ in range(30)}
    hot = {sample(model, vocab, rng, temperature=1.5) for _ in range(30)}
    assert len(cold) < len(hot)
    assert len(invent(model, vocab, rng, 5)) == 5


def test_backprop_to_the_inputs_leaves_the_weights_alone():
    model, vocab = trained()
    before = {k: v.copy() for k, v in model.params.items()}
    report = influence(model, vocab, "stegosa")
    assert report.sizes.shape == (8,)
    assert (report.sizes > 0).all()
    assert vocab.chars[report.top] == "u"
    for k, v in model.params.items():
        np.testing.assert_array_equal(v, before[k])


def test_one_step_toward_a_letter_makes_it_likelier():
    model, vocab = trained()
    target = vocab.chars.index("e")
    change = nudge(model, vocab, "stegosa", target)
    assert change.after[target] > change.before[target]
    assert change.after[target] >= 0.2 or change.lr == 4.0


def test_top_p_keeps_the_fewest_tokens_that_reach_p():
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    np.testing.assert_allclose(nucleus(probs, 0.75), [0.5 / 0.8, 0.3 / 0.8, 0, 0])
    np.testing.assert_allclose(nucleus(probs, 0.5), [1, 0, 0, 0])
    np.testing.assert_array_equal(nucleus(probs, 1.0), probs)


def test_top_p_sampling_only_picks_from_the_nucleus():
    model, vocab = trained()
    rng = np.random.default_rng(3)
    for pick in picks(model, vocab, rng, top_p=0.5):
        assert pick.probs[pick.token] > 0
        assert (pick.probs > 0).sum() <= len(pick.probs)


def test_log_prob_scores_every_letter_and_the_ending():
    model, vocab = trained()
    lp = log_prob(model, vocab, "stegosaurus")
    assert lp.shape == (len("stegosaurus") + 1,)
    assert (lp <= 0).all()
    assert lp.sum() > log_prob(model, vocab, "sgteosuarsu").sum()
