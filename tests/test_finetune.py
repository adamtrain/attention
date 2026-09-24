import numpy as np
import pytest

from attention.corpus import built_in
from attention.finetune import learn_from, niche_trainer, share
from attention.generate import invent
from attention.train import prepare


@pytest.fixture(scope="module")
def pretrained():
    trainer = prepare(built_in("dinosaurs"), seed=3)
    while not trainer.done:
        trainer.step()
    return trainer


def test_fine_tuning_on_a_niche_shifts_what_it_writes(pretrained):
    niche = built_in("dinosaurs").niche
    rng = np.random.default_rng(0)
    before = share(invent(pretrained.model, pretrained.data.vocab, rng, 40, 0.8), niche)
    tuner = niche_trainer(pretrained.model, pretrained.data, niche, seed=3)
    assert all(niche.has(w) for w in tuner.data.train)
    while not tuner.done:
        tuner.step()
    after = share(invent(tuner.model, pretrained.data.vocab, rng, 40, 0.8), niche)
    assert after > before + 0.4
    assert tuner.model is not pretrained.model


def test_feedback_makes_liked_words_likelier_on_a_copy(pretrained):
    vocab = pretrained.data.vocab
    words = invent(pretrained.model, vocab, np.random.default_rng(1), 6, 0.8)
    original = {k: v.copy() for k, v in pretrained.model.params.items()}
    result = learn_from(pretrained.model, pretrained.data, words[:2], words[2:])
    for liked in words[:2]:
        was, now = result.odds(liked)
        assert now < was  # "1 in N" got smaller: likelier
    for name, w in pretrained.model.params.items():
        np.testing.assert_array_equal(w, original[name])
