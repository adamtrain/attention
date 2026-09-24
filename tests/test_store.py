import numpy as np

from attention.corpus import built_in
from attention.store import default_path, load, pick_name
from attention.tour.lab import Lab


def test_a_trained_model_saves_and_loads(model_home):
    lab = Lab.create(built_in("towns"), seed=9)
    lab.train_quietly()
    assert lab.saved_to == default_path() == model_home / "model.npz"
    saved = load(lab.saved_to)
    assert saved.card.name == lab.name
    assert saved.card.seed == 9 and saved.card.corpus == "towns"
    assert saved.vocab == lab.vocab
    assert saved.known == set(lab.corpus.words)
    for name, w in lab.model.params.items():
        np.testing.assert_array_equal(saved.model.params[name], w)


def test_models_name_themselves_with_new_words():
    lab = Lab.create(built_in("dinosaurs"), seed=4)
    lab.train_quietly()
    name = pick_name(lab.model, lab.vocab, np.random.default_rng(0), lab.corpus)
    assert name[0].isupper() and 4 <= len(name) <= 10
    assert name.lower() not in lab.corpus.words
