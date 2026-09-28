import json

import numpy as np
import pytest

from attention.corpus import built_in
from attention.store import (
    Card,
    Outdated,
    Saved,
    chat_path,
    default_path,
    is_model,
    leftovers,
    load,
    pick_name,
    save,
)


def card(name: str = "Knogly") -> Card:
    return Card(
        name, "fables", "Fables", "fable", "fables", 3, 300, 2.9, 3.2, 4.2, Card.now(), "The"
    )


def test_a_trained_model_saves_and_loads(trained, model_home):
    saved = Saved(trained.model, trained.data.tokenizer, card(), trained.data.train_docs)
    path = save(default_path(), saved)
    assert path == model_home / "model.npz" and is_model(path)
    again = load(path)
    assert again.card == saved.card
    assert again.tokenizer.pieces == saved.tokenizer.pieces
    assert again.documents == saved.documents
    for name, w in trained.model.params.items():
        np.testing.assert_array_equal(again.model.params[name], w)
    assert again.memory.longest(trained.data.train[500:520]) == 20
    assert leftovers() == [path]


def test_models_from_the_old_letter_by_letter_version_are_recognized(model_home):
    old = model_home / "model.npz"
    old.parent.mkdir(parents=True)
    np.savez(old, meta=np.array(json.dumps({"format": 1, "card": {"name": "Stegotops"}})))
    assert is_model(old)  # so `attention clean` can still tidy it away
    with pytest.raises(Outdated):
        load(old)


def test_the_chat_add_on_lives_beside_the_model(tmp_path):
    assert chat_path(tmp_path / "model.npz") == tmp_path / "model.chat.npz"


def test_models_name_themselves_with_new_words(trained):
    corpus = built_in("fables")
    name = pick_name(trained.model, trained.data.tokenizer, np.random.default_rng(0), corpus)
    assert name[0].isupper() and name[1:].islower()
    words = {w.lower() for d in corpus.documents for w in d.split()}
    assert name.lower() not in words or name == corpus.title
