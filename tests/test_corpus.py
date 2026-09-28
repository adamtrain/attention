import numpy as np
import pytest

from attention.corpus import (
    BUILT_IN,
    SEPARATOR,
    VOCAB,
    Dataset,
    Niche,
    built_in,
    common_topic,
    documents,
    from_file,
    load,
)
from attention.tokenizer import END


@pytest.mark.parametrize("key", list(BUILT_IN))
def test_built_in_corpora_load_with_a_good_example(key):
    corpus = built_in(key)
    assert len(corpus.documents) > 100
    assert corpus.characters > 500_000
    text = "\n\n".join(corpus.documents)
    assert corpus.example.split("\n\n")[-1][:40] in text
    assert all(SEPARATOR not in d and d.strip() == d for d in corpus.documents)
    assert 0.03 < sum(map(corpus.niche.has, corpus.documents)) / len(corpus.documents) < 0.3


def test_versions_of_one_fable_are_held_back_together():
    corpus = built_in("fables")
    data = Dataset.split(corpus, np.random.default_rng(1))
    held = {corpus.family(d) for d in data.val_docs}
    assert held and not held & {corpus.family(d) for d in data.train_docs}
    grapes = [d for d in corpus.documents if d.startswith("The Fox and the Grapes\n")]
    assert len(grapes) >= 3  # one title, several translations


def test_the_split_is_seeded_and_the_tokenizer_learns_only_from_training_text():
    corpus = built_in("shakespeare")
    a = Dataset.split(corpus, np.random.default_rng(4))
    b = Dataset.split(corpus, np.random.default_rng(4))
    assert a.val_docs == b.val_docs
    assert len(a.val_docs) == round(len(corpus.documents) * 0.1)
    assert len(a.tokenizer) == VOCAB
    assert a.train[0] == END and a.val[0] == END
    assert (a.train == END).sum() == len(a.train_docs) + 1


def test_windows_are_the_next_token_at_every_position():
    data = Dataset.split(built_in("fables"), np.random.default_rng(0))
    inputs, targets = data.windows(np.random.default_rng(1), 4, 16)
    assert inputs.shape == targets.shape == (4, 16)
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])
    fixed = data.fixed(5, 16)
    np.testing.assert_array_equal(fixed[0], data.fixed(5, 16)[0])


def test_documents_split_at_markers_or_blank_lines():
    assert documents(f"one\n{SEPARATOR}\ntwo\n") == ("one", "two")
    many = "\n\n\n".join(f"Doc {i}. Some words here." for i in range(25))
    assert len(documents(many)) == 25
    one_long = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(30))
    assert 1 < len(documents(one_long)) < 30  # cut into pages


def test_your_own_text_files(tmp_path):
    path = tmp_path / "moby.txt"
    path.write_text(
        "\n\n\n".join(
            f"Chapter {i}. Call me Ishmael, said the sailor to Queequeg on the {i}th day. "
            + "The whale swam on. " * 30
            for i in range(40)
        )
    )
    corpus = load(str(path))
    assert corpus.key == "moby" and len(corpus.documents) == 40
    assert corpus.example and corpus.probe and corpus.example.startswith(corpus.probe)
    (tmp_path / "few.txt").write_text("Too short.\n")
    with pytest.raises(ValueError, match="at least"):
        from_file(tmp_path / "few.txt")
    with pytest.raises(ValueError, match="No corpus"):
        load("nonexistent-corpus")


def test_a_niche_can_look_at_just_the_title_or_the_opening():
    assert Niche(r"\bFox\b", "foxes", titles=True).has("The Fox\n\nA Crow")
    assert not Niche(r"\bCrow\b", "crows", titles=True).has("The Fox\n\nA Crow")
    assert Niche(r"\bCrow\b", "crows", opening=20).has("The Fox\n\nA Crow")
    assert not Niche(r"\bCrow\b", "crows", opening=5).has("The Fox\n\nA Crow")


def test_custom_text_gets_a_niche():
    docs = tuple(
        f"Once the Captain met a sailor{', and Queequeg sang' if i % 5 == 0 else ''}. The end."
        for i in range(40)
    )
    niche = common_topic(docs)
    assert niche.has(docs[0]) and not niche.has(docs[1])
