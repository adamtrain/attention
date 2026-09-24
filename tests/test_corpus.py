import numpy as np
import pytest

from attention.corpus import (
    BUILT_IN,
    Dataset,
    Vocab,
    built_in,
    clean,
    common_ending,
    from_file,
    load,
)


@pytest.mark.parametrize("key", list(BUILT_IN))
def test_built_in_corpora_load_and_fit(key):
    corpus = built_in(key)
    assert len(corpus.words) > 300
    assert corpus.example in corpus.words
    assert all(w.isalpha() and w.islower() for w in corpus.words)
    assert len(set(corpus.words)) == len(corpus.words)
    assert 20 <= sum(corpus.niche.has(w) for w in corpus.words) <= len(corpus.words) // 5


def test_clean_keeps_letters_and_drops_repeats_and_giants():
    assert clean(["Tyrannosaurus!", "tyrannosaurus", "  T-Rex ", "x", "a" * 40]) == (
        "tyrannosaurus",
        "trex",
    )


def test_vocab_round_trips_with_the_boundary_first():
    vocab = Vocab.of(("emma", "ava"))
    assert vocab.chars == ".aemv"
    assert vocab.sequence("emma") == [0, 2, 3, 3, 1, 0]
    assert vocab.decode(vocab.encode("ave")) == "ave"


def test_batches_shift_targets_by_one_and_pad_with_minus_one():
    data = Dataset.split(built_in("names"), np.random.default_rng(0))
    inputs, targets = data.batch(["ava", "emma"])
    v = data.vocab
    assert inputs.shape == targets.shape == (2, 5)
    assert v.decode(inputs[1]) == ".emma"
    assert v.decode(targets[1]) == "emma."
    assert list(targets[0, 4:]) == [-1]


def test_split_holds_back_a_tenth_and_is_seeded():
    corpus = built_in("dinosaurs")
    a = Dataset.split(corpus, np.random.default_rng(1))
    b = Dataset.split(corpus, np.random.default_rng(1))
    assert a.val == b.val
    assert len(a.val) == round(len(corpus.words) * 0.1)
    assert not set(a.val) & set(a.train)
    assert a.context == max(len(w) for w in corpus.words) + 1


def test_word_lists_from_files(tmp_path):
    path = tmp_path / "planets.txt"
    path.write_text("\n".join(f"planet{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(40)))
    corpus = load(str(path))
    assert corpus.key == "planets"
    assert corpus.probe and corpus.example.startswith(corpus.probe)
    (tmp_path / "few.txt").write_text("one\ntwo\n")
    with pytest.raises(ValueError, match="at least 20"):
        from_file(tmp_path / "few.txt")
    with pytest.raises(ValueError, match="No corpus"):
        load("nonexistent-corpus")


def test_custom_word_lists_get_a_niche():
    words = tuple(f"{a}{b}ville" for a in "bcdfg" for b in "aeiou") + tuple(
        f"{a}{b}{c}" for a in "bcdfghklm" for b in "aeiou" for c in "npt"
    )
    niche = common_ending(words)
    assert niche.ending == "ville"
    assert niche.label == "words ending in -ville"
