from attention import bpe
from attention.corpus import built_in


def test_the_most_common_pair_is_merged_first():
    merges = bpe.learn(["abab", "abc", "xab"], 2)
    assert merges[0].token == "ab" and merges[0].count == 4
    assert bpe.encode("abab", merges[:1]) == ["ab", "ab"]


def test_merges_stop_when_nothing_repeats():
    assert bpe.learn(["abc"], 10) == []


def test_encoding_replays_merges_in_order():
    merges = [bpe.Merge("a", "b", 2), bpe.Merge("ab", "c", 2)]
    assert bpe.encode("abcab", merges) == ["abc", "ab"]
    assert "".join(bpe.encode("stegosaurus", bpe.learn(["stegosaurus"] * 2, 5))) == "stegosaurus"


def test_dinosaurs_learn_their_favourite_chunk():
    words = built_in("dinosaurs").words
    merges = bpe.learn(words, 40)
    assert len(merges) == 40
    assert "osaurus" in bpe.encode("stegosaurus", merges)
    assert bpe.average_length(words, merges) < 0.6 * bpe.average_length(words, [])
