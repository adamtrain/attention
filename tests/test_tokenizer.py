from collections import Counter
from itertools import pairwise

from attention.corpus import built_in
from attention.tokenizer import (
    ALPHABET,
    END,
    SPECIAL,
    Tokenizer,
    chunks,
    learn,
    merge_pair,
    plain,
)


def naive(text: str, merges: int) -> list[tuple[str, str]]:
    """BPE the slow, obvious way: recount every pair after every merge."""
    words = Counter(tuple(w) for w in chunks(text))
    out = []
    for _ in range(merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for w, n in words.items():
            for p in pairwise(w):
                pairs[p] += n
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], [-ord(ch) for ch in "".join(kv[0])]))
        (a, b), n = min(((p, c) for p, c in pairs.items() if c == best[1]), key=lambda pc: pc[0])
        if n < 2:
            break
        out.append((a, b))
        words = Counter({tuple(merge_pair(list(w), a, b)): c for w, c in words.items()})
    return out


def test_text_is_cut_into_chunks_with_their_spaces():
    assert chunks("The Fox's tail, 12 times.\n\nEnd") == [
        "The", " Fox", "'s", " tail", ",", " 12", " times", ".", "\n\n", "End",
    ]  # fmt: skip


def test_the_most_frequent_pair_is_merged_first():
    merges = learn("abab abc xab", 2)
    assert merges[0].token == "ab" and merges[0].count == 4


def test_keeping_counts_up_to_date_finds_the_same_merges_as_recounting():
    text = "\n\n".join(built_in("fables").documents[:40])
    fast = [(m.left, m.right) for m in learn(text, 60)]
    assert fast == naive(text, 60)


def test_encoding_round_trips_any_text():
    tok = Tokenizer.train("\n\n".join(built_in("fables").documents[:100]), 300)
    assert len(tok) == 300
    for text in ("The Fox and the Grapes", "Zebra quokka 1234 #@!", "  two  spaces\nand lines"):
        assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode("The Fox and the Grapes")) < len("The Fox and the Grapes") / 2


def test_special_tokens_come_first_and_never_come_from_text():
    tok = Tokenizer.train("aaa bbb aaa bbb", 110)
    assert tok.pieces[: len(SPECIAL)] == list(SPECIAL)
    assert tok.pieces[len(SPECIAL) : tok.first_merge] == list(ALPHABET)
    assert END not in tok.encode("<|endoftext|>")
    assert tok.decode([END, *tok.encode("hi")]) == "hi"


def test_saved_merges_make_the_same_tokenizer():
    tok = Tokenizer.train("\n\n".join(built_in("shakespeare").documents[:20]), 200)
    again = Tokenizer.from_json(tok.to_json())
    assert again.pieces == tok.pieces
    text = "ROMEO:\nBut soft, what light through yonder window breaks?"
    assert again.encode(text) == tok.encode(text)


def test_plain_turns_fancy_text_into_ascii():
    assert plain("“Æsop’s” fables — café…\ttab") == '"Aesop\'s" fables -- cafe... tab'
