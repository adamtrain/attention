"""Byte-pair encoding: how real language models choose their tokens.

Start with single letters. Find the pair of neighboring tokens that turns up most often, glue
it into a new token, and repeat. Common chunks ("saurus", "ing", " the") become single tokens,
and text takes fewer of them. Your model doesn't use this; the tour just shows it working.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class Merge:
    left: str
    right: str
    count: int  # how many times the pair turned up when it was chosen

    @property
    def token(self) -> str:
        return self.left + self.right


def merge_pair(tokens: list[str], left: str, right: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
            out.append(left + right)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def learn(words: tuple[str, ...] | list[str], merges: int) -> list[Merge]:
    """Learn up to `merges` merges from a list of words, most frequent pair first."""
    seqs = [list(w) for w in words]
    learned: list[Merge] = []
    for _ in range(merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for seq in seqs:
            pairs.update(pairwise(seq))
        if not pairs:
            break
        (left, right), count = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        if count < 2:
            break
        learned.append(Merge(left, right, count))
        seqs = [merge_pair(seq, left, right) for seq in seqs]
    return learned


def encode(word: str, merges: list[Merge]) -> list[str]:
    """Split a word into tokens by replaying the merges in the order they were learned."""
    tokens = list(word)
    for m in merges:
        tokens = merge_pair(tokens, m.left, m.right)
    return tokens


def average_length(words: tuple[str, ...] | list[str], merges: list[Merge]) -> float:
    return sum(len(encode(w, merges)) for w in words) / max(len(words), 1)
