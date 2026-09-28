"""Byte-pair encoding: how the model splits text into tokens, the way GPT and Llama do.

Text is first cut into chunks at spaces and punctuation, so a token never spans two words; the
space before a word stays with it. Every chunk starts out as single characters. Then, over and
over, the pair of neighboring tokens that turns up most often is glued into one new token, until
the vocabulary is full. Common words end up as single tokens (" the", " Fox"), rarer ones as a
few pieces (" Gr", "apes"), and any text at all can still be written out, one character at a time
if need be.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from itertools import pairwise

# Special tokens aren't text. They mark where a document ends, and (for chat) who is speaking.
END, USER, ASSISTANT = 0, 1, 2
SPECIAL = ("<|endoftext|>", "<|user|>", "<|assistant|>")
# Every printable ASCII character and the newline, whether the text uses it or not, so that any
# prompt can be encoded. (Real tokenizers start from all 256 bytes, for the same reason.)
ALPHABET = "\n" + "".join(chr(c) for c in range(32, 127))

# GPT-2's rule for cutting text into chunks: common English endings like 's, words with the
# space before them, numbers, and runs of punctuation. Runs of newlines stay together, as in
# GPT-4's rule, so a paragraph break can become a single token.
PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)\b| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s*\n+|\s+(?!\S)|\s+"
)


def plain(text: str) -> str:
    """Text as plain ASCII: straight quotes, simple dashes, and accents taken off."""
    for fancy, simple in {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "--", "–": "-",
                          "Æ": "Ae", "æ": "ae", "Œ": "Oe", "œ": "oe", "…": "...",
                          "\t": " ", "\r\n": "\n", "\r": "\n"}.items():  # fmt: skip
        text = text.replace(fancy, simple)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text if ch in ALPHABET)


def chunks(text: str) -> list[str]:
    return PATTERN.findall(text)


def merge_pair(tokens: list[str], left: str, right: str) -> list[str]:
    """Glue every `left` followed by `right` into one token."""
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


@dataclass(frozen=True, slots=True)
class Merge:
    left: str
    right: str
    count: int  # how many times the pair turned up when it was chosen

    @property
    def token(self) -> str:
        return self.left + self.right


def learn(text: str, merges: int) -> list[Merge]:
    """Learn up to `merges` merges, most frequent pair first.

    Counting every pair from scratch after each merge would be slow, so the counts are kept up
    to date instead: a merge only changes the chunks that contain the pair it merged.
    """
    counts = Counter(chunks(text))
    words = [list(w) for w in counts]
    freq = list(counts.values())
    pairs: Counter[tuple[str, str]] = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        for pair in pairwise(w):
            pairs[pair] += freq[i]
            where[pair].add(i)
    # The most frequent pair is on top (ties go to the alphabetically first). Stale entries,
    # whose counts have changed since they were pushed, are skipped when they surface.
    heap = [(-n, pair) for pair, n in pairs.items()]
    heapify(heap)
    learned: list[Merge] = []
    while len(learned) < merges and heap:
        n, pair = heappop(heap)
        if pairs.get(pair, 0) != -n:
            continue
        if -n < 2:
            break
        learned.append(Merge(pair[0], pair[1], -n))
        for i in sorted(where.pop(pair, ())):
            old = words[i]
            new = merge_pair(old, *pair)
            if new == old:
                continue
            before = list(pairwise(old))
            after = list(pairwise(new))
            for p in before:
                pairs[p] -= freq[i]
            for p in after:
                pairs[p] += freq[i]
                where[p].add(i)
            words[i] = new
            for p in set(before) | set(after):
                if p != pair and pairs[p] > 0:
                    heappush(heap, (-pairs[p], p))
        pairs.pop(pair, None)
    return learned


@dataclass(slots=True)
class Tokenizer:
    """Special tokens first, then single characters, then every merge in the order learned."""

    merges: tuple[Merge, ...]
    pieces: list[str] = field(init=False)  # each token's text, by id
    ids: dict[str, int] = field(init=False)
    ranks: dict[tuple[str, str], int] = field(init=False)
    known: dict[str, list[int]] = field(init=False)  # chunks already encoded

    def __post_init__(self) -> None:
        self.pieces = [*SPECIAL, *ALPHABET, *(m.token for m in self.merges)]
        self.ids = {piece: i for i, piece in enumerate(self.pieces) if i >= len(SPECIAL)}
        self.ranks = {(m.left, m.right): rank for rank, m in enumerate(self.merges)}
        self.known = {}

    @classmethod
    def train(cls, text: str, size: int) -> Tokenizer:
        """A tokenizer with `size` tokens in all, learned from `text`."""
        return cls(tuple(learn(text, size - len(SPECIAL) - len(ALPHABET))))

    def __len__(self) -> int:
        return len(self.pieces)

    @property
    def first_merge(self) -> int:
        """The id of the first token made by merging."""
        return len(SPECIAL) + len(ALPHABET)

    def encode_chunk(self, chunk: str) -> list[int]:
        """Replay the merges on one chunk, earliest-learned first."""
        if (done := self.known.get(chunk)) is not None:
            return done
        tokens = list(chunk)
        while len(tokens) > 1:
            rank, at = min(
                (self.ranks.get(pair, len(self.ranks)), i)
                for i, pair in enumerate(pairwise(tokens))
            )
            if rank == len(self.ranks):
                break
            tokens = merge_pair(tokens, tokens[at], tokens[at + 1])
        ids = [self.ids[t] for t in tokens]
        self.known[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for chunk in chunks(plain(text)) for i in self.encode_chunk(chunk)]

    def split(self, text: str) -> list[str]:
        """Text cut into its tokens, as strings."""
        return [self.pieces[i] for i in self.encode(text)]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.pieces[int(i)] for i in ids if int(i) >= len(SPECIAL))

    def is_special(self, token: int) -> bool:
        return token < len(SPECIAL)

    def to_json(self) -> list[list[str | int]]:
        return [[m.left, m.right, m.count] for m in self.merges]

    @classmethod
    def from_json(cls, merges: Sequence[Sequence[str | int]]) -> Tokenizer:
        return cls(tuple(Merge(str(a), str(b), int(n)) for a, b, n in merges))
