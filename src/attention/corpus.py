"""The words a model learns from, and turning them into numbers (tokens) and back."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

BOUNDARY = "."  # marks where a word starts and ends
MAX_LENGTH = 23  # longer words don't fit the model's context


@dataclass(frozen=True, slots=True)
class Niche:
    """A family of words that share an ending: something to fine-tune a model toward."""

    ending: str  # "ceratops"
    label: str  # "horned dinosaurs, the -ceratops family"

    def has(self, word: str) -> bool:
        return word.endswith(self.ending)


@dataclass(frozen=True, slots=True)
class Corpus:
    key: str
    title: str  # "Dinosaurs"
    noun: str  # "dinosaur"
    blurb: str  # what's in it
    words: tuple[str, ...]
    example: str  # a word to take apart in the tour
    probe: str  # a beginning to watch the model's predictions for while it trains
    plural: str  # "dinosaurs"
    niche: Niche


BUILT_IN = {
    "dinosaurs": ("Dinosaurs", "dinosaur", "dinosaur genera, Abelisaurus to Zuul", "stegosaurus",
                  "stegosa", Niche("ceratops", "horned dinosaurs, the -ceratops family")),
    "names": ("Names", "name", "given names from around the world", "olivia", "mar",
              Niche("o", "names ending in -o")),
    "towns": ("English towns", "town", "towns and villages in England", "cheltenham", "bur",
              Niche("ford", "-ford towns, named for a river crossing")),
}  # fmt: skip


def clean(lines: list[str]) -> tuple[str, ...]:
    """Lowercase letters only, one word per line, no repeats, nothing too long."""
    words = []
    seen = set()
    for line in lines:
        word = re.sub(r"[^a-z]", "", line.lower())
        if 1 < len(word) <= MAX_LENGTH and word not in seen:
            seen.add(word)
            words.append(word)
    return tuple(words)


def built_in(key: str) -> Corpus:
    title, noun, blurb, example, probe, niche = BUILT_IN[key]
    text = files("attention").joinpath("corpora", f"{key}.txt").read_text(encoding="utf-8")
    words = clean(text.splitlines())
    plural = "towns" if key == "towns" else title.lower()
    return Corpus(key, title, noun, f"{len(words):,} {blurb}", words, example, probe, plural, niche)


def common_ending(words: tuple[str, ...]) -> Niche:
    """The longest ending shared by a decent handful of words, but not by most of them."""
    enough = max(8, len(words) // 20)
    for length in (5, 4, 3, 2, 1):
        counts = Counter(w[-length:] for w in words if len(w) > length)
        ending, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]), default=("", 0))
        if enough <= n <= len(words) // 2:
            return Niche(ending, f"words ending in -{ending}")
    ending = Counter(w[-1] for w in words).most_common(1)[0][0]
    return Niche(ending, f"words ending in -{ending}")


def from_file(path: Path) -> Corpus:
    words = clean(path.read_text(encoding="utf-8", errors="replace").splitlines())
    if len(words) < 20:
        raise ValueError(
            f"{path.name} has {len(words)} usable words; the model needs at least 20. "
            "Put one word per line (letters a–z)."
        )
    by_length = sorted(words, key=len)
    example = by_length[len(by_length) * 2 // 3]
    return Corpus(
        key=path.stem,
        title=path.stem.replace("_", " ").replace("-", " ").capitalize(),
        noun="word",
        blurb=f"{len(words):,} words from {path.name}",
        words=words,
        example=example,
        probe=example[: max(2, len(example) // 2)],
        plural="words",
        niche=common_ending(words),
    )


def load(name: str) -> Corpus:
    """A built-in corpus by name, or a text file with one word per line."""
    if name in BUILT_IN:
        return built_in(name)
    path = Path(name).expanduser()
    if path.is_file():
        return from_file(path)
    raise ValueError(f"No corpus called {name!r}. Try {', '.join(BUILT_IN)}, or a file path.")


# ── Tokens ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Vocab:
    """Every character the model knows, in order. A character's position is its token id."""

    chars: str

    @classmethod
    def of(cls, words: tuple[str, ...]) -> Vocab:
        return cls(BOUNDARY + "".join(sorted(set("".join(words)))))

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.chars.index(ch) for ch in text]

    def decode(self, ids: list[int] | np.ndarray) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def sequence(self, word: str) -> list[int]:
        """A word as the model sees it: `.word.`, as ids."""
        return self.encode(BOUNDARY + word + BOUNDARY)


@dataclass(frozen=True, slots=True)
class Dataset:
    vocab: Vocab
    train: tuple[str, ...]
    val: tuple[str, ...]  # held back, to check the model isn't just memorizing

    @property
    def context(self) -> int:
        """Room for `.` plus the longest word: the model reads `.word` and predicts `word.`."""
        return max(len(w) for w in self.train + self.val) + 1

    @classmethod
    def split(cls, corpus: Corpus, rng: np.random.Generator, held_out: float = 0.1) -> Dataset:
        words = list(corpus.words)
        rng.shuffle(words)
        n = max(1, round(len(words) * held_out))
        return cls(Vocab.of(corpus.words), tuple(words[n:]), tuple(words[:n]))

    def batch(self, words: list[str] | tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Inputs and targets for some words, padded to the same length.

        For "emma", the input is `.emma` and the target is `emma.`: at every position, the
        next character. Padding targets are -1, which the loss ignores.
        """
        seqs = [self.vocab.sequence(w) for w in words]
        t = max(len(s) for s in seqs) - 1
        inputs = np.zeros((len(seqs), t), dtype=np.int64)
        targets = np.full((len(seqs), t), -1, dtype=np.int64)
        for row, seq in enumerate(seqs):
            inputs[row, : len(seq) - 1] = seq[:-1]
            targets[row, : len(seq) - 1] = seq[1:]
        return inputs, targets
