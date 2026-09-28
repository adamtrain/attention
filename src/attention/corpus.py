"""The text a model learns from: documents, the tokens they become, and stretches of them."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from .tokenizer import END, Tokenizer, plain

SEPARATOR = "<|endoftext|>"  # on a line of its own, between the documents in a corpus file
VOCAB = 2048  # tokens in all: the special ones, single characters, and merges
HELD_OUT = 0.1
SMALLEST = 20_000  # characters: a few pages


@dataclass(frozen=True, slots=True)
class Niche:
    """Some of the documents, all about one thing: something to fine-tune a model toward."""

    pattern: str  # a regular expression that finds the thing
    label: str  # "fables about the Wolf"
    titles: bool = False  # look only at a document's first line

    def has(self, text: str) -> bool:
        where = text.split("\n", 1)[0] if self.titles else text
        return re.search(self.pattern, where, re.M) is not None


@dataclass(frozen=True, slots=True)
class Corpus:
    key: str
    title: str  # "Fables"
    noun: str  # "fable"
    plural: str  # "fables"
    blurb: str  # what's in it
    documents: tuple[str, ...]
    example: str  # a passage to take apart in the tour
    probe: str  # a beginning to watch the model's predictions for while it trains
    niche: Niche
    titled: bool  # each document starts with its title
    play: bool = False  # documents are speeches: a SPEAKER: line, then what they say

    def family(self, document: str) -> str:
        """What groups a document with its relatives: its title, if it has one.

        The same fable told in three translations is one family. When one version is held back
        to test the model, they all are, or the test would be partly an exam it had seen.
        """
        return document.split("\n", 1)[0].lower() if self.titled else document

    @property
    def characters(self) -> int:
        return sum(len(d) for d in self.documents)


BUILT_IN: dict[str, dict] = {
    "fables": {
        "title": "Fables", "noun": "fable", "plural": "fables",
        "blurb": "fables of Aesop, in four translations",
        "example": "The Fox and the Grapes\n\n"
                   "A hungry Fox saw some fine bunches of Grapes hanging from a vine",
        "probe": "The Fox and the",
        "niche": Niche(r"\bWolf\b", "fables about the Wolf", titles=True),
        "titled": True,
    },
    "fairytales": {
        "title": "Fairy tales", "noun": "tale", "plural": "fairy tales",
        "blurb": "tales collected by the Brothers Grimm",
        "example": "Hansel and Grethel\n\nHard by a great forest dwelt a poor wood-cutter with "
                   "his wife and his two children. The boy was called Hansel and the girl Grethel.",
        "probe": "There was once a",
        "niche": Niche(r"\b[Pp]rincess", "tales with a princess in them"),
        "titled": True,
    },
    "shakespeare": {
        "title": "Shakespeare", "noun": "scene", "plural": "scenes",
        "blurb": "scenes from eight of Shakespeare's plays",
        "example": "JULIET:\nO Romeo, Romeo, wherefore art thou Romeo?\n"
                   "Deny thy father and refuse thy name.",
        "probe": "I thank you, my good",
        "niche": Niche(r"^HAMLET:", "scenes where Hamlet speaks"),
        "titled": False,
        "play": True,
    },
}  # fmt: skip


def documents(text: str) -> tuple[str, ...]:
    """Split a corpus into documents: at <|endoftext|> lines, or else at gaps of blank lines."""
    text = plain(text)
    if SEPARATOR in text:
        parts = text.split(SEPARATOR)
    else:
        parts = re.split(r"\n\s*\n\s*\n", text)
        if len(parts) < 20:  # one long text: cut it up at paragraphs instead, a page or so each
            parts, page = [], ""
            for para in re.split(r"\n\s*\n", text):
                page = f"{page}\n\n{para}" if page else para
                if len(page) > 2_000:
                    parts.append(page)
                    page = ""
            parts.append(page)
    return tuple(d for p in parts if (d := p.strip("\n")).strip())


def built_in(key: str) -> Corpus:
    meta = BUILT_IN[key]
    text = files("attention").joinpath("corpora", f"{key}.txt").read_text(encoding="ascii")
    docs = documents(text)
    blurb = f"{len(docs):,} {meta['blurb']}" if key != "shakespeare" else str(meta["blurb"])
    return Corpus(key=key, documents=docs, **{**meta, "blurb": blurb})  # ty: ignore[invalid-argument-type]


def common_topic(docs: tuple[str, ...]) -> Niche:
    """A name that turns up in a decent handful of the documents, but not most of them."""
    counts: Counter[str] = Counter()
    for d in docs:
        counts.update(set(re.findall(r"(?<=[a-z,;] )[A-Z][a-z]{2,}\b", d)))
    few, most = max(2, len(docs) // 20), len(docs) // 3
    fits = [(n, w) for w, n in counts.items() if few <= n <= most]
    word = max(fits)[1] if fits else (counts.most_common(1) or [("The", 0)])[0][0]
    return Niche(rf"\b{word}\b", f"passages that mention {word}")


def from_file(path: Path) -> Corpus:
    docs = documents(path.read_text(encoding="utf-8", errors="replace"))
    chars = sum(len(d) for d in docs)
    if chars < SMALLEST:
        raise ValueError(
            f"{path.name} has {chars:,} characters of text; the model needs at least "
            f"{SMALLEST:,} (a few pages), and a few hundred thousand is much better."
        )
    middle = docs[len(docs) // 2]
    example = middle[:120].rsplit(" ", 1)[0] if len(middle) > 120 else middle
    return Corpus(
        key=path.stem,
        title=path.stem.replace("_", " ").replace("-", " ").capitalize(),
        noun="passage",
        plural="passages",
        blurb=f"{len(docs):,} passages from {path.name}",
        documents=docs,
        example=example,
        probe=" ".join(example.split()[:3]),
        niche=common_topic(docs),
        titled=False,
    )


def load(name: str) -> Corpus:
    """A built-in corpus by name, or a text file."""
    if name in BUILT_IN:
        return built_in(name)
    path = Path(name).expanduser()
    if path.is_file():
        return from_file(path)
    raise ValueError(f"No corpus called {name!r}. Try {', '.join(BUILT_IN)}, or a text file.")


# ── Tokens ────────────────────────────────────────────────────────────────────


def stream(tokenizer: Tokenizer, docs: tuple[str, ...] | list[str]) -> np.ndarray:
    """Documents end to end, as the model reads them: <|endoftext|> before each, and at the end."""
    ids: list[int] = []
    for doc in docs:
        ids.append(END)
        ids.extend(tokenizer.encode(doc))
    ids.append(END)
    return np.array(ids, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class Dataset:
    tokenizer: Tokenizer
    train_docs: tuple[str, ...]
    val_docs: tuple[str, ...]  # held back, to check the model isn't just memorizing
    train: np.ndarray  # every training document's tokens, one after another
    val: np.ndarray

    @classmethod
    def split(
        cls,
        corpus: Corpus,
        rng: np.random.Generator,
        vocab: int = VOCAB,
        held_out: float = HELD_OUT,
        tokenizer: Tokenizer | None = None,
    ) -> Dataset:
        """Hold back a tenth of the documents, then learn a tokenizer from the rest.

        Pass a `tokenizer` to use that one instead: a saved model's, say, to keep its tokens.
        """
        families = sorted({corpus.family(d) for d in corpus.documents})
        n = max(1, round(len(families) * held_out))
        held = {families[int(i)] for i in rng.choice(len(families), n, replace=False)}
        train = tuple(d for d in corpus.documents if corpus.family(d) not in held)
        val = tuple(d for d in corpus.documents if corpus.family(d) in held)
        tokenizer = tokenizer or Tokenizer.train("\n\n".join(train), vocab)
        return cls(tokenizer, train, val, stream(tokenizer, train), stream(tokenizer, val))

    def windows(
        self, rng: np.random.Generator, count: int, length: int, held_back: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random stretches of text, `length` tokens long, and the token after each position.

        That's all pretraining data is: for every position, the input is the text so far, and
        the target is the next token. One stretch of 128 tokens holds 128 quizzes.
        """
        data = self.val if held_back else self.train
        starts = rng.integers(0, len(data) - length, count)
        return pieces(data, starts, length)

    def fixed(
        self, count: int, length: int, held_back: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evenly spaced stretches, the same every time: for measuring progress fairly."""
        data = self.val if held_back else self.train
        starts = np.linspace(0, len(data) - length - 1, count).astype(np.int64)
        return pieces(data, starts, length)


def pieces(data: np.ndarray, starts: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    w = data[starts[:, None] + np.arange(length + 1)]
    return w[:, :-1], w[:, 1:]
