"""Saving your model to disk, and loading it back."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .corpus import Corpus, Vocab
from .generate import sample
from .model import Config, Transformer


def home() -> Path:
    if custom := os.environ.get("ATTENTION_HOME"):
        return Path(custom).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "attention"


def default_path() -> Path:
    return home() / "model.npz"


@dataclass(frozen=True, slots=True)
class Card:
    """Everything about a trained model except its weights."""

    name: str
    corpus: str  # "dinosaurs", or a file's name
    title: str
    noun: str
    seed: int
    steps: int
    loss: float
    val_loss: float
    pair_loss: float  # what letter-pair statistics alone would score
    trained_at: str
    probe: str

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Saved:
    model: Transformer
    vocab: Vocab
    card: Card
    words: tuple[str, ...]  # what it trained on, to tell invented words from memorized ones

    @property
    def known(self) -> set[str]:
        return set(self.words)


def save(path: Path, saved: Saved) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": 1,
        "config": asdict(saved.model.config),
        "vocab": saved.vocab.chars,
        "card": asdict(saved.card),
        "words": "\n".join(saved.words),
    }
    arrays = {f"param/{k}": v for k, v in saved.model.params.items()}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)  # ty: ignore[invalid-argument-type]
    tmp.replace(path)
    return path


def load(path: Path) -> Saved:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        params = {k.removeprefix("param/"): f[k] for k in f.files if k.startswith("param/")}
    model = Transformer(Config(**meta["config"]), params)
    words = tuple(w for w in meta["words"].split("\n") if w)
    return Saved(model, Vocab(meta["vocab"]), Card(**meta["card"]), words)


def pick_name(model: Transformer, vocab: Vocab, rng: np.random.Generator, corpus: Corpus) -> str:
    """Let the model name itself: the first new word it invents that's a nice length."""
    known = set(corpus.words)
    fallback = ""
    for _ in range(300):
        word = sample(model, vocab, rng, temperature=0.6)
        tidy = not any(a == b == c for a, b, c in zip(word, word[1:], word[2:], strict=False))
        if 4 <= len(word) <= 10 and word not in known and tidy:
            return word.capitalize()
        if word and not fallback:
            fallback = word
    return (fallback or corpus.example).capitalize()
