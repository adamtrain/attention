"""Saving your model to disk, and loading it back."""

from __future__ import annotations

import json
import os
import zipfile
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
    tmp = unfinished(path)
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)  # ty: ignore[invalid-argument-type]
    tmp.replace(path)
    return path


def unfinished(path: Path) -> Path:
    """Where a save is written first, before it replaces the real file in one go."""
    return path.with_name(path.name + ".tmp.npz")


def is_model(path: Path) -> bool:
    """Is this a model attention saved? (It has attention's card inside.)"""
    try:
        with np.load(path, allow_pickle=False) as f:
            return "meta" in f.files and "card" in json.loads(str(f["meta"]))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def leftovers(extra: Path | None = None) -> list[Path]:
    """Everything attention has saved: models and unfinished saves in its folder, and `extra`."""
    found: list[Path] = []
    folder = home()
    if folder.is_dir():
        found += sorted(
            p for p in folder.iterdir()
            if p.is_file() and (p.name.endswith(".tmp.npz") or is_model(p))
        )  # fmt: skip
    if extra is not None:
        found += [p for p in (extra, unfinished(extra)) if p.is_file() and p not in found]
    return found


def remove(paths: list[Path]) -> bool:
    """Delete these files, then attention's folder if that left it empty. Was it removed?"""
    for path in paths:
        path.unlink(missing_ok=True)
    folder = home()
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
        return True
    return False


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
