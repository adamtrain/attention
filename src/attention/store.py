"""Saving your model to disk, and loading it back."""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import numpy as np

from .corpus import SEPARATOR, Corpus, stream
from .generate import Memory, write
from .model import Array, Config, Transformer
from .tokenizer import Tokenizer

FORMAT = 2


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


def chat_path(path: Path) -> Path:
    """Where the chat add-on for the model at `path` lives: right beside it."""
    return path.with_name(path.stem + ".chat.npz")


@dataclass(frozen=True, slots=True)
class Card:
    """Everything about a trained model except its weights."""

    name: str
    corpus: str  # "fables", or a file's name
    title: str
    noun: str
    plural: str
    seed: int
    steps: int
    loss: float
    val_loss: float
    pair_loss: float  # what counting token pairs alone would score
    trained_at: str
    probe: str

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Saved:
    model: Transformer
    tokenizer: Tokenizer
    card: Card
    documents: tuple[str, ...]  # what it trained on, to tell what it invents from what it copies

    @cached_property
    def memory(self) -> Memory:
        return Memory(stream(self.tokenizer, self.documents))


class Outdated(ValueError):
    """A model saved by an older version of attention, from before it read tokens."""


def save(path: Path, saved: Saved) -> Path:
    meta = {
        "format": FORMAT,
        "config": asdict(saved.model.config),
        "tokenizer": saved.tokenizer.to_json(),
        "card": asdict(saved.card),
    }
    text = f"\n{SEPARATOR}\n".join(saved.documents).encode("ascii")
    return write_npz(
        path,
        meta=np.array(json.dumps(meta)),
        text=np.frombuffer(text, dtype=np.uint8),
        **{f"param/{k}": v for k, v in saved.model.params.items()},
    )


def write_npz(path: Path, **arrays: Array) -> Path:
    """Write to a temporary file first, then swap it in, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unfinished(path)
    np.savez_compressed(tmp, **arrays)  # ty: ignore[invalid-argument-type]
    tmp.replace(path)
    return path


def unfinished(path: Path) -> Path:
    """Where a save is written first, before it replaces the real file in one go."""
    return path.with_name(path.name + ".tmp.npz")


def is_model(path: Path) -> bool:
    """Is this a file attention saved? (It has attention's card, or is a chat add-on.)"""
    try:
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            return "card" in meta or "adapter" in meta
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def leftovers(extra: Path | None = None) -> list[Path]:
    """Everything attention has saved: models, add-ons and unfinished saves, and `extra`."""
    found: list[Path] = []
    folder = home()
    if folder.is_dir():
        found += sorted(
            p for p in folder.iterdir()
            if p.is_file() and (p.name.endswith(".tmp.npz") or is_model(p))
        )  # fmt: skip
    if extra is not None:
        for p in (extra, unfinished(extra), chat_path(extra)):
            if p.is_file() and p not in found and (p != chat_path(extra) or is_model(p)):
                found.append(p)
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
        if meta.get("format") != FORMAT:
            raise Outdated(
                "it was made by an older version of attention, which read letters, not tokens"
            )
        params = {k.removeprefix("param/"): f[k] for k in f.files if k.startswith("param/")}
        text = f["text"].tobytes().decode("ascii")
    model = Transformer(Config(**meta["config"]), params)
    documents = tuple(text.split(f"\n{SEPARATOR}\n"))
    tokenizer = Tokenizer.from_json(meta["tokenizer"])
    return Saved(model, tokenizer, Card(**meta["card"]), documents)


def pick_name(
    model: Transformer, tokenizer: Tokenizer, rng: np.random.Generator, corpus: Corpus
) -> str:
    """Let the model name itself: the first name it invents that appears nowhere in its text."""
    known = {w.lower() for d in corpus.documents for w in re.findall(r"[A-Za-z]+", d)}
    fallback = ""
    for _ in range(60):
        for word in re.findall(
            r"\b[A-Z][a-z]{3,9}\b", write(model, tokenizer, rng, "", 0.8, limit=48)
        ):
            if word.lower() not in known:
                return word
            fallback = fallback or word
    return fallback or corpus.title.split()[0]
