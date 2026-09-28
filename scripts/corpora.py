"""Rebuild the built-in corpora in src/attention/corpora/ from public-domain books.

    uv run scripts/corpora.py [--cache DIR]

Downloads the books from Project Gutenberg (or reads them from --cache), keeps only the stories
and plays themselves, turns them into plain ASCII, and writes one file per corpus. Documents are
separated by a line holding only <|endoftext|>, the marker the model sees between them.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

from attention.tokenizer import plain

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "attention" / "corpora"
SEPARATOR = "\n<|endoftext|>\n"

BOOKS = {
    21: "Three Hundred Aesop's Fables, tr. George Fyler Townsend (1867)",
    28: "The Fables of Aesop, retold by Joseph Jacobs (1894)",
    11339: "Aesop's Fables, a new translation, tr. V. S. Vernon Jones (1912)",
    19994: "The Aesop for Children, illustrated by Milo Winter (1919)",
    5314: "Household Tales by the Brothers Grimm, tr. Margaret Hunt (1884)",
    100: "The Complete Works of William Shakespeare",
}

PLAYS = [
    "THE TRAGEDY OF ROMEO AND JULIET",
    "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK",
    "THE TRAGEDY OF MACBETH",
    "A MIDSUMMER NIGHT’S DREAM",
    "THE TRAGEDY OF JULIUS CAESAR",
    "THE TEMPEST",
    "TWELFTH NIGHT; OR, WHAT YOU WILL",
    "MUCH ADO ABOUT NOTHING",
]

SMALL = {"a", "an", "and", "the", "of", "in", "on", "to", "at", "by", "for", "with", "his",
         "her", "their", "its", "who", "that", "or", "as", "from", "into", "is", "was"}  # fmt: skip


def fetch(number: int, cache: Path | None) -> str:
    if cache is not None and (path := cache / f"pg{number}.txt").exists():
        return path.read_text(encoding="utf-8")
    url = f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"pg{number}.txt").write_text(text, encoding="utf-8")
    return text


def body(text: str) -> str:
    """Just the book: everything between Project Gutenberg's start and end markers."""
    start = re.search(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*$", text, re.M)
    end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*$", text, re.M)
    assert start and end
    return text[start.end() : end.start()].replace("\r\n", "\n")


def blocks(text: str) -> list[str]:
    """Pieces separated by three or more blank lines: how these books separate stories."""
    return [b.strip("\n") for b in re.split(r"\n[ \t]*(?:\n[ \t]*){3,}", text) if b.strip()]


def paragraphs(block: str) -> list[str]:
    """Prose paragraphs, each unwrapped onto one line."""
    out = []
    for para in re.split(r"\n[ \t]*\n", block):
        lines = [line.strip() for line in para.splitlines()]
        lines = [line for line in lines if line and not line.startswith("[Illustration")]
        if lines:
            out.append(" ".join(lines))
    return out


def title_case(title: str) -> str:
    words = title.lower().split()
    return " ".join(
        w if (i and w in SMALL) else "-".join(part.capitalize() for part in w.split("-"))
        for i, w in enumerate(words)
    )


def tidy(text: str) -> str:
    text = text.replace("_", "")  # italics
    text = re.sub(r"\[(?:\d+|\*)\]|\{\d+\}", "", text)  # footnote marks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def story(title: str, paras: list[str]) -> str:
    return tidy(title) + "\n\n" + "\n\n".join(tidy(p) for p in paras)


# ── Fables ────────────────────────────────────────────────────────────────────


def calm_opening(first: str, rest: str) -> str:
    """Townsend starts each fable in capitals ("A LION was..."): write them normally."""
    words = first.split(" ")
    for i, word in enumerate(words):
        letters = re.sub(r"[^A-Za-z]", "", word)
        if not letters or not letters.isupper() or (len(letters) == 1 and letters not in "AI"):
            break
        fixed = letters.capitalize() if i == 0 or letters.capitalize() in rest else letters.lower()
        words[i] = word.replace(letters, fixed)
    return " ".join(words)


def townsend(text: str) -> list[str]:
    start = re.search(r"^AESOP’S FABLES$", text, re.M)
    end = re.search(r"^FOOTNOTES$", text, re.M)
    assert start and end
    fables = []
    for block in blocks(text[start.end() : end.start()]):
        title, *paras = paragraphs(block)
        if paras:
            paras[0] = calm_opening(paras[0], " ".join(paras))
            fables.append(story(title_case(title), paras))
    return fables


def jacobs(text: str) -> list[str]:
    """Jacobs's retelling: ordinary titles, from The Cock and the Pearl to his closing HURRAH."""
    start = [m.start() for m in re.finditer(r"^The Cock and the Pearl$", text, re.M)][-1]
    end = re.search(r"^And this is the end of", text, re.M)
    assert end
    fables = re.sub(r"\n+(?:WARRA ?)+\n+", " ", text[start : end.start()])  # a picture's caption
    return [story(title, paras) for title, *paras in map(paragraphs, blocks(fables)) if paras]


def all_caps_stories(text: str, after: str) -> list[str]:
    """Books where each story is a title in capitals, then paragraphs (some indented morals)."""
    start = [m.end() for m in re.finditer(rf"^{re.escape(after)}$", text, re.M)][-1]
    out = []
    for block in blocks(text[start:]):
        title, *paras = paragraphs(block)
        if paras and title.isupper():
            out.append(story(title_case(title), paras))
    return out


# ── Fairy tales ───────────────────────────────────────────────────────────────


def hunt(text: str) -> list[str]:
    """Hunt's Grimm: each tale starts with its number (or "Legend" and a number) and its title."""
    tales: list[tuple[str, list[str]]] = []
    for block in blocks(text):
        first, *paras = paragraphs(block)
        if head := re.match(r"^(?:Legend )?\d+ (.+)$", first):
            tales.append((head.group(1), paras))
        elif tales and first != "CONTENTS":  # the second part of a tale told in parts
            tales[-1][1].extend(p for p in [first, *paras] if not re.match(r"^\w+ STORY$", p))
    return [story(title, paras) for title, paras in tales if paras]


# ── Plays ─────────────────────────────────────────────────────────────────────

SPEAKER = re.compile(r"^([A-Z][A-Z’'&,. -]*[A-Z])\.$")
DIRECTION = re.compile(r"\[_[^\]]*_\]|\[[^\]]*\]")


def works(text: str) -> list[str]:
    """The title of everything in the complete works, from its table of contents."""
    contents = re.search(r"^\s+Contents\n\n((?:    .+\n)+)", text, re.M)
    assert contents
    return [line.strip() for line in contents.group(1).splitlines()]


def scenes(text: str, play: str) -> list[str]:
    """Every scene of one play, as speeches: SPEAKER: then the lines, as the verse runs."""
    begin = [m.end() for m in re.finditer(rf"^{re.escape(play)}$", text, re.M)][-1]
    ends = [m.start() for t in works(text) for m in re.finditer(rf"^{re.escape(t)}$", text, re.M)]
    stop = min((e for e in ends if e > begin), default=len(text))
    out = []
    parts = re.split(r"^ ?SCENE [IVX]+\..*$", text[begin:stop], flags=re.M)  # some are indented
    for part in parts[1:]:
        speeches: list[tuple[str, list[str]]] = []
        for raw in part.splitlines():
            if re.match(r"^(ACT [IVX]+|EPILOGUE|PROLOGUE)\b|^ \S", raw):
                continue  # act headings, and stage directions (indented by one space; songs, more)
            line = DIRECTION.sub("", raw).replace("_", "").strip()
            if not line:
                continue  # a speech runs on past a stage direction, to the next speaker
            if m := SPEAKER.match(line):
                speeches.append((m.group(1), []))
            elif speeches:
                speeches[-1][1].append(line)
        spoken = [f"{who}:\n" + "\n".join(lines) for who, lines in speeches if lines]
        if len(spoken) >= 3:
            out.append("\n\n".join(spoken))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, help="keep (and reuse) the downloaded books here")
    args = parser.parse_args()
    books = {n: body(fetch(n, args.cache)) for n in BOOKS}

    fables = (
        townsend(books[21])
        + jacobs(books[28])
        + all_caps_stories(books[11339], "ÆSOP'S FABLES")
        + all_caps_stories(books[19994], "THE ÆSOP FOR CHILDREN")
    )
    corpora = {
        "fables": fables,
        "fairytales": hunt(books[5314]),
        "shakespeare": [s for play in PLAYS for s in scenes(books[100], play)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, docs in corpora.items():
        text = SEPARATOR.join(plain(d) for d in docs) + "\n"
        (OUT / f"{name}.txt").write_text(text, encoding="ascii")
        print(f"{name}: {len(docs)} documents, {len(text) / 1000:.0f} KB")


if __name__ == "__main__":
    main()
