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

FABLES = {
    21: "Three Hundred Aesop's Fables, tr. George Fyler Townsend (1867)",
    28: "The Fables of Aesop, retold by Joseph Jacobs (1894)",
    11339: "Aesop's Fables, a new translation, tr. V. S. Vernon Jones (1912)",
    19994: "The Aesop for Children, illustrated by Milo Winter (1919)",
    13815: "The Talking Beasts, ed. Kate Douglas Wiggin and Nora Archibald Smith (1911)",
    36039: "The Giant Crab and Other Tales from Old India, retold by W. H. D. Rouse (1897)",
    62514: "Jataka Tales, retold by Ellen C. Babbitt (1912)",
    7518: "More Jataka Tales, retold by Ellen C. Babbitt (1922)",
    374: "Fantastic Fables, by Ambrose Bierce (1899)",
}
FAIRY_TALES = {
    5314: "Household Tales by the Brothers Grimm, tr. Margaret Hunt (1884)",
    27200: "Fairy Tales of Hans Christian Andersen",
    503: "The Blue Fairy Book, ed. Andrew Lang (1889)",
    540: "The Red Fairy Book (1890)",
    7277: "The Green Fairy Book (1892)",
    640: "The Yellow Fairy Book (1894)",
    5615: "The Pink Fairy Book (1897)",
    6746: "The Grey Fairy Book (1900)",
    641: "The Violet Fairy Book (1901)",
    2435: "The Crimson Fairy Book (1903)",
    3282: "The Brown Fairy Book (1904)",
    3027: "The Orange Fairy Book (1906)",
    27826: "The Olive Fairy Book (1907)",
    3454: "The Lilac Fairy Book (1910)",
}
PLAYS = {100: "The Complete Works of William Shakespeare"}
BOOKS = FABLES | FAIRY_TALES | PLAYS

# In the complete works, but not plays.
POEMS = {"THE SONNETS", "VENUS AND ADONIS", "THE RAPE OF LUCRECE", "A LOVER’S COMPLAINT",
         "THE PASSIONATE PILGRIM", "THE PHOENIX AND THE TURTLE"}  # fmt: skip

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
    book = text[start.end() : end.start()].replace("\r\n", "\n")
    return re.split(r"^End of (?:the )?Project Gutenberg", book, flags=re.M)[0]  # an older ending


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


# ── Books with a title line before each story ─────────────────────────────────

# Not stories: the front and back of a book, and a book's own title.
FRONT = re.compile(
    r"^(preface|contents|list of illustrations|illustrations|plates|dedication|introduction|"
    r"foreword|publisher.s note|footnotes|notes?|index|the end|to the friendly reader|"
    r"produced by|transcriber|warning|page)\b|fairy book$",
    re.I,
)
TITLE_PAGE = re.compile(
    r"\bcopyright\b|all rights reserved|\b(?:press|published)\b[\s\S]{0,40}?1[89]\d\d", re.I
)
PART = re.compile(r"^(?:chapter\s+|part\s+)?[IVXLC]+\.?$", re.I)  # a numbered part of a story
OPENING_NOTE = re.compile(  # "From the Danish.", "Sicilianische Mahrchen von Laura Gonzenbach..."
    r"^(?:Translated |Adapted |Taken )?[Ff]rom the\b|^(?:Translated|Adapted)\b|"
    r"\b(?:Leipzig|Paris|Barcelona|Berlin|Editeur|Verlag|Libreria)\b|\b1[6-9]\d\d\b|"
    r"M[aä]h?rchen|Murchen|Contes|Cuentos|\bSagen\b|Slavonic story"
)
FOOTNOTE = re.compile(r"^(?:\(\d+\)|\[\d+\]|\d+ \(return\)|\[Footnote|\[[^\]]*\]$)")
SOURCE = re.compile(  # "Adapted from the Portuguese.", "(Grimm.)", "West Highland Tales."
    r"^[(\[]?(?:Adapted|From|Translated|Taken|Told|Collected|Retold|After)\b|"
    r"\b(?:Grimm|Asbjornsen|Moe|d.Aulnoy|Perrault|Bureau|Maerchen|Contes|Magazine|Journal|"
    r"Sagas?|Folk-?[Tt]ales|Tales)\b\W*$"
)


def headings(lines: list[str]) -> list[int]:
    """Lines that look like a story's title: short, with two blank lines above and one below.

    Stories' own lines can look like that too (a song's last line, a line after a picture), so
    a title can't end like a sentence, start in lower case, be a footnote, or be speech.
    """
    found = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) > 70 or i < 2 or lines[i - 1].strip() or lines[i - 2].strip():
            continue
        if i + 1 == len(lines) or lines[i + 1].strip() or PART.match(s):
            continue
        if re.match(r"^\(?\[?\d|^[a-z]", s) or s.rstrip("’'\"”").endswith((".", ",", ";", ":")):
            continue
        if re.search(r"[’”'\"] [a-z]", s) or (s.endswith(("!", "?")) and ", " in s):
            continue  # ...,’ he replied   |   And the days went by, in life as in dreams!
        found.append(i)
    return found


def title_of(line: str) -> str:
    title = re.sub(r"\(\d+\)|\[\d+\]|_", "", line.strip())
    title = re.sub(r"\s{3,}.*$", "", title)  # a note after the title, set off by spaces
    title = re.sub(r"\s*\([^)]*\)?\s*$", "", title)  # (From the Russian)
    title = re.sub(r"^[IVXLC]+\.\s+", "", title)  # VII. The Eagle's Nest
    title = title.strip(" '\"‘’“”.")
    return title_case(title) if title.isupper() else title


def lines_or_prose(block: str) -> list[str]:
    """Paragraphs, unwrapped; but verse (short lines, each a capital) keeps its lines."""
    out = []
    for para in re.split(r"\n[ \t]*\n", block):
        lines = [line.strip() for line in para.splitlines() if line.strip()]
        verse = len(lines) > 1 and all(len(ln) < 55 and ln[0].isupper() for ln in lines)
        if lines:
            out.append(("\n" if verse else " ").join(lines))
    return out


def titled_stories(text: str, smallest: int) -> list[str]:
    """Every story in a book that puts a title line before each one, footnotes and sources
    taken out. Lang's Fairy Books, Andersen, and several books of fables are laid out so."""
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)  # captions, even over several lines
    text = re.sub(r"\n[ \t]*[IVXLC]+\.?[ \t]*\n", "\n\n", text)  # "II" above a title: a gap
    lines = text.split("\n")
    book = title_of(next(line for line in lines if line.strip())).lower()  # the title page's
    heads = headings(lines)
    out = []
    for a, b in zip(heads, [*heads[1:], len(lines)], strict=True):
        title = title_of(lines[a])
        if not title or FRONT.search(title) or title.lower() == book:
            continue
        told = re.split(r"\n\s*Footnotes?:?\s*\n", "\n".join(lines[a + 1 : b]))[0]
        written = [line for line in told.splitlines() if line.strip()]
        if TITLE_PAGE.search(told) or sum(
            bool(re.search(r"\s\d+$", ln)) for ln in written
        ) * 2 > len(written):
            continue  # a title page, or a table of contents with its page numbers
        paras = [p for p in lines_or_prose(told) if not FOOTNOTE.match(p) and not PART.match(p)]
        while (
            paras
            and len(paras[-1]) < 200
            and (SOURCE.search(paras[-1]) or re.fullmatch(r"[A-ZÆŒ .'’-]{3,40}", paras[-1]))
        ):
            paras.pop()  # where the story came from, or who wrote it (JOHN GAY)
        if paras and len(paras[0]) < 250 and OPENING_NOTE.search(paras[0]):
            paras.pop(0)  # where it came from, again, but at the start
        paras = [re.sub(r"\(\d+\)|\[\d+\]|\(\d+$|\s*\[Footnote[^\]]*\]", "", p) for p in paras]
        if sum(map(len, paras)) >= smallest:
            out.append(story(title, paras))
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
        + [f for n in (13815, 36039, 62514, 7518, 374) for f in titled_stories(books[n], 150)]
    )
    tales = hunt(books[5314]) + [
        t for n in FAIRY_TALES if n != 5314 for t in titled_stories(books[n], 800)
    ]
    plays = [w for w in works(books[100]) if w not in POEMS]
    corpora = {
        "fables": fables,
        "fairytales": tales,
        "shakespeare": [s for play in plays for s in scenes(books[100], play)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, docs in corpora.items():
        text = SEPARATOR.join(plain(d) for d in docs) + "\n"
        (OUT / f"{name}.txt").write_text(text, encoding="ascii")
        print(f"{name}: {len(docs)} documents, {len(text) / 1000:.0f} KB")


if __name__ == "__main__":
    main()
