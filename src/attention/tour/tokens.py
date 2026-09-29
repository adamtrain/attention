"""Chapter: text becomes tokens."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from itertools import accumulate

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..generate import prompt
from ..tokenizer import ALPHABET, SPECIAL, chunks, merge_pair
from ..views import chip, chips, label, tiles
from .lab import Lab
from .stage import Frame, Stage, hold

SHOWN = 40  # merges to watch one at a time


def run(stage: Stage, lab: Lab) -> None:
    c, tok, data = lab.corpus, lab.tokenizer, lab.data
    stage.say(
        "A neural network can only do arithmetic, so the first job is to turn text into "
        f"numbers. Here's a little of what your model will learn from, {c.blurb}:"
    )
    stage.show(sample(lab, stage.width))
    stage.wait()

    chars = sum(len(d) for d in data.train_docs)
    stage.say(
        "The simplest way would be a number for each character. But then every word has to be "
        f"spelled out one letter at a time, and the model has to read {chars:,} characters one "
        "by one. Real language models read chunks of text called [b]tokens[/b], and they choose "
        "their chunks with a simple trick called [b]byte-pair encoding[/b] (BPE)."
    )
    stage.say(
        f"Start with single characters: yours has {len(ALPHABET)}, every letter, digit and "
        "symbol on an English keyboard, plus the space and the newline. Find the pair of "
        "neighbors that turns up most often in all of the text, and glue it into one new token. "
        "Then do it again, and again. Common pairs of letters become tokens first, then common "
        f"syllables, then whole words. Here it is, running on your {c.plural}:"
    )
    stage.play(lambda: merging(lab, stage.width), fps=4, start="watch it merge")
    stage.say(
        f"{len(tok.merges):,} merges later, the vocabulary has {len(tok):,} tokens. The later "
        "merges are rarer pairs, so each saves less than the one before. That's your model's "
        "[b]tokenizer[/b]: it was trained once, before the model, and from now on every piece "
        "of text goes through it, in and out. Here's what it knows:"
    )
    stage.show(vocabulary(lab, stage.width))
    stage.wait()

    ids = prompt(tok, c.example)
    stage.say("So this is your example passage, as token ids:")
    stage.show(*tiles(tok, ids[1:], stage.width), gap=True)
    stage.say(pieces_note(lab, ids[1:]))
    stage.wait()

    tokens = len(data.train) - len(data.train_docs) - 1
    words = sum(len(d.split()) for d in data.train_docs)
    stage.say(
        f"The {c.plural} it will train on come to {chars:,} characters and {tokens:,} tokens: "
        f"about {chars / tokens:.1f} characters, or {tokens / words:.1f} tokens per word. Real "
        "tokenizers have far bigger vocabularies (GPT-4's has about 100,000 tokens, Llama 3's "
        "128,000), so more words are whole tokens: in English a token is about four "
        "characters, three-quarters of a word. That's why chatbots count tokens, not words, and "
        "why their limits and prices are measured in tokens. And they start from the 256 "
        "possible bytes instead of a keyboard's characters, so they can read any text at all: "
        "an emoji or a Chinese character is a few bytes, so at worst a few tokens."
    )
    seam = boundary(lab)
    while len(seam) > 5 and sum(len(label(tok, t)) + 3 for t in seam) - 1 > stage.width:
        seam = seam[1:-1]  # a token off each end, keeping the seam in the middle
    stage.say(
        "Between documents goes a [b]special token[/b], `<|endoftext|>`, token 0. It isn't "
        "text; it's a marker meaning “a new document starts here”. This is the seam between two "
        f"{c.plural} in the training text:"
    )
    stage.show(chips(tok, seam, " "))
    stage.say(
        "When your model writes something new, it starts right after one of these. Two more "
        "special tokens, `<|user|>` and `<|assistant|>`, are set aside and not used until the "
        "chat chapter. Real models reserve tokens like these too: Llama 3 set aside 256 of them."
    )
    stage.wait("see the quizzes")

    stage.say(
        "The model's whole job is to [b]predict the next token[/b]. That makes every position "
        "in the text a little quiz: here's everything so far, what comes next?"
    )
    stage.play(
        lambda: quizzes(lab, ids[: min(len(ids), 13)], stage.width), fps=10, start="see the quizzes"
    )
    context = lab.model.config.context
    stage.say(
        f"Your model reads at most {context} tokens at once: that's its [b]context window[/b]. "
        f"To train, the text is cut into stretches of {context} tokens, and each stretch is "
        f"{context} quizzes, all graded at once."
    )
    stage.note(
        "Reading chunks is also why chatbots can be clumsy at spelling, or at counting the r's "
        "in “strawberry”: they never see the letters, only the tokens."
    )


def sample(lab: Lab, width: int, most: int = 9) -> Group:
    """A short document from the corpus, the way it's written."""
    docs = sorted(lab.data.train_docs, key=len)
    doc = docs[len(docs) // 4] if not lab.corpus.play else lab.data.train_docs[0]
    lines = []
    for para in doc.split("\n"):
        while len(para) > width - 4:
            cut = para.rfind(" ", 0, width - 4)
            lines.append(para[:cut])
            para = para[cut + 1 :]
        lines.append(para)
    shown = [Text(line, style="italic") for line in lines[:most]]
    if len(lines) > most:
        shown.append(Text("…", style=viz.FAINT))
    return Group(*shown)


def merging(lab: Lab, width: int) -> Iterator[Frame]:
    """Byte-pair encoding, one merge at a time, on the example, all the way to the last merge.

    The first merges go slowly enough to follow; then faster and faster, several to a frame.
    """
    tok = lab.tokenizer
    merges = tok.merges
    pieces = [list(ch) for ch in chunks(lab.corpus.example)]
    counts = Counter(chunks("\n\n".join(lab.data.train_docs)))
    chars = sum(len(w) * n for w, n in counts.items())  # tokens before any merge, one a letter
    joined = [0, *accumulate(m.count for m in merges)]  # each merge makes that many pairs one
    start = len(SPECIAL) + len(ALPHABET)
    digits = len(f"{len(merges):,}")

    def frame(done: int) -> Group:
        latest = merges[done - 1] if done else None
        head = Text.assemble(
            ("merge ", viz.FAINT), (f"{done:>{digits},}", "bold"),
            (f" of {len(merges):,}   ", viz.FAINT),
        )  # fmt: skip
        if latest:
            head.append_text(piece(latest.left, tok))
            head.append(" + ", style=viz.FAINT)
            head.append_text(piece(latest.right, tok))
            head.append(" → ", style=viz.FAINT)
            head.append_text(piece(latest.token, tok))
            head.append(f"   seen {latest.count:,} times", style=viz.FAINT)
        flat = [p for chunk in pieces for p in chunk]
        shown = tiles(tok, [tok.ids[p] for p in flat], width, numbers=False)
        stats = Text.assemble(
            ("vocabulary ", viz.FAINT), (f"{start + done:,}", "bold"), (" tokens", viz.FAINT),
            ("     characters per token ", viz.FAINT),
            (f"{chars / (chars - joined[done]):.2f}", "bold"),
        )  # fmt: skip
        return Group(head, Text(""), *shown, Text(""), stats)

    yield frame(0)
    done = 0
    while done < len(merges):
        step = 1 if done < SHOWN else done // SHOWN  # speeding up, more merges to a frame
        for m in merges[done : done + step]:
            pieces = [merge_pair(p, m.left, m.right) if m.left in p else p for p in pieces]
        done = min(done + step, len(merges))
        seconds = 0.5 if done <= 8 else 0.2 if done <= SHOWN else 0.04
        yield frame(done), (seconds if done < len(merges) else 1.0)


def piece(text: str, tok) -> Text:
    """A piece of text as a tile, colored as its token will be (every piece is one, by the end)."""
    return chip(tok, tok.ids[text])


def vocabulary(lab: Lab, width: int) -> Table:
    tok = lab.tokenizer
    merged = list(range(tok.first_merge, len(tok)))
    longest = sorted(merged, key=lambda i: (-len(tok.pieces[i]), i))[:8]
    letters = [tok.ids[ch] for ch in "\n abcABC.,!"]
    characters = f"{len(ALPHABET)} characters"
    room = width - len(characters) - 2  # beside the widest label
    more = ("  …", viz.FAINT)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=viz.FAINT, no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    grid.add_row("special", chips(tok, range(len(SPECIAL)), " "))
    grid.add_row(characters, Text.assemble(chips(tok, fitting(tok, letters, room - 3), " "), more))
    grid.add_row(
        "first merges", Text.assemble(chips(tok, fitting(tok, merged[:8], room - 3), " "), more)
    )
    last = fitting(tok, merged[-6:][::-1], room - 3)[::-1]
    grid.add_row("last merges", Text.assemble(("…  ", viz.FAINT), chips(tok, last, " ")))
    grid.add_row("longest", chips(tok, fitting(tok, longest, room), " "))
    return grid


def fitting(tok, ids: list[int], room: int) -> list[int]:
    """As many of these tokens as fit in `room` columns, as tiles with a space between."""
    used, out = 0, []
    for token in ids:
        used += len(label(tok, token)) + 2 + (1 if out else 0)
        if used > room:
            break
        out.append(token)
    return out


def pieces_note(lab: Lab, ids: list[int]) -> str:
    """What to notice about how the example was split, worked out from the example itself."""
    tok = lab.tokenizer
    words: list[list[int]] = []
    for token in ids:
        text = tok.pieces[token]
        if text.startswith(" ") or not words or not text[:1].isalpha():
            words.append([token])
        else:
            words[-1].append(token)
    split = max(words, key=len)
    whole = next(
        (
            w[0]
            for w in words
            if len(w) == 1 and tok.pieces[w[0]].startswith(" ") and len(tok.pieces[w[0]]) > 3
        ),
        None,
    )
    parts = ", ".join(f"`{label(tok, t)}`" for t in split)
    word = "".join(tok.pieces[t] for t in split).strip()
    note = (
        "A few things to notice. Most tokens start with a space (shown as ·): words are chunked "
        "along with the space in front of them"
    )
    if whole is not None:
        note += f", so `{label(tok, whole)}` is one token, space and all"
    note += (
        ". A word at the start of a line has no space before it, so it's a different token "
        "from the same word mid-sentence, and capitals count too. "
    )
    if len(split) > 1:
        note += (
            f"Common words get a token of their own; rarer ones are spelled out in pieces, like "
            f"“{word}”: {parts}."
        )
    return note


def boundary(lab: Lab, around: int = 6) -> list[int]:
    stream = lab.data.train
    ends = [i for i, t in enumerate(stream) if t == 0 and around < i < len(stream) - around]
    at = ends[len(ends) // 2]
    return [int(t) for t in stream[at - around : at + around + 1]]


def quizzes(lab: Lab, ids: list[int], width: int) -> Iterator[Frame]:
    tok = lab.tokenizer
    rows: list[Text] = []
    widths = [len(label(tok, t)) + 2 for t in ids]
    while len(ids) > 3 and sum(widths[: len(ids) - 1]) + len(ids) + 3 + max(widths[1:]) > width:
        ids, widths = ids[:-1], widths[:-1]  # the last line, the longest, has to fit
    room = sum(widths[:-1]) + len(ids)
    for n in range(1, len(ids)):
        line = Text(no_wrap=True)
        context = Text(no_wrap=True)
        for token in ids[:n]:
            context.append_text(chip(tok, token))
            context.append(" ")
        context.align("left", room)
        line.append_text(context)
        line.append(" → ", style=viz.FAINT)
        line.append_text(chip(tok, ids[n]))
        rows.append(line)
        yield Text("\n").join(rows)
    yield hold(Text("\n").join(rows), 0.3)
