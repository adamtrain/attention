"""Chapter: text becomes numbers."""

from __future__ import annotations

from collections.abc import Iterator

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import bpe, viz
from ..views import display, piece_chips, tiles, token_chip
from .lab import Lab
from .stage import Frame, Stage, hold

MERGES = 40


def run(stage: Stage, lab: Lab) -> None:
    c, vocab = lab.corpus, lab.vocab
    stage.say(
        "A neural network can only do arithmetic, so the first job is to turn text into "
        f"numbers. Here are some of the {len(c.words):,} {c.plural} your model will learn from:"
    )
    stage.show(word_grid(lab, stage.width))
    stage.wait()

    stage.say(
        "Each character gets a number, its [b]token ID[/b]. There are "
        f"{len(vocab)} tokens: {len(vocab) - 1} letters plus `.`, which marks where a word "
        "starts and ends."
    )
    stage.show(tiles(vocab, list(range(len(vocab))), stage.width))
    stage.wait()

    word = c.example
    stage.say(f"So this is {display(word)}, as the model sees it:")
    ids = vocab.sequence(word)
    stage.play(spell(lab, ids, stage.width), fps=14)
    stage.wait()

    stage.say(
        "The model's whole job is to [b]predict the next token[/b]. That makes every word "
        f"a little set of quizzes. {display(word)} is {len(word) + 1} of them:"
    )
    stage.play(quizzes(lab, ids), fps=10)
    stage.say(
        "When the answer is `.`, the word is finished. That's how the model will know when to stop."
    )
    stage.wait("see how the big models do it")

    stage.say(
        "One way your model differs from the big ones: it reads one letter at a time. Real "
        "language models read chunks of words, and they pick their chunks with a simple trick "
        "called [b]byte-pair encoding[/b]. Find the pair of neighboring tokens that turns up most "
        f"often, glue it into one new token, and repeat. Here it is, running on your {c.plural}:"
    )
    merges = bpe.learn(c.words, MERGES)
    stage.play(merging(lab, merges), fps=4)
    before, after = bpe.average_length(c.words, []), bpe.average_length(c.words, merges)
    stage.say(
        f"After {len(merges)} merges, the average {c.noun} takes {after:.1f} tokens instead of "
        f"{before:.1f}. GPT-3's tokenizer did this 50,000 times, on text from the internet, so "
        "chunks like ` the`, `ing` and `tion` are single tokens to it. That's also why language "
        "models can be clumsy at spelling, or at counting the r's in “strawberry”: they never "
        "see the letters, only the chunks."
    )
    stage.note(
        "Your model sticks to single letters. That keeps it tiny, and with only a few hundred "
        "words to learn from, every token gets plenty of examples."
    )


def word_grid(lab: Lab, width: int) -> Table:
    words = [lab.corpus.words[int(i)] for i in lab.rng.choice(len(lab.corpus.words), 24, False)]
    col_w = max(len(w) for w in words) + 2
    cols = max(1, min(6, width // col_w))
    grid = Table.grid(padding=(0, 2))
    for _ in range(cols):
        grid.add_column(no_wrap=True)
    for start in range(0, len(words) - cols + 1, cols):
        grid.add_row(*(Text(display(w), style="italic") for w in words[start : start + cols]))
    return grid


def spell(lab: Lab, ids: list[int], width: int) -> Iterator[Frame]:
    for n in range(1, len(ids) + 1):
        yield Text("\n").join(tiles(lab.vocab, ids[:n], width))


def quizzes(lab: Lab, ids: list[int]) -> Iterator[Frame]:
    vocab = lab.vocab
    rows: list[Text] = []
    width = 4 * (len(ids) - 1)
    for n in range(1, len(ids)):
        line = Text(no_wrap=True)
        context = Text(no_wrap=True)
        for token in ids[:n]:
            context.append_text(token_chip(vocab, token))
            context.append(" ")
        context.align("left", width)
        line.append_text(context)
        line.append(" → ", style=viz.FAINT)
        line.append_text(token_chip(vocab, ids[n]))
        rows.append(line)
        yield Text("\n").join(rows)
    yield hold(Text("\n").join(rows), 0.3)


def merging(lab: Lab, merges: list[bpe.Merge]) -> Iterator[Frame]:
    """Byte-pair encoding, one merge at a time, on a few of the words."""
    c = lab.corpus
    others = [w for w in c.words if w != c.example and len(w) >= 7]
    shown = [c.example, *(others[int(i)] for i in lab.rng.choice(len(others), 2, replace=False))]
    seqs = [list(w) for w in c.words]
    letters = len(set("".join(c.words)))
    start = sum(map(len, seqs)) / len(seqs)
    pieces = {w: list(w) for w in shown}

    def frame(done: int) -> Group:
        latest = merges[done - 1] if done else None
        head = Text.assemble(
            ("merge ", viz.FAINT), (f"{done:>2}", "bold"), (f" of {len(merges)}   ", viz.FAINT)
        )
        if latest:
            head.append_text(piece_chips([latest.left], lab.vocab))
            head.append(" + ", style=viz.FAINT)
            head.append_text(piece_chips([latest.right], lab.vocab))
            head.append(" → ", style=viz.FAINT)
            head.append_text(piece_chips([latest.token], lab.vocab))
            head.append(f"   seen {latest.count} times", style=viz.FAINT)
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="italic", no_wrap=True)
        rows.add_column(no_wrap=True)
        for w in shown:
            rows.add_row(display(w), piece_chips(pieces[w], lab.vocab))
        average = sum(map(len, seqs)) / len(seqs)
        stats = Text.assemble(
            ("vocabulary ", viz.FAINT), (f"{letters} → {letters + done}", "bold"), (" tokens", viz.FAINT),
            ("     average tokens per word ", viz.FAINT), (f"{start:.1f} → {average:.1f}", "bold"),
        )  # fmt: skip
        newest = Text("newest  ", style=viz.FAINT, no_wrap=True)
        newest.append_text(
            piece_chips([m.token for m in merges[max(0, done - 6) : done]][::-1], lab.vocab)
        )
        return Group(head, Text(""), rows, Text(""), stats, newest)

    yield hold(frame(0), 1.5)
    for done, m in enumerate(merges, 1):
        seqs = [bpe.merge_pair(seq, m.left, m.right) for seq in seqs]
        for w in shown:
            pieces[w] = bpe.merge_pair(pieces[w], m.left, m.right)
        yield frame(done), 0.35 if done <= 8 else 0.15
    yield hold(frame(len(merges)), 0.5)
