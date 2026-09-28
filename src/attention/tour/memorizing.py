"""Chapter: overfitting. What happens when a model has too little to read, for too long."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..corpus import Dataset, stream
from ..generate import Memory, picks
from ..model import Config, Transformer
from ..tokenizer import END
from ..train import Trainer, evaluate
from ..views import running
from .lab import Lab
from .stage import Frame, Stage

SHARE = 0.05  # of the training documents the overfitting model gets to read
CONTEXT = 64  # shorter windows, to train faster
SECONDS = 20
FPS = 12
EVERY = 50  # steps between check-ups


@dataclass
class Run:
    """The model with too little to read, training, and what it recites along the way."""

    trainer: Trainer
    memory: Memory
    seen: list[int]  # the start of a document it trains on
    unseen: list[int]  # the start of one it never sees
    own: tuple[np.ndarray, np.ndarray]  # stretches of its own text, to measure it on
    val: list[tuple[int, float]] = field(default_factory=list)
    mine: list[tuple[int, float]] = field(default_factory=list)
    recited: list[int] = field(default_factory=list)
    guessed: list[int] = field(default_factory=list)

    @classmethod
    def start(cls, lab: Lab) -> Run:
        data, tok = lab.data, lab.tokenizer
        rng = lab.dice(7)
        count = max(8, round(len(data.train_docs) * SHARE))
        docs = tuple(
            data.train_docs[int(i)]
            for i in sorted(rng.choice(len(data.train_docs), count, replace=False))
        )
        little = Dataset(tok, docs, data.val_docs, stream(tok, docs), data.val)
        sizes = {
            k: getattr(lab.model.config, k)
            for k in ("width", "layers", "heads", "kv_heads", "hidden")
        }
        model = Transformer.create(Config(len(tok), CONTEXT, **sizes), rng)
        trainer = Trainer(model, little, lab.dice(8), steps=lab.budget.memorizing, decay=0.0)
        long_enough = [d for d in docs if len(tok.encode(d)) > 60] or list(docs)
        seen = [END, *tok.encode(opening(long_enough[0]))]
        other = min(data.val_docs, key=lambda d: abs(len(d) - len(long_enough[0])))
        unseen = [END, *tok.encode(opening(other))]
        own = little.fixed(16, CONTEXT, held_back=False)
        run = cls(trainer, Memory(little.train), seen, unseen, own)
        run.check()
        return run

    def check(self) -> None:
        tr = self.trainer
        self.val.append((tr.step_number, evaluate(tr.model, *self.trainer.data.fixed(16, CONTEXT))))
        self.mine.append((tr.step_number, evaluate(tr.model, *self.own)))
        self.recited = recite(tr.model, self.seen)
        self.guessed = recite(tr.model, self.unseen)

    def step(self) -> None:
        self.trainer.step()
        if self.trainer.step_number % EVERY == 0 or self.trainer.done:
            self.check()

    @property
    def copied(self) -> float:
        marks = self.memory.copied(self.seen + self.recited)[len(self.seen) :]
        return float(marks.mean()) if len(marks) else 0.0

    @property
    def best(self) -> tuple[int, float]:
        return min(self.val, key=lambda sv: sv[1])


def opening(doc: str, words: int = 3) -> str:
    """How a document begins: its title (or first line) and its first few words."""
    head, sep, rest = doc.partition("\n\n" if "\n\n" in doc[:100] else "\n")
    return head + sep + " ".join(rest.split(" ")[:words])


def recite(model: Transformer, start: list[int], most: int = 40) -> list[int]:
    """Carry on from `start`, cautiously (temperature 0.3), with the same dice every time."""
    rng = np.random.default_rng(0)
    return [p.token for p in picks(model, start, rng, temperature=0.3, limit=most)]


def run(stage: Stage, lab: Lab) -> None:
    c, tr = lab.corpus, lab.trainer
    n = max(8, round(len(lab.data.train_docs) * SHARE))
    stage.say(
        f"Why did your model train on all {len(lab.data.train_docs)} {c.plural}, with weight "
        "decay (more on that below), and stop when it did? Let's break the rules and see. Here's "
        f"a model the same size as yours, given just {n} of the {c.plural} to read, trained on "
        f"them for {lab.budget.memorizing:,} steps (about "
        f"{lab.budget.memorizing * tr.batch_size * CONTEXT / max(1, sum(len(lab.tokenizer.encode(d)) for d in lab.data.train_docs[:n])):.0f} "
        "passes over its text), with no weight decay. Your own model isn't touched."
    )
    stage.say(
        "Watch the two lines: [accent]its loss on its own text[/accent], and "
        f"[amber]its loss on held-back {c.plural}[/amber] it never sees. The green line is your "
        "model's held-back loss, for comparison. On the right: give it the first few tokens of "
        f"one of its own {c.plural}, and of one it never saw, and see how it carries on."
    )

    yours = tr.val_losses[-1][1] if tr.val_losses else lab.baselines.pairs
    big = Run.start(lab)

    def training() -> Iterator[Frame]:
        nonlocal big
        if big.trainer.step_number:  # a replay: a new model, from new random numbers
            big = Run.start(lab)
        return overfit(lab, big, yours, stage.width)

    stage.play(training, fps=FPS, start="break the rules", again="try another")

    at, low = big.best
    last = big.val[-1][1]
    uniform = lab.baselines.uniform
    worse = f", worse than the {uniform:.2f} of pure guessing" if last > uniform else ""
    if big.copied >= 0.8:
        recites = f"it now recites it, word for word ({big.copied:.0%} of what it wrote)"
    elif big.copied >= 0.2:
        recites = f"it recites whole stretches of it ({big.copied:.0%} of what it wrote)"
    else:
        recites = f"it has begun to recite bits of it ({big.copied:.0%} of what it wrote)"
    stage.say(
        f"On its own text it kept improving, down to {big.mine[-1][1]:.2f}. But its held-back "
        f"loss bottomed out at {low:.2f} around step {at}, then climbed to [b]{last:.2f}[/b]"
        f"{worse}. And given the start of one of its {c.plural}, {recites}."
    )
    mine = recite(lab.model, big.seen)
    stage.say("Your model, given the same start, carries on in its own words:")
    stage.show(
        *running(
            lab.tokenizer,
            big.seen + mine,
            stage.width - 4,
            lab.memory.copied(big.seen + mine),
            start=len(big.seen),
        )
    )
    stage.say(
        "The other model [b]memorized[/b] its text instead of learning what it has in common "
        "with everything else, so it got better at the test it had seen and worse at "
        "everything else. That's called [b]overfitting[/b], and it's why the dashboard kept an "
        f"eye on held-back text. Yours ended at {yours:.2f}. It had three defenses: far more "
        "text to learn from, fewer passes over it, and weight decay."
    )
    stage.say(
        "[b]Weight decay[/b] shrinks every weight a little toward zero, every step. A weight "
        "only stays big if the gradient keeps pushing it back up. Memorizing one particular "
        "passage takes weights that only that passage pushes on, now and then, so they wear "
        "away. Patterns that help with lots of text get pushed on at every step, so they "
        "survive. Adam with weight decay is AdamW, the optimizer from the gradient descent "
        "chapter."
    )
    stage.note(
        "The real cure is more data. Large language models learn from so much text that they "
        "see most of it only once or twice. But text that turns up again and again, like famous "
        "quotes, licenses and the opening lines of books, can still end up memorized word for "
        "word, and researchers have pulled memorized text out of real models just like this: "
        "by giving them the start and letting them carry on."
    )


def overfit(lab: Lab, big: Run, yours: float, width: int) -> Iterator[Frame]:
    yield view(lab, big, yours, width)
    total, shown = SECONDS * FPS, 0
    steps = big.trainer.steps
    while not big.trainer.done:
        shown += 1
        target = steps * min(1.0, shown / total)
        big.step()
        while big.trainer.step_number < target and not big.trainer.done:
            big.step()
        yield view(lab, big, yours, width)


def view(lab: Lab, big: Run, yours: float, width: int) -> Table:
    tr = big.trainer
    tok = lab.tokenizer
    uniform = lab.baselines.uniform
    hi = max(uniform + 0.3, max(v for _, v in big.val) + 0.2)
    plot_w = max(30, min(46, width - 44))
    plot = viz.Plot(plot_w, 10, x_max=tr.steps, lo=0.0, hi=float(np.ceil(hi)))
    plot.guide(uniform, viz.FAINT, "guessing")
    plot.guide(yours, viz.GREEN, "yours")
    plot.line(big.mine, viz.ACCENT)
    plot.line(big.val, viz.AMBER)  # on top: it's the one that counts
    at, low = big.best
    if tr.step_number > at + 200:
        plot.mark(at, low, "▼", f"bold {viz.GREEN}")
    legend = Text.assemble(
        ("━ ", viz.ACCENT), ("its own text ", viz.FAINT), (f"{big.mine[-1][1]:.2f}", "bold"),
        ("   ━ ", viz.AMBER), ("held back ", viz.FAINT), (f"{big.val[-1][1]:.2f}", "bold"),
    )  # fmt: skip
    progress = Text.assemble(
        ("step ", viz.FAINT), (f"{tr.step_number:>4}", "bold"), (f" / {tr.steps}  ", viz.FAINT),
        viz.bar(tr.step_number / tr.steps, 16, viz.ACCENT),
    )  # fmt: skip
    left = Group(progress, Text(""), Text("loss", style="bold"), *plot.render(fmt="{:.0f}"), legend)

    side = max(24, width - plot_w - 14)
    seen_ids = big.seen + big.recited
    unseen_ids = big.unseen + big.guessed
    right = [Text("its own text, from the start:", style="bold")]
    right += running(tok, seen_ids, side, big.memory.copied(seen_ids), start=len(big.seen))[:5]
    right += [Text(""), Text("text it never saw:", style="bold")]
    right += running(tok, unseen_ids, side, None, start=len(big.unseen))[:4]
    right += [
        Text(""),
        Text.assemble(
            ("recited ", viz.FAINT),
            viz.bar(big.copied, 10, viz.RED),
            (f" {big.copied:.0%}", "bold"),
        ),
    ]
    grid = Table.grid(padding=(0, 4))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, vertical="top")
    grid.add_row(left, Group(*right))
    return grid
