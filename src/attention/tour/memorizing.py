"""Chapter: overfitting. What happens when a model is too big, or trains too long."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..generate import invent
from ..model import Config, Transformer
from ..train import Trainer, evaluate, smooth
from ..views import display
from .lab import Lab
from .stage import Frame, Stage

WIDTH, HIDDEN = 32, 128  # about three and a half times your model
STEPS = 1600  # four times as long
SECONDS = 16
FPS = 15
SAMPLES = 8


@dataclass
class Run:
    """The oversized model, training, and what it writes along the way."""

    trainer: Trainer
    known: set[str]  # the words it trains on
    rng: np.random.Generator
    samples: list[str] = field(default_factory=list)
    val: list[tuple[int, float]] = field(default_factory=list)

    @classmethod
    def start(cls, lab: Lab) -> Run:
        data = lab.data
        config = Config(len(data.vocab), data.context, width=WIDTH, heads=2, hidden=HIDDEN)
        model = Transformer.create(config, lab.dice(7))
        trainer = Trainer(model, data, lab.dice(8), steps=STEPS)
        run = cls(trainer, set(data.train), lab.dice(9))
        run.check()
        return run

    def check(self) -> None:
        tr = self.trainer
        self.val.append((tr.step_number, evaluate(tr.model, tr.data, tr.data.val)))
        self.samples = invent(tr.model, tr.data.vocab, self.rng, SAMPLES, temperature=0.8)

    def step(self) -> None:
        self.trainer.step()
        if self.trainer.step_number % 40 == 0:
            self.check()

    @property
    def copies(self) -> int:
        return sum(w in self.known for w in self.samples)

    @property
    def best(self) -> tuple[int, float]:
        return min(self.val, key=lambda sv: sv[1])


def run(stage: Stage, lab: Lab) -> None:
    c, tr = lab.corpus, lab.trainer
    stage.say(
        f"Why did training stop at {tr.steps} steps? And why such a small model? Let's break "
        "both rules and see. Here's a model about three and a half times bigger "
        f"({WIDTH} numbers per token instead of {lab.model.config.width}), trained four times as "
        f"long, on the same {len(lab.data.train)} {c.plural}. Your own model isn't touched."
    )
    stage.say(
        "Watch the two lines: [accent]training loss[/accent] on the words it studies, and "
        f"[amber]held-back loss[/amber] on the {c.plural} it never sees. And watch how many of "
        "its samples are [b]copies[/b] of words it trained on. The green line is your own "
        "model's held-back loss, for comparison."
    )

    yours = tr.val_losses[-1][1] if tr.val_losses else lab.baselines.pairs
    big = Run.start(lab)

    def training() -> Iterator[Frame]:
        nonlocal big
        if big.trainer.step_number:  # a replay: a new big model, from new random numbers
            big = Run.start(lab)
        return overfit(lab, big, yours, stage.width)

    stage.play(training, fps=FPS, start="break the rules", again="try another")
    if big.trainer.step_number % 40:
        big.check()

    at, low = big.best
    last = big.val[-1][1]
    uniform = lab.baselines.uniform
    worse = f", worse than the {uniform:.2f} of pure guessing" if last > uniform else ""
    trained_loss = smooth(big.trainer.losses, 0.08)[-1]
    stage.say(
        f"On the words it studied, it kept improving, down to {trained_loss:.2f}. But its "
        f"held-back loss bottomed out at {low:.2f} around step {at}, then climbed to "
        f"[b]{last:.2f}[/b]{worse}. And {big.copies} of its last {SAMPLES} samples are copies."
    )
    stage.say(
        "It [b]memorized[/b] its training words instead of learning what they have in common, "
        "so it got better at the test it had already seen and worse at everything else. That's "
        "called [b]overfitting[/b]. Your model avoided it by being small and stopping in time "
        f"(its held-back loss ended at {yours:.2f}), which is why the dashboard kept an eye on "
        "the held-back words."
    )
    stage.note(
        "The real cure is more data. Large language models learn from so much text that they "
        "see most of it only once or twice. Text that turns up again and again, like famous "
        "quotes, can still end up memorized word for word."
    )


def overfit(lab: Lab, big: Run, yours: float, width: int) -> Iterator[Frame]:
    yield view(lab, big, yours, width)
    total, shown = SECONDS * FPS, 0
    while not big.trainer.done:
        shown += 1
        target = STEPS * min(1.0, shown / total)
        big.step()
        while big.trainer.step_number < target and not big.trainer.done:
            big.step()
        yield view(lab, big, yours, width)


def view(lab: Lab, big: Run, yours: float, width: int) -> Table:
    tr = big.trainer
    uniform = lab.baselines.uniform
    hi = max(uniform + 0.3, max(v for _, v in big.val) + 0.2)
    plot_w = max(30, min(52, width - 34))
    plot = viz.Plot(plot_w, 10, x_max=STEPS, lo=0.0, hi=float(np.ceil(hi * 2) / 2))
    plot.guide(uniform, viz.FAINT, "guessing")
    plot.guide(yours, viz.GREEN, "your model")
    plot.line(big.val, viz.AMBER)
    if tr.losses:
        plot.line(list(enumerate(smooth(tr.losses, 0.08), 1)), viz.ACCENT)
    at, low = big.best
    if tr.step_number > at + 120:
        plot.mark(at, low, "▼", f"bold {viz.GREEN}")
    now = smooth(tr.losses, 0.08)[-1] if tr.losses else uniform
    legend = Text.assemble(
        ("━ ", viz.ACCENT), ("training ", viz.FAINT), (f"{now:.2f}", "bold"),
        ("   ━ ", viz.AMBER), ("held back ", viz.FAINT), (f"{big.val[-1][1]:.2f}", "bold"),
        ("   ▼ ", viz.GREEN), ("its best", viz.FAINT),
    )  # fmt: skip
    progress = Text.assemble(
        ("step ", viz.FAINT), (f"{tr.step_number:>4}", "bold"), (f" / {STEPS}  ", viz.FAINT),
        viz.bar(tr.step_number / STEPS, 20, viz.ACCENT),
        (f"   {tr.model.size:,} parameters", viz.FAINT),
    )  # fmt: skip
    left = Group(progress, Text(""), Text("loss", style="bold"), *plot.render(), legend)

    lines = [Text.assemble(("it writes", "bold"), (f" at step {tr.step_number}", viz.FAINT))]
    for w in big.samples:
        copied = w in big.known
        shown = display(w) if len(w) <= 16 else display(w)[:15] + "…"
        lines.append(
            Text.assemble(
                ("· " if copied else "✦ ", viz.RED if copied else viz.GREEN),
                (shown, viz.FAINT if copied else ""),
                (" copy" if copied else "", viz.RED),
            )
        )
    lines.append(Text(""))
    lines.append(
        Text.assemble(
            ("copies ", viz.FAINT),
            viz.bar(big.copies / SAMPLES, 10, viz.RED),
            (f" {big.copies} of {SAMPLES}", "bold"),
        )
    )
    grid = Table.grid(padding=(0, 4))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, vertical="bottom")
    grid.add_row(left, Group(*lines))
    return grid
