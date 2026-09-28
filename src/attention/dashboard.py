"""The live training dashboard: watch the loss fall, the layers wake up and the samples improve."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import viz
from .corpus import Corpus
from .generate import Memory, picks, prompt
from .model import groups
from .train import Baselines, Step, Trainer, smooth
from .views import chip, prob_bars, running

SAMPLE_EVERY = 100
EVAL_EVERY = 50
SAMPLE_TOKENS = 48


@dataclass
class Watch:
    """Everything the dashboard shows, kept up to date as training runs."""

    trainer: Trainer
    baselines: Baselines
    corpus: Corpus
    seed: int
    rng: np.random.Generator  # for samples, so watching doesn't change how training goes
    sample: list[int] = field(default_factory=list)
    sample_step: int = 0
    last: Step | None = None
    compute: float = 0.0  # seconds spent actually training, not drawing
    memory: Memory = field(init=False)

    def __post_init__(self) -> None:
        self.memory = Memory(self.trainer.data.train)
        if not self.trainer.val_losses:
            self.trainer.evaluate()
        self.resample()

    def resample(self) -> None:
        tr = self.trainer
        steps = list(picks(tr.model, prompt(tr.data.tokenizer), self.rng, 0.8, limit=SAMPLE_TOKENS))
        self.sample = [*steps[-1].context, steps[-1].token] if steps else []
        self.sample_step = tr.step_number

    def step(self) -> Step:
        start = time.perf_counter()
        step = self.trainer.step()
        self.compute += time.perf_counter() - start
        self.last = step
        n = step.number
        if n % EVAL_EVERY == 0 or self.trainer.done:
            self.trainer.evaluate()
        if n % SAMPLE_EVERY == 0 or self.trainer.done:
            self.resample()
        return step

    # ── Pictures ──────────────────────────────────────────────────────────────

    def render(self, width: int) -> RenderableType:
        inner = width - 6
        tr = self.trainer
        tokens = tr.step_number * tr.batch_size * tr.model.config.context
        passes = tokens / max(len(tr.data.train), 1)
        progress = Text.assemble(
            ("step ", viz.FAINT), (f"{tr.step_number:>4}", "bold"), (f" / {tr.steps}  ", viz.FAINT),
            viz.bar(tr.step_number / tr.steps, max(10, inner - 70), viz.ACCENT),
            (f" {tr.step_number / tr.steps:4.0%}", "bold"),
            ("   lr ", viz.FAINT), (f"{tr.learning_rate():.4f}", viz.AMBER),
            ("   read ", viz.FAINT), (f"{tokens / 1e6:.1f}M", "bold"), (" tokens", viz.FAINT),
            (f" ({passes:.1f}× the text)", viz.FAINT),
        )  # fmt: skip
        wide = inner >= 84
        panels = [
            self.loss_panel(min(52, inner - 42) if wide else min(52, inner - 4)),
            self.samples_panel(38 if wide else min(52, inner - 4)),
            self.gradient_panel(),
            self.probe_panel(),
            self.lens_panel(),
        ]
        rows = pack(panels, inner, gap=3)
        body = Group(progress, Text(""), *(r for row in rows for r in (row, Text(""))))
        title = Text.assemble(
            (" pretraining ", f"bold {viz.ACCENT}"),
            (f"· {self.corpus.title} · seed {self.seed} ", viz.FAINT),
        )
        panel = Panel(body, box=box.ROUNDED, border_style=viz.FAINT, title=title, title_align="left",
                      padding=(1, 2, 0, 2), width=width)  # fmt: skip
        return Group(panel, self.caption())

    def loss_panel(self, width: int) -> tuple[int, RenderableType]:
        tr, b = self.trainer, self.baselines
        smoothed = smooth(tr.losses, 0.05)
        lo = min([b.pairs, *smoothed, *(v for _, v in tr.val_losses)]) - 0.3
        lo = max(0.0, np.floor(lo * 2) / 2)
        hi = np.ceil((b.uniform + 0.05) * 2) / 2
        plot = viz.Plot(width - 6, 8, x_max=tr.steps, lo=lo, hi=hi)
        plot.guide(b.tokens, viz.FAINT, "token counts")
        plot.guide(b.pairs, viz.FAINT, "token pairs")
        if smoothed:
            plot.line(list(enumerate(smoothed, 1)), viz.ACCENT)
        if tr.val_losses:
            plot.line(tr.val_losses, viz.AMBER)  # on top: it's the one that counts
        now = smoothed[-1] if smoothed else tr.val_losses[0][1]
        val = tr.val_losses[-1][1] if tr.val_losses else now
        legend = Text.assemble(
            ("━ ", viz.ACCENT), ("training ", viz.FAINT), (f"{now:.2f}", "bold"),
            ("   ━ ", viz.AMBER), ("held back ", viz.FAINT), (f"{val:.2f}", "bold"),
        )  # fmt: skip
        return width, Group(Text("loss", style="bold"), *plot.render(fmt="{:.1f}"), legend)

    def samples_panel(self, width: int) -> tuple[int, RenderableType]:
        tok = self.trainer.data.tokenizer
        head = Text.assemble(("it writes", "bold"), (f" at step {self.sample_step}", viz.FAINT))
        copied = self.memory.copied(self.sample) if self.sample else None
        lines = running(tok, self.sample, width, copied)[:8]
        lines += [Text("")] * (8 - len(lines))
        key = Text.assemble(("▆ ", viz.RED), ("copied word for word", viz.FAINT))
        return width, Group(head, *lines, key)

    def gradient_panel(self) -> tuple[int, RenderableType]:
        config = self.trainer.model.config
        names = groups(config)
        sizes = self.last.group_sizes(config) if self.last else dict.fromkeys(names, 0.0)
        lines = [Text("backprop's push", style="bold"), Text("loss", style=viz.FAINT)]
        for group in reversed(names):
            n = sizes[group]
            length = max(0.0, 1 + np.log10(max(n, 1e-6)) / 3) if n else 0.0  # 0.001 … 1, log scale
            lines.append(Text.assemble(("↓ ", viz.AMBER), (f"{group:<11}", ""),
                                       viz.bar(min(length, 1.0), 6, viz.AMBER), (f" {n:.3f}", viz.FAINT)))  # fmt: skip
        size = self.last.size if self.last else 0.0
        clipped = size > self.trainer.most
        lines.append(Text.assemble(
            ("whole gradient ", viz.FAINT), (f"{size:.2f}", f"bold {viz.RED if clipped else ''}"),
            (" clipped" if clipped else "", viz.RED),
        ))  # fmt: skip
        return 25, Group(*lines)

    def probe_ids(self) -> list[int]:
        return prompt(self.trainer.data.tokenizer, self.corpus.probe)

    def probe_panel(self) -> tuple[int, RenderableType]:
        tr = self.trainer
        tok = tr.data.tokenizer
        probs = tr.model.forward(np.array([self.probe_ids()])).probs[0, -1]
        head = Text.assemble(
            ("next after ", viz.FAINT), (self.corpus.probe.replace("\n", "↵"), f"bold {viz.ACCENT}")
        )
        return 24, Group(head, prob_bars(probs, tok, top=6, width=8))

    def lens_panel(self) -> tuple[int, RenderableType]:
        """What the model would guess at the end of the probe if it stopped after each layer."""
        tr = self.trainer
        tok = tr.data.tokenizer
        model = tr.model
        trace = model.forward(np.array([self.probe_ids()]))
        lens = model.lens(trace)[:, 0, -1]  # (depths, vocab)
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style=viz.FAINT, no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True, justify="right")
        names = ["embeddings", *(f"after layer {i + 1}" for i in range(model.config.layers))]
        for name, probs in zip(names, lens, strict=True):
            best = int(np.argmax(probs))
            grid.add_row(
                name, chip(tok, best), Text(viz.percent(float(probs[best])), style=viz.FAINT)
            )
        return 30, Group(Text("each layer's best guess", style="bold"), grid)

    def caption(self) -> Text:
        """What's going on, judged on the held-back text so memorizing doesn't count."""
        tr, b = self.trainer, self.baselines
        loss = tr.val_losses[-1][1] if tr.val_losses else b.uniform
        if tr.step_number == 0:
            words = "Untrained: every token equally likely, so what it writes is gibberish."
        elif loss > b.tokens + 0.1:
            words = "Finding its feet: learning which tokens turn up at all."
        elif loss > b.pairs + 0.05:
            words = "It knows which tokens are common. Now: which tokens follow which?"
        elif loss > b.pairs - 0.3:
            words = "About as good as a table of token pairs. Can attention look further back?"
        else:
            words = "Better than any table of token pairs: attention is putting the context to use."
        return Text.assemble(("  ", ""), ("◇ ", viz.ACCENT), (words, f"italic {viz.FAINT}"))


def pack(panels: list[tuple[int, RenderableType]], width: int, gap: int) -> list[Table]:
    """Lay panels out left to right, starting a new row when one won't fit."""
    rows: list[list[RenderableType]] = [[]]
    used = 0
    for w, panel in panels:
        if rows[-1] and used + gap + w > width:
            rows.append([])
            used = 0
        rows[-1].append(panel)
        used += (gap if used else 0) + w
    tables = []
    for row in rows:
        grid = Table.grid(padding=(0, gap))
        for _ in row:
            grid.add_column(no_wrap=True, vertical="top")
        grid.add_row(*row)
        tables.append(grid)
    return tables


FPS = 10


def frames(
    watch: Watch, width: int, seconds: float, fps: float = FPS
) -> Iterator[tuple[RenderableType, float]]:
    """Train to the end, a picture at a time, taking about `seconds` if drawing keeps up.

    Every picture is at least one step further on, and early steps get more screen time than
    later ones: that's when the most happens. The first picture is the untrained model. When
    the arithmetic takes longer than `seconds`, pictures come as fast as it allows.
    """
    tr = watch.trainer
    total = max(1, round(seconds * fps))
    yield watch.render(width), 1.0
    shown = 0
    while not tr.done:
        shown += 1
        target = tr.steps * min(1.0, shown / total) ** 1.6
        watch.step()
        while tr.step_number < target and not tr.done:
            watch.step()
        yield watch.render(width), 1 / fps


def run(
    watch: Watch,
    update: Callable[[RenderableType], None],
    width: int,
    seconds: float,
    interrupted: Callable[[float], bool],
) -> None:
    """Train to the end, redrawing as it goes.

    `interrupted(timeout)` waits up to `timeout` seconds and says whether to hurry up and
    finish, which still shows a glimpse now and then.
    """
    hurry = seconds <= 0
    glimpsed = 0.0
    for picture, pause in frames(watch, width, seconds):
        if not hurry:
            update(picture)
            hurry = interrupted(min(pause, 1.5))
        elif time.monotonic() - glimpsed > 0.25:
            update(picture)
            glimpsed = time.monotonic()
    update(watch.render(width))
