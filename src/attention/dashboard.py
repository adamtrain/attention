"""The live training dashboard: watch the loss fall, the weights shift and the samples improve."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import viz
from .corpus import Corpus
from .generate import invent
from .model import LAYERS, softmax
from .train import Baselines, Step, Trainer, smooth
from .views import display, token_chip

SAMPLE_EVERY = 20
EVAL_EVERY = 20
LAYER_NAMES = {
    "head": "head",
    "mlp": "MLP",
    "attention": "attention",
    "embeddings": "embeddings",
}


@dataclass
class Watch:
    """Everything the dashboard shows, kept up to date as training runs."""

    trainer: Trainer
    baselines: Baselines
    corpus: Corpus
    seed: int
    rng: np.random.Generator  # for samples, so watching doesn't change how training goes
    samples: list[str] = field(default_factory=list)
    sample_step: int = 0
    last: Step | None = None
    weight_scale: float = 0.0
    compute: float = 0.0  # seconds spent actually training, not drawing

    def __post_init__(self) -> None:
        self.weight_scale = viz.scale_of(self.trainer.model.params["embed.token"])
        if not self.trainer.val_losses:
            self.trainer.evaluate()
        self.resample()

    @property
    def known(self) -> set[str]:
        return set(self.corpus.words)

    def resample(self) -> None:
        tr = self.trainer
        self.samples = invent(tr.model, tr.data.vocab, self.rng, 6, temperature=0.8)
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
        progress = Text.assemble(
            ("step ", viz.FAINT), (f"{tr.step_number:>3}", "bold"), (f" / {tr.steps}  ", viz.FAINT),
            viz.bar(tr.step_number / tr.steps, max(10, inner - 44), viz.ACCENT),
            (f" {tr.step_number / tr.steps:4.0%}", "bold"),
            ("   learning rate ", viz.FAINT), (f"{tr.learning_rate():.4f}", viz.AMBER),
        )  # fmt: skip
        panels = [
            self.loss_panel(min(50, inner - 51) if inner >= 84 else min(50, inner - 23)),
            self.samples_panel(),
            self.gradient_panel(),
            self.probe_panel(),
            self.weights_panel(),
            self.nudge_panel(),
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
        smoothed = smooth(tr.losses, 0.08)
        lo = min([b.pairs, *smoothed, *(v for _, v in tr.val_losses)]) - 0.3
        lo = max(0.0, np.floor(lo * 2) / 2)
        hi = np.ceil((b.uniform + 0.05) * 2) / 2
        plot_w = width - 6
        plot = viz.Plot(plot_w, 8, x_max=tr.steps, lo=lo, hi=hi)
        plot.guide(b.letters, viz.FAINT, "letter counts")
        plot.guide(b.pairs, viz.FAINT, "letter pairs")
        if tr.val_losses:
            plot.line(tr.val_losses, viz.AMBER)
        if smoothed:
            plot.line(list(enumerate(smoothed, 1)), viz.ACCENT)
        now = smoothed[-1] if smoothed else tr.val_losses[0][1]
        val = tr.val_losses[-1][1] if tr.val_losses else now
        legend = Text.assemble(
            ("━ ", viz.ACCENT), ("training ", viz.FAINT), (f"{now:.2f}", "bold"),
            ("   ━ ", viz.AMBER), ("held back ", viz.FAINT), (f"{val:.2f}", "bold"),
        )  # fmt: skip
        return width, Group(Text("loss", style="bold"), *plot.render(fmt="{:.1f}"), legend)

    def samples_panel(self) -> tuple[int, RenderableType]:
        lines = [Text.assemble(("samples", "bold"), (f" at step {self.sample_step}", viz.FAINT))]
        for word in self.samples[:6]:
            new = word not in self.known
            shown = display(word)
            if len(shown) > 16:
                shown = shown[:15] + "…"
            lines.append(Text.assemble(
                ("✦ " if new else "· ", viz.GREEN if new else viz.FAINT), (shown, "" if new else viz.FAINT),
                ("" if new else " copy", viz.FAINT),
            ))  # fmt: skip
        lines += [Text("")] * (7 - len(lines))
        lines.append(
            Text.assemble(
                ("✦ ", viz.GREEN),
                ("new  ", viz.FAINT),
                ("· ", viz.FAINT),
                ("copied", viz.FAINT),
            )
        )
        return 20, Group(*lines)

    def gradient_panel(self) -> tuple[int, RenderableType]:
        lines = [Text("backprop's push", style="bold")]
        norms = self.last.layer_norms() if self.last else dict.fromkeys(LAYERS, 0.0)
        lines.append(Text("loss", style=viz.FAINT))
        for layer in reversed(LAYERS):
            n = norms[layer]
            length = max(0.0, 1 + np.log10(max(n, 1e-6)) / 3) if n else 0.0  # 0.001 … 1, log scale
            lines.append(Text.assemble(("↓ ", viz.AMBER), (f"{LAYER_NAMES[layer]:<11}", ""),
                                       viz.bar(min(length, 1.0), 6, viz.AMBER), (f" {n:.3f}", viz.FAINT)))  # fmt: skip
        lines.append(Text(""))
        lines.append(Text("gradient size (log)", style=viz.FAINT))
        return 25, Group(*lines)

    def probe_panel(self) -> tuple[int, RenderableType]:
        tr = self.trainer
        vocab = tr.data.vocab
        ids = np.array([vocab.encode("." + self.corpus.probe)])
        probs = softmax(tr.model.forward(ids).logits[0, -1])
        lines = [
            Text.assemble(
                ("next after ", viz.FAINT),
                (f".{self.corpus.probe}", f"bold {viz.ACCENT}"),
            )
        ]
        for i in np.argsort(-probs)[:7]:
            lines.append(Text.assemble(token_chip(vocab, int(i)), " ", viz.bar(float(probs[i]), 10, viz.ACCENT),
                                       (f" {viz.percent(float(probs[i])):>5}", "")))  # fmt: skip
        return 22, Group(*lines)

    def weights_panel(self) -> tuple[int, RenderableType]:
        table = self.trainer.model.params["embed.token"].T
        lines = viz.signed_blocks(table, self.weight_scale)
        letters = Text(self.trainer.data.vocab.chars, style=viz.FAINT, no_wrap=True)
        return table.shape[1], Group(Text("token embeddings", style="bold"), *lines, letters)

    def nudge_panel(self) -> tuple[int, RenderableType]:
        vocab = self.trainer.data.vocab
        if self.last is None:
            moved = np.zeros((self.trainer.model.config.width, len(vocab)))
        else:
            moved = self.last.moved["embed.token"].T
        lines = viz.nudge_blocks(moved)
        letters = Text(vocab.chars, style=viz.FAINT, no_wrap=True)
        return moved.shape[1], Group(Text("this step's nudge", style="bold"), *lines, letters)

    def caption(self) -> Text:
        """What's going on, judged on the held-back words so memorizing doesn't count."""
        tr, b = self.trainer, self.baselines
        loss = tr.val_losses[-1][1] if tr.val_losses else b.uniform
        if tr.step_number == 0:
            words = "Untrained: every letter is equally likely, so the samples are gibberish."
        elif loss > b.letters + 0.1:
            words = "Finding its feet: learning which letters turn up at all."
        elif loss > b.pairs + 0.05:
            words = "It knows which letters are common. Now: which letters follow which?"
        elif loss > b.pairs - 0.1:
            words = "About as good as a table of letter pairs. Can attention beat that?"
        else:
            words = "Better than any letter-pair table: attention lets it look further back."
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


def run(
    watch: Watch,
    update: Callable[[RenderableType], None],
    width: int,
    seconds: float,
    interrupted: Callable[[float], bool],
) -> float:
    """Train to the end, redrawing as it goes. Early steps get more screen time than later ones.

    `interrupted(timeout)` waits up to `timeout` seconds and says whether to hurry up and
    finish. Returns how long training took.
    """
    tr = watch.trainer
    fps = 15
    start = time.monotonic()
    update(watch.render(width))
    hurry = seconds <= 0 or interrupted(min(1.5, seconds / 10) if seconds > 0 else 0)
    while not tr.done:
        if hurry:
            watch.step()
            if tr.step_number % 25 == 0 or tr.done:
                update(watch.render(width))
            continue
        elapsed = time.monotonic() - start
        target = tr.steps * min(1.0, elapsed / seconds) ** 1.6
        watch.step()
        while tr.step_number < target and not tr.done:
            watch.step()
        update(watch.render(width))
        hurry = interrupted(1 / fps)
    update(watch.render(width))
    return time.monotonic() - start
