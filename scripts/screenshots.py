"""Regenerate the README screenshots in docs/.

    uv run scripts/screenshots.py

Every picture is drawn by the same code the tour and the commands use, from real models trained
with fixed seeds, then saved with Rich's SVG export. Nothing touches your own saved model. It
takes several minutes: the models are trained for real.
"""

from __future__ import annotations

import io
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.padding import Padding
from rich.terminal_theme import TerminalTheme
from rich.text import Text

from attention import longhand, tour, viz
from attention.cli import show_explain
from attention.corpus import built_in
from attention.dashboard import Watch
from attention.finetune import AdapterTrainer, LoRA, narrow, niche_trainer, share
from attention.generate import picks, prompt, samples
from attention.induction import Race, habits, sequences
from attention.store import load
from attention.tour import attention as attention_chapter
from attention.tour import cache, finetuning, layers, learned, memorizing
from attention.tour.lab import Lab
from attention.tour.stage import THEME as STYLES
from attention.tour.stage import Stage
from attention.train import evaluate
from attention.views import running

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PROMPT = "❯ "  # a shell-prompt chevron
SEED = 21

THEME = TerminalTheme(
    background=(16, 18, 25),
    foreground=(226, 228, 236),
    normal=[
        (32, 34, 44),
        (242, 80, 110),
        (31, 191, 143),
        (235, 154, 18),
        (124, 131, 247),
        (168, 113, 247),
        (86, 182, 194),
        (200, 202, 212),
    ],
    bright=[
        (92, 96, 112),
        (255, 110, 136),
        (70, 214, 170),
        (250, 184, 60),
        (152, 158, 255),
        (190, 146, 255),
        (120, 208, 220),
        (255, 255, 255),
    ],
)


def terminal(width: int) -> Console:
    return Console(
        record=True,
        width=width,
        height=60,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        file=io.StringIO(),
        theme=STYLES,
    )


def prompt_line(console: Console, command: str) -> None:
    console.print(Text.assemble((PROMPT, f"bold {viz.ACCENT}"), (command, "bold")))


def chapter(stage: Stage, title: str) -> None:
    number, c = next((i, c) for i, c in enumerate(tour.CHAPTERS, 1) if c.title == title)
    stage.header(number, len(tour.CHAPTERS), c.title, c.subtitle)


def last(frames):
    """The final frame of an animation."""
    *_, frame = frames
    return frame[0] if isinstance(frame, tuple) else frame


class Snapped(Exception):
    """The picture has been taken."""


@dataclass
class Snapshot(Stage):
    """Draws only the step called `title`, and its first animation stopped at frame `frame`."""

    title: str = ""
    frame: int = 0
    drawing: bool = False

    def header(self, number: int, total: int, title: str, subtitle: str) -> None:
        self.drawing = title == self.title
        if self.drawing:
            super().header(number, total, title, subtitle)

    def say(self, text: str, gap: bool = True) -> None:
        if self.drawing:
            super().say(text, gap)

    def show(self, *items, gap: bool = True) -> None:
        if self.drawing:
            super().show(*items, gap=gap)

    def play(self, frames, *args, **kwargs) -> None:
        if self.drawing:
            picture = list(frames())[self.frame]
            self.show(picture[0] if isinstance(picture, tuple) else picture)
            raise Snapped


def save(console: Console, name: str, title: str) -> None:
    console.save_svg(str(DOCS / f"{name}.svg"), title=title, theme=THEME)
    print(f"wrote docs/{name}.svg")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    home = Path(tempfile.mkdtemp())
    lab = Lab.create(built_in("fables"), SEED, home / "model.npz")

    # attention math: the little model's attention scores, worked out one row at a time.
    con = terminal(88)
    prompt_line(con, "attention math")
    with suppress(Snapped):
        longhand.run(Snapshot(con, animate=False, title="Layer 1 · scores", frame=7))
    save(con, "math", "attention math")

    # Attention, from the tour: the lookup with made-up numbers, then the three jobs.
    con = terminal(88)
    stage = Stage(con, animate=False)
    chapter(stage, "Attention")
    stage.show(last(attention_chapter.toy()))
    stage.show(attention_chapter.jobs())
    save(con, "attention", "attention · tour")

    # The hero: pretraining, part way through.
    watch = Watch(lab.trainer, lab.baselines, lab.corpus, lab.seed, lab.rng)
    while lab.trainer.step_number < round(lab.trainer.steps * 0.45, -2):
        watch.step()
    hero = terminal(100)
    prompt_line(hero, f"attention train --corpus fables --seed {SEED}")
    hero.print()
    hero.print(Padding(watch.render(96), (0, 0, 0, 2)))
    save(hero, "hero", "attention train")
    while not lab.trainer.done:
        watch.step()
    lab.finish()

    # Why layers stack: the race, and the circuit it grew.
    race = Race(layers.WORDS + 1, np.random.default_rng(layers.FIRST))
    while not race.done:
        race.step()
    con = terminal(96)
    stage = Stage(con, animate=False)
    chapter(stage, "Why layers stack")
    stage.show(layers.view(race, stage.width))
    two = race.learners[1].model
    found = habits(two, sequences(np.random.default_rng(5), 64, layers.WORDS + 1))
    shown = sequences(np.random.default_rng(2), 1, layers.WORDS + 1)
    stage.show(layers.circuit_view(lab, two, shown, layers.words(lab), found, stage.width))
    save(con, "layers", "attention · tour")

    # What it learned: the logit lens on the example.
    con = terminal(96)
    stage = Stage(con, animate=False)
    chapter(stage, "What it learned")
    inputs, _ = lab.example()
    lens = lab.model.lens(lab.model.forward(inputs))[:, 0]
    stage.show(learned.lens_view(lab, lens, learned.pick_rows(lab, lens, 10), stage.width))
    save(con, "lens", "attention · tour")

    # The KV cache: filling up, and what it costs at scale.
    con = terminal(88)
    stage = Stage(con, animate=False)
    chapter(stage, "The KV cache")
    stage.show(last(cache.filling(lab, stage.width)))
    stage.show(cache.memory_table(lab))
    save(con, "cache", "attention · tour")

    # Memorizing: a model with too little to read, reciting.
    big = memorizing.Run.start(lab)
    while not big.trainer.done:
        big.step()
    con = terminal(100)
    stage = Stage(con, animate=False)
    chapter(stage, "Memorizing")
    stage.show(memorizing.view(lab, big, lab.trainer.val_losses[-1][1], stage.width))
    save(con, "memorizing", "attention · tour")

    # Fine-tuning: all of it, and with LoRA.
    niche = lab.corpus.niche
    full = niche_trainer(lab.model, lab.data, niche, np.random.default_rng([SEED, 6]))
    while not full.done:
        full.step()
    lora = LoRA.create(lab.model, np.random.default_rng([SEED, 16]))
    data = narrow(lab.data, niche)
    context = lab.model.config.context
    tuner = AdapterTrainer(
        lora, lambda r: data.windows(r, 16, context), np.random.default_rng([SEED, 17]),
        steps=full.steps, lr=finetuning.LORA_RATE,
    )  # fmt: skip
    while not tuner.done:
        tuner.step()
    merged = lora.merged()
    rng = np.random.default_rng(11)

    def measure(model) -> float:
        texts = samples(model, lab.tokenizer, rng, finetuning.MEASURE, temperature=0.8, limit=48)
        return share(texts, niche)

    con = terminal(88)
    stage = Stage(con, animate=False)
    chapter(stage, "Fine-tuning")
    stage.show(
        finetuning.comparison(
            lab, full.model, merged, lora, measure(full.model), measure(merged),
            evaluate(full.model, *lab.trainer.held), evaluate(merged, *lab.trainer.held),
        )
    )  # fmt: skip
    save(con, "finetuning", "attention · tour")

    # The commands, on the trained model.
    saved = load(lab.model_path)
    con = terminal(96)
    prompt_line(con, 'attention generate -n 1 "The Fox and the"')
    con.print()
    ids = prompt(saved.tokenizer, "The Fox and the")
    steps = list(picks(saved.model, ids, np.random.default_rng(3), 0.8, limit=60))
    written = [*ids, *(p.token for p in steps)]
    for line in running(saved.tokenizer, written, 88, saved.memory.copied(written), start=len(ids)):
        con.print(Padding(line, (0, 0, 0, 4)))
    con.print()
    prompt_line(con, 'attention explain "The Fox and the"')
    show_explain(con, saved, ids)
    save(con, "commands", "attention")


if __name__ == "__main__":
    main()
