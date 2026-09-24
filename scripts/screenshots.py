"""Regenerate the README screenshots in docs/.

    uv run scripts/screenshots.py

Every picture is drawn by the same code the tour and the commands use, from real models trained
with fixed seeds, then saved with Rich's SVG export. Nothing touches your own saved model.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.padding import Padding
from rich.terminal_theme import TerminalTheme
from rich.text import Text

from attention import tour, viz
from attention.cli import show_explain, show_words
from attention.corpus import built_in
from attention.dashboard import Watch
from attention.finetune import niche_trainer, share
from attention.generate import invent, picks
from attention.store import load
from attention.tour import attention as attention_chapter
from attention.tour import backprop, descent, finetuning, memorizing
from attention.tour.lab import Lab
from attention.tour.stage import THEME as STYLES
from attention.tour.stage import Stage
from attention.views import step_view

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PROMPT = "\u276f "  # a shell-prompt chevron
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


def prompt(console: Console, command: str) -> None:
    console.print(Text.assemble((PROMPT, f"bold {viz.ACCENT}"), (command, "bold")))


def chapter(stage: Stage, title: str) -> None:
    number, c = next((i, c) for i, c in enumerate(tour.CHAPTERS, 1) if c.title == title)
    stage.header(number, len(tour.CHAPTERS), c.title, c.subtitle)


def last(frames):
    """The final frame of an animation."""
    *_, frame = frames
    return frame[0] if isinstance(frame, tuple) else frame


def save(console: Console, name: str, title: str) -> None:
    console.save_svg(str(DOCS / f"{name}.svg"), title=title, theme=THEME)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    home = Path(tempfile.mkdtemp())

    # The hero: pretraining, part way through.
    lab = Lab.create(built_in("dinosaurs"), SEED, home / "model.npz")
    watch = Watch(lab.trainer, lab.baselines, lab.corpus, lab.seed, lab.rng)
    while lab.trainer.step_number < 180:
        watch.step()
    hero = terminal(100)
    prompt(hero, f"attention train --corpus dinosaurs --seed {SEED}")
    hero.print()
    hero.print(Padding(watch.render(96), (0, 0, 0, 2)))
    save(hero, "hero", "attention train")

    # Attention, from the tour (untrained, as the tour shows it).
    fresh = Lab.create(built_in("dinosaurs"), SEED, home / "unused.npz")
    con = terminal(88)
    stage = Stage(con, animate=False)
    chapter(stage, "Attention")
    inputs, _ = fresh.example()
    tr = fresh.model.forward(inputs)
    chars = [fresh.vocab.chars[int(i)] for i in inputs[0]]
    raw = (tr.q[0, 0] @ tr.k[0, 0].T) / np.sqrt(fresh.model.config.head_width)
    stage.show(last(attention_chapter.scoring(raw, tr.weights[0, 0], chars)))
    save(con, "attention", "attention · tour")

    # Backpropagation: the neuron, then the transformer's output gradient.
    con = terminal(88)
    stage = Stage(con, animate=False)
    chapter(stage, "Backpropagation")
    stage.show(last(backprop.graph_story()))
    focus = max(1, round(len(fresh.corpus.example) * 0.65))
    right = int(fresh.example()[1][0, focus])
    seen = fresh.vocab.decode(inputs[0, : focus + 1])
    stage.say(
        "In the transformer, backprop starts with the probabilities. For the quiz "
        f"`{seen}` → `{fresh.vocab.chars[right]}`:",
    )
    stage.show(backprop.output_gradient(fresh, tr.probs[0, focus], right))
    save(con, "backprop", "attention · tour")

    # Gradient descent: the three walkers.
    con = terminal(100)
    stage = Stage(con, animate=False)
    chapter(stage, "Gradient descent")
    stage.show(last(descent.race(stage.width)))
    save(con, "descent", "attention · tour")

    # Inference: a word being written, mid-way.
    lab.train_quietly()
    rng = np.random.default_rng(5)
    steps = list(picks(lab.model, lab.vocab, rng, 0.8))
    con = terminal(88)
    stage = Stage(con, animate=False)
    chapter(stage, "Inference")
    mid = steps[min(5, len(steps) - 1)]
    stage.show(step_view(lab.vocab, mid, stage.width, mid.roll))
    save(con, "inference", "attention · tour")

    # Memorizing: the oversized model, trained too long.
    big = memorizing.Run.start(lab)
    while not big.trainer.done:
        big.step()
    con = terminal(100)
    stage = Stage(con, animate=False)
    chapter(stage, "Memorizing")
    stage.show(memorizing.view(lab, big, lab.trainer.val_losses[-1][1], stage.width))
    save(con, "memorizing", "attention · tour")

    # Fine-tuning: a copy of the model, specialized on horned dinosaurs.
    niche = lab.corpus.niche
    rng = np.random.default_rng(11)
    before = invent(lab.model, lab.vocab, rng, 40, 0.8)
    tuner = niche_trainer(lab.model, lab.data, niche, lab.dice(6))
    while not tuner.done:
        tuner.step()
    after = invent(tuner.model, lab.vocab, rng, 40, 0.8)
    con = terminal(88)
    stage = Stage(con, animate=False)
    chapter(stage, "Fine-tuning")
    stage.show(
        finetuning.compare(
            before, after, share(before, niche), share(after, niche), niche.ending,
            set(lab.corpus.words),
        )
    )  # fmt: skip
    save(con, "finetuning", "attention · tour")

    # The commands, on the trained model.
    saved = load(lab.model_path)
    con = terminal(84)
    prompt(con, "attention generate -n 6")
    show_words(con, saved, invent(saved.model, saved.vocab, np.random.default_rng(3), 6), 0.8)
    prompt(con, "attention explain stegosa --teach e")
    show_explain(con, saved, "stegosa", "e")
    save(con, "commands", "attention")

    for path in sorted(DOCS.glob("*.svg")):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
