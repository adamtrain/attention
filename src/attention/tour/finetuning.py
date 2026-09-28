"""Chapter: fine-tuning on a speciality, the full way and with LoRA."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..finetune import (
    LEARNING_RATE,
    RANK,
    AdapterTrainer,
    LoRA,
    drift,
    narrow,
    niche_trainer,
    share,
)
from ..generate import samples
from ..train import LEARNING_RATE as PRETRAINING_RATE
from ..train import Trainer, evaluate
from ..views import display
from .lab import Lab
from .stage import Frame, Stage

LORA_RATE = 0.003
SAMPLES = 6
MEASURE = 48  # samples to measure the share with: fewer, and luck moves it a lot
FPS = 8


def run(stage: Stage, lab: Lab) -> None:
    c, data = lab.corpus, lab.data
    niche = c.niche
    narrowed = narrow(data, niche)  # the niche's documents, tokenized once
    docs = narrowed.train_docs
    stage.say(
        f"Pretraining gave you a model that writes any kind of {c.noun}. [b]Fine-tuning[/b] is "
        f"how you get one that does something more particular. Say you only want {niche.label}. "
        f"There are {len(docs)} of them in the training text:"
    )
    stage.show(examples(docs, stage.width))
    stage.say(
        "Fine-tuning is the pretraining loop all over again: run some examples forward, measure "
        "the loss, backpropagate, and step every weight downhill. Only three things change:",
        gap=False,
    )
    stage.console.print()
    stage.show(recipe(lab, len(docs)))
    limit = 48 if niche.titles else 96
    rng = lab.dice(11)
    before = share(
        samples(lab.model, lab.tokenizer, rng, MEASURE, temperature=0.8, limit=limit), niche
    )
    trainer = niche_trainer(lab.model, data, niche, lab.dice(6), steps=lab.budget.tuning)
    stage.say(
        "Starting from pretrained weights is what makes it work with so few examples: the model "
        f"already knows how {c.plural} go, so all it has to learn is which ones to prefer. "
        f"Right now, {before:.0%} of what your model writes is {niche.label}. Here's a copy of "
        "it, fine-tuning. Your own model isn't touched:"
    )

    def tuning() -> Iterator[Frame]:
        nonlocal trainer
        if trainer.step_number:  # a replay: start again from your model, with new dice
            trainer = niche_trainer(lab.model, data, niche, lab.dice(6), steps=lab.budget.tuning)
        return training(lab, trainer, before, limit, "full fine-tuning")

    stage.play(tuning, fps=FPS, start="fine-tune it", again="try again")
    after = share(
        samples(trainer.model, lab.tokenizer, rng, MEASURE, temperature=0.8, limit=limit), niche
    )
    general_before = evaluate(lab.model, *lab.trainer.held)
    general_after = evaluate(trainer.model, *lab.trainer.held)
    stage.say(
        f"Measured over {MEASURE} fresh samples: from {before:.0%} to {after:.0%}, from "
        f"{len(docs)} examples and a few seconds of "
        f"training. And its weights hardly moved: measured against their size, fine-tuning "
        f"changed them by [b]{drift(trainer.model, lab.model):.0%}[/b], where pretraining had "
        f"moved them {drift(lab.model, lab.initial):.0%} from where they started. Fine-tuning "
        "doesn't build new knowledge so much as tilt the model toward part of what it has."
    )
    stage.say(
        f"There's a price. Its loss on held-back {c.plural} of every kind rose from "
        f"{general_before:.2f} to [b]{general_after:.2f}[/b]: it got worse at everything else. "
        "Fine-tune too hard, or on too narrow a diet, and a model forgets what it used to know. "
        "That's called [b]catastrophic forgetting[/b], and it's why fine-tuning is kept short "
        "and gentle."
    )
    stage.wait("try LoRA")

    lora = LoRA.create(lab.model, lab.dice(16))
    grid = lab.model.params["layers.0.mlp.up"]
    m, n = grid.shape
    stage.say(
        f"That changed every one of your model's {lab.model.size:,} numbers, and for a model "
        "with billions of them that's expensive: every fine-tune is a whole new copy of the "
        "model, and training it needs memory for every weight's gradient and Adam's running "
        "averages. So most people fine-tune open models with [b]LoRA[/b] (low-rank adaptation) "
        "instead."
    )
    stage.say(
        "LoRA freezes the model, and learns a small correction for each grid of weights, made "
        f"from two thin grids multiplied together. For one of your MLP's {m} × {n} grids "
        f"({m * n:,} numbers), that's A, {m} × {RANK}, times B, {RANK} × {n}: "
        f"{RANK * (m + n):,} numbers, and the model uses W + A × B wherever it used W. B starts "
        "at zero, so the model starts out exactly as it was. Backprop works out the gradient for "
        "W + A × B as usual, and the chain rule turns it into gradients for A and B. Here it is, "
        "on the same examples:"
    )
    stage.show(lora_shapes(m, n))
    tuner = AdapterTrainer(
        lora, lambda r: narrowed.windows(r, 16, lab.model.config.context), lab.dice(17),
        steps=lab.budget.tuning, lr=LORA_RATE,
    )  # fmt: skip

    def adapting() -> Iterator[Frame]:
        nonlocal lora, tuner
        if tuner.step_number:
            lora = LoRA.create(lab.model, lab.dice(16))
            tuner = AdapterTrainer(
                lora, lambda r: narrowed.windows(r, 16, lab.model.config.context),
                lab.dice(17), steps=lab.budget.tuning, lr=LORA_RATE,
            )  # fmt: skip
        return training(lab, tuner, before, limit, f"LoRA, rank {RANK}")

    stage.play(adapting, fps=FPS, start="fine-tune with LoRA", again="try again")
    merged = lora.merged()
    lora_share = share(
        samples(merged, lab.tokenizer, rng, MEASURE, temperature=0.8, limit=limit), niche
    )
    lora_loss = evaluate(merged, *lab.trainer.held)
    stage.show(
        comparison(lab, trainer.model, merged, lora, after, lora_share, general_after, lora_loss)
    )
    close, behind = lora_share >= after - 0.05, lora_share < after - 0.15  # 48 samples: ±5 or so
    verdict = "as well or better" if close else "less well" if behind else "nearly as well"
    if lora_loss < general_after:
        verdict += ", but forgot less" if behind else ", and forgot less"
    stage.say(
        f"LoRA learned {lora.size:,} numbers instead of {lab.model.size:,}, and did {verdict}"
        + ": a correction built from thin grids can only change the model in a few directions, "
        "which keeps it from wandering far."
        + (
            " It can't touch the embedding table either, which full fine-tuning changed directly."
            if behind
            else ""
        )
        + f" On your small model, rank {RANK} is a fair slice of each grid; on a big one it's a "
        "sliver. "
        "Llama 3 8B with rank-16 LoRA on every grid learns about 0.5% as many numbers as the "
        "model has."
    )
    stage.note(
        "And the correction is its own small file. One base model can have many LoRA adapters "
        "(one for legal writing, one for a house style, one for a language), swapped in and out "
        "as needed, or merged into the weights for speed."
    )


def examples(docs: tuple[str, ...], width: int, most: int = 8) -> Group:
    lines = [
        Text(display(d.split("\n\n")[0] if "\n\n" in d[:80] else d, width - 4), style="italic")
        for d in docs[:most]
    ]
    if len(docs) > most:
        lines.append(Text(f"…and {len(docs) - most} more", style=viz.FAINT))
    return Group(*lines)


def recipe(lab: Lab, count: int) -> Table:
    """What fine-tuning does differently from pretraining."""
    pre = lab.trainer
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column()
    grid.add_row("starts from", "your pretrained weights, instead of random ones")
    grid.add_row(
        "learns from",
        f"only the {count} in the niche, instead of all {len(lab.data.train_docs)} {lab.corpus.plural}",
    )
    grid.add_row(
        "goes gently",
        f"{lab.budget.tuning} steps instead of {pre.steps:,}, with a learning rate of "
        f"{LEARNING_RATE} instead of {PRETRAINING_RATE}, and no weight decay",
    )
    return grid


def training(
    lab: Lab, trainer: Trainer | AdapterTrainer, before: float, limit: int, what: str
) -> Iterator[Frame]:
    """Fine-tuning, a picture at a time: what it writes, and how much of it is in the niche."""
    niche = lab.corpus.niche
    rng = np.random.default_rng([lab.seed, 18])
    written = samples(trainer.model, lab.tokenizer, rng, SAMPLES, temperature=0.8, limit=limit)
    yield progress(lab, trainer, written, before, what)
    while not trainer.done:
        target = min(trainer.steps, trainer.step_number + max(1, trainer.steps // 10))
        while trainer.step_number < target:
            trainer.step()
        written = samples(trainer.model, lab.tokenizer, rng, SAMPLES, temperature=0.8, limit=limit)
        yield progress(lab, trainer, written, share(written, niche), what)


def progress(lab: Lab, trainer, written: list[str], now: float, what: str) -> Group:
    niche = lab.corpus.niche
    head = Text.assemble(
        (f"{what}   ", "bold"), ("step ", viz.FAINT), (f"{trainer.step_number:>3}", "bold"),
        (f" / {trainer.steps}  ", viz.FAINT), viz.bar(trainer.step_number / trainer.steps, 20, viz.ACCENT),
    )  # fmt: skip
    meter = Text.assemble(
        (f"{niche.label}, in these samples  ", viz.FAINT),
        viz.bar(now, 20, viz.GREEN),
        (f" {now:4.0%}", "bold"),
    )
    lines = []
    for text in written:
        hit = niche.has(text)
        lines.append(
            Text.assemble(
                ("✓ " if hit else "· ", viz.GREEN if hit else viz.FAINT),
                (display(text, 70), "" if hit else viz.FAINT),
            )
        )
    return Group(head, Text(""), meter, Text(""), *lines)


def lora_shapes(m: int, n: int) -> Table:
    grid = Table.grid(padding=(0, 1))
    for _ in range(7):
        grid.add_column(no_wrap=True)
    full = Text(f" W: {m} × {n} ", style=viz.style("#f4f4ff", "#3d4270"))
    a = Text(f" A: {m}×{RANK} ", style=viz.style(viz.DARK, viz.AMBER))
    b = Text(f" B: {RANK}×{n} ", style=viz.style(viz.DARK, viz.AMBER))
    grid.add_row(
        full,
        Text("frozen", style=viz.FAINT),
        Text("+"),
        a,
        Text("×"),
        b,
        Text("learned", style=viz.AMBER),
    )
    return grid


def comparison(
    lab: Lab,
    full,
    merged,
    lora: LoRA,
    full_share: float,
    lora_share: float,
    full_loss: float,
    lora_loss: float,
) -> Table:
    grid = Table.grid(padding=(0, 3))
    for justify in ("left", "right", "right"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(
        "", Text("full fine-tuning", style="bold"), Text(f"LoRA, rank {RANK}", style="bold")
    )
    grid.add_row(Text("numbers learned", style=viz.FAINT), f"{lab.model.size:,}", f"{lora.size:,}")
    grid.add_row(
        Text(lab.corpus.niche.label, style=viz.FAINT), f"{full_share:.0%}", f"{lora_share:.0%}"
    )
    grid.add_row(
        Text("weights moved", style=viz.FAINT),
        f"{drift(full, lab.model):.0%}",
        f"{drift(merged, lab.model):.0%}",
    )
    grid.add_row(Text("held-back loss", style=viz.FAINT), f"{full_loss:.2f}", f"{lora_loss:.2f}")
    grid.add_row(
        Text("file to keep (16-bit)", style=viz.FAINT),
        f"{lab.model.size * 2 / 1024:,.0f} KB",
        f"{lora.size * 2 / 1024:,.0f} KB",
    )
    return grid
