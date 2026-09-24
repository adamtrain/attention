"""Chapter: your model, and how it compares."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from .. import viz
from .lab import Lab
from .pretraining import tidy
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    c = lab.corpus
    stage.say(
        f"Your model, [bold {viz.PURPLE}]{lab.name}[/], is saved at "
        f"`{tidy(lab.model_path)}`. From any terminal:"
    )
    stage.show(commands(lab))
    stage.wait()

    stage.say("How does it compare with a real large language model? Here's GPT-3, from 2020:")
    stage.show(comparison(lab))
    stage.wait()

    stage.say("So that's the whole story, end to end:")
    stage.show(recap())
    stage.say(
        "Every large language model is this, scaled up: more tokens, more layers, more text, "
        "more steps. Underneath, it's the loop you just watched. Predict the next token, measure "
        "the surprise, backpropagate, nudge every weight a little downhill, and do it again."
    )
    stage.show(
        Text.assemble(
            ("Thanks for taking the tour. ", "bold"),
            (f"Enjoy your {c.plural}.", viz.FAINT),
        )
    )


def commands(lab: Lab) -> Table:
    c = lab.corpus
    probe = c.probe
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column(style=viz.FAINT)
    grid.add_row("attention generate", f"invent ten new {c.plural}")
    grid.add_row("attention generate -t 1.5", "…stranger ones (higher temperature)")
    grid.add_row(f"attention generate {probe}", f"…ones that start with “{probe}”")
    grid.add_row(f"attention explain {probe}", "watch one prediction up close")
    grid.add_row("attention train", "train a new one (try --corpus names, or your own list)")
    grid.add_row("attention info", "everything about your model")
    grid.add_row("attention tour", "take the tour again")
    grid.add_row("attention clean", "delete your model when you're done with it")
    return grid


def comparison(lab: Lab) -> Table:
    tr, cfg = lab.trainer, lab.model.config
    chars = sum(len(w) + 1 for w in lab.data.train)
    tokens = (
        tr.step_number
        * tr.batch_size
        * (sum(len(w) + 1 for w in lab.data.train) / len(lab.data.train))
    )
    flops = (
        6 * lab.model.size * tokens
    )  # a forward and backward pass costs ~6 operations per weight
    grid = Table.grid(padding=(0, 4))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row("", Text(lab.name, style=f"bold {viz.PURPLE}"), Text("GPT-3", style="bold"))
    rows = [
        ("parameters", f"{lab.model.size:,}", "175,000,000,000"),
        ("vocabulary", f"{cfg.vocab} characters", "50,257 pieces of words"),
        ("context", f"{cfg.context} characters", "2,048 tokens"),
        ("layers", "1", "96"),
        ("attention heads", f"{cfg.heads}", "96 in each layer"),
        ("training text", f"{chars:,} characters", "300 billion tokens"),
        (
            "arithmetic",
            f"~{float(f'{flops:.2g}'):,.0f}",
            "~314,000,000,000,000,000,000,000",
        ),
    ]
    for name, mine, theirs in rows:
        grid.add_row(name, mine, theirs)
    return grid


RECAP = [
    ("tokens", "text is split into chunks, and each chunk becomes a number", "Tokens"),
    ("embeddings", "each token looks up a vector, and so does its position", "Embeddings"),
    ("attention", "each position gathers what it needs from the ones before", "Attention"),
    ("MLP", "each position thinks over what it gathered", "The whole model"),
    ("softmax", "a score for every possible next token becomes a probability", "Softmax"),
    ("loss", "how surprised it was by the right answer", "Loss"),
    ("backprop", "which way every weight should move, by the chain rule", "Backpropagation"),
    ("gradient descent", "move them all a little way downhill", "Gradient descent"),
    ("pretraining", "do that thousands of times, on lots of text", "Pretraining"),
    ("overfitting", "…but stop before it memorizes", "Memorizing"),
    ("inference", "sample a token, add it, repeat, with the weights frozen", "Inference"),
    ("fine-tuning", "a little more training, on examples of what you want", "Fine-tuning"),
    ("feedback", "make the answers people liked more likely", "Fine-tuning"),
]


def recap() -> Table:
    from . import CHAPTERS  # here, not at the top: this module is part of that list

    number = {c.title: i for i, c in enumerate(CHAPTERS, 1)}
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column()
    grid.add_column(style=viz.FAINT, no_wrap=True, justify="right")
    for name, what, chapter in RECAP:
        grid.add_row(name, what, f"ch. {number[chapter]}")
    return grid
