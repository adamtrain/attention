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
        f"`{tidy(lab.model_path)}`, with its chat version beside it. From any terminal:"
    )
    stage.show(commands(lab))
    stage.wait()

    stage.say("How does it compare with real large language models?")
    stage.show(comparison(lab))
    stage.say(
        "Every row is something you've now seen at work. The differences are of size: more "
        "tokens in the vocabulary, more numbers per token, more layers and heads, a longer "
        "context, and vastly more text. Llama 3 8B read about 15 trillion tokens, nearly 2,000 "
        "for every one of its weights; yours read about "
        f"{lab.trainer.steps * lab.trainer.batch_size * lab.model.config.context / lab.model.size:.0f}."
    )
    stage.wait()

    stage.say("So that's the whole story, end to end:")
    stage.show(recap())
    stage.say(
        "Every large language model is this, scaled up. Underneath, it's the loop you just "
        "watched: predict the next token, measure the surprise, backpropagate, nudge every "
        "weight a little downhill, and do it again, trillions of tokens at a time."
    )
    stage.show(
        Text.assemble(
            ("Thanks for taking the tour. ", "bold"),
            (f"Enjoy your {c.plural}.", viz.FAINT),
        )
    )


def commands(lab: Lab) -> Table:
    c = lab.corpus
    probe = c.probe.replace("\n", " ")
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column(style=viz.FAINT)
    grid.add_row("attention generate", f"write new {c.plural}")
    grid.add_row(f'attention generate "{probe}"', "…carrying on from a start you give it")
    grid.add_row("attention generate -t 1.2 -k 20", "…with the dials: temperature, top-k, top-p")
    grid.add_row("attention chat", "talk to the chat version")
    grid.add_row(f'attention explain "{probe}"', "one prediction up close, layer by layer")
    grid.add_row('attention tokenize "any text"', "see how your tokenizer splits it")
    grid.add_row("attention math", "the whole forward pass, number by number")
    grid.add_row("attention info", "everything about your model, and its config.json")
    grid.add_row("attention train", "train a new one (try --corpus shakespeare)")
    grid.add_row("attention tour", "take the tour again")
    grid.add_row("attention clean", "delete your models when you're done with them")
    return grid


def comparison(lab: Lab) -> Table:
    tr, cfg = lab.trainer, lab.model.config
    tokens = tr.steps * tr.batch_size * cfg.context
    flops = 6 * lab.model.size * tokens  # a forward and backward pass is ~6 operations per weight
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(style="bold", no_wrap=True, justify="right")
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True, justify="right")
    grid.add_row(
        "",
        Text(lab.name, style=f"bold {viz.PURPLE}"),
        Text("GPT-3", style="bold"),
        Text("Llama 3 8B", style="bold"),
    )
    rows = [
        ("parameters", f"{lab.model.size:,}", "175 billion", "8 billion"),
        ("vocabulary", f"{cfg.vocab:,} tokens", "50,257", "128,256"),
        ("numbers per token", f"{cfg.width}", "12,288", "4,096"),
        ("layers", f"{cfg.layers}", "96", "32"),
        ("attention heads", f"{cfg.heads} per layer", "96", "32"),
        ("key/value heads", f"{cfg.kv_heads} per layer", "96", "8"),
        ("positions", "RoPE", "learned", "RoPE"),
        ("context", f"{cfg.context} tokens", "2,048", "8,192"),
        ("training text", f"{tokens / 1e6:.1f}M tokens", "300 billion", "15 trillion"),
        ("arithmetic", f"~{flops:.1e}".replace("e+", " × 10^"), "~3.1 × 10^23", "~7.2 × 10^23"),
    ]  # fmt: skip
    for row in rows:
        grid.add_row(*row)
    return grid


RECAP = [
    (
        "tokens",
        "text is split into chunks by byte-pair encoding, and each becomes a number",
        "Tokens",
    ),
    ("embeddings", "each token looks up a vector: the start of the residual stream", "Embeddings"),
    ("attention", "each position asks (query), is found (key) and hands over (value)", "Attention"),
    ("RoPE", "queries and keys are turned by position, so distance changes the match", "Position"),
    ("softmax", "a score for every possible next token becomes a probability", "Softmax"),
    ("MLP", "each position works on what it gathered, through a gated middle layer", "The stack"),
    (
        "the stack",
        "layers add onto a shared stream, and later ones build on earlier ones",
        "The stack",
    ),
    ("loss", "how surprised it was by the right answer", "Loss"),
    ("backprop", "which way every weight should move, by the chain rule", "Backpropagation"),
    (
        "AdamW",
        "move them all a little way downhill, with momentum and weight decay",
        "Gradient descent",
    ),
    ("pretraining", "do that thousands of times, on lots of text", "Pretraining"),
    ("overfitting", "…but not so much, or so narrowly, that it memorizes", "Memorizing"),
    (
        "induction heads",
        "two layers of attention, composing, copy from earlier in the text",
        "Why layers stack",
    ),
    ("inference", "sample a token, add it, repeat, with the weights frozen", "Inference"),
    (
        "KV cache",
        "keep every position's keys and values, instead of recomputing them",
        "The KV cache",
    ),
    ("quantization", "store each weight in fewer bits", "Quantization"),
    (
        "fine-tuning",
        "a little more training, on examples of what you want, maybe with LoRA",
        "Fine-tuning",
    ),
    ("chat", "fine-tune on conversations in a template, grading only the replies", "Chat"),
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
