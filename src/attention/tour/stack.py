"""Chapter: the whole model. One block, then a stack of them."""

from __future__ import annotations

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import BLOCK, Transformer, silu
from ..views import label, prob_bars
from .code import excerpt
from .lab import Lab
from .stage import Stage

# Llama 3 8B's config.json, for comparison.
LLAMA = {
    "vocab_size": "128,256",
    "hidden_size": "4,096",
    "num_hidden_layers": "32",
    "num_attention_heads": "32",
    "num_key_value_heads": "8",
    "head_dim": "128",
    "intermediate_size": "14,336",
    "max_position_embeddings": "8,192",
    "rope_theta": "500,000",
    "tie_word_embeddings": "false",
    "hidden_act": "silu",
}
MEANING = {
    "vocab_size": "tokens in the vocabulary",
    "hidden_size": "numbers per token in the stream",
    "num_hidden_layers": "blocks in the stack",
    "num_attention_heads": "query heads per layer",
    "num_key_value_heads": "key/value heads per layer",
    "head_dim": "numbers per head",
    "intermediate_size": "width of the MLP's middle",
    "max_position_embeddings": "the context window",
    "rope_theta": "how slowly RoPE's slowest hand turns",
    "tie_word_embeddings": "one table for input and output?",
    "hidden_act": "the MLP's switch",
}


def run(stage: Stage, lab: Lab) -> None:
    c = lab.model.config
    inputs, targets = lab.example()
    t = inputs.shape[1]
    tr = lab.model.forward(inputs)
    stage.say(
        "Here's the whole model, top to bottom. The purple numbers on the right are the shape "
        f"of the grid flowing through it for the {t} tokens of your example: rows, times "
        "numbers per row."
    )
    stage.show(diagram(lab, t, stage.width))
    stage.wait()

    focus = t - 1
    name = label(lab.tokenizer, int(inputs[0, focus]))
    block = tr.blocks[0]
    stage.say(
        "You've met the embeddings and attention. The new box is the [b]MLP[/b] (short for "
        "multi-layer perceptron, an old name for the simplest kind of neural network). "
        "Attention moves information [i]between[/i] positions. The MLP then works on what each "
        "position has gathered, one position at a time, without looking at the others."
    )
    stage.say(
        f"Your model's MLP is the kind nearly every modern model uses, called [b]SwiGLU[/b]. "
        f"It has three grids. Two of them, [b]gate[/b] and [b]up[/b], each turn a position's "
        f"{c.width} numbers into {c.hidden}: {c.hidden} different weighted sums, like "
        "detectors, each looking for some pattern in the vector. The gate's go through a "
        "switch, called SiLU, which lets positive numbers through and squashes negative ones "
        "to about zero. Then the two are multiplied, number by number: the gate decides how "
        "much of each [i]up[/i] detector gets through. The third grid, [b]down[/b], mixes the "
        f"results back into {c.width} numbers, which are added onto the stream. Here it is at "
        f"`{name}`, in layer 1:"
    )
    stage.show(
        mlp(
            block.m_in[0, focus],
            block.gate[0, focus],
            block.up[0, focus],
            block.out[0, focus] - block.mid[0, focus],
            stage.width,
        )
    )
    stage.say(
        "The switch is what makes it more than arithmetic. Grids of weights in a row, with "
        "nothing between them, can only ever do what a single grid could; the on-or-off step "
        "lets the network react to combinations, like “a name, right after `said`”. Older models "
        "used just two grids with a plain switch between them; the gate works better for the "
        "same number of weights, so Llama, Mistral, Qwen and Gemma all use it. Research "
        "suggests much of what a big model knows, facts included, is stored in its MLPs."
    )
    stage.wait()

    stage.say(
        "The [b]⊕[/b] is a [b]residual connection[/b]: each block's output is [i]added[/i] onto "
        "its input instead of replacing it. Picture the grid running down the left side as a "
        "shared notebook, the residual stream. Each block reads it and adds its notes back in. "
        "Whatever a block doesn't change flows straight past it. That also gives backprop a "
        "clear path back down, which is a big part of why stacks of dozens of blocks can be "
        "trained at all."
    )
    stage.say(
        "Before each block reads the notebook, a [b]norm[/b] (RMSNorm) rescales each "
        "position's vector to a steady size without changing its direction, so no block gets "
        "thrown by numbers that happen to be huge or tiny. Only the block's copy is rescaled; "
        "the notebook itself isn't."
    )
    stage.say(
        f"Your model has {c.layers} of these blocks, one after another. They're built the same "
        "way, but each has its own weights, so each can do a different job. What does stacking "
        "buy? Each layer can build on what the layers before it wrote. The first layer's "
        "attention can only find tokens by what they are. The second can find them by what the "
        "first layer [i]wrote about them[/i]: say, “I come right after `Fox`”. You'll see "
        "exactly that, trained, in the chapter on why layers stack. Big models stack many "
        "more: Llama 3 8B has 32 layers, GPT-3 had 96."
    )
    stage.wait()

    stage.say(
        "At the bottom, one last norm, and then the [b]unembedding[/b]: a score for each of the "
        f"{c.vocab} tokens, called its [b]logit[/b]. Your model reuses the embedding table for "
        "this: each token's score is the dot product of the final vector with that token's row. "
        "A vector ends up meaning “the tokens it points toward”. That's called tying the "
        "embeddings; small models often do it to save parameters, and big ones often don't. "
        "Softmax turns the scores into probabilities."
    )
    stage.say(
        "Because every layer adds onto the same stream, you can stop after any layer and read "
        "the stream the same way: norm, unembed, softmax. That shows what the model would guess "
        "if it stopped there. It's called the [b]logit lens[/b], and once your model is trained "
        "you'll use it to watch a prediction take shape, layer by layer."
    )
    stage.wait("count the parameters")

    stage.say(f"Where its {lab.model.size:,} parameters live:")
    stage.show(breakdown(lab, stage.width))
    table = lab.model.params["embed"].size / lab.model.size
    stage.say(
        "Most of the model is in its layers, and most of each layer is its MLP. "
        + (
            f"The embedding table is a big slice too, {table:.0%}: {c.vocab:,} tokens is a lot "
            "of rows for a model this narrow. Llama 3 8B has 128,256 tokens, yet its embedding "
            "and unembedding tables come to only 13% of it: a table grows in step with the "
            "numbers per token, but a layer's grids grow with the square of them."
            if table > 0.2
            else "Big models are the same: Llama 3 8B's embedding and unembedding tables, for "
            "all their 128,256 tokens, come to only 13% of it."
        )
    )
    stage.wait()

    stage.say(
        "Every model shared on Hugging Face comes with a file called config.json, which sets "
        "these sizes. Here are your model's, under the names a Llama model's file uses, beside "
        "Llama 3 8B's:"
    )
    stage.show(config_table(lab))
    stage.say(
        "Every one of those is now something you've seen. The architecture is the same; the "
        "difference is size. Here's your model's entire forward pass, from its source code:"
    )
    stage.show(excerpt(Transformer.forward, "# 1. Look up", "return Trace"))
    stage.wait()

    right = int(targets[0, focus])
    probs = tr.probs[0, focus]
    stage.say("Let's run it. After the example, the untrained model thinks the next token is:")
    stage.show(prob_bars(probs, lab.tokenizer, top=8, width=30, right=right))
    stage.say(
        f"Every token gets about 1 in {c.vocab}, around {1 / c.vocab:.1%}. It hasn't learned a "
        "thing yet. Its weights are random, so its guesses are too."
    )


def mlp(x: np.ndarray, gate: np.ndarray, up: np.ndarray, out: np.ndarray, width: int) -> Group:
    """One position through the MLP: widen twice, switch one, multiply, narrow."""
    switched = silu(gate)
    mixed = switched * up
    room = width - len("SiLU(gate)") - 2
    shown = min(len(gate), 2 * room)  # as many detectors as fit, two to a character at most
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(no_wrap=True)
    wide = viz.scale_of(np.concatenate([gate, up]))
    grid.add_row("in", viz.signed_fitted(x, viz.scale_of(x), room))
    grid.add_row("gate", viz.signed_fitted(gate[:shown], wide, room))
    grid.add_row("SiLU(gate)", viz.signed_fitted(switched[:shown], wide, room))
    grid.add_row("up", viz.signed_fitted(up[:shown], wide, room))
    grid.add_row("SiLU × up", viz.signed_fitted(mixed[:shown], viz.scale_of(mixed), room))
    grid.add_row("down", viz.signed_fitted(out, viz.scale_of(out), room))
    open_ = int((switched > 0.1 * np.abs(switched).max()).sum())
    some = f" (the first {shown} shown)" if shown < len(gate) else ""
    caption = Text.assemble(
        (f"{len(x)} numbers in; {len(gate)} gate and up detectors{some}; ", viz.FAINT),
        (f"{open_} gates open", "bold"),
        (f"; {len(out)} numbers out, added onto the stream", viz.FAINT),
    )
    return Group(grid, Text(""), caption)


def diagram(lab: Lab, t: int, width: int) -> Table:
    c = lab.model.config
    box, wire = viz.ACCENT, viz.FAINT
    rows: list[tuple[Text, str | Text, str]] = []

    def line(*parts: tuple[str, str]) -> Text:
        return Text.assemble(*parts, no_wrap=True)

    def block(name: str, what: str, shape: str, extra: str = "") -> None:
        rows.append((line(("╭" + "─" * 16 + "╮", box)), "", ""))
        rows.append((line(("│", box), (f"{name:^16}", "bold"), ("│", box)), what, shape))
        if extra:
            rows.append((line(("│", box), (f"{extra:^16}", viz.FAINT), ("│", box)), "", ""))
        rows.append((line(("╰" + "─" * 7 + "┬" + "─" * 8 + "╯", box)), "", ""))

    def sub(name: str, what: str) -> None:
        rows.append((line(("        ├" + "─" * 10 + "╮", wire)), "", ""))
        rows.append(
            (line(("        │  ", wire), ("╭" + "─" * 7 + "┴" + "─" * 8 + "╮", box)), "", "")
        )
        rows.append(
            (
                line(("        │  ", wire), ("│", box), (f"{'norm':^16}", viz.FAINT), ("│", box)),
                "",
                "",
            )
        )
        rows.append(
            (
                line(("        │  ", wire), ("│", box), (f"{name:^16}", "bold"), ("│", box)),
                what,
                f"{t} × {c.width}",
            )
        )
        rows.append(
            (line(("        │  ", wire), ("╰" + "─" * 7 + "┬" + "─" * 8 + "╯", box)), "", "")
        )
        rows.append(
            (
                line(("        ⊕", f"bold {viz.AMBER}"), ("◂" + "─" * 9 + "╯", wire)),
                "add onto the stream",
                "",
            )
        )

    rows.append((line(("      tokens", "bold")), "", f"{t}"))
    rows.append((line(("        │", wire)), "", ""))
    block("embeddings", "look up each token's row", f"{t} × {c.width}")
    rows.append((line(("        │", wire), ("  the residual stream", viz.FAINT)), "", ""))
    rows.append((line(("  ┌ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ┐", viz.PURPLE)), "", ""))
    sub("attention", "look back at earlier tokens")
    sub("MLP", "work on each position")
    rows.append(
        (
            line(("  └ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ┘", viz.PURPLE)),
            Text(f"one layer: {c.layers} in all, stacked", style=viz.PURPLE),
            "",
        )
    )
    rows.append((line(("        │", wire)), "", ""))
    block("unembed", "a score for every token", f"{t} × {c.vocab}", "(the same table)")
    rows.append((line(("        │", wire)), "", ""))
    rows.append(
        (line(("     softmax", "bold")), "probabilities for the next token", f"{t} × {c.vocab}")
    )

    widest = [max(len(str(r[i]).rstrip()) for r in rows) for i in range(3)]
    grid = Table.grid(padding=(0, 3 if sum(widest) + 6 <= width else 2))
    grid.add_column(no_wrap=True)
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(style=viz.PURPLE, no_wrap=True, justify="right")
    for graphic, what, shape in rows:
        grid.add_row(graphic, what, shape)
    return grid


def breakdown(lab: Lab, width: int) -> Table:
    p, c = lab.model.params, lab.model.config

    def count(names: list[str]) -> int:
        return sum(p[f"layers.{i}.{n}"].size for i in range(c.layers) for n in names)

    attention = ["attn.query", "attn.key", "attn.value", "attn.out"]
    mlp_names = ["mlp.gate", "mlp.up", "mlp.down"]
    groups = [
        ("embeddings", f"{c.vocab} tokens × {c.width} (and the unembedding)", p["embed"].size),
        ("attention", f"{c.layers} layers × query, key, value, output", count(attention)),
        ("MLP", f"{c.layers} layers × gate, up, down", count(mlp_names)),
        (
            "norms",
            f"{2 * c.layers + 1} × {c.width}",
            count(["attn.norm", "mlp.norm"]) + p["norm"].size,
        ),
    ]
    assert sum(n for *_, n in groups) == lab.model.size and len(BLOCK) == 9
    biggest = max(n for *_, n in groups)
    total = f"{lab.model.size:,}"
    beside = max(len(g) for g, *_ in groups) + max(len(d) for _, d, _ in groups) + len(total) + 6
    bar = max(6, min(20, width - beside))
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    for name, detail, n in groups:
        grid.add_row(name, detail, f"{n:,}", viz.bar(n / biggest, bar, viz.PURPLE, track=False))
    grid.add_row("", Text("total", style="bold"), Text(total, style="bold"), "")
    return grid


def config_table(lab: Lab) -> Table:
    mine = lab.model.config.hugging_face()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(style=viz.FAINT)  # the one that wraps, in a narrow terminal
    grid.add_row("", Text("yours", style="bold"), Text("Llama 3 8B", style="bold"), "")
    for key, theirs in LLAMA.items():
        grid.add_row(key, shown(mine[key]), theirs, MEANING[key])
    return grid


def shown(value: object) -> str:
    """A config value the way it reads in a table."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
