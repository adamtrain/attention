"""Chapter: position, and rotary position embeddings (RoPE)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import angles, rmsnorm, rope, softmax, split_heads
from ..views import label
from .code import excerpt
from .lab import Lab
from .stage import Frame, Stage, hold

ARROWS = "→↗↑↖←↙↓↘"  # a direction for every eighth of a turn
ROWS = 16  # positions shown on the dials


def run(stage: Stage, lab: Lab) -> None:
    c = lab.model.config
    stage.say(
        "Something is missing from attention as you've seen it. A query scores a key the same "
        "wherever that key sits: the dot product only looks at the two vectors. Shuffle the "
        "earlier tokens and each one still gets exactly the same weight, so the blend comes out "
        "exactly the same. “The Fox ate the Hen” and “The Hen ate the Fox” would look alike."
    )
    order, shuffled, plain_w, rope_w = shuffle_test(lab)
    stage.say(
        "Here's layer 1's first head, at the last token, reading two orders of the same tokens:"
    )
    stage.show(shuffle_view(lab, order, shuffled, plain_w, rope_w))
    stage.say(
        "Without position, each token gets the same weight in either order. With it, the weights "
        "change, because now it matters where each token is."
    )
    stage.say(
        "GPT-2 and GPT-3 fixed this by adding a learned position vector to every token at the "
        "start, one for each place in the context (GPT-2 had 1,024 of them, and couldn't read "
        "any further). Today's models use [b]RoPE[/b], rotary position embedding, instead. It "
        "adds nothing. It [i]turns[/i] the queries and keys."
    )
    stage.wait("see it turn")

    speeds = angles(np.array([1.0]), c.head_width, c.rope_base)[0]
    stage.say(
        f"Each head's queries and keys have {c.head_width} numbers. Take them in pairs, "
        f"{c.head_width // 2} pairs, and treat each pair as the tip of a clock hand. RoPE turns "
        "every hand by an angle that grows with the position: the first pair turns fast, "
        f"{speeds[0]:.0f} radian (about 57°) per position, and each pair after it turns more "
        f"slowly, down to {speeds[-1]:.4f} radians per position for the last. Here are the "
        f"{c.head_width // 2} hands at the first {ROWS} positions:"
    )
    stage.play(lambda: dials(speeds), fps=6, start="turn them")
    stage.say(
        "The fast hands spin round and round, like a clock's second hand; the slow ones hardly "
        "move over a few positions, like its hour hand. Between them, every position gets its "
        "own combination of angles."
    )
    stage.wait("see why that works")

    distances, scores = by_distance(lab)
    stage.say(
        "Here's the clever part. When a query at position m meets a key at position n, the "
        "query has been turned by m steps and the key by n. A dot product only cares about the "
        "angle [i]between[/i] two vectors, so turning both by the same amount changes nothing: "
        "what's left depends only on n − m, how far apart they are. Here's one real query and "
        "one real key from your model, scored at every distance apart:"
    )
    stage.show(distance_plot(distances, scores, min(60, stage.width - 12)))
    stage.say(
        "Move the pair anywhere in the text, and at the same distance they score the same. So "
        "a head can learn “look at the token just before me” or “look about ten back”, and that "
        "works at every position. Nothing else in the model knows where it is."
    )
    stage.say(
        "Notice what RoPE turns: only the [b]queries and keys[/b], never the values. Position "
        "decides [i]where[/i] a head looks; it doesn't change [i]what[/i] gets handed over. In "
        "your model's code, it's the two lines with `rope` in them:"
    )
    stage.show(excerpt(lab.model.block, "q = rope", "if cache"))
    stage.show(excerpt(rope))
    llama = 500_000.0 ** (-126 / 128)  # rope_theta 500,000, heads of 128 numbers
    stage.note(
        f"Your model reads at most {c.context} tokens, so it has only ever seen distances up to "
        f"{c.context - 1}. Llama 3 sets its rope_theta to 500,000 instead of 10,000, and its "
        f"slowest hand turns about {speeds[-1] / llama:,.0f} times more slowly than yours, so "
        "that even across its 128,000-token context no two positions look alike. Tricks with "
        "names like YaRN stretch the angles to read further than a model was trained on."
    )


# ── Order ─────────────────────────────────────────────────────────────────────


def shuffle_test(
    lab: Lab, shown: int = 7
) -> tuple[list[int], list[int], list[np.ndarray], list[np.ndarray]]:
    """The last token's attention to the rest, in two orders, without and with RoPE."""
    inputs, _ = lab.example()
    ids = [int(t) for t in inputs[0, : shown + 1]]
    middle = ids[1:-1]
    shuffled = [ids[0], *middle[::-1], ids[-1]]
    plain, turned = [], []
    for order in (ids, shuffled):
        p, turn = weights(lab, order)
        plain.append(p)
        turned.append(turn)
    return ids, shuffled, plain, turned


def weights(lab: Lab, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Layer 1, head 1: the last position's attention, without RoPE and with it."""
    c, p = lab.model.config, lab.model.params
    x = p["embed"][np.array([ids])]
    a_in, _ = rmsnorm(x, p["layers.0.attn.norm"])
    q = split_heads(a_in @ p["layers.0.attn.query"], c.heads)[0, 0]
    k = split_heads(a_in @ p["layers.0.attn.key"], c.kv_heads)[0, 0]
    turn = angles(np.arange(len(ids)), c.head_width, c.rope_base)
    plain = softmax(q[-1] @ k.T / np.sqrt(c.head_width))
    turned = softmax(rope(q, turn)[-1] @ rope(k, turn).T / np.sqrt(c.head_width))
    return plain, turned


def shuffle_view(lab: Lab, order, shuffled, plain, turned) -> Table:
    tok = lab.tokenizer
    grid = Table.grid(padding=(0, 2))
    for justify in ("right", "right", "right", "right", "right"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(
        "", Text("without position", style="bold"), "", Text("with RoPE", style="bold"), "",
    )  # fmt: skip
    grid.add_row(
        Text("token", style=viz.FAINT), Text("in order", style=viz.FAINT), Text("shuffled", style=viz.FAINT),
        Text("in order", style=viz.FAINT), Text("shuffled", style=viz.FAINT),
    )  # fmt: skip
    for token in order[:-1]:
        i, j = order.index(token), shuffled.index(token)
        same = abs(plain[0][i] - plain[1][j]) < 1e-9
        grid.add_row(
            Text(label(tok, token), style=f"bold {viz.ACCENT}"),
            Text(viz.percent(float(plain[0][i]))),
            Text(viz.percent(float(plain[1][j])), style=viz.GREEN if same else ""),
            Text(viz.percent(float(turned[0][i]))),
            Text(viz.percent(float(turned[1][j])), style=viz.AMBER),
        )
    return grid


# ── The dials ─────────────────────────────────────────────────────────────────


def arrow(angle: float) -> str:
    return ARROWS[round(angle / (np.pi / 4)) % 8]


def dials(speeds: np.ndarray) -> Iterator[Frame]:
    def frame(shown: int) -> Table:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="right", style=viz.FAINT, no_wrap=True)
        for _ in speeds:
            grid.add_column(justify="center", no_wrap=True)
        grid.add_row(
            Text("pair", style=viz.FAINT),
            *(Text(f"{j + 1}", style="bold") for j in range(len(speeds))),
        )
        for pos in range(ROWS):
            if pos < shown:
                row = [
                    Text(
                        arrow(pos * s),
                        style=f"bold {viz.mix(viz.AMBER, viz.BLUE, j / (len(speeds) - 1))}",
                    )
                    for j, s in enumerate(speeds)
                ]
            else:
                row = [Text("") for _ in speeds]
            grid.add_row(f"position {pos}", *row)
        grid.add_row(
            Text("turns", style=viz.FAINT),
            *(
                Text("fast" if j == 0 else "slow" if j == len(speeds) - 1 else "", style=viz.FAINT)
                for j in range(len(speeds))
            ),
        )
        return grid

    yield frame(1)
    for shown in range(2, ROWS + 1):
        yield frame(shown), 0.25
    yield hold(frame(ROWS), 0.3)


# ── Distance ──────────────────────────────────────────────────────────────────


def by_distance(lab: Lab, far: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One query and one key from the model, scored at every distance apart (and two spots)."""
    c, p = lab.model.config, lab.model.params
    inputs, _ = lab.example()
    x = p["embed"][inputs]
    a_in, _ = rmsnorm(x, p["layers.0.attn.norm"])
    q = split_heads(a_in @ p["layers.0.attn.query"], c.heads)[0, 0, -1]
    k = split_heads(a_in @ p["layers.0.attn.key"], c.kv_heads)[0, 0, 1]
    far = far or c.context - 1
    distances = np.arange(0, far + 1)
    at = far  # the query sits at the end; the key moves back
    qa = rope(q, angles(np.array([at]), c.head_width, c.rope_base)[0])
    keys = rope(np.tile(k, (len(distances), 1)), angles(at - distances, c.head_width, c.rope_base))
    return distances, keys @ qa / np.sqrt(c.head_width)


def distance_plot(distances: np.ndarray, scores: np.ndarray, width: int) -> Group:
    lo, hi = float(scores.min()), float(scores.max())
    pad = (hi - lo) * 0.1 + 1e-6
    plot = viz.Plot(width, 7, x_max=float(distances[-1]), lo=lo - pad, hi=hi + pad)
    plot.line(list(zip(distances.astype(float), scores, strict=True)), viz.ACCENT)
    axis = Text.assemble(
        ("       0", viz.FAINT),
        (" " * (width - 18)),
        (f"{int(distances[-1])} tokens apart", viz.FAINT),
    )
    return Group(Text("  score", style=viz.FAINT), *plot.render(fmt="{:+.1f}"), axis)
