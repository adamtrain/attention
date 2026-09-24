"""Chapter: backpropagation, from one neuron to the whole transformer."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..model import PARTS, cross_entropy, linear_backward
from ..scalar import Value, neuron
from ..views import token_chip
from .code import excerpt
from .lab import Lab
from .stage import Frame, Stage, hold

W0, B0, LR = 0.5, 0.0, 0.05


def run(stage: Stage, lab: Lab) -> None:
    n = lab.model.size
    stage.say(
        f"Your model has {n:,} weights. To make the loss smaller, which way should each one "
        "move, and how far? [b]Backpropagation[/b] answers that for every weight at once, "
        "using the chain rule from calculus, run backwards through the calculation."
    )
    stage.say(
        "Start with the smallest model there is: one input [b]x[/b], two weights [b]w[/b] and "
        "[b]b[/b], a prediction [b]p = w·x + b[/b], and a loss [b]L = (p − y)²[/b], where "
        "[b]y[/b] is the right answer."
    )
    stage.play(graph_story(), fps=1, start="run it forward, then backward")
    stage.say(
        "Each number in amber is a [b]gradient[/b]: how much the loss would change if that "
        "number were nudged up a little. w's is −8, so nudging w up by 0.01 would cut the "
        "loss by about 0.08. Every gradient is the slopes along the path back from L, "
        "multiplied together. That's the chain rule, and that's all backprop is."
    )
    stage.wait("follow the gradient")

    stage.say(
        "Now [b]gradient descent[/b]: move every weight a little way against its gradient, "
        f"with a step size called the [b]learning rate[/b] (here {LR}). Then do it again:"
    )
    stage.show(Text.assemble(
        ("w ← w − ", "bold"), (f"{LR}", f"bold {viz.AMBER}"), (" × ∂L/∂w", "bold"),
        ("      b ← b − ", "bold"), (f"{LR}", f"bold {viz.AMBER}"), (" × ∂L/∂b", "bold"),
    ))  # fmt: skip
    stage.play(descent(), fps=1, start="take the steps")
    stage.say(
        "The prediction closes in on the right answer, 3. That's learning. Everything else "
        "is scale: your transformer does exactly this with its thousands of weights."
    )
    stage.wait("see it in the transformer")

    inputs, targets = lab.example()
    tr = lab.model.forward(inputs)
    focus = max(1, round(len(lab.corpus.example) * 0.65))
    seen = lab.vocab.decode(inputs[0, : focus + 1])
    right = int(targets[0, focus])
    probs = tr.probs[0, focus]
    stage.say(
        "In the transformer, backprop starts at the very end, with the probabilities. For the "
        f"quiz `{seen}` → `{lab.vocab.chars[right]}`, the gradient for each token's score is "
        "wonderfully simple: its probability, minus 1 if it's the right answer."
    )
    stage.show(output_gradient(lab, probs, right))
    stage.say(
        "Negative means [green]push this score up[/green]: that's the right answer. Every "
        "other score gets pushed [red]down[/red], a little, in proportion to how likely the "
        "model thought it was."
    )
    stage.wait()

    loss, dlogits = cross_entropy(tr.logits, targets)
    grads, _ = lab.model.backward(tr, dlogits)
    stage.say(
        "Then each layer's backward function takes the gradient of its output and works out "
        "the gradient of its input and its weights, and hands the first one back to the layer "
        f"before. Here it is flowing back from the loss ({loss:.2f}) through every weight:"
    )
    stage.play(flow(grads, stage.width), fps=8, start="send it back through the layers")
    stage.wait()

    stage.say(
        "Where did it end up? This is the token embedding table again. Green shows which way "
        f"each number will move (up), red down. Only the letters in "
        f"{lab.corpus.example.capitalize()} got a push. A model can only learn from what it sees."
    )
    stage.show(embedding_push(lab, -grads["embed.token"].T, set(inputs[0].tolist())))
    stage.say(
        "Every layer has a backward function. Here's the one for multiplying by a grid of "
        "weights, which does most of the work:"
    )
    stage.show(excerpt(linear_backward))


# ── The neuron ────────────────────────────────────────────────────────────────

LEAVES = {"w": (1, 0), "x": (1, 3), "b": (1, 5), "y": (1, 7)}
NODES = {"w·x": (16, 2), "p": (32, 4), "e": (48, 6), "L": (62, 6)}
GRADS = {
    "w": (4, 1),
    "x": (4, 4),
    "b": (4, 6),
    "w·x": (20, 3),
    "p": (34, 5),
    "e": (50, 7),
    "L": (64, 7),
}
WIRES = [
    (10, 0, "──╮"), (12, 1, "│"), (12, 2, "×──"), (10, 3, "──╯"), (28, 3, "│"),
    (25, 2, " ──╮"), (28, 4, "+──"), (10, 5, "─" * 18 + "╯"), (44, 5, "│"),
    (39, 4, " ────╮"), (44, 6, "−──"), (10, 7, "─" * 34 + "╯"), (55, 6, " ──²──"),
]  # fmt: skip
OPS = {"w·x": (12, 2), "p": (28, 4), "e": (44, 6), "L": (58, 6)}


def graph(v: dict[str, Value], shown: set[str], grads: set[str], active: str | None) -> Text:
    canvas = viz.Canvas(70, 8)
    for x, y, s in WIRES:
        canvas.put(x, y, s, viz.FAINT)
    for x, y in OPS.values():
        canvas.restyle(x, y, 1, "bold")
    for name, (x, y) in {**LEAVES, **NODES}.items():
        learnable = name in ("w", "b")
        hot = name == active
        label_style = (
            f"bold {viz.ACCENT}" if hot else "bold" if learnable or name in NODES else viz.FAINT
        )
        canvas.put(x, y, name, label_style)
        if name in shown:
            vx = x + len(name) + 1 if name in NODES else 4
            canvas.put(vx, y, f"{v[name].data:5.2f}", f"bold {viz.ACCENT}" if hot else "")
        if name in grads and name in GRADS:
            gx, gy = GRADS[name]
            canvas.put(
                gx,
                gy,
                f"{v[name].grad:5.2f}",
                f"bold {viz.AMBER}" if hot else viz.AMBER,
            )
    if active in OPS:
        x, y = OPS[active]
        canvas.restyle(x, y, 1, f"bold {viz.ACCENT}")
    return canvas.text()


def graph_story() -> Iterator[Frame]:
    v = neuron(W0, B0)
    v["L"].backward()
    leaves = {"w", "x", "b", "y"}

    def frame(
        shown: set[str],
        grads: set[str],
        active: str | None,
        caption: str,
        seconds: float,
    ):
        legend = Text.assemble(
            ("values", "bold"), (" are computed forward →    ", viz.FAINT), ("gradients", f"bold {viz.AMBER}"),
            (" are computed backward ←", viz.FAINT),
        )  # fmt: skip
        body = Group(
            graph(v, shown, grads, active),
            Text(""),
            legend,
            Text(""),
            Text.from_markup(caption),
        )
        return hold(body, seconds)

    d = {k: val.data for k, val in v.items()}
    g = {k: val.grad for k, val in v.items()}
    yield frame(
        leaves,
        set(),
        None,
        "[b]Forward.[/b] Start with the inputs, the weights and the answer.",
        2.0,
    )
    forward = [
        ("w·x", f"w·x = {d['w']:.2f} × {d['x']:.2f} = [b]{d['w·x']:.2f}[/b]"),
        (
            "p",
            f"p = w·x + b = {d['w·x']:.2f} + {d['b']:.2f} = [b]{d['p']:.2f}[/b]   the prediction",
        ),
        (
            "e",
            f"e = p − y = {d['p']:.2f} − {d['y']:.2f} = [b]{d['e']:.2f}[/b]   the error",
        ),
        ("L", f"L = e² = [b]{d['L']:.2f}[/b]   the loss"),
    ]
    shown = set(leaves)
    for name, caption in forward:
        shown.add(name)
        yield frame(shown, set(), name, caption, 1.6)
    back = [
        ({"L"}, "L", "[b]Backward.[/b] Start at the end: nudge L, and L moves the same amount. Gradient [amber]1[/amber]."),
        ({"e"}, "e", f"L = e², whose slope is 2e: 2 × {d['e']:.2f} × 1 = [amber]{g['e']:.2f}[/amber]"),
        ({"p"}, "p", f"e = p − y: nudge p and e moves just as much, so p gets e's gradient: [amber]{g['p']:.2f}[/amber]"),
        ({"w·x", "b"}, "p", f"p = w·x + b: a sum hands its gradient to both inputs: [amber]{g['b']:.2f}[/amber] each"),
        ({"w", "x"}, "w·x", f"w·x: nudge w and it moves x times as much: {d['x']:.2f} × {g['w·x']:.2f} = [amber]{g['w']:.2f}[/amber]"),
    ]  # fmt: skip
    grads: set[str] = set()
    for names, active, caption in back:
        grads |= names
        yield frame(shown, set(grads), active, caption, 2.4)
    yield frame(shown, grads, None, "Every gradient, from one pass backward.", 0.2)


def descent(steps: int = 7) -> Iterator[Frame]:
    grid_rows: list[tuple[str, ...]] = []
    w, b = W0, B0
    first = None
    for step in range(steps + 1):
        v = neuron(w, b)
        v["L"].backward()
        loss = v["L"].data
        first = first or loss
        grid_rows.append(
            (
                str(step),
                f"{w:.2f}",
                f"{b:.2f}",
                f"{v['p'].data:.2f}",
                f"{loss:.3f}",
                str(loss / first),
            )
        )
        yield hold(descent_table(grid_rows), 0.9)
        w -= LR * v["w"].grad
        b -= LR * v["b"].grad


def descent_table(rows: list[tuple[str, ...]]) -> Table:
    grid = Table.grid(padding=(0, 3))
    for _ in range(6):
        grid.add_column(justify="right", no_wrap=True)
    grid.add_row(*(Text(h, style=viz.FAINT) for h in ("step", "w", "b", "p", "loss", "")))
    for i, (step, w, b, p, loss, frac) in enumerate(rows):
        latest = i == len(rows) - 1
        grid.add_row(
            Text(step, style=viz.FAINT), Text(w, style="bold" if latest else ""), Text(b, style="bold" if latest else ""),
            Text(p), Text(loss, style=f"bold {viz.GREEN}" if latest else ""), viz.bar(float(frac), 24, viz.GREEN, track=False),
        )  # fmt: skip
    return grid


# ── The transformer ───────────────────────────────────────────────────────────


def output_gradient(lab: Lab, probs: np.ndarray, right: int, top: int = 7) -> Table:
    grad = probs.copy()
    grad[right] -= 1.0
    order = [right, *[int(i) for i in np.argsort(-probs) if int(i) != right][: top - 1]]
    grid = Table.grid(padding=(0, 2))
    for justify in ("left", "right", "right", "right", "left", "left"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(
        *(Text(h, style=viz.FAINT) for h in ("", "probability", "right?", "gradient", "", ""))
    )
    for i in order:
        is_right = i == right
        grid.add_row(
            token_chip(lab.vocab, i),
            viz.percent(float(probs[i])),
            Text("1" if is_right else "0", style="bold" if is_right else viz.FAINT),
            Text(f"{grad[i]:+.3f}", style="bold"),
            viz.signed_bar(float(grad[i]), 1.0, 12, neg=viz.GREEN, pos=viz.RED),
            Text(
                "push up" if is_right else "push down",
                style=viz.GREEN if is_right else viz.RED,
            ),
        )
    grid.add_row(Text("…", style=viz.FAINT), "", "", "", "", "")
    return grid


def flow(grads: dict[str, np.ndarray], width: int) -> Iterator[Frame]:
    names = list(PARTS)
    norms = {n: float(np.sqrt((grads[n] ** 2).sum())) for n in names}
    top = max(norms.values())
    decades = 3  # the bars are on a log scale, or the biggest would dwarf the rest

    def length(norm: float) -> float:
        return max(0.0, 1 + np.log10(max(norm, 1e-12) / top) / decades)

    bar_w = max(10, min(30, width - 50))
    for revealed in range(len(names) + 1):
        grid = Table.grid(padding=(0, 2))
        for justify in ("left", "left", "left", "right", "left"):
            grid.add_column(justify=justify, no_wrap=True)
        grid.add_row("", "", Text("size of the gradient (log scale)", style=viz.FAINT), "", "")
        last_layer = ""
        for i, name in enumerate(names):
            part = PARTS[name]
            shown = i >= len(names) - revealed
            front = i == len(names) - revealed
            layer = part.layer if part.layer != last_layer else ""
            last_layer = part.layer
            grid.add_row(
                Text(layer, style="bold"),
                Text(part.label, style="" if shown else viz.FAINT),
                viz.bar(length(norms[name]), bar_w, viz.AMBER)
                if shown
                else Text("─" * bar_w, style=viz.FAINT),
                Text(f"{norms[name]:.3f}" if shown else "", style=viz.FAINT),
                Text(
                    "◀ starts here"
                    if name == names[-1]
                    else "▲"
                    if front and 0 < revealed < len(names)
                    else "",
                    style=viz.AMBER,
                ),
            )
        yield grid, (0.7 if revealed == 0 else 0.25)
    yield hold(grid, 0.3)


def embedding_push(lab: Lab, push: np.ndarray, used: set[int]) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    lines = viz.nudge_blocks(push, width=2)
    for line in lines:
        grid.add_row(line)
    letters = Text(no_wrap=True)
    for i, ch in enumerate(lab.vocab.chars):
        letters.append(f"{ch:<2}", style=f"bold {viz.ACCENT}" if i in used else viz.FAINT)
    grid.add_row(letters)
    grid.add_row(Text(""))
    grid.add_row(viz.legend(viz.NUDGE, "down", "up"))
    return grid
