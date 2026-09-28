"""Chapter: what the model learned. Before and after, embeddings, heads, and the logit lens."""

from __future__ import annotations

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..views import attention_grid, chip, label
from .lab import Lab
from .loss import quiz_losses
from .stage import Stage

NEIGHBORS = [" said", " she", " the", " king", " Fox", " Lion", " lord", " good", " love", " man"]
SHOWN = 16  # tokens of the example in the attention pictures


def run(stage: Stage, lab: Lab) -> None:
    stage.say("Remember the quizzes in your example? Before training, and now:")
    stage.show(before_after(lab, stage.width))
    stage.say(
        "Where it used to shrug at every token, it now knows what's coming much of the time. "
        "The biggest losses left are the genuinely hard calls, like the first word of a new "
        "sentence, where plenty of answers would do."
    )
    stage.wait()

    stage.say(
        "What did the embeddings learn? Tokens the model uses in similar ways should have "
        "rows that point in similar directions. Here, for a few tokens, are the ones whose rows "
        "point most nearly the same way (by [b]cosine similarity[/b]: 1 means the same "
        "direction, 0 unrelated):"
    )
    stage.show(neighbors(lab, stage.width))
    stage.say(
        "Nobody told it which words are alike. Tokens that turn up in the same kinds of places "
        "get pushed the same way by backprop, step after step, until their rows end up "
        "neighbors. In big models this goes much further: directions in the space come to mean "
        "things like plural, past tense, or female."
    )
    stage.wait("look at the heads")

    stats = habits(lab)
    stage.say(
        f"And attention? Your model has {len(stats)} heads, {lab.model.config.heads} in each "
        "layer. Here's what each one tends to do, averaged over held-back text: how much of its "
        "attention goes to the token itself, to the token just before, and how far back it "
        "looks on average:"
    )
    stage.show(habits_view(stats, stage.width))
    stage.say(habit_story(stats, lab.model.config.layers))
    inputs, _ = lab.example()
    ids = [int(t) for t in inputs[0, :SHOWN]]
    names = [label(lab.tokenizer, t) for t in ids]
    weights = lab.model.forward(np.array([ids])).weights[:, 0]  # (layers, heads, t, t)
    far = max(stats, key=lambda s: s.reach)
    prev = max(stats, key=lambda s: s.previous)
    stage.say("Two of them, reading your example:")
    stage.show(pair_view(weights, names, prev, far, stage.width))
    stage.say(
        "Nobody told any head what to look for. Those habits are simply what lowered the loss."
    )
    stage.wait("watch a guess form")

    lens = lab.model.lens(lab.model.forward(inputs))[:, 0]  # (depths, time, vocab)
    rows = pick_rows(lab, lens)
    guesses = guess_room(lab, rows, len(lens), stage.width) > 0
    stage.say(
        "Finally, the [b]logit lens[/b]: stop after each layer, read the residual stream as if "
        "it were the end of the model, and see what it would guess. Here's the right answer's "
        "probability at each depth, for some of the quizzes in your example"
        + (", with its top guess when that isn't the right one:" if guesses else ":")
    )
    stage.show(lens_view(lab, lens, rows, stage.width))
    stage.say(lens_story(lab, lens))


# ── Before and after ──────────────────────────────────────────────────────────


def before_after(lab: Lab, width: int) -> Table:
    grid = Table.grid(padding=(0, 4))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(Text("untrained", style=f"bold {viz.FAINT}"), Text("trained", style="bold"))
    bar = 6 if width < 110 else 14
    trained = 8 + 1 + 6 + bar  # the compact table beside it: chance, and loss with its bar
    grid.add_row(
        quiz_losses(lab, lab.initial, bar_width=bar, room=width - trained - 4),
        quiz_losses(lab, lab.model, bar_width=bar, compact=True),
    )
    return grid


# ── Embeddings ────────────────────────────────────────────────────────────────


def neighbors(lab: Lab, width: int, count: int = 5, per: int = 5) -> Group:
    tok = lab.tokenizer
    table = lab.model.params["embed"]
    unit = table / np.linalg.norm(table, axis=1, keepdims=True)
    freq = np.bincount(lab.data.train, minlength=len(tok))
    chosen = [tok.ids[w] for w in NEIGHBORS if w in tok.ids and freq[tok.ids[w]] >= 30][:count]
    if len(chosen) < count:  # fill up with the commonest whole words
        words = [
            int(i)
            for i in np.argsort(-freq)
            if tok.pieces[i].startswith(" ") and tok.pieces[i][1:].isalpha()
        ]
        chosen += [w for w in words if w not in chosen][: count - len(chosen)]
    near = {}
    for token in chosen:
        sims = unit @ unit[token]
        sims[token] = -2
        near[token] = [(int(i), float(sims[i])) for i in np.argsort(-sims)[:per]]

    def line(token: int, most: int) -> Text:
        row = Text.assemble(chip(tok, token), ("  is nearest  ", viz.FAINT), no_wrap=True)
        for i, sim in near[token][:most]:
            row.append_text(chip(tok, i))
            row.append(f" {sim:.2f}  ", style=viz.FAINT)
        row.rstrip()
        return row

    while per > 1 and any(len(line(token, per)) > width for token in chosen):
        per -= 1  # the same number on every line, as many as fit
    return Group(*(line(token, per) for token in chosen))


# ── Heads ─────────────────────────────────────────────────────────────────────


class Habit:
    def __init__(self, layer: int, head: int, weights: np.ndarray):
        # weights: (batch, time, time) for one head, over held-back text
        t = weights.shape[-1]
        idx = np.arange(1, t)
        self.layer, self.head = layer, head
        self.itself = float(weights[:, idx, idx].mean())
        self.previous = float(weights[:, idx, idx - 1].mean())
        distance = np.arange(t)[:, None] - np.arange(t)[None, :]
        self.reach = float((weights[:, 1:] * np.maximum(distance, 0)[1:]).sum(-1).mean())

    @property
    def name(self) -> str:
        return f"layer {self.layer + 1}, head {self.head + 1}"


def habits(lab: Lab) -> list[Habit]:
    inputs, _ = lab.data.fixed(12, lab.model.config.context)
    weights = lab.model.forward(inputs).weights  # (layers, batch, heads, t, t)
    return [
        Habit(i, h, weights[i, :, h])
        for i in range(weights.shape[0])
        for h in range(weights.shape[2])
    ]


def habits_view(stats: list[Habit], width: int) -> Table:
    beside = max(len(s.name) for s in stats) + 13 + len("at the token before") + 3 * 2
    reach = max(6, min(12, width - beside - len(" 00.0 tokens")))
    grid = Table.grid(padding=(0, 2))
    for justify in ("left", "left", "left", "left"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(
        *(
            Text(h, style=viz.FAINT)
            for h in ("", "at itself", "at the token before", "looks back, on average")
        )
    )
    for s in stats:
        grid.add_row(
            Text(s.name, style="bold" if s.head == 0 else ""),
            Text.assemble(viz.bar(s.itself, 8, viz.ACCENT), (f" {s.itself:4.0%}", viz.FAINT)),
            Text.assemble(viz.bar(s.previous, 8, viz.GREEN), (f" {s.previous:4.0%}", viz.FAINT)),
            Text.assemble(
                viz.bar(min(s.reach / 30, 1.0), reach, viz.AMBER),
                (f" {s.reach:4.1f} tokens", viz.FAINT),
            ),
        )
    return grid


def habit_story(stats: list[Habit], layers: int) -> str:
    def mean(values: list[float]) -> float:
        return float(np.mean(values))

    first = [s for s in stats if s.layer == 0]
    later = [s for s in stats if s.layer > 0]
    prev = max(stats, key=lambda s: s.previous)
    far = max(stats, key=lambda s: s.reach)
    parts = []
    if mean([s.itself for s in first]) > mean([s.itself for s in later]) + 0.1:
        parts.append(
            "The first layer's heads mostly look at their own token: early on, there's little "
            "in the stream worth gathering yet."
        )
    parts.append(
        f"The strongest [b]previous-token head[/b] is {prev.name}, putting {prev.previous:.0%} "
        "of its attention on the token just before. Previous-token heads turn up in nearly "
        "every language model: knowing which token came before is useful everywhere."
    )
    parts.append(
        f"The one that reaches furthest is {far.name}, looking back {far.reach:.0f} tokens on "
        "average: gathering context from across the passage, like which animal or which "
        "character it's about."
    )
    return " ".join(parts)


def pair_view(
    weights: np.ndarray, names: list[str], a: Habit, b: Habit, width: int
) -> Table | Group:
    grids = [attention_grid(weights[s.layer, s.head], names, numbers=False) for s in (a, b)]
    cells = [
        Group(Text(s.name, style="bold"), *lines) for s, lines in zip((a, b), grids, strict=True)
    ]
    if sum(max(line.cell_len for line in lines) for lines in grids) + 4 > width:
        return Group(cells[0], Text(""), cells[1])
    grid = Table.grid(padding=(0, 4))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(*cells)
    return grid


# ── The logit lens ────────────────────────────────────────────────────────────


def settles(lens: np.ndarray, pos: int, right: int) -> int | None:
    """The first depth from which the right answer stays the top guess, if it ever does."""
    tops = lens[:, pos].argmax(-1)
    for depth in range(len(tops)):
        if (tops[depth:] == right).all():
            return depth
    return None


def pick_rows(lab: Lab, lens: np.ndarray, most: int = 12) -> list[int]:
    _, targets = lab.example()
    t = targets.shape[1]
    return list(range(min(t, most)))


def guess_room(lab: Lab, rows: list[int], depths: int, width: int) -> int:
    """How many characters of each top guess fit beside the probabilities: 0 if too few."""
    inputs, targets = lab.example()
    tiles = sum(
        max(len(label(lab.tokenizer, int(ids[0, pos]))) + 2 for pos in rows)
        for ids in (inputs, targets)
    )
    room = (width - tiles - (depths + 1)) // depths - len(" 00.0% ") - 1
    return min(6, room) if room >= 4 else 0


def lens_view(lab: Lab, lens: np.ndarray, rows: list[int], width: int) -> Table:
    tok = lab.tokenizer
    inputs, targets = lab.example()
    depths = lens.shape[0]
    room = guess_room(lab, rows, depths, width)
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    for _ in range(depths):
        grid.add_column(no_wrap=True, justify="left")
    heads = ["embeddings", *(f"layer {i}" for i in range(1, depths))]
    grid.add_row(
        Text("after", style=viz.FAINT),
        Text("answer", style=viz.FAINT),
        *(Text(f"{h:<10}", style=viz.FAINT) for h in heads),
    )
    for pos in rows:
        right = int(targets[0, pos])
        cells = []
        for d in range(depths):
            p = float(lens[d, pos, right])
            top = int(lens[d, pos].argmax())
            bg = viz.HEAT.color(p**0.6)
            cell = Text(f" {viz.percent(p):>5} ", style=viz.style(viz.ink_for(bg), bg))
            if top != right and room:
                cell.append(f" {label(tok, top)[:room]}", style=viz.FAINT)
            cells.append(cell)
        grid.add_row(chip(tok, int(inputs[0, pos])), chip(tok, right), *cells)
    return grid


def lens_story(lab: Lab, lens: np.ndarray) -> str:
    tok = lab.tokenizer
    inputs, targets = lab.example()
    last = lens.shape[0] - 1
    early, late = None, None
    for pos in range(targets.shape[1]):
        right = int(targets[0, pos])
        depth = settles(lens, pos, right)
        if depth == 1 and early is None:
            early = pos
        if depth == last and late is None and lens[last, pos, right] > 0.3:
            late = pos
    parts = [
        "Each layer adds its notes to the stream, and the guess sharpens as they pile up. (At "
        "the embeddings, before any layer, the guess is mostly the token itself: the same "
        "table is used at both ends, and a token's row points most toward its own token.)"
    ]
    if early is not None:
        a, b = label(tok, int(inputs[0, early])), label(tok, int(targets[0, early]))
        parts.append(
            f"Some answers are clear after the very first layer: after `{a}`, `{b}` is the "
            "top guess from layer 1 on, the sort of thing a table of token pairs could know."
        )
    if late is not None:
        a, b = label(tok, int(inputs[0, late])), label(tok, int(targets[0, late]))
        parts.append(
            f"Others only come together at the end: after `{a}`, `{b}` isn't the top guess "
            f"until the last layer has had its say."
        )
    parts.append(
        "Big models work the same way, with dozens of layers, and researchers use this lens "
        "(and sharper tools like it) to see where in the stack a model works something out."
    )
    return " ".join(parts)
