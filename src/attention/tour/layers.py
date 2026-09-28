"""Chapter: why stack layers? Induction heads, and why a key is not a value."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..induction import Batch, Habits, Race, habits, sequences
from ..tokenizer import END
from ..views import heat_lines
from .lab import Lab
from .stage import Frame, Stage

WORDS = 40  # the vocabulary of the experiment: this many common words, plus a start marker
FIRST = [7, 2]  # the first race's random numbers: they grow the textbook circuit
FPS = 12


def words(lab: Lab) -> list[int]:
    """Common whole words from your tokenizer, to make the experiment's text from."""
    tok = lab.tokenizer
    freq = np.bincount(lab.data.train, minlength=len(tok))
    whole = [
        int(i) for i in np.argsort(-freq)
        if tok.pieces[i].startswith(" ") and tok.pieces[i][1:].isalpha() and len(tok.pieces[i]) > 2
    ]  # fmt: skip
    return whole[:WORDS]


def real(ids: np.ndarray, vocab: list[int]) -> list[int]:
    """The experiment's token numbers (0 to WORDS) as your tokenizer's tokens."""
    return [END if i == 0 else vocab[int(i) - 1] for i in ids]


def run(stage: Stage, lab: Lab) -> None:
    vocab = words(lab)
    stage.say(
        "Back in the chapter on the stack, there was a claim: stacking layers lets later ones "
        "build on what earlier ones wrote. Here's an experiment that shows exactly that, and "
        "shows, along the way, why a key and a value have to be different things."
    )
    example = sequences(np.random.default_rng(1), 1, WORDS + 1)
    ids = real(example.inputs[0], vocab)
    stage.say(
        f"The task: strings of random words, drawn from {WORDS} common ones in your "
        f"{lab.corpus.plural}. Somewhere in each string, a stretch of words appears twice. "
        "The model is graded only inside the second copy, where it's possible to know what "
        "comes next: find where the current word turned up before, and say the word that "
        "came after it. Here's one, with the repeat marked:"
    )
    stage.show(*repeat_view(lab, example, ids))
    stage.say(
        "You do this all the time: read “Mrs. Dursley” once, and the next “Mrs.” makes you "
        "expect “Dursley”. It's called [b]in-context copying[/b], and it's one of the most "
        "useful things a language model does: repeating names, following the pattern of an "
        "example, picking up an unusual word from earlier in the prompt."
    )
    stage.wait("race them")

    stage.say(
        "Two models, built exactly like yours but much smaller, learn this task side by side, "
        "from the same batches: one has a single layer, the other has two. The chart shows "
        "each one's loss inside the repeats. Guessing blindly among the words scores "
        f"{np.log(WORDS):.2f}."
    )
    race = Race(WORDS + 1, np.random.default_rng(FIRST), steps=lab.budget.race)

    def racing() -> Iterator[Frame]:
        nonlocal race
        if race.step_number:  # a replay: new random numbers
            race = Race(WORDS + 1, lab.dice(12), steps=lab.budget.race)
        return animate(race, stage.width)

    stage.play(racing, fps=FPS, start="start the race", again="race again")
    one, two = race.learners
    drop = next((s for s, loss in two.history if loss < 1.0), None)
    stage.say(
        f"The one-layer model crept down to {one.loss:.2f} and stalled. The two-layer model "
        + (
            f"sat just as stuck, then around step {drop} dropped suddenly, to {two.loss:.2f}. "
            if drop
            else f"reached {two.loss:.2f}. "
        )
        + "That sudden drop has a name, a phase change: it happens when a small circuit "
        "clicks into place. It shows up in the training of large language models too."
    )
    stage.wait("see why one layer can't")

    stage.say(
        "Why can't one layer do it? At the second copy's `A`, the answer is whatever came after "
        "the first `A`. To find it, the query at `A` must match the key at the position "
        "[i]after[/i] the first `A`. But a key is made from its own position's vector, and in "
        "the first layer that vector only knows its own word (and, through RoPE, its place). "
        "Nothing there says “the word before me was `A`”. With one layer, the best it can do is "
        "notice which words have turned up already, and guess among those."
    )
    batch = sequences(np.random.default_rng(5), 64, WORDS + 1)
    found = habits(two.model, batch)
    stage.say(circuit_story(found))
    shown = sequences(np.random.default_rng(2), 1, WORDS + 1)
    stage.show(circuit_view(lab, two.model, shown, vocab, found, stage.width))
    stage.say(
        "So in layer 2, one position's key and value carry different things. Its [b]key[/b] "
        "describes the word [i]before[/i] it, so it can be found by what came before. Its "
        "[b]value[/b] describes the word [i]itself[/i], the answer to hand over. If keys and "
        "values were one and the same, this couldn't work. And it needs two layers, because "
        "layer 2's keys can only read what layer 1 wrote: researchers call it [b]composition[/b]."
    )
    stage.say(
        "This pair of heads is called an [b]induction head[/b]. Every large language model "
        "that's been looked at has them, and they're thought to be a big part of how models "
        "learn from their prompts: the examples you give a chatbot work partly through "
        "circuits like this one, built from queries, keys and values in two layers."
    )
    stage.wait("test your model")

    first, second = repeat_test(lab)
    stage.say(
        "Does your pretrained model have one? Here's its loss on a passage of held-back text "
        "read once, then read again straight after, as if repeated:"
    )
    stage.show(repeat_losses(first, second))
    if second < first * 0.7:
        stage.say(
            "The second time through, it's far less surprised: it's copying from the first "
            "copy. Your model grew induction heads of its own."
        )
    else:
        stage.say(
            "No better the second time: it isn't copying. Models this small, trained this "
            "briefly on this little text, usually don't grow induction heads; the two-layer "
            "model above had a task that demanded one, and nothing else. In large models "
            "trained on vast amounts of text, they always appear."
        )


# ── The task ──────────────────────────────────────────────────────────────────


def repeat_view(lab: Lab, batch: Batch, ids: list[int]) -> list[Text]:
    graded = batch.targets[0] >= 0
    first = np.zeros(len(ids), dtype=bool)
    gap = int(batch.gaps[0])
    for i in np.flatnonzero(graded):
        first[i - gap] = first[i - gap + 1] = True
    second = graded.copy()
    second[np.flatnonzero(graded)[-1] + 1 :] = False
    second[np.flatnonzero(graded) + 1] = True
    line = Text(no_wrap=False)
    for i, token in enumerate(ids):
        if token == END:
            continue
        style = (
            viz.style("#f4f4ff", "#1d5a47") if second[i] else
            viz.style("#f4f4ff", "#3d4270") if first[i] else viz.FAINT
        )  # fmt: skip
        line.append(lab.tokenizer.pieces[token], style)
    key = Text.assemble(
        ("■ ", "#3d4270"), ("the first copy   ", viz.FAINT), ("■ ", "#1d5a47"),
        ("the repeat, where it's graded", viz.FAINT),
    )  # fmt: skip
    return [line, Text(""), key]


# ── The race ──────────────────────────────────────────────────────────────────


def animate(race: Race, width: int) -> Iterator[Frame]:
    yield view(race, width)
    while not race.done:
        target = min(race.steps, race.step_number + 25)
        while race.step_number < target:
            race.step()
        yield view(race, width)


def view(race: Race, width: int) -> Group:
    plot_w = max(30, min(60, width - 16))
    hi = float(np.ceil(np.log(WORDS) + 0.5))
    plot = viz.Plot(plot_w, 10, x_max=race.steps, lo=0.0, hi=hi)
    plot.guide(float(np.log(WORDS)), viz.FAINT, "guessing")
    one, two = race.learners
    plot.line(one.history, viz.AMBER)
    plot.line(two.history, viz.GREEN)
    progress = Text.assemble(
        ("step ", viz.FAINT), (f"{race.step_number:>4}", "bold"), (f" / {race.steps}  ", viz.FAINT),
        viz.bar(race.step_number / race.steps, 20, viz.ACCENT),
    )  # fmt: skip
    legend = Text.assemble(
        ("━ ", viz.AMBER), ("one layer ", viz.FAINT), (f"{one.loss:.2f}", "bold"),
        ("   ━ ", viz.GREEN), ("two layers ", viz.FAINT), (f"{two.loss:.2f}", "bold"),
    )  # fmt: skip
    return Group(
        progress, Text(""), Text("loss inside the repeats", style="bold"), *plot.render(), legend
    )


# ── The circuit ───────────────────────────────────────────────────────────────


def circuit_story(found: Habits) -> str:
    prev_head, prev = found.previous()
    ind_head, ind = found.induction()
    if found.textbook:
        return (
            "Two layers can do it in two steps, and the trained model shows exactly how. In "
            f"layer 1, head {prev_head + 1} became a [b]previous-token head[/b]: at every "
            f"position it looks at the word just before ({prev:.0%} of its attention, on "
            "average), and its value writes that word into the position's vector. After layer "
            "1, each position's vector holds its own word [i]and[/i] “the word before me was "
            f"…”. Then in layer 2, head {ind_head + 1} does the lookup. At the second `A`, its "
            "query asks “who came right after an `A`?”; the keys, made from layer 1's output, "
            "can now answer “the word before me was `A`”; and it looks straight at the answer "
            f"({ind:.0%} of its attention). Here are the two heads, reading one string:"
        )
    back = found.back[:, 0]  # (3, heads): layer 1 at 1, 2 and 3 back
    past = found.past[:, -1]  # (3, heads): the last layer at the answer and just past it
    k, h = np.unravel_index(int(np.argmax(back)), back.shape)
    d, g = np.unravel_index(int(np.argmax(past)), past.shape)
    return (
        "Two layers can do it in two steps, though this model found its own variation. In "
        f"layer 1, head {h + 1} looks {k + 1} word{'s' if k else ''} back ({back[k, h]:.0%} "
        "of its attention), writing that word into each position's vector. Then in layer 2, "
        f"head {g + 1} looks {'at the answer' if d == 0 else f'{d} past the answer'} "
        f"({past[d, g]:.0%} of its attention): the position whose layer-1 note matches the "
        "current word, and whose vector carries the word to copy. The textbook version, with "
        "a head that looks one back, is what most runs find. Here are the two heads, reading "
        "one string:"
    )


def circuit_view(
    lab: Lab, model, batch: Batch, vocab: list[int], found: Habits, width: int
) -> Group:
    """The two heads at work on one string: each lit up by where it looks."""
    tok = lab.tokenizer
    weights = model.forward(batch.inputs).weights[:, 0]  # (layers, heads, t, t)
    ids = real(batch.inputs[0], vocab)
    graded = np.flatnonzero(batch.targets[0] >= 0)
    at = int(graded[len(graded) // 2])  # a word in the middle of the repeat
    answer = at - int(batch.gaps[0]) + 1  # the word after the same word, in the first copy
    window = list(range(max(1, int(graded[0]) - int(batch.gaps[0]) - 1), at + 1))
    words = [ids[i] for i in window]
    first, second = int(np.argmax(found.back[0, 0])), found.induction()[0]

    def strip(layer: int, head: int, where: int) -> Group:
        w = weights[layer, head, where, window]
        return Group(*heat_lines(tok, words, w / max(float(w.max()), 1e-9), width))

    a, b = tok.pieces[ids[at]].strip(), tok.pieces[ids[answer]].strip()
    lines = [
        Text(f"layer 1, head {first + 1}, at the first “{b}”: it looks one word back", style=viz.FAINT),
        strip(0, first, answer),
        Text(""),
        Text(f"layer 2, head {second + 1}, at the second “{a}”: it looks at the answer", style=viz.FAINT),
        strip(1, second, at),
        Text(""),
        Text.assemble(
            ("So after the second “", viz.FAINT), (a, "bold"), ("”, it predicts “", viz.FAINT),
            (b, f"bold {viz.GREEN}"), ("”: the word that came after the first one.", viz.FAINT),
        ),
    ]  # fmt: skip
    return Group(*lines)


# ── Your model ────────────────────────────────────────────────────────────────


def repeat_test(lab: Lab, length: int = 40) -> tuple[float, float]:
    """Loss on a held-back passage, read once and then again."""
    doc = max(lab.data.val_docs, key=len)
    part = lab.tokenizer.encode(doc)[:length]
    ids = [END, *part, *part][: lab.model.config.context + 1]
    probs = lab.model.forward(np.array([ids[:-1]])).probs[0]
    lp = -np.log(probs[np.arange(len(ids) - 1), ids[1:]])
    return float(lp[:length].mean()), float(lp[length + 1 :].mean())


def repeat_losses(first: float, second: float) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    top = max(first, second, 1e-9)
    grid.add_row(
        "read the first time",
        Text(f"{first:.2f}", style="bold"),
        viz.bar(first / top, 24, viz.AMBER),
    )
    grid.add_row(
        "read again", Text(f"{second:.2f}", style="bold"), viz.bar(second / top, 24, viz.GREEN)
    )
    return grid
