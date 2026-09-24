"""Chapter: inference. Writing, sampling, the context window, and hallucination."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..generate import Pick, invent, log_prob, nucleus, picks
from ..views import display, spectrum, step_view, token_chip
from .lab import Lab
from .stage import Frame, Stage, hold

TEMPERATURE = 0.8
TOP_P = 0.9


def run(stage: Stage, lab: Lab) -> None:
    stage.say(
        "Generating text is a loop: run the model forward, get the probabilities for the next "
        "token, pick one at random according to those probabilities, add it to the end, and go "
        "again, until it picks the `.` that ends a word."
    )
    stage.say(
        "Picking at random means lining the probabilities up end to end, from 0 to 1, and "
        "throwing a dart. Likely letters are wide targets, unlikely ones narrow. Watch your "
        f"model invent a {lab.corpus.noun} (temperature {TEMPERATURE}):"
    )
    stage.wait("generate")
    steps = list(picks(lab.model, lab.vocab, lab.rng, TEMPERATURE))
    stage.play(writing(lab, steps, stage.width), fps=4)
    stage.wait()

    n = len(steps)
    stage.say(
        f"Two things about that loop. First, [b]every letter reran the model on everything so "
        f"far[/b]: {n} steps meant {n * (n + 1) // 2} positions' worth of work. But thanks to the "
        "causal mask, adding a letter never changes anything about the letters before it. So "
        "real models keep each position's keys and values from the step before, in a "
        "[b]KV cache[/b], and only work out the newest one:"
    )
    stage.show(caching(lab, steps))
    c = lab.model.config
    stage.say(
        f"Second, [b]it can only read so far[/b]. It has {c.context} position embeddings, one "
        f"for each place a letter can be, so {c.context} characters is all it can see. That's its "
        "[b]context window[/b]. A chatbot's works the same way, just often 100,000 tokens or "
        "more long. When a conversation outgrows it, the beginning falls out of view."
    )
    stage.wait("turn the dials")

    stage.say(
        "Then there are the dials. [b]Temperature[/b], from the softmax chapter, makes every "
        "choice bolder or safer. Low temperatures play it safe and repeat what they've seen; "
        "high ones get creative, then weird. ✦ marks words that aren't in the training data:"
    )
    stage.show(temperatures(lab))
    later = [p for p in steps if len(p.context) > 1] or steps
    wide = max(later, key=lambda p: float(-(p.probs * np.log(p.probs + 1e-12)).sum()))
    kept = int((nucleus(wide.probs, TOP_P) > 0).sum())
    stage.say(
        f"[b]Top-p[/b] throws away the long shots before the dart flies: keep the likeliest "
        f"tokens until they add up to p (say {TOP_P:.0%}), and share their probability out among "
        f"them. Here's the step after `{wide.context}`, where your model was least sure. Top-p "
        f"keeps {kept} of its {len(wide.probs)} tokens and drops the faded ones:"
    )
    stage.show(top_p_view(lab, wide, stage.width))
    stage.say(
        "When the model is sure, top-p changes almost nothing. When it isn't, top-p stops it "
        "from ever picking something truly unlikely. Most chatbots use both dials together."
    )
    if stage.keys is not None and stage.auto is None:
        playground(stage, lab)
    stage.wait("see what it doesn't know")

    stage.say(
        "Every ✦ word you've seen is a [b]hallucination[/b]: fluent, plausible, and made up. "
        f"Your model even named itself [bold {viz.PURPLE}]{lab.name}[/], and there's no such "
        f"{lab.corpus.noun}. Does it know that? Here's how likely it finds each letter of its "
        f"own name, of a real {lab.corpus.noun} it never saw in training, and of its name "
        "with the letters scrambled:"
    )
    rows = hallucination(lab)
    stage.show(confidence_view(rows))
    invented, real, scrambled = (typical(lp) for _, _, lp in rows)
    everyone = float(np.mean([typical(log_prob(lab.model, lab.vocab, w)) for w in lab.data.val]))
    verdict = "more convincing than" if invented > real * 1.25 else "about as convincing as"
    stage.say(
        f"It finds its invention {verdict} a real one"
        + (", and far more than the scramble" if scrambled < min(invented, real) else "")
        + f". (Across all {len(lab.data.val)} held-back {lab.corpus.plural}, the typical letter "
        f"gets {viz.percent(everyone)}.) It has learned what {lab.corpus.plural} look like, not "
        "which ones exist, so it can't tell them apart. A chatbot that invents a book, a quote "
        "or a court case is doing the same thing at scale: writing what's likely, which isn't "
        "always what's true."
    )


# ── Writing a word ────────────────────────────────────────────────────────────


def writing(lab: Lab, steps: list[Pick], width: int) -> Iterator[Frame]:
    for i, pick in enumerate(steps):
        yield hold(step_view(lab.vocab, pick, width, None), 0.9 if i < 3 else 0.5)
        yield hold(step_view(lab.vocab, pick, width, pick.roll), 0.9 if i < 3 else 0.5)
    word = "".join(lab.vocab.chars[p.token] for p in steps if not p.done)
    new = word not in set(lab.corpus.words)
    done = Text.assemble(
        ("→ ", viz.FAINT), (display(word), f"bold {viz.GREEN if new else ''}"),
        ("   ✦ new: it's not in the training data" if new else "   (this one is in the training data)", viz.FAINT),
    )  # fmt: skip
    yield hold(Group(step_view(lab.vocab, steps[-1], width, steps[-1].roll), Text(""), done), 0.5)


def caching(lab: Lab, steps: list[Pick], most: int = 12) -> Table:
    """The work each step does, without a cache and with one."""
    vocab = lab.vocab
    grid = Table.grid(padding=(0, 4))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(Text("without a cache", style="bold"), Text("with a KV cache", style="bold"))
    for pick in steps[:most]:
        ids = vocab.encode(pick.context)
        redo = Text(no_wrap=True)
        for token in ids:
            redo.append_text(token_chip(vocab, token))
        cached = Text(no_wrap=True)
        cached.append(" · " * (len(ids) - 1), style=viz.FAINT)
        cached.append_text(token_chip(vocab, ids[-1]))
        grid.add_row(redo, cached)
    if len(steps) > most:
        grid.add_row(Text("…", style=viz.FAINT), Text("…", style=viz.FAINT))
    n = len(steps)
    grid.add_row(
        Text.assemble((f"{n * (n + 1) // 2}", "bold"), (" positions computed", viz.FAINT)),
        Text.assemble((f"{n}", f"bold {viz.GREEN}"), (" positions computed", viz.FAINT)),
    )
    return grid


# ── The dials ─────────────────────────────────────────────────────────────────


def temperatures(lab: Lab) -> Table:
    known = set(lab.corpus.words)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column()
    for t in (0.4, 0.8, 1.2, 1.8):
        words = invent(lab.model, lab.vocab, lab.rng, 5, temperature=t)
        grid.add_row(Text(f"{t}", style=f"bold {viz.AMBER}"), word_list(words, known))
    return grid


def word_list(words: list[str], known: set[str]) -> Text:
    text = Text()
    for i, w in enumerate(words):
        if i:
            text.append("  ")
        new = w not in known
        text.append("✦" if new else "·", style=viz.GREEN if new else viz.FAINT)
        text.append(display(w), style="" if new else viz.FAINT)
    return text


def top_p_view(lab: Lab, pick: Pick, width: int) -> Table:
    strip = min(56, width - 16)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True, vertical="top")
    grid.add_column(no_wrap=True)
    dropped = nucleus(pick.probs, TOP_P) == 0
    grid.add_row(
        Text("every token", style="bold"), spectrum(pick.probs, lab.vocab, strip, faded=dropped)
    )
    grid.add_row(
        Text.assemble(("top-p ", "bold"), (f"{TOP_P}", f"bold {viz.AMBER}")),
        spectrum(nucleus(pick.probs, TOP_P), lab.vocab, strip),
    )
    return grid


def playground(stage: Stage, lab: Lab) -> None:
    """Keys: space for another word, ↑ ↓ for the temperature, p for top-p, → to move on."""
    t = TEMPERATURE
    top_ps = (1.0, 0.9, 0.5)
    p = 0
    known = set(lab.corpus.words)
    history: list[tuple[float, float, str]] = []
    help_line = Text.assemble(
        ("space", "bold"), (" another   ", viz.FAINT), ("↑ ↓", "bold"), (" temperature   ", viz.FAINT),
        ("p", "bold"), (" top-p   ", viz.FAINT), ("→", "bold"), (" move on", viz.FAINT),
    )  # fmt: skip

    def dials(temp: float, top: float) -> str:
        return f"{temp:.1f}" + (f" · p {top:g}" if top < 1 else "")

    def view() -> Group:
        rows = [
            Text.assemble(
                ("temperature ", viz.FAINT), (f"{t:.1f}", f"bold {viz.AMBER}"),
                ("   top-p ", viz.FAINT), ("off" if top_ps[p] >= 1 else f"{top_ps[p]:g}", f"bold {viz.AMBER}"),
            )
        ]  # fmt: skip
        for temp, top, w in history[-8:]:
            new = w not in known
            rows.append(Text.assemble(
                (f"{dials(temp, top):<12}", viz.FAINT), ("✦ " if new else "· ", viz.GREEN if new else viz.FAINT),
                (display(w), "bold" if new else viz.FAINT),
            ))  # fmt: skip
        return Group(Text("Your turn.", style="bold"), help_line, Text(""), *rows)

    with stage.live() as live:
        live.update(stage.pad(view()), refresh=True)
        while True:
            key = stage.key()
            if key in ("right", "enter", "tab"):
                break
            if key == "up":
                t = min(3.0, round(t + 0.1, 1))
            elif key == "down":
                t = max(0.1, round(t - 0.1, 1))
            elif key == "p":
                p = (p + 1) % len(top_ps)
            elif key == "space":
                word = invent(lab.model, lab.vocab, lab.rng, 1, temperature=t, top_p=top_ps[p])[0]
                history.append((t, top_ps[p], word))
            live.update(stage.pad(view()), refresh=True)


# ── Hallucination ─────────────────────────────────────────────────────────────


def hallucination(lab: Lab) -> list[tuple[str, str, np.ndarray]]:
    """The model's own name, a real word it never trained on, and a scramble, with log-probs."""
    invented = lab.name.lower()
    if invented in lab.corpus.words:  # it named itself after a real one: find an invention
        known = set(lab.corpus.words)
        found = [w for w in invent(lab.model, lab.vocab, lab.rng, 30, 0.6) if w not in known]
        invented = found[0] if found else invented
    # A representative real one: the held-back word the model rates most like a typical one.
    scores = {w: typical(log_prob(lab.model, lab.vocab, w)) for w in lab.data.val}
    middle = float(np.median(list(scores.values())))
    real = min(scores, key=lambda w: (abs(scores[w] - middle), w))
    rng = np.random.default_rng([lab.seed, 10])
    scrambled = invented
    for _ in range(20):
        scrambled = "".join(rng.permutation(list(invented)))
        if scrambled != invented:
            break
    return [
        (f"its name, {display(invented)}", invented, log_prob(lab.model, lab.vocab, invented)),
        (f"a real one, {display(real)}", real, log_prob(lab.model, lab.vocab, real)),
        ("its name, scrambled", scrambled, log_prob(lab.model, lab.vocab, scrambled)),
    ]


def typical(log_probs: np.ndarray) -> float:
    """The typical chance it gave each letter: the geometric mean of the probabilities."""
    return float(np.exp(log_probs.mean()))


def confidence_view(rows: list[tuple[str, str, np.ndarray]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(
        "",
        Text("how likely it found each letter, and the ending", style=viz.FAINT),
        Text("typical", style=viz.FAINT),
    )
    for label, word, lp in rows:
        letters = Text(no_wrap=True)
        for ch, p in zip(word + ".", np.exp(lp), strict=True):
            bg = viz.HEAT.color(float(p) ** 0.5)
            letters.append(f" {ch} ", viz.style(viz.ink_for(bg), bg, bold=True))
        grid.add_row(label, letters, Text(viz.percent(typical(lp)), style="bold"))
    return grid
