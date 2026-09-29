"""Chapter: inference. Writing, sampling and its dials, and hallucination."""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..generate import Pick, log_probs, nucleus, picks, prompt, top_k, write
from ..views import display, running, spectrum, step_view, wrap
from .lab import Lab
from .stage import Frame, Stage, hold

TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 10
SHOWN = 24  # tokens to write in the animation


def run(stage: Stage, lab: Lab) -> None:
    c = lab.corpus
    stage.say(
        "Writing text is a loop: run the model forward, get the probabilities for the next "
        "token, pick one at random according to those probabilities, add it to the end, and go "
        "again. It stops when it picks `<|endoftext|>`, or runs out of room."
    )
    stage.say(
        "Picking at random means lining the probabilities up end to end, from 0 to 1, and "
        "throwing a dart. Likely tokens are wide targets, unlikely ones narrow. Starting from "
        f"a blank page (right after `<|endoftext|>`), watch your model begin a {c.noun} "
        f"(temperature {TEMPERATURE}). You'll also see where the newest token looked, in each "
        "layer, on its way to the guess:"
    )
    steps: list[Pick] = []

    def writing_frames() -> Iterator[Frame]:
        steps[:] = picks(lab.model, prompt(lab.tokenizer), lab.rng, TEMPERATURE, limit=SHOWN)
        return writing(lab, steps, stage.width)

    stage.play(writing_frames, fps=4, start="throw the dart", again="write another")
    stage.say(
        "Notice that the weights never changed. Writing only runs the model forward; nothing "
        "is learned. A chatbot doesn't learn from talking to you either (unless its makers "
        "later train it on the conversation)."
    )
    stage.wait("turn the dials")

    stage.say(
        "Then there are the dials, the same ones you'll find in any chatbot's API. "
        "[b]Temperature[/b], from the softmax chapter, divides the scores before softmax: low "
        "temperatures play it safe and repeat themselves; high ones get creative, then "
        f"garbled. Here's how your model begins a {c.noun} at four temperatures:"
    )
    stage.show(temperatures(lab, stage.width))
    later = [p for p in steps if len(p.context) > 2] or steps
    wide = max(later, key=lambda p: float(-(p.probs * np.log(p.probs + 1e-12)).sum()))
    kept_p = int((nucleus(wide.probs, TOP_P) > 0).sum())
    stage.say(
        "Two more throw away the long shots before the dart flies. [b]Top-k[/b] keeps only the "
        f"k likeliest tokens (say {TOP_K}). [b]Top-p[/b] keeps the likeliest ones until they "
        f"add up to p (say {TOP_P:.0%}): few when the model is sure, many when it isn't. Either "
        "way, what's kept shares out the probability. Here's the step where your model was "
        f"least sure, when top-p kept {kept_p} of its {len(wide.probs)} tokens:"
    )
    stage.show(cutoffs(lab, wide, stage.width))
    greedy = write(lab.model, lab.tokenizer, lab.rng, "", temperature=0.01, limit=60)
    stage.say(
        "And at a temperature of zero, the dart always lands on the likeliest token. That's "
        "called [b]greedy[/b] decoding: the same prompt always gives the same text. It sounds "
        "safest, but small models (and big ones, less often) fall into loops, because the "
        "likeliest thing after a phrase is often the phrase again:"
    )
    stage.show(*running(lab.tokenizer, lab.tokenizer.encode(greedy), stage.width - 4)[:5])
    if stage.keys is not None and stage.auto is None:
        playground(stage, lab)
    stage.wait("see what it doesn't know")

    rows = hallucination(lab)
    stage.say(
        f"Everything your model writes is a [b]hallucination[/b]: fluent, plausible, and made "
        f"up. It even named itself [bold {viz.PURPLE}]{lab.name}[/], a name that appears "
        f"nowhere in the {c.plural}. Does it know? Here's how likely it finds each token of "
        "an opening it made up, of a real one it never saw in training, and of its own "
        "opening with the words shuffled:"
    )
    stage.show(confidence_view(lab, rows, stage.width))
    invented, real, scrambled = (typical(lp) for _, _, lp in rows)
    verdict = "more convincing than" if invented > real * 1.25 else "about as convincing as"
    stage.say(
        f"It finds its own invention {verdict} the real thing"
        + (", and far more than the shuffle" if scrambled < min(invented, real) else "")
        + f". It has learned what {c.plural} sound like, not what's true, so it can't tell "
        "them apart. A chatbot that invents a book, a quote or a court case is doing the same "
        "thing at scale: writing what's likely, which isn't always what's so."
    )


# ── Writing ───────────────────────────────────────────────────────────────────


def writing(lab: Lab, steps: list[Pick], width: int) -> Iterator[Frame]:
    for i, pick in enumerate(steps):
        yield hold(step_view(lab.tokenizer, pick, width, None), 0.9 if i < 3 else 0.45)
        yield hold(step_view(lab.tokenizer, pick, width, pick.roll), 0.9 if i < 3 else 0.45)


# ── The dials ─────────────────────────────────────────────────────────────────


def temperatures(lab: Lab, width: int) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    rng = lab.dice(13)
    for t in (0.4, 0.8, 1.2, 1.8):
        text = write(lab.model, lab.tokenizer, rng, "", temperature=t, limit=28)
        grid.add_row(
            Text(f"{t}", style=f"bold {viz.AMBER}"), Text(display(text, width - 10), style="italic")
        )
    return grid


def cutoffs(lab: Lab, pick: Pick, width: int) -> Table:
    strip = min(56, width - 16)
    tok = lab.tokenizer
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True, vertical="top")
    grid.add_column(no_wrap=True)
    kept_k = top_k(pick.probs, TOP_K)
    kept_p = nucleus(pick.probs, TOP_P)
    grid.add_row(
        Text("every token", style="bold"), spectrum(pick.probs, tok, strip, faded=kept_p == 0)
    )
    grid.add_row(
        Text.assemble(("top-k ", "bold"), (f"{TOP_K}", f"bold {viz.AMBER}")),
        spectrum(kept_k, tok, strip),
    )
    grid.add_row(
        Text.assemble(("top-p ", "bold"), (f"{TOP_P}", f"bold {viz.AMBER}")),
        spectrum(kept_p, tok, strip),
    )
    return grid


def playground(stage: Stage, lab: Lab) -> None:
    """Keys: space for another, ↑ ↓ for the temperature, p for top-p, k for top-k, → to move on."""
    t = TEMPERATURE
    top_ps, top_ks = (1.0, 0.9, 0.5), (0, 40, 10, 3)
    p = k = 0
    history: list[tuple[str, str]] = []
    help_line = Text.assemble(
        ("space", "bold"), (" another   ", viz.FAINT), ("↑ ↓", "bold"), (" temperature   ", viz.FAINT),
        ("p", "bold"), (" top-p   ", viz.FAINT), ("k", "bold"), (" top-k   ", viz.FAINT),
        ("→", "bold"), (" move on", viz.FAINT),
    )  # fmt: skip

    def settings() -> str:
        parts = [f"t {t:.1f}"]
        if top_ps[p] < 1:
            parts.append(f"p {top_ps[p]:g}")
        if top_ks[k]:
            parts.append(f"k {top_ks[k]}")
        return " · ".join(parts)

    def view() -> Group:
        rows = [
            Text.assemble(
                ("temperature ", viz.FAINT), (f"{t:.1f}", f"bold {viz.AMBER}"),
                ("   top-p ", viz.FAINT), ("off" if top_ps[p] >= 1 else f"{top_ps[p]:g}", f"bold {viz.AMBER}"),
                ("   top-k ", viz.FAINT), ("off" if not top_ks[k] else f"{top_ks[k]}", f"bold {viz.AMBER}"),
            )
        ]  # fmt: skip
        for shown, text in history[-6:]:
            rows.append(
                Text.assemble(
                    (f"{shown:<18}", viz.FAINT), (display(text, stage.width - 22), "italic")
                )
            )
        return Group(Text("Your turn.", style="bold"), help_line, Text(""), *rows)

    with stage.live() as live:
        live.update(stage.pad(view(), live=True), refresh=True)
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
            elif key == "k":
                k = (k + 1) % len(top_ks)
            elif key == "space":
                text = write(
                    lab.model, lab.tokenizer, lab.rng, "", t, top_ps[p], top_ks[k], limit=32
                )
                history.append((settings(), text))
            live.update(stage.pad(view(), live=True), refresh=True)


# ── Hallucination ─────────────────────────────────────────────────────────────


def first_sentence(text: str, most: int = 90) -> str:
    body = text.split("\n\n", 1)[-1] if "\n\n" in text[:80] else text
    body = body.replace("\n", " ")
    end = re.search(r"[.!?]", body[20:])
    cut = body[: 20 + end.end()] if end else body
    return cut[:most].rsplit(" ", 1)[0] if len(cut) > most else cut


def hallucination(lab: Lab) -> list[tuple[str, str, np.ndarray]]:
    """An opening the model made up, a real one it never saw, and its own shuffled, with log-probs."""
    rng = np.random.default_rng([lab.seed, 10])
    invented = first_sentence(write(lab.model, lab.tokenizer, rng, "", temperature=0.7, limit=60))
    real = first_sentence(sorted(lab.data.val_docs, key=len)[len(lab.data.val_docs) // 2])
    words = invented.split()
    shuffled = " ".join(rng.permutation(words)) if len(words) > 2 else invented[::-1]
    return [
        ("made up", invented, log_probs(lab.model, lab.tokenizer, invented)),
        ("real", real, log_probs(lab.model, lab.tokenizer, real)),
        ("shuffled", shuffled, log_probs(lab.model, lab.tokenizer, shuffled)),
    ]


def typical(lp: np.ndarray) -> float:
    """The typical chance it gave each token: the geometric mean of the probabilities."""
    return float(np.exp(lp.mean()))


def confidence_view(lab: Lab, rows: list[tuple[str, str, np.ndarray]], width: int) -> Table:
    tok = lab.tokenizer
    room = width - max(len(name) for name, *_ in rows) - len("typical") - 2 * 2
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(
        "",
        Text("each token, lit by how likely it found it", style=viz.FAINT),
        Text("typical", style=viz.FAINT),
    )
    for name, text, lp in rows:
        lines = [Text(no_wrap=True)]  # broken between words, so no token is split
        for token, p in zip(tok.encode(text), np.exp(lp), strict=True):
            bg = viz.HEAT.color(float(p) ** 0.5)
            wrap(lines, tok.pieces[token], viz.style(viz.ink_for(bg), bg), room)
        grid.add_row(name, Group(*lines), Text(viz.percent(typical(lp)), style="bold"))
    return grid
