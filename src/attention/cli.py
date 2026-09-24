"""`attention`: take the tour, or use the model you made on it."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.text import Text

from . import __version__, dashboard, tour, viz
from . import corpus as corpora
from .explain import influence, nudge
from .generate import Pick, invent
from .keys import Keys
from .model import softmax
from .store import Saved, default_path, load
from .tour.lab import Lab
from .tour.pretraining import summary, tidy
from .tour.stage import THEME, Stage
from .train import STEPS
from .views import (
    display,
    influence_view,
    model_card,
    nudge_view,
    reading,
    tally,
    word_row,
)

out = Console(highlight=False, theme=THEME)
err = Console(stderr=True, highlight=False, theme=THEME)

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

Model = Annotated[
    Path | None,
    typer.Option(
        "--model", "-m", help="Model file (default: your saved model).", show_default=False
    ),
]
CorpusOption = Annotated[
    str | None,
    typer.Option(
        "--corpus",
        "-c",
        help="dinosaurs, names, towns, or a file with one word per line. Default: you choose.",
        show_default=False,
    ),
]
Seed = Annotated[
    int | None,
    typer.Option(
        "--seed", "-s", help="Seed for a reproducible model (default: random).", show_default=False
    ),
]


def fail(message: str, hint: str | None = None) -> typer.Exit:
    err.print(Text.assemble(("✗ ", f"bold {viz.RED}"), (message, "bold")))
    if hint:
        err.print(Text.from_markup(f"  {hint}"))
    return typer.Exit(1)


def open_model(path: Path | None) -> tuple[Saved, Path]:
    path = path or default_path()
    if not path.exists():
        raise fail(
            f"There's no model at {tidy(path)} yet.",
            "Take the tour ([bold]attention[/]) or run [bold]attention train[/] to make one.",
        )
    try:
        return load(path), path
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise fail(
            f"Couldn't read the model at {tidy(path)}: {e}",
            "Train a new one: [bold]attention train[/]",
        ) from e


def check_prefix(saved: Saved, prefix: str) -> str:
    prefix = prefix.strip().lower()
    unknown = sorted({ch for ch in prefix if ch not in saved.vocab.chars[1:]})
    if unknown:
        raise fail(
            f"{saved.card.name} doesn't know {', '.join(repr(ch) for ch in unknown)}.",
            f"It knows these letters: [bold]{saved.vocab.chars[1:]}[/]",
        )
    if len(prefix) + 1 >= saved.model.config.context:
        most = saved.model.config.context - 1
        raise fail(f"That's too long: {saved.card.name} reads at most {most} letters.")
    return prefix


def pick_corpus(name: str | None, rng: np.random.Generator) -> corpora.Corpus:
    if name is None:
        return corpora.built_in(list(corpora.BUILT_IN)[int(rng.integers(len(corpora.BUILT_IN)))])
    try:
        return corpora.load(name)
    except (ValueError, OSError) as e:
        raise fail(str(e)) from e


def name_line(saved: Saved, extra: str = "") -> Text:
    return Text.assemble(
        ("◆ ", viz.PURPLE),
        (saved.card.name, f"bold {viz.PURPLE}"),
        (f" · a {saved.card.noun} model{extra}", viz.FAINT),
    )


def _version(value: bool) -> None:
    if value:
        out.print(f"attention {__version__}")
        raise typer.Exit()


# ── Commands ──────────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", callback=_version, is_eager=True, help="Show the version."),
    ] = None,
) -> None:
    """[bold]attention[/]: build a tiny transformer, watch it learn, and keep it.

    With no command, starts the guided tour.
    """
    if ctx.invoked_subcommand is None:
        take_tour()


@app.command("tour")
def take_tour(
    corpus: CorpusOption = None,
    seed: Seed = None,
    chapter: Annotated[
        int,
        typer.Option(
            "--chapter",
            help=f"Start at this chapter (1–{len(tour.CHAPTERS)}).",
            min=1,
            max=len(tour.CHAPTERS),
        ),
    ] = 1,
    auto: Annotated[
        bool, typer.Option("--auto", help="Play by itself, without waiting for keys.")
    ] = False,
    fast: Annotated[bool, typer.Option("--fast", help="Speed up the animations.")] = False,
    model: Model = None,
) -> None:
    """Take the guided tour: tokens, attention, backprop, training, and a model to keep."""
    if out.is_terminal and out.width < 80:
        err.print(f"[faint]attention looks best at least 80 columns wide (this is {out.width}).[/]")
    interactive = Keys.available() and not auto
    # With a screen but no keyboard to press, play by itself rather than racing past the text.
    autopilot = auto or (out.is_terminal and not interactive)
    stage = Stage(
        out, animate=out.is_terminal, auto=3.5 if autopilot else None, speed=2.5 if fast else 1.0
    )
    try:
        if interactive:
            with Keys() as keys:
                stage.keys = keys
                outcome = tour.run(stage, corpus=corpus, seed=seed, start=chapter, model_path=model)
        else:
            outcome = tour.run(stage, corpus=corpus, seed=seed, start=chapter, model_path=model)
    except KeyboardInterrupt:
        out.print("\n[faint]Stopped.[/]")
        raise typer.Exit(130) from None
    except ValueError as e:
        raise fail(str(e)) from e
    if not outcome.finished:
        out.print()
        lab = outcome.lab
        resume = f"attention tour --chapter {outcome.reached}"
        if lab:
            resume += f" --corpus {corpus or lab.corpus.key} --seed {lab.seed}"
        out.print(
            Text.assemble(
                ("  See you later. To pick up where you left off: ", viz.FAINT), (resume, "bold")
            )
        )
        out.print()


@app.command()
def train(
    corpus: CorpusOption = None,
    seed: Seed = None,
    steps: Annotated[int, typer.Option("--steps", help="How many training steps.", min=1)] = STEPS,
    fast: Annotated[
        bool, typer.Option("--fast", help="Don't slow down to watch; just train.")
    ] = False,
    model: Model = None,
) -> None:
    """Train a new model with the live dashboard, and save it (replacing the old one)."""
    rng = np.random.default_rng()
    chosen = pick_corpus(corpus, rng)
    seed = int(rng.integers(1, 10_000)) if seed is None else seed
    path = model or default_path()
    previous = None
    if path.exists():
        try:
            previous = load(path).card.name
        except (OSError, ValueError, KeyError, TypeError):
            previous = None
    lab = Lab.create(chosen, seed, path, steps)
    watch = dashboard.Watch(lab.trainer, lab.baselines, chosen, seed, lab.rng)
    width = min(out.width, 100) - 4
    out.print()
    try:
        if out.is_terminal:
            seconds = 0.0 if fast else 10.0
            with Live(console=out, auto_refresh=False) as live, keyboard() as keys:

                def update(frame) -> None:
                    live.update(Padding(frame, (0, 0, 0, 2)), refresh=True)

                def interrupted(timeout: float) -> bool:
                    if keys is None:
                        time.sleep(timeout)
                        return False
                    return keys.read(timeout) is not None

                dashboard.run(watch, update, width, seconds, interrupted)
        else:
            dashboard.run(watch, lambda _: None, width, 0, lambda _: False)
    except KeyboardInterrupt:
        out.print("\n[faint]Stopped. Nothing was saved.[/]")
        raise typer.Exit(130) from None
    lab.seconds = watch.compute
    saved_to = lab.finish()
    out.print()
    out.print(Padding(summary(lab, saved_to), (0, 0, 0, 2)))
    if previous:
        out.print(Text(f"    (replacing {previous})", style=viz.FAINT))
    out.print()
    out.print(
        Text.assemble(
            ("  Try: ", viz.FAINT),
            ("attention generate", "bold"),
            ("  or  ", viz.FAINT),
            (f"attention explain {chosen.probe}", "bold"),
        )
    )
    out.print()


@contextmanager
def keyboard() -> Iterator[Keys | None]:
    """Single keypresses if there's a keyboard to press, otherwise None."""
    if not Keys.available():
        yield None
        return
    with Keys() as keys:
        yield keys


@app.command()
def generate(
    start: Annotated[
        str, typer.Argument(help="Letters every word should start with.", show_default=False)
    ] = "",
    count: Annotated[
        int, typer.Option("--count", "-n", help="How many words.", min=1, max=200)
    ] = 10,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", help="Lower is safer, higher is wilder.", min=0.05, max=5.0
        ),
    ] = 0.8,
    top_p: Annotated[
        float,
        typer.Option(
            "--top-p",
            "-p",
            help="Only pick from the likeliest letters adding up to this.",
            min=0.01,
            max=1.0,
        ),
    ] = 1.0,
    plain: Annotated[bool, typer.Option("--plain", help="Just the words, one per line.")] = False,
    model: Model = None,
) -> None:
    """Invent new words with your model."""
    saved, _ = open_model(model)
    start = check_prefix(saved, start)
    words = invent(
        saved.model, saved.vocab, np.random.default_rng(), count, temperature, start, top_p
    )
    if plain or not out.is_terminal:
        for w in words:
            print(display(w))
        return
    show_words(out, saved, words, temperature, start, animate=True, top_p=top_p)


def show_words(
    console: Console,
    saved: Saved,
    words: list[str],
    temperature: float,
    start: str = "",
    animate: bool = False,
    top_p: float = 1.0,
) -> None:
    extra = f" · temperature {temperature:g}" + (f" · top-p {top_p:g}" if top_p < 1 else "")
    extra += f" · starting with “{start}”" if start else ""
    console.print()
    console.print(Padding(name_line(saved, extra), (0, 0, 0, 2)))
    console.print()
    known = saved.known
    for w in words:
        row = word_row(w, known)
        if animate and console.is_terminal:
            with Live(console=console, auto_refresh=False, transient=False) as live:
                for i in range(1, len(row.plain) + 1):
                    live.update(Padding(row[:i], (0, 0, 0, 4)), refresh=True)
                    time.sleep(0.012)
        else:
            console.print(Padding(row, (0, 0, 0, 4)))
    console.print()
    console.print(Padding(tally(words, known), (0, 0, 0, 2)))
    console.print()


@app.command()
def explain(
    start: Annotated[
        str, typer.Argument(help="The beginning of a word, e.g. stego.", show_default=False)
    ],
    teach: Annotated[
        str | None,
        typer.Option(
            "--teach",
            help="Show what one step of learning toward this letter would do.",
            show_default=False,
        ),
    ] = None,
    model: Model = None,
) -> None:
    """Watch your model make one prediction up close, with backprop as a microscope."""
    saved, _ = open_model(model)
    prefix = check_prefix(saved, start)
    letter = None
    if teach:
        letter = teach.strip().lower()[:1]
        if not letter or letter not in saved.vocab.chars:
            raise fail(
                f"{saved.card.name} doesn't know {teach!r}.",
                f"It knows: [bold]{saved.vocab.chars}[/]",
            )
    show_explain(out, saved, prefix, letter)


def show_explain(console: Console, saved: Saved, prefix: str, teach: str | None = None) -> None:
    vocab = saved.vocab
    report = influence(saved.model, vocab, prefix)
    probs = softmax(report.trace.logits[0, -1])
    pick = Pick("." + prefix, probs, report.trace.weights[0, :, -1], 0.0, report.top)
    top = vocab.chars[report.top]
    parts: list = [
        reading(vocab, pick, saved.card.name),
        Text(""),
        Text.assemble(
            ("which letters mattered", "bold"),
            (f"  backprop from “{top}” down to each input letter", viz.FAINT),
        ),
        influence_view(report),
        Text(""),
    ]
    if teach:
        change = nudge(saved.model, vocab, prefix, vocab.chars.index(teach))
        parts += [
            Text.assemble(
                ("one step of learning toward “", "bold"),
                (teach, f"bold {viz.GREEN}"),
                ("”", "bold"),
                ("  on a copy: your model is unchanged", viz.FAINT),
            ),
            nudge_view(vocab, change),
            Text(""),
        ]
    else:
        parts.append(
            Text.assemble(
                ("The weights stay frozen: nothing learns at inference. Add ", viz.FAINT),
                (f"--teach {vocab.chars[report.runner_up]}", "bold"),
                (" to see what one step of learning would change.", viz.FAINT),
            )
        )
        parts.append(Text(""))
    console.print()
    console.print(Padding(Group(*parts), (0, 0, 0, 2)))


@app.command()
def info(model: Model = None) -> None:
    """Everything about your model."""
    saved, path = open_model(model)
    show_info(out, saved, path)


def show_info(console: Console, saved: Saved, path: Path) -> None:
    card = saved.card
    console.print()
    console.print(Padding(name_line(saved), (0, 0, 0, 2)))
    console.print()
    table = model_card(
        card,
        saved.model.size,
        saved.model.config,
        saved.vocab,
        len(saved.words),
        tidy(path),
        path.stat().st_size / 1024,
    )
    console.print(Padding(table, (0, 0, 0, 4)))
    if card.corpus in corpora.BUILT_IN:
        steps = f" --steps {card.steps}" if card.steps != STEPS else ""
        console.print()
        console.print(
            Text.assemble(
                ("  Make it again: ", viz.FAINT),
                (f"attention train --corpus {card.corpus} --seed {card.seed}{steps}", "bold"),
            )
        )
    console.print()


def main() -> None:
    app()
