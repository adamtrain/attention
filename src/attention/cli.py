"""`attention`: take the tour, or use the model you made on it."""

from __future__ import annotations

import json
import sys
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
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__, dashboard, longhand, tour, viz
from . import corpus as corpora
from .corpus import Dataset
from .explain import influence, nudge
from .finetune import CHAT_STEPS, answer, chat_trainer, exchanges
from .generate import Pick, picks, prompt
from .keys import Keys
from .model import part, softmax
from .store import (
    Outdated,
    Saved,
    chat_path,
    default_path,
    home,
    is_model,
    leftovers,
    load,
    remove,
    save,
)
from .tour.lab import Lab
from .tour.pretraining import summary, tidy
from .tour.stage import THEME, Quit, Stage
from .views import chip, influence_view, label, looking, nudge_view, running

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
        help="fables, fairytales, shakespeare, or a text file. Default: you choose.",
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
    except Outdated as e:
        raise fail(
            f"The model at {tidy(path)} can't be used: {e}.",
            "Train a new one: [bold]attention train[/]",
        ) from e
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise fail(
            f"Couldn't read the model at {tidy(path)}: {e}",
            "Train a new one: [bold]attention train[/]",
        ) from e


def check_prompt(saved: Saved, text: str) -> list[int]:
    """The text as the model will read it, if it fits in the context."""
    ids = prompt(saved.tokenizer, text)
    most = saved.model.config.context - 1
    if len(ids) > most:
        raise fail(f"That's {len(ids)} tokens: {saved.card.name} reads at most {most}.")
    return ids


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


def indent(renderable, left: int = 2) -> Padding:
    return Padding(renderable, (0, 0, 0, left))


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
    """[bold]attention[/]: build a small language model, watch it learn, and keep it.

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
    """Take the guided tour: tokens, attention, layers, training, the KV cache, and more."""
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
    steps: Annotated[
        int | None,
        typer.Option(
            "--steps",
            help="Training steps (default: enough for the text).",
            min=1,
            show_default=False,
        ),
    ] = None,
    fast: Annotated[bool, typer.Option("--fast", help="Don't stop to draw; just train.")] = False,
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
    from .tour.lab import Budget

    lab = Lab.create(chosen, seed, path, Budget(steps=steps))
    watch = dashboard.Watch(lab.trainer, lab.baselines, chosen, seed, lab.rng)
    width = min(out.width, 100) - 4
    out.print()
    try:
        if out.is_terminal and not fast:
            with Live(console=out, auto_refresh=False) as live, keyboard() as keys:

                def update(frame) -> None:
                    live.update(indent(frame), refresh=True)

                def interrupted(timeout: float) -> bool:
                    if keys is None:
                        time.sleep(timeout)
                        return False
                    return keys.read(timeout) is not None

                dashboard.run(watch, update, width, 30.0, interrupted)
        elif out.is_terminal:
            with Live(console=out, auto_refresh=False) as live:
                dashboard.run(
                    watch, lambda f: live.update(indent(f), refresh=True), width, 0, lambda _: False
                )
        else:
            dashboard.run(watch, lambda _: None, width, 0, lambda _: False)
    except KeyboardInterrupt:
        out.print("\n[faint]Stopped. Nothing was saved.[/]")
        raise typer.Exit(130) from None
    lab.seconds = watch.compute
    saved_to = lab.finish()
    chat_path(saved_to).unlink(missing_ok=True)  # it belonged to the old model
    out.print()
    out.print(indent(summary(lab, saved_to)))
    if previous:
        out.print(Text(f"    (replacing {previous})", style=viz.FAINT))
    out.print()
    out.print(
        Text.assemble(
            ("  Try: ", viz.FAINT),
            ("attention generate", "bold"),
            ("  or  ", viz.FAINT),
            ("attention chat", "bold"),
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
        str,
        typer.Argument(help="Text to carry on from (default: a blank page).", show_default=False),
    ] = "",
    count: Annotated[int, typer.Option("--count", "-n", help="How many.", min=1, max=50)] = 3,
    tokens: Annotated[
        int, typer.Option("--tokens", help="The most tokens to write, each time.", min=1, max=512)
    ] = 80,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", help="Lower is safer, higher is wilder.", min=0.01, max=5.0
        ),
    ] = 0.8,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k", "-k", help="Only pick from the k likeliest tokens (0: off).", min=0
        ),
    ] = 0,
    top_p: Annotated[
        float,
        typer.Option(
            "--top-p",
            "-p",
            help="Only pick from the likeliest tokens adding up to this.",
            min=0.01,
            max=1.0,
        ),
    ] = 1.0,
    plain: Annotated[bool, typer.Option("--plain", help="Just the text.")] = False,
    model: Model = None,
) -> None:
    """Write with your model: new documents, or carrying on from some text."""
    saved, _ = open_model(model)
    ids = check_prompt(saved, start)
    rng = np.random.default_rng()
    results = []
    for _ in range(count):
        steps = list(picks(saved.model, ids, rng, temperature, top_p, top_k, tokens))
        results.append([*ids, *(p.token for p in steps)])
    tok = saved.tokenizer
    if plain or not out.is_terminal:
        for written in results:
            print(tok.decode(written))
            print()
        return
    dials = f" · temperature {temperature:g}"
    dials += f" · top-k {top_k}" if top_k else ""
    dials += f" · top-p {top_p:g}" if top_p < 1 else ""
    out.print()
    out.print(indent(name_line(saved, dials)))
    out.print()
    width = min(out.width, 100) - 6
    for written in results:
        copied = saved.memory.copied(written)
        for line in running(tok, written, width, copied, start=len(ids)):
            out.print(indent(line, 4))
        out.print()
    note = "word for word from the training text (a run of 12 tokens or more)"
    out.print(indent(Text.assemble(("▆ ", "#7a2d40"), (note, viz.FAINT))))
    out.print()


@app.command()
def chat(
    model: Model = None,
    temperature: Annotated[
        float, typer.Option("--temperature", "-t", help="Lower is safer.", min=0.01, max=5.0)
    ] = 0.5,
    steps: Annotated[int, typer.Option("--steps", hidden=True, min=1)] = CHAT_STEPS,
) -> None:
    """Talk to the chat version of your model (it's made the first time, if need be)."""
    saved, path = open_model(model)
    where = chat_path(path)
    if where.exists():
        try:
            talker = load(where)
        except (OSError, ValueError, KeyError, TypeError, Outdated):
            talker = None
    else:
        talker = None
    if talker is None or talker.tokenizer.pieces != saved.tokenizer.pieces:
        talker = make_chat(saved, where, steps)
    tok = talker.tokenizer
    out.print()
    out.print(indent(name_line(talker, " · chat")))
    out.print(
        indent(Text("Ask it something (an empty line, or Ctrl-D, to finish).", style=viz.FAINT))
    )
    out.print()
    rng = np.random.default_rng()
    width = min(out.width, 100) - 8
    while True:
        try:
            request = out.input(Text.assemble(("  you  ", f"bold {viz.ACCENT}"))).strip()
        except (EOFError, KeyboardInterrupt):
            out.print()
            break
        if not request:
            break
        reply = answer(talker.model, tok, request, rng, temperature, 100)
        lines = running(
            tok, tok.encode(reply), width, talker.memory.copied(tok.encode(reply)), start=0
        )
        for i, line in enumerate(lines):
            out.print(
                Text.assemble(("  it   " if i == 0 else "       ", f"bold {viz.GREEN}"), line)
            )
        out.print()


def make_chat(saved: Saved, where: Path, steps: int = CHAT_STEPS) -> Saved:
    """Fine-tune a copy of the model on conversations, and save it beside the model."""
    corpus = corpora.built_in(saved.card.corpus) if saved.card.corpus in corpora.BUILT_IN else None
    if corpus is None:
        raise fail(
            "There's no chat version yet, and this model's text isn't one of the built-in ones.",
            "Take the tour's chat chapter with it to make one.",
        )
    data = Dataset.split(
        corpus, np.random.default_rng([saved.card.seed, 0]), tokenizer=saved.tokenizer
    )
    trainer = chat_trainer(
        saved.model, data, exchanges(corpus, data.train_docs), np.random.default_rng(), steps
    )
    out.print()
    columns = (
        TextColumn("  Teaching it to chat (once)", style=viz.FAINT),
        BarColumn(complete_style=viz.ACCENT, finished_style=viz.GREEN),
        TextColumn("{task.completed}/{task.total} steps", style=viz.FAINT),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=out, transient=True) as progress:
        task = progress.add_task("chat", total=trainer.steps)
        while not trainer.done:
            trainer.step()
            progress.update(task, completed=trainer.step_number)
    talker = Saved(trainer.model, saved.tokenizer, saved.card, saved.documents)
    save(where, talker)
    return talker


@app.command()
def explain(
    start: Annotated[
        str, typer.Argument(help='Some text, e.g. "The Fox and the".', show_default=False)
    ],
    teach: Annotated[
        str | None,
        typer.Option(
            "--teach",
            help="Show what one step of learning toward this word would do.",
            show_default=False,
        ),
    ] = None,
    model: Model = None,
) -> None:
    """Watch your model make one prediction up close: attention, layers, and backprop."""
    saved, _ = open_model(model)
    ids = check_prompt(saved, start)
    target, word = None, None
    if teach:
        word = teach if teach.startswith(" ") or not start else " " + teach
        pieces = saved.tokenizer.encode(word)
        if not pieces:
            raise fail(f"{teach!r} isn't something {saved.card.name} can write.")
        target = pieces[0]
        word = word.strip() if len(pieces) > 1 else None  # worth mentioning if it's in pieces
    show_explain(out, saved, ids, target, word)


def show_explain(
    console: Console,
    saved: Saved,
    ids: list[int],
    teach: int | None = None,
    word: str | None = None,
) -> None:
    tok = saved.tokenizer
    report = influence(saved.model, ids)
    probs = softmax(report.trace.logits[0, -1])
    weights = report.trace.weights[:, 0, :, -1]  # (layers, heads, positions)
    pick = Pick(tuple(ids), probs, weights, 0.0, report.top)
    width = min(console.width, 100) - 4
    lens = saved.model.lens(report.trace)[:, 0, -1]
    title = Text.assemble(
        ("◆ ", viz.PURPLE), (saved.card.name, f"bold {viz.PURPLE}"), (" reads ", viz.FAINT),
        (tok.decode(ids).replace("\n", "↵"), f"bold {viz.ACCENT}"),
    )  # fmt: skip
    parts: list = [
        title,
        Text(""),
        looking(tok, pick, width, top=6),
        Text(""),
        Text.assemble(("each layer's best guess", "bold"), ("  the logit lens", viz.FAINT)),
        lens_row(saved, lens),
        Text(""),
        Text.assemble(
            ("which tokens mattered", "bold"),
            (f"  backprop from {label(tok, report.top)} down to each token read", viz.FAINT),
        ),
        influence_view(tok, report),
        Text(""),
    ]
    if teach is not None:
        change = nudge(saved.model, ids, teach)
        parts += [
            Text.assemble(
                ("one step of learning toward ", "bold"),
                chip(tok, teach),
                (f", the first token of “{word}”" if word else "", viz.FAINT),
                ("  on a copy: your model is unchanged", viz.FAINT),
            ),
            nudge_view(tok, change),
            Text(""),
        ]
    else:
        parts.append(
            Text.assemble(
                ("The weights stay frozen: nothing learns at inference. Add ", viz.FAINT),
                (f'--teach "{suggestion(tok, probs, report.top)}"', "bold"),
                (" to see what one step of learning would change.", viz.FAINT),
            )
        )
        parts.append(Text(""))
    console.print()
    console.print(indent(Group(*parts)))


def suggestion(tok, probs: np.ndarray, top: int) -> str:
    """A likely next word, other than the likeliest, to try teaching toward."""
    for token in (int(t) for t in np.argsort(-probs)):
        word = tok.pieces[token].strip()
        if token != top and not tok.is_special(token) and word.isalpha() and len(word) > 2:
            return word
    return "the"


def lens_row(saved: Saved, lens: np.ndarray) -> Table:
    tok = saved.tokenizer
    grid = Table.grid(padding=(0, 2))
    names = ["embeddings", *(f"layer {i + 1}" for i in range(saved.model.config.layers))]
    for _ in names:
        grid.add_column(no_wrap=True)
    grid.add_row(*(Text(n, style=viz.FAINT) for n in names))
    grid.add_row(
        *(
            Text.assemble(
                chip(tok, int(p.argmax())), (f" {viz.percent(float(p.max()))}", viz.FAINT)
            )
            for p in lens
        )
    )
    return grid


@app.command("math")
def longhand_math(
    text: Annotated[
        str, typer.Argument(help="What the little model reads: words from fable titles.")
    ] = "The Fox and the",
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Train the little model differently.")
    ] = 1,
    auto: Annotated[
        bool, typer.Option("--auto", help="Play by itself, without waiting for keys.")
    ] = False,
    fast: Annotated[bool, typer.Option("--fast", help="Speed up the animations.")] = False,
) -> None:
    """Watch every operation of the forward pass, on a model small enough to print."""
    small = out.is_terminal and (out.width < 80 or out.height < longhand.ROWS)
    if small:
        err.print(
            f"[faint]attention math draws tables up to 80 columns wide and {longhand.ROWS} rows "
            f"tall, and this terminal is {out.width} × {out.height}: make it bigger if you can.[/]"
        )
    interactive = Keys.available() and not auto
    autopilot = auto or (out.is_terminal and not interactive)
    stage = Stage(
        out, animate=out.is_terminal, auto=3.5 if autopilot else None, speed=2.5 if fast else 1.0
    )

    def start() -> None:
        if small:
            stage.wait("start")  # a moment to make the terminal bigger first
        longhand.run(stage, text, seed)

    try:
        if interactive:
            with Keys() as keys:
                stage.keys = keys
                start()
        else:
            start()
    except ValueError as e:
        raise fail(str(e)[0].upper() + str(e)[1:] + ".") from e
    except (Quit, KeyboardInterrupt):
        out.print("\n[faint]Stopped.[/]")


@app.command()
def tokenize(
    text: Annotated[str, typer.Argument(help="Any text.", show_default=False)],
    model: Model = None,
) -> None:
    """See how your model's tokenizer splits some text into tokens."""
    saved, _ = open_model(model)
    tok = saved.tokenizer
    ids = tok.encode(text)
    width = min(out.width, 100) - 6
    out.print()
    for line in running(tok, ids, width):
        out.print(indent(line, 4))
    out.print()
    lines = [Text(no_wrap=True)]
    for token in ids:
        tile = Text.assemble(chip(tok, token), (f" {token} ", viz.FAINT))
        if len(lines[-1]) + len(tile) > width:
            lines.append(Text(no_wrap=True))
        lines[-1].append_text(tile)
    for line in lines:
        out.print(indent(line, 4))
    out.print()
    out.print(indent(Text.assemble(
        (f"{len(text):,} characters → ", viz.FAINT), (f"{len(ids):,} tokens", "bold"),
        (f"  ({len(text) / max(len(ids), 1):.1f} characters per token)", viz.FAINT),
    )))  # fmt: skip
    out.print()


@app.command()
def info(model: Model = None) -> None:
    """Everything about your model, including its config.json and its weights."""
    saved, path = open_model(model)
    show_info(out, saved, path)


def show_info(console: Console, saved: Saved, path: Path) -> None:
    from datetime import datetime

    card, model, tok = saved.card, saved.model, saved.tokenizer
    when = datetime.fromisoformat(card.trained_at).astimezone()
    console.print()
    console.print(indent(name_line(saved)))
    console.print()
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column()
    grid.add_row("made from", f"{len(saved.documents):,} {card.plural} (a tenth more held back)")
    grid.add_row(
        "trained",
        f"{when.day} {when:%b %Y} at {when:%H:%M} · {card.steps:,} steps · seed {card.seed}",
    )
    c = model.config
    grid.add_row(
        "parameters",
        f"{model.size:,}  ({c.layers} layers, {c.heads} heads, {c.width} numbers per token)",
    )
    grid.add_row(
        "vocabulary", f"{len(tok)} tokens, {len(tok.merges)} of them learned by byte-pair encoding"
    )
    grid.add_row("context", f"{c.context} tokens")
    grid.add_row(
        "loss",
        Text.assemble(
            (f"{card.loss:.2f}", "bold"), (" on its training text, ", viz.FAINT),
            (f"{card.val_loss:.2f}", "bold"), (" on held-back text", viz.FAINT),
            (f"  (counting token pairs: {card.pair_loss:.2f})", viz.FAINT),
        ),
    )  # fmt: skip
    kb = path.stat().st_size / 1024
    grid.add_row("file", f"{tidy(path)}  ({kb:,.0f} KB)")
    if chat_path(path).exists():
        grid.add_row("chat version", tidy(chat_path(path)))
    console.print(indent(grid, 4))
    console.print()
    console.print(indent(Text("config.json, as a Llama model would have it", style="bold")))
    config = json.dumps(model.config.hugging_face(), indent=2)
    console.print(indent(Syntax(config, "json", theme="ansi_dark", background_color="default"), 4))
    console.print()
    console.print(indent(Text("its weights, by the names a checkpoint uses", style="bold")))
    weights = Table.grid(padding=(0, 2))
    weights.add_column(no_wrap=True, style=viz.ACCENT)
    weights.add_column(no_wrap=True, justify="right", style=viz.FAINT)
    weights.add_column(no_wrap=True, justify="right")
    names = [n for n in model.params if not n.startswith("layers.") or n.startswith("layers.0.")]
    for name in names:
        w = model.params[name]
        shape = " × ".join(str(s) for s in w.shape)
        weights.add_row(part(name).hugging_face, shape, f"{w.size:,}")
    weights.add_row(Text(f"…and the same for layers 1 to {c.layers - 1}", style=viz.FAINT), "", "")
    console.print(indent(weights, 4))
    console.print(
        indent(
            Text(
                "(Shapes are in × out, the way your model multiplies. PyTorch checkpoints store "
                "each grid the other way round, out × in.)",
                style=viz.FAINT,
            ),
            4,
        )
    )
    console.print()
    if card.corpus in corpora.BUILT_IN:
        console.print(
            Text.assemble(
                ("  Make it again: ", viz.FAINT),
                (f"attention train --corpus {card.corpus} --seed {card.seed}", "bold"),
            )
        )
        console.print()


@app.command()
def clean(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Delete without asking first.")] = False,
    model: Annotated[
        Path | None,
        typer.Option(
            "--model", "-m", help="Also delete a model saved somewhere else.", show_default=False
        ),
    ] = None,
) -> None:
    """Delete everything attention has saved: your models, and their folder if that empties it."""
    if model is not None and model.exists() and not is_model(model):
        raise fail(f"{tidy(model)} isn't an attention model, so it's staying put.")
    doomed = leftovers(model)
    folder = home()
    out.print()
    if not doomed:
        gone = remove([])  # an empty folder of ours is still ours to tidy away
        where = f" in {tidy(folder)}" if folder.exists() or gone else ""
        out.print(
            Text(f"  Nothing to clean up: attention hasn't saved anything{where}.", style=viz.FAINT)
        )
        if gone:
            out.print(
                Text.assemble(
                    ("  ✓ ", f"bold {viz.GREEN}"), (f"Removed the empty folder {tidy(folder)}", "")
                )
            )
        out.print()
        return

    out.print(Text("  This deletes, for good:", style="bold"))
    for path in doomed:
        out.print(indent(Text.assemble(("• ", viz.FAINT), (tidy(path), "bold")), 4))
        out.print(indent(Text(describe(path), style=viz.FAINT), 6))
    out.print()
    if not yes:
        try:
            go = typer.confirm(f"  Delete {'it' if len(doomed) == 1 else 'them'}?", default=False)
        except typer.Abort:
            go = False
        if not sys.stdin.isatty():
            out.print()  # the answer came from a pipe, so nothing ended the prompt's line
        if not go:
            out.print(Text("  Nothing deleted. (Add --yes to skip the question.)", style=viz.FAINT))
            out.print()
            return
    gone = remove(doomed)
    n = len(doomed)
    out.print(
        Text.assemble(
            ("  ✓ ", f"bold {viz.GREEN}"), (f"Deleted {n} file{'s' if n != 1 else ''}", "")
        )
    )
    if gone:
        out.print(Text.assemble(("  ✓ ", f"bold {viz.GREEN}"), (f"Removed {tidy(folder)}", "")))
    elif folder.is_dir():
        out.print(
            Text(f"  Left {tidy(folder)} in place: it has other files in it.", style=viz.FAINT)
        )
    out.print()


def describe(path: Path) -> str:
    """A few words about a file attention saved."""
    size = f"{path.stat().st_size / 1024:,.0f} KB"
    if path.name.endswith(".tmp.npz"):
        return f"an unfinished save · {size}"
    try:
        card = load(path).card
    except Outdated:
        return f"a model from an older version of attention · {size}"
    except (OSError, ValueError, KeyError, TypeError):
        return f"a saved model · {size}"
    kind = "chat version" if path.name.endswith(".chat.npz") else f"{card.noun} model"
    return f"{card.name}, a {kind} · {size}"


def main() -> None:
    app()
