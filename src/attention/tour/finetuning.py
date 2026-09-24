"""Chapter: fine-tuning and learning from feedback, the steps between pretraining and a chatbot."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..finetune import LEARNING_RATE, STEPS, Feedback, drift, learn_from, niche_trainer, share
from ..generate import invent
from ..train import LEARNING_RATE as PRETRAINING_RATE
from ..train import Trainer, evaluate
from ..views import display
from .lab import Lab
from .stage import Frame, Stage

SAMPLES = 8
CHOICES = 6
SECONDS = 8
FPS = 8


def run(stage: Stage, lab: Lab) -> None:
    c, data = lab.corpus, lab.data
    niche = c.niche
    examples = [w for w in data.train if niche.has(w)]
    stage.say(
        f"Pretraining gave you a model that writes any kind of {c.noun}. [b]Fine-tuning[/b] "
        f"is how you get one that does something more particular. Say you only want "
        f"{niche.label}. There are {len(examples)} in the training data:"
    )
    stage.show(word_grid(examples, stage.width))
    stage.say(
        "Fine-tuning is the pretraining loop all over again: run some examples forward, "
        "measure the loss, backpropagate, and step every weight downhill. Only three things "
        "change:",
        gap=False,
    )
    stage.console.print()
    stage.show(recipe(lab, len(examples)))
    trainer = niche_trainer(lab.model, data, niche, lab.dice(6))
    batch = trainer.batch_size
    stage.say(
        f"Starting from pretrained weights is what makes it work with so few examples: the "
        f"model already knows how {c.plural} are put together, so all it has to learn is "
        "which ones to prefer. The gentle pace is so that it adjusts what it knows rather "
        f"than overwriting it. And with only {len(examples)} examples, {STEPS} steps of "
        f"{batch} means it sees each one about {STEPS * batch // len(examples)} times."
    )
    rng = np.random.default_rng([lab.seed, 11])
    before_words = invent(lab.model, lab.vocab, rng, 40, 0.8)
    before = share(before_words, niche)
    stage.say(
        f"Right now, {before:.0%} of what your model invents ends in -{niche.ending}. Here's "
        "a copy of it, fine-tuning. Your own model isn't touched:"
    )

    def tuning() -> Iterator[Frame]:
        nonlocal trainer
        if trainer.step_number:  # a replay: start again from your model, with new dice
            trainer = niche_trainer(lab.model, data, niche, lab.dice(6))
        return training(lab, trainer, rng, before_words[:SAMPLES], before)

    stage.play(tuning, fps=FPS, start="fine-tune it", again="try again")

    after_words = invent(trainer.model, lab.vocab, rng, 40, 0.8)
    after = share(after_words, niche)
    general_before = evaluate(lab.model, data, data.val)
    general_after = evaluate(trainer.model, data, data.val)
    stage.show(compare(before_words, after_words, before, after, niche.ending, set(c.words)))
    stage.say(
        f"From {before:.0%} to {after:.0%}, from {len(examples)} examples and a fraction of a "
        "second of training. With so few examples, each seen so many times, some of what it "
        "writes now are straight copies of them."
    )
    stage.say(
        "And its weights hardly moved. Measured against their size, fine-tuning changed them "
        f"by [b]{drift(trainer.model, lab.model):.0%}[/b]; pretraining had moved them "
        f"{drift(lab.model, lab.initial):.0%} from where they started. Fine-tuning doesn't "
        "build new knowledge so much as tilt the model toward part of what it already has."
    )
    stage.say(
        f"There's a price. Its loss on held-back {c.plural} of every kind rose from "
        f"{general_before:.2f} to [b]{general_after:.2f}[/b]: it got worse at everything "
        "else. Fine-tune too hard, or on too narrow a diet, and a model forgets what it used to "
        "know. That's called [b]catastrophic forgetting[/b], and it's why fine-tuning is kept "
        "short and gentle."
    )
    stage.note(
        "Fine-tuning a big model changes billions of weights and needs a lot of memory, so "
        "people often freeze the original weights and train a small add-on instead. The most "
        "popular kind, LoRA, learns a correction to each grid of weights that takes a tiny "
        "fraction of the numbers."
    )
    stage.wait("see how this makes a chatbot")

    stage.say(
        "This is how chat assistants are made. Pretrained on the internet, a model will continue "
        "any text the way a web page might, so a question could be followed by more questions. "
        "To fix that, it's fine-tuned on many thousands of example conversations, each a request and a "
        "good reply."
    )
    stage.say(
        "Each conversation is written out as one piece of text, with special tokens marking "
        "who's talking, something like `<user>` … `<assistant>` …, and the loss only counts "
        "the assistant's parts, so the model learns to write replies, not requests. When you "
        "chat with one, your message goes after `<user>`, and the model predicts what comes "
        "after `<assistant>`. Same model, same next-token prediction, a different taste in "
        "what comes next."
    )
    stage.say(
        "Then comes learning from [b]feedback[/b]. People compare a model's answers and pick "
        "the ones they prefer, and training makes those likelier and the others less so. It's "
        "the same gradient descent, aimed at what people liked instead of at what the text "
        "said."
    )
    if stage.keys is not None and stage.auto is None:
        stage.say("Your turn. Here are some fresh inventions from your pretrained model.")
        rounds(stage, lab)
    else:
        words = invent(lab.model, lab.vocab, rng, CHOICES, 0.8)
        liked = words[:2]
        stage.say(f"Say you liked {display(liked[0])} and {display(liked[1])}, and not the rest:")
        result = learn_from(lab.model, data, liked, words[2:])
        stage.show(feedback_view(result, words, set(liked)))
        if note := resemblance(result, words[2:]):
            stage.say(note)
    stage.note(
        "Real systems don't learn straight from each click. They train a second model to predict "
        "which answers people will prefer (a reward model), then train the chatbot to score "
        "well with it, at enormous scale. Push too hard and the model learns to please the "
        "scorer rather than to get better, so it's done with care."
    )


def recipe(lab: Lab, examples: int) -> Table:
    """What fine-tuning does differently from pretraining."""
    pre = lab.trainer
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {viz.ACCENT}", no_wrap=True)
    grid.add_column()
    grid.add_row("starts from", "your pretrained weights, instead of random ones")
    grid.add_row(
        "learns from",
        f"only the {examples} that end in -{lab.corpus.niche.ending}, instead of all "
        f"{len(lab.data.train)} {lab.corpus.plural}",
    )
    grid.add_row(
        "goes gently",
        f"{STEPS} steps instead of {pre.steps}, with a learning rate of {LEARNING_RATE} instead "
        f"of {PRETRAINING_RATE}",
    )
    return grid


def word_grid(words: list[str], width: int, most: int = 18) -> Table:
    shown = words[:most]
    col = max(len(w) for w in shown) + 2
    cols = max(1, min(6, width // col))
    grid = Table.grid(padding=(0, 2))
    for _ in range(cols):
        grid.add_column(no_wrap=True)
    for start in range(0, len(shown), cols):
        row = [Text(display(w), style="italic") for w in shown[start : start + cols]]
        grid.add_row(*row, *([""] * (cols - len(row))))
    if len(words) > most:
        grid.add_row(Text(f"…and {len(words) - most} more", style=viz.FAINT), *([""] * (cols - 1)))
    return grid


def training(
    lab: Lab, trainer: Trainer, rng: np.random.Generator, samples: list[str], before: float
) -> Iterator[Frame]:
    """Fine-tuning, a picture at a time: what it writes, and how much of it is in the niche."""
    niche = lab.corpus.niche
    yield progress(trainer, samples, before, niche.ending)
    total, shown = SECONDS * FPS, 0
    while not trainer.done:
        shown += 1
        target = trainer.steps * min(1.0, shown / total)
        trainer.step()
        while trainer.step_number < target and not trainer.done:
            trainer.step()
        if trainer.step_number % 5 == 0 or trainer.done:
            samples = invent(trainer.model, lab.vocab, rng, SAMPLES, 0.8)
        now = share(invent(trainer.model, lab.vocab, rng, 20, 0.8), niche)
        yield progress(trainer, samples, now, niche.ending)


def progress(trainer: Trainer, samples: list[str], now: float, ending: str) -> Group:
    head = Text.assemble(
        ("step ", viz.FAINT), (f"{trainer.step_number:>2}", "bold"), (f" / {trainer.steps}  ", viz.FAINT),
        viz.bar(trainer.step_number / trainer.steps, 20, viz.ACCENT),
    )  # fmt: skip
    meter = Text.assemble(
        (f"ends in -{ending}  ", viz.FAINT), viz.bar(now, 30, viz.GREEN), (f" {now:4.0%}", "bold")
    )
    lines = [Text.assemble(("✦ ", viz.GREEN), highlight(w, ending)) for w in samples]
    return Group(head, Text(""), meter, Text(""), *lines)


def highlight(word: str, ending: str) -> Text:
    shown = display(word)
    if word.endswith(ending):
        cut = len(shown) - len(ending)
        return Text.assemble((shown[:cut], ""), (shown[cut:], f"bold {viz.GREEN}"))
    return Text(shown)


def compare(
    before: list[str], after: list[str], was: float, now: float, ending: str, known: set[str]
) -> Table:
    def mark(word: str) -> Text:
        new = word not in known
        return Text.assemble(
            ("✦ " if new else "· ", viz.GREEN if new else viz.FAINT), highlight(word, ending)
        )

    grid = Table.grid(padding=(0, 6))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(
        Text.assemble(("before ", "bold"), (f"{was:.0%} end in -{ending}", viz.FAINT)),
        Text.assemble(("after ", "bold"), (f"{now:.0%} end in -{ending}", viz.FAINT)),
    )
    for a, b in zip(before[:SAMPLES], after[:SAMPLES], strict=False):
        grid.add_row(mark(a), mark(b))
    grid.add_row(
        Text.assemble(("✦ ", viz.GREEN), ("new   ", viz.FAINT), ("· ", viz.FAINT),
                      ("straight from the training data", viz.FAINT)),
        "",
    )  # fmt: skip
    return grid


def odds(n: float) -> str:
    for size, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if n >= size:
            return f"1 in {n / size:,.1f} {name}"
    return f"1 in {n:,.0f}"


def resemblance(result: Feedback, disliked: list[str]) -> str:
    """A note for when words nobody picked got likelier anyway."""
    risers = [w for w in disliked if result.after[w] > result.before[w]]
    if not risers:
        return ""
    return (
        f"{display(risers[0])} got likelier even though you didn't pick it: it shares patterns "
        "with the ones you did. The model learns patterns, not a list of answers."
    )


def feedback_view(result: Feedback, words: list[str], liked: set[str]) -> Table:
    grid = Table.grid(padding=(0, 2))
    for justify in ("left", "left", "right", "center", "right", "left"):
        grid.add_column(justify=justify, no_wrap=True)  # type: ignore[arg-type]
    grid.add_row("", "", Text("chance of writing it", style=viz.FAINT), "", "", "")
    for w in words:
        was, now = result.odds(w)
        up = now < was
        factor = was / now if up else now / was
        grid.add_row(
            Text("♥" if w in liked else " ", style=f"bold {viz.RED}"),
            Text(display(w), style="bold" if w in liked else viz.FAINT),
            Text(odds(was), style=viz.FAINT),
            Text("→", style=viz.FAINT),
            Text(odds(now), style="bold"),
            Text(
                f"{'↑' if up else '↓'} {factor:,.1f}× {'likelier' if up else 'less likely'}",
                style=viz.GREEN if up else viz.RED,
            ),
        )
    return grid


def rounds(stage: Stage, lab: Lab, most: int = 5) -> None:
    """Pick favorites, watch the model shift toward them; again, as often as you like."""
    model = lab.model
    rng = np.random.default_rng([lab.seed, 12])
    for number in range(1, most + 1):
        words = invent(model, lab.vocab, rng, CHOICES, 0.8)
        liked: set[str] = set()
        with stage.live() as live:
            while True:
                live.update(stage.pad(choosing(words, liked, number)), refresh=True)
                key = stage.key()
                if key and key.isdigit() and 1 <= int(key) <= len(words):
                    liked ^= {words[int(key) - 1]}
                elif key in ("enter", "space") and liked:
                    break
                elif key == "right":
                    return
        disliked = [w for w in words if w not in liked]
        result = learn_from(model, lab.data, sorted(liked), disliked)
        stage.show(feedback_view(result, words, liked))
        if note := resemblance(result, disliked):
            stage.say(note)
        fresh = invent(result.model, lab.vocab, rng, CHOICES, 0.8)
        stage.show(
            Text.assemble(
                ("what it writes now  ", viz.FAINT), (", ".join(map(display, fresh)), "italic")
            )
        )
        model = result.model
        if number < most and stage.wait("pick again, or → to move on") == "right":
            return


def choosing(words: list[str], liked: set[str], number: int) -> Group:
    lines = [
        Text.assemble(("round ", viz.FAINT), (str(number), "bold"),
                      ("   press the numbers of the ones you like, then enter  ·  → skip", viz.FAINT))
    ]  # fmt: skip
    for i, w in enumerate(words, 1):
        on = w in liked
        lines.append(
            Text.assemble(
                (f" {i} ", f"bold {viz.ACCENT}"),
                ("♥ " if on else "  ", f"bold {viz.RED}"),
                (display(w), "bold" if on else ""),
            )
        )
    return Group(*lines)
