"""Chapter: from predicting text to answering. Chat templates, loss masks and feedback."""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..finetune import (
    Exchange,
    Feedback,
    answer,
    chat_ids,
    chat_trainer,
    exchanges,
    learn_from,
)
from ..generate import prompt, write
from ..model import Transformer
from ..store import Card, Saved, chat_path, save
from ..tokenizer import ASSISTANT
from ..train import Trainer, smooth
from ..views import chip, display, running
from .lab import Lab
from .pretraining import tidy
from .stage import Frame, Stage

ASKED = 3  # requests held back from chat training, to ask it afterwards
CHOICES = 4
FPS = 6


def run(stage: Stage, lab: Lab) -> None:
    c, tok = lab.corpus, lab.tokenizer
    pairs = exchanges(c, lab.data.train_docs)
    order = lab.dice(19).permutation(len(pairs))
    asked = [pairs[int(i)] for i in order[:ASKED]]
    taught = [pairs[int(i)] for i in order[ASKED:]]
    first = asked[0]

    stage.say(
        "Your model continues documents. Give it a request as plain text, and it carries on as "
        "if it were reading a document that happened to start that way:"
    )
    start = prompt(tok, first.request)
    carried = write(lab.model, tok, lab.dice(20), first.request, 0.7, limit=40)
    stage.show(*running(tok, start + tok.encode(carried), stage.width - 4, start=len(start))[:5])
    stage.say(
        "That's all a pretrained model is: a very good document completer. It has no idea "
        "anyone is asking it anything. ChatGPT and Claude started out like this too. Turning "
        "one into an assistant is more fine-tuning, on conversations."
    )
    stage.wait("see a conversation")

    shown = Exchange(first.request, first.reply[:80].rsplit(" ", 1)[0])
    ids = chat_ids(tok, shown.request, shown.reply)
    stage.say(
        "Each conversation is written out as one piece of text, in a fixed pattern called a "
        "[b]chat template[/b], with special tokens marking who's speaking. Your model's uses the "
        "two it has had reserved since the first chapter, `<|user|>` and `<|assistant|>`, and "
        "ends each reply with `<|endoftext|>`:"
    )
    stage.show(template_view(lab, ids))
    stage.say(
        "Real templates are the same idea with different markers. Llama 3 writes "
        "`<|start_header_id|>user<|end_header_id|>` … `<|eot_id|>`; the ChatML format many "
        "others use writes `<|im_start|>user` … `<|im_end|>`. Use a model with the wrong "
        "template and it goes to pieces, because it has never seen that pattern. When you send "
        "a chatbot a message, it's slotted into the template, and the model is asked what "
        "comes after the assistant marker."
    )
    stage.say(
        "One more change: [b]the loss only counts the reply[/b]. Every target up to and "
        "including `<|assistant|>` is set to −1, which the loss skips, so the model learns to "
        "answer requests, not to write them. Here's which quizzes in that conversation count:"
    )
    stage.show(mask_view(lab, ids, stage.width))
    stage.wait("teach it to answer")

    steps = lab.budget.chat
    trainer = chat_trainer(lab.model, lab.data, taught, lab.dice(21), steps)
    stage.say(
        f"Here's a copy of your model learning the template from {len(taught):,} "
        f"conversations made from the {c.plural}, for {steps} steps. Below the loss, its "
        "reply to the request above, as it learns:"
    )

    def teaching() -> Iterator[Frame]:
        nonlocal trainer
        if trainer.step_number:
            trainer = chat_trainer(lab.model, lab.data, taught, lab.dice(21), steps)
        return training(lab, trainer, first.request, stage.width)

    stage.play(teaching, fps=FPS, start="teach it", again="teach it again")
    chat = trainer.model
    stage.say(
        f"Now some requests it wasn't taught, though it read those {c.plural} in pretraining:"
    )
    answered = [
        (
            ex.request,
            answer(chat, tok, ex.request, np.random.default_rng([lab.seed, 24, i]), 0.3, 60),
        )
        for i, ex in enumerate(asked)
    ]
    stage.show(replies(answered, stage.width))
    followed = sum(bool(names(request) & names(reply)) for request, reply in answered)
    if followed == len(answered):
        picked = "and each time it picked up a word from your request"
    elif followed:
        picked = f"and {followed} of {len(answered)} times it picked up a word from your request"
    else:
        picked = (
            "but it didn't pick up the words of your request: copying from the prompt is exactly "
            "what the induction heads of the last chapters do, and yours never grew any"
        )
    stage.say(
        f"It answers now, in the right shape, {picked}. Much of the rest is made up: it read "
        f"these {c.plural} during pretraining, but not well enough to tell them back. What a "
        "chat model knows comes from pretraining; fine-tuning only teaches it how to answer. "
        "And when it doesn't know, it answers anyway, just as fluently. That's where a "
        "chatbot's confident mistakes come from."
    )
    path = save_chat(lab, chat)
    if stage.keys is not None and stage.auto is None:
        conversation(stage, lab, chat, first.request)
    stage.say(
        f"It's saved beside your model, at `{tidy(path)}`. From any terminal, "
        "`attention chat` talks to it."
    )
    stage.wait("give it feedback")

    stage.say(
        "Last comes learning from [b]feedback[/b]. People compare a model's answers and pick "
        "the ones they prefer, and training makes those likelier and the others less so. It's "
        "the same gradient descent, aimed at what people liked instead of at what a text said."
    )
    if stage.keys is not None and stage.auto is None:
        stage.say("Your turn. Here are some replies to one request.")
        rounds(stage, lab, chat, first.request)
    else:
        options = [answer(chat, tok, first.request, lab.dice(22), 0.9, 40) for _ in range(CHOICES)]
        liked = options[:1]
        stage.say(f"Say you liked the first reply to “{first.request}”, and not the rest:")
        result = learn_from(chat, tok, liked, options[1:], request=first.request)
        stage.show(feedback_view(result, options, set(liked), stage.width))
    stage.note(
        "Real systems don't learn straight from each click. They train a second model to "
        "predict which answers people will prefer (a reward model), then train the chatbot to "
        "score well with it, at enormous scale: that's RLHF, reinforcement learning from human "
        "feedback. Push too hard and the model learns to please the scorer rather than to get "
        "better, so it's done with care."
    )


# ── The template ──────────────────────────────────────────────────────────────


def template_view(lab: Lab, ids: list[int]) -> Text:
    tok = lab.tokenizer
    text = Text()
    for token in ids:
        if tok.is_special(token):
            text.append_text(chip(tok, token))
        else:
            text.append(tok.pieces[token].replace("\n", "↵\n"), style="italic")
    return text


def mask_view(lab: Lab, ids: list[int], width: int, most: int = 18) -> Group:
    tok = lab.tokenizer
    start = ids.index(ASSISTANT)
    shown = ids[: min(len(ids) - 1, start + most - 6)][-most:]
    first = min(len(ids) - 1, start + most - 6) - len(shown)  # the position of shown[0]
    row = Text("reads    ", style=viz.FAINT, no_wrap=True)
    marks = Text("graded?  ", style=viz.FAINT, no_wrap=True)
    for i, token in enumerate(shown):
        tile = chip(tok, token)
        if len(row) + len(tile) + 1 > width:
            break
        row.append_text(tile)
        row.append(" ")
        graded = first + i >= start
        marks.append(
            f"{'✓' if graded else '−':^{len(tile)}} ", style=viz.GREEN if graded else viz.FAINT
        )
    return Group(row, marks)


# ── Teaching ──────────────────────────────────────────────────────────────────


def training(lab: Lab, trainer: Trainer, request: str, width: int) -> Iterator[Frame]:
    rng = np.random.default_rng([lab.seed, 23])
    reply = answer(trainer.model, lab.tokenizer, request, rng, 0.3, 40)
    yield progress(lab, trainer, request, reply, width)
    while not trainer.done:
        target = min(trainer.steps, trainer.step_number + max(1, trainer.steps // 12))
        while trainer.step_number < target:
            trainer.step()
        reply = answer(
            trainer.model, lab.tokenizer, request, np.random.default_rng([lab.seed, 23]), 0.3, 40
        )
        yield progress(lab, trainer, request, reply, width)


def progress(lab: Lab, trainer: Trainer, request: str, reply: str, width: int) -> Group:
    losses = smooth(trainer.losses, 0.1)
    head = Text.assemble(
        ("step ", viz.FAINT), (f"{trainer.step_number:>3}", "bold"), (f" / {trainer.steps}  ", viz.FAINT),
        viz.bar(trainer.step_number / trainer.steps, 20, viz.ACCENT),
        ("   loss on the replies ", viz.FAINT), (f"{losses[-1]:.2f}" if losses else "–", "bold"),
    )  # fmt: skip
    return Group(
        head,
        Text(""),
        Text.assemble(("you  ", f"bold {viz.ACCENT}"), (request, "italic")),
        Text.assemble(("it   ", f"bold {viz.GREEN}"), (display(reply or "…", width - 10), "")),
    )


def names(text: str) -> set[str]:
    """The capitalized words in a text, besides the ones any request starts with."""
    return set(re.findall(r"\b[A-Z][a-z]+\b", text)) - {"Tell", "The", "A", "An", "Go"}


def replies(answered: list[tuple[str, str]], width: int) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, style="bold")
    grid.add_column()
    for request, reply in answered:
        grid.add_row(Text("you", style=viz.ACCENT), Text(request, style="italic"))
        grid.add_row(Text("it", style=viz.GREEN), Text(display(reply, 3 * (width - 8))))
        grid.add_row("", "")
    return grid


def save_chat(lab: Lab, chat: Transformer):
    card = Card(
        name=lab.name, corpus=lab.corpus.key, title=lab.corpus.title, noun=lab.corpus.noun,
        plural=lab.corpus.plural, seed=lab.seed, steps=lab.trainer.steps, loss=0.0,
        val_loss=lab.trainer.val_losses[-1][1] if lab.trainer.val_losses else float("nan"),
        pair_loss=lab.baselines.pairs, trained_at=Card.now(), probe=lab.corpus.probe,
    )  # fmt: skip
    return save(chat_path(lab.model_path), Saved(chat, lab.tokenizer, card, lab.data.train_docs))


def conversation(stage: Stage, lab: Lab, chat: Transformer, example: str) -> None:
    """Type requests and read the replies, until the viewer moves on."""
    history: list[tuple[str, str]] = []
    rng = lab.dice(25)

    def picture(typed: str) -> Group:
        lines = [
            Text("Your turn: type a request and press enter.", style="bold"),
            Text.assemble(
                ("for example  ", viz.FAINT), (example, "italic"), ("     → to move on", viz.FAINT)
            ),
            Text(""),
        ]
        for request, reply in history[-3:]:
            lines.append(Text.assemble(("you  ", f"bold {viz.ACCENT}"), (request, "italic")))
            lines.append(
                Text.assemble(
                    ("it   ", f"bold {viz.GREEN}"), (display(reply, 3 * stage.width - 30), "")
                )
            )
            lines.append(Text(""))
        lines.append(
            Text.assemble(("you  ", f"bold {viz.ACCENT}"), (typed, "italic"), ("▌", viz.ACCENT))
        )
        return Group(*lines)

    with stage.live() as live:
        while (request := stage.line(live, picture)) is not None:
            history.append((request, answer(chat, lab.tokenizer, request, rng, 0.5, 60)))
        live.update(stage.pad(picture("")), refresh=True)


# ── Feedback ──────────────────────────────────────────────────────────────────


def feedback_view(result: Feedback, texts: list[str], liked: set[str], width: int) -> Group:
    most = min(36, width - 42)  # the reply, beside ♥, the chances, and "↓ 1.00× less likely"
    grid = Table.grid(padding=(0, 2))
    for justify in ("left", "left", "right", "center", "left", "left"):
        grid.add_column(justify=justify, no_wrap=True)
    for text in texts:
        was, now = result.chance(text)
        up = now > was
        factor = now / was if up else was / now
        grid.add_row(
            Text("♥" if text in liked else " ", style=f"bold {viz.RED}"),
            Text(display(text, most), style="bold" if text in liked else viz.FAINT),
            Text(viz.percent(was), style=viz.FAINT),
            Text("→", style=viz.FAINT),
            Text(viz.percent(now), style="bold"),
            Text(
                f"{'↑' if up else '↓'} {factor:.2f}× {'likelier' if up else 'less likely'}",
                style=viz.GREEN if up else viz.RED,
            ),
        )
    heading = Text(
        "the typical chance it gives each token of the reply, before and after", style=viz.FAINT
    )
    return Group(heading, grid)


def rounds(stage: Stage, lab: Lab, chat: Transformer, request: str, most: int = 3) -> None:
    """Pick favorites, watch the model shift toward them; again, as often as you like."""
    model = chat
    rng = lab.dice(26)
    for number in range(1, most + 1):
        options = [answer(model, lab.tokenizer, request, rng, 0.9, 40) for _ in range(CHOICES)]
        liked: set[str] = set()
        with stage.live() as live:
            while True:
                live.update(stage.pad(choosing(request, options, liked, number)), refresh=True)
                key = stage.key()
                if key and key.isdigit() and 1 <= int(key) <= len(options):
                    liked ^= {options[int(key) - 1]}
                elif key in ("enter", "space") and liked:
                    break
                elif key == "right":
                    return
        disliked = [t for t in options if t not in liked]
        result = learn_from(model, lab.tokenizer, sorted(liked), disliked, request=request)
        stage.show(feedback_view(result, options, liked, stage.width))
        model = result.model
        if number < most and stage.wait("pick again, or → to move on") == "right":
            return


def choosing(request: str, options: list[str], liked: set[str], number: int) -> Group:
    lines = [
        Text.assemble(("round ", viz.FAINT), (str(number), "bold"),
                      ("   press the numbers of the replies you like, then enter  ·  → skip", viz.FAINT)),
        Text.assemble(("you  ", f"bold {viz.ACCENT}"), (request, "italic")),
    ]  # fmt: skip
    for i, text in enumerate(options, 1):
        on = text in liked
        lines.append(
            Text.assemble(
                (f" {i} ", f"bold {viz.ACCENT}"),
                ("♥ " if on else "  ", f"bold {viz.RED}"),
                (display(text, 70), "bold" if on else ""),
            )
        )
    return Group(*lines)
