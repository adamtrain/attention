"""Chapter: backprop after training, used as a microscope instead of for learning."""

from __future__ import annotations

from .. import viz
from ..explain import influence, nudge
from ..views import influence_view, nudge_view
from .lab import Lab
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    stage.say(
        "Is there backprop when a model writes? [b]No.[/b] Generating only runs the model "
        "forward, and the weights stay frozen. Your model didn't learn a thing while it wrote in "
        "the last chapter, and a chatbot doesn't learn while it talks to you."
    )
    inputs, _ = lab.example()
    focus = max(1, round(len(lab.corpus.example) * 0.65))
    prefix = lab.vocab.decode(inputs[0, 1 : focus + 1])
    report = influence(lab.model, lab.vocab, prefix)
    stage.say(
        "But backprop still makes a good microscope. Take one prediction and backpropagate from "
        "it, not to change any weights, but all the way down to the input letters. The size of "
        "the gradient at each letter says how much that letter could sway the prediction. "
        f"After `.{prefix}`, the model says `{lab.vocab.chars[report.top]}` "
        f"({viz.percent(report.probs[report.top])}). Which letters mattered?"
    )
    stage.show(influence_view(report))
    stage.say(
        "Researchers use tricks like this to work out why a model said what it said. It's one "
        "small corner of a field called interpretability."
    )
    stage.wait("see what one step of learning would do")

    alternative = report.runner_up
    change = nudge(lab.model, lab.vocab, prefix, alternative)
    stage.say(
        f"And what if we insisted the answer was `{lab.vocab.chars[alternative]}`? One backward "
        "pass from that answer, one step downhill on a copy of the model, and look again:"
    )
    stage.show(nudge_view(lab.vocab, change))
    stage.say(
        f"That's learning in miniature, and exactly what happened {lab.trainer.steps} times "
        "during pretraining. We threw the copy away: your saved model is unchanged. The next "
        "chapter does it on purpose."
    )
