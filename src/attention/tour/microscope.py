"""Chapter: backprop after training, used as a microscope instead of for learning."""

from __future__ import annotations

from .. import viz
from ..explain import influence, nudge
from ..views import influence_view, label, nudge_view
from .lab import Lab
from .stage import Stage


def run(stage: Stage, lab: Lab) -> None:
    tok = lab.tokenizer
    stage.say(
        "Is there backprop when a model writes? [b]No.[/b] Generating only runs the model "
        "forward, and the weights stay frozen, for your model and for chatbots alike."
    )
    inputs, _ = lab.example()
    ids = [int(t) for t in inputs[0, : min(inputs.shape[1], 16)]]
    report = influence(lab.model, ids)
    text = tok.decode(ids).replace("\n", "↵")
    stage.say(
        "But backprop still makes a good microscope. Take one prediction and backpropagate from "
        "it, not to change any weights, but all the way down to the tokens that went in. The "
        "size of the gradient at each token says how much that token could sway the prediction. "
        f"After `{text}`, your model says `{label(tok, report.top)}` "
        f"({viz.percent(float(report.probs[report.top]))}). Which tokens mattered?"
    )
    stage.show(influence_view(tok, report))
    stage.say(
        "Researchers use tools like this, and much sharper ones, to work out why a model said "
        "what it said. It's one corner of a field called interpretability."
    )
    stage.wait("see what one step of learning would do")

    alternative = report.runner_up
    change = nudge(lab.model, ids, alternative)
    stage.say(
        f"And what if we insisted the answer was `{label(tok, alternative)}`? One backward pass "
        "from that answer, one step downhill on a copy of the model, and look again:"
    )
    stage.show(nudge_view(tok, change))
    stage.say(
        f"That's learning in miniature, and exactly what happened {lab.trainer.steps:,} times "
        "during pretraining. We threw the copy away: your saved model is unchanged. The "
        "chapters after next do this on purpose."
    )
