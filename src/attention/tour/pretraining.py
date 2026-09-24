"""Chapter: pretraining, live."""

from __future__ import annotations

from pathlib import Path

from rich.table import Table
from rich.text import Text

from .. import viz
from ..dashboard import Watch
from ..dashboard import run as train_live
from .lab import Lab
from .stage import Stage

SECONDS = 28


def run(stage: Stage, lab: Lab) -> None:
    tr, c = lab.trainer, lab.corpus
    stage.say(
        f"Now do that {tr.steps} times. Each step: grab {tr.batch_size} random {c.plural}, run "
        "them forward, measure the loss, backpropagate, and nudge every weight downhill. That's "
        "[b]pretraining[/b]: the same first stage every large language model goes through."
    )
    stage.say(
        f"A tenth of the {c.plural} are held back, and the model never trains on them. The "
        "amber line is its loss on those, to check it's learning patterns rather than "
        "memorizing. The dashed lines are what you'd score with no neural network at all, "
        "just by counting letters, or counting which letter follows which. Here's your "
        "model before its first step:"
    )

    watch = Watch(tr, lab.baselines, c, lab.seed, lab.rng)
    if stage.animate:
        with stage.live() as live:
            train_live(
                watch,
                lambda frame: live.update(stage.pad(frame), refresh=True),
                stage.width,
                SECONDS / stage.speed,
                stage.pressed,
                ready=lambda frame: stage.ready(live, frame, "start training"),
            )
    else:
        train_live(watch, lambda frame: None, stage.width, 0, lambda _: False)
        stage.show(watch.render(stage.width))
    lab.seconds = watch.compute
    path = lab.finish()
    stage.show(summary(lab, path))
    stage.say(
        "You just pretrained a language model. The big ones do exactly this, on trillions of "
        "tokens instead of a few thousand, for months, on thousands of GPUs."
    )


def summary(lab: Lab, path: Path) -> Table:
    tr, b = lab.trainer, lab.baselines
    first_val = tr.val_losses[0][1]
    last_val = tr.val_losses[-1][1]
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column()
    words = tr.step_number * tr.batch_size
    grid.add_row(
        Text("✓", style=f"bold {viz.GREEN}"),
        Text.assemble(
            (f"{tr.step_number} steps", "bold"), (f" · {words:,} {lab.corpus.plural} read · ", viz.FAINT),
            (f"{lab.seconds:.1f} seconds", "bold"), (" of actual arithmetic", viz.FAINT),
        ),
    )  # fmt: skip
    grid.add_row(
        Text("↘", style=f"bold {viz.GREEN}"),
        Text.assemble(
            ("held-back loss ", viz.FAINT), (f"{first_val:.2f}", "bold"), (" → ", viz.FAINT),
            (f"{last_val:.2f}", f"bold {viz.GREEN}"),
            (f"   (counting letter pairs: {b.pairs:.2f})", viz.FAINT),
        ),
    )  # fmt: skip
    grid.add_row(
        Text("◆", style=f"bold {viz.PURPLE}"),
        Text.assemble(("It named itself ", viz.FAINT), (lab.name, f"bold {viz.PURPLE}")),
    )
    grid.add_row(
        Text("↓", style=f"bold {viz.ACCENT}"),
        Text.assemble(("Saved to ", viz.FAINT), (tidy(path), "bold")),
    )
    return grid


def tidy(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)
