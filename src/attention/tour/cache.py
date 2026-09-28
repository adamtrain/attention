"""Chapter: the KV cache. Remembering keys and values instead of rereading everything."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .. import viz
from ..generate import picks, prompt
from ..model import Cache, Transformer
from ..views import label
from .code import excerpt
from .lab import Lab
from .stage import Frame, Stage, hold

WRITE = 120  # tokens to write when timing
SHOWN = 10  # rows of the cache in the picture


def run(stage: Stage, lab: Lab) -> None:
    c = lab.model.config
    stage.say(
        "There's something wasteful about that writing loop. Every new token means running the "
        "model again, and the obvious way is to run it on [i]everything so far[/i]: to write "
        f"{WRITE} tokens, that's 1 + 2 + … + {WRITE} = {WRITE * (WRITE + 1) // 2:,} positions' "
        "worth of work, when only one new position appears each time."
    )
    stage.say(
        "Look at what the new token actually needs. In each layer it makes its own query, then "
        "compares it with the [b]keys[/b] of every position so far and blends their "
        "[b]values[/b]. And those keys and values never change once they're made: the causal "
        "mask means a position never looks ahead, so nothing written later can change what it "
        "computed. So keep them. That's the [b]KV cache[/b]: every layer's keys and values, "
        "for every position read so far, kept in memory."
    )
    stage.say(
        "What about the queries? A query is only ever used by its own position, once, to look "
        "back. Later positions never look at earlier queries; they look at earlier keys and "
        "values. So each query is made, used and thrown away. That's the query's job in one "
        "sentence, and the key's and value's: a query is a question asked once, keys and values "
        "are what every later token will consult."
    )
    stage.wait("watch the cache fill")

    stage.say(
        "Writing happens in two phases. First the [b]prefill[/b]: the whole prompt goes through "
        "the model in one pass, and every position's keys and values go into the cache. Then "
        "[b]decode[/b]: one token at a time, each step running the model on just the newest "
        "token, which adds one row to the cache. Here's your model's cache as it writes. Each "
        f"square is a vector of {c.head_width} numbers: a key (K) or a value (V) for one of "
        f"the {c.kv_heads} key/value heads in each of the {c.layers} layers:"
    )
    stage.play(lambda: filling(lab, stage.width), fps=3, start="prefill, then decode")
    stage.wait("check it")

    same, with_cache, without = check(lab)
    stage.say(
        "It isn't an approximation. Writing with the cache gives exactly the same "
        f"probabilities as rereading everything (the biggest difference was {same:.0e}, "
        "rounding error). And here's what it saves, on your model, writing "
        f"{WRITE} tokens:"
    )
    stage.show(savings(with_cache, without))
    stage.wait("see what it costs")

    per = 2 * c.layers * c.kv_heads * c.head_width
    stage.say(
        "The price is memory. For every token, your model keeps 2 (a key and a value) × "
        f"{c.layers} layers × {c.kv_heads} heads × {c.head_width} numbers = {per} numbers. "
        f"At its full {c.context} tokens, that's {per * c.context:,} numbers. Tiny. Now a real "
        "model, with each number in 16 bits:"
    )
    stage.show(memory_table(lab))
    stage.say(
        "One long conversation with Llama 3 8B can need gigabytes of cache, on top of the "
        "model's own 16 GB of weights, and a server has to keep one for every conversation "
        "it's working on. That's why grouped-query attention exists: with 8 key/value heads "
        "instead of 32, the cache is four times smaller. It's also why long contexts cost more, "
        "and why some models share keys and values across layers, or squeeze them into fewer "
        "bits."
    )
    stage.wait("see the two phases")

    prefill, decode = phases(lab)
    stage.say(
        f"The two phases feel different, too. On your model, the prefill read {c.context // 2} "
        f"tokens at {prefill:,.0f} tokens a second, all in one pass; decoding wrote at "
        f"{decode:,.0f} tokens a second, one at a time. The prefill can do every position at "
        "once, which is exactly what a GPU is good at; decoding can't, because each token "
        "needs the one before. That's why a chatbot pauses before its first word (the time to "
        "first token) and then streams the rest steadily."
    )
    stage.say(
        "And when an API offers [b]prompt caching[/b], this is what it keeps: the KV cache for "
        "the start of your prompt, so that a later request beginning with exactly the same "
        "tokens can skip that part of the prefill. It has to be exactly the same, from the very "
        "first token, because every position's keys and values depend on everything before it."
    )
    stage.say(
        f"The cache also makes the [b]context window[/b] concrete. Your model's cache has room "
        f"for {c.context} positions, because that's as far as it was trained to read; when it's "
        "full, writing has to stop, or the oldest tokens have to go. Here's the part of your "
        "model's code that adds to the cache:"
    )
    stage.show(excerpt(Cache.add))
    stage.show(excerpt(Transformer.block, "if cache is not None", "scores ="))


# ── Filling up ────────────────────────────────────────────────────────────────


def filling(lab: Lab, width: int, start: int = 5, steps: int = 6) -> Iterator[Frame]:
    tok, c = lab.tokenizer, lab.model.config
    ids = prompt(tok, lab.corpus.example)[: start + 1]
    cache = Cache()
    lab.model.forward(np.array([ids]), cache)
    yield hold(
        cache_view(
            lab,
            ids,
            cache.length,
            len(ids),
            "prefill",
            f"the prompt, {len(ids)} tokens, in one pass",
        ),
        2.5,
    )
    rng = np.random.default_rng([lab.seed, 14])
    written = list(ids)
    for pick in picks(lab.model, ids, rng, 0.8, limit=steps):
        if pick.done:
            break
        written.append(pick.token)
        yield hold(
            cache_view(
                lab,
                written,
                len(written) - 1,
                len(written) - 1,
                "decode",
                f"write {label(tok, pick.token)}: one new query, used and thrown away; one new row, kept",
            ),
            1.4,
        )
    yield hold(
        cache_view(
            lab,
            written,
            len(written),
            len(written),
            "decode",
            f"{len(written)} rows kept, {2 * c.layers * c.kv_heads * c.head_width * len(written):,} numbers",
        ),
        0.5,
    )


def cache_view(lab: Lab, ids: list[int], cached: int, newest: int, phase: str, what: str) -> Group:
    tok, c = lab.tokenizer, lab.model.config
    shown = ids[-SHOWN:]
    offset = len(ids) - len(shown)
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, justify="right")
    for _ in range(c.layers):
        grid.add_column(no_wrap=True)
    grid.add_row("", *(Text(f"layer {i + 1}", style=viz.FAINT) for i in range(c.layers)))
    grid.add_row(
        "",
        *(
            Text(" ".join(f"K{h + 1} V{h + 1}" for h in range(c.kv_heads)), style=viz.FAINT)
            for _ in range(c.layers)
        ),
    )
    for i, token in enumerate(shown):
        pos = offset + i
        if pos >= cached and pos != newest:
            continue  # not in the cache yet
        fresh = pos == newest or (phase == "prefill" and pos < cached)
        color = viz.AMBER if fresh else viz.ACCENT
        cells = Text(" ".join(" ■ " + " ■ " for _ in range(c.kv_heads)), style=color)
        grid.add_row(Text(label(tok, token), style="bold" if fresh else ""), *([cells] * c.layers))
    head = Text.assemble((phase, f"bold {viz.AMBER}"), ("   ", ""), (what, viz.FAINT))
    return Group(head, Text(""), grid)


# ── Checking and timing ───────────────────────────────────────────────────────


def check(lab: Lab) -> tuple[float, float, float]:
    """How far apart cached and uncached answers are, and how long each took to write."""
    model, ids = lab.model, prompt(lab.tokenizer)

    def write() -> list:  # at temperature 1: the model's own probabilities, to compare
        return list(picks(model, ids, np.random.default_rng(1), 1.0, limit=WRITE, stop=()))

    start = time.perf_counter()
    steps = write()
    with_cache = time.perf_counter() - start
    worst, start = 0.0, time.perf_counter()
    for pick in steps:
        again = model.forward(np.array([pick.context])).probs[0, -1]
        worst = max(worst, float(np.abs(again - pick.probs).max()))
    return worst, with_cache, time.perf_counter() - start


def savings(with_cache: float, without: float) -> Table:
    n = WRITE
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=viz.FAINT, no_wrap=True)
    grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, justify="right")
    grid.add_row("", Text("positions computed", style=viz.FAINT), "", Text("time", style=viz.FAINT))
    grid.add_row(
        "rereading everything",
        Text(f"{n * (n + 1) // 2:,}", style="bold"),
        viz.bar(1.0, 24, viz.RED),
        f"{without:.2f} s",
    )
    grid.add_row(
        "with a KV cache",
        Text(f"{n:,}", style=f"bold {viz.GREEN}"),
        viz.bar(2 / (n + 1), 24, viz.GREEN),
        f"{with_cache:.2f} s",
    )
    return grid


def memory_table(lab: Lab) -> Table:
    c = lab.model.config
    rows = [
        ("yours", c.layers, c.kv_heads, c.head_width, c.context),
        ("Llama 3 8B", 32, 8, 128, 8_192),
        ("…with a long chat", 32, 8, 128, 131_072),
        ("…without GQA", 32, 32, 128, 131_072),
    ]
    grid = Table.grid(padding=(0, 2))
    for justify in ("left", "right", "right", "right", "right", "right"):
        grid.add_column(justify=justify, no_wrap=True)
    grid.add_row(
        *(
            Text(h, style=viz.FAINT)
            for h in ("", "layers", "K/V heads", "head size", "tokens", "cache (16-bit)")
        )
    )
    for name, layers, heads, width, tokens in rows:
        size = 2 * layers * heads * width * tokens * 2
        grid.add_row(
            Text(name, style="bold"),
            str(layers),
            str(heads),
            str(width),
            f"{tokens:,}",
            Text(human(size), style=f"bold {viz.AMBER}"),
        )
    return grid


def human(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


def phases(lab: Lab) -> tuple[float, float]:
    """Tokens per second: reading a prompt in one pass, and writing one token at a time."""
    model = lab.model
    half = model.config.context // 2
    ids = [int(t) for t in lab.data.val[:half]]
    start = time.perf_counter()
    for _ in range(5):
        model.forward(np.array([ids]), Cache())
    prefill = 5 * half / (time.perf_counter() - start)
    start = time.perf_counter()
    written = list(picks(model, ids[:4], np.random.default_rng(2), 0.8, limit=half, stop=()))
    decode = len(written) / (time.perf_counter() - start)
    return prefill, decode
