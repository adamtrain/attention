import numpy as np
from rich.text import Text

from attention import viz


def test_bars_have_half_cell_precision():
    assert viz.bar(0.5, 10, "red").plain == "━━━━━─────"
    assert viz.bar(0.55, 10, "red").plain == "━━━━━╸────"
    assert viz.bar(2.0, 4, "red").plain == "━━━━"
    assert viz.signed_bar(-1.0, 1.0, 3).plain == "━━━┃   "


def test_heatmaps_fold_two_rows_into_each_line():
    lines = viz.signed_blocks(np.arange(15.0).reshape(5, 3) - 7)
    assert len(lines) == 3
    assert all(line.plain == "▀▀▀" for line in lines)


def test_plots_are_the_size_asked_for():
    plot = viz.Plot(20, 5, x_max=10, lo=0, hi=1)
    plot.line([(0, 1.0), (10, 0.0)], viz.ACCENT)
    plot.guide(0.5, viz.FAINT, "half")
    lines = plot.render()
    assert len(lines) == 5
    assert {len(line.plain) for line in lines} == {20 + 5}
    assert "half" in "".join(line.plain for line in lines)


def test_colors_mix_and_ramps_stay_in_range():
    assert viz.mix("#000000", "#ffffff", 0.5) == "#808080"
    assert viz.SIGNED.color(-5) == viz.SIGNED.color(0)
    assert viz.ink_for("#ffffff") == viz.DARK


def test_canvas_writes_where_its_told():
    canvas = viz.Canvas(6, 2)
    canvas.put(2, 1, "hi")
    canvas.put(5, 0, Text("xyz"))
    assert [line.plain for line in canvas.lines()] == ["     x", "  hi"]


def test_the_wordmark_spells_attention():
    assert len(viz.wordmark().plain.splitlines()) == 2
