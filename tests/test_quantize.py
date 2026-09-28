import numpy as np

from attention.quantize import BLOCK, bfloat16, quantize, shrink
from attention.train import evaluate


def test_bfloat16_keeps_about_three_significant_digits():
    x = np.array([1.0, 3.14159265, -1e-3, 65504.0])
    y = bfloat16(x)
    assert y[0] == 1.0
    assert np.all(np.abs(y - x) <= np.abs(x) / 128)


def test_each_block_is_rounded_to_its_own_scale():
    rows = np.array([np.linspace(-1, 1, BLOCK), np.linspace(-100, 100, BLOCK)])
    q = quantize(rows, bits=4)
    assert len(np.unique(q[0])) <= 15 and len(np.unique(q[1])) <= 15  # -7..7 steps
    assert np.abs(q[0] - rows[0]).max() <= 1 / 7 / 2 + 1e-12  # half a step, in its own scale
    assert np.abs(q - rows).max() < 10
    assert set(np.unique(quantize(rows, bits=2)[0] / (1.0))) <= {-1.0, 0.0, 1.0}


def test_fewer_bits_make_a_smaller_rougher_model(trained):
    held = trained.held
    results = [shrink(trained.model, bits) for bits in (16, 8, 4, 2)]
    sizes = [r.size for r in results]
    errors = [r.error for r in results]
    assert sizes == sorted(sizes, reverse=True)
    assert errors == sorted(errors)
    losses = [evaluate(r.model, *held) for r in results]
    assert abs(losses[0] - evaluate(trained.model, *held)) < 0.01  # bfloat16: barely any change
    assert losses[-1] > losses[0] + 0.1  # 2 bits: a lot worse
    assert results[2].label == "4-bit" and results[0].label == "bfloat16"
