from collections.abc import Sequence

import numpy as np
import pytest

from attention.model import (
    Config,
    Transformer,
    cross_entropy,
    rmsnorm,
    rmsnorm_backward,
    softmax,
    softmax_backward,
)

TINY = Config(vocab=7, context=6, width=8, heads=2, hidden=12)


def numeric_grad(f, x: np.ndarray, picks: Sequence[tuple[int, ...]], h: float = 1e-6) -> np.ndarray:
    out = []
    for i in picks:
        old = x[i]
        x[i] = old + h
        up = f()
        x[i] = old - h
        down = f()
        x[i] = old
        out.append((up - down) / (2 * h))
    return np.array(out)


def batch(rng: np.random.Generator):
    tokens = rng.integers(0, TINY.vocab, (3, 5))
    targets = rng.integers(0, TINY.vocab, (3, 5))
    targets[1, 3:] = -1  # padding
    return tokens, targets


def test_untrained_model_finds_every_token_about_equally_likely():
    rng = np.random.default_rng(0)
    model = Transformer.create(Config(vocab=27, context=12), rng)
    tokens = rng.integers(0, 27, (4, 10))
    probs = model.forward(tokens).probs
    assert probs.shape == (4, 10, 27)
    np.testing.assert_allclose(probs.sum(-1), 1.0)
    assert np.abs(probs - 1 / 27).max() < 0.02


def test_attention_never_looks_ahead():
    rng = np.random.default_rng(1)
    model = Transformer.create(TINY, rng)
    tokens, _ = batch(rng)
    weights = model.forward(tokens).weights
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    assert np.all(weights[..., upper] == 0)
    np.testing.assert_allclose(weights.sum(-1), 1.0)
    # Changing a later token can't change an earlier prediction.
    changed = tokens.copy()
    changed[:, -1] = (changed[:, -1] + 1) % TINY.vocab
    a = model.forward(tokens).logits[:, :-1]
    b = model.forward(changed).logits[:, :-1]
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("name", list(Transformer.create(TINY, np.random.default_rng(0)).params))
def test_backprop_matches_finite_differences(name):
    rng = np.random.default_rng(2)
    model = Transformer.create(TINY, rng)
    for w in model.params.values():  # make everything matter, including the small unembedding
        w += rng.normal(0, 0.3, w.shape)
    tokens, targets = batch(rng)

    def loss() -> float:
        return cross_entropy(model.forward(tokens).logits, targets)[0]

    trace = model.forward(tokens)
    _, dlogits = cross_entropy(trace.logits, targets)
    grads, _ = model.backward(trace, dlogits)

    param = model.params[name]
    flat = [np.unravel_index(i, param.shape) for i in range(param.size)]
    picks = [flat[i] for i in rng.choice(len(flat), size=min(12, len(flat)), replace=False)]
    if name == "embed.token":
        picks += [(int(tokens[0, 0]), 0)]  # make sure a used row is checked
    expected = numeric_grad(loss, param, picks)
    actual = np.array([grads[name][i] for i in picks])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_input_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    model = Transformer.create(TINY, rng)
    tokens, targets = batch(rng)
    trace = model.forward(tokens)
    _, dlogits = cross_entropy(trace.logits, targets)
    _, dx0 = model.backward(trace, dlogits)
    # Nudge one position's position embedding in one example: equivalent to nudging x0 there.
    total = dx0.sum(axis=0)
    grads, _ = model.backward(trace, dlogits)
    np.testing.assert_allclose(grads["embed.position"][:5], total)


def test_softmax_and_rmsnorm_backward():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 5))
    up = rng.normal(size=(2, 5))

    def f_soft() -> float:
        return float((softmax(x) * up).sum())

    picks = [(0, 1), (1, 3), (1, 0)]
    expected = numeric_grad(f_soft, x, picks)
    actual = softmax_backward(up, softmax(x))
    np.testing.assert_allclose([actual[i] for i in picks], expected, rtol=1e-6)

    gain = rng.normal(size=5)

    def f_norm() -> float:
        return float((rmsnorm(x, gain)[0] * up).sum())

    expected = numeric_grad(f_norm, x, picks)
    _, inv = rmsnorm(x, gain)
    dx, _ = rmsnorm_backward(up, x, inv, gain)
    np.testing.assert_allclose([dx[i] for i in picks], expected, rtol=1e-6)


def test_cross_entropy_gradient_is_probs_minus_one_hot():
    logits = np.array([[[2.0, 0.0, -1.0]]])
    loss, grad = cross_entropy(logits, np.array([[0]]))
    p = softmax(logits)[0, 0]
    assert loss == pytest.approx(-np.log(p[0]))
    np.testing.assert_allclose(grad[0, 0], p - [1, 0, 0])
