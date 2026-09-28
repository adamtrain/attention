from collections.abc import Sequence

import numpy as np
import pytest

from attention.model import (
    Cache,
    Config,
    Transformer,
    angles,
    causal_mask,
    cross_entropy,
    groups,
    part,
    rmsnorm,
    rmsnorm_backward,
    rope,
    silu,
    silu_backward,
    softmax,
    softmax_backward,
)

# Two layers, grouped-query attention (4 query heads sharing 2 key/value heads), RoPE: small
# enough to check by brute force, with every part of the real model in it.
TINY = Config(
    vocab=11, context=9, width=16, layers=2, heads=4, kv_heads=2, hidden=12, rope_base=100
)


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
    tokens = rng.integers(0, TINY.vocab, (3, 7))
    targets = rng.integers(0, TINY.vocab, (3, 7))
    targets[1, 4:] = -1  # positions that don't count
    return tokens, targets


def test_untrained_model_finds_every_token_about_equally_likely():
    rng = np.random.default_rng(0)
    model = Transformer.create(Config(vocab=50, context=16), rng)
    tokens = rng.integers(0, 50, (4, 12))
    probs = model.forward(tokens).probs
    assert probs.shape == (4, 12, 50)
    np.testing.assert_allclose(probs.sum(-1), 1.0)
    assert np.abs(np.log(probs * 50)).max() < np.log(2)  # all within a factor of 2 of 1 in 50


def test_attention_never_looks_ahead():
    rng = np.random.default_rng(1)
    model = Transformer.create(TINY, rng)
    tokens, _ = batch(rng)
    weights = model.forward(tokens).weights  # (layers, batch, heads, time, time)
    assert weights.shape == (2, 3, 4, 7, 7)
    assert np.all(weights[..., ~causal_mask(7)] == 0)
    np.testing.assert_allclose(weights.sum(-1), 1.0)
    # Changing a later token can't change an earlier prediction.
    changed = tokens.copy()
    changed[:, -1] = (changed[:, -1] + 1) % TINY.vocab
    np.testing.assert_allclose(
        model.forward(tokens).logits[:, :-1], model.forward(changed).logits[:, :-1]
    )


@pytest.mark.parametrize("name", list(Transformer.create(TINY, np.random.default_rng(0)).params))
def test_backprop_matches_finite_differences(name):
    rng = np.random.default_rng(2)
    model = Transformer.create(TINY, rng)
    for w in model.params.values():  # make everything matter, including the small embeddings
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
    if name == "embed":
        picks += [(int(tokens[0, 0]), 0)]  # make sure a row that was read is checked
    expected = numeric_grad(loss, param, picks)
    actual = np.array([grads[name][i] for i in picks])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_the_input_gradient_is_what_reaches_the_embeddings():
    rng = np.random.default_rng(3)
    model = Transformer.create(TINY, rng)
    tokens, targets = batch(rng)
    trace = model.forward(tokens)
    _, dlogits = cross_entropy(trace.logits, targets)
    grads, dx = model.backward(trace, dlogits)
    # The embedding table's gradient is the output side's plus each token's input gradient.
    out_side = dlogits.reshape(-1, TINY.vocab).T @ trace.f_in.reshape(-1, TINY.width)
    in_side = np.zeros_like(out_side)
    np.add.at(in_side, tokens, dx)
    np.testing.assert_allclose(grads["embed"], out_side + in_side)


def test_the_kv_cache_gives_exactly_the_same_answers():
    rng = np.random.default_rng(4)
    model = Transformer.create(TINY, rng)
    tokens = rng.integers(0, TINY.vocab, (1, 8))
    full = model.forward(tokens).logits[0]
    cache = Cache()
    parts = [model.forward(tokens[:, :3], cache).logits[0]]  # prefill: the prompt in one go
    for t in range(3, 8):  # then one token at a time, reading the rest from the cache
        parts.append(model.forward(tokens[:, t : t + 1], cache).logits[0])
    np.testing.assert_allclose(np.concatenate(parts), full, atol=1e-12)
    assert cache.length == 8
    # Two numbers (a key and a value) per layer, key/value head, position and head width.
    assert cache.size == 2 * TINY.layers * TINY.kv_heads * 8 * TINY.head_width


def test_the_context_window_is_a_hard_limit():
    model = Transformer.create(TINY, np.random.default_rng(5))
    with pytest.raises(ValueError, match="at most 9 tokens"):
        model.forward(np.zeros((1, 10), dtype=int))
    cache = Cache()
    model.forward(np.zeros((1, 9), dtype=int), cache)
    with pytest.raises(ValueError):
        model.forward(np.zeros((1, 1), dtype=int), cache)


def test_rope_makes_matches_depend_on_distance_not_position():
    rng = np.random.default_rng(6)
    q, k = rng.normal(size=8), rng.normal(size=8)

    def score(at_q: int, at_k: int) -> float:
        turned_q = rope(q, angles(np.array([at_q]), 8, 100.0)[0])
        turned_k = rope(k, angles(np.array([at_k]), 8, 100.0)[0])
        return float(turned_q @ turned_k)

    assert score(5, 2) == pytest.approx(score(9, 6))  # three apart either way
    assert score(5, 2) != pytest.approx(score(5, 4))
    # Turning back by the same angle undoes it.
    turn = angles(np.arange(4), 8, 100.0)
    x = rng.normal(size=(4, 8))
    np.testing.assert_allclose(rope(rope(x, turn), -turn), x)


def test_the_logit_lens_ends_at_the_real_answer():
    rng = np.random.default_rng(7)
    model = Transformer.create(TINY, rng)
    tokens, _ = batch(rng)
    trace = model.forward(tokens)
    lens = model.lens(trace)
    assert lens.shape == (TINY.layers + 1, 3, 7, TINY.vocab)
    np.testing.assert_allclose(lens[-1], trace.probs)
    assert model.lens(trace, mid=True).shape[0] == 2 * TINY.layers + 1


def test_the_pieces_backward():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 5))
    up = rng.normal(size=(2, 5))
    picks = [(0, 1), (1, 3), (1, 0)]

    def f_soft() -> float:
        return float((softmax(x) * up).sum())

    actual = softmax_backward(up, softmax(x))
    np.testing.assert_allclose(
        [actual[i] for i in picks], numeric_grad(f_soft, x, picks), rtol=1e-6
    )

    gain = rng.normal(size=5)

    def f_norm() -> float:
        return float((rmsnorm(x, gain)[0] * up).sum())

    _, inv = rmsnorm(x, gain)
    dx, _ = rmsnorm_backward(up, x, inv, gain)
    np.testing.assert_allclose([dx[i] for i in picks], numeric_grad(f_norm, x, picks), rtol=1e-6)

    def f_silu() -> float:
        return float((silu(x) * up).sum())

    actual = silu_backward(up, x)
    np.testing.assert_allclose(
        [actual[i] for i in picks], numeric_grad(f_silu, x, picks), rtol=1e-6
    )


def test_cross_entropy_gradient_is_probs_minus_one_hot():
    logits = np.array([[[2.0, 0.0, -1.0]]])
    loss, grad = cross_entropy(logits, np.array([[0]]))
    p = softmax(logits)[0, 0]
    assert loss == pytest.approx(-np.log(p[0]))
    np.testing.assert_allclose(grad[0, 0], p - [1, 0, 0])


def test_every_weight_has_a_name_people_and_checkpoints_use():
    model = Transformer.create(TINY, np.random.default_rng(9))
    names = {part(n).hugging_face for n in model.params}
    assert "model.layers.1.self_attn.k_proj.weight" in names
    assert "model.embed_tokens.weight" in names
    assert {part(n).group for n in model.params} == set(groups(TINY))
    assert TINY.hugging_face()["num_key_value_heads"] == 2
    assert model.params["layers.0.attn.key"].shape == (16, 2 * 4)  # 2 key heads of width 4
