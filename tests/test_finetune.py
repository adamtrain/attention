import numpy as np
import pytest

from attention.corpus import built_in
from attention.finetune import (
    AdapterTrainer,
    LoRA,
    chat_batch,
    chat_ids,
    drift,
    exchanges,
    learn_from,
    niche_trainer,
    share,
)
from attention.generate import samples
from attention.model import Config, Transformer, cross_entropy
from attention.tokenizer import ASSISTANT, END, USER


def test_fine_tuning_on_a_niche_shifts_what_it_writes(trained):
    niche = built_in("fables").niche
    tok = trained.data.tokenizer
    rng = np.random.default_rng(0)
    before = share(samples(trained.model, tok, rng, 20, temperature=0.8, limit=12), niche)
    tuner = niche_trainer(trained.model, trained.data, niche, np.random.default_rng(6), steps=60)
    assert all(niche.has(d) for d in tuner.data.train_docs)
    assert drift(tuner.model, trained.model) == 0
    while not tuner.done:
        tuner.step()
    after = share(samples(tuner.model, tok, rng, 20, temperature=0.8, limit=12), niche)
    assert after > before + 0.2
    assert 0 < drift(tuner.model, trained.model) < 0.5


def test_lora_starts_as_the_same_model_and_learns_only_its_own_grids():
    rng = np.random.default_rng(0)
    base = Transformer.create(Config(vocab=9, context=8, width=16, layers=2, hidden=32), rng)
    lora = LoRA.create(base, rng, rank=2)
    tokens = rng.integers(0, 9, (2, 6))
    np.testing.assert_allclose(lora.merged().forward(tokens).logits, base.forward(tokens).logits)
    assert lora.size < base.size / 4
    # Check the chain rule for A and B against finite differences.
    for w in lora.b.values():
        w += rng.normal(0, 0.1, w.shape)
    targets = rng.integers(0, 9, (2, 6))

    def loss() -> float:
        return cross_entropy(lora.merged().forward(tokens).logits, targets)[0]

    merged = lora.merged()
    trace = merged.forward(tokens)
    grads, _ = merged.backward(trace, cross_entropy(trace.logits, targets)[1])
    adapter = lora.gradients(grads)
    for name, w in lora.params.items():
        i = (1, 1)
        old = w[i]
        w[i] = old + 1e-6
        up = loss()
        w[i] = old - 1e-6
        down = loss()
        w[i] = old
        assert adapter[name][i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-9)


def test_lora_training_leaves_the_base_model_untouched(trained):
    base = {k: v.copy() for k, v in trained.model.params.items()}
    lora = LoRA.create(trained.model, np.random.default_rng(1))
    data = trained.data

    def batches(rng):
        return data.windows(rng, 8, 32)

    tuner = AdapterTrainer(lora, batches, np.random.default_rng(2), steps=5, lr=0.003)
    while not tuner.done:
        tuner.step()
    assert len(tuner.losses) == 5
    for k, v in trained.model.params.items():
        np.testing.assert_array_equal(v, base[k])
    assert drift(tuner.model, trained.model) > 0


def test_conversations_follow_the_chat_template(trained):
    tok = trained.data.tokenizer
    ids = chat_ids(tok, "Tell me a fable.", "A Fox once...")
    assert ids[:2] == [END, USER] and ASSISTANT in ids and ids[-1] == END
    examples = exchanges(built_in("fables"), trained.data.train_docs[:5])
    assert examples[0].request.startswith("Tell me the fable of ")
    inputs, targets = chat_batch(tok, examples, np.random.default_rng(0), 3, 64)
    for row in range(3):
        at = list(inputs[row]).index(ASSISTANT)
        assert (targets[row, :at] == -1).all()  # the request isn't graded
        assert targets[row, at] >= 0  # the reply is


def test_plays_turn_into_lines_and_replies():
    corpus = built_in("shakespeare")
    first = exchanges(corpus, corpus.documents[:1])[0]
    assert first.request == "Gregory, on my word, we'll not carry coals."
    assert first.reply.startswith("GREGORY:\n")


def test_feedback_makes_liked_texts_likelier_on_a_copy(trained):
    tok = trained.data.tokenizer
    texts = samples(trained.model, tok, np.random.default_rng(1), 4, temperature=0.8, limit=16)
    original = {k: v.copy() for k, v in trained.model.params.items()}
    result = learn_from(trained.model, tok, texts[:2], texts[2:])
    for liked in texts[:2]:
        assert result.change(liked) > 1
    for name, w in trained.model.params.items():
        np.testing.assert_array_equal(w, original[name])
