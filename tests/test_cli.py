import json
import re

import numpy as np
from typer.testing import CliRunner

from attention import tour
from attention.cli import app
from attention.corpus import VOCAB, built_in
from attention.tour.lab import Budget, Lab

runner = CliRunner()
TRAIN = ["train", "--corpus", "fables", "--seed", "8", "--steps", "20", "--fast"]


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("attention ")


def test_commands_need_a_model_first():
    for command in (["generate"], ["info"], ["explain", "The"], ["chat"], ["tokenize", "hi"]):
        result = runner.invoke(app, command)
        assert result.exit_code == 1
        assert "no model" in plain(result.output).lower()


def test_train_then_use_the_model():
    result = runner.invoke(app, TRAIN)
    assert result.exit_code == 0, result.output
    assert "It named itself" in plain(result.output)

    result = runner.invoke(app, ["generate", "-n", "2", "--tokens", "8", "--plain", "The Fox"])
    assert result.exit_code == 0, result.output
    # Two samples, each carrying on from the prompt. (Not split on blank lines: a sample can
    # have one of its own, between a title and its story.)
    assert result.output.lstrip().startswith("The Fox") and result.output.count("The Fox") >= 2

    result = runner.invoke(app, ["generate", "-n", "1", "-k", "5", "-p", "0.5", "--tokens", "6"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["explain", "The Fox and the", "--teach", "Crow"])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "which tokens mattered" in out and "one step of learning" in out and "logit lens" in out

    result = runner.invoke(app, ["tokenize", "The Fox and the Grapes"])
    assert result.exit_code == 0
    assert "tokens" in plain(result.output)

    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    out = plain(result.output)
    assert "--seed 8" in out and "num_key_value_heads" in out
    assert "model.layers.0.self_attn.k_proj.weight" in out

    result = runner.invoke(app, ["chat", "--steps", "3"], input="Tell me a fable.\n\n")
    assert result.exit_code == 0, result.output
    assert "it" in plain(result.output)


def test_a_model_with_a_different_number_of_tokens_can_still_chat():
    lab = Lab.create(built_in("fables"), 5, budget=Budget.tiny(vocab=512))
    assert len(lab.tokenizer) != VOCAB  # saved by an older version, say
    lab.train_quietly()
    result = runner.invoke(app, ["chat", "--steps", "2"], input="Tell me a fable.\n\n")
    assert result.exit_code == 0, result.output


def test_bad_input_is_explained(model_home):
    result = runner.invoke(app, ["train", "--corpus", "nope"])
    assert result.exit_code == 1
    assert "No corpus" in plain(result.output)
    runner.invoke(app, TRAIN)
    result = runner.invoke(app, ["generate", "word " * 200])
    assert result.exit_code == 1
    assert "reads at most" in plain(result.output)


def test_models_from_the_old_version_are_explained(model_home):
    model_home.mkdir(parents=True)
    np.savez(model_home / "model.npz", meta=np.array(json.dumps({"format": 1, "card": {}})))
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 1
    assert "older version" in plain(result.output)


def test_the_tour_runs_without_a_keyboard(monkeypatch):
    real = tour.run
    monkeypatch.setattr(tour, "run", lambda stage, **kw: real(stage, budget=Budget.tiny(), **kw))
    result = runner.invoke(app, ["tour", "--corpus", "fairytales", "--seed", "2"])
    assert result.exit_code == 0, result.output
    assert "Thanks for taking the tour" in plain(result.output)


def test_clean_with_nothing_saved():
    result = runner.invoke(app, ["clean"])
    assert result.exit_code == 0
    assert "Nothing to clean up" in plain(result.output)


def test_clean_asks_first_then_deletes_the_models_and_their_folder(model_home):
    runner.invoke(app, TRAIN)
    runner.invoke(app, ["chat", "--steps", "2"], input="\n")
    path = model_home / "model.npz"
    assert path.exists() and (model_home / "model.chat.npz").exists()

    result = runner.invoke(app, ["clean"], input="n\n")
    assert path.exists()
    assert "Nothing deleted" in plain(result.output)

    result = runner.invoke(app, ["clean"], input="y\n")
    assert result.exit_code == 0
    assert "Deleted 2 files" in plain(result.output)
    assert not model_home.exists()


def test_clean_leaves_everything_else_alone(model_home):
    runner.invoke(app, TRAIN)
    (model_home / "notes.txt").write_text("mine")
    (model_home / "model.npz.tmp.npz").write_bytes(b"half-written")
    result = runner.invoke(app, ["clean", "--yes"])
    assert result.exit_code == 0
    assert sorted(p.name for p in model_home.iterdir()) == ["notes.txt"]
    assert "Left" in plain(result.output)


def test_clean_only_deletes_attention_models(tmp_path):
    precious = tmp_path / "precious.npz"
    np.savez(precious, x=np.zeros(3))
    result = runner.invoke(app, ["clean", "--yes", "--model", str(precious)])
    assert result.exit_code == 1
    assert precious.exists()

    elsewhere = tmp_path / "mine.npz"
    runner.invoke(app, [*TRAIN, "--model", str(elsewhere)])
    assert elsewhere.exists()
    result = runner.invoke(app, ["clean", "--yes", "--model", str(elsewhere)])
    assert result.exit_code == 0
    assert not elsewhere.exists()
