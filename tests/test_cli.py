import re

import numpy as np
from typer.testing import CliRunner

from attention.cli import app

runner = CliRunner()


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("attention ")


def test_commands_need_a_model_first():
    for command in (["generate"], ["info"], ["explain", "ab"]):
        result = runner.invoke(app, command)
        assert result.exit_code == 1
        assert "no model" in plain(result.output).lower()


def test_train_then_use_the_model():
    result = runner.invoke(app, ["train", "--corpus", "dinosaurs", "--seed", "8", "--fast"])
    assert result.exit_code == 0, result.output
    assert "It named itself" in plain(result.output)

    result = runner.invoke(app, ["generate", "-n", "5", "--plain", "stego"])
    assert result.exit_code == 0
    words = result.output.split()
    assert len(words) == 5 and all(w.lower().startswith("stego") for w in words)

    result = runner.invoke(app, ["generate", "-n", "3", "--top-p", "0.5", "--plain"])
    assert result.exit_code == 0 and len(result.output.split()) == 3

    result = runner.invoke(app, ["explain", "stegosa", "--teach", "e"])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "which letters mattered" in out and "taught" in out

    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "--seed 8" in plain(result.output)


def test_bad_input_is_explained():
    runner.invoke(app, ["train", "--corpus", "names", "--seed", "1", "--fast"])
    result = runner.invoke(app, ["explain", "b3"])
    assert result.exit_code == 1
    assert "doesn't know" in plain(result.output)
    result = runner.invoke(app, ["train", "--corpus", "nope"])
    assert result.exit_code == 1
    assert "No corpus" in plain(result.output)


def test_the_tour_runs_without_a_keyboard():
    result = runner.invoke(app, ["tour", "--corpus", "towns", "--seed", "2"])
    assert result.exit_code == 0, result.output
    assert "Thanks for taking the tour" in plain(result.output)


def test_clean_with_nothing_saved():
    result = runner.invoke(app, ["clean"])
    assert result.exit_code == 0
    assert "Nothing to clean up" in plain(result.output)


def test_clean_asks_first_then_deletes_the_model_and_its_folder(model_home):
    runner.invoke(app, ["train", "--corpus", "towns", "--seed", "2", "--fast"])
    path = model_home / "model.npz"
    assert path.exists()

    result = runner.invoke(app, ["clean"], input="n\n")
    assert path.exists()
    assert "Nothing deleted" in plain(result.output)

    result = runner.invoke(app, ["clean"], input="y\n")
    assert result.exit_code == 0
    assert "Deleted 1 file" in plain(result.output)
    assert not model_home.exists()


def test_clean_leaves_everything_else_alone(model_home):
    runner.invoke(app, ["train", "--corpus", "names", "--seed", "4", "--fast"])
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
    runner.invoke(
        app, ["train", "--corpus", "names", "--seed", "3", "--fast", "--model", str(elsewhere)]
    )
    assert elsewhere.exists()
    result = runner.invoke(app, ["clean", "--yes", "--model", str(elsewhere)])
    assert result.exit_code == 0
    assert not elsewhere.exists()
