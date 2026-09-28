import numpy as np

from attention.induction import LENGTH, Race, habits, sequences


def test_each_row_repeats_a_stretch_and_only_the_repeat_is_graded():
    batch = sequences(np.random.default_rng(0), 50, 20)
    assert batch.inputs.shape == batch.targets.shape == (50, LENGTH)
    assert (batch.inputs[:, 0] == 0).all()
    for row in range(50):
        graded = np.flatnonzero(batch.targets[row] >= 0)
        assert 5 <= len(graded) <= 9
        gap = batch.gaps[row]
        for i in graded:  # the answer is the token after the same token, gap places back
            assert batch.inputs[row, i] == batch.inputs[row, i - gap]
            assert batch.targets[row, i] == batch.inputs[row, i - gap + 1]


def test_the_race_trains_both_models_on_the_same_batches():
    race = Race(vocab=12, rng=np.random.default_rng(1), steps=30)
    while not race.done:
        race.step()
    one, two = race.learners
    assert one.model.config.layers == 1 and two.model.config.layers == 2
    assert [s for s, _ in one.history] == [s for s, _ in two.history] == [0, 25, 30]
    found = habits(two.model, sequences(np.random.default_rng(2), 8, 12))
    assert found.back.shape == (3, 2, 2) and found.past.shape == (4, 2, 2)
    assert found.start.shape == (2, 2)
    assert np.all((found.back >= 0) & (found.back <= 1))
