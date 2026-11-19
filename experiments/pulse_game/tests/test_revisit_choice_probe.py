"""Verify disjoint calibration and the physically paired probe controls."""

import numpy as np
import torch

from experiments.pulse_game import revisit_choice_probe as probe, self_param_writer as fw


def test_balanced_worlds_record_native_action_with_no_taste_or_swap_leak():
    cal = probe.worlds(20261124, 731, 24, calibration=True)
    test = probe.worlds(20261124, 731, 24, calibration=False)
    assert not torch.equal(cal.left, test.left)
    assert not torch.equal(cal.draws, test.draws)
    graph = fw.spawn(731, 8)
    raw = probe.live(graph, test, 4)
    activity = raw["hidden"].reshape(24, 4, len(probe.PHASES), graph.width)
    np.testing.assert_array_equal(activity[:, :1, 0].repeat(4, axis=1), activity[:, :, 0])
    np.testing.assert_array_equal(activity[:, ::2, :4], activity[:, 1::2, :4])
    true_first, true_return = raw["targets"][:, 1:].T
    np.testing.assert_array_equal(raw["targets"][:, 0], true_first * raw["first_side"])
    np.testing.assert_array_equal(true_return, true_first * np.where(raw["swap"], -1, 1))
    np.testing.assert_array_equal(raw["first_side"].reshape(24, 4)[:, [0, 1]],
                                  raw["first_side"].reshape(24, 4)[:, [2, 3]])


def test_ridge_uses_calibration_coefficients_and_balance_is_per_action_swap_taste_cell():
    rng = np.random.default_rng(12)
    action, swap, taste = np.indices((2, 2, 2)).reshape(3, 8)
    repeats = np.arange(1, 9)
    action, swap, taste = (np.repeat(value, repeats) for value in (action, swap, taste))
    action, taste = action * 2 - 1, taste * 2 - 1
    weights, count = probe.balanced_weights(action, swap, taste)
    np.testing.assert_array_equal(count.flatten(), repeats)
    np.testing.assert_allclose(np.bincount(np.repeat(np.arange(8), repeats), weights=weights), 1 / 8)
    first, returned = action * taste, action * taste * (1 - 2 * swap)
    labels = np.stack((taste, first, returned), axis=1)
    calibrate = (labels + rng.normal(scale=.01, size=labels.shape))[:, None, :]
    heldout = (labels + rng.normal(scale=.01, size=labels.shape))[:, None, :]
    head = probe.fit_readout(calibrate, labels, weights, .1)
    np.testing.assert_allclose(probe.accuracy(probe.score_readout(heldout, head), labels, weights), 1.)
    np.testing.assert_allclose(probe.accuracy(np.zeros((len(taste), 1, 3)), labels, weights), .5)


def test_end_to_end_independent_pre_taste_and_unseen_return_labels_are_at_chance():
    graph = fw.spawn(731, 8)
    metrics, raw = probe.diagnose(graph, 731, 20261124, 96, 128, 4, .1, keep_connected=True)
    scores = np.array(metrics["accuracy"])
    np.testing.assert_allclose(scores[0], .5)
    np.testing.assert_allclose(scores[:4, 2], .5)
    assert raw["score"].shape == (128 * 4, len(probe.PHASES), len(probe.TARGETS))
    assert raw["coef"].shape == (len(probe.PHASES), graph.width, len(probe.TARGETS))
    assert scores[1, 0] > .9
