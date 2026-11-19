"""Check actuator reversibility, sensor isolation, and physical branch matching."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.pulse_game import self_write_view as env
from experiments.pulse_game import self_write_view_study as study


def test_disturbance_and_every_accepted_command_have_exact_inverses():
    trial = env.random_trial(71, 128)
    inverse = env.apply(trial.damaged, trial.inverse)
    assert env.exact_rows(inverse.body, trial.reference).all()
    assert inverse.valid.all()
    old = trial.damaged.values.copy()
    support = trial.damaged.exists.copy()
    failures = 0
    for command in range(env.ACTION_COUNT):
        actions = np.full(len(old), command)
        receipt = env.apply(trial.damaged, actions)
        differences = ((receipt.body.exists != support) | (receipt.body.values != old)).sum(axis=1)
        np.testing.assert_array_equal(differences, receipt.changed)
        failures += (~receipt.valid).sum()
        reversed_actions = env.inverse_action(trial.damaged, actions, receipt)
        reversed_body = env.apply(receipt.body, reversed_actions)
        assert reversed_body.valid.all()
        assert env.exact_rows(reversed_body.body, trial.damaged).all()
    assert failures > 0
    np.testing.assert_array_equal(trial.damaged.values, old)
    np.testing.assert_array_equal(trial.damaged.exists, support)
    with pytest.raises(ValueError):
        env.apply(trial.damaged, np.full(len(old), env.ACTION_COUNT))


def test_observation_arms_vary_only_the_structural_channels():
    trial = env.random_trial(802, 64)
    panel = {mode: env.local_features(trial, mode) for mode in env.MODES}
    genuine = panel["veridical"]
    start = env.ADDRESS_CHANNELS
    end = start + env.STRUCTURE_CHANNELS
    for observation in panel.values():
        assert observation.shape == (64, env.SLOTS, env.FEATURES)
        np.testing.assert_array_equal(observation[:, :, :start], genuine[:, :, :start])
        np.testing.assert_array_equal(observation[:, :, end:], genuine[:, :, end:])
    assert len(np.unique(genuine[0, :, :start], axis=0)) == env.SLOTS
    np.testing.assert_array_equal(genuine[:, :, start], trial.reference.exists)
    np.testing.assert_array_equal(genuine[:, :, start + 1], trial.reference.values)
    np.testing.assert_array_equal(genuine[:, :, start + 2], trial.damaged.exists)
    np.testing.assert_array_equal(genuine[:, :, start + 3], trial.damaged.values)
    np.testing.assert_array_equal(panel["blind"][:, :, start:end], 0)
    np.testing.assert_array_equal(panel["current_only"][:, :, start:start + 2], 0)
    np.testing.assert_array_equal(panel["current_only"][:, :, start + 2:end], genuine[:, :, start + 2:end])
    assert np.all(trial.yoke != np.arange(len(trial.drive)))
    np.testing.assert_array_equal(panel["yoked"][:, :, start:end], genuine[trial.yoke, :, start:end])
    np.testing.assert_array_equal(panel["stale"][:, :, start + 2:end], genuine[:, :, start:start + 2])
    wrong_labels = replace(trial, disturbance=np.full(64, -9), inverse=np.full(64, -9))
    for mode, observation in panel.items():
        np.testing.assert_array_equal(env.local_features(wrong_labels, mode), observation)


def test_one_real_write_repairs_and_counterfactual_undo_reopens_damage():
    trial = env.random_trial(5436, 128)
    target, damaged = study.measurements(trial)
    disturbed = trial.disturbance != env.NOOP
    assert np.all(damaged[disturbed] > 0)
    assert np.all(damaged[~disturbed] == 0)
    controls, potentials, valid = study.oracle_controls(trial, target, damaged, 883)
    rows = np.arange(len(trial.drive))
    chosen = controls["oracle"]
    ranking = np.random.default_rng(5496).random((len(chosen), env.SLOTS))
    oracle = study.score(trial, target, damaged, chosen, potentials[rows, chosen], valid, ranking)
    assert oracle["restored"].all()
    assert oracle["undone"].all()
    np.testing.assert_array_equal(oracle["error"], 0)
    np.testing.assert_array_equal(oracle["undo_error"], damaged)
    has_retarget = oracle["retarget_available"]
    assert has_retarget.any()
    retarget = env.apply(trial.damaged, oracle["retarget_action"])
    np.testing.assert_array_equal(retarget.delta[has_retarget], oracle["delta"][has_retarget])
    np.testing.assert_array_equal(retarget.operation[has_retarget], oracle["operation"][has_retarget])
    assert np.all(retarget.slot[has_retarget] != oracle["slot"][has_retarget])
    assert np.all(oracle["retarget_error"][has_retarget] > oracle["error"][has_retarget])


def test_paired_interval_preserves_pairing_and_damaged_filter():
    first = np.arange(60).reshape(3, 4, 5)
    mask = np.indices((4, 5)).sum(axis=0) % 2 == 0
    second = first - np.where(mask, 4, -9)
    assert study.paired_interval(first, second, mask) == {"difference": 4.0, "ci95": [4.0, 4.0]}
