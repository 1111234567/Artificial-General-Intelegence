"""Checks for physical receipt alignment and the declared sensory prior."""

import numpy as np

from experiments.pulse_game import mixture_world as taste, world
from experiments.pulse_game.symmetry_spike import Skill, SpikingSkill
from experiments.pulse_game.symmetry_spike_matrix import MatrixLearner, MatrixSkill, alternate
from experiments.pulse_game.symmetry_spike_run import make_pair, paired


def test_matched_arm_preserves_a_receipts_with_a_native_spiking_body():
    family_pair = make_pair([9011, 9012, 9013, 9014], probes=1)
    report = paired(family_pair, Skill(), 22)
    assert all(report["same_a_observed_waveform"])
    assert all(report["both_contact"])


def test_opponent_prior_is_odd_under_matched_initial_state():
    brain = SpikingSkill(Skill(), birth=8)
    brain.reset(2)
    raw = np.zeros((2, world.OBS_SIZE), dtype=np.float32)
    value = np.array([.86, .45, .31, .64], dtype=np.float32)
    raw[0, taste.LEFT] = value
    raw[1, taste.LEFT] = 1 - value
    raw[:, taste.RIGHT] = .5
    brain.sensory(raw)
    raw[:, taste.LEFT] = .5
    for _ in range(4):
        brain.sensory(raw)
    address = brain.address()[:, 0]
    assert abs(address[0]).sum() > 0
    np.testing.assert_allclose(address[0], -address[1], atol=1e-7)


def test_quarter_turn_keeps_rewards_and_native_receipt_times():
    ref = make_pair([9011, 9012, 9013, 9014], probes=1)
    changed = alternate(ref, "quarter-turn")
    for old, new in zip(ref, changed, strict=True):
        np.testing.assert_array_equal(new.menu.realized, old.menu.realized)
        np.testing.assert_array_equal(new.menu.probability, old.menu.probability)
        assert new.menu.body is old.menu.body
        witnesses = []
        for row in range(len(new.menu.realized)):
            for trial in range(new.exposed):
                parent = new.pairmates[row, trial]
                if parent >= 0:
                    x = new.menu.chemicals[row, parent, 0] * 2 - 1
                    y = new.menu.chemicals[row, trial, 0] * 2 - 1
                    witnesses.append(np.dot(x, y))
        assert np.max(np.abs(witnesses)) < .25
    result = paired(changed, MatrixSkill(encoding="linear"), 22, actor=MatrixLearner)
    assert all(result["same_a_observed_waveform"])
    assert all(result["both_contact"])
