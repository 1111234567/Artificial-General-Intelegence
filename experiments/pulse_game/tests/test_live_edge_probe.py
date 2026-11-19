import numpy as np
import pytest

from .. import joint_live_model as joint
from .. import live_edge_probe as probe, live_edge_probe_history as history


def test_edge_generations_and_post_birth_receipts():
    start = np.zeros((1, 4, 4), dtype=bool)
    start[0, 1, 0] = start[0, 3, 0] = True
    edits = np.array([[[9, 4], [4, 9], [9, -1], [6, -1]]])
    final = start.copy()
    final[0, 2, 1] = final[0, 1, 2] = True
    birth, generation, cohort = history.edge_identity(start, edits, final)
    assert (birth[0, 3, 0], generation[0, 3, 0], cohort[0, 3, 0]) == (-1, 1, 0)
    assert (birth[0, 1, 0], generation[0, 1, 0], cohort[0, 1, 0]) == (1, 2, 3)
    assert (birth[0, 2, 1], generation[0, 2, 1], cohort[0, 2, 1]) == (2, 2, 2)
    assert (birth[0, 1, 2], generation[0, 1, 2], cohort[0, 1, 2]) == (3, 1, 1)
    receipts = history.age_in_receipts(birth, np.array([[1, -1, 0, -1]]))
    np.testing.assert_array_equal(receipts[final], [1, 0, 1, 3])
    with pytest.raises(RuntimeError, match='support differs'):
        history.edge_identity(start, edits, start)


def test_resets_only_apply_to_requested_generation_class():
    values = np.array([-.5, -.1, .7, -.2, 0.], dtype=np.float32)
    cohort = np.array([0, 1, 2, 3, -1])
    reference = np.array([-1., .1, 0., .2, 0.], dtype=np.float32)
    arrays = probe.interventions(values, reference, cohort)
    np.testing.assert_array_equal(arrays['learned'], values)
    np.testing.assert_array_equal(arrays['reset_newborn'], np.r_[values[0], reference[1:]])
    np.testing.assert_array_equal(arrays['reset_initial'], np.r_[reference[0], values[1:]])
    np.testing.assert_array_equal(arrays['reset_all'], reference)
    np.testing.assert_array_equal(arrays['reset_first_novel'], np.r_[values[0], reference[1], values[2:]])


def test_silent_scoring_preserves_both_branch_state_and_receipt_visibility():
    width = joint.WIDTH
    mask = np.zeros((1, width, width), dtype=bool)
    nodes = np.arange(width)
    mask[0, (nodes + 1) % width, nodes] = True
    arrays = {
        'final_mask': mask,
        'final_state': np.full((1, width), .5, dtype=np.float32),
        'starting_gain': np.ones((1, width), dtype=np.float32),
        'final_drive': np.zeros((1, probe.sensory.SENSES, 2 * width), dtype=np.float32),
        'final_bias': np.zeros((1, 2 * width), dtype=np.float32),
        'final_readout': np.zeros((1, width), dtype=np.float32),
        'final_offset': np.ones(1, dtype=np.float32),
    }
    raw = np.zeros_like(mask, dtype=np.float32)
    rows = [[0, 0, 0, 0]]
    first = probe.score_probes(joint.Population, arrays, probe.Scorer(rows, False), raw)
    reverse = probe.score_probes(joint.Population, arrays, probe.Scorer(rows, True), raw)
    np.testing.assert_array_equal(first[..., :2], reverse[..., :2])
    np.testing.assert_array_equal(first[..., 2], -reverse[..., 2])
    np.testing.assert_array_equal(first[..., 0], np.ones_like(first[..., 0]))
    assert (arrays['final_state'] == .5).all()
