import json

import numpy as np
import pytest

from experiments.pulse_game import self_dose_life as life
from experiments.pulse_game.self_dose_life_analysis import CrossedBootstrap


def test_private_complement_and_donor_ecology_balanced_at_every_dose():
    plan = json.loads(life.PLAN.read_text(encoding="utf-8"))
    donor = life.donor_tape(plan)
    receiver = life.receiver_tape(plan)
    n = plan["contacts_per_donor_problem"]
    life.verify_private_preferences(donor, receiver, n)
    for dose in plan["doses"]:
        np.testing.assert_array_equal(donor.ecology[:dose * n:n].sum(axis=0), dose // 2)
    np.testing.assert_array_equal(donor.good[::n].repeat(n, axis=0), donor.good)
    left, right, target, tastes = receiver.present(0, np.arange(receiver.good.shape[1]))
    np.testing.assert_array_equal(left, -right)
    np.testing.assert_array_equal(target[::2], -target[1::2])
    np.testing.assert_array_equal(tastes[::2], -tastes[1::2])


def test_writer_only_graft_preserves_a_disconnection_without_copying_donor_body():
    dimension = life.core.N
    mask = np.zeros((1, dimension, dimension), dtype=np.uint8)
    index = np.arange(dimension)
    mask[0, (index + 1) % dimension, index] = 1
    base = life.checked_graph(mask.astype(np.float32) * 0.25, np.zeros((1, dimension), np.float32), mask)
    donor = base.clone()
    for target, source, weight in ((12, 11, 0), (12, 10, 0.3), (10, 11, 0.45)):
        donor.mask[0, target, source] = bool(weight)
        donor.w[0, target, source] = weight
    donor.b[0, [10, 12]] = 1
    assert life.core.connected(base.mask[0]) and life.core.connected(donor.mask[0])
    graft = life.writer_payload(base, donor)
    assert not life.core.connected(graft.mask[0])
    assert not graft.mask[0, 10, 11]
    assert graft.b[0, 10] == 0
    assert graft.b[0, 12] == 1
    assert graft.mask[0, 12, 10]
    with pytest.raises(ValueError, match="binary"):
        life.checked_graph(base.w, base.b, 2 * mask)


def test_crossed_weights_preserve_one_cell_per_cross_and_paired_zero():
    resample = CrossedBootstrap((3, 4, 2), {"root": 1, "draws": 100, "interval": [0.025, 0.975]})
    np.testing.assert_allclose(resample.weights.sum(axis=1), 1)
    ones = np.ones((3, 4, 2))
    assert resample.interval(ones)["mean"] == 1
    assert resample.interval(ones - ones) == {"mean": 0, "ci95": [0, 0]}
