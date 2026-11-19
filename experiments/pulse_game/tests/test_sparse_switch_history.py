import numpy as np

from experiments.pulse_game.sparse_pairs_model import RECORD
from experiments.pulse_game.sparse_switch_history import HistoryGraph
from experiments.pulse_game.unified_live_graph import Recipe


def make_graph():
    return HistoryGraph([17, 23], Recipe(), world_seeds=[4, 9], pair_rate=.65, pair_decay=.993)


def test_history_filter_only_restricts_original_samples():
    actor = make_graph()
    fresh = np.array([True, True])
    active = np.empty(0, dtype=np.int32)
    candidates, priority = actor._sample(3, fresh, active)
    assert len(candidates) > 2
    actor.allowed_pair_births = candidates[::2]
    filtered, filtered_priority = actor._sample(3, fresh, active)
    np.testing.assert_array_equal(filtered, candidates[::2])
    np.testing.assert_array_equal(filtered_priority, priority[::2])


def test_snapshot_protection_leaves_support_exit_checks_to_the_base_policy():
    actor = make_graph()
    actor.receipts[:] = actor.policy.grace + 1
    records = np.zeros(1, dtype=RECORD)
    support = np.empty(0, dtype=np.int32)
    fresh = np.array([True, True])
    assert actor._evidence_retire(records, support, fresh, 777, None).all()
    actor.protect_from_evidence_exit = True
    assert not actor._evidence_retire(records, support, fresh, 777, None).any()
