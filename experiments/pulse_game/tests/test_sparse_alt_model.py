import numpy as np
import pytest

from experiments.pulse_game import graph_interactions, world
from experiments.pulse_game.dendrite_ports_graph import DendriteGraph, FIRST, SECOND
from experiments.pulse_game.sparse_alt_model import CorroboratingGraph, ProbePolicy
from experiments.pulse_game.unified_live_graph import Recipe
from experiments.pulse_game.unified_live_graph_study import Body


def make_graph(policy=ProbePolicy()):
    return CorroboratingGraph([88731], Recipe(.045, .23, .991), world_seeds=[7635],
                             policy=policy, pair_rate=.65, pair_decay=.993)


def first_legal(graph):
    receiver, pair = np.argwhere(graph.mask[0][:, FIRST] & graph.mask[0][:, SECOND])[0]
    return receiver, FIRST[pair], SECOND[pair]


def test_sparse_current_and_credit_match_same_pairs_in_dense_graph():
    sparse = make_graph()
    dense = DendriteGraph([88731], sparse.recipe, product=True, pair_rate=.65, pair_decay=.993)
    np.testing.assert_array_equal(sparse.weight, dense.weight)
    allowed = dense.valid_pairs
    indices = np.argwhere(allowed)[::29]
    lanes, receivers, pairs = indices.T
    values = np.random.default_rng(6753).normal(0, .1, size=len(indices)).astype(np.float32)
    sparse.live = np.stack((lanes, receivers, FIRST[pairs], SECOND[pairs]))
    sparse.values = values
    dense.pair[lanes, receivers, pairs] = values
    dense.refresh_pair_support()
    menu, _, _, _ = graph_interactions.make_menu([7635], "stable", 32)
    body = Body(menu, None, 1)
    observation = body.observe()
    for brain in (sparse, dense):
        brain.start(np.array([.13], dtype=np.float32))
    for _ in range(world.TICKS):
        action = sparse.advance(observation)
        np.testing.assert_array_equal(action, dense.advance(observation))
        np.testing.assert_allclose(sparse.state, dense.state, rtol=1e-5, atol=1e-6)
        observation = body.step(action)
    assert sparse.receipt[0] != 0
    credit, sent = sparse.trial_credit()
    first_grad, pair_grad = dense.gradients()
    np.testing.assert_allclose(np.matmul(credit, sent), first_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sparse.pair_gradient(credit, sent, *sparse.live),
                               pair_grad[lanes, receivers, pairs], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("later_sign, admits", [(1, True), (-1, False)])
def test_probe_needs_disjoint_signed_receipt_blocks(monkeypatch, later_sign, admits):
    graph = make_graph(ProbePolicy(slots=1, half_window=2, evidence=1.5, minimum_mean=.001))
    receiver, first, second = first_legal(graph)
    address = (0, receiver, 0)
    graph.probe_first[address] = first
    graph.probe_second[address] = second
    monkeypatch.setattr(graph, "_reserve_probes", lambda fresh: None)
    sent = np.ones((1, 1, graph.mask.shape[-1]), dtype=np.float32)
    credit = np.zeros((1, graph.mask.shape[-1], 1), dtype=np.float32)
    for index, sign in enumerate((1, 1, later_sign, later_sign)):
        credit[0, receiver, 0] = sign * .1
        graph._evaluate_probes(np.array([True]), credit, sent)
        if index < 3:
            assert graph.values.size == 0
    assert graph.probe_first[address] == -1
    assert graph.values.size == int(admits)
    if admits:
        assert graph.values[0] > 0
        graph.mask[0, receiver, second] = False
        graph._drop_illegal()
        assert graph.values.size == 0
        assert graph.pair_events[0, 6] == 1


def test_silent_trials_advance_neither_probe_nor_weights():
    graph = make_graph()
    graph.start(np.zeros(1, dtype=np.float32))
    receiver, first, second = first_legal(graph)
    graph.probe_first[0, receiver, 0] = first
    graph.probe_second[0, receiver, 0] = second
    graph.probe_age[0, receiver, 0] = 3
    before = graph.mask.copy(), graph.weight.copy()
    graph.update(0)
    assert graph.probe_age[0, receiver, 0] == 3
    assert graph.values.size == 0 and not graph.receipts.any()
    np.testing.assert_array_equal(graph.mask, before[0])
    np.testing.assert_array_equal(graph.weight, before[1])


def test_soft_threshold_removes_exact_zero_live_record():
    graph = make_graph()
    receiver, first, second = first_legal(graph)
    graph.live = np.array([[0], [receiver], [first], [second]])
    graph.values = np.array([1e-5], dtype=np.float32)
    credit = np.zeros((1, graph.mask.shape[-1], 1), dtype=np.float32)
    sent = np.zeros((1, 1, graph.mask.shape[-1]), dtype=np.float32)
    graph._learn_live(np.array([True]), credit, sent)
    assert graph.values.size == 0
    assert graph.pair_events[0, 5] == 1
    assert graph.pair_events[0, 7] == 0
