import numpy as np

from experiments.pulse_game import self_write_core as core
from experiments.pulse_game import self_write_core_lifetime as life


def edit_for(graph, *, delta=0.01, birth=-1, retire=-1):
    return core.Proposal(np.full_like(graph.w, delta), np.full_like(graph.b, -delta),
                         np.array([birth]), np.array([-0.12], dtype=np.float32), np.array([retire]),
                         np.array([0.25]), np.array([0.25]))


def test_all_graph_rows_can_be_actuated_and_primitive_protects_the_emitter():
    graph = core.spawn_graph(4)
    edit = edit_for(graph)
    updated, receipt = core.apply(graph, edit)
    fixed, fixed_receipt = core.apply(graph, edit, "primitive")
    null, null_receipt = core.apply(graph, edit, "none")
    assert np.all(receipt.edge_delta[graph.mask] > 0)
    assert np.all(receipt.bias_delta < 0)
    assert np.array_equal(fixed.w[:, :12], updated.w[:, :12])
    assert np.array_equal(fixed.w[:, 12:], graph.w[:, 12:])
    assert np.array_equal(fixed.b[:, 12:], graph.b[:, 12:])
    assert np.all(fixed_receipt.edge_delta[:, 12:] == 0)
    assert np.array_equal(null.w, graph.w)
    assert not np.any(null_receipt.bias_delta)


def test_birth_uses_emitted_signed_weight_and_retirement_has_connectivity_guard():
    mask = np.zeros((1, core.N, core.N), dtype=bool)
    order = np.arange(core.N)
    mask[:, (order + 1) % core.N, order] = True
    graph = core.Graph(mask.astype(np.float32), np.zeros((1, core.N), np.float32), mask)
    no_birth, receipt = core.apply(graph, edit_for(graph, retire=core.N))
    assert receipt.retire_status[0] == 4
    assert np.array_equal(no_birth.mask, graph.mask)
    new_graph, born = core.apply(graph, edit_for(graph, birth=2 * core.N))
    assert born.birth_status[0] == 1
    assert born.edge_count[0] == core.N + 1
    assert new_graph.w[0, 2, 0] == np.float32(-0.12)
    retired, gone = core.apply(new_graph, edit_for(new_graph, retire=2 * core.N))
    assert gone.retire_status[0] == 1
    assert gone.edge_count[0] == core.N
    assert retired.w[0, 2, 0] == 0
    assert core.connected(retired.mask[0])
    _, same_round = core.apply(graph, edit_for(graph, birth=2 * core.N, retire=2 * core.N))
    assert same_round.retire_status[0] == 1
    assert same_round.retired_value[0] == np.float32(-0.12)


def test_blind_proposals_cannot_see_support_or_values_with_same_activity():
    a = core.spawn_graph(0, 48)
    b = core.spawn_graph(1, 48)
    pre = np.ones_like(a.b) * 0.1
    post = np.ones_like(a.b) * 0.9
    blind_a = core.proposal(a, pre, post, 4)
    blind_b = core.proposal(b, pre, post, 4)
    assert np.any(blind_a.birth >= 0)
    assert np.any(blind_a.retire >= 0)
    for field in core.Proposal.__dataclass_fields__:
        assert np.array_equal(getattr(blind_a, field), getattr(blind_b, field))
    own_a = core.proposal(a, pre, post, 4, view="own")
    own_b = core.proposal(a, pre, post, 4, view="own", view_graph=b)
    assert not np.array_equal(own_a.edge_delta, own_b.edge_delta)


def test_transplant_uses_writer_changes_without_swapping_rest_of_graph():
    base = core.spawn_graph(8)
    donor, _ = core.apply(base, edit_for(base))
    receiver = core.spawn_graph(8)
    receiver.b[:, 0] += 0.15
    result = core.transplant_writer(receiver, donor, base)
    assert np.array_equal(result.w[:, :12], receiver.w[:, :12])
    assert np.array_equal(result.b[:, :12], receiver.b[:, :12])
    assert np.array_equal(result.w[:, 12:], donor.w[:, 12:])
    assert np.array_equal(result.b[:, 12:], donor.b[:, 12:])


def test_first_surprise_is_physically_identical_across_hidden_world_regimes():
    graph = core.spawn_graph(0)
    world = life.worlds(0, 4)
    stationary, _, a, _, _ = life.run_world(graph, world, "stationary", "active",
                                             keep_log=False, snapshots=False)
    changing, _, b, _, _ = life.run_world(graph, world, "changing", "active",
                                            keep_log=False, snapshots=False)
    np.testing.assert_array_equal(a["actual_taste"][:49], b["actual_taste"][:49])
    np.testing.assert_array_equal(a["side"][:49], b["side"][:49])
    for graph in (stationary, changing):
        assert all(map(core.connected, graph.mask))

def test_sparse_writer_slots_have_no_true_support_oracle_and_bounded_size():
    a = core.spawn_graph(10, 24)
    b = core.spawn_graph(11, 24)
    rng = np.random.default_rng(33)
    pre = rng.uniform(-1, 1, size=a.b.shape).astype(np.float32)
    post = rng.uniform(-1, 1, size=a.b.shape).astype(np.float32)
    pa = core.proposal(a, pre, post, 33, sparse=True)
    pb = core.proposal(b, pre, post, 33, sparse=True)
    np.testing.assert_array_equal(pa.edge_delta, pb.edge_delta)
    np.testing.assert_array_equal(pa.bias_delta, pb.bias_delta)
    assert np.all(np.count_nonzero(pa.edge_delta, axis=(1, 2)) <= 4)
    assert np.all(np.count_nonzero(pa.bias_delta, axis=1) <= 1)
    assert not np.any(pa.edge_delta[:, np.arange(core.N), np.arange(core.N)])
    changed, receipt = core.apply(a, pa)
    assert np.all(np.count_nonzero(receipt.edge_delta, axis=(1, 2)) <= 4)
    assert np.all(np.count_nonzero(receipt.bias_delta, axis=1) <= 1)
    assert all(map(core.connected, changed.mask))
