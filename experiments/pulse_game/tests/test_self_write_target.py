import numpy as np

from experiments.pulse_game import self_write_core as core
from experiments.pulse_game import self_write_target as target


def test_target_current_value_has_the_correct_control_sign_and_needs_own_readback():
    graph = core.spawn_graph(1)
    graph.w[0, 10, 0] = 2.40
    graph.b[0, 10] = 1.40
    pre = np.full_like(graph.b, 0.80)
    post = np.full_like(graph.b, 0.84)
    post[:, 14:16] = 0
    visible = target.proposal(graph, pre, post, 2, view="own")
    masked = target.proposal(graph, pre, post, 2, view="blind")
    assert visible.edge_delta[0, 10, 0] < 0 < masked.edge_delta[0, 10, 0]
    assert visible.bias_delta[0, 10] < 0 < masked.bias_delta[0, 10]
    sequence = []
    for _ in range(5):
        edit = target.proposal(graph, pre, post, 2, view="own")
        assert edit.birth[0] == edit.retire[0] == -1
        graph, _ = core.apply(graph, edit)
        sequence.append(float(graph.w[0, 10, 0]))
    assert np.all(np.diff(sequence) < 0)


def test_hidden_true_support_cannot_change_blind_address_or_numeric_proposals():
    first = core.spawn_graph(1, 48)
    second = core.spawn_graph(2, 48)
    pre = np.full_like(first.b, 0.72)
    post = np.full_like(first.b, 0.79)
    a = target.proposal(first, pre, post, 7, view="blind")
    b = target.proposal(second, pre, post, 7, view="blind")
    for field in core.Proposal.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
