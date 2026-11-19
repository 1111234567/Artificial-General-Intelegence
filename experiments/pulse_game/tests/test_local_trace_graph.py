"""Physical invariants and causal controls for retained local plasticity."""

import torch

from experiments.pulse_game import local_trace_graph as g
from experiments.pulse_game import self_param_writer as native


def _experience(rule, state, side=1):
    left = state.hidden.new_tensor([0.5, -0.3, 0.9]).expand(len(state.hidden), -1)
    action = g.act(rule, state, left, -left, side_override=side)
    return g.observe(rule, action, left, -left, torch.full_like(action.side, 1.0))


def test_native_ticks_trace_hold_and_sham():
    rule = g.Rule(42)
    state = g.start(g.spawn(42), 2)
    experience = _experience(rule, state)
    left = state.hidden.new_tensor([[0.5, -0.3, 0.9]]).expand(2, -1)
    side = torch.ones(2)
    h = native.advance(state.graph, state.hidden, native.sensory(state.graph, left, -left), 5)
    h = native.advance(state.graph, h, native.sensory(state.graph, left, -left, side), 3)
    h = native.advance(state.graph, h, native.sensory(state.graph, left, -left, side, side), 3)
    assert torch.equal(experience.state.hidden, h)
    assert experience.state.sender_trace.abs().sum() > 0
    assert experience.state.receiver_trace.abs().sum() > 0

    result = g.commit(rule, experience, torch.zeros(2, 4), intervention="sham")
    assert result.state.graph is state.graph
    assert torch.equal(result.state.hidden, native.advance(state.graph, h, torch.zeros_like(h), 6))
    assert result.state.sender_trace is experience.state.sender_trace
    assert result.state.receiver_trace is experience.state.receiver_trace
    assert (result.receipt.proposed_birth >= 0).all() and (result.receipt.proposed_retirement >= 0).all()
    assert (result.receipt.birth == -1).all() and (result.receipt.retired == -1).all()
    assert not result.receipt.weight_delta.any() and not result.receipt.bias_delta.any()

    mixed = g.wait(rule, result.state, [0, 4])
    assert torch.equal(mixed.hidden[:1], result.state.hidden[:1])
    torch.testing.assert_close(mixed.hidden[1:], g.wait(rule, result.state.subset(1), 4).hidden)
    assert mixed.sender_trace is result.state.sender_trace
    donor = result.state.subset([1, 0])
    exchanged = g.swap(mixed, donor, "trace")
    assert exchanged.sender_trace is donor.sender_trace
    assert exchanged.graph is mixed.graph and exchanged.hidden is mixed.hidden


def test_independent_birth_and_retirement_allow_self_edges_and_reconcile_connectivity():
    first = torch.arange(g.NODES)
    present = torch.zeros(1, g.NODES, g.NODES, dtype=torch.bool)
    present[0, (first + 1) % g.NODES, first] = True
    graph = g.Graph(0.125 * present, torch.zeros(1, g.NODES), present)
    rule = g.Rule(1)
    with torch.no_grad():
        for parameter in rule.parameters():
            parameter.zero_()

    experience = _experience(rule, g.start(graph))
    result = g.commit(rule, experience, torch.zeros(1, 4))
    assert result.receipt.proposed_retirement.item() == g.NODES - 1
    assert result.receipt.retired.item() == -1
    assert result.receipt.birth.item() == 0
    assert result.state.graph.present[0, 0, 0]
    assert result.state.graph.weights[0, 0, 0] == g.MIN_WEIGHT
    assert not result.receipt.weight_delta.any() and not result.receipt.bias_delta.any()
    assert result.state.graph.present.sum() - present.sum() == 1
    assert native.strongly_connected(result.state.graph.present[0].numpy())

    experience = _experience(rule, result.state)
    result = g.commit(rule, experience, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert result.receipt.birth.item() == -1
    assert result.receipt.retired.item() == 0
    assert result.receipt.retired_weight.item() > 0
    assert torch.equal(result.state.graph.present, present)
    assert torch.equal(result.state.graph.weights, graph.weights)


def test_shuffle_uses_realized_deltas_and_preserves_birth_dose():
    rng = torch.Generator().manual_seed(921)
    rule = g.Rule(11)
    state = g.start(g.spawn(11), 3)
    experience = _experience(rule, state)
    edit = torch.zeros(3, 4)
    priorities = torch.rand(3, g.SHUFFLE_DRAWS, generator=rng)
    active = g.commit(rule, experience, edit)
    for mode in ("shuffle", "shuffle_values"):
        result = g.commit(rule, experience, edit, intervention=mode, shuffle=priorities)
        rc = result.receipt
        assert torch.equal(rc.match.reference.weight_delta, active.receipt.weight_delta)
        assert rc.match.weight_matches.all() and rc.match.bias_matches.all()
        assert torch.equal(rc.birth_weight, active.receipt.birth_weight)
        assert torch.equal(rc.birth >= 0, active.receipt.birth >= 0)
        assert torch.equal(rc.retired >= 0, active.receipt.retired >= 0)
        assert not torch.equal(rc.weight_delta, active.receipt.weight_delta)
        assert not torch.equal(rc.bias_delta, active.receipt.bias_delta)
        graph = result.state.graph
        common = state.graph.present & graph.present
        assert torch.equal(rc.weight_delta, torch.where(common, graph.weights - state.graph.weights, 0.0))
        assert torch.equal(graph.present, graph.weights != 0)
        assert all(native.strongly_connected(p.numpy()) for p in graph.present)
        if mode == "shuffle_values":
            assert rc.match.retirement_matches.all()
            assert not rc.match.birth_relocated.any() and not rc.match.retirement_relocated.any()
        else:
            assert rc.match.birth_relocated.all() and rc.match.retirement_relocated.all()


def test_grouped_roots_keep_rule_gradients_independent():
    roots = [14, 18]
    rule = g.Rule(roots)
    view = rule.expand(2)
    initial = g.start(g.spawn(roots), 2)
    experience = _experience(view, initial)
    result = g.commit(view, experience, torch.full((4, 4), 0.01))
    loss = result.state.graph.weights[:2].sum() + result.state.graph.bias[:2].sum()
    loss = loss + result.edit_log_prob[:2].sum()
    loss.backward()
    for parameter in rule.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad[0]) > 0
        assert torch.count_nonzero(parameter.grad[1]) == 0
    for index, root in enumerate(roots):
        solo = g.Rule(root)
        for name, value in rule.state_dict().items():
            assert torch.equal(value[index], solo.state_dict()[name][0])


def test_edit_sampling_accounts_for_gates_no_support_and_cdf_boundaries():
    gate = torch.zeros(4, requires_grad=True)
    score = torch.zeros(4, 4, requires_grad=True)
    available = torch.tensor([[False, True, False, True], [False, False, False, False],
                              [True, False, True, False], [False, True, False, True]])
    draws = torch.tensor([[0, 0], [0, 1], [0, 1], [1, 0]], dtype=torch.float64)
    index, lp = g._sample_edit(gate, score, available, draws)
    assert index.tolist() == [1, -1, 2, -1]
    assert torch.allclose(lp.detach().exp(), torch.tensor([0.25, 1, 0.25, 0.5]))
    lp.sum().backward()
    assert torch.isfinite(score.grad).all() and torch.isfinite(gate.grad).all()
    assert not score.grad[1].any() and gate.grad[1] == 0
