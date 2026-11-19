"""Causal and permission checks for the native edit receipt loop."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from experiments.pulse_game import edit_receipt_graph as native, self_param_writer as fw


def contact(state):
    batch = len(state.hidden)
    rng = torch.Generator().manual_seed(630)
    left = torch.randn(batch, 3, generator=rng)
    outcome = (torch.arange(batch) % 2).float() * 2 - 1
    draws = torch.rand(batch, 4, generator=rng)
    return left, -left, outcome, draws


def test_receipt_classifies_written_rejected_withheld_and_noop_commands():
    root = fw.spawn(631, 8)
    width = root.width
    state = native.start(root, 7)
    left, right, outcome, draws = contact(state)
    missing = torch.where(~root.present[0].flatten())[0][0]
    loop = torch.where(root.present[0].diag())[0][0]
    existing = loop * (width + 1)
    address = torch.tensor([missing, existing, existing, width * width + 2,
                            width * width + 3, width * width + 4, width * width + 5])
    op = torch.tensor([fw.Op.BIRTH, fw.Op.RETIRE, fw.Op.BIRTH, fw.Op.ADJUST,
                       fw.Op.NONE, fw.Op.ADJUST, fw.Op.ADJUST])
    payload = torch.tensor([0.0, 0.0, .2, .5, 1.0, 0.0, 4.0])
    proposal = replace(native.propose(state, left, right, outcome, draws),
                       command=fw.Command(address, op, payload))
    apply = torch.tensor([True, True, True, False, True, True, True])
    result = native.commit(state, proposal, draws, accept_override=apply)
    receipt = result.receipt

    assert receipt.status.tolist() == [fw.Status.APPLIED, fw.Status.APPLIED, fw.Status.SUPPORT_MISMATCH,
                                       native.WITHHELD, fw.Status.DECLINED, fw.Status.UNCHANGED, fw.Status.APPLIED]
    assert receipt.kind.tolist() == [native.EditKind.APPLIED, native.EditKind.APPLIED, native.EditKind.REJECTED,
                                     native.EditKind.WITHHELD, native.EditKind.NOOP, native.EditKind.NOOP,
                                     native.EditKind.APPLIED]
    assert receipt.topology_delta.tolist() == [1, -1, 0, 0, 0, 0, 0]
    assert receipt.value_delta[0] == 0
    assert receipt.value_delta[1] == -root.weights[0, loop, loop]
    assert torch.count_nonzero(receipt.effect_target[2:6, 1:]) == 0
    assert torch.count_nonzero(receipt.activation_delta[2:6]) > 0
    assert receipt.value_after[-1] > fw.LIMIT
    assert result.state.graph.present[0].flatten()[missing]
    assert not result.state.graph.present[1, loop, loop]
    assert torch.count_nonzero(result.state.graph.weights[~result.state.graph.present]) == 0
    torch.testing.assert_close(result.sensed_receipt, receipt.features)


def test_before_after_are_local_actual_commit_observations_and_override_is_only_sensory():
    root = fw.spawn(632, 4)
    state = native.start(root, 2)
    left, right, outcome, draws = contact(state)
    # A writer bias is reachable and writable, as are ordinary node biases.
    receivers = torch.tensor([fw.BODY + 3, 9])
    command = fw.Command(root.width ** 2 + receivers,
                         torch.full_like(receivers, fw.Op.ADJUST), torch.tensor([.6, -.4]))
    with patch.object(fw, "commit", side_effect=AssertionError("Propose should precede commit")):
        proposal = replace(native.propose(state, left, right, outcome, draws), command=command)
    actual = native.commit(state, proposal, draws, accept_override=True)
    hidden_receipt = native.commit(state, proposal, draws, accept_override=True,
                                   receipt_override=torch.zeros(2, native.RECEIPT_WIDTH))
    refused = native.commit(state, proposal, draws, accept_override=False)
    protected = native.commit(state, proposal, draws, accept_override=True, intervention="protect_writer")

    assert actual.receipt.status.tolist() == [fw.Status.APPLIED, fw.Status.APPLIED]
    assert refused.receipt.kind.tolist() == [native.EditKind.WITHHELD, native.EditKind.WITHHELD]
    assert protected.receipt.status.tolist() == [fw.Status.WRITER_PROTECTED, fw.Status.APPLIED]
    torch.testing.assert_close(actual.state.graph.bias, hidden_receipt.state.graph.bias)
    torch.testing.assert_close(actual.receipt.features, hidden_receipt.receipt.features)
    assert not torch.equal(actual.state.hidden, hidden_receipt.state.hidden)
    assert torch.count_nonzero(hidden_receipt.sensed_receipt) == 0
    assert torch.count_nonzero(refused.receipt.effect_target[:, 1:]) == 0
    assert torch.count_nonzero(refused.receipt.activation_delta) > 0
    rows = torch.arange(2)
    neutral = torch.zeros_like(state.hidden)
    before = proposal.hidden
    after = fw.advance(actual.state.graph, before, neutral, 1)
    torch.testing.assert_close(actual.receipt.activation_before[:, 0], before[rows, receivers])
    torch.testing.assert_close(actual.receipt.activation_after[:, 0], after[rows, receivers])
    torch.testing.assert_close(actual.receipt.activation_delta[:, 1], torch.zeros(2))
    torch.testing.assert_close(hidden_receipt.state.hidden, fw.advance(actual.state.graph, after, neutral, 1))


def test_sensing_controls_match_signal_donors_and_preserve_executor_and_clock():
    state = native.start(fw.spawn(633, 8), 4)
    args = contact(state)
    donors = torch.tensor([2, 3, 0, 1])
    own = native.propose(state, *args)
    blind = native.propose(state, *args, sensing="blind")
    yoked = native.propose(state, *args, sensing="yoked")
    shuffled = native.propose(state, *args, sensing="shuffled", donor=donors)
    assert torch.count_nonzero(blind.sense) == 0
    torch.testing.assert_close(yoked.sense, own.sense[[1, 0, 3, 2]])
    torch.testing.assert_close(shuffled.sense, own.sense[donors])
    for proposal in (blind, yoked, shuffled):
        torch.testing.assert_close(proposal.command.address, own.command.address)
        torch.testing.assert_close(proposal.command.op, own.command.op)
        torch.testing.assert_close(proposal.command.payload, own.command.payload)
    with pytest.raises(ValueError, match="no self matches"):
        native.propose(state, *args, sensing="shuffled", donor=torch.arange(4))

    result = native.commit(state, shuffled, args[-1], gate="open")
    torch.testing.assert_close(result.sensed_receipt, result.receipt.features[donors])
    unread = native.commit(state, shuffled, args[-1], gate="open", sensing="zero")
    torch.testing.assert_close(result.state.graph.weights, unread.state.graph.weights)
    torch.testing.assert_close(result.state.graph.bias, unread.state.graph.bias)
    torch.testing.assert_close(result.state.graph.present, unread.state.graph.present)
    torch.testing.assert_close(result.receipt.features, unread.receipt.features)
    assert torch.count_nonzero(unread.sensed_receipt) == 0


def test_native_effect_and_acceptance_have_gradients_and_motor_cannot_see_untasted_outcome():
    original = fw.spawn(634, 8)
    birth = fw.Graph(original.weights.requires_grad_(), original.bias.requires_grad_(), original.present)
    state = native.start(birth, 4)
    left, right, outcome, draws = contact(state)
    proposed = native.propose(state, left, right, outcome, draws)
    complement = native.propose(state, left, right, -outcome, draws)
    torch.testing.assert_close(proposed.side, complement.side)
    torch.testing.assert_close(proposed.motor_logit, complement.motor_logit)
    torch.testing.assert_close(proposed.taste, -complement.taste)
    assert not torch.equal(proposed.hidden, complement.hidden)

    outcome = native.commit(state, proposed, draws)
    for loss in (proposed.effect_prediction.square().mean(), -outcome.gate_log_prob.mean()):
        grad = torch.autograd.grad(loss, (birth.weights, birth.bias), retain_graph=True)
        assert all(torch.isfinite(x).all() and torch.count_nonzero(x) > 0 for x in grad)
        assert torch.count_nonzero(grad[0][~birth.present]) == 0
    assert torch.count_nonzero(outcome.gate_log_prob[proposed.command.op == fw.Op.NONE]) == 0
