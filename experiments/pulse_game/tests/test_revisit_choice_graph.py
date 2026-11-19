"""Check the causal boundary and clock of native two-candidate editing."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from experiments.pulse_game import revisit_choice_graph as native, self_param_writer as fw


def sample_experience(width=8, batch=4, *, differentiable=False):
    root = fw.spawn(741, width)
    if differentiable:
        root.weights.requires_grad_()
        root.bias.requires_grad_()
    state = native.start(root, batch)
    rng = torch.Generator().manual_seed(742)
    odors = torch.randn(batch, 2, 3, generator=rng)
    outcome = (torch.arange(batch) % 2).float() * 2 - 1
    motor_draw = torch.rand(batch, generator=rng)
    draws = torch.rand(batch, 2, 2, generator=rng)
    return root, state, odors[:, 0], odors[:, 1], outcome, motor_draw, draws


@pytest.mark.parametrize("width", fw.WRITER_WIDTHS)
def test_physical_choice_precedes_taste_and_proposals_share_unchanged_prefix(width):
    root, state, left, right, outcome, draw, draws = sample_experience(width)
    action = native.act(state, left, right, draw)
    observed = action.side * outcome
    exp = native.taste(action, left, right, observed)
    reverse = native.taste(action, left, right, observed, taste_mode="reverse")
    absent = native.taste(action, left, right, observed, taste_mode="zero")
    yoked = native.taste(action, left, right, observed, taste_mode="yoked", donor=[2, 3, 0, 1])
    assert torch.equal(exp.contact_hidden, reverse.contact_hidden)
    assert torch.equal(exp.side, reverse.side)
    assert torch.equal(exp.motor_logit, reverse.motor_logit)
    assert torch.equal(reverse.taste, observed)
    assert torch.equal(reverse.sensed_taste, -observed)
    assert torch.count_nonzero(absent.sensed_taste) == 0
    assert torch.equal(yoked.sensed_taste, observed[[2, 3, 0, 1]])
    assert not torch.equal(exp.state.hidden, reverse.state.hidden)

    with patch.object(fw, "commit", side_effect=AssertionError("Propose cannot execute")):
        proposals = native.propose(exp, draws)
        reversed_order = native.propose(exp, draws.flip(1))
        duplicate = native.propose(exp, draws[:, :1].expand_as(draws))
    assert exp.state.graph is state.graph
    for field in ("address", "op", "payload"):
        torch.testing.assert_close(getattr(proposals.command, field),
                                   getattr(reversed_order.command, field).flip(1))
    torch.testing.assert_close(proposals.logits[:, :2], reversed_order.logits[:, :2].flip(1))
    torch.testing.assert_close(proposals.logits[:, 2], reversed_order.logits[:, 2])
    torch.testing.assert_close(duplicate.logits[:, 0], duplicate.logits[:, 1])
    torch.testing.assert_close(proposals.benefit, reversed_order.benefit.flip(1))
    torch.testing.assert_close(proposals.effect_prediction, reversed_order.effect_prediction.flip(1))
    assert proposals.effect_prediction.shape == (len(draw), 2, 3)
    neutral = torch.zeros_like(exp.state.hidden)
    expected_withhold = fw.advance(state.graph, exp.state.hidden, neutral, 4)
    torch.testing.assert_close(proposals.withheld_hidden, expected_withhold)


def test_only_selected_edit_commits_and_native_receipt_matches_exact_sham_stimulus():
    root, state, left, right, outcome, draw, draws = sample_experience()
    exp = native.observe(state, left, right, outcome, draw)
    proposals = native.propose(exp, draws)
    receivers = torch.tensor([fw.BODY + 4, fw.BODY + 5])
    command = fw.Command((root.width ** 2 + receivers).expand_as(proposals.command.address),
                         torch.full_like(proposals.command.op, fw.Op.ADJUST),
                         torch.tensor([+0.8, -0.7]).expand_as(proposals.command.payload))
    proposals = replace(proposals, command=command)
    decision = native.choose(proposals, override=[0, 1, native.WITHHOLD, native.WITHHOLD])
    with patch.object(fw, "commit", wraps=fw.commit) as execution:
        real = native.commit(exp, proposals, decision)
    assert execution.call_count == 1
    disabled = native.commit(exp, proposals, decision, intervention="disabled")
    sham = native.commit(exp, proposals, decision, intervention="disabled", receipt_override=real.sensed_receipt)

    assert real.accepted.tolist() == [True, True, False, False]
    assert real.raw_receipt.status.tolist() == [fw.Status.APPLIED, fw.Status.APPLIED,
                                                fw.Status.DECLINED, fw.Status.DECLINED]
    assert real.receipt.status.tolist() == [fw.Status.APPLIED, fw.Status.APPLIED, native.WITHHELD, native.WITHHELD]
    assert sham.raw_receipt.status.tolist() == [fw.Status.DISABLED, fw.Status.DISABLED,
                                                fw.Status.DECLINED, fw.Status.DECLINED]
    delta = real.state.graph.bias - state.graph.bias
    assert torch.count_nonzero(delta) == 2
    torch.testing.assert_close(delta[:2, receivers].diag(), torch.tensor([0.8, -0.7]))
    torch.testing.assert_close(real.state.graph.weights, state.graph.weights)
    torch.testing.assert_close(real.state.graph.present, state.graph.present)
    torch.testing.assert_close(sham.state.graph.bias, state.graph.bias)
    torch.testing.assert_close(disabled.state.graph.weights, real.state.graph.weights)
    torch.testing.assert_close(sham.sensed_receipt, real.receipt.features)
    assert not torch.equal(sham.receipt.features, sham.sensed_receipt)
    assert not torch.equal(sham.state.hidden[:2], real.state.hidden[:2])
    assert not torch.equal(sham.state.hidden[:2], disabled.state.hidden[:2])
    torch.testing.assert_close(sham.state.hidden[2:], real.state.hidden[2:])


def test_batched_counterfactual_wait_keeps_continuous_activity_and_graph():
    _, state, left, right, outcome, draw, draws = sample_experience()
    batch = len(draw)
    exp = native.observe(state, left, right, outcome, draw)
    proposals = native.propose(exp, draws)
    forked = proposals.fork_batch(3)
    result = native.commit(exp.fork_batch(3), forked,
                           native.choose(forked, override=torch.arange(3).repeat_interleave(batch)))
    ticks = torch.tensor([0, 3, 1, 7]).repeat(3)
    waited = native.wait(result.state, ticks)
    assert waited.graph is result.state.graph
    for index, delay in enumerate(ticks.tolist()):
        individual = native.wait(result.state.subset(index), delay)
        torch.testing.assert_close(waited.hidden[index], individual.hidden[0])
    equiv = native.wait(result.state, ticks, max_ticks=9)
    torch.testing.assert_close(waited.hidden, equiv.hidden)
    with pytest.raises(ValueError, match="exceeds max_ticks"):
        native.wait(result.state, ticks, max_ticks=6)
    assert not torch.equal(waited.hidden, result.state.hidden)
    replay = native.act(waited, right.repeat(3, 1), left.repeat(3, 1), draw.repeat(3))
    assert replay.state.graph is waited.graph
    assert replay.side.shape == (3 * batch,)


def test_choice_score_is_independent_of_benefit_and_all_heads_reach_graph_parameters():
    root, state, left, right, outcome, draw, draws = sample_experience(differentiable=True)
    exp = native.observe(state, left, right, outcome, draw)
    proposals = native.propose(exp, draws)
    decision = native.choose(proposals, draw)
    altered = replace(proposals, benefit=1000 * (1 - proposals.benefit))
    same = native.choose(altered, draw)
    torch.testing.assert_close(decision.index, same.index)
    torch.testing.assert_close(decision.log_prob, same.log_prob)

    result = native.commit(exp, proposals, decision)
    post = native.wait(result.state, [1, 3, 2, 0])
    revisit = native.act(post, right, left, draw)
    losses = (proposals.benefit.square().sum(), proposals.effect_prediction.square().sum(),
              decision.log_prob.sum(), (proposals.address_log_prob + proposals.op_log_prob).sum(),
              revisit.motor_log_prob.sum())
    for loss in losses:
        grad_w, grad_b = torch.autograd.grad(loss, (root.weights, root.bias), retain_graph=True)
        for grad in (grad_w, grad_b):
            assert torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0
        assert torch.count_nonzero(grad_w[~root.present]) == 0
