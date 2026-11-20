from dataclasses import replace
from itertools import groupby
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.pulse_game import interleave_train as tr


def test_tape_balances_both_modalities_without_exposing_laws():
    tape = tr.make_tape((731, 732), tr.Config(pairs=32), 517)
    replay = tr.make_tape((731, 732), tr.Config(pairs=32), 517)
    for field in tape.__dataclass_fields__:
        assert torch.equal(getattr(tape, field), getattr(replay, field))
    for value in (tape.left, tape.right, tape.draws, tape.waits, tape.context, tape.probe):
        assert torch.equal(value[:, :, 0], value[:, :, 1])
    for value in (tape.right_factor, tape.fixed_factor, tape.law):
        assert torch.equal(value[:, :, 0], -value[:, :, 1])
    assert torch.allclose(tape.objective_weight.sum(dim=-1), torch.ones(tape.batch_shape))
    assert torch.allclose((tape.objective_weight * (tape.context == tr.MARK_CONTEXT)).sum(dim=-1),
                          torch.full(tape.batch_shape, .5))
    assert set(tape.waits.unique().tolist()) == set(range(9))
    assert (tape.right_factor != tape.donor_right_factor).any()

    distinct_layouts = set()
    for context, probe, left, right in zip(tape.context[:, :, 0].flatten(0, 1),
                                            tape.probe[:, :, 0].flatten(0, 1),
                                            tape.left[:, :, 0].flatten(0, 1),
                                            tape.right[:, :, 0].flatten(0, 1), strict=True):
        odor = context != tr.MARK_CONTEXT
        layout = tuple(odor.tolist())
        distinct_layouts.add(layout)
        assert sum(layout) == 7 and len(layout) == 14
        assert max(len(list(run)) for _, run in groupby(layout)) <= 3
        assert sorted(context[odor][:3].tolist()) == [0, 1, 2]
        assert (probe[odor][:3] == -1).all()
        assert sorted(probe[odor][3:].unique().tolist()) == [0, 1, 2]
        ambient = (left[odor] + right[odor]) / 2
        assert len(ambient.round(decimals=4).unique(dim=0)) == 3
        delta = (right[odor] - left[odor]) / 2
        early_ab = delta[(context[odor] < 2).nonzero().flatten()[:2]]
        assert torch.allclose(early_ab[0].abs(), early_ab[1].abs())

        mark_left, mark_right, mark_probe = left[~odor], right[~odor], probe[~odor]
        first, second = ((mark_probe == label).nonzero().item() for label in (3, 4))
        for start, stop in ((0, first), (first + 1, second)):
            assert torch.equal(mark_left[start:stop], mark_right[start:stop])
            signs = mark_left[start:stop, 0].sign()
            assert set(signs.tolist()) == {-1., 1.}
        assert torch.equal(mark_left[[first, second]], -mark_right[[first, second]])
    assert len(distinct_layouts) > 40


def test_independent_controller_never_observes_true_or_future_feedback():
    config = tr.Config(pairs=2, max_wait=3)
    tape = tr.make_tape((731,), config, 18)
    graph, rule = tr.local.spawn(731), tr.local.Rule(731)
    negated = replace(tape, right_factor=-tape.right_factor, fixed_factor=-tape.fixed_factor)
    altered_future = tape.fixed_factor.clone()
    altered_future[..., -1] = -altered_future[..., -1]
    altered_side = tape.right_factor.clone()
    altered_side[..., -1] = -altered_side[..., -1]
    future = replace(tape, fixed_factor=altered_future, right_factor=altered_side)
    with torch.no_grad():
        own = tr.rollout(graph, rule, tape, "local_active", config.max_wait)
        other = tr.rollout(graph, rule, future, "local_active", config.max_wait)
        control = tr.rollout(graph, rule, tape, "local_independent", config.max_wait)
        changed_truth = tr.rollout(graph, rule, negated, "local_independent", config.max_wait)
    assert torch.equal(own.observed, own.resource)
    assert torch.equal(own.observed[..., :-1], other.observed[..., :-1])
    assert torch.equal(own.side, other.side)
    assert torch.equal(own.resource[..., -1], -other.resource[..., -1])
    for name in ("side", "observed", "motor_score", "edit_score", "numeric_contacts"):
        assert torch.equal(getattr(control, name), getattr(changed_truth, name))
    assert torch.equal(control.resource, -changed_truth.resource)
    assert torch.equal(control.final_state.graph.weights, changed_truth.final_state.graph.weights)
    assert (control.numeric_contacts > 0).all()
    assert not torch.equal(own.final_state.graph.weights, graph.weights.expand_as(own.final_state.graph.weights))


def test_score_credits_edits_with_only_strictly_later_return():
    resource = torch.tensor([[[[1., 1.], [1., -1.]], [[-1., -1.], [-1., 1.]]]])
    objective = torch.tensor([0., 1.])
    reward_to_go = tr.causal_advantage(resource, objective)
    assert torch.equal(reward_to_go[..., 0], resource[..., 1])
    assert torch.equal(reward_to_go[..., 1], resource[..., 1])
    config = tr.Config(pairs=2, max_wait=0)
    tape = tr.make_tape((731,), config, 80)
    graph, rule = tr.local.spawn(731), tr.local.Rule(731)
    rolled = tr.rollout(graph, rule, tape, "local_active", config.max_wait)
    edit_score = rolled.edit_score.detach().requires_grad_()
    loss, _ = tr.score_and_measure(replace(rolled, edit_score=edit_score), tape)
    edit_gradient, = torch.autograd.grad(loss, edit_score)
    assert torch.any(edit_gradient[..., :-1] != 0)
    assert torch.all(edit_gradient[..., -1] == 0)


def test_interrupted_fit_restores_adam_graph_and_world_sequence(tmp_path):
    config = tr.Config(updates=4, pairs=2, max_wait=2, checkpoint_every=2)
    birth, arm = 733, "local_active"
    interrupted, uninterrupted = tmp_path / "interrupted", tmp_path / "uninterrupted"
    sample = tr.make_tape

    def stop_at_third_sample(births, config, step):
        if step == 3:
            raise InterruptedError("synthetic interruption")
        return sample(births, config, step)

    with patch.object(tr, "make_tape", side_effect=stop_at_third_sample):
        with pytest.raises(InterruptedError):
            tr.run(interrupted, config, (birth,), (arm,), workers=1, progress=0)
    checkpoints = sorted((interrupted / arm).glob("step-*.pt"))
    assert [file.name for file in checkpoints] == ["step-000000.pt", "step-000002.pt"]
    tr.run(interrupted, config, (birth,), (arm,), workers=1, progress=0)
    tr.run(uninterrupted, config, (birth,), (arm,), workers=1, progress=0)
    loaded = [torch.load(tr.checkpoint_path(root, arm, config.updates), weights_only=True)
              for root in (interrupted, uninterrupted)]
    for component in ("graph", "rule"):
        for key in loaded[0][component]:
            assert torch.equal(loaded[0][component][key], loaded[1][component][key])
    for key, left in loaded[0]["optimizer"]["state"].items():
        for field, value in left.items():
            assert torch.equal(value, loaded[1]["optimizer"]["state"][key][field])
    np.testing.assert_array_equal(loaded[0]["metrics"].numpy(), loaded[1]["metrics"].numpy())
    graph, rule = tr.load_checkpoint(interrupted, arm, birth, config.updates)
    assert graph.weights.shape == (1, 20, 20)
    assert all(not parameter.requires_grad for parameter in rule.parameters())
    with pytest.raises(ValueError, match="Immutable run protocol"):
        tr.run(interrupted, replace(config, world_seed=12), (birth,), (arm,), workers=1, progress=0)
