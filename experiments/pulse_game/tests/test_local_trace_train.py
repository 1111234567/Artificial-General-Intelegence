from dataclasses import replace
from itertools import product

import torch
from torch.nn import functional as F

from experiments.pulse_game import local_trace_train as train


def test_sampled_terminal_resource_gradient_matches_exact_binary_expectation():
    logit = torch.tensor([[[.41, -.1], [.87, 1.1]]], dtype=torch.float64, requires_grad=True)
    factors = torch.tensor([[[1, -1], [-1, 1]]])
    estimated = 0
    for choices in product((-1, 1), repeat=4):
        side = torch.tensor(choices, dtype=logit.dtype).reshape_as(logit)
        sampled_score = F.logsigmoid(side * logit)
        sampled_resource = side * factors
        weight = sampled_score.detach().sum().exp()
        advantage = train.leave_pair_out_advantage(sampled_resource)
        estimated += weight * train.score_loss(sampled_score, advantage).sum()
    expected_utility = ((2 * logit.sigmoid() - 1) * factors).mean()
    estimate_grad = torch.autograd.grad(estimated, logit)[0]
    utility_grad = torch.autograd.grad(expected_utility, logit)[0]
    torch.testing.assert_close(estimate_grad, -utility_grad)


def test_a_pair_baseline_excludes_both_twins_and_has_no_reward_derivative():
    rewards = torch.tensor([[[1., -1.], [1., 1.], [-1., -1.]]], requires_grad=True)
    advantage = train.leave_pair_out_advantage(rewards)
    assert not advantage.requires_grad
    torch.testing.assert_close(advantage, torch.tensor([[[1., -1.], [1.5, 1.5], [-1.5, -1.5]]]))


def test_feeder_matches_archival_physical_demonstrations():
    config = train.Config(updates=1, pairs=16)
    tape = train.world_for_update(config, "feeder", 731, 1)
    torch.testing.assert_close(tape.left[:, :, :2], tape.right[:, :, :2])
    torch.testing.assert_close(tape.right[:, :, 0], -tape.right[:, :, 1])
    torch.testing.assert_close(tape.left[:, :, 2], -tape.right[:, :, 2])
    torch.testing.assert_close(tape.support_factor, tape.right[:, :, :2, 0].sign() * tape.law[..., None])
    torch.testing.assert_close(tape.terminal_factor, tape.right[:, :, -1, 0].sign() * tape.law)
    assert (tape.wait == 0).all()
    assert (tape.donor_factor != tape.support_factor).any()
    assert (tape.donor_factor == tape.support_factor).any()


def test_unseen_terminal_law_has_no_effect_on_trajectory_before_outcome():
    torch.set_num_threads(1)
    config = train.Config(updates=1, pairs=3, max_wait=2)
    tape = train.world_for_update(config, "odor", 731, 1)
    graph, rule = train.native.spawn(731), train.native.Rule(731)
    initial = train.rollout(graph, rule, tape, "odor", "active", config)
    changed = train.rollout(graph, rule, replace(tape, law=-tape.law, terminal_factor=-tape.terminal_factor),
                            "odor", "active", config)
    torch.testing.assert_close(initial.terminal_side, changed.terminal_side)
    torch.testing.assert_close(initial.terminal_logit, changed.terminal_logit)
    torch.testing.assert_close(initial.support_motor_score, changed.support_motor_score)
    torch.testing.assert_close(initial.edit_score, changed.edit_score)
    torch.testing.assert_close(initial.terminal_resource, -changed.terminal_resource)


def test_zero_early_feedback_cancels_all_score_directions_in_opposite_law_twins():
    torch.set_num_threads(1)
    config = train.Config(pairs=3, max_wait=2)
    for task in train.TASKS:
        tape = train.world_for_update(config, task, 731, 1)
        blank = replace(tape, support_factor=torch.zeros_like(tape.support_factor))
        graph = train.native.spawn(731)
        rule = train.native.Rule(731)
        loss, metrics = train.training_step(graph, rule, blank, task, "active", config)
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        assert metrics[0, 1] == .5
        for gradient in torch.autograd.grad(loss, tuple(rule.parameters()), allow_unused=True):
            if gradient is not None:
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)


def test_grouped_fit_checkpoints_round_trip_for_both_controllers(tmp_path):
    torch.set_num_threads(1)
    for controller in train.CONTROLLERS:
        config = train.Config(updates=2, pairs=3, max_wait=1, controller=controller)
        output = tmp_path / controller
        train.fit(output, config, (731, 732), "odor", "active", (0, 2), 0)
        for birth in (731, 732):
            initial, initial_rule = train.load_checkpoint(output, "odor", "active", birth, 0)
            trained, trained_rule = train.load_checkpoint(output, "odor", "active", birth, 2)
            torch.testing.assert_close(initial.present, trained.present)
            assert not torch.equal(initial.weights, trained.weights)
            if controller == "legacy":
                assert trained_rule is None
            else:
                assert not any(parameter.requires_grad for parameter in trained_rule.parameters())
                assert any(not torch.equal(before, after)
                           for before, after in zip(initial_rule.parameters(), trained_rule.parameters(), strict=True))
