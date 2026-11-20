from dataclasses import fields

import numpy as np
import torch

from experiments.pulse_game import interleave_joint_continue as joint, interleave_train as train


def test_short_lives_preserve_pairing_feedback_and_all_study_requirements():
    tape = train.make_tape((731, 732), train.Config(pairs=8), 32)
    for mode in joint.MODES[:-1]:
        sampled = joint.short_life(tape, mode, np.random.default_rng(9))
        visits = 4 if mode == "joint4" else 3
        assert sampled.left.shape[3] == visits
        assert torch.equal(sampled.draws, tape.draws[..., :visits, :])
        assert torch.equal(sampled.waits, tape.waits[..., :visits])
        assert torch.equal(sampled.law, tape.law)
        assert torch.equal(sampled.objective_weight.sum(-1), torch.ones(sampled.batch_shape))
        assert torch.equal(sampled.right_factor[:, :, 0], -sampled.right_factor[:, :, 1])
        for context, probe, menus in zip(sampled.context.reshape(-1, visits), sampled.probe.reshape(-1, visits),
                                         sampled.left.reshape(-1, visits, 3), strict=True):
            for target in (probe >= 0).nonzero().flatten().tolist():
                studied = ((probe[:target] == -1) & (context[:target] == context[target])).nonzero().flatten()
                assert len(studied)
            if mode == "context3":
                assert sorted(context[:2].tolist()) == [0, 1]
                assert int(probe[-1]) in (0, 1)
            if mode == "mark3":
                assert torch.equal(menus[0], -menus[1])
                assert int(probe[-1]) == 3


def test_last_half_replays_unmodified_primary_world_at_same_coordinates():
    config = train.Config(updates=8000, pairs=4)
    for step in (4000, 7000, 8000):
        tape, mode = joint.world((731,), config, step)
        exact = train.make_tape((731,), config, step)
        assert mode == "full14"
        for field in fields(train.Tape):
            assert torch.equal(getattr(tape, field.name), getattr(exact, field.name))


def test_short_modes_score_physical_choices_with_a_gradient_to_local_rule():
    config = train.Config(pairs=4, max_wait=0)
    tape = train.make_tape((731,), config, 38)
    for mode in joint.MODES[:-1]:
        graph = train.local.spawn(731)
        rule = train.local.Rule(731)
        sampled = joint.short_life(tape, mode, np.random.default_rng(5))
        rolled = joint._short_rollout(graph, rule, sampled, config.max_wait)
        assert torch.equal(rolled.resource, rolled.side * sampled.right_factor + sampled.fixed_factor)
        loss, values = joint.training_step(graph, rule, sampled, mode, config)
        loss.backward()
        assert values[0, 1].isfinite()
        assert all(parameter.grad.isfinite().all() for parameter in rule.parameters())
