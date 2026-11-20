from dataclasses import replace

import numpy as np
import torch

from experiments.pulse_game import interleave_quest_eval as evaluation, interleave_quest_train as quest
from experiments.pulse_game import local_trace_graph as local, world


def test_quest_world_and_law_crossing_are_exogenous():
    births = (811, 812)
    config = quest.Config(worlds=5)
    train = quest.make_tape(births, config, 1)
    novel = quest.make_tape(births, config, 1, domain=quest.VALIDATION_DOMAIN, novel=True, shared=True)
    assert len(quest.ORDERS) == 720
    assert not (train.order[..., :2] >= quest.MARK_PLUS).all(dim=-1).any()
    assert (novel.order[..., :2] >= quest.MARK_PLUS).all()
    np.testing.assert_array_equal(train.law[0, 0], quest.LAWS)
    assert torch.equal(train.left[:, :, 0], train.left[:, :, 7])
    assert torch.equal(train.draws[:, :, 0], train.draws[:, :, 7])
    assert not torch.equal(train.left[0], novel.left[0])
    assert torch.equal(novel.left[0], novel.left[1])
    assert torch.equal(novel.quest.sort(-1).values[0, 0, 0], torch.tensor([quest.A, quest.B, quest.MARK_CHOICE]))
    assert set(train.order[0, 0, 0, :quest.SUPPORT].tolist()) >= {quest.A, quest.B, quest.MARK_PLUS, quest.MARK_MINUS}


def test_contact_ticks_match_the_physical_world_for_both_actions():
    physical = world.make_tape((31, 43, 69, 80), trials=12, distance_jitter=.34, wind_scale=.11)
    expected = quest.travel_ticks(physical.distances, physical.wind)
    for side in (0, 1):
        sim = world.PulseWorld(physical, feedback="none")
        pulses = np.zeros((physical.n, 2), bool)
        pulses[:, side] = True
        observation = sim.observe()
        while observation is not None:
            observation = sim.step(pulses)
        result = sim.result()
        assert np.all(result.contact_side == side)
        np.testing.assert_array_equal(result.contact_tick, expected[..., side])


def test_hidden_terminal_rewards_cannot_cause_any_of_the_serial_actions():
    torch.set_num_threads(1)
    tape = quest.make_tape((731,), quest.Config(worlds=2), 1)
    graph, rule = local.spawn(731), local.Rule(731)
    changed = tape.resources.clone()
    changed[..., quest.SUPPORT:, :] *= -1
    with torch.no_grad():
        intact = quest.play(graph, rule, tape, 8)
        reversed_truth = quest.play(graph, rule, replace(tape, resources=changed), 8)
    np.testing.assert_array_equal(intact.side, reversed_truth.side)
    np.testing.assert_array_equal(intact.taste, reversed_truth.taste)
    np.testing.assert_array_equal(intact.edit_score, reversed_truth.edit_score)
    assert not intact.taste[..., quest.SUPPORT:].any()
    assert torch.equal(intact.resource[..., quest.SUPPORT:], -reversed_truth.resource[..., quest.SUPPORT:])


def test_outer_objective_uses_the_three_earned_resources_only():
    torch.set_num_threads(1)
    tape = quest.make_tape((731,), quest.Config(worlds=3), 1)
    graph, rule = local.spawn(731), local.Rule(731)
    record = quest.play(graph, rule, tape, 8)
    change_support = record.resource.clone()
    change_support[..., :quest.SUPPORT] *= -1
    for objective in quest.ARMS:
        before, _ = quest.score(record, tape, objective)
        after, _ = quest.score(replace(record, resource=change_support), tape, objective)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    joint, _ = quest.score(record, tape, "conjunction")
    mean, _ = quest.score(record, tape, "mean")
    assert not torch.isclose(joint, mean)
    (joint + mean).backward()
    assert all(value.grad is not None for value in rule.parameters())


def test_native_public_solver_and_constant_reference():
    config = quest.Config(worlds=8)
    tape = quest.make_tape((811, 812), config, 1, domain=quest.VALIDATION_DOMAIN, novel=True, shared=True)
    results = {mode: evaluation.native_payload(evaluation.native_reference(tape, mode), tape)
               for mode in ("uniform", "constant_left", "public_lookup")}
    assert results["public_lookup"]["success/both_modalities"].all()
    for mode in ("uniform", "constant_left"):
        assert results[mode]["success/both_modalities"].mean() == .125
        assert not results[mode]["taste"][..., quest.SUPPORT:].any()
