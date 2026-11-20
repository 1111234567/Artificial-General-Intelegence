"""Verify the physical timings, paired branches, and surgical bridge controls."""

from dataclasses import fields, replace

import torch

from experiments.pulse_game import fork_lesson_world as world
from experiments.pulse_game import native_rehearsal_trial as trial


class RecordedBody(world.ForkGraph):
    def __init__(self):
        super().__init__(731)
        self.pulses = []

    def tick(self, s, x, **kwargs):
        self.pulses.append(x.clone())
        return super().tick(s, x, **kwargs)


@torch.no_grad()
def test_support_reaches_both_objects_before_exposing_either_private_value():
    torch.set_num_threads(1)
    model = RecordedBody()
    episode = world.scenes(357, 24)
    support, visits = world.support(model, episode, return_receipts=True)
    presented = torch.stack(model.pulses)
    assert len(presented) == world.SUPPORT_TICKS
    rows = torch.arange(len(episode.first))
    assert torch.equal(visits[0].object_id, episode.first)
    assert torch.equal(visits[1].object_id, 1 - episode.first)
    for rank, visit in enumerate(visits):
        begin = rank * (world.SUPPORT_TICKS // 2) + world.SUPPORT_GAP_TICKS
        assert torch.equal(presented[begin:begin + len(visit.inputs)], visit.inputs)
        assert torch.equal(visit.cue, episode.vectors[rows, visit.object_id])
        assert torch.equal(visit.side, episode.approach_sides[:, rank])
        assert torch.equal(visit.payoff, episode.values[rows, visit.object_id])
        assert torch.equal(visit.taste[:, 0] - visit.taste[:, 1], visit.payoff)
        until_contact = world.APPROACH_TICKS + world.TRAVERSE_TICKS - 1
        assert not visit.inputs[:until_contact, :, 6:10].any()
        assert torch.equal(visit.inputs[until_contact, :, 6:8].argmax(1), visit.side)
        assert not visit.inputs[until_contact, :, 8:10].any()
        assert torch.equal(visit.inputs[until_contact + 1, :, 8:10], visit.taste)

    poison = replace(world.swap_doors(episode), draws=1 - episode.draws, flips=-episode.flips)
    new_support = world.support(model, poison)
    for field in fields(trial.State):
        assert torch.equal(getattr(support, field.name), getattr(new_support, field.name))
    changed = world.support(model, replace(episode, values=-episode.values))
    assert not torch.equal(support.h, changed.h)


@torch.no_grad()
def test_only_the_traversed_door_settles_after_the_same_zero_reward_delay():
    torch.set_num_threads(1)
    model = RecordedBody()
    episode = world.scenes(2743, 128)
    support = world.support(model, episode)
    model.pulses.clear()
    cue = world.cue(model, support, episode)
    assert len(model.pulses) == world.CUE_TICKS
    for pulse in model.pulses:
        assert torch.equal(pulse, world.door_cues(episode))
        assert not pulse[:, 8:10].any()
    model.pulses.clear()
    ready, activities, _ = world.choice_window(model, cue)
    assert len(model.pulses) == world.BLANK_TICKS
    assert not torch.stack(model.pulses).any()
    assert torch.equal(model.forecast(ready), activities[-1, :, 12:14])
    paid = []
    for side in (0, 1):
        model.pulses.clear()
        choice = world.choose(model, ready, episode, side=side)
        assert not choice.immediate.any()
        assert (choice.side == side).all()
        _, outcome = world.commit(model, ready, episode, choice)
        pulses = torch.stack(model.pulses)
        assert len(pulses) == 1 + world.DELAY_TICKS + world.TASTE_TICKS
        assert (pulses[0, :, 6 + side] == 1).all()
        assert not pulses[:1 + world.DELAY_TICKS, :, 8:10].any()
        assert not pulses[1:1 + world.DELAY_TICKS].any()
        assert torch.equal(pulses[-world.TASTE_TICKS, :, 8:10], outcome.taste)
        assert torch.equal(outcome.taste[:, 0] - outcome.taste[:, 1], outcome.payoff)
        assert torch.equal(choice.score, outcome.score)
        paid.append(outcome.payoff)
    actual = torch.stack(paid, dim=1)
    assert torch.equal(actual, world.oracle(episode))
    assert torch.equal(actual.flip(1), world.oracle(world.swap_doors(episode)))
    assert set(actual.unique().tolist()) == {-1., 1.}
    assert (actual[:, 0] == actual[:, 1]).any()
    assert (actual[:, 0] > actual[:, 1]).any()
    assert (actual[:, 0] < actual[:, 1]).any()


@torch.no_grad()
def test_bridge_reexpression_preserves_parent_and_same_history_replacement_is_exact():
    torch.set_num_threads(1)
    model = world.ForkGraph(731)
    parent = trial.TrialGraph(731, "trial")
    episode = world.scenes(39127, 8)
    support = world.support(model, episode)
    state = world.cue(model, support, episode)
    pulse = world.door_cues(episode)
    actual, actual_back = model.tick(state, pulse)
    expected, expected_back = parent.tick(state, pulse)
    for field in fields(trial.State):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), rtol=1e-5, atol=1e-6)
    assert torch.equal(actual_back, expected_back)

    natural, activity, back, outward = world.choice_window(model, state, return_outgoing=True)
    replacement, replaced_activity, replaced_back, replaced_outward = world.choice_window(
        model, state, override=back, outbound_override=outward, return_outgoing=True,
    )
    for field in fields(trial.State):
        assert torch.equal(getattr(natural, field.name), getattr(replacement, field.name))
    assert torch.equal(activity, replaced_activity)
    assert torch.equal(back, replaced_back)
    assert torch.equal(outward, replaced_outward)

    _, no_back, silent_back, _ = world.choice_window(model, state, cut=True, return_outgoing=True)
    _, no_out, _, silent_out = world.choice_window(model, state, cut_outbound=True, return_outgoing=True)
    assert not silent_back.any()
    assert not silent_out.any()
    assert torch.equal(no_back[0, :, trial.REGION:], activity[0, :, trial.REGION:])
    assert torch.equal(no_out[0, :, :trial.REGION], activity[0, :, :trial.REGION])
    assert not torch.equal(no_back[0, :, :trial.REGION], activity[0, :, :trial.REGION])
    assert not torch.equal(no_out[0, :, trial.REGION:], activity[0, :, trial.REGION:])
