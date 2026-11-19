import pytest
import torch

from experiments.pulse_game.scout_neural import NeuralScoutConfig
from experiments.pulse_game.twostep_flash_control import (
    compare_states, equal_cut_state, equal_dose_state, match_acquisition_and_visits, scheduled_corpus,
)
from experiments.pulse_game.twostep_latent import LearnedMemory
from experiments.pulse_game.twostep_neural import INPUT_SIZE, TwoStepPulseNet
from experiments.pulse_game.twostep_neural_train import Corpus
from experiments.pulse_game.twostep_reward import discounted_life_returns, terminal_returns
from experiments.pulse_game.twostep_world import EPISODE_TICKS, OBS_SIZE, make_tape
from experiments.pulse_game.world import ACTIVE


@pytest.mark.parametrize("symmetry", ("none", "physical_context", "full"))
def test_sparse_forward_matches_training_sequence(symmetry):
    net = TwoStepPulseNet(NeuralScoutConfig(hidden=16, degree=4, symmetry=symmetry))
    g = torch.Generator().manual_seed(1272)
    frames = torch.randn(4, 10, INPUT_SIZE, generator=g)
    state = net.initial(4)
    reference, _ = net.sequence(frames, state)
    output = []
    for present in frames.unbind(1):
        logits, state = net(present, state)
        output.append(logits)
    torch.testing.assert_close(reference, torch.stack(output, dim=1))


def test_two_step_action_and_label_permutations_are_exact():
    net = TwoStepPulseNet(NeuralScoutConfig(hidden=16, degree=4, symmetry="full"))
    g = torch.Generator().manual_seed(1273)
    inputs = torch.randn(4, 10, INPUT_SIZE, generator=g)
    reference, value = net.sequence(inputs, net.initial(4))
    for group in range(8):
        changed = inputs[..., net._input_orders[group]] * net._input_signs[group]
        output, output_value = net.sequence(changed, net.initial(4))
        torch.testing.assert_close(reference, output[..., net._outputs[group]])
        torch.testing.assert_close(value, output_value)


def test_learned_banks_write_only_with_the_associated_physical_event():
    memory = LearnedMemory()
    bank = memory.prepare()
    empty = memory.initial(2)
    entry = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    room = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    fruit = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    sign = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    used = memory.update(empty, bank, entry, room, fruit, sign)
    assert torch.count_nonzero(used.routes[0]) > 0
    assert torch.count_nonzero(used.routes[1]) == 0
    assert torch.count_nonzero(used.outcomes[0]) == 0
    assert torch.count_nonzero(used.outcomes[1]) > 0
    zero_room = torch.zeros_like(room)
    masked = memory.update(used, bank, entry, zero_room, fruit, sign)
    torch.testing.assert_close(masked.routes, used.routes)
    torch.testing.assert_close(masked.outcomes, used.outcomes)


def test_later_recall_loss_reaches_the_earlier_event_encoder():
    model = LearnedMemory()
    associations = model.prepare()
    start = model.initial(2)
    identities = torch.eye(2)
    linked = model.update(start, associations, identities, identities.flip(0), identities, identities)
    combined = model.view(linked, associations)
    loss = combined.square().sum()
    loss.backward()
    assert model.entrance_keys.weight.grad.norm() > 0
    assert model.room_encoder.weight.grad.norm() > 0
    assert model.sign_encoder.weight.grad.norm() > 0


def test_equal_dose_state_controls_preserve_the_intended_radii():
    rng = torch.Generator().manual_seed(90057)
    left = torch.randn(8, 4, 12, generator=rng)
    right = torch.randn(8, 4, 12, generator=rng)
    against_swap = equal_dose_state(left, right, rng)
    torch.testing.assert_close((against_swap - left).flatten(1).norm(dim=1), (right - left).flatten(1).norm(dim=1))
    torch.testing.assert_close(against_swap.flatten(1).norm(dim=1), right.flatten(1).norm(dim=1))
    against_cut = equal_cut_state(left, rng)
    torch.testing.assert_close((against_cut - left).flatten(1).norm(dim=1), left.flatten(1).norm(dim=1))
    torch.testing.assert_close(against_cut.flatten(1).norm(dim=1), left.flatten(1).norm(dim=1))


def test_flash_dataset_supplies_recall_before_movement_and_masks_it_during_motion():
    sensory = torch.ones(24, EPISODE_TICKS, INPUT_SIZE)
    actions = torch.ones(24, EPISODE_TICKS, dtype=torch.int64)
    weights = torch.ones(24, EPISODE_TICKS)
    base = Corpus(sensory, actions, weights)
    flashed = scheduled_corpus(base, seed=39010, protocol="flash")
    phase = flashed.sensory.reshape(24, 2, 49, INPUT_SIZE)
    movement = phase[..., ACTIVE] > 0
    recalled = phase[..., OBS_SIZE:].abs().sum(dim=-1) > 0
    assert not (movement & recalled).any()
    torch.testing.assert_close(recalled.sum(dim=-1), torch.ones(24, 2, dtype=torch.long))
    offsets = recalled.float().argmax(dim=-1)
    assert set(torch.unique(offsets).tolist()) == {0, 16, 24}


def test_replaying_controlled_probes_does_not_change_their_shared_prefix():
    actor = TwoStepPulseNet(NeuralScoutConfig(hidden=8, degree=3, symmetry="physical"))
    tape = make_tape([57, 58], acquisition=2, visits=4)
    start = match_acquisition_and_visits(actor, tape, delay=8, hold=False, seed=9)
    before, _ = compare_states(start, seed=9, mode="intact")
    compare_states(start, seed=9, mode="equal_dose")
    after, _ = compare_states(start, seed=9, mode="intact")
    assert before == after
    assert start.worlds[0].episode_index == tape.probe
    assert start.worlds[0].tick_index == 0


def test_strict_terminal_return_only_uses_shown_endpoint_outcomes():
    displayed = torch.zeros(3, EPISODE_TICKS, dtype=torch.int8).numpy()
    displayed[1, 18] = -1
    displayed[2, 29] = 1
    assert terminal_returns(displayed, only_contact=True).tolist() == [0.0, -1.0, 1.0]
    assert terminal_returns(displayed, positive_only=True).tolist() == [0.0, 0.0, 1.0]
    assert terminal_returns(displayed).tolist() == [-1.0, -1.0, 1.0]


def test_life_return_only_adds_discounted_actual_future_receipts():
    recorded = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]]).numpy()
    assert discounted_life_returns(recorded, 0.5).tolist() == [[0.375, -0.25, -0.5], [0.0, 0.0, 0.0]]
    assert discounted_life_returns(recorded, 0).tolist() == recorded.tolist()
