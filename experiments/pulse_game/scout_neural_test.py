import numpy as np
import pytest
import torch

from experiments.pulse_game.scout_neural import INPUT_SIZE, NeuralScoutConfig, ReceiptFeatures, ScoutPulseNet
from experiments.pulse_game.scout_world import SCOUT_OBS_SIZE
from experiments.pulse_game.world import CONTACT_ID, CTX, CUE, REWARD_POS, TRIAL_START


@pytest.mark.parametrize("symmetry", ("none", "physical", "physical_context"))
def test_recurrent_sequence_matches_stepwise_execution(symmetry):
    generator = torch.Generator().manual_seed(802)
    model = ScoutPulseNet(NeuralScoutConfig(hidden=12, degree=3, symmetry=symmetry))
    inputs = torch.randn(5, 12, INPUT_SIZE, generator=generator)
    state = model.initial(5)
    expected, _ = model.sequence(inputs, state)
    outputs = []
    for frame in inputs.unbind(dim=1):
        logits, state = model(frame, state)
        outputs.append(logits)
    torch.testing.assert_close(expected, torch.stack(outputs, dim=1))


def test_body_equivariance_and_context_relabeling_preserve_predictions():
    generator = torch.Generator().manual_seed(803)
    model = ScoutPulseNet(NeuralScoutConfig(hidden=12, degree=3, symmetry="physical_context"))
    x = torch.randn(5, 12, INPUT_SIZE, generator=generator)
    reference, reference_values = model.sequence(x, model.initial(5))
    for index in range(4):
        relabeled = x[..., model._input_orders[index]] * model._input_signs[index]
        logits, values = model.sequence(relabeled, model.initial(5))
        torch.testing.assert_close(reference, logits[..., model._outputs[index]])
        torch.testing.assert_close(reference_values, values)


def test_public_reward_and_contact_identity_are_both_required_by_fast_writer():
    features = ReceiptFeatures(2, seed=77)
    obs = np.zeros((2, SCOUT_OBS_SIZE), dtype=np.float32)
    obs[:, CUE.start] = 1
    obs[:, TRIAL_START] = 1
    readout = slice(SCOUT_OBS_SIZE, SCOUT_OBS_SIZE + 4)
    np.testing.assert_array_equal(features.observe(obs)[:, readout], 0)

    obs[:, TRIAL_START] = 0
    obs[:, CTX.start] = 1
    np.testing.assert_array_equal(features.observe(obs)[:, readout], 0)
    obs[:, CTX] = 0
    obs[0, REWARD_POS] = 1
    obs[1, CONTACT_ID.start] = 1
    np.testing.assert_array_equal(features.observe(obs)[:, readout], 0)

    obs[0, CONTACT_ID.start] = 1
    read = features.observe(obs)[:, readout]
    assert np.all(read[0, :2] > 0)
    assert np.all(read[0, 2:] < 0)
    np.testing.assert_array_equal(read[1], 0)
