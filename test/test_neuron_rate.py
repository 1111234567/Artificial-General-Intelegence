import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.dynamics.model import DynamicsModel
from experiments.pulse_game.scout_neural import INPUT_SIZE, NeuralScoutConfig, ScoutPulseNet
from src.neurons.rate import PulseRateRNN, TanhRNN, projected_tanh_step
from src.neurons.rate_numpy import NumpyElman2
from src.neurons.training import train_chunk


@pytest.mark.parametrize("leak,learned", [(1., False), (.4, False), (.65, True)])
def test_tanh_rollout_matches_steps_and_carries_gradient(leak, learned):
    torch.manual_seed(42)
    model = TanhRNN(3, 5, leak=leak, learnable_leak=learned, dtype=torch.float64)
    inputs = torch.randn(7, 2, 3, dtype=torch.float64, requires_grad=True)
    initial = model.initial_state(2)
    output, state = model(inputs, initial)
    reference, sequential, stepped = [], initial, initial
    gate = model.leak_logit.sigmoid() if learned else leak
    for tick in inputs:
        candidate = (tick @ model.input_projection.weight.T + model.input_projection.bias
                     + sequential @ model.recurrent.T).tanh()
        sequential = candidate if leak == 1. else torch.lerp(sequential, candidate, gate)
        stepped = model.step(tick, stepped)
        reference.append(sequential)
    torch.testing.assert_close(output, torch.stack(reference))
    torch.testing.assert_close(state, sequential)
    torch.testing.assert_close(state, stepped)
    first, carry = model(inputs[:3], initial)
    later, end = model(inputs[3:], carry)
    torch.testing.assert_close(torch.cat((first, later)), output)
    torch.testing.assert_close(end, state)
    later.square().mean().backward()
    assert inputs.grad[:3].abs().sum() > 0
    assert model.recurrent.grad.abs().sum() > 0
    if learned:
        assert model.leak_logit.grad.abs().sum() > 0


def test_projected_tanh_step_agrees_with_scout_mask_and_observation_gate():
    model = ScoutPulseNet(NeuralScoutConfig(hidden=8, degree=3, symmetry="physical_context"))
    inputs = torch.randn(3, INPUT_SIZE)
    state = torch.randn_like(model.initial(3))
    drive, gate = model._observation_drives(inputs)
    recurrent = model.prepare()
    expected = model._step(drive, gate, state, recurrent)
    actual = projected_tanh_step(drive, state, recurrent, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("train_recurrent", [False, True])
def test_elapsed_rate_matches_dynamics_with_same_weights(dtype, train_recurrent):
    torch.manual_seed(42)
    model = PulseRateRNN(3, 5, max_step=.17, train_recurrent=train_recurrent, dtype=dtype)
    original = DynamicsModel(input_size=3, width=5, max_step=.17, kind="learned").to(dtype=dtype)
    with torch.no_grad():
        model.recurrent.copy_(original.recurrent)
        model.encoder.weight.copy_(original.encoder.weight)
    inputs = torch.randn(4, 2, 3, dtype=dtype)
    initial = torch.randn(2, 5, dtype=dtype)
    durations = (0., .04, .27, .8)
    expected, reference = [], initial
    for tick, duration in zip(inputs, durations):
        _, reference = original.step(tick, reference, duration)
        expected.append(reference.tanh())
    activities, end = model(inputs, initial, elapsed=durations)
    torch.testing.assert_close(activities, torch.stack(expected), rtol=0, atol=0)
    torch.testing.assert_close(end, reference, rtol=0, atol=0)
    first, state = model(inputs[:2], initial, elapsed=durations[:2])
    last, end = model(inputs[2:], state, elapsed=durations[2:])
    torch.testing.assert_close(torch.cat((first, last)), activities, rtol=0, atol=0)
    torch.testing.assert_close(end, reference, rtol=0, atol=0)
    activities.square().mean().backward()
    assert model.encoder.weight.grad.abs().sum() > 0
    if train_recurrent:
        assert model.recurrent.grad.abs().sum() > 0
    else:
        assert model.recurrent.grad is None


@pytest.mark.parametrize("family,kwargs", [(TanhRNN, {}), (PulseRateRNN, {"elapsed": .3})])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))])
def test_train_chunk_updates_rate_parameters_and_truncates_between_chunks(family, kwargs, device):
    torch.manual_seed(42)
    model = family(3, 5, device=device)
    head = torch.nn.Linear(5, 2, device=device)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=.01)
    inputs = torch.randn(8, 2, 3, device=device, requires_grad=True)
    targets = torch.randn(8, 2, 2, device=device)
    initial = model.initial_state(2).requires_grad_()
    with torch.no_grad():
        _, reference = model(inputs[:4], initial, **kwargs)
        prior = model.recurrent.clone()
    loss, state = train_chunk(model, optimizer, inputs[:4], initial,
                              lambda output: F.mse_loss(head(output), targets[:4]),
                              max_grad_norm=1., **kwargs)
    assert torch.isfinite(loss) and loss.grad_fn is None and state.grad_fn is None
    assert initial.grad is None and not torch.equal(model.recurrent, prior)
    torch.testing.assert_close(state, reference)
    inputs.grad = None
    _, end = train_chunk(model, optimizer, inputs[4:], state,
                         lambda output: F.mse_loss(head(output), targets[4:]), **kwargs)
    assert inputs.grad[4:].abs().sum() > 0
    assert inputs.grad[:4].count_nonzero() == 0
    assert end.shape == initial.shape and torch.isfinite(end).all()


def test_empty_rate_chunks_keep_state_and_elapsed_alignment():
    inputs = torch.empty(0, 2, 3)
    for model, kwargs in [(TanhRNN(3, 5), {}), (PulseRateRNN(3, 5), {"elapsed": ()})]:
        initial = torch.randn(2, 5, requires_grad=True)
        outputs, carry = model(inputs, initial, **kwargs)
        assert outputs.shape == (0, 2, 5) and carry is initial
    model = PulseRateRNN(3, 5)
    with pytest.raises(ValueError, match="per tick"):
        model(torch.zeros(2, 2, 3), elapsed=(0.,))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        model(torch.zeros(1, 2, 3), elapsed=float("nan"))


def test_numpy_elman_carry_and_masked_bptt_agree_with_reference():
    rng = np.random.default_rng(42)
    model = NumpyElman2(rng, 2, 4, 3)
    inputs = rng.normal(size=(6, 2))
    initial = rng.normal(size=(2, 4))
    logits, h1, h2 = model.rollout(inputs, initial)
    first, h1a, h2a = model.rollout(inputs[:3], initial)
    last, h1b, h2b = model.rollout(inputs[3:], (h1a[-1], h2a[-1]))
    np.testing.assert_array_equal(np.concatenate((first, last)), logits)
    np.testing.assert_array_equal(h1b[-1], h1[-1])
    np.testing.assert_array_equal(h2b[-1], h2[-1])
    mask = [True, False, False, True, True, False]
    upstream = rng.normal(size=logits.shape)
    gradients = model.backward(inputs, h1, h2, upstream, state_mask=mask)
    p = {key: torch.tensor(value, requires_grad=True) for key, value in zip(model.keys(), model.params())}
    ref1, ref2 = torch.from_numpy(initial).unbind()
    outputs = []
    for tick, use_state in zip(torch.from_numpy(inputs), mask):
        ref1 = (p['W1'] @ tick + p['U1'] @ ref1 + p['b1']).tanh()
        ref2 = (p['W2'] @ ref1 + p['U2'] @ ref2 + p['b2']).tanh()
        outputs.append(p['W3'] @ (ref2 if use_state else ref2.detach()) + p['b3'])
    (torch.stack(outputs) * torch.from_numpy(upstream)).sum().backward()
    for key in model.keys():
        np.testing.assert_allclose(gradients[key], p[key].grad.numpy(), rtol=1e-12, atol=1e-12)
