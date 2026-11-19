import pytest
import torch

from .model import DynamicsModel


def test_frozen_and_learned_start_identical_but_only_one_trains_recurrence():
    torch.manual_seed(8)
    frozen = DynamicsModel(width=12, kind="reservoir")
    torch.manual_seed(8)
    learned = DynamicsModel(width=12, kind="learned")
    obs = torch.randn(4, 48)
    state = torch.randn(4, 12)
    a, _ = frozen.step(obs, state, .5)
    b, _ = learned.step(obs, state, .5)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    assert frozen.recurrent.grad is None
    assert learned.recurrent.grad.abs().sum() > 0
    assert frozen.encoder.weight.grad.abs().sum() > 0


def test_silent_interval_has_no_repeated_event_injection():
    model = DynamicsModel(width=8)
    with torch.no_grad():
        model.recurrent.zero_()
    initial = model.initial_state(2)
    pulse = torch.ones(2, 48)
    _, state = model.step(pulse, initial, 0)
    _, after = model.step(torch.zeros_like(pulse), state, 2., max_step=.01)
    torch.testing.assert_close(after, state * torch.exp(torch.tensor(-2.)), rtol=5e-5, atol=2e-6)
    # Dividing a silent interval preserves the same flow up to integration error.
    divided = model.flow(model.flow(state, .7, .01), 1.3, .01)
    torch.testing.assert_close(after, divided)


def test_substeps_do_not_multiply_the_observation_pulse():
    model = DynamicsModel(width=8)
    pulse = torch.randn(3, 48)
    state = model.initial_state(3)
    _, fine = model.step(pulse, state, .25, max_step=.03125)
    _, coarse = model.step(pulse, state, .25, max_step=.25)
    torch.testing.assert_close(fine, coarse)  # autonomous zero remains zero
    torch.testing.assert_close(fine, model.encoder(pulse))


@pytest.mark.parametrize("kind", ["reservoir", "learned", "gru"])
def test_history_changes_later_outputs_and_receives_gradients(kind):
    model = DynamicsModel(width=12, kind=kind)
    cue = torch.randn(2, 48, requires_grad=True)
    _, state = model.step(cue, model.initial_state(2), 0)
    for _ in range(6):
        logits, state = model.step(torch.zeros_like(cue), state, .25)
    logits.square().mean().backward()
    assert cue.grad.abs().sum() > 0
    assert torch.isfinite(state).all()


def test_elapsed_time_matters_and_solver_converges():
    model = DynamicsModel(width=12)
    state = torch.randn(2, 12)
    quick = model.flow(state, .25, .01)
    slow = model.flow(state, 1., .01)
    assert not torch.allclose(quick, slow)
    reference = model.flow(state, 1., .005)
    coarse = model.flow(state, 1., .25)
    fine = model.flow(state, 1., .125)
    assert (fine - reference).square().sum() < (coarse - reference).square().sum()


@pytest.mark.parametrize("dt", [-1, float("nan"), float("inf")])
def test_invalid_time_rejected(dt):
    model = DynamicsModel(width=8)
    with pytest.raises(ValueError):
        model.step(torch.zeros(1, 48), model.initial_state(1), dt)
