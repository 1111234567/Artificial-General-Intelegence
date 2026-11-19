import pytest
import torch

from .model import AgentModel


@pytest.mark.parametrize("jobs", [1, 2, 4])
@pytest.mark.parametrize("kind", ["learned", "reservoir", "gru"])
def test_job_shape_and_temporal_credit(jobs, kind):
    model = AgentModel(input_size=24 * jobs, n_jobs=jobs, width=12, kind=kind)
    cue = torch.randn(3, jobs * 24, requires_grad=True)
    _, state = model.step(cue, model.initial_state(3), 0)
    for _ in range(4):
        logits, state = model.step(torch.zeros_like(cue), state, .25)
    assert logits.shape == (3, jobs, 11)
    logits.square().mean().backward()
    assert cue.grad.abs().sum() > 0
    if kind == "reservoir":
        assert model.recurrent.grad is None


def test_event_not_repeated_when_solver_refined():
    model = AgentModel(width=12)
    cue = torch.randn(2, 96)
    h = model.initial_state(2)
    _, a = model.step(cue, h, .5, .25)
    _, b = model.step(cue, h, .5, .0625)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, model.encoder(cue))


def test_frozen_and_plastic_recurrence_share_initial_function():
    torch.manual_seed(16)
    frozen = AgentModel(width=16, kind="reservoir")
    torch.manual_seed(16)
    learned = AgentModel(width=16, kind="learned")
    x, h = torch.randn(2, 96), torch.randn(2, 16)
    for a, b in zip(frozen.step(x, h, .375), learned.step(x, h, .375)):
        torch.testing.assert_close(a, b)
