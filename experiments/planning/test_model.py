from dataclasses import replace

import pytest
import torch

from .model import Planner, optimal_set_loss, tensorize
from .world import generate_problem, permute_problem


@pytest.fixture
def batch():
    torch.set_num_threads(2)
    problems = [generate_problem(seed, 4, family="fork") for seed in (37, 38)]
    return problems, *tensorize(problems)


def test_metadata_cannot_enter_features(batch):
    problems, features, legal = batch
    changed = [replace(p, metadata={"plan": [0], "distance": -900, "seed": 99}) for p in problems]
    new_features, new_legal = tensorize(changed)
    assert torch.equal(features, new_features)
    assert torch.equal(legal, new_legal)


def test_rule_permutation_equivariance(batch):
    problems, features, legal = batch
    torch.manual_seed(5)
    model = Planner(width=32, rounds=3).eval()
    order = list(reversed(range(problems[0].n_actions)))
    permuted = [permute_problem(p, item_order=range(p.n_items), action_order=order) for p in problems]
    new_features, new_legal = tensorize(permuted)
    with torch.no_grad():
        expected = model(features, legal)[:, order]
        actual = model(new_features, new_legal)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_feedback_detach_preserves_forward_and_encoder_gradients(batch):
    _, features, legal = batch
    torch.manual_seed(6)
    joint = Planner(width=32, rounds=3, kind="recurrent")
    detached = Planner(width=32, rounds=3, kind="detach")
    detached.load_state_dict(joint.state_dict())
    # Comparing all legal scores avoids assigning arbitrary single-action labels.
    joint_output = joint(features, legal)
    detached_output = detached(features, legal)
    torch.testing.assert_close(joint_output, detached_output)
    joint_output[legal].square().mean().backward()
    detached_output[legal].square().mean().backward()
    for model in (joint, detached):
        assert model.encoder.weight.grad is not None
        assert torch.isfinite(model.encoder.weight.grad).all()
        assert model.encoder.weight.grad.abs().sum() > 0
    assert not torch.allclose(joint.encoder.weight.grad, detached.encoder.weight.grad)


def test_sharing_parameter_count_and_initialization(batch):
    _, features, legal = batch
    models = {}
    for kind in ("shallow", "recurrent", "untied"):
        torch.manual_seed(123)
        models[kind] = Planner(width=32, rounds=4, kind=kind).eval()
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(models["shallow"]) == count(models["recurrent"])
    assert count(models["untied"]) > count(models["recurrent"])
    with torch.no_grad():
        shallow = models["shallow"](features, legal)
        for kind in ("recurrent", "untied"):
            torch.testing.assert_close(shallow, models[kind](features, legal, rounds=1))
    with pytest.raises(ValueError):
        models["untied"](features, legal, rounds=5)


def test_optimal_set_mass_handles_multiple_answers_and_illegal_actions():
    logits = torch.tensor([[0., 0., -torch.inf, 0.]], requires_grad=True)
    target = torch.tensor([[True, True, False, False]])
    loss = optimal_set_loss(logits, target)
    torch.testing.assert_close(loss, torch.tensor(1.5).log())
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1/6, -1/6, 0., 1/3]]))
    with pytest.raises(ValueError):
        optimal_set_loss(logits, torch.zeros_like(target))
    with pytest.raises(ValueError):
        optimal_set_loss(logits, ~torch.isfinite(logits))


def test_input_mask_is_only_current_applicability(batch):
    problems, features, legal = batch
    from .world import solve, step
    for index, problem in enumerate(problems):
        solution = solve(problem)
        dead = [a for a in range(problem.n_actions)
                if legal[index, a] and solution.distances[step(problem, problem.initial, a)] is None]
        assert dead, "the input mask must leave legal but fatal actions visible"
    output = Planner(width=32)(features, legal)
    assert torch.isneginf(output[~legal]).all()
    assert torch.isfinite(output[legal]).all()
