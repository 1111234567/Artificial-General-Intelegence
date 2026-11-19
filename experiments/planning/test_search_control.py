import math

import pytest
import torch

from . import run, search_control as control, world


def duplicate_problem():
    return world.Problem(
        (1, 0, 0), (0, 0, 1),
        ((1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1)),
        metadata={"oracle_plan": [1, 3]},
    )


def test_unique_prior_equalizes_observable_recipe_groups():
    features, legal = run.tensorize([duplicate_problem()])
    unique = control.UniformPolicy("unique")
    logits = unique(features, legal)
    probabilities = logits.softmax(-1)[0]
    torch.testing.assert_close(probabilities[0], probabilities[1] + probabilities[2])
    assert probabilities[3] == 0
    action = control.UniformPolicy("action")(features, legal).softmax(-1)[0]
    torch.testing.assert_close(action[1] + action[2], action[0] * 2)
    assert sum(parameter.numel() for parameter in unique.parameters()) == 0
    assert logits[0, 1] == pytest.approx(-math.log(2))


def test_prior_ignores_goal_inventory_metadata_and_invalidates_recipe_cache():
    problem = duplicate_problem()
    features, legal = run.tensorize([problem])
    policy = control.UniformPolicy("unique")
    first = policy(features, legal)
    changed = features.clone()
    n_items = problem.n_items
    changed[:, :, 2*n_items:4*n_items] = 9876
    torch.testing.assert_close(first, policy(changed, legal))
    # Changing recipe 2 makes it a separate observable semantic action.
    changed[0, 2, 0] = 2
    changed_result = policy(changed, legal)
    assert changed_result[0, 1] == 0
    assert changed_result[0, 2] == 0


def test_control_reuses_real_search_but_has_no_neural_calls_or_oracle(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("solver must not enter a search-only policy")
    original_tensorize = run.tensorize
    def public_only(problems, *args, **kwargs):
        assert all(problem.metadata == {} for problem in problems)
        return original_tensorize(problems, *args, **kwargs)
    monkeypatch.setattr(world, "solve", forbidden)
    monkeypatch.setattr(run, "tensorize", public_only)
    result = control.search_only(duplicate_problem(), mode="unique", budget=2,
                                 max_steps=24, branch=4, beam_width=32)
    assert result["success"] and world.validate_plan(duplicate_problem(), result["plan"])
    assert result["core_calls"] == result["neural_core_calls"] == 0
    assert result["expanded_scored_states"] == 2
    assert result["transition_calls"] <= 2 * duplicate_problem().n_actions
    assert result["posthoc"] is True


def test_summary_keeps_neural_cost_zero_and_reports_expansions():
    result = control.search_only(duplicate_problem(), mode="unique", budget=2,
                                 max_steps=24, branch=4, beam_width=32)
    row = dict(result, case_id="case", method="uniform_unique_c2", distribution="iid",
               family="chain", depth=2, optimal_length=2)
    summaries = control.aggregate([row])
    assert summaries
    assert all(summary["mean_core_calls"] == 0 for summary in summaries)
    assert all(summary["mean_expanded_scored_states"] == 2 for summary in summaries)
    assert all(summary["posthoc"] for summary in summaries)
