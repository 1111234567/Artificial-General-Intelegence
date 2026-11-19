from dataclasses import replace

import pytest

from .heuristics import heuristic_plan
from .world import Problem, validate_plan


def protected_resource_problem():
    # Coordinates: protected resource, initial material, intermediate, goal.
    return Problem(
        initial=(1, 1, 0, 0), goal=(0, 0, 0, 1),
        consume=((1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 1, 0)),
        produce=((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )


@pytest.mark.parametrize("mode", ["relaxed", "fanout"])
def test_public_only_policies_avoid_one_resource_trap(mode, monkeypatch):
    problem = protected_resource_problem()
    from . import world

    def forbidden(*args, **kwargs):
        raise AssertionError("a diagnostic policy must not use an oracle/environment helper")

    for name in ("solve", "optimal_actions", "step", "applicable_actions", "is_goal"):
        monkeypatch.setattr(world, name, forbidden)
    assert heuristic_plan(problem, mode) == (1, 2)
    changed = replace(problem, metadata={"plan": [0], "distances": "incorrect", "depth": 9000})
    assert heuristic_plan(changed, mode) == (1, 2)
    assert problem.initial == (1, 1, 0, 0)


def test_relaxed_and_executed_work_are_counted_without_repeating_chosen_transition():
    result = heuristic_plan(protected_resource_problem(), "relaxed", return_stats=True)
    assert result["plan"] == (1, 2)
    # Two legal actions at each of the two decisions; selected updates cached.
    assert result["transition_calls"] == 4
    assert result["relaxed_closure_calls"] == 4
    assert result["relaxed_rule_checks"] >= 4 * 3
    assert result["core_calls"] == 0
    assert result["executed_actions"] == 2
    assert result["goal_reached"] is True
    assert result["termination_reason"] == "goal"
    assert validate_plan(protected_resource_problem(), result["plan"])
    guarded = heuristic_plan(protected_resource_problem(), "fanout", return_stats=True)
    assert guarded["transition_calls"] == 2
    assert guarded["relaxed_closure_calls"] == 0


def test_identical_rule_copies_do_not_create_false_multiple_consumers():
    problem = Problem(initial=(1, 0), goal=(0, 1),
                      consume=((1, 0), (1, 0)), produce=((0, 1), (0, 1)))
    assert heuristic_plan(problem, "fanout") == (0,)


def test_relaxed_probe_can_miss_future_delete_interactions():
    # After bad rule 0, only Y remains to produce BOTH B and C. The relaxed
    # closure overlooks this competition. A real successful plan is 1,3,4.
    # Coordinates: X, Y, useless output, B, C, goal.
    problem = Problem(
        initial=(1, 1, 0, 0, 0, 0), goal=(0, 0, 0, 0, 0, 1),
        consume=((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0),
                 (0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0)),
        produce=((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0),
                 (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)),
    )
    result = heuristic_plan(problem, "relaxed", return_stats=True)
    assert result["plan"] == (0,)
    assert result["goal_reached"] is False
    assert result["termination_reason"] == "filter_rejected_all"
    assert validate_plan(problem, (1, 3, 4))
    assert not validate_plan(problem, result["plan"])


@pytest.mark.parametrize("mode", ["relaxed", "fanout"])
def test_multiset_counts_limits_and_empty_goal(mode):
    problem = Problem(initial=(2, 0), goal=(0, 1), consume=((2, 0),), produce=((0, 1),))
    assert heuristic_plan(problem, mode) == (0,)
    missing = replace(problem, initial=(1, 0))
    result = heuristic_plan(missing, mode, return_stats=True)
    assert result["plan"] == ()
    assert result["termination_reason"] == "no_legal_action"
    assert result["transition_calls"] == 0
    assert heuristic_plan(problem, mode, max_steps=0) == ()
    already_solved = replace(problem, initial=(0, 1))
    result = heuristic_plan(already_solved, mode, max_steps=0, return_stats=True)
    assert result["plan"] == ()
    assert result["goal_reached"] is True


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_step_limits(bad):
    with pytest.raises(ValueError):
        heuristic_plan(protected_resource_problem(), max_steps=bad)


def test_invalid_policy_mode():
    with pytest.raises(ValueError):
        heuristic_plan(protected_resource_problem(), "oracle")
