"""Inventory semantics and exact search checked against a separate enumerator."""

from collections import deque
from dataclasses import replace
from itertools import product
from random import Random

import pytest

from experiments.planning.world import (
    Problem,
    SearchLimitExceeded,
    applicable_actions,
    execute_plan,
    generate_problem,
    is_goal,
    observation,
    optimal_actions,
    permute_problem,
    solve,
    step,
    validate_plan,
)


def _reference_successors(problem, inventory):
    """No use of world.step, masks, compiled rules, or solver distances."""
    successors = []
    for action in range(len(problem.consume)):
        counts = {item: inventory[item] for item in range(len(inventory))}
        legal = True
        for item in range(len(inventory)):
            if counts[item] < problem.consume[action][item]:
                legal = False
                break
        if not legal:
            continue
        for item in range(len(inventory)):
            counts[item] = counts[item] - problem.consume[action][item]
        for item in range(len(inventory)):
            counts[item] = counts[item] + problem.produce[action][item]
        successors.append((action, tuple(counts[item] for item in range(len(inventory)))))
    return successors


def _reference_distance(problem, start):
    """Independent forward brute force; used only for small finite problems."""
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        inventory, distance = queue.popleft()
        if all(inventory[item] >= problem.goal[item] for item in range(len(inventory))):
            return distance
        for _, following in _reference_successors(problem, inventory):
            if following not in seen:
                seen.add(following)
                queue.append((following, distance + 1))
    return None


def test_consumes_all_inputs_and_preserves_multiplicities():
    problem = Problem(
        initial=(2, 1, 0),
        goal=(0, 0, 2),
        consume=((2, 1, 0), (1, 0, 0)),
        produce=((0, 0, 2), (0, 1, 0)),
    )
    assert applicable_actions(problem, problem.initial) == (0, 1)
    assert step(problem, problem.initial, 0) == (0, 0, 2)
    assert execute_plan(problem, [0]) == (0, 0, 2)
    assert validate_plan(problem, [0])
    assert not validate_plan(problem, [1, 0])
    assert not validate_plan(problem, [0, 0])
    with pytest.raises(ValueError, match="not applicable"):
        step(problem, (1, 1, 0), 0)


def test_one_material_cannot_be_used_for_two_different_branches():
    # A delete-relaxed closure reaches items 1 and 2, but the actual single
    # initial unit can go to only one of them, so their join is impossible.
    problem = Problem(
        initial=(1, 0, 0, 0),
        goal=(0, 0, 0, 1),
        consume=((1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 1, 0)),
        produce=((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )
    solution = solve(problem)
    assert solution.plan is None
    assert solution.distance is None
    assert solution.optimal_actions == ()
    assert all(distance is None for distance in solution.distances.values())
    assert _reference_distance(problem, problem.initial) is None
    assert not validate_plan(problem, [0, 1, 2])


def test_inputs_are_checked_before_outputs_and_self_loops_are_finite():
    problem = Problem((0,), (1,), ((1,),), ((2,),))
    assert applicable_actions(problem, (0,)) == ()
    assert not validate_plan(problem, [0])
    with pytest.raises(ValueError):
        step(problem, (0,), 0)
    loop = Problem((1, 0), (0, 1), ((1, 0),), ((1, 0),))
    result = solve(loop, max_states=1)
    assert result.states_explored == 1
    assert result.distance is None


def test_all_shortest_first_actions_and_interleavings_are_valid():
    problem = Problem(
        initial=(1, 1, 0, 0, 0),
        goal=(0, 0, 0, 0, 1),
        consume=((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 1, 0), (1, 0, 0, 0, 0)),
        produce=((0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1), (0, 0, 1, 0, 0)),
    )
    result = solve(problem)
    assert result.distance == 3
    assert result.optimal_actions == (0, 1, 3)
    for plan in ((0, 1, 2), (1, 0, 2), (3, 1, 2), (1, 3, 2)):
        assert validate_plan(problem, plan)
    for inventory, distance in result.distances.items():
        assert distance == _reference_distance(problem, inventory)
        labels = optimal_actions(problem, inventory, result)
        if distance is None or distance == 0:
            assert labels == ()
        else:
            assert labels == tuple(
                action for action, following in _reference_successors(problem, inventory)
                if _reference_distance(problem, following) == distance - 1
            )


def test_goal_is_a_lower_bound_and_distance_table_includes_states_after_goal():
    problem = Problem((2, 0), (1, 0), ((1, 0),), ((0, 1),))
    result = solve(problem)
    assert result.plan == ()
    assert result.distance == 0
    assert result.optimal_actions == ()
    assert is_goal(problem, (2, 7))
    assert result.distances == {(2, 0): 0, (1, 1): 0, (0, 2): None}
    assert solve(problem, include_distances=False).distances == {}


def test_search_limit_never_turns_unknown_into_unsolvable():
    unbounded = Problem((1,), (100,), ((1,),), ((2,),))
    with pytest.raises(SearchLimitExceeded, match="solvability is unknown"):
        solve(unbounded, max_states=5)
    bounded = Problem((1, 0), (0, 1), ((1, 0),), ((0, 1),))
    assert solve(bounded, max_states=2).distance == 1
    with pytest.raises(SearchLimitExceeded):
        solve(bounded, max_states=1)


@pytest.mark.parametrize("family", ["chain", "fork", "cross"])
@pytest.mark.parametrize("depth", [2, 3, 4, 6, 8, 12])
@pytest.mark.parametrize("seed", [0, 17, 839])
def test_generator_has_exact_length_resource_dead_end_and_fixed_shape(family, depth, seed):
    if family == "cross" and depth < 4:
        with pytest.raises(ValueError, match="cross needs depth >= 4"):
            generate_problem(seed, depth, family=family)
        return
    problem = generate_problem(seed, depth, family=family)
    result = solve(problem)
    assert problem.n_items == 40
    assert problem.n_actions == 32
    assert all(any(required) for required in problem.consume)
    assert result.distance == depth
    assert result.states_explored < 10_000
    assert result.plan is not None
    assert len(result.plan) == depth
    assert validate_plan(problem, result.plan)
    assert execute_plan(problem, result.plan) == _execute_reference(problem, result.plan)
    assert result.optimal_actions == optimal_actions(problem, problem.initial, result)
    assert len(result.optimal_actions) >= 2  # duplicates or independent lane orders
    first_actions = _reference_successors(problem, problem.initial)
    dead_ends = [following for _, following in first_actions if result.distances[following] is None]
    assert dead_ends
    assert all(_reference_distance(problem, following) is None for following in set(dead_ends))
    assert _reference_distance(problem, problem.initial) == depth


def _execute_reference(problem, plan):
    inventory = problem.initial
    for action in plan:
        choices = dict(_reference_successors(problem, inventory))
        if action not in choices:
            raise ValueError("illegal reference plan")
        inventory = choices[action]
    return inventory


def test_generated_all_state_distances_match_separate_forward_search():
    for family, seed in product(("chain", "fork", "cross"), range(3)):
        problem = generate_problem(seed, 6, family=family)
        result = solve(problem)
        for inventory, distance in result.distances.items():
            assert distance == _reference_distance(problem, inventory)


def test_fixed_dimensions_across_depths_and_reproducibility():
    for family in ("chain", "fork", "cross"):
        for depth in ((4, 6, 12) if family == "cross" else (2, 6, 12)):
            first = generate_problem(4201, depth, family=family)
            assert first == generate_problem(4201, depth, family=family)
            assert first != generate_problem(4202, depth, family=family)
            assert len(first.initial) == len(first.goal) == 40
            assert len(first.consume) == len(first.produce) == 32
    fork = generate_problem(5, 6, family="fork")
    assert fork.metadata["critical_path"] < fork.metadata["depth"]


def test_random_item_and_action_permutations_preserve_every_label():
    problem = generate_problem(41, 6, family="cross")
    result = solve(problem)
    rng = Random(99)
    items = list(range(problem.n_items))
    actions = list(range(problem.n_actions))
    rng.shuffle(items)
    rng.shuffle(actions)
    changed = permute_problem(problem, item_order=items, action_order=actions)
    new_result = solve(changed)
    assert changed.initial != problem.initial
    assert changed.consume != problem.consume
    assert new_result.distance == result.distance
    assert new_result.optimal_actions == tuple(
        action for action, previous in enumerate(actions) if previous in result.optimal_actions
    )
    inverse_actions = {previous: action for action, previous in enumerate(actions)}
    assert validate_plan(changed, [inverse_actions[action] for action in result.plan])
    assert new_result.distances == {
        tuple(inventory[item] for item in items): distance
        for inventory, distance in result.distances.items()
    }


def test_metadata_is_not_observable_and_cannot_affect_search_or_execution():
    problem = generate_problem(219, 5, family="fork")
    poisoned = replace(problem, metadata={"family": "fake", "depth": -1, "plan": [999], "secret": object()})
    assert set(observation(problem)) == {"initial", "goal", "consume", "produce"}
    assert problem.observation() == poisoned.observation()
    assert solve(problem) == solve(poisoned)
    result = solve(poisoned)
    assert validate_plan(poisoned, result.plan)
    following = step(problem, problem.initial, result.plan[0])
    assert observation(problem, following)["initial"] == following
    assert "metadata" not in observation(poisoned, following)


def test_independent_validator_does_not_call_step_or_solver(monkeypatch):
    problem = generate_problem(37, 4)
    plan = solve(problem).plan

    def forbidden(*args, **kwargs):
        raise AssertionError("validator reused the planner/step implementation")

    monkeypatch.setattr("experiments.planning.world.step", forbidden)
    monkeypatch.setattr("experiments.planning.world.solve", forbidden)
    assert validate_plan(problem, plan)
    assert not validate_plan(problem, list(plan) + [plan[0]])


@pytest.mark.parametrize("action", [-1, 2, True, 0.5, "0"])
def test_invalid_actions_are_rejected(action):
    problem = Problem((1, 0), (0, 1), ((1, 0),), ((0, 1),))
    with pytest.raises(ValueError):
        step(problem, problem.initial, action)
    assert not validate_plan(problem, [action])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial": (-1, 0)},
        {"initial": (True, 0)},
        {"goal": (1,)},
        {"consume": ((0, 0),)},
        {"consume": ((1, 0), (1, 0))},
        {"produce": ((0, -1),)},
    ],
)
def test_malformed_problems_are_rejected(kwargs):
    values = {"initial": (1, 0), "goal": (0, 1), "consume": ((1, 0),), "produce": ((0, 1),)}
    values.update(kwargs)
    with pytest.raises(ValueError):
        Problem(**values)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_items": 2}, {"max_rules": 5}, {"distractors": 0}, {"family": "unknown"}, {"branches": 1, "family": "fork"}],
)
def test_impossible_generation_requests_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        generate_problem(0, 4, **kwargs)


def _delete_relaxed_possible(problem, inventory):
    """Independent count-aware closure that deliberately never deletes inputs."""
    available = list(inventory)
    caps = [max([problem.goal[item], available[item]] + [row[item] for row in problem.consume])
            for item in range(problem.n_items)]
    changed = True
    while changed:
        changed = False
        for required, produced in zip(problem.consume, problem.produce):
            if any(available[item] < required[item] for item in range(problem.n_items)):
                continue
            for item in range(problem.n_items):
                updated = min(caps[item], available[item] + produced[item])
                if updated != available[item]:
                    available[item] = updated
                    changed = True
    return all(available[item] >= problem.goal[item] for item in range(problem.n_items))


def _goal_relevant_actions(problem):
    needed = {item for item, count in enumerate(problem.goal) if count}
    actions = set()
    changed = True
    while changed:
        changed = False
        for action, (required, produced) in enumerate(zip(problem.consume, problem.produce)):
            if action not in actions and any(produced[item] for item in needed):
                actions.add(action)
                needed.update(item for item, count in enumerate(required) if count)
                changed = True
    return actions


@pytest.mark.parametrize("family,depth", [("chain", 4), ("fork", 6), ("cross", 6), ("cross", 12)])
@pytest.mark.parametrize("seed", [0, 17, 839])
def test_hall_conflict_stays_relaxed_reachable_but_cannot_reach_joint_goal(family, depth, seed):
    problem = generate_problem(seed, depth, family=family)
    assert problem.metadata["allocation_count"] >= 3
    result = solve(problem)
    relevant = _goal_relevant_actions(problem)
    hard_dead_ends = [
        action for action, following in _reference_successors(problem, problem.initial)
        if action in relevant
        and result.distances[following] is None
        and _delete_relaxed_possible(problem, following)
    ]
    assert hard_dead_ends, "no genuine joint-resource conflict survived the independent relaxed check"


@pytest.mark.parametrize("family,depth", [("chain", 2), ("chain", 6), ("fork", 8), ("cross", 4), ("cross", 12)])
def test_counterfactual_goal_changes_reverse_optimal_and_fatal_resource_choices(family, depth):
    first = generate_problem(33, depth, family=family, goal_variant=0)
    second = generate_problem(33, depth, family=family, goal_variant=1)
    assert first.initial == second.initial
    assert first.consume == second.consume
    assert first.produce == second.produce
    assert first.goal != second.goal
    assert observation(first) | {"goal": second.goal} == observation(second)
    first_solution, second_solution = solve(first), solve(second)
    assert first_solution.distance == second_solution.distance == depth
    assert any(
        second_solution.distances[step(second, second.initial, action)] is None
        for action in first_solution.optimal_actions
    )
    assert any(
        first_solution.distances[step(first, first.initial, action)] is None
        for action in second_solution.optimal_actions
    )


@pytest.mark.parametrize("depth", [4, 5, 6, 8, 12])
def test_cross_has_real_multiple_input_and_multiple_output_processing(depth):
    problem = generate_problem(391, depth, family="cross")
    assert problem.metadata["processing_count"] >= 1
    assert problem.metadata["has_cross"]
    goal_items = {item for item, count in enumerate(problem.goal) if count}
    assert any(
        sum(count > 0 for count in required) >= 2
        and sum(count > 0 for count in produced) >= 2
        and not any(produced[item] for item in goal_items)
        for required, produced in zip(problem.consume, problem.produce)
    )


def test_depth_increases_allocation_decisions_and_metadata_records_actual_shape():
    for family in ("chain", "fork", "cross"):
        counts = []
        for depth in (4, 6, 8, 12):
            problem = generate_problem(18, depth, family=family)
            allocated = problem.metadata["allocation_count"]
            processed = problem.metadata["processing_count"]
            assert allocated + processed + 1 == depth
            assert 1 <= problem.metadata["critical_path"] <= depth
            counts.append(allocated)
        assert counts[-1] == 6
        assert counts[-1] > counts[0]


def test_default_goal_is_rng_sampled_and_both_variants_occur_with_fixed_seed_parity():
    variants = set()
    for seed in range(0, 20, 2):
        chosen = generate_problem(seed, 4)
        first = generate_problem(seed, 4, goal_variant=0)
        variants.add(0 if chosen.goal == first.goal else 1)
    assert variants == {0, 1}
