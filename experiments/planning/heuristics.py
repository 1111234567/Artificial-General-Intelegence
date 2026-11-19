"""Observable-only diagnostics for shortcuts in consumable recipe tasks.

These deliberately simple policies are not exact planners. ``relaxed`` tests
each currently legal action with an actual inventory update, then ignores all
later consumption while computing a bounded production closure. ``fanout``
uses only a static structural guard and does not look ahead at all.

The final plan must be checked by the experiment's independent executor. No
solver, distance table, generator metadata, or oracle action enters a policy.
"""

from __future__ import annotations

from numbers import Integral

from .world import Problem


def _goal(inventory: tuple[int, ...] | list[int], goal: tuple[int, ...]) -> bool:
    return all(have >= need for have, need in zip(inventory, goal))


def _legal(inventory: tuple[int, ...], consume: tuple[tuple[int, ...], ...]) -> list[int]:
    return [
        action for action, required in enumerate(consume)
        if all(have >= need for have, need in zip(inventory, required))
    ]


def _transition(
    inventory: tuple[int, ...], required: tuple[int, ...], produced: tuple[int, ...],
) -> tuple[int, ...]:
    # Callers have already checked applicability. This implementation is local
    # so the diagnostic never asks the environment to classify a future state.
    return tuple(have - used + made for have, used, made in zip(inventory, required, produced))


def _relaxed_reachable(
    inventory: tuple[int, ...],
    goal: tuple[int, ...],
    consume: tuple[tuple[int, ...], ...],
    produce: tuple[tuple[int, ...], ...],
    caps: tuple[int, ...],
) -> tuple[bool, int]:
    """Delete-relaxed multiset closure, returning reachability and rule checks.

    Inputs enable a recipe but are never subtracted. Outputs accumulate, with
    each coordinate capped at its largest relevant recipe/goal threshold.
    Thus closure terminates even for a system that can produce indefinitely.
    This relaxation can miss competition between future uses of one resource.
    """
    reachable = [min(have, cap) for have, cap in zip(inventory, caps)]
    changed = True
    rule_checks = 0
    while changed:
        changed = False
        for required, produced in zip(consume, produce):
            rule_checks += 1
            if all(have >= need for have, need in zip(reachable, required)):
                following = [
                    min(cap, have + made)
                    for have, made, cap in zip(reachable, produced, caps)
                ]
                if following != reachable:
                    reachable = following
                    changed = True
    return _goal(reachable, goal), rule_checks


def heuristic_plan(
    problem: Problem,
    mode: str = "relaxed",
    max_steps: int = 24,
    *,
    return_stats: bool = False,
) -> tuple[int, ...] | dict[str, object]:
    """Return a legal candidate plan using only the public observation.

    ``fanout`` deduplicates equivalent recipes, finds nonrenewable resources
    with multiple distinct consumers, and vetoes rules whose sole input is
    such a resource. It can reject good rules and is only a shortcut probe.

    Statistics count actual work: ``transition_calls`` includes every exact
    candidate inventory update, plus selected-plan updates when not cached.
    The relaxed policy reuses its chosen candidate's update. Relaxed closure
    work is reported separately because those are not real state transitions.
    The external final verifier is outside these policy operation counts.
    """
    if mode not in {"relaxed", "fanout"}:
        raise ValueError("mode must be 'relaxed' or 'fanout'")
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    observation = problem.observation()
    inventory = tuple(observation["initial"])
    goal = tuple(observation["goal"])
    consume = tuple(tuple(row) for row in observation["consume"])
    produce = tuple(tuple(row) for row in observation["produce"])
    caps = tuple(
        max([goal[item]] + [required[item] for required in consume])
        for item in range(len(inventory))
    )
    unique = set(zip(consume, produce))
    fanout = tuple(sum(required[item] > 0 for required, _ in unique) for item in range(len(inventory)))
    renewable = tuple(any(produced[item] > 0 for _, produced in unique) for item in range(len(inventory)))
    plan: list[int] = []
    transition_calls = 0
    closure_calls = 0
    rule_checks = 0
    applicability_checks = 0
    reason = "max_steps"
    for _ in range(int(max_steps)):
        if _goal(inventory, goal):
            reason = "goal"
            break
        actions = _legal(inventory, consume)
        applicability_checks += len(consume)
        if not actions:
            reason = "no_legal_action"
            break
        candidates: list[int] = []
        following_states: dict[int, tuple[int, ...]] = {}
        if mode == "relaxed":
            for action in actions:
                following = _transition(inventory, consume[action], produce[action])
                transition_calls += 1
                following_states[action] = following
                reachable, checked = _relaxed_reachable(following, goal, consume, produce, caps)
                closure_calls += 1
                rule_checks += checked
                if reachable:
                    candidates.append(action)
        else:
            for action in actions:
                inputs = [item for item, count in enumerate(consume[action]) if count]
                guarded = len(inputs) == 1 and fanout[inputs[0]] > 1 and not renewable[inputs[0]]
                if not guarded:
                    candidates.append(action)
        if not candidates:
            reason = "filter_rejected_all"
            break
        action = candidates[0]
        plan.append(action)
        if mode == "relaxed":
            inventory = following_states[action]
        else:
            inventory = _transition(inventory, consume[action], produce[action])
            transition_calls += 1
    reached = _goal(inventory, goal)
    if reached:
        reason = "goal"
    result = tuple(plan)
    if not return_stats:
        return result
    return {
        "plan": result,
        "transition_calls": transition_calls,
        "relaxed_closure_calls": closure_calls,
        "relaxed_rule_checks": rule_checks,
        "applicability_rule_checks": applicability_checks,
        "core_calls": 0,
        "executed_actions": len(result),
        "goal_reached": reached,
        "termination_reason": reason,
    }
