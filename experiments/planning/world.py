"""Consumable inventory planning with exact, bounded state-space search.

An action subtracts its complete input multiset and then adds its output
multiset.  Seeing an item once never makes it permanently available.  The
generator produces finite acyclic recipe systems, while the generic solver
raises ``SearchLimitExceeded`` if a supplied system has too many states.

``metadata`` is for offline analysis.  ``observation`` is the complete public
problem representation and deliberately has no seed, family, depth, or plan.
Depth in this module means shortest plan length, not dependency critical path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from random import Random
from typing import Iterable, Mapping, Sequence


State = tuple[int, ...]
Recipe = tuple[int, ...]


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _vector(name: str, values: Iterable[int], size: int | None = None) -> State:
    try:
        result = tuple(_integer(name, value, 0) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of nonnegative integers") from exc
    if size is not None and len(result) != size:
        raise ValueError(f"{name} has length {len(result)}, expected {size}")
    return result


@dataclass(frozen=True)
class Problem:
    """A fully observed multiset-rewriting problem.

    Rules and inventory coordinates have no intrinsic names or order.  Counts
    can exceed one.  Every rule must consume something; zero-output disposal
    rules and finite self-loops are legal for hand-written problems.
    """

    initial: State
    goal: State
    consume: tuple[Recipe, ...]
    produce: tuple[Recipe, ...]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = _vector("initial", self.initial)
        if not initial:
            raise ValueError("a problem needs at least one item coordinate")
        goal = _vector("goal", self.goal, len(initial))
        consume = tuple(_vector("consume", row, len(initial)) for row in self.consume)
        produce = tuple(_vector("produce", row, len(initial)) for row in self.produce)
        if len(consume) != len(produce):
            raise ValueError("consume and produce must contain the same number of rules")
        if any(not any(row) for row in consume):
            raise ValueError("every action must consume at least one item")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "consume", consume)
        object.__setattr__(self, "produce", produce)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_items(self) -> int:
        return len(self.initial)

    @property
    def n_actions(self) -> int:
        return len(self.consume)

    def observation(self, state: Sequence[int] | None = None) -> dict:
        return observation(self, state)


def observation(problem: Problem, state: Sequence[int] | None = None) -> dict:
    """Return only observable inventory, recipes, and goal, without labels.

    When ``state`` is provided, it replaces the initial inventory for a policy
    decision at that state.  The representation is also JSON serializable.
    """
    inventory = problem.initial if state is None else _vector("state", state, problem.n_items)
    return {
        "initial": inventory,
        "goal": problem.goal,
        "consume": problem.consume,
        "produce": problem.produce,
    }


def is_goal(problem: Problem, state: Sequence[int]) -> bool:
    inventory = _vector("state", state, problem.n_items)
    return all(have >= need for have, need in zip(inventory, problem.goal))


def applicable_actions(problem: Problem, state: Sequence[int]) -> tuple[int, ...]:
    inventory = _vector("state", state, problem.n_items)
    return tuple(
        action
        for action, required in enumerate(problem.consume)
        if all(have >= need for have, need in zip(inventory, required))
    )


def step(problem: Problem, state: Sequence[int], action: int) -> State:
    """Execute one action, rejecting missing inputs before adding any output."""
    inventory = _vector("state", state, problem.n_items)
    action = _integer("action", action, 0)
    if action >= problem.n_actions:
        raise ValueError(f"action {action} is outside [0, {problem.n_actions})")
    required = problem.consume[action]
    if any(have < need for have, need in zip(inventory, required)):
        raise ValueError(f"action {action} is not applicable to this inventory")
    return tuple(
        have - used + made
        for have, used, made in zip(inventory, required, problem.produce[action])
    )


def execute_plan(
    problem: Problem,
    plan: Iterable[int],
    state: Sequence[int] | None = None,
) -> State:
    """Independently execute a plan without calling ``step`` or the solver.

    This intentionally uses a separate imperative inventory update.  It is the
    final validation path for generated and model-produced plans.
    """
    inventory = list(
        problem.initial if state is None else _vector("state", state, problem.n_items)
    )
    for position, action in enumerate(plan):
        action = _integer(f"plan[{position}]", action, 0)
        if action >= len(problem.consume):
            raise ValueError(f"plan[{position}] refers to missing action {action}")
        for item in range(len(inventory)):
            if inventory[item] < problem.consume[action][item]:
                raise ValueError(f"plan[{position}] lacks input item {item}")
        for item in range(len(inventory)):
            inventory[item] -= problem.consume[action][item]
        for item in range(len(inventory)):
            inventory[item] += problem.produce[action][item]
    return tuple(inventory)


def validate_plan(
    problem: Problem,
    plan: Iterable[int],
    state: Sequence[int] | None = None,
    *,
    require_goal: bool = True,
) -> bool:
    """Whether every action is legal and, by default, the final goal holds."""
    try:
        final = execute_plan(problem, plan, state)
    except (TypeError, ValueError):
        return False
    return not require_goal or all(
        final[item] >= problem.goal[item] for item in range(problem.n_items)
    )


class SearchLimitExceeded(RuntimeError):
    """The exact search was stopped; this must never be reported as unsolvable."""


@dataclass(frozen=True)
class Solution:
    """Exact shortest plan and labels for every forward-reachable state.

    ``None`` distances denote genuine dead ends after complete enumeration.
    ``distances`` is empty only when ``include_distances=False`` was requested.
    All equal-length best first actions are included in ``optimal_actions``.
    """

    plan: tuple[int, ...] | None
    distance: int | None
    optimal_actions: tuple[int, ...]
    distances: dict[State, int | None]
    states_explored: int


def _compiled_rules(problem: Problem) -> list[tuple[tuple, tuple, tuple[int, ...]]]:
    """Group equivalent recipes for search while preserving their action IDs."""
    groups: dict[tuple[Recipe, Recipe], list[int]] = {}
    for action, rows in enumerate(zip(problem.consume, problem.produce)):
        groups.setdefault(rows, []).append(action)
    rules = []
    for (required, produced), actions in groups.items():
        inputs = tuple((item, count) for item, count in enumerate(required) if count)
        delta = tuple(
            (item, made - used)
            for item, (used, made) in enumerate(zip(required, produced))
            if made != used
        )
        rules.append((inputs, delta, tuple(actions)))
    return rules


def _successor(state: State, inputs: tuple, delta: tuple) -> State | None:
    if any(state[item] < count for item, count in inputs):
        return None
    result = list(state)
    for item, change in delta:
        result[item] += change
    return tuple(result)


def optimal_actions(
    problem: Problem,
    state: Sequence[int],
    distances: Mapping[State, int | None] | Solution,
) -> tuple[int, ...]:
    """Read all shortest next actions from a completed exact-search result."""
    inventory = _vector("state", state, problem.n_items)
    table = distances.distances if isinstance(distances, Solution) else distances
    if inventory not in table:
        raise ValueError("state is not present in the supplied exact distance table")
    distance = table[inventory]
    if distance is None or distance == 0:
        return ()
    choices = []
    for action in applicable_actions(problem, inventory):
        following = step(problem, inventory, action)
        if following not in table:
            raise ValueError("distance table is incomplete for an applicable successor")
        if table[following] == distance - 1:
            choices.append(action)
    return tuple(choices)


def solve(
    problem: Problem,
    state: Sequence[int] | None = None,
    *,
    max_states: int = 50_000,
    include_distances: bool = True,
) -> Solution:
    """Enumerate reachable states, then reverse-BFS from every goal state.

    Unlike a monotone recipe closure, this graph retains the actual multiset
    after every action.  Enumeration includes states reached after a goal, so
    the optional distance table covers *all* states reachable from ``state``.
    An infinite or unexpectedly large system raises ``SearchLimitExceeded``.
    """
    limit = _integer("max_states", max_states, 1)
    start = problem.initial if state is None else _vector("state", state, problem.n_items)
    rules = _compiled_rules(problem)
    states = [start]
    indices = {start: 0}
    predecessors: list[list[int]] = [[]]
    goal_indices = []
    cursor = 0
    while cursor < len(states):
        current = states[cursor]
        if all(have >= need for have, need in zip(current, problem.goal)):
            goal_indices.append(cursor)
        following_ids = set()
        for required, delta, _ in rules:
            following = _successor(current, required, delta)
            if following is None:
                continue
            next_id = indices.get(following)
            if next_id is None:
                if len(states) >= limit:
                    raise SearchLimitExceeded(
                        f"more than {limit} states are reachable; exact solvability is unknown"
                    )
                next_id = len(states)
                indices[following] = next_id
                states.append(following)
                predecessors.append([])
            if next_id not in following_ids:
                predecessors[next_id].append(cursor)
                following_ids.add(next_id)
        cursor += 1

    values: list[int | None] = [None] * len(states)
    queue = deque(goal_indices)
    for goal_id in goal_indices:
        values[goal_id] = 0
    while queue:
        following_id = queue.popleft()
        following_distance = values[following_id]
        assert following_distance is not None
        for previous_id in predecessors[following_id]:
            if values[previous_id] is None:
                values[previous_id] = following_distance + 1
                queue.append(previous_id)

    distance = values[0]
    choices = ()
    plan = None
    if distance is not None:
        path = []
        current = start
        remaining = distance
        while remaining:
            candidates = []
            for required, delta, actions in rules:
                following = _successor(current, required, delta)
                if following is not None and values[indices[following]] == remaining - 1:
                    candidates.append((actions[0], following, actions))
            if not path:
                choices = tuple(sorted(action for _, _, actions in candidates for action in actions))
            action, current, _ = min(candidates, key=lambda candidate: candidate[0])
            path.append(action)
            remaining -= 1
        plan = tuple(path)
    table = dict(zip(states, values)) if include_distances else {}
    return Solution(plan, distance, choices, table, len(states))


def permute_problem(
    problem: Problem,
    *,
    item_order: Sequence[int],
    action_order: Sequence[int],
) -> Problem:
    """Reindex coordinates and recipes: new position ``i`` uses old ``order[i]``."""
    items = tuple(item_order)
    actions = tuple(action_order)
    if sorted(items) != list(range(problem.n_items)):
        raise ValueError("item_order must be a permutation of all item indices")
    if sorted(actions) != list(range(problem.n_actions)):
        raise ValueError("action_order must be a permutation of all action indices")
    return Problem(
        initial=tuple(problem.initial[item] for item in items),
        goal=tuple(problem.goal[item] for item in items),
        consume=tuple(tuple(problem.consume[action][item] for item in items) for action in actions),
        produce=tuple(tuple(problem.produce[action][item] for item in items) for action in actions),
        metadata=problem.metadata,
    )


def _has_matching(
    edges: Sequence[set[int]],
    targets: set[int],
    excluded_resource: int | None = None,
) -> bool:
    """Small independent augmenting-path test for the allocation generator."""
    owners: dict[int, int] = {}

    def assign(resource: int, seen: set[int]) -> bool:
        for component in sorted(edges[resource] & targets):
            if component in seen:
                continue
            seen.add(component)
            previous = owners.get(component)
            if previous is None or assign(previous, seen):
                owners[component] = resource
                return True
        return False

    for resource in range(len(edges)):
        if resource != excluded_resource:
            assign(resource, set())
    return len(owners) == len(targets)


def _has_hall_dead_end(edges: Sequence[set[int]], targets: set[int]) -> bool:
    """A needed component can be made wrongly despite relaxed reachability."""
    for resource, products in enumerate(edges):
        for component in products & targets:
            remaining = targets - {component}
            individually_possible = all(
                any(other != resource and needed in edges[other] for other in range(len(edges)))
                for needed in remaining
            )
            if individually_possible and not _has_matching(edges, remaining, resource):
                return True
    return False


def generate_problem(
    seed: int,
    depth: int,
    *,
    branches: int = 2,
    distractors: int = 4,
    n_items: int = 40,
    max_rules: int = 32,
    family: str = "chain",
    max_states: int = 50_000,
    goal_variant: int | None = None,
) -> Problem:
    """Generate an exactly checked finite recipe DAG with competing resources.

    The shortest action count is exactly depth.  A random bipartite graph
    allocates finite initial resource batches to required components, with a
    planted matching guaranteeing a solution.  Resource count grows with depth
    up to six.  For at least three resources the graph is rejection-sampled to
    contain a genuine Hall conflict for each of two possible targets: after a
    wrong relevant allocation, every remaining component is individually
    producible, but they cannot all be produced with the remaining resources.
    This cannot be detected by delete-relaxed reachability.

    Zero to several processing steps follow allocation.  The chain family
    merges component lanes and then processes serially, fork processes separate
    lanes, and cross consumes and replaces multiple lanes together.  These are
    small synthetic allocation/recipe DAGs, not general planning domains.
    Cross requires depth >= 4 and always includes a genuine two-input,
    two-output intermediate operation.  The other families allow depth >= 2.
    Counts of two occur in inventories and rules.  Fixed-width remaining rule
    slots duplicate legal recipes; no zero-input or infinite-loop padding is
    used.  All item symbols and complete recipe rows are permuted.

    goal_variant=0/1 gives counterfactual pairs with exactly the same initial
    inventory and recipes and different goals.  Some allocations are optimal
    for one goal and fatal for the other.  The default chooses the variant from
    the seeded RNG, never from seed parity.  Metadata is offline.
    """
    seed = _integer("seed", seed, 0)
    depth = _integer("depth", depth, 2)
    branches = _integer("branches", branches, 1)
    distractors = _integer("distractors", distractors, 1)
    n_items = _integer("n_items", n_items, 1)
    max_rules = _integer("max_rules", max_rules, 1)
    if family not in {"chain", "fork", "cross"}:
        raise ValueError(f"unknown family {family!r}; expected chain, fork, or cross")
    if family != "chain" and branches < 2:
        raise ValueError("fork and cross need at least two branches")
    if family == "cross" and depth < 4:
        raise ValueError("cross needs depth >= 4 for a real two-input two-output processing step")
    if goal_variant is not None:
        goal_variant = _integer("goal_variant", goal_variant, 0)
        if goal_variant not in (0, 1):
            raise ValueError("goal_variant must be 0 or 1")
    rng = Random(seed)
    # Reserve a random 0..2 processing operations; allocation complexity grows
    # with depth rather than using a fixed three-resource prefix.
    reserve = (
        1 + rng.randrange(min(2, depth - 3))
        if family == "cross"
        else rng.randrange(min(2, max(0, depth - 4)) + 1)
    )
    n_resources = min(6, depth - 1 - reserve)
    processing_steps = depth - n_resources - 1
    pivot = rng.randrange(n_resources)
    target_sets = [
        set(range(n_resources)),
        (set(range(n_resources)) - {pivot}) | {n_resources},
    ]
    sampled_variant = rng.randrange(2)
    variant = sampled_variant if goal_variant is None else goal_variant

    mandatory = {(resource, resource) for resource in range(n_resources)}
    mandatory.add((pivot, n_resources))
    candidates = [
        (resource, component)
        for resource in range(n_resources)
        for component in range(n_resources + 1)
        if (resource, component) not in mandatory
    ]
    processing_rule_count = processing_steps * (2 if family == "cross" else 1)
    base_rule_count = n_resources + 1 + processing_rule_count + 2
    edge_budget = max_rules - base_rule_count
    if edge_budget < 0:
        raise ValueError("max_rules is too small for allocation and goal recipes")
    extra_count = min(len(candidates), max(distractors, n_resources), edge_budget)
    if n_resources >= 3 and extra_count < 2:
        raise ValueError("max_rules needs room for alternative allocation edges")
    chosen_edges = None
    adjacency = None
    for _ in range(4_096):
        proposed = mandatory | set(rng.sample(candidates, extra_count))
        graph = [{component for resource, component in proposed if resource == row}
                 for row in range(n_resources)]
        if n_resources < 3 or all(_has_hall_dead_end(graph, target) for target in target_sets):
            chosen_edges, adjacency = sorted(proposed), graph
            break
    if chosen_edges is None or adjacency is None:
        raise RuntimeError("could not sample a Hall-conflict allocation graph within 4096 attempts")

    used_items = 0

    def fresh() -> int:
        nonlocal used_items
        if used_items >= n_items:
            raise ValueError(f"n_items={n_items} is too small for the recipe DAG")
        result = used_items
        used_items += 1
        return result

    def amount() -> int:
        return 2 if rng.random() < 0.35 else 1

    resource_items = [fresh() for _ in range(n_resources)]
    resource_counts = [amount() for _ in resource_items]
    component_items = [fresh() for _ in range(n_resources + 1)]
    component_counts = [amount() for _ in component_items]
    initial = dict(zip(resource_items, resource_counts))
    recipes: list[tuple[dict[int, int], dict[int, int]]] = [
        ({resource_items[resource]: resource_counts[resource]},
         {component_items[component]: component_counts[component]})
        for resource, component in chosen_edges
    ]
    goal_items = [fresh(), fresh()]
    common_components = sorted(target_sets[0] & target_sets[1])
    if family == "cross":
        # Include the goal-specific lane, so even depth 4 can have two real
        # lanes.  The two goals have separate processing branches of identical
        # shape, both present in the same public recipe system.
        lane_order = list(range(n_resources))
        rng.shuffle(lane_order)
        active_lanes = lane_order[:min(branches, n_resources)]
        forced_cross = rng.randrange(processing_steps)
        pattern = []
        for position in range(processing_steps):
            if position == forced_cross or rng.random() < 0.35:
                pattern.append(rng.sample(active_lanes, 2))
            else:
                pattern.append([active_lanes[position % len(active_lanes)]])
        for target, goal_item in zip(target_sets, goal_items):
            frontier = [(component_items[component], component_counts[component])
                        for component in sorted(target)]
            lane_paths = [1] * len(frontier)
            for lanes in pattern:
                required = {frontier[lane][0]: frontier[lane][1] for lane in lanes}
                produced = {}
                level = 1 + max(lane_paths[lane] for lane in lanes)
                for lane in lanes:
                    item, count = fresh(), amount()
                    produced[item] = count
                    frontier[lane] = (item, count)
                    lane_paths[lane] = level
                recipes.append((required, produced))
            recipes.append(({item: count for item, count in frontier}, {goal_item: 1}))
        critical_path = max(lane_paths) + 1
    else:
        frontier = [(component_items[component], component_counts[component])
                    for component in common_components]
        lane_order = list(range(len(frontier)))
        rng.shuffle(lane_order)
        active_lanes = lane_order[:min(branches, len(frontier))]
        lane_paths = [1] * len(frontier)
        for position in range(processing_steps):
            lanes = (list(range(len(frontier))) if family == "chain"
                     else [active_lanes[position % len(active_lanes)]])
            required = {frontier[lane][0]: frontier[lane][1] for lane in lanes}
            produced = {}
            level = 1 + max(lane_paths[lane] for lane in lanes)
            if family == "chain":
                item, count = fresh(), amount()
                produced[item] = count
                frontier, lane_paths = [(item, count)], [level]
            else:
                for lane in lanes:
                    item, count = fresh(), amount()
                    produced[item] = count
                    frontier[lane] = (item, count)
                    lane_paths[lane] = level
            recipes.append((required, produced))
        for target, goal_item in zip(target_sets, goal_items):
            specific = next(iter(target - set(common_components)))
            required = {item: count for item, count in frontier}
            required[component_items[specific]] = component_counts[specific]
            recipes.append((required, {goal_item: 1}))
        critical_path = max(lane_paths, default=1) + 1

    initial_recipes = list(recipes[:len(chosen_edges)])
    all_recipes = list(recipes)
    for recipe in initial_recipes:
        if len(all_recipes) < max_rules:
            all_recipes.append(recipe)
    while len(all_recipes) < max_rules:
        all_recipes.append(rng.choice(recipes))

    def dense(counts: Mapping[int, int]) -> State:
        return tuple(counts.get(item, 0) for item in range(n_items))

    problem = Problem(
        initial=dense(initial),
        goal=dense({goal_items[variant]: 1}),
        consume=tuple(dense(required) for required, _ in all_recipes),
        produce=tuple(dense(produced) for _, produced in all_recipes),
        metadata={
            "seed": seed,
            "family": family,
            "depth": depth,
            "critical_path": critical_path,
            "n_resources": n_resources,
            "allocation_count": n_resources,
            "processing_count": processing_steps,
            "has_cross": family == "cross",
        },
    )
    item_order = list(range(n_items))
    action_order = list(range(max_rules))
    rng.shuffle(item_order)
    rng.shuffle(action_order)
    problem = permute_problem(problem, item_order=item_order, action_order=action_order)
    solution = solve(problem, max_states=max_states)
    if solution.distance != depth:
        raise RuntimeError(f"generator requested optimal length {depth}, got {solution.distance}")
    if solution.plan is None or not validate_plan(problem, solution.plan):
        raise RuntimeError("generated exact plan failed independent execution")
    if not any(
        solution.distances[step(problem, problem.initial, action)] is None
        for action in applicable_actions(problem, problem.initial)
    ):
        raise RuntimeError("generated problem lacks a legal dead-end first action")
    return problem
