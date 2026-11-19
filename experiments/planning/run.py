"""Reproducible training and simulator-only evaluation of recipe planners.

Run ``python -m experiments.planning.run --help``.  Training freezes datasets,
configuration and source hashes in a fresh directory.  Held-out evaluation is
an explicit, one-shot action; the exact solver never runs inside a policy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
from random import Random
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from . import world
from .model import Planner, optimal_set_loss, tensorize


TRAIN_GROUPS = tuple((family, depth) for family in ("chain", "fork")
                     for depth in (2, 3, 4, 5, 6))
TEST_GROUPS = TRAIN_GROUPS + tuple(
    (family, depth) for family in ("chain", "fork") for depth in (7, 8, 10, 12)
) + tuple(("cross", depth) for depth in (4, 6, 8, 10))
SOURCE_FILES = ("__init__.py", "world.py", "model.py", "run.py", "heuristics.py")
SCHEMA_VERSION = 1


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def public_problem(problem: world.Problem) -> world.Problem:
    """Make the policy boundary explicit: offline metadata is not a feature."""
    return world.Problem(**problem.observation())


def problem_from_record(record: dict) -> world.Problem:
    return world.Problem(**record["observation"])


def distribution(family: str, depth: int) -> str:
    if family == "cross":
        return "structural_ood" if depth <= 6 else "structural_length_ood"
    return "iid" if (family, depth) in TRAIN_GROUPS else "length_ood"


def split_specs(split: str, count: int, seed: int) -> list[dict]:
    """Disjoint, auditable seed namespaces; count is per group outside train."""
    if split not in {"train", "val", "test"} or count < 1 or seed < 0:
        raise ValueError("invalid split, count or seed")
    groups = TEST_GROUPS if split == "test" else TRAIN_GROUPS
    size = count if split == "train" else count * len(groups)
    if size >= 100_000_000:
        raise ValueError("dataset exceeds its reserved seed namespace")
    base = (seed + 1) * 1_000_000_000 + {"train": 0, "val": 100_000_000,
                                       "test": 200_000_000}[split]
    specs = []
    # Interleaving groups makes every prefix, including a smoke test, balanced.
    for index in range(size):
        family, depth = groups[index % len(groups)]
        specs.append({"case_id": f"{split}:{index:07d}", "seed": base + index,
                      "family": family, "depth": depth,
                      "distribution": distribution(family, depth)})
    return specs


def labelled_states(problem: world.Problem, *, seed: int, max_states: int = 8,
                    solver_max_states: int = 50_000) -> list[dict]:
    """Initial plus sampled reachable solvable states, with ALL optimal actions.

    States are sampled from the complete exact graph, not a canonical path.
    The training sampler later gives initial and other states equal mass.
    Search-limit exceptions deliberately propagate instead of becoming labels.
    """
    if max_states < 1:
        raise ValueError("max_states must be positive")
    solution = world.solve(problem, max_states=solver_max_states)
    if solution.distance is None or solution.distance == 0:
        raise ValueError("a training problem must be solvable and non-goal")
    others = [state for state, distance in solution.distances.items()
              if state != problem.initial and distance is not None and distance > 0]
    Random(seed).shuffle(others)
    states = [problem.initial] + others[:max_states - 1]
    result = []
    for state in states:
        optimal = set(world.optimal_actions(problem, state, solution))
        if not optimal:
            raise RuntimeError("a sampled training state lacks an optimal action")
        result.append({"inventory": state,
                       "target_mask": [action in optimal for action in range(problem.n_actions)]})
    return result


def generate_dataset(path: Path, specs: Sequence[dict], config: dict,
                     *, training: bool) -> dict:
    total_states = 0
    start = time.perf_counter()
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, spec in enumerate(specs):
            problem = world.generate_problem(
                spec["seed"], spec["depth"], family=spec["family"],
                n_items=config["n_items"], max_rules=config["n_rules"],
                max_states=config["solver_max_states"],
            )
            record = dict(spec, observation=problem.observation(),
                          generator_metadata=problem.metadata)
            if training:
                record["states"] = labelled_states(
                    problem, seed=spec["seed"] ^ 0xA9C3,
                    max_states=config["states_per_problem"],
                    solver_max_states=config["solver_max_states"],
                )
                total_states += len(record["states"])
            else:
                # Only offline metrics survive. No solver plan or future-state
                # distance table is serialized into evaluation policy inputs.
                solution = world.solve(problem, max_states=config["solver_max_states"],
                                       include_distances=False)
                if solution.distance is None:
                    raise RuntimeError("generated evaluation problem is unsolvable")
                record["optimal_length"] = solution.distance
                record["exact_states_explored"] = solution.states_explored
            stream.write(json_text(record) + "\n")
            if (index + 1) % 512 == 0:
                print(f"dataset {path.stem}: {index + 1}/{len(specs)} problems", flush=True)
    return {"file": path.name, "sha256": file_sha256(path), "problems": len(specs),
            "labelled_states": total_states, "seconds": time.perf_counter() - start,
            "specs": list(specs)}


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@dataclass
class StatePool:
    features: torch.Tensor
    legal: torch.Tensor
    targets: torch.Tensor
    initial_indices: torch.Tensor  # CPU sampling tables, independent of model RNG.
    other_indices: torch.Tensor
    other_counts: torch.Tensor


def build_pool(records: Sequence[dict], device: str, chunk_size: int = 256) -> StatePool:
    if not records:
        raise ValueError("empty training dataset")
    feature_chunks, legal_chunks, targets = [], [], []
    initial_indices, other_lists = [], []
    pending_problems, pending_states = [], []
    offset = 0

    def flush() -> None:
        if pending_problems:
            features, legal = tensorize(pending_problems, pending_states, device=device)
            feature_chunks.append(features)
            legal_chunks.append(legal)
            pending_problems.clear()
            pending_states.clear()

    for record in records:
        problem = problem_from_record(record)
        states = record["states"]
        if not states or tuple(states[0]["inventory"]) != problem.initial:
            raise ValueError("the first labelled state must be the initial state")
        initial_indices.append(offset)
        other_lists.append(list(range(offset + 1, offset + len(states))))
        for state in states:
            pending_problems.append(problem)
            pending_states.append(tuple(state["inventory"]))
            target = state["target_mask"]
            if len(target) != problem.n_actions or not any(target):
                raise ValueError("invalid optimal action mask")
            targets.append(target)
            offset += 1
            if len(pending_problems) >= chunk_size:
                flush()
    flush()
    feature_tensor = torch.cat(feature_chunks)
    legal_tensor = torch.cat(legal_chunks)
    target_tensor = torch.tensor(targets, dtype=torch.bool, device=device)
    if (target_tensor & ~legal_tensor).any():
        raise ValueError("an optimal target is illegal")
    width = max(1, max(map(len, other_lists)))
    other_indices = torch.zeros((len(records), width), dtype=torch.long)
    for index, others in enumerate(other_lists):
        if others:
            other_indices[index, :len(others)] = torch.tensor(others)
    return StatePool(feature_tensor, legal_tensor, target_tensor,
                     torch.tensor(initial_indices), other_indices,
                     torch.tensor([len(others) for others in other_lists]))


def sample_batch_indices(pool: StatePool, batch_size: int,
                         generator: torch.Generator) -> torch.Tensor:
    """Choose problems uniformly, then initial/other state with 50% probability."""
    problems = torch.randint(len(pool.initial_indices), (batch_size,), generator=generator)
    counts = pool.other_counts[problems]
    other_position = (torch.rand(batch_size, generator=generator) * counts).long()
    other = pool.other_indices[problems, other_position]
    use_other = (torch.rand(batch_size, generator=generator) < 0.5) & (counts > 0)
    return torch.where(use_other, other, pool.initial_indices[problems])


def synchronize(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def finish_rollout(problem: world.Problem, plan: Sequence[int], *, core_calls: int,
                   transition_calls: int | None, latency_ms: float, stop_reason: str,
                   simulator_state: world.State | None = None) -> dict:
    plan = tuple(int(action) for action in plan)
    legal_plan = world.validate_plan(problem, plan, require_goal=False)
    validated_goal = world.validate_plan(problem, plan, require_goal=True)
    return {"plan": plan, "plan_length": len(plan), "success": validated_goal,
            "valid_plan": legal_plan, "independent_validation": validated_goal,
            "simulator_goal_reached": (None if simulator_state is None
                                       else world.is_goal(problem, simulator_state)),
            "core_calls": core_calls, "transition_calls": transition_calls,
            "validation_actions": 2 * len(plan), "latency_ms": latency_ms,
            "stop_reason": stop_reason}


@torch.inference_mode()
def greedy_rollouts(model: Planner, problems: Sequence[world.Problem], *, rounds: int,
                    device: str = "cpu", core_budget: int = 96, max_steps: int = 24,
                    batch_size: int = 128) -> list[dict]:
    """Batched full-plan greedy inference. No solver or labels are consulted.

    core_calls counts one core-layer evaluation for one problem. Latencies
    are measured batched wall time apportioned across active cases, not
    single-case request latencies. Goal halting is supplied by the simulator.
    """
    if rounds < 1 or core_budget < 0 or max_steps < 0 or batch_size < 1:
        raise ValueError("invalid rollout budget")
    public = [public_problem(problem) for problem in problems]
    results = []
    model.eval()
    for begin in range(0, len(public), batch_size):
        batch = public[begin:begin + batch_size]
        states = [problem.initial for problem in batch]
        plans = [[] for _ in batch]
        calls = [0] * len(batch)
        transitions = [0] * len(batch)
        latencies = [0.0] * len(batch)
        reasons: list[str | None] = [None] * len(batch)
        while True:
            active = []
            for index, problem in enumerate(batch):
                if reasons[index] is not None:
                    continue
                if world.is_goal(problem, states[index]):
                    reasons[index] = "goal"
                elif len(plans[index]) >= max_steps:
                    reasons[index] = "max_steps"
                elif calls[index] + rounds > core_budget:
                    reasons[index] = "core_budget"
                elif not world.applicable_actions(problem, states[index]):
                    reasons[index] = "dead_end"
                else:
                    active.append(index)
            if not active:
                break
            synchronize(device)
            start = time.perf_counter()
            features, legal = tensorize([batch[index] for index in active],
                                       [states[index] for index in active], device=device)
            logits = model(features, legal, rounds=rounds)
            chosen = logits.masked_fill(~legal, -torch.inf).argmax(dim=-1).tolist()
            for index, action in zip(active, chosen):
                calls[index] += rounds
                states[index] = world.step(batch[index], states[index], action)
                transitions[index] += 1
                plans[index].append(action)
            synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000 / len(active)
            for index in active:
                latencies[index] += elapsed
        for index, problem in enumerate(batch):
            result = finish_rollout(problem, plans[index], core_calls=calls[index],
                                    transition_calls=transitions[index],
                                    latency_ms=latencies[index], stop_reason=str(reasons[index]),
                                    simulator_state=states[index])
            result["latency_kind"] = "amortized_batch_wall"
            results.append(result)
    return results


@torch.inference_mode()
def neural_best_first(model: Planner, problem: world.Problem, *, device: str = "cpu",
                      core_budget: int = 96, max_steps: int = 24, branch: int = 4,
                      beam_width: int = 32) -> dict:
    """A real external search baseline using only shallow neural scores.

    Priority is cumulative log policy probability. Each queried distinct
    state costs one shallow core call. Expansion keeps the top ``branch``
    distinct successor states. Equivalent recipes first share their summed
    policy probability, so duplicate action IDs do not dilute a branch's
    priority. A single representative action executes each equivalent group.
    Every candidate ``world.step`` counts, with a separate hard transition
    budget of core_budget*n_actions. No future distance or oracle is used.
    """
    if core_budget < 0 or max_steps < 0 or branch < 1 or beam_width < 1:
        raise ValueError("invalid search budget")
    problem = public_problem(problem)
    model.eval()
    calls = transitions = serial = probability_terms = 0
    transition_budget = core_budget * problem.n_actions
    # Heap ordering uses a deterministic serial to break probability ties.
    frontier = [(0.0, serial, problem.initial, ())]
    open_best = {problem.initial: (0.0, serial)}
    closed: set[world.State] = set()
    discovered = {problem.initial}
    rule_groups: dict[tuple, list[int]] = defaultdict(list)
    for action, recipe in enumerate(zip(problem.consume, problem.produce)):
        rule_groups[recipe].append(action)
    best_plan: tuple[int, ...] = ()
    best_state = problem.initial
    reason = "frontier_exhausted"
    synchronize(device)
    start = time.perf_counter()
    while frontier:
        cost, entry_serial, state, plan = heapq.heappop(frontier)
        if open_best.get(state) != (cost, entry_serial) or state in closed:
            continue
        del open_best[state]
        if len(plan) >= len(best_plan):
            best_plan, best_state = plan, state
        if world.is_goal(problem, state):
            reason = "goal"
            break
        if len(plan) >= max_steps:
            continue
        legal_actions = world.applicable_actions(problem, state)
        if not legal_actions:
            closed.add(state)
            continue
        if calls >= core_budget or transitions >= transition_budget:
            reason = "core_budget" if calls >= core_budget else "transition_budget"
            break
        features, legal = tensorize([problem], [state], device=device)
        logits = model(features, legal, rounds=1)[0].masked_fill(~legal[0], -torch.inf)
        log_probs = torch.log_softmax(logits, dim=-1).tolist()
        calls += 1
        closed.add(state)
        probability_terms += len(legal_actions)
        legal_set = set(legal_actions)
        grouped = []
        for actions in rule_groups.values():
            if actions[0] in legal_set:
                maximum = max(log_probs[action] for action in actions)
                group_log_prob = maximum + math.log(sum(
                    math.exp(log_probs[action] - maximum) for action in actions))
                grouped.append((group_log_prob, actions[0]))
        ordered = sorted(grouped, key=lambda value: (-value[0], value[1]))
        local_successors = set()
        found_goal = False
        for group_log_prob, action in ordered:
            if transitions >= transition_budget:
                reason = "transition_budget"
                break
            following = world.step(problem, state, action)
            transitions += 1
            if following in local_successors:
                continue
            local_successors.add(following)
            following_plan = plan + (action,)
            if world.is_goal(problem, following):
                best_plan, best_state = following_plan, following
                reason, found_goal = "goal", True
                break
            following_cost = cost - group_log_prob
            if following not in closed and (following not in open_best
                                             or following_cost < open_best[following][0]):
                discovered.add(following)
                serial += 1
                open_best[following] = (following_cost, serial)
                heapq.heappush(frontier, (following_cost, serial, following, following_plan))
            if len(local_successors) >= branch:
                break
        if found_goal:
            break
        if len(frontier) > beam_width:
            live = [entry for entry in frontier
                    if open_best.get(entry[2]) == (entry[0], entry[1])]
            frontier = heapq.nsmallest(beam_width, live)
            # A pruned, unexpanded state can be discovered along a later path.
            open_best = {entry[2]: (entry[0], entry[1]) for entry in frontier}
            heapq.heapify(frontier)
    synchronize(device)
    result = finish_rollout(problem, best_plan, core_calls=calls,
                            transition_calls=transitions,
                            latency_ms=(time.perf_counter() - start) * 1000,
                            stop_reason=reason, simulator_state=best_state)
    result.update(latency_kind="single_case_wall", queried_distinct_states=calls,
                  visited_states=len(discovered), transition_budget=transition_budget,
                  branch=branch, beam_width=beam_width,
                  recipe_grouping_rule_checks=problem.n_actions,
                  group_probability_terms=probability_terms)
    return result


def random_rollout(problem: world.Problem, *, seed: int, max_steps: int = 24) -> dict:
    problem = public_problem(problem)
    rng = Random(seed)
    state, plan = problem.initial, []
    start = time.perf_counter()
    reason = "max_steps"
    for _ in range(max_steps):
        if world.is_goal(problem, state):
            reason = "goal"
            break
        legal = world.applicable_actions(problem, state)
        if not legal:
            reason = "dead_end"
            break
        action = rng.choice(legal)
        state = world.step(problem, state, action)
        plan.append(action)
    if world.is_goal(problem, state):
        reason = "goal"
    return finish_rollout(problem, plan, core_calls=0, transition_calls=len(plan),
                          latency_ms=(time.perf_counter() - start) * 1000,
                          stop_reason=reason, simulator_state=state)


def heuristic_rollout(problem: world.Problem, *, mode: str, max_steps: int) -> dict:
    from .heuristics import heuristic_plan
    problem = public_problem(problem)
    start = time.perf_counter()
    stats = heuristic_plan(problem, mode=mode, max_steps=max_steps, return_stats=True)
    plan = stats["plan"]
    result = finish_rollout(problem, plan, core_calls=0,
                            transition_calls=int(stats["transition_calls"]),
                            latency_ms=(time.perf_counter() - start) * 1000,
                            stop_reason=stats.get("termination_reason", "stopped"))
    result["heuristic_stats"] = {key: value for key, value in stats.items() if key != "plan"}
    result["latency_kind"] = "single_case_wall"
    return result


def paired_subset(records: Sequence[dict], limit: int) -> list[dict]:
    """Round-robin groups so the same bounded search cases cover each group."""
    if limit < 0:
        raise ValueError("subset limit cannot be negative")
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        buckets[(record["family"], record["depth"])].append(record)
    result = []
    groups = sorted(buckets)
    offset = 0
    while len(result) < limit:
        added = False
        for group in groups:
            if offset < len(buckets[group]) and len(result) < limit:
                result.append(buckets[group][offset])
                added = True
        if not added:
            break
        offset += 1
    return result


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / count
                           + z * z / (4 * count * count)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def aggregate_results(rows: Sequence[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["method"], "all", "all", "all")].append(row)
        buckets[(row["method"], row["distribution"], "all", "all")].append(row)
        buckets[(row["method"], row["distribution"], row["family"], row["depth"])].append(row)
    results = []
    for (method, split, family, depth), values in sorted(buckets.items(), key=lambda pair: str(pair[0])):
        count = len(values)
        success = sum(bool(value["success"]) for value in values)
        optimal = sum(value["success"] and value["plan_length"] == value["optimal_length"]
                      for value in values)
        extra = [value["plan_length"] - value["optimal_length"]
                 for value in values if value["success"]]
        mean = lambda key: sum(value[key] for value in values) / count
        results.append({"method": method, "distribution": split, "family": family,
                        "depth": depth, "n": count, "success_rate": success / count,
                        "success_wilson95": wilson(success, count),
                        "optimal_success_rate": optimal / count,
                        "legal_plan_rate": mean("valid_plan"),
                        "mean_core_calls": mean("core_calls"),
                        "mean_transition_calls": (mean("transition_calls") if all(
                            value["transition_calls"] is not None for value in values) else None),
                        "mean_latency_ms": mean("latency_ms"),
                        "mean_success_extra_steps": sum(extra) / len(extra) if extra else None,
                        "exact_oracle_success_upper_bound": 1.0})
    return results


def configure_runtime(seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment(device: str) -> dict:
    result = {"python": sys.version, "platform": platform.platform(),
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "device": device,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "threads": torch.get_num_threads(), "command": sys.argv}
    if torch.device(device).type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(torch.device(device))
        result["device_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    for key, command in (("git_head", ["git", "rev-parse", "HEAD"]),
                         ("git_status", ["git", "status", "--short"])):
        try:
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                       capture_output=True, text=True, timeout=10, check=False)
            result[key] = completed.stdout.strip()[:64000]
        except (OSError, subprocess.TimeoutExpired):
            result[key] = None
    return result


def freeze_experiment(out: Path, config: dict, device: str) -> dict:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    write_json(out / "environment.json", environment(device))
    source_dir = out / "source"
    source_dir.mkdir()
    sources = {}
    for filename in SOURCE_FILES:
        source = Path(__file__).with_name(filename)
        if source.exists():
            shutil.copyfile(source, source_dir / filename)
            sources[filename] = file_sha256(source)
    datasets = {}
    for split, count in (("train", config["train_problems"]),
                         ("val", config["val_per_group"]),
                         ("test", config["test_per_group"])):
        datasets[split] = generate_dataset(out / f"{split}.jsonl",
                                           split_specs(split, count, config["seed"]),
                                           config, training=split == "train")
    manifest = {"schema_version": SCHEMA_VERSION,
                "config_sha256": file_sha256(out / "config.json"),
                "environment_sha256": file_sha256(out / "environment.json"),
                "sources": sources, "datasets": datasets,
                "policy_input": ["inventory", "goal", "consume", "produce", "current_legality"],
                "labels": "all optimal first actions; train only",
                "state_sampling": "uniform problem; 50% initial, 50% sampled other state",
                "halting": "simulator goal predicate, not a learned stop action",
                "budget_matching": "core-layer example count, not exact FLOPs or wall time"}
    write_json(out / "manifest.json", manifest)
    verify_experiment(out)
    return manifest


def verify_experiment(out: Path) -> tuple[dict, dict]:
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported experiment schema")
    if file_sha256(out / "config.json") != manifest["config_sha256"]:
        raise ValueError("frozen configuration was modified")
    if file_sha256(out / "environment.json") != manifest["environment_sha256"]:
        raise ValueError("recorded training environment was modified")
    present_sources = {filename for filename in SOURCE_FILES
                       if Path(__file__).with_name(filename).exists()}
    if present_sources != set(manifest["sources"]):
        raise ValueError("source-file set changed since experiment freeze")
    for filename, expected in manifest["sources"].items():
        if file_sha256(Path(__file__).with_name(filename)) != expected:
            raise ValueError(f"source changed since experiment freeze: {filename}")
        if file_sha256(out / "source" / filename) != expected:
            raise ValueError(f"source snapshot changed: {filename}")
    for split, data in manifest["datasets"].items():
        if file_sha256(out / data["file"]) != data["sha256"]:
            raise ValueError(f"frozen {split} dataset was modified")
    return config, manifest


def checkpoint(path: Path, model: Planner, *, method: str, config_sha: str,
               optimizer_steps: int, external_update: int, seed: int) -> None:
    torch.save({"method": method, "n_items": model.n_items, "width": model.width,
                "rounds": model.rounds, "state_dict": model.state_dict(),
                "config_sha256": config_sha, "optimizer_steps": optimizer_steps,
                "external_update": external_update, "seed": seed,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters())}, path)


def load_models(out: Path, config: dict, manifest: dict, device: str) -> dict[str, Planner]:
    models = {}
    for method in config["methods"]:
        saved = torch.load(out / f"{method}.pt", map_location=device, weights_only=True)
        if saved["config_sha256"] != manifest["config_sha256"] or saved["method"] != method:
            raise ValueError(f"checkpoint does not belong to the frozen config: {method}")
        model = Planner(n_items=saved["n_items"], width=saved["width"],
                        rounds=saved["rounds"], kind=method).to(device)
        model.load_state_dict(saved["state_dict"])
        models[method] = model.eval()
    return models


def make_row(record: dict, method: str, result: dict) -> dict:
    offline = {key: record[key] for key in ("case_id", "seed", "family", "depth",
                                           "distribution", "optimal_length")}
    return dict(offline, method=method, **result)


def evaluate_models(models: dict[str, Planner], records: Sequence[dict], config: dict,
                    device: str, *, include_search: bool) -> list[dict]:
    problems = [problem_from_record(record) for record in records]
    rows = []
    for method, model in models.items():
        round_counts = ([1] if method == "shallow" else [config["rounds"]]
                        if method == "untied" or not include_search
                        else sorted({1, 2, config["rounds"], 2 * config["rounds"]}))
        for rounds in round_counts:
            results = greedy_rollouts(model, problems, rounds=rounds, device=device,
                                       core_budget=config["core_budget"],
                                       max_steps=config["max_steps"],
                                       batch_size=config["eval_batch_size"])
            name = f"{method}_r{rounds}"
            rows.extend(make_row(record, name, result) for record, result in zip(records, results))
            print(f"eval {name}: {sum(result['success'] for result in results)}/{len(results)}",
                  flush=True)
    if include_search:
        subset = paired_subset(records, config["beam_limit"])
        for budget in config["beam_budgets"]:
            for record in subset:
                result = neural_best_first(
                    models["shallow"], problem_from_record(record), device=device,
                    core_budget=budget, max_steps=config["max_steps"],
                    branch=config["beam_branch"], beam_width=config["beam_width"],
                )
                rows.append(make_row(record, f"shallow_bestfirst_c{budget}", result))
            print(f"eval shallow_bestfirst_c{budget}: {len(subset)} paired cases", flush=True)
        heuristic_available = Path(__file__).with_name("heuristics.py").exists()
        for record, problem in zip(records, problems):
            rows.append(make_row(record, "random", random_rollout(
                problem, seed=record["seed"] ^ 0xD91E, max_steps=config["max_steps"])))
            if heuristic_available:
                for mode in ("relaxed", "fanout"):
                    rows.append(make_row(record, f"heuristic_{mode}", heuristic_rollout(
                        problem, mode=mode, max_steps=config["max_steps"])))
    return rows


def evaluate(out: Path, *, split: str, device: str, beam_limit: int | None = None) -> None:
    config, manifest = verify_experiment(out)
    if beam_limit is not None and beam_limit != config["beam_limit"]:
        raise ValueError("--beam-limit must equal the frozen training configuration")
    configure_runtime(config["seed"], config["threads"])
    models = load_models(out, config, manifest, device)
    records = load_records(out / manifest["datasets"][split]["file"])
    # Atomic creation prevents accidental repeated exposure to the held-out set.
    marker = out / f"evaluation_{split}.started.json"
    with marker.open("x", encoding="utf-8") as stream:
        stream.write(json_text({"split": split, "config_sha256": manifest["config_sha256"],
                                "environment": environment(device)}) + "\n")
    start = time.perf_counter()
    rows = evaluate_models(models, records, config, device, include_search=True)
    with (out / f"results_{split}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json_text(row) + "\n")
    subset_ids = {record["case_id"] for record in paired_subset(records, config["beam_limit"])}
    summary = {"split": split, "config_sha256": manifest["config_sha256"],
               "seconds": time.perf_counter() - start,
               "all_cases": aggregate_results(rows),
               "paired_search_subset_ids": sorted(subset_ids),
               "paired_search_subset": aggregate_results(
                   [row for row in rows if row["case_id"] in subset_ids]),
               "notes": ["Final plans independently executed and validated.",
                         "Exact solver metadata used only for reporting optimal length.",
                         "Simulator supplies legality, transitions, and goal stopping.",
                         "Greedy latency is apportioned batched wall time; search latency is sequential.",
                         "Core counts exclude encoding, heads, optimizer, simulation, and validation costs.",
                         "Beam sums equivalent recipe probabilities before expanding distinct successors.",
                         "Recipe grouping checks and probability terms are reported separately from transitions."]}
    write_json(out / f"summary_{split}.json", summary)
    print(f"saved {out / f'summary_{split}.json'}", flush=True)


def train(config: dict, *, out: Path, device: str, run_test: bool = False) -> None:
    configure_runtime(config["seed"], config["threads"])
    manifest = freeze_experiment(out, config, device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records = load_records(out / "train.jsonl")
    pool = build_pool(records, device=device)
    del records
    validation = load_records(out / "val.jsonl")
    models, optimizers, steps, train_seconds = {}, {}, {}, {}
    for method in config["methods"]:
        # Identical seeds share encoder, head, and first-core initialization.
        torch.manual_seed(config["seed"])
        model = Planner(n_items=config["n_items"], width=config["width"],
                        rounds=config["rounds"], kind=method).to(device)
        models[method] = model
        optimizers[method] = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                              weight_decay=config["weight_decay"])
        steps[method], train_seconds[method] = 0, 0.0
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] ^ 0x41AF)
    batch_digest = hashlib.sha256()
    start = time.perf_counter()
    with (out / "curves.jsonl").open("x", encoding="utf-8", buffering=1) as curves, \
            (out / "validation.jsonl").open("x", encoding="utf-8", buffering=1) as val_stream:
        for external in range(1, config["updates"] + 1):
            indices = sample_batch_indices(pool, config["batch_size"], generator)
            batch_digest.update(indices.numpy().tobytes())
            selected = indices.to(device)
            features, legal, targets = (value.index_select(0, selected)
                                        for value in (pool.features, pool.legal, pool.targets))
            last_losses = {}
            for method, model in models.items():
                model.train()
                optimizer = optimizers[method]
                repetitions = config["rounds"] if method == "shallow" else 1
                losses, hits = [], []
                synchronize(device)
                method_start = time.perf_counter()
                for _ in range(repetitions):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features, legal)
                    loss = optimal_set_loss(logits, targets)
                    losses.append(float(loss.detach()))
                    hits.append(float(targets.gather(1, logits.argmax(-1, keepdim=True)).float().mean()))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    optimizer.step()
                    steps[method] += 1
                    if method == "shallow" and steps[method] == config["updates"]:
                        checkpoint(out / "shallow_equal_updates.pt", model, method=method,
                                   config_sha=manifest["config_sha256"], optimizer_steps=steps[method],
                                   external_update=external, seed=config["seed"])
                synchronize(device)
                train_seconds[method] += time.perf_counter() - method_start
                last_losses[method] = sum(losses) / len(losses)
                curves.write(json_text({"method": method, "external_update": external,
                                        "optimizer_steps": steps[method],
                                        "loss": last_losses[method],
                                        "optimal_action_accuracy": sum(hits) / len(hits),
                                        "core_layer_examples": external * config["batch_size"] * config["rounds"],
                                        "train_seconds": train_seconds[method],
                                        "paired_batch_sha256_so_far": batch_digest.hexdigest()}) + "\n")
            if external == 1 or external % config["log_every"] == 0 or external == config["updates"]:
                print(f"update {external}/{config['updates']} " + " ".join(
                    f"{method}={loss:.4f}" for method, loss in last_losses.items()), flush=True)
            if config["val_every"] and (external % config["val_every"] == 0
                                          or external == config["updates"]):
                rows = evaluate_models(models, validation, config, device, include_search=False)
                val_stream.write(json_text({"external_update": external,
                                            "groups": aggregate_results(rows)}) + "\n")
    verify_experiment(out)
    for method, model in models.items():
        checkpoint(out / f"{method}.pt", model, method=method,
                   config_sha=manifest["config_sha256"], optimizer_steps=steps[method],
                   external_update=config["updates"], seed=config["seed"])
    write_json(out / "training_summary.json", {
        "seconds": time.perf_counter() - start, "optimizer_steps": steps,
        "train_seconds": train_seconds,
        "parameter_counts": {method: sum(parameter.numel() for parameter in model.parameters())
                             for method, model in models.items()},
        "core_layer_examples_each": config["updates"] * config["batch_size"] * config["rounds"],
        "paired_batch_sha256": batch_digest.hexdigest(),
        "sampling_seed": config["seed"] ^ 0x41AF,
        "training_states": len(pool.features),
        "feature_cache_bytes": pool.features.numel() * pool.features.element_size(),
        "feature_cache_device": str(pool.features.device),
        "peak_cuda_memory_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                              if torch.device(device).type == "cuda" else None),
        "notes": ["All methods use exactly the same batch indices at each external update.",
                  "Shallow repeats the same batch; recurrent/untied perform one optimizer update.",
                  "Equal core-layer examples do not imply equal FLOPs or wall time.",
                  "shallow_equal_updates.pt stops after as many optimizer updates as final recurrent.",
                  "Final checkpoints are used; validation does not select checkpoints."]})
    print(f"training complete: {out}", flush=True)
    # Free feature cache before the optional full evaluation.
    del pool, models, optimizers
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    if run_test:
        evaluate(out, split="test", device=device)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="freeze data/config, train; test only with --test")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--train-problems", type=int, default=4096)
    training.add_argument("--val-per-group", type=int, default=32)
    training.add_argument("--test-per-group", type=int, default=64)
    training.add_argument("--updates", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--width", type=int, default=128)
    training.add_argument("--rounds", type=int, default=4)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--include-detach", action="store_true")
    training.add_argument("--states-per-problem", type=int, default=8)
    training.add_argument("--solver-max-states", type=int, default=50_000)
    training.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    training.add_argument("--log-every", type=int, default=25)
    training.add_argument("--val-every", type=int, default=200, help="0 disables training validation")
    training.add_argument("--eval-batch-size", type=int, default=128)
    training.add_argument("--core-budget", type=int, default=96)
    training.add_argument("--max-steps", type=int, default=24)
    training.add_argument("--beam-budget", type=int, nargs="+", default=[32, 96])
    training.add_argument("--beam-limit", type=int, default=64)
    training.add_argument("--beam-branch", type=int, default=4)
    training.add_argument("--beam-width", type=int, default=32)
    training.add_argument("--test", action="store_true", help="one-shot final held-out evaluation")
    evaluation = commands.add_parser("evaluate", help="one-shot evaluation of frozen checkpoints")
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluation.add_argument("--beam-limit", type=int,
                            help="optional assertion; must match the frozen configuration")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "evaluate":
        evaluate(args.out.resolve(), split=args.split, device=args.device, beam_limit=args.beam_limit)
        return
    config = vars(args).copy()
    for key in ("command", "out", "device", "test"):
        config.pop(key)
    config["beam_budgets"] = sorted(set(config.pop("beam_budget")))
    config.update(methods=["shallow", "recurrent", "untied"] + (["detach"] if args.include_detach else []),
                  n_items=40, n_rules=32, schema_version=SCHEMA_VERSION)
    config.pop("include_detach")
    positive = ("train_problems", "val_per_group", "test_per_group", "updates", "batch_size",
                "width", "rounds", "states_per_problem", "solver_max_states", "threads",
                "log_every", "eval_batch_size", "core_budget", "max_steps", "beam_branch", "beam_width")
    if any(config[key] < 1 for key in positive) or config["width"] % 4:
        raise ValueError("counts must be positive and width must be divisible by four")
    if config["seed"] < 0 or config["beam_limit"] < 0 or config["val_every"] < 0:
        raise ValueError("seed, beam-limit, and val-every must be nonnegative")
    if any(budget < 1 for budget in config["beam_budgets"]):
        raise ValueError("beam budgets must be positive")
    if config["lr"] <= 0 or config["weight_decay"] < 0 or config["grad_clip"] <= 0:
        raise ValueError("invalid optimizer hyperparameters")
    train(config, out=args.out.resolve(), device=args.device, run_test=args.test)


if __name__ == "__main__":
    main()
