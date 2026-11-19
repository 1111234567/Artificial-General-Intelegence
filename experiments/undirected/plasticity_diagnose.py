"""Replay the v1 champion on its published holdout with action interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from experiments.seed_evolution.large_runner import _write

from .evolved_plasticity import INPUTS, GraphBatch
from .evolved_plasticity_world import (
    FIRST_FINISH, N_ACTIONS, N_SLOTS, N_SYMBOLS, N_VERSIONS, PHASES, READY, VERSION, WAIT,
    make_schedule, rollout,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "data/undirected/2026-11-16-evolved-plasticity-v1"
DEFAULT_OUTPUT = ROOT / "data/undirected/2026-11-16-evolved-plasticity-diagnostic/holdout.json"


def finish_code(action):
    return (action >= FIRST_FINISH) & (action < N_ACTIONS)


class HoldoutPolicy:
    """Wrap the native graph policy and record the actions executed by the world."""

    def __init__(self, graph=None, *, timing=False, version=False, direct_cue=False,
                 guess=None, second_guess=False):
        self.graph = graph
        self.timing = timing
        self.version = version
        self.direct_cue = direct_cue
        self.guess = guess
        self.second_guess = second_guess
        self.previous_ready = None
        self.executed = []
        self.readiness = []
        self.direct_changes = []

    def act(self, obs):
        ready = obs[..., READY].bool()
        version = obs[..., VERSION].argmax(-1)
        first_ready = ready if self.previous_ready is None else ready & ~self.previous_ready
        if self.graph is None:
            action = torch.zeros((*ready.shape, N_SLOTS), dtype=torch.long, device=obs.device)
            if self.second_guess:
                symbol = (self.guess + (~first_ready).long()) % N_SYMBOLS
                action[..., 0] = torch.where(ready, FIRST_FINISH + N_SYMBOLS * version + symbol, WAIT)
            else:
                action[..., 0] = torch.where(
                    first_ready, FIRST_FINISH + N_SYMBOLS * version + self.guess, WAIT,
                )
        else:
            action = self.graph.act(obs)
            if self.direct_cue:
                native = action[..., 0].clone()
                action[..., 0] = self.same_tick_cue_readout(obs)
                self.direct_changes.append((action[..., 0] != native).squeeze(0))
            if self.timing:
                scores = self.graph.h[..., INPUTS + FIRST_FINISH:INPUTS + N_ACTIONS]
                action[..., 0] = torch.where(first_ready, FIRST_FINISH + scores.argmax(-1), WAIT)
            if self.version:
                action0 = action[..., 0]
                symbol = (action0 - FIRST_FINISH) % N_SYMBOLS
                repaired = FIRST_FINISH + N_SYMBOLS * version + symbol
                action[..., 0] = torch.where(ready & finish_code(action0), repaired, action0)
        self.previous_ready = ready
        self.readiness.append(ready.squeeze(0))
        self.executed.append(action.squeeze(0))
        return action

    def same_tick_cue_readout(self, obs):
        """Let the increment from current version/READY traverse direct edges in this readout."""
        graph = self.graph
        indices = [*range(VERSION.start, VERSION.stop), READY]
        input_state = graph.h[..., indices]
        without_cue = input_state - .25 * obs[..., indices]
        cue_delta = graph.activity[..., indices] - without_cue.tanh()
        edges = graph.matrix[..., INPUTS:INPUTS + N_ACTIONS, indices]
        scores = graph.h[..., INPUTS:INPUTS + N_ACTIONS]
        return (scores + .25 * (edges * cue_delta.unsqueeze(-2)).sum(-1)).argmax(-1)


def trace_counts(policy, schedule, completion, announced_waits):
    requests, ticks, actors = schedule.stage.shape
    ready = torch.stack(policy.readiness).reshape(requests, ticks, actors)
    action = torch.stack(policy.executed).reshape(requests, ticks, actors, N_SLOTS)[..., 0]
    target = schedule.target.unsqueeze(1)
    ready_finish = ready & finish_code(action)
    ready_wait = ready & (action == WAIT)
    version_correct = ((action - FIRST_FINISH) // N_SYMBOLS) == ((target - FIRST_FINISH) // N_SYMBOLS)
    symbol_correct = ((action - FIRST_FINISH) % N_SYMBOLS) == ((target - FIRST_FINISH) % N_SYMBOLS)
    accepted = ready_finish & version_correct & symbol_correct
    open_ticks = (schedule.stage == 2).cumsum(1)
    early_finish = (schedule.stage == 1) & finish_code(action)
    direct_changes = None
    if policy.direct_cue:
        direct_changes = torch.stack(policy.direct_changes).reshape_as(ready)

    if not math.isclose(float(accepted.any(1).float().mean()), completion, rel_tol=0., abs_tol=1e-7):
        raise ValueError("recorded FINISH acceptance disagrees with the original rollout")

    def count(mask):
        contract_mask = mask.unsqueeze(1)

        def contracts(events):
            return int((mask & events.any(1)).sum())

        def ticks_count(events):
            return int((contract_mask & events).sum())

        result = {
            "contracts": int(mask.sum()),
            "contracts_with_ready_finish": contracts(ready_finish),
            "contracts_with_ready_nonwait": contracts(ready & ~ready_wait),
            "contracts_all_ready_wait": int((mask & (~ready | ready_wait).all(1)).sum()),
            "contracts_with_current_version_ready_finish": contracts(ready_finish & version_correct),
            "contracts_with_early_finish": contracts(early_finish),
            "contracts_completed": contracts(accepted),
            "ready_ticks": ticks_count(ready),
            "ready_wait_ticks": ticks_count(ready_wait),
            "ready_finish_ticks": ticks_count(ready_finish),
            "ready_current_version_finish_ticks": ticks_count(ready_finish & version_correct),
            "ready_correct_symbol_finish_ticks_any_version": ticks_count(ready_finish & symbol_correct),
            "ready_correct_symbol_finish_ticks_current_version": ticks_count(accepted),
        }
        if direct_changes is not None:
            result["readout_changes"] = ticks_count(direct_changes)
            result["readout_changes_on_ready"] = ticks_count(direct_changes & ready)
        for tick in range(1, schedule.finish_window + 1):
            opportunity = ready & (open_ticks == tick)
            finish = ready_finish & opportunity
            result[f"window_tick_{tick}"] = {
                "ready_ticks": ticks_count(opportunity),
                "ready_finish_ticks": ticks_count(finish),
                "ready_wait_ticks": ticks_count(ready_wait & opportunity),
                "ready_current_version_finish_ticks": ticks_count(finish & version_correct),
                "ready_correct_symbol_finish_ticks_current_version": ticks_count(accepted & opportunity),
            }
        return result

    all_contracts = torch.ones_like(schedule.target, dtype=torch.bool)
    phases = torch.arange(requests, device=ready.device) // schedule.requests_per_phase
    announced_wait = schedule.public[..., 0, 9].sum(1).long()
    return {
        "overall": count(all_contracts),
        "by_phase": {name: count(all_contracts & (phases[:, None] == i)) for i, name in enumerate(PHASES)},
        "by_version": {
            str(i): count(((schedule.target - FIRST_FINISH) // N_SYMBOLS) == i)
            for i in range(N_VERSIONS)
        },
        "by_announced_wait": {
            str(wait): count(announced_wait == wait)
            for wait in announced_waits
        },
    }


def merge_counts(rows):
    return {
        key: merge_counts([row[key] for row in rows]) if isinstance(value, dict)
        else sum(row[key] for row in rows)
        for key, value in rows[0].items()
    }


def ratios(count):
    numbers = {
        "contract_ready_finish_coverage": ("contracts_with_ready_finish", "contracts"),
        "contract_all_ready_wait": ("contracts_all_ready_wait", "contracts"),
        "contract_current_version_ready_finish_coverage": (
            "contracts_with_current_version_ready_finish", "contracts",
        ),
        "completion_given_current_version_ready_finish_contract": (
            "contracts_completed", "contracts_with_current_version_ready_finish",
        ),
        "current_version_given_ready_finish_tick": (
            "ready_current_version_finish_ticks", "ready_finish_ticks",
        ),
        "correct_symbol_given_current_version_ready_finish_tick": (
            "ready_correct_symbol_finish_ticks_current_version", "ready_current_version_finish_ticks",
        ),
        "correct_symbol_given_any_ready_finish_tick": (
            "ready_correct_symbol_finish_ticks_any_version", "ready_finish_ticks",
        ),
    }
    return {
        key: count[numerator] / count[denominator] if count[denominator] else None
        for key, (numerator, denominator) in numbers.items()
    }


def inspect_birth_cue_edges(genome):
    """Count direct edges between public cue inputs and first-port action nodes."""
    nodes = genome["node_bias"].numel()
    u, v = torch.triu_indices(nodes, nodes, offset=1)
    action = (INPUTS <= v) & (v < INPUTS + N_ACTIONS)
    return {
        name: int((genome["birth_mask"] & (u == port) & action).sum())
        for name, port in {"version_0": VERSION.start, "version_1": VERSION.start + 1, "ready": READY}.items()
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_diagnostic(run, device):
    config = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    published = json.loads((run / "holdout.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "champion.pt", map_location="cpu", weights_only=True)
    genome = checkpoint["genome"]
    candidate = checkpoint["row"]["id"]
    actors = config["holdout_actors"]
    seeds = config["holdout_world_seeds"]
    if (candidate, actors, seeds) != (published["candidate"], published["actors"], published["seeds"]):
        raise ValueError("published holdout does not match the supplied protocol and checkpoint")
    source = Path(__file__).resolve().parent
    for name, expected in config["sources"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"local source differs from the v1 protocol: {name}")

    graph_arms = {
        "native": {},
        "timing_only": {"timing": True},
        "version_only": {"version": True},
        "timing_and_version": {"timing": True, "version": True},
        "same_tick_cue_readout": {"direct_cue": True},
    }
    fixed_arms = {
        f"fixed_guess_symbol_{symbol}": {"guess": symbol} for symbol in range(N_SYMBOLS)
    }
    fixed_arms["two_fixed_guesses"] = {"guess": 0, "second_guess": True}
    results = {name: [] for name in (*graph_arms, *fixed_arms)}
    baseline_max_delta = 0.
    published_native = {row["seed"]: row for row in published["paired_scores"]["full"]}
    announced_waits = range(config["min_wait"], config["max_wait"] + 1)
    for seed in seeds:
        schedule = make_schedule(
            seed=seed, actors=actors, device=device, requests_per_phase=config["requests_per_phase"],
            min_wait=config["min_wait"], max_wait=config["max_wait"],
        )
        for name, options in (graph_arms | fixed_arms).items():
            graph = GraphBatch([genome], actors=actors, device=device, seed=seed) if name in graph_arms else None
            policy = HoldoutPolicy(graph, **options)
            metrics = {key: value.item() for key, value in rollout(policy, schedule, 1).items()}
            if name == "native":
                max_delta = max(abs(value - published_native[seed][key]) for key, value in metrics.items())
                baseline_max_delta = max(baseline_max_delta, max_delta)
                if max_delta > 1e-7:
                    raise ValueError(f"native trace on {device} differs from published holdout seed {seed}: {max_delta}")
            counts = trace_counts(policy, schedule, metrics["completion"], announced_waits)
            results[name].append({"seed": seed, "metrics": metrics, "counts": counts})

    arms = {}
    for name, rows in results.items():
        counts = merge_counts([row["counts"] for row in rows])
        arms[name] = {
            "mean_metrics": {key: statistics.fmean(row["metrics"][key] for row in rows) for key in rows[0]["metrics"]},
            "counts": counts,
            "overall_rates": ratios(counts["overall"]),
            "per_seed": rows,
        }
    fixed_names = [f"fixed_guess_symbol_{symbol}" for symbol in range(N_SYMBOLS)]
    arms["uniform_fixed_guess"] = {
        "mean_metrics": {
            key: statistics.fmean(arms[name]["mean_metrics"][key] for name in fixed_names)
            for key in arms[fixed_names[0]]["mean_metrics"]
        },
        "method": "Equal mixture of four paired deterministic single-guess arms.",
    }
    return {
        "schema": 1,
        "candidate": candidate,
        "device": str(device),
        "torch_version": torch.__version__,
        "run": str(run),
        "source_hashes": config["sources"],
        "diagnostic_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(run / "champion.pt"),
        "holdout_sha256": sha256(run / "holdout.json"),
        "published_full_behavioral_metrics_max_abs_delta_across_seeds": baseline_max_delta,
        "holdout_actors_per_seed": actors,
        "holdout_world_seeds": seeds,
        "finish_window": schedule.finish_window,
        "birth_cue_edges_to_first_port_actions": inspect_birth_cue_edges(genome),
        "interventions": {
            "native": "Unmodified full-plasticity graph.",
            "timing_only": "Port 0 emits its highest-logit FINISH across both versions at first public READY and WAIT otherwise.",
            "version_only": "When a native FINISH occurs on public READY, set its version to the visible version.",
            "timing_and_version": "Apply timing_only, then version_only to the emitted FINISH.",
            "same_tick_cue_readout": "For port 0, add the current version/READY input increment through existing direct edges to the readout scores. Leave graph state update, learning rule, and timing decision native.",
            "fixed_guess_symbol_N": "Guess symbol N at first public READY in the public version, WAIT otherwise.",
            "two_fixed_guesses": "Guess symbol 0 at first public READY and symbol 1 at the second if still open.",
        },
        "method": "All arms rerun the original world state and receive receipts for their actually executed actions. Graph arms keep port 1 native. Private targets are used by the original world and by diagnostic counting, and policies act only on public observations and graph state.",
        "arms": arms,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    report = run_diagnostic(args.run.resolve(), device)
    _write(args.output, report)
    for name, arm in report["arms"].items():
        score = arm["mean_metrics"]
        print(f"{name:25s} completion={score['completion']:.6f}  strict={score['strict']:.6f}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
