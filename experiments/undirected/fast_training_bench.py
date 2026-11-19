"""Paired full-life benchmark on archived recipes, with exact trace/state validation."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import median
import time

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import local_rule_evaluator as reference
from . import local_rule_genetics as genetics
from . import local_rule_run as run
from .fast_training_evaluator import FastEvaluation


def evaluator(name):
    if name == "reference":
        return reference.Evaluation("cuda", "training")
    family, size = name.split("-")
    return FastEvaluation("cuda", "training", backend=family,
                          requests=4 if family == "arcs" else int(size), tiles=int(size) if family == "arcs" else 1)


def archived_inputs(root, arm, generation, count):
    folder = root / "arms" / arm / "generations"
    parents = torch.load(folder / f"{generation-1:04d}" / "elites.pt", weights_only=True, map_location="cpu")
    recipes = json.loads((folder / f"{generation:04d}" / "plan.json").read_text(encoding="utf-8"))
    # Cover alternating islands, ordinary offspring and protected contests.
    indices = np.linspace(0, len(recipes) - 1, count, dtype=int)
    return [genetics.candidate(recipes[i], parents, 96) for i in indices], [recipes[i]["id"] for i in indices]


def reference_trace(e, genes, arm):
    replay, = e.replays.values()
    tape, policy = e.tape, replay.policy
    policy.reset(reference.stack_birth(genes, e.device, arm))
    replay.receipts.zero_()
    replay.flags.zero_()
    replay.counts.zero_()
    trace = replay.trace.new_empty((len(tape.raw), *replay.trace.shape))
    for i in range(len(tape.raw)):
        replay.frame.copy_(tape.public[i])
        replay.stage.copy_(tape.stage[i])
        replay.target.copy_(tape.target[i])
        replay.graph.replay()
        trace[i].copy_(replay.trace)
    return trace


def state_digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def paired_ci(base, times):
    ratios = np.array(base) / np.array(times)
    draws = np.random.default_rng(547).integers(len(ratios), size=(10000, len(ratios)))
    return [float(np.median(ratios)), *np.quantile(np.median(ratios[draws], axis=1), [.025, .975]).tolist()]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=["chunks-4"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--generation", type=int, default=88)
    p.add_argument("--arms", nargs="+", default=["local", "modulated"])
    args = p.parse_args()
    torch.set_num_threads(1)
    names = ["reference", *args.variants]
    protocol = json.loads((run.OUT / "protocol.json").read_text(encoding="utf-8"))
    devseed = protocol["seed"] + protocol["development_seed_offset"] + args.generation
    report = {"generation": args.generation, "genomes_per_batch": args.batch, "requests_per_life": 64,
              "actors": 8, "gpu": torch.cuda.get_device_name(), "development_seed": devseed, "arms": {}}
    order_rng = random.Random(8291)
    for arm in args.arms:
        genes, ids = archived_inputs(run.OUT, arm, args.generation, args.batch)
        evs = {name: evaluator(name) for name in names}
        timings = {name: [] for name in names}
        checks = {}
        try:
            for seed in (devseed, devseed + 310_000_003):
                base = evs["reference"]
                official = base(genes, arm=arm, seed=seed)
                expected = reference_trace(base, genes, arm)
                ref, = base.replays.values()
                state = ref.policy.state_tensors()
                pool = {}
                for name in args.variants:
                    candidate = evs[name]
                    rows = candidate(genes, arm=arm, seed=seed)
                    trial, = candidate.replays.values()
                    outputs = trial.policy.state_tensors()
                    numeric_diff = sorted({k for a, b in zip(official, rows, strict=True) for k in a.keys() | b.keys()
                                           if a.get(k) != b.get(k)})
                    different = [key for key in state if not torch.equal(state[key], outputs[key])]
                    action_diff = int((expected != trial.trace).sum().item())
                    pool[name] = dict(mismatched_metrics=numeric_diff, mismatched_states=different,
                                      mismatched_primary_actions=action_diff)
                    if different or numeric_diff or action_diff:
                        raise ValueError(f"parity failed on {arm}/{seed}/{name}: {pool[name]}")
                checks[str(seed)] = {"primary_actions": expected.numel(), "state_fields": len(state),
                                     "metrics": len(official[0]), "trace_sha256": state_digest(expected), "variants": pool}
            # Randomized pairs of reversed run orders balance short-term clock drift.
            for pair in range((args.rounds + 1) // 2):
                order = names.copy()
                order_rng.shuffle(order)
                for turn, series in enumerate((order, list(reversed(order)))):
                    if 2 * pair + turn >= args.rounds:
                        break
                    for name in series:
                        start = time.perf_counter()
                        evs[name](genes, arm=arm, seed=devseed)
                        torch.cuda.synchronize()
                        timings[name].append(time.perf_counter() - start)
            results = {name: {"seconds": times, "candidates_per_second": args.batch/median(times),
                              "paired_speed_ratio_and_ci95": paired_ci(timings["reference"], times)}
                       for name, times in timings.items()}
            report["arms"][arm] = {"ids": ids, "checks": checks, "timings": results}
            _write(args.out, report)
            print(arm, {name: round(x["candidates_per_second"],2) for name,x in results.items()}, flush=True)
        finally:
            for ev in evs.values():
                ev.clear_cache()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
