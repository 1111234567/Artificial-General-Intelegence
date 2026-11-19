"""Evolve directed and reciprocal reward-plastic graphs on paired A/B lives."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
from statistics import fmean
import sys

import torch

from . import uncapped_directed_breeding as breeding
from . import uncapped_plasticity_run as runner
from .uncapped_directed_audit import audit
from .uncapped_directed_reward import RewardEvaluation
from .uncapped_island_crossover_run import DELAYED, OUT as ARCHIVE, TIMING, block_ticks
from .uncapped_ready_signal_run import schedule


OUT = ARCHIVE.with_name("2026-11-17-plasticity-directed-reward-v1")
NODES = 96
ARCS = NODES * (NODES - 1)
ISLANDS = 4
WORLD_BLOCKS = 4
SCORING = {
    "first_return_mean": .70, "first_return_worst_context": .20,
    "first_finish_correct": .08, "ready_coverage": .02,
    "wrong_answer_rate": -.03, "premature_request_rate": -.02, "idle": -.01,
}
SOURCE_FILES = (
    "uncapped_directed_reward_run.py", "uncapped_directed_reward.py", "uncapped_directed_breeding.py",
    "uncapped_directed_audit.py", "uncapped_reward_precision.py", "uncapped_bf16_backend.py",
    "uncapped_cuda_backend.py", "uncapped_answer_feedback.py", "uncapped_ready_signal_run.py",
    "plasticity_roomy_world.py", "variable_birth_genome.py", "uncapped_island_recombination.py",
    "uncapped_archived_founders.py",
)


def training_schedule(*, seed, actors, device, visits):
    """Split lives across four independent seeds without adding actors or ticks."""
    per_block, rest = divmod(actors, WORLD_BLOCKS)
    if rest or per_block < 2 or per_block % 2:
        raise ValueError("training needs four equally sized, context-balanced actor blocks")
    seeds = [seed * WORLD_BLOCKS + i for i in range(WORLD_BLOCKS)]
    tapes = [schedule(seed=world_seed, actors=per_block, device=device, visits=visits, timing=TIMING)
             for world_seed in seeds]
    return replace(
        tapes[0],
        **{field: torch.cat([getattr(tape, field) for tape in tapes], dim=2)
           for field in ("public", "stage")},
        **{field: torch.cat([getattr(tape, field) for tape in tapes], dim=1)
           for field in ("target", "context", "raw")},
        maps=torch.cat([tape.maps for tape in tapes], dim=0),
    )


def score_fields(row, max_pairs):
    a = row["first_finish_correct_a_first_return"]
    b = row["first_finish_correct_b_first_return"]
    components = {**row, "first_return_mean": (a + b) / 2, "first_return_worst_context": min(a, b)}
    score = sum(weight * components[key] for key, weight in SCORING.items())
    result = {**components, "selection_score": score, "fitness": row["strict"],
              "average_edges": row["mean_edges"]}
    if min(a, b) > .25:
        result["sparsity_bonus"] = 1 - row["mean_edges"] / max_pairs
    return result


def rank(row):
    return (row["selection_score"], row["first_return_worst_context"], row["first_return_mean"],
            row["strict"], row.get("sparsity_bonus", -1.))


def snapshot(rows):
    result = runner_snapshot(rows)
    partitions = {name: [] for name in ("directed", "reciprocal")}
    for row in rows:
        partitions["directed" if row["directed"] else "reciprocal"].append(row)
    arms = {}
    for name, group in partitions.items():
        if not group:
            continue
        best = max(group, key=rank)
        arms[name] = {
            "candidates": len(group), "best_candidate": best["id"], "selection_score": best["selection_score"],
            "first_return_mean": best["first_return_mean"], "mean_first_return": fmean(row["first_return_mean"] for row in group),
            "strict": best["strict"], "ready_coverage": best["ready_coverage"], "idle": best["idle"],
            "premature_request_rate": best["premature_request_rate"], "final_reciprocity": best["final_reciprocity"],
        }
    return {**result, "topology_groups": arms, "execution_precision": "fp16 / fp32 matmul accumulation"}


runner_snapshot = runner.snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--warm-start", type=int, help="Historical founder slots across all four islands.")
    defaults, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if any(flag in rest for flag in ("-h", "--help")):
        parser.print_help()

    historical = []
    genomes = {}
    train = RewardEvaluation(training_schedule)
    regular = RewardEvaluation(partial(schedule, timing=TIMING))
    regular_padded = RewardEvaluation(partial(schedule, timing=TIMING, pad_to=block_ticks(DELAYED)))
    delayed = RewardEvaluation(partial(schedule, timing=DELAYED))

    def protocol(args):
        if args.islands != ISLANDS or args.population < 16 or args.actors % (2 * WORLD_BLOCKS):
            raise ValueError("Use four islands, at least 16 candidates, and a multiple of eight training actors")
        warm_count = args.population // 16 if defaults.warm_start is None else defaults.warm_start
        warm_count -= warm_count % ISLANDS if defaults.warm_start is None else 0
        if not 0 <= warm_count <= args.population or warm_count % ISLANDS:
            raise ValueError("Warm slots must be a multiple of four within the population")
        descriptors, bank, source_record = breeding.load_warm(defaults.archive.resolve(), warm_count, args.seed)
        historical[:] = descriptors
        genomes.clear()
        genomes.update(bank)
        result = existing_protocol(args)
        result.update(
            nodes=NODES, edge_pairs=ARCS, edge_unit="directed matrix entries in both groups",
            birth_edges=16 * NODES, experiment_kind=runner.KIND,
            precision="FP16 state and edges, FP32 matrix accumulation",
            timing={**TIMING, "visit_ticks": block_ticks(TIMING) * 4,
                    "ticks_per_request": block_ticks(TIMING), "life_ticks": block_ticks(TIMING) * 4 * args.visits},
            warm_start={"slots": warm_count, "archive": str(defaults.archive.resolve()), "source": source_record},
            island_pairs=[[0, 1], [2, 3]], island_topologies=["directed", "reciprocal", "directed", "reciprocal"],
            training_worlds={"blocks": WORLD_BLOCKS, "actors_per_block": args.actors // WORLD_BLOCKS,
                             "seed_rule": "4 * generation development_seed + block index"},
            reward_gate="ACK +1 or REJ -1 on first legal primary FINISH only, write at next tick",
            selection_score=SCORING, ordering=["selection_score", "first_return_worst_context", "first_return_mean",
                                              "strict", "sparsity tie-break if both contexts exceed .25"],
            audit_modes=["full", "no_reward_write", "frozen", "fixed_guess"],
        )
        source = Path(__file__).resolve().parent
        result["sources"].update({name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in SOURCE_FILES})
        return result

    def audit_generation(config, out, generation, checkpoint, publish):
        train.clear_cache()
        return audit(config, out, generation, checkpoint, publish, evaluate=regular, delayed_evaluate=delayed,
                     matched_regular_evaluate=regular_padded, rank=rank)

    def evaluate_training(offspring, *, device, **settings):
        return train(offspring, device=device, capture=torch.device(device).type == "cuda", **settings)

    existing_protocol = runner.protocol
    overrides = {
        "OUT": OUT, "protocol": protocol,
        "plan": partial(breeding.plan, warm_founders=historical),
        "candidate": partial(breeding.candidate, warm_genomes=genomes),
        "evaluate_continuous": evaluate_training, "score_fields": score_fields, "rank": rank,
        "snapshot": snapshot, "audit": audit_generation,
    }
    previous = {key: getattr(runner, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(runner, key, value)
        runner.main(["--population", "4096", "--generations", "512", "--islands", "4", "--visits", "4",
                     "--actors", "16", "--audit-every", "64", *rest])
    finally:
        for key, value in previous.items():
            setattr(runner, key, value)


if __name__ == "__main__":
    main()
