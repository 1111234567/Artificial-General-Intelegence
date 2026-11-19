"""Evaluate consolidation over all old/new partitions of the same cue set."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_consolidate import Consolidation
from .spike_consolidate_run import CONFIRM_CODES, STOP, TrialSchedule, behavior, curriculum, row_scores
from .spike_policy import SpikeConfig
from .world import rollout


PARTITIONS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))


def partition(tape, order):
    cue = np.asarray(order, dtype=tape.cue.dtype)[tape.cue]
    good = tape.good_map[np.arange(tape.n)[:, None], tape.ctx, cue]
    return replace(tape, cue=cue, good_id=good)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    candidate = Consolidation(**frozen["mechanism"])
    cfg = SpikeConfig(**frozen["config"])
    tape = curriculum(list(range(100201, 100221)), codes=CONFIRM_CODES)
    seeds = (31, 37, 41)
    result_rows = []
    for order in PARTITIONS:
        for reversal in (False, True):
            perm = order[2:] + order[:2] if reversal else order
            world_tape = partition(tape, perm)
            for seed in seeds:
                for name, mechanism in (("candidate", candidate), ("ordinary", Consolidation())):
                    policy = TrialSchedule(behavior(world_tape, mechanism, cfg, seed))
                    result = rollout(policy, world_tape)
                    scores = row_scores(tape, result, stop=STOP)
                    result_rows.append({"name": name, "partition": order, "reversal": reversal, "birth_seed": seed,
                                        "cue_order": perm, "scores": scores, "correct": result.correct.tolist()})
                    print(f"{seed:2d} {perm} {name:9s} old/new quiet {np.mean(scores['old_revisit']):.3f} "
                          f"/ {np.mean(scores['new_revisit']):.3f}", flush=True)
    comparison = {}
    for group in PARTITIONS:
        comparison[str(group)] = {}
        for key in ("old_learning", "new_learning", "old_revisit", "new_revisit"):
            arms = {name: np.mean([row["scores"][key] for row in result_rows
                                   if row["name"] == name and row["partition"] == group], axis=0)
                    for name in ("candidate", "ordinary")}
            comparison[str(group)][key] = {name: estimates(values) for name, values in arms.items()}
            comparison[str(group)][key]["difference"] = estimates(arms["candidate"] - arms["ordinary"])
    comparison["all"] = {}
    for metric in ("old_revisit", "new_revisit"):
        arms = {name: np.mean([row["scores"][metric] for row in result_rows if row["name"] == name], axis=0)
                for name in ("candidate", "ordinary")}
        comparison["all"][metric] = {name: estimates(value) for name, value in arms.items()}
        comparison["all"][metric]["difference"] = estimates(arms["candidate"] - arms["ordinary"])
    report = {"life_seeds": tape.seeds, "map_codes": tape.map_codes.tolist(), "birth_seeds": seeds,
              "summary": comparison, "results": result_rows}
    args.artifact.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
