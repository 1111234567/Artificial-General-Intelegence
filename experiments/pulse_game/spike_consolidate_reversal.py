"""Stress-test the learned activity filter when a familiar rule changes."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_consolidate import Consolidation
from .spike_consolidate_run import ACQUIRE, CONFIRM_CODES, STOP, TrialSchedule, behavior, curriculum, row_scores
from .spike_policy import SpikeConfig
from .world import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    candidate = Consolidation(**frozen["mechanism"])
    cfg = SpikeConfig(**frozen["config"])
    tape = curriculum(list(range(100301, 100325)), codes=CONFIRM_CODES, phases=3)
    # From the third block onwards, old-cue preferences invert once. Nothing
    # in the visible interface announces that former answers have changed.
    good = tape.good_id.copy()
    good[:, STOP:] ^= (tape.cue[:, STOP:] < 2).astype(good.dtype)
    tape = replace(tape, good_id=good)
    cutoff = STOP + ACQUIRE
    arms = (
        ("ordinary", Consolidation(), cutoff),
        ("candidate", candidate, cutoff),
        ("candidate-refresh", replace(candidate, refresh=.006), cutoff),
        ("candidate-no-reversal-receipts", candidate, STOP),
    )
    rows = []
    for seed in frozen["birth_seeds"]:
        for name, rule, quiet in arms:
            policy = TrialSchedule(behavior(tape, rule, cfg, seed), quiet_from=quiet)
            result = rollout(policy, tape)
            scores = row_scores(tape, result, stop=cutoff)
            scores["first_reversal16"] = result.correct[:, STOP:STOP + 16].mean(1).tolist()
            scores["last_reversal16"] = result.correct[:, cutoff - 16:cutoff].mean(1).tolist()
            rows.append({"name": name, "birth_seed": seed, "scores": scores, "correct": result.correct.tolist()})
            print(f"{seed:2d} {name:28s} reversal {np.mean(scores['first_reversal16']):.3f} → "
                  f"{np.mean(scores['last_reversal16']):.3f}  old/new quiet "
                  f"{np.mean(scores['old_revisit']):.3f} / {np.mean(scores['new_revisit']):.3f}", flush=True)
    metrics = ("first_reversal16", "last_reversal16", "old_revisit", "new_revisit")
    means = {name: {score: np.mean([row["scores"][score] for row in rows if row["name"] == name], axis=0)
                    for score in metrics} for name, _, _ in arms}
    summary = {name: {metric: {"value": estimates(v), "minus_ordinary": estimates(v - means["ordinary"][metric])}
                      for metric, v in values.items()}
               for name, values in means.items()}
    out = {"study": "follow-up reversal", "world_seeds": tape.seeds, "map_codes_at_birth": tape.map_codes.tolist(),
           "reversal_trial_0_based": STOP, "quiet_mixed_test_start_0_based": cutoff, "summary": summary, "results": rows}
    args.artifact.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
