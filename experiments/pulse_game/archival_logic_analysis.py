"""Summarize paired physical outcomes for the archived directed support."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np

from . import archival_logic_model as model, archival_logic_study as study
from . import ring_logic_study as shared, ring_logic_world as env


PROFILES = ("standard", "long")


def _bootstrap_weights(rng, sample):
    count = len(sample)
    choices = rng.multinomial(count, np.full(count, 1 / count), size=study.BOOTSTRAP_DRAWS)
    return choices.astype(np.float32) / count


def _metrics(reward, rule):
    correct = reward > 0
    by_start = correct.mean(axis=(-2, -1))
    cells = correct.mean(axis=(0, 1))
    swap = correct & correct[..., env.SWAP]
    changed = env.truth()[rule] != env.truth()[rule][env.SWAP]
    return {"mean_accuracy": float(correct.mean()), "timeout_rate": float((reward == 0).mean()),
            "full_20_pair_world_fraction": float(correct.all(axis=-1).mean()),
            "starts_at_least_95pct": int((by_start >= .95).sum()),
            "median_worst_pair_accuracy": float(np.median(correct.mean(axis=1).min(axis=-1))),
            "accuracy_by_start": by_start.tolist(), "accuracy_by_pair": cells.tolist(),
            "swapped_pair_joint_accuracy_on_changed_label": float(swap[..., changed].mean()) if changed.any() else None,
            "swapped_pair_joint_accuracy_on_same_label": float(swap[..., ~changed].mean()) if (~changed).any() else None}


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(out):
    registration, verification = study.frozen(out)
    scored = shared.read(out / "scored.json")
    if (shared.sha(out / "scores.npz") != scored["scores_sha256"] or
            shared.sha(out / "frozen.json") != scored["frozen_sha256"]):
        raise RuntimeError("Frozen or scored artifact changed")
    rng = np.random.default_rng(76324429)
    boot_starts = _bootstrap_weights(rng, registration["births"])
    boot_roots = {profile: _bootstrap_weights(rng, registration["world_roots"][profile])
                  for profile in PROFILES}
    family_size = len(PROFILES) * len(study.STEPS) * len(env.RULES)
    tail = .05 / (2 * family_size)
    complete = []
    per_start = []
    with np.load(out / "scores.npz", allow_pickle=False) as data:
        oracle = {profile: dict(zip(env.RULES, (data[f"oracle-{profile}"][..., 2] > 0).mean(axis=(1, 2)).tolist(),
                                    strict=True)) for profile in PROFILES}
        for steps in study.STEPS:
            for profile in PROFILES:
                table = data[f"{steps}-{profile}"]
                rewards = table[..., 2]
                expected = (study.STARTS, len(env.RULES), len(model.ARMS), len(study.ROOTS[profile]), len(env.PAIRS))
                if rewards.shape != expected or not np.isin(rewards, (-1, 0, 1)).all():
                    raise RuntimeError("Physical outcomes differ from the registered grid")
                for rule, name in enumerate(env.RULES):
                    archive, control = rewards[:, rule].transpose(1, 0, 2, 3)
                    paired = (archive > 0).mean(axis=-1) - (control > 0).mean(axis=-1)
                    draws = ((boot_starts @ paired.astype(np.float32)) * boot_roots[profile]).sum(axis=1)
                    archive_metrics, control_metrics = (_metrics(v, rule) for v in (archive, control))
                    label = {"steps": steps, "profile": profile, "rule": name}
                    complete.append({**label, "archive": archive_metrics, "degree_switch": control_metrics,
                                     "archive_minus_switch": float(paired.mean()),
                                     "simultaneous_95pct_interval": np.quantile(draws, (tail, 1 - tail)).tolist()})
                    for start, birth in enumerate(registration["births"]):
                        per_start.append({**label, "start": start, "seed": birth,
                                          "switch_seed": registration["control_seeds"][start],
                                          "archive_accuracy": archive_metrics["accuracy_by_start"][start],
                                          "switch_accuracy": control_metrics["accuracy_by_start"][start],
                                          "archive_minus_switch": float(paired[start].mean())})
    summary = {"design": f"Paired crossed bootstrap over starts and episode roots; all {family_size} family contrasts use Bonferroni simultaneous 95% percentile intervals. All 20 cells are an exhaustive within-root block.",
               "notes": "One fixed champion. The start bootstrap samples paired optimization seeds and their independently switched supports together. Intervals do not generalize across champions.",
               "public_oracle_accuracy": oracle, "training": verification["checkpoints"], "results": complete}
    shared.write(out / "summary.json", summary)
    _write_csv(out / "per-start.csv", per_start)
    entries = []
    for row in complete:
        lower, upper = row["simultaneous_95pct_interval"]
        entries.append({"steps": row["steps"], "profile": row["profile"], "rule": row["rule"],
                        "archive_accuracy": row["archive"]["mean_accuracy"],
                        "switch_accuracy": row["degree_switch"]["mean_accuracy"],
                        "archive_minus_switch": row["archive_minus_switch"],
                        "simultaneous_lower": lower, "simultaneous_upper": upper})
    _write_csv(out / "full-table.csv", entries)
    return entries


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.DEFAULT)
    args = parser.parse_args()
    for row in summarize(args.out):
        print(row)


if __name__ == "__main__":
    main()
