"""Portable seed-level probe evidence and figure, without refitting anything."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/timing-boundary/probe-v1"
OUT = ROOT / "docs/evidence/timing-boundary-2026-11-04"
ARMS = ("midpoint", "interior", "first", "random")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect():
    from experiments.timing_boundary.data import sample_path, facts_path
    pins, rows, native, behavior, coverage, time_errors = {}, [], [], [], [], []

    def pin(path):
        pins[Path(path).relative_to(ROOT).as_posix()] = sha(path)

    for seed in range(3):
        for arm in ARMS:
            histories = [("teacher", "validation"), ("teacher", "test")]
            if arm != "random":
                histories.append(("own", "test"))
            for mode, split in histories:
                fact_meta = facts_path(seed, split, mode=mode, arm=arm).with_suffix(".json")
                pin(fact_meta)
                facts = json.loads(fact_meta.read_text(encoding="utf8"))
                coverage.append(dict(seed=seed, arm=arm, mode=mode, split=split,
                                     counts=facts["counts"], exclusions=facts["exclusions"]))
                if mode == "own":
                    matured = [r for r in facts["requests"] if r["eligible"] and r["issued"] >= 512 and r["mature"]]
                    groups = {"all": matured}
                    for family in (0, 1):
                        groups[f"family_{'AB'[family]}_conflict"] = [r for r in matured if r["family"] == family and r["conflict"]]
                    for group, requests in groups.items():
                        n = len(requests)
                        behavior.append(dict(seed=seed, arm=arm, group=group, matured=n,
                            ordinary=sum(r["ordinary"] for r in requests), strict=sum(r["strict"] for r in requests),
                            ordinary_rate=sum(r["ordinary"] for r in requests)/n,
                            strict_rate=sum(r["strict"] for r in requests)/n,
                            idle_actions=facts["idle_actions"], idle_opportunities=facts["idle_opportunities"],
                            idle_per_1000=1000*facts["idle_actions"]/facts["idle_opportunities"]))
                for task in ("timing", "content", "remaining"):
                    path = SOURCE / f"seed_{seed:03d}/{arm}/predictions/{mode}__{split}__{task}.json"
                    value = json.loads(path.read_text(encoding="utf8"))
                    pin(path)
                    prediction_path = path.with_suffix(".npz")
                    pin(prediction_path)
                    assert sha(prediction_path) == value["predictions_sha256"]
                    for feature, groups in value["metrics"].items():
                        for group, metrics in groups.items():
                            rows.append(dict(seed=seed, arm=arm, mode=mode, split=split,
                                             task=task, feature=feature, group=group, **metrics))
                    if task == "remaining":
                        with np.load(prediction_path) as prediction:
                            for feature in ("h", "obs"):
                                for offset in (-8, -4, -2, -1, 0, 1, 2):
                                    mask = prediction["offset"] == offset
                                    error = np.abs(prediction[feature][mask] - prediction["remaining_ticks"][mask])
                                    time_errors.append(dict(seed=seed, arm=arm, mode=mode, split=split,
                                        feature=feature, offset=offset, rows=len(error),
                                        mae_ticks=float(error.mean()), median_abs_ticks=float(np.median(error)),
                                        p90_abs_ticks=float(np.quantile(error, .9))))
                path = sample_path(seed, split, arm, mode=mode)
                pin(path)
                with np.load(path) as z:
                    selected = np.flatnonzero(np.isin(z["offset"], (-1, 0)))
                    last = selected[z["offset"][selected] == -1]
                    first = selected[z["offset"][selected] == 0]
                    last = last[np.argsort(z["pair_id"][last])]
                    first = first[np.argsort(z["pair_id"][first])]
                    assert np.array_equal(z["pair_id"][last], z["pair_id"][first])
                    assert np.array_equal(z["tick"][last] + 1, z["tick"][first])
                    before = z["logits"][last, z["job"][last]].argmax(-1)
                    after = z["logits"][first, z["job"][first]].argmax(-1)
                    masks = {"all": np.ones(len(last), dtype=bool)}
                    for family in (0, 1):
                        masks[f"family_{'AB'[family]}"] = z["family"][last] == family
                        masks[f"family_{'AB'[family]}_conflict"] = (z["family"][last] == family) & z["conflict"][last]
                    identical = (z["obs"][last] == z["obs"][first]).all(-1)
                    for group, mask in masks.items():
                        n = int(mask.sum())
                        correct = (before == 0) & (after == z["target"][first])
                        both_time = (before == 0) & (after >= 3)
                        native.append(dict(seed=seed, arm=arm, mode=mode, split=split, group=group,
                            pairs=n, last_wait=int(((before == 0) & mask).sum()),
                            first_finish=int(((after >= 3) & mask).sum()),
                            both_time=int((both_time & mask).sum()),
                            both_time_rate=float(both_time[mask].mean()),
                            both_correct=int((correct & mask).sum()),
                            both_correct_rate=float(correct[mask].mean()),
                            first_correct=int(((after == z["target"][first]) & mask).sum()),
                            identical_current_observation=int((identical & mask).sum())))

    for path in (SOURCE / "protocol.json", SOURCE / "freeze.json",
                 ROOT / "data/timing-boundary/v1/protocol.json",
                 ROOT / "data/timing-boundary/v1/workers.json",
                 ROOT / "data/timing-boundary/audit-v1/protocol.json",
                 ROOT / "data/timing-boundary/audit-v1/audit.json", Path(__file__)):
        pin(path)
    audit = json.loads((ROOT / "data/timing-boundary/audit-v1/audit.json").read_text(encoding="utf8"))
    assert audit["passed"] and audit["complete"]
    aggregate = []
    keys = ("arm", "mode", "split", "task", "feature", "group")
    unique = sorted({tuple(r[k] for k in keys) for r in rows})
    rate_keys = ("row_accuracy", "both_accuracy", "first_accuracy", "rank_accuracy",
                 "rank_strict_accuracy", "false_early_rate", "missed_first_legal_rate",
                 "mae_ticks", "boundary_sign_accuracy", "boundary_both_accuracy", "boundary_rank_accuracy")
    count_keys = ("rows", "pairs", "correct_rows", "both_correct", "first_correct", "rank_wins",
                  "rank_ties", "false_early", "missed_first_legal", "boundary_rows", "boundary_pairs", "boundary_both_correct")
    for key in unique:
        subset = [r for r in rows if tuple(r[k] for k in keys) == key]
        assert sorted(r["seed"] for r in subset) == [0, 1, 2], (key, "missing seed")
        entry = dict(zip(keys, key), contributing_seeds=[0, 1, 2])
        for name in rate_keys:
            if name in subset[0]:
                values = [r[name] for r in subset]
                assert all(v is not None for v in values)
                entry[name] = dict(mean=float(np.mean(values)), seed_values=values,
                                   minimum=float(min(values)), maximum=float(max(values)))
        for name in count_keys:
            if name in subset[0]:
                entry[name] = sum(r[name] for r in subset)
        aggregate.append(entry)
    value = dict(description="Readout rates are equal-weight means over 3 model seeds. Counts are pooled, with per-seed rows retained. Probes never control actions.",
                 rows=rows, aggregate=aggregate, native_head=native, free_behavior=behavior,
                 data_coverage=coverage, time_errors_by_offset=time_errors,
                 audit={k: audit[k] for k in ("passed", "complete", "native_streams", "recurrent_model_streams",
                     "probe_coefficient_artifacts", "probe_prediction_artifacts", "full_tickwise_hidden_logits_exact",
                     "original_models_and_raw_unchanged")}, source_files=pins)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    return value


def plot(value):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"h": "#2463a5", "h_mlp64": "#c76029", "obs": "#8b929a", "original_conditional_argmax": "#619c78"}
    names = {"h": "Linear state readout", "h_mlp64": "64-unit nonlinear readout", "obs": "Current input only", "original_conditional_argmax": "Original FINISH ranking"}

    def points(arm, task, feature, metric, mode="teacher"):
        found = [r for r in value["rows"] if r["arm"] == arm and r["task"] == task and r["feature"] == feature
                 and r["mode"] == mode and r["split"] == "test" and r["group"] == "all"]
        assert len(found) == 3
        return [100 * r[metric] for r in sorted(found, key=lambda r: r["seed"])]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), layout="constrained")
    configs = ((axes[0, 0], "timing", ("h", "h_mlp64", "obs"), "both_accuracy", "Both boundary decisions correct", "teacher"),
               (axes[0, 1], "timing", ("h", "h_mlp64", "obs"), "rank_accuracy", "Score increases across the boundary", "teacher"),
               (axes[1, 0], "content", ("h", "original_conditional_argmax", "obs"), "first_accuracy", "Correct version + content at first legal tick", "teacher"),
               (axes[1, 1], "timing", ("h", "h_mlp64", "obs"), "both_accuracy", "Same teacher-fitted probes on own-action histories", "own"))
    for ax, task, features, metric, title, mode in configs:
        arms = ARMS if mode == "teacher" else ARMS[:3]
        for j, feature in enumerate(features):
            x = np.arange(len(arms)) + (j - 1) * .24
            vals = [points(arm, task, feature, metric, mode) for arm in arms]
            ax.bar(x, [np.mean(v) for v in vals], width=.22, color=colors[feature], label=names[feature], alpha=.85)
            for at, v in zip(x, vals):
                ax.scatter(at + np.array([-.035, 0, .035]), v, s=13, c="#24313a", zorder=3)
        ax.set_xticks(np.arange(len(arms)), [dict(midpoint="Original time", interior="Interior", first="First legal", random="Random GRU")[a] for a in arms])
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False)
    fig.suptitle("Timing boundary diagnosis with frozen recurrent models\nBars: equal seed means; dots: 3 model seeds. Readouts do not control actions.", fontsize=14)
    fig.savefig(OUT / "timing-boundary.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot(collect())
