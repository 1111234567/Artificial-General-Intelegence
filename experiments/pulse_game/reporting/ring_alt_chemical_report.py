"""Summarize chemical representation and retention with world-level pairings."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..spike_analysis import estimates


def load(root, name):
    return json.loads((root / f"ring-alt-chemical-{name}.json").read_text(encoding="utf-8"))


def at(result, name, *, phase=1, key="old"):
    return np.mean([row["phases"][phase][key] for row in result["rows"] if row["name"] == name], axis=0)


def summary(noisy, novel, shift, both, replay):
    static = {}
    for style, data in (("noisy", noisy), ("novel", novel)):
        groups = {}
        for name in dict.fromkeys(row["name"] for row in data["rows"]):
            block = [row for row in data["rows"] if row["name"] == name]
            group = dict(data["summary"]["arms"][name])
            for sense in ("fruit", "cue"):
                field = "organizing_" + sense
                if field in block[0]:
                    group[field] = {
                        key: float(np.mean([r[field][key] for r in block])) for key in block[0][field]
                    }
                    group[sense + "_active"] = float(np.mean([r["network"][sense]["active"] for r in block]))
                    group[sense + "_retirements"] = float(np.mean([r["network"][sense]["retirements"] for r in block]))
            groups[name] = group
        static[style] = groups
    a = noisy["summary"]["candidate_minus_comparator"]["negative-yoked"]["late_first"]
    capacity = {}
    for setting, dataset in (("fruit_only", shift), ("both", both)):
        ability = {}
        for phase, prefix in ((0, "early"), (1, "late")):
            for key in ("old", "new"):
                before = at(dataset, "growing-4", phase=phase, key=key)
                after = at(dataset, "growing-8", phase=phase, key=key)
                ability[prefix + "_" + key + "_8_minus_4"] = estimates(after - before)
        ability["retention_DID_8_minus_4"] = estimates(
            at(dataset, "growing-8") - at(dataset, "growing-4") - at(dataset, "growing-8", phase=0) + at(dataset, "growing-4", phase=0))
        ability["phase_1_4_8_12"] = {
            c: {key: dataset["summary"][f"growing-{c}"][f"phase_1_{key}"]
                for key in ("old", "new", "active_fruit", "retirements")} for c in (4, 8, 12)
        }
        capacity[setting] = ability
    budget = [row["ratio_absolute_mismatch"] for row in replay["rows"] if row["name"] == "competitive-matched"]
    old, new = (capacity["fruit_only"][f"late_{key}_8_minus_4"] for key in ("old", "new"))
    checks = {"late_first_gt_70_noisy": static["noisy"]["negative-live"]["late_first"]["mean"] > .70,
              "noisy_live_vs_bite_aligned_yoke": a["mean"] >= .25 and a["ci95"][0] > 0,
              "old_quiet_8_vs_4_gt_10pp": old["mean"] >= .10 and old["ci95"][0] > 0,
              "new_quiet_8_vs_4_noninferior": new["ci95"][0] > -.07,
              "both_budget_errors_below_one_percent": float(np.max(budget)) < .01}
    return {"checks": checks, "stationary": static, "capacity": capacity,
            "replay_summary": replay["summary"], "budget_match_max_relative_error": np.max(budget, axis=0).tolist()}


def chart(widths, noisy, shift, replay, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"grow": "#166861", "control": "#a86241", "overlap": "#7b58aa", "plain": "#66717f"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout="constrained")
    ax = axes[0, 0]
    curves = (("negative-live", "Grown, overlapping", "grow", "-"),
              ("winner-take-all", "Grown, one winner", "overlap", "--"),
              ("competition", "Fixed capacity competition", "control", "-."),
              ("raw-continuous", "Raw receptor mixtures", "plain", "-"),
              ("negative-yoked", "Bite-aligned yoke", "plain", ":"))
    for name, text, role, style in curves:
        a = np.array([row["first_correct"] for row in noisy["rows"] if row["name"] == name], dtype=float)
        x = a.reshape(a.shape[0], a.shape[1], -1, 32).mean(axis=(0, 1, 3))
        ax.plot(16.5 + np.arange(len(x)) * 32, x, color=colors[role], lw=1.5, ls=style, marker="o", ms=3, label=text)
    ax.set_xlabel("Trial (32-trial bins)")
    ax.set_ylabel("Correct first bite")
    ax.set_title("A   Independent noisy odor mixtures", loc="left", fontweight="bold")
    ax.set_ylim(.17, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="center right", bbox_to_anchor=(1., .31), frameon=True, edgecolor="none", fontsize=8)

    ax = axes[0, 1]
    for name, label, role in (("grow-winner", "One winner", "plain"), ("wide", "Wide, ordinary credit", "control"),
                              ("wide-orthogonal-strong", "Wide, decorrelated credit", "grow"),
                              ("fixed-competition-medium", "Fixed competition", "overlap"),
                              ("raw-with-orthogonal", "Raw channels", "plain"), ("narrow", "Narrow", "plain")):
        rows = [r for r in widths["rows"] if r["name"] == name]
        x = np.mean([r["organizing_fruit"]["mean_active_support"] for r in rows])
        y = np.mean([r["scores"]["late_first"] for r in rows])
        ax.scatter(x, y, c=colors[role], label=label, s=40)
    ax.set_xlabel("Effective participating odor branches")
    ax.set_ylabel("Late first-bite accuracy")
    ax.set_ylim(.67, .96)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("B   Odor overlap and local learning", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, edgecolor="none", fontsize=8)

    def groups(ax, table, labels, title, pair_names, keys, *, legend="lower right"):
        for i, (label, name) in enumerate(zip(labels, pair_names, strict=True)):
            for j, (key, text, role, marker, offset) in enumerate(zip(keys, ("Old odors", "New odors"),
                                                                    ("grow", "overlap"), ("o", "s"), (.12, -.12))):
                d = table[name][key]
                mid, lo, hi = d["mean"], *d["ci95"]
                ax.errorbar(mid, i + offset, xerr=[[mid - lo], [hi - mid]], color=colors[role], ms=4,
                            marker=marker, capsize=2, label=text if not i else None)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels) - .5, -.5)
        ax.set_xlim(.12, 1.)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Correct first bite in quiet probe")
        ax.legend(loc=legend, frameon=True, edgecolor="none", fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold")

    ax = axes[1, 0]
    groups(ax, shift["summary"], ("2 branches", "4 branches", "8 branches", "12 branches", "Plasticity frozen"),
           "C   Retention and relearning after odor change",
           ("growing-2", "growing-4", "growing-8", "growing-12", "freeze-all-8"),
           ("phase_1_old", "phase_1_new"), legend="upper right")
    ax = axes[1, 1]
    groups(ax, replay["summary"], ("Growth", "Ordinary competition", "Competition, same write norms", "Reassign fixed bank"),
           "D   Same bites and matched write norms",
           ("growing", "competitive", "competitive-matched", "fixed-bank-reassignment"), ("old", "new"))
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax in axes[0] else "x", color="#dfdfdf", lw=.6)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def forward_evidence(native, paired):
    members = ["ordinary", "age", "sham", "winner"]
    raw = {name: {key: np.mean([row[key] for row in native["rows"] if row["name"] == name], axis=0)
                  for key in ("old_before", "old_after", "new_after")} for name in members}
    comparison = {
        control: {key: estimates(raw["age"][key] - raw[control][key]) for key in raw["age"]}
        for control in ("ordinary", "sham", "winner")
    }
    front = comparison["ordinary"]
    shuffle = comparison["sham"]["old_after"]
    match = paired["summary"]["matched_minus_ordinary"]
    residual = np.max([row["relative_prototype_and_association_errors"] for row in paired["rows"]
                       if row["name"] == "age-matched"], axis=0)
    checks = {"native_old_gain_12pp": front["old_after"]["mean"] >= .12 and front["old_after"]["ci95"][0] > 0,
              "native_new_noninferior": front["new_after"]["ci95"][0] > -.05,
              "native_old_gain_against_shuffled_ages": shuffle["mean"] >= .05 and shuffle["ci95"][0] > 0,
              "matched_old_positive_new_noninferior": match["old"]["ci95"][0] > 0 and match["new"]["ci95"][0] > -.05,
              "equal_write_amplitude": float(residual.max()) < .01}
    return {"checks": checks, "native": native["summary"], "matched": paired["summary"],
            "paired_native_age_minus": comparison, "max_prototype_and_association_budget_error": residual.tolist(),
            "mean_active_fruit_support": {name: float(np.mean([row["network"]["fruit"]["participation"]
                                                                for row in native["rows"] if row["name"] == name]))
                                          for name in members}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    p = args.data
    noisy, novel, shift, both, replay = (load(p, name) for name in (
        "confirm-noisy", "confirm-novel", "shift-fruit", "shift-both", "match-confirm"))
    report = summary(noisy, novel, shift, both, replay)
    if all((p / f"ring-alt-chemical-{name}.json").exists() for name in ("forward-confirm", "forward-match")):
        report["forward_age_competition"] = forward_evidence(load(p, "forward-confirm"), load(p, "forward-match"))
    (p / "ring-alt-chemical-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        chart(load(p, "width-discovery"), noisy, shift, replay, p / "ring-alt-chemical-results.png")


if __name__ == "__main__":
    main()
