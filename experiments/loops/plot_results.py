"""Standalone scientific figures from strictly validated loop reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_report(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not value.get("complete") or value.get("audit", {}).get("status") != "passed":
        raise ValueError("plots require a complete validated report")
    return value


def audit_figure(report, folder):
    index = {(row["mode"], row["condition"], row["intervention"]): row for row in report["tables"]}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True)
    conditions = [("iid", "Old regime: wait 1-4"), ("long_sparse", "New regime: wait 4-8, sparse"),
                  ("longer", "Unseen regime: wait 8-16")]
    for row_index, mode in enumerate(("bp64", "bp_replay")):
        for col, (condition, title) in enumerate(conditions):
            ax = axes[row_index, col]
            for label, color, names, offset in (
                ("Toward DAG", "#ac3e45", ("none", "dag_quarter", "dag_norm_matched"), -.018),
                ("Toward random cut", "#227e9e", ("none", "random_quarter", "random_norm_matched"), .018)):
                stats = [index[(mode, condition, name)]["metrics"]["strict_clean_completion"] for name in names]
                xs = np.array([0, .25, 1.])
                ax.plot(xs, [100*s["mean"] for s in stats], color=color, lw=1.6, label=label)
                ax.errorbar(xs+offset, [100*s["mean"] for s in stats],
                            yerr=[100*s["sample_sd"] for s in stats], color=color, fmt="o", ms=4, capsize=3)
                for x, stat in zip(xs, stats):
                    ax.scatter([x+offset]*3, [100*s["value"] for s in stat["per_seed"]],
                               color=color, alpha=.45, s=12, marker="x", zorder=4)
            ax.set_title(f"{mode} | {title}", fontsize=10)
            ax.set_xticks([0, .25, 1.], ["0%", "25%", "100%"])
            if row_index == 1:
                ax.set_xlabel("Interpolation toward cut endpoint")
            ax.set_xlim(-.09, 1.09)
            ax.set_ylim(bottom=-.3)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Strict clean completion (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Existing recurrent circuits are sensitive to connection interventions", fontsize=13, y=.99)
    fig.text(.5, .015, "Three trained seeds; bars = sample SD; masks averaged within seed. Endpoints match incoming counts and row norms.\n"
             "25% interpolation retains cycles and is not norm-matched. Large cuts change many properties; a floor cannot isolate loop causality.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"existing-loop-interventions.{suffix}", dpi=180)
    plt.close(fig)


def extension_figure(report, folder):
    index = {(row["arm"], row["condition"], row["metric"]): row for row in report["main_summary"]}
    arms = [("baseline", "Original field", "#6d778a"), ("source_square", "Source-square extension", "#d38b31"),
            ("interaction", "Recipient/source extension", "#267f9b")]
    conditions = ["iid", "long_wait", "longer", "sparse", "long_sparse"]
    labels = ["Old\nwait 1-4", "New, dense\nwait 4-8", "Unseen\nwait 8-16", "Old, sparse\nwait 1-4", "New, sparse\nwait 4-8"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.3), sharex=True)
    for ax, metric, scale, ylabel in zip(axes,
            ("strict_completion", "inactive_per_1000_idle"), (100, 1),
            ("Strict clean completion (%)", "Erroneous actions per 1,000\nidle opportunities")):
        for arm_index, (arm, label, color) in enumerate(arms):
            xs = np.arange(5) + (arm_index-1)*.23
            stats = [index[(arm, condition, metric)] for condition in conditions]
            ax.bar(xs, [scale*s["mean"] for s in stats], width=.22, color=color, alpha=.85, label=label)
            ax.errorbar(xs, [scale*s["mean"] for s in stats], yerr=[scale*s["sample_sd"] for s in stats],
                        fmt="none", color="#222222", capsize=3, lw=1)
            for x, stat in zip(xs, stats):
                ax.scatter([x-.045, x, x+.045], [scale*s["value"] for s in stat["per_seed"]],
                           color="#20252a", s=12, zorder=4)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, ncols=3, loc="upper center", bbox_to_anchor=(.5, 1.14))
    axes[1].set_xticks(np.arange(5), labels)
    fig.suptitle("Continuous-state learning with two matched nonlinear extensions", y=.99, fontsize=13)
    fig.text(.5, .015, "Three trained seeds; dots = individual seeds; bars = mean; error bars = sample SD.\n"
             "Same 128-state size and 65,536-tick learning budget. Extensions: 50,732 parameters; original: 34,348.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .96))
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"nonlinear-extension-comparison.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_figure(read_report(args.audit), args.output)
    if args.extension:
        extension_figure(read_report(args.extension), args.output)


if __name__ == "__main__":
    main()
