"""Export the completed experiment's metrics and a standalone results figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/persistent")
    parser.add_argument("--output", default="docs/evidence/persistent-2026-10-26")
    args = parser.parse_args()
    source, output = ROOT / args.data, ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    experiments = {}
    expected = {"main-v1":18, "read-budget-v1":3, "retention-v1":6,
                "heldout-main-v1":18, "heldout-budget-v1":3, "heldout-retention-v1":6,
                "adapt-v1":9}
    for folder, count in expected.items():
        path = source / folder
        runs = [read(p) for p in sorted(path.glob("*-seed*.json"))
                if not p.name.startswith("initial-")]
        if len(runs) != count:
            raise ValueError(f"{folder}: expected {count} complete runs, got {len(runs)}")
        experiments[folder] = dict(manifest=read(path / "manifest.json"),
                                   summary=read(path / "summary.json"), runs=runs)
    artifact = dict(description="Small synthetic mechanism tests, not a general agent benchmark",
                    statistics="mean and sample standard deviation across 3 model seeds; 2048 heldout queries per seed per case",
                    experiments=experiments)
    (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    core = experiments["heldout-main-v1"]["summary"]
    budget = experiments["heldout-budget-v1"]["summary"]
    retention = experiments["heldout-retention-v1"]["summary"]
    adapt = experiments["adapt-v1"]["summary"]["summary"]
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    names = ["two_facts", "two_gap16", "two_gap64", "two_gap256", "two_gap1024", "two_gap4096"]
    x = [0, 16, 64, 256, 1024, 4096]
    for mode, label, color in [("gru", "GRU state", "#b94942"), ("single", "Addressable records", "#197f75")]:
        means = [100*retention[f"{mode}/{name}"]["mean"] for name in names]
        stds = [100*retention[f"{mode}/{name}"]["std"] for name in names]
        axes[0].errorbar(x, means, yerr=stds, label=label, color=color, marker="o", capsize=3)
    axes[0].set_xscale("symlog", linthresh=16, base=2)
    axes[0].set_xticks(x, labels=[str(v) for v in x], rotation=35)
    axes[0].axhline(50, color="gray", linestyle=":", label="Identity baseline (expected)")
    axes[0].set(title="A. Retention of 2 learned facts", xlabel="NOOP delay (training: 0, 4, 16)", ylabel="Exact answer (%)", ylim=(0,108))
    axes[0].legend(loc="lower left", fontsize=8)
    options = [(core, "single", "1-step\n1k upd"), (budget,"single","1-step\n4k upd"),
               (core,"detach","Loop\ncut grad"), (core,"joint","Loop\nfull grad"),
               (core,"judge_aux","Loop\njudge loss"), (core,"judge","Loop\nfeedback")]
    for i, (dataset, mode, label) in enumerate(options):
        value = dataset[f"{mode}/capacity64"]
        axes[1].bar(i, value["mean"]*100, yerr=value["std"]*100, capsize=3,
                    color="#7894ad" if i==0 else "#197f75")
    axes[1].set_xticks(range(len(options)), [item[2] for item in options], fontsize=8)
    axes[1].set(title="B. 64 records, 4 inference steps", ylabel="Exact answer (%)", ylim=(0,108))
    for i, arm in enumerate(["frozen", "reward", "critic"]):
        value = adapt[f"{arm}/new"]
        axes[2].bar(i, value["mean"]*100, yerr=value["std"]*100, capsize=4,
                    color=["#7894ad", "#197f75", "#527d55"][i])
    axes[2].set_xticks(range(3), ["Frozen", "Real reward", "Real reward\n+ shared critic"])
    axes[2].set(title="C. Adaptation with binary reward", ylabel="Heldout success (%)", ylim=(0,108))
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Persistent memory and integrated loops: controls change the conclusion\n"
                 "3 seeds; 2048 fresh queries per seed and condition; mean +/- SD", fontsize=12)
    fig.savefig(output / "results.png", dpi=180)
    fig.savefig(output / "results.svg")
    plt.close(fig)
    lines = ["| 模式 | 16 条事实 / 4 跳 | 64 条事实 / 4 跳 | 16 条事实 / 8 跳 |", "|---|---:|---:|---:|"]
    for dataset, mode, label in options:
        cells = [f'{100*dataset[f"{mode}/{case}"]["mean"]:.2f} ± {100*dataset[f"{mode}/{case}"]["std"]:.2f}%'
                 for case in ["four_hop", "capacity64", "eight_hop"]]
        lines.append("| " + " | ".join([label.replace("\n", " ")] + cells) + " |")
    (output / "tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(str(output))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
