"""Draw paired held-out outcomes from the finalized policy-data comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/action-return/refit-policy-v1/summary.json"
OUTPUT = ROOT / "docs/evidence/action-return-2026-11-03/paired-refit.png"


def main():
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert summary["audit"] == "passed"
    rates = {(r["condition"], r["arm"], r["seed"]): r["rates"] for r in summary["heldout"]}
    colors = ("#087d80", "#b35412", "#6654a5")
    metrics = (
        ("feasible_begin_pp", "有效启动（%）", 100),
        ("idle_actions_per1000", "空闲误动作（每千次机会）", 1000),
        ("ordinary_pp", "普通完成（%）", 10),
    )
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.1))
    fig.subplots_adjust(left=.13, right=.975, top=.82, bottom=.14, wspace=.40, hspace=.39)
    fig.text(.045, .953, "用自己的行动轨迹再训练，哪些行为变了？", fontsize=17, weight="bold")
    fig.text(.045, .894, "同一条线连接同一训练种子，两端用等量的监督数据。评估时模型自行行动。", color="#40505c")
    for i, (condition, name) in enumerate((("short", "短任务"), ("long_both", "长处理\n长空档"))):
        for j, (metric, title, limit) in enumerate(metrics):
            ax = axes[i, j]
            for seed, color in enumerate(colors):
                values = [rates[condition, arm, seed][metric] for arm in ("assisted_archive", "own_policy")]
                ax.plot((0, 1), values, "o-", color=color, alpha=.88, lw=1.9, markersize=5, label=f"种子 {seed}")
            ax.set(xlim=(-.16, 1.16), ylim=(-.035 * limit, 1.04 * limit), yticks=[0, limit / 2, limit])
            ax.set_xticks((0, 1), ("帮助轨迹", "自主轨迹"))
            ax.grid(axis="y", color="#d9dfe3", lw=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#9ca5aa")
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title(title, fontsize=12, pad=10)
            if j == 0:
                ax.set_ylabel(name, rotation=0, labelpad=28, va="center", weight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.035, .016), ncol=3, frameon=False)
    fig.text(.51, .052, "两种训练条件在此次对照中的严格完成数均为零。", color="#40505c")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
