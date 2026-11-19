"""Scientific plots from the complete validated recurrent rewiring report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [("dense", "Dense", "#606b7e"), ("retire", "Retire only", "#c0812e"),
        ("learned", "Directed growth", "#167f9b"), ("random", "Random growth", "#a64c72")]


def draw(report, output):
    if not report["complete"] or report["audit"]["status"] != "passed":
        raise ValueError("plots require a complete validated report")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, dose in enumerate(("low", "high")):
        for arm, label, color in ARMS:
            identifier = arm if arm == "dense" else f"{arm}_{dose}"
            rows = [r for r in report["capacity_trajectory"] if r["condition_id"] == identifier]
            seeds = [sorted((r for r in rows if r["seed"] == seed), key=lambda r:r["stage_ticks"]) for seed in range(3)]
            x = np.array([r["stage_ticks"] for r in seeds[0]])/1024
            for row, metric in enumerate(("active", "initial_survivors")):
                values = np.array([[r[metric] for r in values] for values in seeds])
                mean, sd = values.mean(0), values.std(0, ddof=1)
                axes[row,col].plot(x, mean, color=color, label=label, lw=1.6)
                axes[row,col].fill_between(x, mean-sd, mean+sd, color=color, alpha=.13)
        axes[0,col].set_title(f"{dose.capitalize()} shrink: " + ("0.00005" if dose=="low" else "0.00020"))
        for ax in axes[:,col]:
            ax.axvline(64,color="#333333",ls="--",lw=.8)
            ax.set_ylim(bottom=0, top=16900)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.16)
            ax.set_xticks([0,32,64,96,128])
        axes[1,col].set_xlabel("Continuous adaptation (1,024 ticks); return to A at 64")
    axes[0,0].set_ylabel("Active recurrent connections")
    axes[1,0].set_ylabel("Original edges never retired")
    axes[0,0].legend(frameon=False,fontsize=9,ncols=2)
    fig.suptitle("Connection birth and gradual retirement in one continuing state",fontsize=13)
    fig.text(.5,.01,"Lines: mean across three trained seeds; shaded regions: sample SD. Births and deaths can revisit the same connection.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96))
    for ext in ("png","svg"): fig.savefig(output/f"connection-history.{ext}",dpi=170)
    plt.close(fig)
    index={(r["condition_id"],r["learning_phase"],r["condition"],r["metric"]):r for r in report["main_summary"]}
    fig, axes = plt.subplots(2,3,figsize=(12,7),sharey=True)
    for row,dose in enumerate(("low","high")):
        for col,(condition,title) in enumerate((("iid","Old A conditions"),("long_sparse","Long and sparse B conditions"),("longer","Unseen wait 8-16"))):
            ax=axes[row,col]
            for k,(arm,label,color) in enumerate(ARMS):
                identifier=arm if arm=="dense" else f"{arm}_{dose}"
                vals=[index[(identifier,p,condition,"strict_completion")] for p in ("B","A2")]
                xs=np.array([0,1])+(k-1.5)*.19
                ax.bar(xs,[100*r["mean"] for r in vals],width=.18,color=color,label=label,alpha=.85)
                ax.errorbar(xs,[100*r["mean"] for r in vals],yerr=[100*r["sample_sd"] for r in vals],fmt="none",color="#222222",capsize=2,lw=1)
                for x,r in zip(xs,vals):
                    ax.scatter([x-.035,x,x+.035],[100*s["value"] for s in r["per_seed"]],s=9,color="#222222",zorder=4)
            ax.set_title(f"{dose} | {title}",fontsize=10)
            ax.set_xticks([0,1],["After B","After A return"])
            ax.set_ylim(0,100)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.16)
            ax.set_axisbelow(True)
            if col==0: ax.set_ylabel("Strict clean completion (%)")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncols=4,frameon=False,bbox_to_anchor=(.5,.955))
    fig.suptitle("Adaptation and retention under recurrent topology changes",fontsize=13,y=.995)
    fig.text(.5,.01,"Three trained seeds; dots show each seed; error bars show sample SD. Both predeclared shrink strengths are reported.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.90))
    for ext in ("png","svg"): fig.savefig(output/f"phase-comparison.{ext}",dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    draw(json.loads(args.report.read_text(encoding="utf-8")),args.output)
