"""Plot paired behavior changes for the learned rule and the separate rehearsal reference."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVIDENCE=Path(__file__).resolve().parents[3]/"docs/evidence/edge-site-20261120"


def main():
    learned=json.loads((EVIDENCE/"evaluation-heldout-1024.json").read_text(encoding="utf8"))["primary"]
    replay=json.loads((EVIDENCE/"matched-b-results.json").read_text(encoding="utf8"))["cohorts"]
    colors=("#1b7169","#ba623d","#3f587e")
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,
                         "font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.55),sharex=True,constrained_layout=True)
    panels=[
        ("学得的局部选边规则",["相对原生全写","相对同剂量随机选址"],
         [learned["contrasts"][k] for k in ("full_minus_native","full_minus_random")],["a","b","both"]),
        ("显式回放旧回执后选择地址",["原六个体","独立六个体"],
         [replay[c]["scores"]["replay minus mixed"] for c in ("original","independent")],
         ["conflict/A","conflict/B","conflict/both"])]
    labels=("旧 A 正确","新 B 正确","两题全对")
    for ax,(title,rows,values,keys) in zip(axes,panels,strict=True):
        ax.axvline(0,color="#64748b",linewidth=1)
        for row,(name,data) in enumerate(zip(rows,values,strict=True)):
            for i,key in enumerate(keys):
                record=data[key]
                mean=100*record["mean"]
                low,high=(100*v for v in record["ci95"])
                y=1-row+(1-i)*.145
                ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt="o",color=colors[i],capsize=3,
                            label=labels[i] if row==0 else None)
        ax.set(yticks=[1,0],yticklabels=rows,ylim=(-.48,1.48),xlim=(-12,16))
        ax.set_title(title,fontsize=13,pad=12)
        ax.grid(axis="x",alpha=.13)
        ax.set_xlabel("实际正确率配对差（百分点）")
    axes[0].legend(loc="lower left",fontsize=9,frameon=False,ncol=3,handletextpad=.15,columnspacing=.5)
    axes[1].text(.5,.04,"对照在另一批世界按新 B 正确率配平",ha="center",va="center",
                 transform=axes[1].transAxes,fontsize=9,color="#475569")
    fig.savefig(EVIDENCE/"edge-site-outcomes.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
