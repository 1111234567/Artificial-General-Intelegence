"""Draw the actual three primary graphs and plot their confirmed physical accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..relation_study import ROOT


ARMS = ("self", "directed", "multiloop")
NAMES = ("自环", "有向单环", "无端点多环")
COLORS = ("#887341", "#196280", "#79559a")
RULES = ("orientation", "adjacency", "lookup")


def render(study, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch

    graphs = json.loads((study/"graphs.json").read_text(encoding="utf-8"))
    summary = json.loads((study/"summary.json").read_text(encoding="utf-8"))
    options = {item.name for item in font_manager.fontManager.ttflist}
    chinese = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if name in options),None)
    if chinese:
        plt.rcParams["font.family"] = chinese
    names = NAMES if chinese else ("Self connections", "Directed ring", "Closed multi-loop")
    rules = ("顺序方向", "对称邻接", "混合查表") if chinese else ("Orientation", "Adjacency", "Mixed table")
    fig = plt.figure(figsize=(9.0,5.5),layout="constrained")
    grid = fig.add_gridspec(2,3,height_ratios=(2.1,1.25))
    for col,arm in enumerate(ARMS):
        row, = [r for r in graphs if r["split"]=="confirmation" and r["group"]==r["restart"]==0 and r["arm"]==arm]
        n=len(row["sources_per_target"])
        t=np.arange(n)*2*np.pi/n + np.pi/2
        xy=np.stack([np.cos(t),np.sin(t)],axis=1)
        ax=fig.add_subplot(grid[0,col],aspect="equal")
        for dest,senders in enumerate(row["sources_per_target"]):
            for source in senders:
                if source==dest:
                    u=xy[dest]
                    v=np.array([-u[1],u[0]])
                    theta=np.linspace(-.75*np.pi,.75*np.pi,30)
                    loop=1.10*u + .083*np.cos(theta)[:,None]*u + .052*np.sin(theta)[:,None]*v
                    ax.plot(loop[:,0],loop[:,1],color=COLORS[col],linewidth=.9)
                    ax.add_patch(FancyArrowPatch(loop[-4],loop[-1],arrowstyle="-|>",mutation_scale=5,
                                                 color=COLORS[col],shrinkA=0,shrinkB=0,linewidth=.6))
                else:
                    backbone=(dest-source)%n==1
                    ax.add_patch(FancyArrowPatch(xy[source],xy[dest],arrowstyle="-|>",
                                                 connectionstyle="arc3,rad=0.08" if backbone else "arc3,rad=0.025",
                                                 shrinkA=1.9,shrinkB=2.6,mutation_scale=6,
                                                 linewidth=.95 if backbone else .66,
                                                 color=COLORS[col] if backbone else "#a995bb",
                                                 alpha=1 if backbone else .79))
        ax.scatter(*xy.T,c=COLORS[col],s=9,zorder=4)
        ax.set(xlim=(-1.21,1.21),ylim=(-1.21,1.21),title=names[col])
        ax.set_axis_off()
    for col,rule in enumerate(RULES):
        ax=fig.add_subplot(grid[1,col])
        for i,arm in enumerate(ARMS):
            ys=[summary["steps"][step]["standard"][rule]["arms"][arm]["accuracy"]*100 for step in ("240","1200")]
            ax.plot([0,1],ys,color=COLORS[i],marker=("o","s","^")[i],ms=5,linewidth=1.8)
        ax.axhline(50,color="#aaaaaa",linewidth=.7)
        ax.set(xlim=(-.17,1.17),ylim=(42,102),xticks=(0,1),xticklabels=("240","1,200"),yticks=(50,75,100),
               xlabel="训练更新次数" if chinese else "Training updates",title=rules[col])
        if col==0:
            ax.set_ylabel("实际选对率（%）" if chinese else "Correct contact (%)")
        ax.spines[["top","right"]].set_visible(False)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT/"data/pulse-game/ring-logic-20261119")
    parser.add_argument("--out",type=Path,default=ROOT/"docs/evidence/ring-logic-2026-11-19/closed-topologies.png")
    args=parser.parse_args()
    render(args.study,args.out)
