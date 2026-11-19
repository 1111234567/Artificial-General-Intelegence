"""Show one preregistered birth from each of the three matched graph families."""
import argparse
from pathlib import Path

import numpy as np

from .. import free_graph_model as model, ring_logic_study as old


def draw(seed, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    chinese=next((font for font in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
                  if font in {f.name for f in font_manager.fontManager.ttflist}),None)
    if chinese: plt.rcParams["font.family"]=chinese
    labels=("固定两入两出","保留主环，额外边自由","全部连接自由") if chinese else (
        "Regular (2 in, 2 out)", "Ring with free chords", "Free connected graph")
    n=model.WIDTH
    theta=np.pi/2+2*np.pi*np.arange(n)/n
    xy=np.stack((np.cos(theta),np.sin(theta)),axis=-1)
    fig,axes=plt.subplots(2,3,figsize=(9.4,4.5),height_ratios=(2.6,1),layout="constrained")
    max_degree=max(np.bincount(model.graph_for(seed,a)[:,i],minlength=n).max() for a in model.ARMS for i in (0,1))
    for col,family in enumerate(model.ARMS):
        arcs=model.graph_for(seed,family)
        top,bottom=axes[:,col]
        for source,target in arcs:
            top.add_patch(FancyArrowPatch(xy[source],xy[target],arrowstyle="-|>",shrinkA=2.3,shrinkB=2.5,
                         alpha=.68,mutation_scale=6,lw=.8,color="#37688b",connectionstyle="arc3,rad=0.10"))
        top.scatter(*xy.T,s=11,c="#24445b",zorder=4)
        top.set(aspect="equal",xlim=(-1.12,1.12),ylim=(-1.12,1.12),title=labels[col]);top.set_axis_off()
        axis=np.arange(1,max_degree+1)
        for offset,which,color in ((-.15,1,"#226489"),(.15,0,"#b97526")):
            degrees=np.bincount(arcs[:,which],minlength=n)
            bottom.bar(axis+offset,np.bincount(degrees,minlength=max_degree+1)[1:],width=.28,color=color)
        bottom.set(ylim=(0,34),yticks=(0,16,32),xticks=axis,xlabel="连接条数" if chinese else "Node degree")
        if col==0:bottom.set_ylabel("节点数" if chinese else "Nodes")
        bottom.spines[["top","right"]].set_visible(False)
    handles=[Line2D([],[],marker="s",ls="",color=color) for color in ("#226489","#b97526")]
    fig.legend(handles,("入度","出度") if chinese else ("Incoming","Outgoing"),
               loc="outside lower center",ncol=2,frameon=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=155);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed",type=int)
    parser.add_argument("--output",type=Path,default=old.REPO/"docs/evidence/free-graph-2026-11-19/wiring.png")
    args=parser.parse_args()
    draw(args.seed,args.output)
