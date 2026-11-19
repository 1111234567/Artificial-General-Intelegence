"""Draw the same preregistered birth before and after adding inter-ring bridges."""
from pathlib import Path

import numpy as np

from .. import module_graph_study as study, module_graphs


def draw(seed,dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch
    font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","SimHei")
               if f in {v.name for v in font_manager.fontManager.ttflist}),None)
    if font:plt.rcParams["font.family"]=font
    dense=module_graphs.sample(seed)
    sparse=module_graphs.sample(seed,"minimum")
    rings=dense.rings;k=len(rings)
    angle=np.linspace(0,2*np.pi,k,endpoint=False)+np.pi/4
    centers=np.stack((np.cos(angle),np.sin(angle)),axis=1)
    xy=np.zeros((sum(map(len,rings)),2))
    scale=.84*np.sin(np.pi/k)
    for center,ring in zip(centers,rings):
        local=np.linspace(0,2*np.pi,len(ring),endpoint=False)+np.pi/2
        radius=scale*(.30+.70*np.sqrt(len(ring)/max(map(len,rings))))
        xy[list(ring)]=center+radius*np.stack((np.cos(local),np.sin(local)),axis=1)
    fig,axs=plt.subplots(1,3,figsize=(9.6,3.3),layout="constrained")
    titles=(("出生时的独立环",f"加入 {len(sparse.bridges)} 条跨环边",f"加入 {len(dense.bridges)} 条跨环边") if font else
            ("Disjoint cycles",f"{len(sparse.bridges)} bridges",f"{len(dense.bridges)} bridges"))
    for ax,bridges,title in zip(axs,((),sparse.bridges,dense.bridges),titles):
        for group,color,radius,size in ((bridges,"#b47e43",.075,.7),(dense.within,"#28677e",.20,1.1)):
            for s,t in group:
                ax.add_patch(FancyArrowPatch(xy[s],xy[t],arrowstyle="-|>",shrinkA=2,shrinkB=2,
                                             mutation_scale=6.5,connectionstyle=f"arc3,rad={radius}",
                                             alpha=.82,linewidth=size,color=color))
        ax.scatter(*xy.T,s=13,c="#254957",zorder=4)
        ax.set(aspect="equal",title=title,xlim=(xy[:,0].min()-.12,xy[:,0].max()+.12),
               ylim=(xy[:,1].min()-.12,xy[:,1].max()+.12));ax.set_axis_off()
    fig.savefig(dest,dpi=165);plt.close(fig)


if __name__=="__main__":
    destination=study.EVIDENCE/"modules.png"
    destination.parent.mkdir(parents=True,exist_ok=True)
    draw(int(study.population()[0,0]),destination)
