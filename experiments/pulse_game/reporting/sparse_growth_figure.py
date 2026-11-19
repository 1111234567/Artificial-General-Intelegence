"""Show actual choices and installed interactions while a life experiences a hidden reversal."""
from pathlib import Path

import numpy as np

from ..ring_logic_study import REPO


def draw(source, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","SimHei")
               if f in {v.name for v in font_manager.fontManager.ttflist}),None)
    if font:plt.rcParams["font.family"]=font
    fig,(left,right)=plt.subplots(1,2,figsize=(10.1,3.75),layout="constrained")
    colors=["#166880","#ba8247","#898d92"]
    arms=["guided","dense","random"]
    names=["回执选位置","稠密更新","随机选位置"] if font else ["Receipt-guided","Dense","Random locations"]
    with np.load(source,allow_pickle=False) as f:
        length=f["guided/correct"].shape[1]
        cuts=list(range(0,length-112,48))+[length-112,length]
        cuts=sorted(set(cuts))
        ticks=(np.array(cuts[1:])+np.array(cuts[:-1]))/2
        for col,arm,label in zip(colors,arms,names):
            score=f[arm+"/correct"].mean(axis=0)
            percent=[score[i:j].mean()*100 for i,j in zip(cuts[:-1],cuts[1:])]
            left.plot(ticks,percent,color=col,linewidth=2,label=label)
            count=f[arm+"/active"].mean(axis=0)
            if len(count)!=length+1:raise ValueError("Expected the birth and all post-request support counts")
            right.plot(np.arange(length+1),count,color=col,label=label,lw=1.7)
        for ax in (left,right):
            ax.axvspan(np.min(f["change"]),np.max(f["change"]),color="#a9879d",alpha=.12)
            ax.axvspan(length-112,length,color="#8faa8b",alpha=.16)
            ax.set(xlim=(0,length),xlabel="一生经历的请求数" if font else "Requests in one lifetime")
            ax.spines[["top","right"]].set_visible(False)
    left.set(ylabel="实际选对率（%）" if font else "Correct contacts (%)",ylim=(32,86),
             title="原生图在偏好反转前后如何行动" if font else "Physical choices before and after reversal")
    left.axhline(50,lw=.7,color="#666",alpha=.8);left.legend(frameon=False,loc="upper left",fontsize=8)
    right.set_yscale("symlog",linthresh=10)
    right.set(ylim=(0,1100),yticks=[0,10,30,100,300,800],yticklabels=["0","10","30","100","300","800"],
              title="当时装在图里的二阶系数" if font else "Installed interaction coefficients",
              ylabel="每图平均数量（分段对数刻度）" if font else "Mean count (symlog)")
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=155);plt.close(fig)


if __name__=="__main__":
    draw(REPO/"data/pulse-game/sparse-native-20261119/reverse.npz",
         REPO/"docs/evidence/sparse-native-2026-11-19/growth-and-performance.png")
