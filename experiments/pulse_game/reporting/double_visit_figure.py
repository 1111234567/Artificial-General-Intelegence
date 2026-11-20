"""Plot readout and causal evidence about the first local observation across the second."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EVIDENCE=Path(__file__).resolve().parents[3]/"docs/evidence/double-visit-20261120"


def main():
    decoded=json.loads((EVIDENCE/"memory-results.json").read_text(encoding="utf8"))["split"]["odd_heldout"]["result"]
    exchange=json.loads((EVIDENCE/"trace-swap-results.json").read_text(encoding="utf8"))["conditions"]["identity"]["heldout"]
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"font.size":10,"axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    palette=("#11756a","#ad6234")
    for i,(key,label) in enumerate((("primary","主区 20 个活动值"),("full","完整神经状态"))):
        x=np.arange(2)+(i-.5)*.12
        rows=[decoded["identity/"+key][m] for m in ("first/updated","second/different/first")]
        y=[r["mean"]*100 for r in rows]
        intervals=np.array([r["ci95"] for r in rows])*100
        axes[0].errorbar(x,y,yerr=[np.asarray(y)-intervals[:,0],intervals[:,1]-y],
                         marker="o",lw=2,capsize=3,label=label,color=palette[i])
    axes[0].axhline(100/3,color="#86929d",lw=.8)
    axes[0].set(ylabel="读对第一间房当前终点（%）",ylim=(20,105),title="同一笔旧事实还能否被读出")
    for i,(key,label) in enumerate((("activity_trace","换活动和慢迹"),("fast_candidate","换快权重和待写增量"))):
        x=np.arange(2)+(i-.5)*.12
        rows=[exchange[t][key]["both_correct_change"] for t in ("after_first","after_second")]
        y=-100*np.array([r["mean"] for r in rows])
        bounds=-100*np.array([r["interval95"][::-1] for r in rows])
        axes[1].errorbar(x,y,yerr=[y-bounds[:,0],bounds[:,1]-y],marker="o",lw=2,capsize=3,
                         label=label,color=palette[i])
    axes[1].axhline(0,color="#86929d",lw=.8)
    axes[1].set(ylabel="互换后配对双对率的损失（百分点）",ylim=(-4,28),title="互换哪部分更会改变实际选择")
    for i,ax in enumerate(axes):
        ax.set(xticks=[0,1],xticklabels=["第一笔结束（18 拍）","第二笔结束（24 拍）"],xlim=(-.2,1.2))
        ax.grid(axis="y",alpha=.16);ax.legend(frameon=False,loc="lower left" if i==0 else "upper left",fontsize=9)
    fig.savefig(EVIDENCE/"first-update.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
