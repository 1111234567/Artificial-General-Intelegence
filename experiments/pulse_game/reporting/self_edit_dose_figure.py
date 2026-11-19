"""Plot portable editing skill against actual autonomous life experience."""
from pathlib import Path

import numpy as np

from .. import ring_logic_study as io


def draw(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=next((name for name in ("Microsoft YaHei","Noto Sans CJK SC","SimHei")
               if name in {f.name for f in font_manager.fontManager.ttflist}),None)
    if font:plt.rcParams["font.family"]=font
    reference=io.read(folder/"life/summary.json")
    expanded=io.read(folder/"life/expanded-summary.json")["writer_widths"]
    fig,ax=plt.subplots(figsize=(7.1,3.65),layout="constrained")
    doses=(0,8,32,128,512)
    for j,(bank,name,en,color,mark) in enumerate((
        (reference,"原写入器","Earlier writer","#286a83","o"),
        (expanded["4"],"全地址 · 4 个写入节点","Full address / four","#b06d3c","s"),
        (expanded["8"],"全地址 · 8 个写入节点","Full address / eight","#67824d","^")
    )):
        val=[bank["doses"][str(d)]["contrasts"]["writer_vs_birth_null"]["mean_post_receipt_gain"] for d in doses]
        mean=np.array([v["mean"] for v in val])*100
        low,high=np.array([v["ci95"] for v in val]).T*100
        x=np.arange(len(doses))+(j-1)*.06
        ax.errorbar(x,mean,yerr=(mean-low,high-mean),capsize=3,color=color,marker=mark,lw=1.5,
                    label=name if font else en)
    ax.axhline(0,color="#777",lw=.8);ax.set(xticks=np.arange(len(doses)),xticklabels=doses,
          xlabel="供体在同一生中完成的新题数" if font else "New problems in one donor lifetime",
          ylabel="新身体的早期学习收益差（百分点）" if font else "Transfer minus naive (percentage points)")
    ax.legend(loc="lower center",fontsize=8,ncol=3,frameon=False);ax.set_ylim(-18,8)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(folder/"lifespan-transfer.png",dpi=175);plt.close(fig)


if __name__=="__main__":
    draw(io.REPO/"docs/evidence/self-edit-dose-2026-11-19")
