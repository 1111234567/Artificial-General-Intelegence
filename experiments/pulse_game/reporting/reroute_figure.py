"""Plot actual first-choice differences for the developmental and within-body comparisons."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVIDENCE=Path(__file__).resolve().parents[3]/"docs/evidence/reroute-20261120"


def main():
    primary=json.loads((EVIDENCE/"eval-results.json").read_text(encoding="utf8"))["worlds"]
    held=json.loads((EVIDENCE/"novel-map-results.json").read_text(encoding="utf8"))["results"]["unseen"]
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,
                         "axes.spines.right":False,"axes.spines.top":False,"font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,title,comparison,limits in zip(axes,("多教下一房是否有用","同一个体的内部试探是否有用"),
                    ("next_room/natural minus terminal/natural","next_room/natural minus next_room/blank"),
                    ((-35,50),(-5,8)),strict=True):
        for rep,(pseed,hseed) in enumerate(zip(primary, (5469301,5469361),strict=True)):
            observed=primary[pseed]["contrasts"][comparison]
            strict=held[comparison]
            entries=[observed["flip_optimal"],observed["stay_optimal"],
                     strict[f"flip/{hseed}/choice"],strict[f"stay/{hseed}/choice"]]
            for i,entry in enumerate(entries):
                mean=entry["mean"]*100;lo,hi=(100*x for x in entry["ci95"])
                ax.errorbar(mean,3-i+(.11 if rep==0 else -.11),
                            xerr=[[mean-lo],[hi-mean]],fmt="o",capsize=3,color=("#196e65","#ab5b38")[rep],
                            label=f"抽样{rep+1}" if i==3 else None)
        ax.axvline(0,color="#6e7883",lw=1)
        ax.axhline(1.5,color="#e3e6e9",lw=.8)
        ax.set(title=title,xlim=limits,ylim=(-.55,3.5),yticks=[3,2,1,0],
               yticklabels=["同域 · 应换门","同域 · 应保留","未见旧图 · 应换门","未见旧图 · 应保留"])
        ax.set_xlabel("首次真实选优率差（百分点）")
        ax.grid(axis="x",alpha=.16);ax.legend(loc="lower right",frameon=False,fontsize=8)
    fig.savefig(EVIDENCE/"first-choice.png",dpi=165)
    plt.close(fig)


if __name__=="__main__":
    main()
