"""Paired uncertainty for slow consolidation, with world and birth as separate units."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/"docs/evidence/interleaved-learning-2026-11-20"
RAW=ROOT/"data/interleaved-learning-20261120/slow-reference"


def run():
    data=json.loads((RAW/"summary.json").read_text())
    fields=data["manifest"]["fields"]
    roots=data["confirmation_births"]
    pairs=data["manifest"]["test_worlds"]
    rng=np.random.default_rng(887132)
    draws=5000
    row=rng.multinomial(len(roots),np.ones(len(roots))/len(roots),draws)/len(roots)
    col=rng.multinomial(pairs,np.ones(pairs)/pairs,draws)/pairs
    def estimate(v):
        samples=((row@v)*col).sum(1)
        return dict(mean=float(v.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),by_birth=v.mean(1).tolist())
    out=dict(source_sha256=sha256((RAW/"summary.json").read_bytes()).hexdigest(),births=roots,
             worlds_per_birth=pairs,beta=data["selected_beta"],gamma=.12,draws=draws,
             scoring="Average over paired A,B,X laws within each independent geometry, then prespecified ages "
                     "16,64,128. Paired resampling across training births and shared geometries. Intervals pointwise.",
             results={})
    for ec in ("absent_a","recurring"):
        for amount in (1,4):
            tables={}
            for name in ("native","static","selected","opposite_A_slow"):
                rows=[]
                for b in roots:
                    fname=f"{b}-{ec}-{amount}-{name}.npz"
                    f=RAW/"confirmation"/fname
                    if sha256(f.read_bytes()).hexdigest()!=data["raw_sha256"][fname]:
                        raise ValueError(f"Changed data: {b}/{name}")
                    with np.load(f) as z:rows.append(z["correct"].mean((2,3)))
                tables[name]=np.stack(rows)
            cells={}
            for age,index in (("ages16_64_128",[2,3,4]),("age128",[4])):
                sliced={k:v[:,index].mean(1) for k,v in tables.items()}
                for a,b in (("static","native"),("selected","static"),("selected","native"),("selected","opposite_A_slow")):
                    sliced[a+"-minus-"+b]=sliced[a]-sliced[b]
                cells[age]={name:{key:estimate(v[:,:,i]) for i,key in enumerate(fields)} for name,v in sliced.items()}
            out["results"][f"{ec}/{amount}"]=cells
    (EVIDENCE/"slow-reference-contrasts.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")
    fig(out)


def fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(1,2,figsize=(10.1,3.8),layout="constrained",sharey=True)
    for ax,ec,title in zip(axes,("absent_a","recurring"),("旧 A 在随后经历中缺席","旧 A 在随后经历中仍会重现"),strict=True):
        tab=out["results"][ec+"/4"]["ages16_64_128"]
        xp=np.arange(3)
        for i,(name,description,color) in enumerate((("native","原局部写入","#7d8795"),
                                                    ("static","向固定出生值回归","#c47140"),
                                                    ("selected","向慢速经验均值回归","#187b74"))):
            vals=[tab[name][k] for k in ("old_after","new_X","all_three")]
            y=np.array([v["mean"] for v in vals])*100
            bounds=np.array([v["ci95"] for v in vals])*100
            ax.errorbar(xp+(i-1)*.11,y,yerr=np.vstack((y-bounds[:,0],bounds[:,1]-y)),fmt="o",
                        color=color,label=description,capsize=3,ms=6)
        ax.set_title(title);ax.set_xticks(xp,["旧 A","新 X","A、B、X 都正确"])
        ax.set_xlim(-.45,2.45);ax.set_ylim(0,100);ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("实际正确率（%）")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncols=3,frameon=False)
    fig.savefig(EVIDENCE/"slow-reference.png",dpi=170);plt.close(fig)


if __name__=="__main__":run()
