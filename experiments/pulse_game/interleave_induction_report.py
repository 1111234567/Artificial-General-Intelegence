"""Re-score paired unseen-object actions and uncertainty from the saved physical records."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/interleaved-learning-20261120"
EVIDENCE=ROOT/"docs/evidence/interleaved-learning-2026-11-20"
CUTS=(0,1,2,4,8,16,32,64)


class Nested:
    def __init__(self,rows,worlds,seed,draws=3000):
        rng=np.random.default_rng(seed)
        self.roots=rng.integers(rows,size=(draws,rows),dtype=np.int16)
        self.worlds=rng.integers(worlds,size=(draws,rows,worlds),dtype=np.int16)

    def interval(self,mat):
        values=np.asarray(mat)
        sample=values[self.roots[:,:,None],self.worlds].mean((1,2))
        return dict(mean=float(values.mean()),ci95=np.quantile(sample,[.025,.975]).tolist(),
                    by_birth=values.mean(1).tolist())


def process(label):
    protocol=json.loads((EVIDENCE/f"induction-{label}.json").read_text())
    roots=protocol["births"];fields=protocol["fields"]
    bs=Nested(len(roots),protocol["worlds_per_birth"],7157)
    result=dict(births=roots,worlds_per_birth=protocol["worlds_per_birth"],source=label,profiles={},sha256={})
    saved=[]
    try:
        for b in roots:
            f=DATA/"induction"/label/f"{b}.npz"
            digest=sha256(f.read_bytes()).hexdigest()
            if digest!=protocol["sha256"][f.name]:raise ValueError(f"Raw induction file changed {f}")
            result["sha256"][f.name]=digest;saved.append(np.load(f))
        for geom,p in (("linear",0.),("linear",.2),("cubic",0.)):
            prefix=f"{geom}/{p}"
            if f"{prefix}/coherent/0" not in saved[0]:continue
            profile={}
            for k in protocol["cuts"]:
                probes={name:np.stack([z[f"{prefix}/{name}/{k}"] for z in saved]).reshape(
                    len(roots),-1,4,len(fields)).astype(float) for name in ("coherent","false_history","unrelated_history","sham_history")}
                # Within-geometry columns are [negative rule exceptional X, negative rule usual X,
                # positive rule exceptional X, positive rule usual X]; none of these are independent births.
                def get(mode,field,pos):
                    return probes[mode][:,:,pos,fields.index(field)].mean(2)
                pos=[1,3];neg=[0,2]
                x=get("coherent","novel_X",pos);y=get("coherent","novel_Y",pos)
                ex=get("coherent","X_after_feedback",neg)
                stats={
                    "first_unseen":x,"second_unseen":y,
                    "first_unseen_minus_independent_history":x-get("unrelated_history","novel_X",pos),
                    "first_unseen_minus_opposite_history":x-get("false_history","novel_X",pos),
                    "first_unseen_minus_history_without_writes":x-get("sham_history","novel_X",pos),
                    "public_sum":get("coherent","public_sum_X",pos),"public_nearest":get("coherent","nearest_X",pos),
                    "exception_first":get("coherent","novel_X",neg),"exception_after_one":ex,
                    "exception_gain":ex-get("coherent","novel_X",neg),
                    "new_typical_after_exception":get("coherent","Y_after_X",neg),
                    "exception_and_new_typical":get("coherent","both_after_X",neg),
                    "typical_after_one":get("coherent","X_after_feedback",pos),
                }
                profile[k]={name:bs.interval(arr) for name,arr in stats.items()}
            result["profiles"][prefix]=profile
    finally:
        for f in saved:f.close()
    return result


def figure(result,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    data=result["profiles"]["linear/0.0"]
    ks=list(data); xx=np.arange(len(ks))
    fig,ax=plt.subplots(figsize=(7.4,3.8),layout="constrained")
    for name,label,color in (("first_unseen","全新对象，未给反馈","#137c79"),
                             ("exception_after_one","例外对象，获得一次纠正后","#b65b37"),
                             ("public_sum","手写三维向量参照，全新对象","#858891")):
        means=100*np.array([data[k][name]["mean"] for k in ks])
        bounds=100*np.array([data[k][name]["ci95"] for k in ks])
        ax.plot(xx,means,marker="o",linewidth=1.7,label=label,color=color)
        if name=="first_unseen":ax.fill_between(xx,bounds[:,0],bounds[:,1],alpha=.14,color=color)
    ax.axhline(50,color="#b5b5b5",lw=.8)
    ax.set_xticks(xx,ks);ax.set_ylim(0,103);ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel("此前体验的不同对象数（测点等距）");ax.set_ylabel("实际选择正确率（%）")
    ax.grid(axis="y",alpha=.16);ax.legend(loc="lower center",frameon=False,fontsize=9)
    fig.savefig(path,dpi=170);plt.close(fig)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--labels",nargs="+",required=True)
    p.add_argument("--name",default="induction-checked")
    a=p.parse_args()
    results={key:process(key) for key in a.labels}
    (EVIDENCE/(a.name+".json")).write_text(json.dumps(results,indent=1)+"\n",encoding="utf8")
    figure(results[a.labels[0]],EVIDENCE/(a.name+".png"))
