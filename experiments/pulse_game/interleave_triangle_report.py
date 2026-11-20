"""Score difficult sign histories, real actions and forward-order sensitivity separately."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .interleave_induction_report import Nested


ROOT=Path(__file__).resolve().parents[2]
D=ROOT/"data/interleaved-learning-20261120"
E=ROOT/"docs/evidence/interleaved-learning-2026-11-20"


def per_world(z):
    labels=z["laws"]
    orders=len([s for s in z.files if s.startswith("active_logit_")])
    raw=np.stack([z[f"active_logit_{j}"].reshape(*labels.shape,2) for j in range(orders)])
    sampled=np.stack([z[f"active_selected_{j}"].reshape(*labels.shape,2) for j in range(orders)])
    correct=sampled==labels[None,...,None]
    signed=(raw[...,0]-raw[...,1])*labels[None]/2
    # [order, geometry, map, item]; maps 0 and 7 assign all the same canonical sign.
    hard=signed[:,:,[0,7]]
    easy=signed[:,:,1:7]
    worst=hard.min(axis=(2,3))
    raw_signed=raw*labels[None,...,None]*np.array([1.,-1.])
    conditional=1/(1+np.exp(-raw_signed))
    actual_three=correct.all(3)
    expected_three=conditional.prod(3)
    result={
        "paired_hard_margin005":(worst>.05).mean(0),
        "paired_hard_margin05":(worst>.5).mean(0),
        "paired_hard_margin1":(worst>1).mean(0),
        "paired_hard_margin2":(worst>2).mean(0),
        "paired_hard_margin005_every_order":(worst>.05).all(0),
        "all_eight_margin005_every_order":(signed>.05).all((0,2,3)),
        "mixed_six_margin005":(easy>.05).all(3).mean((0,2)),
        "hard_actual_all3":actual_three[:,:,[0,7]].mean((0,2,3)),
        "easy_actual_all3":actual_three[:,:,1:7].mean((0,2,3)),
        "hard_actual_each":correct[:,:,[0,7]].mean((0,2,3,4)),
        "hard_expected_all3":expected_three[:,:,[0,7]].mean((0,2,3)),
        "hard_min_signed_margin":worst.mean(0),
    }
    return result


def collect(label,births,folder,step):
    data=[];hashes={}
    manifest=json.loads((E/f"triangle-orders-{label}-{step}.json").read_text())
    for b in births:
        p=folder/label/f"{b}-{step}.npz"
        h=sha256(p.read_bytes()).hexdigest()
        if h!=manifest["sha256"][p.name]:raise ValueError(f"Changed order replay {p}")
        hashes[p.name]=h
        with np.load(p) as z:data.append(per_world(z))
    return {k:np.stack([v[k] for v in data]) for k in data[0]},hashes


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--steps",nargs="+",type=int,default=[32000,48000,64000])
    p.add_argument("--out",type=Path,default=E/"triangle-long-results.json")
    a=p.parse_args()
    out=dict(interpretation="Sampled physical all-three correctness is reported separately on the two hard and six easy "
                            "law maps. The paired margin criterion symmetrizes motor sides and requires all three strict "
                            "signs for both opposite hard histories. A geometry/order is the unit of measurement; inference "
                            "resamples births then independent geometries, keeping six orders and all laws paired.",
             threshold=.05,results={},sha256={})
    for label,roots in (("long-original",range(731,737)),("long-independent",range(811,817))):
        series={};stats={}
        for t in a.steps:
            x,hashes=collect(label,roots,D/"long-order-verification",t)
            bs=Nested(*x["hard_actual_all3"].shape,781351,3000)
            stats[t]={k:bs.interval(v) for k,v in x.items()}
            series[t]=x;out["sha256"][f"{label}/{t}"]=hashes
        if len(a.steps)>1:
            start,end=a.steps[0],a.steps[-1]
            stats["paired_change"]={k:bs.interval(v-series[start][k]) for k,v in series[end].items()}
        out["results"][label]=stats
    a.out.write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")
    chart(out,a.steps,a.out.with_name("triangle-learning.png"))


def chart(out,steps,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    fig,(a,b)=plt.subplots(1,2,figsize=(9.8,3.7),layout="constrained")
    for lab,title,color in (("long-original","最初六个出生","#087e79"),("long-independent","独立六个出生","#b65f35")):
        for ax,metric,ytitle in ((a,"paired_hard_margin005","两种相反历史都通过的几何比例"),
                                 (b,"hard_actual_all3","困难历史下三项实际全对率")):
            vals=out["results"][lab]
            mu=np.array([vals[t][metric]["mean"] for t in steps])*100
            lo=np.array([vals[t][metric]["ci95"] for t in steps])*100
            ax.plot(steps,mu,marker="o",color=color,label=title)
            ax.fill_between(steps,lo[:,0],lo[:,1],color=color,alpha=.12)
            ax.set_title(ytitle);ax.set_ylim(0,100);ax.set_xlabel("三角形任务累计更新次数")
            ax.grid(axis="y",alpha=.15);ax.spines[["top","right"]].set_visible(False);ax.set_xticks(steps)
    a.set_ylabel("百分比")
    h,l=a.get_legend_handles_labels();fig.legend(h,l,loc="outside lower center",ncols=2,frameon=False)
    fig.savefig(path,dpi=165);plt.close(fig)


if __name__=="__main__":main()
