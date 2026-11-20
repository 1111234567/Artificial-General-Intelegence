"""Measure renewable learning and original-item retention at the same naturally reached ages."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


CUTS=(0,1,4,16,64,128,256)
MEASURES=("first_old","x_one_taste","x_after_y","y_after_x_y","xy_both","old_after_xy","x_y_old")


def make(seed,birth,worlds,horizon,history):
    rng=np.random.default_rng([941733,seed,birth])
    pool=age.directions(rng,worlds,max(horizon,4)+2)
    labels=age.signs(rng,pool.shape[:2])
    ids=np.tile(np.arange(horizon),(worlds,1))
    repeat_ids=rng.integers(0,4,size=(worlds,horizon))
    if horizon:repeat_ids[:,0]=0
    if history=="recurring":ids=repeat_ids
    elif history!="novel":raise ValueError("Unknown life history")
    suffix=np.stack((np.full(worlds,pool.shape[1]-2),np.full(worlds,pool.shape[1]-1)),1)
    order=np.concatenate((ids,suffix),1)
    rows=np.arange(worlds)[:,None]
    ori=age.signs(rng,order.shape)
    r=pool[rows,order]*ori[...,None]
    value=labels[rows,order]*ori
    # Full crossing of past, X and Y hidden assignments within every geometry.
    signs=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)],dtype=np.float32)
    flip=np.tile(signs,(worlds,1))
    def rep(x):return torch.as_tensor(np.repeat(x,8,axis=0))
    target=rep(value)
    target[:,:horizon]*=torch.from_numpy(flip[:,0,None])
    target[:,horizon:]*=torch.from_numpy(flip[:,1:])
    return dict(right=rep(r),goal=target,draw=rep(rng.random((*order.shape,6),dtype=np.float32)),
                expose_gap=rep(rng.integers(0,9,size=order.shape)),probe_flip=rep(age.signs(rng,(worlds,3))),
                probe_draw=rep(rng.random((worlds,3),dtype=np.float32)),horizon=horizon)


def see(r,state,t,i,mode):
    right=t["right"][:,i];goal=t["goal"][:,i];draw=t["draw"][:,i]
    state,_,_=stream.contact(r,state,-right,right,goal,torch.zeros_like(goal),draw,mode)
    return stream.wait(r,state,t["expose_gap"][:,i])


def read(r,state,t,i,k):
    right=t["right"][:,i]*t["probe_flip"][:,k,None]
    target=t["goal"][:,i]*t["probe_flip"][:,k]
    a=stream.action(r,state,-right,right,t["probe_draw"][:,k])
    return a.side==target


@torch.no_grad()
def evaluate(g,rule,tape,cuts=CUTS):
    n=len(tape["goal"]);r=None if rule is None else rule.expand(n)
    results={}
    for prefix in ("active","sham"):
        state=stream.start(g,rule,n)
        h=tape["horizon"]
        for i in range(h+1):
            if i in cuts:
                old=read(r,state,tape,0,0)
                for mode in ("active","sham"):
                    with_x=see(r,state,tape,h,mode)
                    x1=read(r,with_x,tape,h,1)
                    with_xy=see(r,with_x,tape,h+1,mode)
                    x=read(r,with_xy,tape,h,1);y=read(r,with_xy,tape,h+1,2)
                    ao=read(r,with_xy,tape,0,0)
                    results[f"{prefix}/{mode}/{i}"]=torch.stack((old,x1,x,y,x&y,ao,ao&x&y),1).numpy()
            if i<h:
                state=see(r,state,tape,i,prefix)
    return results


def run(source,label,births,worlds,seed,cuts=CUTS):
    torch.set_num_threads(1)
    dest=DATA/"adult-evaluation"/label;dest.mkdir(parents=True,exist_ok=True)
    out=dict(seed=seed,worlds_per_birth=worlds,births=births,cuts=cuts,outcome_fields=MEASURES,
             note="Past private law vector and two independent fresh laws are fully crossed. The same two never-seen "
                  "X,Y objects and same study/query draws are reused at all ages of a matched life. Each measurement "
                  "forks the full live state. First_old is queried before its first study when age is zero. "
                  "Active/sham mean prefix write status and fresh suffix write status, respectively.",roots={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        rows={}
        for cond in ("novel","recurring"):
            tape=make(seed,birth,worlds,max(cuts),cond)
            raw=evaluate(g,r,tape,cuts)
            rows[cond]={k:v.mean(0).tolist() for k,v in raw.items()}
            p=dest/f"{cond}-{birth}.npz";np.savez_compressed(p,**raw)
            out["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
        out["roots"][birth]=rows
        print(label,birth,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"adult-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=48);p.add_argument("--seed",type=int,default=347311)
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
