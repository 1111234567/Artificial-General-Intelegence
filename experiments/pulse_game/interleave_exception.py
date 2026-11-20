"""An identical surprising X can be one exception or the first sign of a global reversal."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_stream as stream
from .interleave_interference import DATA,EVIDENCE,ROOT


DOSES=(0,1,2,4,8,16)
QUESTIONS=("exception_x","unseen_y","near_x","far_x","old_object")


def worlds(seed,birth,n):
    rng=np.random.default_rng([711031,seed,birth])
    # Eight coherent historical examples, one surprising X, then distinct diagnostic objects.
    v=age.directions(rng,n)/1.35
    h=age.directions(rng,n,8)
    x=age.directions(rng,n)/1.35
    z=age.directions(rng,n,16)
    y=age.directions(rng,n)
    tangent=age.directions(rng,n)/1.35
    tangent-=(tangent*x).sum(1,keepdims=True)*x
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    def nearby(deg):
        d=np.deg2rad(deg)
        return (1.35*(np.cos(d)*x+np.sin(d)*tangent)).astype(np.float32)
    objects=np.concatenate((h,(1.35*x)[:,None],z),1)
    queries=np.stack((1.35*x,y,nearby(15),nearby(75),h[:,0]),1)
    def repeat(q):return torch.from_numpy(np.repeat(q,4,axis=0))
    # Four lanes are (original world sign ±) × (exception vs switch).
    old=np.tile(np.array([-1.,-1.,1.,1.],dtype=np.float32),n)
    flip=np.tile(np.array([1.,-1.,1.,-1.],dtype=np.float32),n)
    original=np.repeat(np.where((objects*v[:,None]).sum(2)>=0,1.,-1.).astype(np.float32),4,0)*old[:,None]
    values=original.copy();values[:,8]*=-1;values[:,9:]*=flip[:,None]
    test=np.repeat(np.where((queries*v[:,None]).sum(2)>=0,1.,-1.).astype(np.float32),4,0)*old[:,None]
    test[:,0]*=-1;test[:,1:]*=flip[:,None]
    return dict(objects=repeat(objects),values=torch.tensor(values),queries=repeat(queries),truth=torch.tensor(test),
                flip=torch.tensor(flip),orient=repeat(age.signs(rng,(n,len(objects[0])))),
                probe_orient=repeat(age.signs(rng,(n,len(queries[0])))),
                draws=repeat(rng.random((n,len(objects[0]),6),dtype=np.float32)),
                probe_draw=repeat(rng.random((n,len(queries[0])),dtype=np.float32)))


@torch.no_grad()
def evaluate(g,rule,w,mode):
    n=len(w["values"]);state=stream.start(g,rule,n);coeff=None if rule is None else rule.expand(n)
    history=[];sides=[];prob=[]
    for t in range(w["objects"].shape[1]):
        x=w["objects"][:,t]*w["orient"][:,t,None]
        value=w["values"][:,t]*w["orient"][:,t]
        state,side,r=stream.contact(coeff,state,-x,x,value,torch.zeros_like(value),w["draws"][:,t],
                                    "sham" if mode=="sham" else "active")
        state=stream.wait(coeff,state,4)
        history.append(torch.stack((side,r),1).to(torch.int8))
        if t-8 not in DOSES:continue
        acts=[];ps=[]
        for q in range(len(QUESTIONS)):
            v=w["queries"][:,q]*w["probe_orient"][:,q,None]
            a=stream.action(coeff,state,-v,v,w["probe_draw"][:,q])
            goal=w["truth"][:,q]*w["probe_orient"][:,q]
            acts.append(a.side.to(torch.int8));ps.append((a.motor_logit*goal).sigmoid())
        sides.append(torch.stack(acts,1));prob.append(torch.stack(ps,1))
    hist=torch.stack(history,1)
    # Both put the very same food on the chosen side of X. No unannounced change
    # can be distinguished from the local exception at this point.
    if not torch.equal(hist[::2,:9],hist[1::2,:9]) or not torch.equal(sides[0][::2],sides[0][1::2]):
        raise AssertionError("Exception and reversal became distinguishable before diagnostic feedback")
    return dict(side=torch.stack(sides).numpy(),p=torch.stack(prob).numpy(),physical=hist.numpy(),
                target=(w["truth"]*w["probe_orient"]).numpy())


def run(src,label,births,worlds_count,seed):
    torch.set_num_threads(1)
    dest=DATA/"exception"/label;dest.mkdir(parents=True,exist_ok=True)
    result=dict(births=births,worlds_per_birth=worlds_count,seed=seed,doses=DOSES,queries=QUESTIONS,
                notes="The two causal worlds are identical through the unexpected taste of X. The unannounced "
                      "global reversal and isolated local exception yield different actual subsequent tastes on new samples. "
                      "Each diagnostic dose is followed by independent no-feedback queries from one state. Near/far "
                      "denotes angles to X, not guaranteed agreement with the pre-change label.",
                roots={},sha256={})
    for b in births:
        g,r=src.load(b);w=worlds(seed,b,worlds_count);packed={};scores={}
        for mode in ("active","sham"):
            z=evaluate(g,r,w,mode);good=z["side"]==z["target"][None]
            for i,d in enumerate(DOSES):
                for labelname,mask in (("local",w["flip"].numpy()>0),("global",w["flip"].numpy()<0)):
                    score={k:float(x) for k,x in zip(QUESTIONS,good[i,mask].mean(0),strict=True)}
                    score["X_and_Y"]=float(good[i,mask,:2].all(1).mean())
                    score["all_five"]=float(good[i,mask].all(1).mean())
                    scores[f"{mode}/{d}/{labelname}"]=score
            packed.update({mode+"/"+k:v for k,v in z.items()})
        result["roots"][b]=scores
        path=dest/f"{b}.npz";np.savez_compressed(path,**packed)
        result["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"exception-{label}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")
    print(label,"complete",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True);p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=128);p.add_argument("--seed",type=int,default=4561937)
    a=p.parse_args();run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
