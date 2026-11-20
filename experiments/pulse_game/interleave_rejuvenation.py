"""Locate cumulative damage to a graph's learning ability by restoring small sets of incoming rows."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_age_train as age, interleave_stream as stream, local_trace_graph as net
from .interleave_interference import DATA,EVIDENCE,ROOT


AGES=(4,16,64,128)
NAME=("intact",*(f"row_{i}" for i in range(20)),"sensory","contact_taste","motor","internal","all_common","birth_graph")
ROW_SETS=((),*((i,) for i in range(20)),tuple(range(6)),tuple(range(6,10)),(10,11),
          tuple(range(12,20)),tuple(range(20)),())


def tapes(seed,birth,worlds):
    rng=np.random.default_rng([865319,seed,birth])
    def public(v):return torch.as_tensor(np.repeat(v,4,axis=0))
    max_k=max(AGES)
    law=np.repeat(age.signs(rng,(worlds,max_k+1)),4,axis=0)
    law[:,0]=np.tile([-1,-1,1,1],worlds)
    return dict(x=public(age.directions(rng,worlds,max_k+1)),law=torch.from_numpy(law),
                xlaw=torch.tensor([-1.,1.,-1.,1.]).repeat(worlds),
                flip=public(age.signs(rng,(worlds,max_k+4))),u=public(rng.random((worlds,max_k+4,6),dtype=np.float32)),
                old_probe_flip=public(age.signs(rng,worlds)),old_probe_u=public(rng.random(worlds,dtype=np.float32)))


def restore(g,original,state,rows,birth_graph):
    if birth_graph:return replace(state,graph=original)
    if not rows:return state
    selected=torch.zeros(g.width,dtype=torch.bool)
    selected[list(rows)]=True
    common=selected[None,:,None]&g.present&original.present
    return replace(state,graph=replace(g,weights=torch.where(common,original.weights,g.weights)))


@torch.no_grad()
def record(g,rule,t,chunk=64):
    n=len(t["law"])
    coeff=rule.expand(n)
    g0=net.start(g,n)
    state=g0
    data={}
    for k in range(max(AGES)+1):
        if k in AGES:
            collected=[]
            # Dozens of physical surgeries on the same snapshot; small chunks cap memory.
            for offset in range(0,n,chunk):
                sl=slice(offset,min(n,offset+chunk))
                s=state.subset(sl);original=g0.subset(sl).graph
                versions=[restore(s.graph,original,s,rows,name=="birth_graph") for name,rows in zip(NAME,ROW_SETS,strict=True)]
                graph=net.Graph(*(torch.cat([getattr(v.graph,q) for v in versions]) for q in ("weights","bias","present")))
                many=net.State(graph,*(torch.cat([getattr(v,q) for v in versions])
                                       for q in ("hidden","sender_trace","receiver_trace")))
                def rep(v):return torch.cat([v[sl]]*len(versions))
                active=rule.expand(len(many.hidden))
                old=rep(t["x"][:,0]*t["old_probe_flip"][:,None])
                old_goal=rep(t["old_probe_flip"]*t["law"][:,0])
                old_u=rep(t["old_probe_u"])
                old_before=stream.action(active,many,-old,old,old_u)
                pos=rep(t["x"][:,-1]*t["flip"][:,-2,None])
                target=rep(t["xlaw"]*t["flip"][:,-2])
                seen,_,_=stream.contact(active,many,-pos,pos,target,torch.zeros_like(target),rep(t["u"][:,-2]),"active")
                seen=stream.wait(active,seen,4)
                pos=rep(t["x"][:,-1]*t["flip"][:,-1,None])
                target=rep(t["xlaw"]*t["flip"][:,-1])
                new=stream.action(active,seen,-pos,pos,rep(t["u"][:,-1,0]))
                old_after=stream.action(active,seen,-old,old,old_u)
                numeric_cost=(graph.weights-torch.cat([s.graph.weights]*len(versions))).flatten(1).abs().sum(1)
                values=torch.stack((old_before.side==old_goal,new.side==target,old_after.side==old_goal,
                                    (new.side==target)&(old_after.side==old_goal),numeric_cost),1)
                collected.append(values.reshape(len(NAME),-1,5).numpy())
            data[str(k)]=np.concatenate(collected,axis=1)
        if k==max(AGES):break
        pos=t["x"][:,k]*t["flip"][:,k,None]
        law=t["law"][:,k]*t["flip"][:,k]
        state,_,_=stream.contact(coeff,state,-pos,pos,law,torch.zeros_like(law),t["u"][:,k],"active")
        state=stream.wait(coeff,state,4)
    return data


def run(source,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"rejuvenation"/label
    dest.mkdir(parents=True,exist_ok=True)
    out=dict(seed=seed,worlds_per_split=worlds,births=births,history_ages=AGES,branches=NAME,
             outcomes=["old_before","new","old_after","old_and_new","weight_L1_intervention"],
             note="Old A sign crossed independently with fresh X sign, common intervening B worlds. "
                  "Restore only surviving incoming weights before exposing X; keep current support, bias, activity and "
                  "traces unless using the explicit full birth-graph diagnostic. Individual-row selection uses a "
                  "separate calibration split and is external to the model.",roots={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        cal=record(g,r,tapes(seed,birth,worlds))
        test=record(g,r,tapes(seed+1,birth,worlds))
        result={};packed={}
        for k in map(str,AGES):
            train=cal[k].mean(1);held=test[k].mean(1)
            choice=int(train[1:21,3].argmax())+1
            result[k]={"means":held.tolist(),"selected_row":choice-1,
                       "selected_joint_gain":float(held[choice,3]-held[0,3])}
            packed["cal/"+k]=cal[k];packed["test/"+k]=test[k]
        path=dest/f"{birth}.npz";np.savez_compressed(path,**packed)
        out["roots"][birth]=result;out["sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
        print(label,birth,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"rejuvenation-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=48);p.add_argument("--seed",type=int,default=792703)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
