"""Paired read-only capacity curves for any frozen overnight fit, and public-observation references."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_stream as stream
from .interleave_interference import DATA, ROOT


@torch.no_grad()
def evaluate_readback(g,r,tape,ks,seed):
    """Replay the same existing capacity tape using the new local readback calculation."""
    coeff=r.expand(tape.lanes)
    rng=np.random.default_rng([seed,84373])
    perms=torch.from_numpy(rng.random((tape.worlds,max(ks)+1,tape.contacts,20)).argsort(-1)).repeat_interleave(4,0)
    def study(state,obj,mode):
        for t in range(tape.contacts):
            draws=torch.cat((tape.motor_draw[:,obj,t:t+1],tape.edit_draw[:,obj,t]),1)
            state,_,_=stream.contact(coeff,state,tape.left[:,obj,t],tape.right[:,obj,t],tape.right_value[:,obj,t],
                                     tape.constant_value[:,obj,t],draws,mode,perms[:,obj,t])
        return state
    anchor=study(stream.start(g,r,tape.lanes),0,"active")
    states=dict.fromkeys(cap.BRANCHES,anchor)
    maxk=max(ks)
    shape=(len(cap.BRANCHES),len(ks),maxk+1,tape.lanes)
    side=np.zeros(shape,dtype=np.int8)
    logit=np.full(shape,np.nan,dtype=np.float32)
    distances=[]
    for k in range(maxk+1):
        if k:
            for mode in states:
                if mode=="neutral":
                    states[mode]=stream.wait(coeff,states[mode],cap.CONTACT_TICKS*tape.contacts)
                else:
                    states[mode]=study(states[mode],k,"active" if mode=="write" else "sham")
        if k not in ks:continue
        out=[]
        for b,state in enumerate(states.values()):
            snap=stream.wait(coeff,state,4 if tape.task=="odor" else 0)
            out.append(torch.stack(((state.graph.weights-anchor.graph.weights).abs().flatten(1).sum(1),
                                    (state.graph.bias-anchor.graph.bias).abs().sum(1),
                                    (state.graph.present!=anchor.graph.present).flatten(1).sum(1)),1).numpy())
            for item in range(k+1):
                a=stream.action(coeff,snap,tape.left[:,item,-1],tape.right[:,item,-1],tape.motor_draw[:,item,-1])
                side[b,ks.index(k),item]=a.side.numpy()
                logit[b,ks.index(k),item]=a.motor_logit.numpy()
        distances.append(np.stack(out))
    return dict(side=side,logit=logit,right_value=tape.right_value.numpy(),law=tape.law.numpy(),
                capacities=np.asarray(ks),parameter_distance_from_post_A=np.stack(distances,1))


def reference(tape, method):
    """A three-number linear memory; it sees only the item it physically selected and its taste."""
    x, y = tape.left.numpy(), tape.right.numpy()
    B, objects = x.shape[:2]
    vector = np.zeros((B, 3), dtype=np.float64)
    values = {}
    def decide(left,right,draw):
        margin = ((right-left)*vector).sum(-1)
        return np.where(margin > 1e-10, 1, np.where(margin < -1e-10, -1, np.where(draw<.5,1,-1)))
    for j in range(objects):
        for c in range(tape.contacts):
            left,right=x[:,j,c],y[:,j,c]
            side = decide(left,right,tape.motor_draw[:,j,c].numpy())
            selected = np.where(side[:,None]>0,right,left).astype(float)
            selected /= np.linalg.norm(selected,axis=1,keepdims=True)
            earned = tape.constant_value[:,j,c].numpy() + side*tape.right_value[:,j,c].numpy()
            if method=="sum":
                vector += earned[:,None]*selected
            elif method=="error":
                prediction = (selected*vector).sum(1)
                vector += (earned-prediction)[:,None]*selected
            elif method=="recent":
                vector = earned[:,None]*selected
            else:
                raise ValueError(f"Unknown reference rule {method}")
        if j in cap.CAPACITIES:
            choices = [decide(x[:,k,-1],y[:,k,-1],tape.motor_draw[:,k,-1].numpy())
                       for k in range(j+1)]
            correct = np.stack(choices,1) == tape.right_value[:,:j+1,-1].numpy()
            values[j] = {"A":float(correct[:,0].mean()),"all":float(correct.all(1).mean())}
            if j:
                values[j].update(B=float(correct[:,1:].mean()),B_latest=float(correct[:,-1].mean()),
                                 A_B=float((correct[:,0]&correct[:,-1]).mean()))
    return values


def run(source,label,births,seed,worlds,tasks=cap.TASKS,ks=cap.CAPACITIES):
    torch.set_num_threads(1)
    dest=DATA/"scorecards"/label
    dest.mkdir(parents=True,exist_ok=True)
    report=dict(seed=seed,worlds=worlds,births=births,capacities=list(ks),
                controller=source.family,arm=source.arm,step=source.step,source=str(source.path),
                tasks={},raw_sha256={},references={})
    for task in tasks:
        tape=cap.make_tape(task,seed,worlds,max(ks))
        vals=[]
        for birth in births:
            g,r=source.load(birth)
            z=(evaluate_readback(g,r,tape,ks,seed) if source.family=="readback"
               else cap.evaluate(g,r,tape,ks,snapshot_worlds=0))
            # Parameter distances and all sample outcomes are replayable from the seed.
            path=dest/f"{task}-{birth}.npz"
            keep=("side","logit","right_value","law","capacities","parameter_distance_from_post_A")
            np.savez_compressed(path,**{k:z[k] for k in keep})
            report["raw_sha256"][path.name]=sha256(path.read_bytes()).hexdigest()
            entry={}
            for i,k in enumerate(ks):
                entry[k]={}
                for b,branch in enumerate(cap.BRANCHES):
                    metrics,_=cap._metrics(z,b,i,k)
                    entry[k][branch]=metrics
            vals.append(entry)
        report["tasks"][task]=dict(zip(births,vals,strict=True))
        report["references"][task]={name:reference(tape,name) for name in ("sum","error","recent")}
    (dest/"summary.json").write_text(json.dumps(report,indent=1,allow_nan=False)+"\n",encoding="utf8")
    print(label,"completed",flush=True)
    return report


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","readback","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=96)
    p.add_argument("--seed",type=int,default=871831)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--tasks",nargs="+",choices=cap.TASKS,default=cap.TASKS)
    p.add_argument("--capacities",nargs="+",type=int,default=cap.CAPACITIES)
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.seed,a.worlds,a.tasks,a.capacities)
