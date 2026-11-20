"""Decode query-specific values from native pre-action activity on never-seen two-context histories."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_context_time as context
from . import interleave_stream as stream, local_trace_graph as net
from .interleave_interference import DATA,EVIDENCE,ROOT


STATES=("intact","graph_only","activity_only","cold")


@torch.no_grad()
def record(g,rule,birth,seed,nworld,timing):
    cfg=context.Config(worlds=nworld,seed=seed)
    tape=context.make(cfg,birth,1,domain=678902,forced_angle=0)
    n=len(tape["law"])
    row=torch.arange(n)
    view=rule.expand(n)
    cold=net.start(g,n)
    state=cold
    for t in range(2):
        if t:state=net.wait(view,state,tape["gap"][:,t],max_ticks=8)
        item=tape["order"][:,t];cue=tape["cue"][row,item];sign=tape["sign"][:,t]
        odor=tape["odor"][row,item]*sign[:,None];u=tape["u"][:,t]
        a=context.choose(view,state,odor,cue,u[:,0],timing)
        reward=a.side*sign*tape["law"][row,item]
        ex=net.observe(view,a,cue-odor,cue+odor,reward)
        state=net.commit(view,ex,u[:,1:]).state
    snap=net.wait(view,state,tape["gap"][:,2],max_ticks=8)
    states=(snap,net.State(snap.graph,cold.hidden,cold.sender_trace,cold.receiver_trace),
            net.State(cold.graph,snap.hidden,snap.sender_trace,snap.receiver_trace),cold)
    observations={}
    for name,s in zip(STATES,states,strict=True):
        by_item=[];motors=[]
        for i in range(2):
            h=[];m=[]
            for sign in (1,-1):
                a=context.choose(view,s,sign*tape["odor"][:,i],tape["cue"][:,i],tape["u"][:,2,0],timing)
                h.append(a.state.hidden);m.append(a.motor_logit)
            by_item.append(torch.stack(h,1));motors.append(torch.stack(m,1))
        observations[name]=torch.stack(by_item,1).reshape(nworld,4,2,2,-1).numpy()
        observations[name+"/native"]=torch.stack(motors,1).reshape(nworld,4,2,2).numpy()
    truth=tape["law"][:,:,None]*torch.tensor([1.,-1.])[None,None,:]
    observations["right_law"]=truth.reshape(nworld,4,2,2).numpy()
    return observations


def batches(g,r,birth,seed,count,timing):
    parts=[record(g,r,birth,seed+i,min(64,count-i),timing) for i in range(0,count,64)]
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def design(h,degree):
    if degree==1:return h
    i,j=np.triu_indices(h.shape[1])
    return np.column_stack((h,h[:,i]*h[:,j]))


def ridge(train_x,train_y,test_x,degree,ridge=.001):
    flat=train_x.reshape(-1,train_x.shape[-1]); test=test_x.reshape(-1,test_x.shape[-1])
    x=design(flat,degree).astype(float);z=design(test,degree).astype(float)
    mean=x.mean(0);scale=x.std(0).clip(1e-5)
    x=(x-mean)/scale;z=(z-mean)/scale
    y=train_y.ravel().astype(float)
    coef=np.linalg.solve(x.T@x/len(x)+ridge*np.eye(x.shape[1]),x.T@(y-y.mean())/len(x))
    return (z@coef+y.mean()).reshape(test_x.shape[:-1]),coef/scale


def metrics(score,labels):
    signed=score*labels
    correct=(signed>1e-9)+.5*(np.abs(signed)<=1e-9)
    # Symmetrize both physical side layouts and classify the canonical positive item.
    margin=.5*(score[:,:,:,0]-score[:,:,:,1])
    canon=labels[:,:,:,0]
    ok=margin*canon>1e-9
    tied=np.abs(margin)<=1e-9
    fractional=ok.astype(float)+.5*tied
    conflict=canon[:,:,0]!=canon[:,:,1]
    return dict(accuracy=float(correct.mean()),conflict_accuracy=float(correct[conflict].mean()),
                conflict_joint=float(fractional.prod(2)[conflict].mean()),
                agree_joint=float(fractional.prod(2)[~conflict].mean()),
                both_orientations_correct=float((signed>1e-9).all((2,3))[conflict].mean()))


def run(source,label,births,seed,cal_n,test_n,timing):
    torch.set_num_threads(1)
    dest=DATA/"context-probe"/label
    dest.mkdir(parents=True,exist_ok=True)
    out=dict(births=births,seed=seed,calibration_worlds=cal_n,test_worlds=test_n,timing=timing,roots={},sha256={},
             note="Two privately valued contexts use exactly the same odor. Study both once, then freeze a common "
                  "snapshot, test both possible menus in both physical orientations. Independent ridge decoders predict "
                  "the good physical side from the twenty post-sensory activations only. No decoder prediction enters "
                  "the actor. Greedy/symmetrized readout and model motor scores are distinct from stochastic actions.")
    for birth in births:
        g,r=source.load(birth)
        train=batches(g,r,birth,seed,cal_n,timing)
        test=batches(g,r,birth,seed+1000000,test_n,timing)
        y=train["right_law"];target=test["right_law"]
        raw={"labels":target};summary={}
        for name in STATES:
            summary[name]={"native":metrics(test[name+"/native"],target)}
            raw[name+"/native"]=test[name+"/native"]
            for degree in (1,2):
                pred,coef=ridge(train[name],y,test[name],degree)
                raw[name+f"/d{degree}"]=pred.astype(np.float32)
                raw[name+f"/coef{degree}"]=coef
                summary[name][f"d{degree}"]=metrics(pred,target)
        # Labels cannot predict first choices from a cold graph with balanced laws.
        if abs(summary["cold"]["native"]["accuracy"]-.5)>1e-8:
            raise AssertionError("Private label leaked into cold baseline")
        p=dest/f"{birth}.npz";np.savez_compressed(p,**raw)
        out["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
        out["roots"][birth]=summary
        print(label,birth,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"context-probe-{label}.json").write_text(json.dumps(out,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--cal",type=int,default=384);p.add_argument("--test",type=int,default=512)
    p.add_argument("--seed",type=int,default=1837531)
    p.add_argument("--timing",choices=("full","preview","blank"),default="full")
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.seed,a.cal,a.test,a.timing)
