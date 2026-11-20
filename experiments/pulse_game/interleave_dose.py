"""Measure the tradeoff between retaining A and learning B as intervening write magnitude varies."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_stream as stream, local_trace_graph as local
from .interleave_interference import DATA,EVIDENCE,ROOT


GAINS=(0.,.125,.25,.5,.75,1.,1.5,2.)
CUTS=(1,2,4,8,16)


def bounded(old,proposed,limit):
    return proposed.clamp(min=old.clamp_max(-limit),max=old.clamp_min(limit))


def scaled_numeric(coeff,observation,tape,obj,gain):
    u=tape.edit_draw[:,obj,0]
    native=local.commit(coeff,observation,u,intervention="weights_and_bias")
    if gain==1:
        return native.state
    before=observation.state.graph
    r=native.receipt
    if gain==0:
        after=before
    else:
        w=bounded(before.weights,before.weights+gain*r.weight_delta,local.WEIGHT_LIMIT)
        w=torch.where(before.present,local._nonzero(w),0.)
        b=bounded(before.bias,before.bias+gain*r.bias_delta,local.BIAS_LIMIT)
        after=replace(before,weights=w,bias=b)
    ready=local.wait(coeff,observation.state,local.PRE_WRITE_TICKS)
    return local.wait(coeff,replace(ready,graph=after),local.POST_WRITE_TICKS)


@torch.no_grad()
def evaluate(graph,rule,tape):
    coeff=rule.expand(tape.lanes)
    cold=local.start(graph,tape.lanes)
    anchor,_=cap._study(coeff,cold,tape,0,write=True)
    out={}
    for gain in GAINS:
        state=anchor
        for obj in range(1,max(CUTS)+1):
            left=tape.left[:,obj,0]
            right=tape.right[:,obj,0]
            a=local.act(coeff,state,left,right,tape.motor_draw[:,obj,0])
            r=a.side*tape.right_value[:,obj,0]
            o=local.observe(coeff,a,left,right,r)
            state=scaled_numeric(coeff,o,tape,obj,gain)
            if obj in CUTS:
                snapshot=local.wait(coeff,state,4)
                actions=torch.stack([cap._probe(coeff,snapshot,tape,i).side for i in range(obj+1)],1)
                out[f"{gain}/{obj}"]=actions.numpy().astype(np.int8)
    return out


def run(source,label,births,seed,worlds):
    torch.set_num_threads(1)
    tape=cap.make_tape("odor",seed,worlds,max(CUTS))
    targets=tape.right_value[:,:,-1].numpy()
    dest=DATA/"dose"/label
    dest.mkdir(parents=True,exist_ok=True)
    summary=dict(seed=seed,worlds=worlds,births=births,gains=GAINS,cuts=CUTS,
                 notes="A always receives full native write. All subsequent objects use the same numeric gain, "
                 "holding B-stage topology fixed. Physical clipping is retained. Independent read-only probes.",
                 results={},sha256={})
    for birth in births:
        g,r=source.load(birth)
        if r is None:raise ValueError("Numeric local-rule dose requires a local graph")
        raw=evaluate(g,r,tape)
        metrics={}
        for k,val in raw.items():
            correct=val==targets[:,:val.shape[1]]
            metrics[k]={"A":float(correct[:,0].mean()),"latest":float(correct[:,-1].mean()),
                        "both":float((correct[:,0]&correct[:,-1]).mean()),
                        "all":float(correct.all(1).mean()),"B_avg":float(correct[:,1:].mean())}
        p=dest/f"{birth}.npz"
        np.savez_compressed(p,**raw,target=targets)
        summary["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
        summary["results"][birth]=metrics
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"dose-{label}.json").write_text(json.dumps(summary,indent=1)+"\n",encoding="utf8")
    print(label,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True)
    p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active")
    p.add_argument("--step",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=128)
    p.add_argument("--seed",type=int,default=871831)
    p.add_argument("--births",type=int,nargs="+",default=list(range(731,737)))
    args=p.parse_args()
    run(stream.Source(args.family,args.run,args.arm,args.step),args.label,args.births,args.seed,args.worlds)
