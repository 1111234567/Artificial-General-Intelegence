"""Separate elapsed neutral time, persistent activation, and trace evolution during the gap."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as cap, interleave_stream as stream, local_trace_graph as net
from .interleave_interference import DATA,EVIDENCE,ROOT


MODES=("hold","decay","evolve","evolve_all","clear_on_next_input")
CUTS=(1,4,8)
GAPS=(0,4,16,64)


def neutral(rule,state,ticks,mode):
    if mode=="evolve" or mode=="evolve_all":
        return net._advance(rule,state,torch.zeros_like(state.hidden),ticks)
    after=net.wait(rule,state,ticks)
    if mode=="decay":
        keep=rule.trace_keep.sigmoid()**ticks
        return replace(after,sender_trace=state.sender_trace*keep[:,0,None,:],
                       receiver_trace=state.receiver_trace*keep[:,1,None,:])
    return after


def apply(rule,obs,draw,mode):
    if mode!="evolve_all":
        return net.commit(rule,obs,draw).state
    ready=neutral(rule,obs.state,net.PRE_WRITE_TICKS,mode)
    tagged=replace(obs.state,sender_trace=ready.sender_trace,receiver_trace=ready.receiver_trace)
    result=net.commit(rule,replace(obs,state=tagged),draw)
    return neutral(rule,replace(ready,graph=result.state.graph),net.POST_WRITE_TICKS,mode)


@torch.no_grad()
def replay(g,r,tape):
    rule=r.expand(tape.lanes)
    start=net.start(g,tape.lanes)
    anchor,_=cap._study(rule,start,tape,0,write=True)
    raw={}
    for gap in GAPS:
        for mode in MODES:
            s=anchor
            for i in range(1,max(CUTS)+1):
                s=neutral(rule,s,gap,mode)
                if mode=="clear_on_next_input":
                    s=replace(s,sender_trace=torch.zeros_like(s.sender_trace),receiver_trace=torch.zeros_like(s.receiver_trace))
                left,right=tape.left[:,i,0],tape.right[:,i,0]
                a=net.act(rule,s,left,right,tape.motor_draw[:,i,0])
                e=net.observe(rule,a,left,right,a.side*tape.right_value[:,i,0])
                s=apply(rule,e,tape.edit_draw[:,i,0],mode)
                if i not in CUTS:continue
                snap=neutral(rule,s,4,mode)
                actions=[cap._probe(rule,snap,tape,j).side.to(torch.int8) for j in range(i+1)]
                raw[f"{gap}/{mode}/{i}"]=torch.stack(actions,1).numpy()
    for k in CUTS:
        for mode in ("decay","evolve"):
            np.testing.assert_array_equal(raw[f"0/hold/{k}"],raw[f"0/{mode}/{k}"])
    return raw


def run(src,label,births,worlds,seed):
    torch.set_num_threads(1)
    dest=DATA/"clock"/label;dest.mkdir(parents=True,exist_ok=True)
    t=cap.make_tape("odor",seed,worlds,max(CUTS))
    goals=t.right_value[:,:,-1].numpy()
    result=dict(births=births,worlds=worlds,seed=seed,gaps=GAPS,capacities=CUTS,modes=MODES,roots={},sha256={},
                note="All branches begin from exactly the same post-A state. Between successive new objects, h always "
                     "takes the same number of ordinary neutral graph ticks. hold freezes trace values; decay uses "
                     "the existing learned keep factors without new evidence; evolve lets tags follow actual neutral "
                     "activations; evolve_all also evolves them for the 4+2 internal commit ticks. Clear only changes "
                     "traces at input onset. All modes freely commit normal parameters; no external optimizer.")
    for b in births:
        g,r=src.load(b)
        raw=replay(g,r,t)
        cells={}
        for name,v in raw.items():
            k=int(name.split("/")[-1]);good=v==goals[:,:k+1]
            cells[name]=dict(a=float(good[:,0].mean()),latest=float(good[:,-1].mean()),
                             both=float((good[:,0]&good[:,-1]).mean()),all=float(good.all(1).mean()),
                             bmean=float(good[:,1:].mean()))
        path=dest/f"{b}.npz";np.savez_compressed(path,**raw,goals=goals)
        result["sha256"][path.name]=sha256(path.read_bytes()).hexdigest();result["roots"][b]=cells
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"clock-{label}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")
    print(label,flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True);p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=64);p.add_argument("--seed",type=int,default=871831)
    a=p.parse_args()
    run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
