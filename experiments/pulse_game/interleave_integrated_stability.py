"""Do independently calibrated stabilizers transfer to a single long-lived multi-task graph?"""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import interleave_stream as stream, interleave_slow_reference as slow,interleave_homeostasis as home
from . import interleave_age_train as age,local_trace_graph as net
from .interleave_interference import DATA,EVIDENCE,ROOT


BLOCKS=(0,1,2,4,8,16,32,64)
BRANCHES=(("native",0.,0.),("birth",.12,0.),("slow",.12,.02))
PROBES=("old_anchor","first_B","cold_old_and_B","fresh","old_after_fresh","old_and_fresh",
        "ctx_A","ctx_B","mark","ctx_both","ctx_both_mark")


def step(view,m,left,right,factor,fixed,u,gamma,beta):
    a=net.act(view,m.live,left,right,u[:,0])
    taste=a.side*factor+fixed
    e=net.observe(view,a,left,right,taste)
    w=net.commit(view,e,u[:,1:5])
    ref=net.Graph(m.slow_w,m.slow_b,m.ancestry)
    after,eligible=home.interpolate(view,e,w,ref,m.ancestry,"birth",gamma)
    sw=torch.where(eligible,m.slow_w.lerp(after.graph.weights,beta),m.slow_w) if beta else m.slow_w
    sb=m.slow_b.lerp(after.graph.bias,beta) if beta else m.slow_b
    return slow.Memory(after,sw,sb,eligible),a.side,taste


def context_questions(tape,birth,seed):
    n=len(tape.factor);row=torch.arange(n)
    rng=np.random.default_rng([seed,birth,933517])
    layouts=torch.from_numpy(np.repeat(age.signs(rng,(n//2,2)),2,0))
    draws=torch.from_numpy(np.repeat(rng.random((n//2,3),dtype=np.float32),2,0))
    masks=(tape.obj==-1,tape.obj==-2,(tape.kind==age.FEEDER)&(tape.factor!=0))
    stages=[m.to(torch.int64).argmax(1) for m in masks]
    visible=torch.stack([m.any(1) for m in masks],1)
    axis=(tape.right[row,stages[0]]-tape.left[row,stages[0]])*.5
    result=[]
    for i,at in enumerate(stages):
        left,right=tape.left[row,at],tape.right[row,at]
        midpoint=(right+left)*.5; vec=(right-left)*.5
        sign=layouts[:,0]*(vec*axis).sum(1).sign() if i<2 else layouts[:,1]
        result.append((midpoint-vec*sign[:,None],midpoint+vec*sign[:,None],
                       tape.factor[row,at]*sign,draws[:,i]))
    valid=visible.all(1)
    return result, valid,valid & (result[0][2]!=result[1][2]),torch.stack(stages,1)


def measure(rule,m,probes,ctx,gamma,beta):
    n=len(m.live.hidden)
    view=rule.expand(n);double=rule.expand(2*n)
    old=[]
    for l,r,goal,u in probes.menus:old.append(net.act(view,m.live,l,r,u).side==goal)
    split=m.subset(probes.rows)
    after,_,_=step(double,split,-probes.novel_study,probes.novel_study,probes.learn_factor,
                   torch.zeros_like(probes.learn_factor),probes.uniforms[:,:5],gamma,beta)
    final=net.wait(double,after.live,4)
    x=net.act(double,final,-probes.novel_return,probes.novel_return,probes.uniforms[:,6]).side==probes.return_factor
    l,r,goal,u=(v[probes.rows] for v in probes.menus[0])
    old_after=net.act(double,final,l,r,u).side==goal
    known=[net.act(view,m.live,l,r,u).side==goal for l,r,goal,u in ctx]
    def two(v):return v[:,None].expand(n,2)
    return torch.stack((two(old[0]),two(old[1]),two(old[0]&old[1]),x.reshape(n,2),old_after.reshape(n,2),
                        (x&old_after).reshape(n,2),*(two(v) for v in known),two(known[0]&known[1]),
                        two(known[0]&known[1]&known[2]))).numpy()


@torch.no_grad()
def evaluate(g,r,tape,birth,seed,blocks,branches=BRANCHES):
    n,t=tape.factor.shape
    view=r.expand(n)
    probe=stream.ColdProbes(tape,seed,birth)
    ctx,valid,conflict,firsts=context_questions(tape,birth,seed)
    raw=dict(ctx_valid=valid.numpy(),ctx_conflict=conflict.numpy(),context_firsts=firsts.numpy(),
             kind=tape.kind.numpy().astype(np.int8),law=tape.laws.numpy().astype(np.int8))
    for name,gamma,beta in branches:
        s=net.start(g,n);m=slow.Memory(s,s.graph.weights,s.graph.bias,s.graph.present)
        snaps=[];earned=[];stats=[]
        for t0 in range(t+1):
            if t0%4==0 and t0//4 in blocks:
                snaps.append(measure(r,m,probe,ctx,gamma,beta))
                stats.append(torch.stack(((s.graph.weights-m.live.graph.weights).abs().flatten(1).mean(1),
                                          (s.graph.weights-m.slow_w).abs().flatten(1).mean(1),
                                          m.ancestry.flatten(1).sum(1)),1).numpy())
            if t0==t:break
            if t0:m=replace(m,live=net.wait(view,m.live,tape.wait[:,t0]))
            m,_,reward=step(view,m,tape.left[:,t0],tape.right[:,t0],tape.factor[:,t0],
                            tape.fixed[:,t0],tape.draws[:,t0],gamma,beta)
            earned.append((reward>0).numpy())
        raw[name+"/probe"]=np.stack(snaps);raw[name+"/earned"]=np.stack(earned,1)
        raw[name+"/distance"]=np.stack(stats)
    return raw


def run(src,label,births,worlds,seed,blocks=BLOCKS):
    torch.set_num_threads(1)
    dest=DATA/"integrated-stability"/label;dest.mkdir(parents=True,exist_ok=True)
    result=dict(births=births,seed=seed,independent_worlds_per_birth=worlds,blocks=blocks,metric_names=PROBES,
                notes="Stable mixed habitats, same life up to 256 contacts, persistent odors and marks plus new distractors. "
                      "Use the previously selected .12 fast and .02 slow rates unchanged. Probe both contradictory context "
                      "laws and a feeder from the same state, with independent motor draws. Cold-probe correctness "
                      "for contexts is summarized only on worlds where all three cues have actually been presented. "
                      "The fresh X fork sees one real X feedback. No scored probe returns to the stream.",
                roots={},sha256={})
    for b in births:
        g,r=src.load(b);t=stream.tape_for_worlds(b,worlds,max(blocks),seed)
        raw=evaluate(g,r,t,b,seed,blocks);cells={}
        for branch,_,_ in BRANCHES:
            for j,k in enumerate(blocks):
                z=raw[branch+"/probe"][j]
                observed=raw["ctx_valid"]&(raw["context_firsts"]<4*k).all(1)
                conflict=observed&raw["ctx_conflict"]
                entry={name:float(z[i].mean()) for i,name in enumerate(PROBES[:6])}
                entry["conflict_lanes"]=int(conflict.sum())
                for i,name in enumerate(PROBES[6:],6):
                    entry[name]=float(z[i,conflict].mean()) if conflict.any() else None
                if k:
                    lo=4*blocks[j-1];hi=4*k;qs=np.arange(lo,hi)%4>=2
                    earned=raw[branch+"/earned"][:,lo:hi][:,qs]
                    entry["natural_quiz"]=float(earned.mean())
                    for id_,name in ((age.ODOR,"odor"),(age.CONTEXT,"context"),(age.FEEDER,"feeder")):
                        idx=raw["kind"][:,lo:hi][:,qs]==id_
                        entry[name+"_natural"]=float(earned[idx].mean()) if idx.any() else None
                cells[f"{branch}/{k}"]=entry
        f=dest/f"{b}.npz";np.savez_compressed(f,**raw)
        result["sha256"][f.name]=sha256(f.read_bytes()).hexdigest();result["roots"][b]=cells
        print(label,b,flush=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"integrated-stability-{label}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--label",required=True);p.add_argument("--family",choices=("joint","lifetime","objective","odor","feeder"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"data/local-trace-20261119/local")
    p.add_argument("--arm",default="active");p.add_argument("--step",type=int,default=8000)
    p.add_argument("--births",nargs="+",type=int,default=list(range(731,737)))
    p.add_argument("--worlds",type=int,default=64);p.add_argument("--seed",type=int,default=9750121)
    a=p.parse_args();run(stream.Source(a.family,a.run,a.arm,a.step),a.label,a.births,a.worlds,a.seed)
