"""Test whether comparing past prediction errors can improve reversal without blanket forgetting."""

import os
for _var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_var]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import json
from pathlib import Path
import time

import numpy as np

from . import ring_spikes,ring_world as world,volatile_ring,synaptic_arbitration as alg
from .motor_search import write,OUT


DIRECTORY=OUT/"ring-self-evaluation-v1"


def draw_tape(seeds,trials,regime,jitter=True):
    t=world.make_tape(seeds,trials)
    goals=t.goal.copy()
    affected=np.zeros_like(goals,bool); changed=np.zeros_like(goals,bool)
    times=[]
    for i,s in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([s,829651]))
        pair=rng.choice(4,2,replace=False)
        at=int(rng.integers(trials//3,trials*2//3)) if jitter else trials//2
        stop=min(trials,at+trials//4) if regime=="return" else trials
        affected[i]=np.isin(t.cue[i],pair)
        changed[i,at:stop]=affected[i,at:stop]
        surrogate=np.where(t.cue[i]==pair[0],pair[1],np.where(t.cue[i]==pair[1],pair[0],t.cue[i]))
        if regime!="stable":goals[i,changed[i]]=t.preferences[i,t.ctx[i,changed[i]],surrogate[changed[i]]]
        times.append(at)
    return replace(t,goal=goals),np.array(times),affected


def view(t,change,affected,result,gate):
    index=np.arange(t.trials)[None,:]
    post=index>=change[:,None]
    hits=result.first_correct
    masks={"stable_pre":~post & (index>=change[:,None]-32),"all_post":post,"unaffected":post&~affected,
           "all_changed":post&affected,"late_changed":affected & (index>=t.trials-32)}
    rank=np.zeros_like(index+np.zeros_like(t.ctx),dtype=int)
    for c in (0,1):
        for k in range(4):
            v=(t.ctx==c)&(t.cue==k)&affected&post
            rank+=v*np.cumsum(v,1)
    for lo,hi in ((1,1),(2,3),(4,7),(8,10000)):
        masks[f"exposure_{lo}_{hi}"]=(rank>=lo)&(rank<=hi)
    return {"overall":hits.mean(1).tolist(),"coverage":(result.first>=0).mean(1).tolist(),"gate":gate.tolist(),
            **{k:volatile_ring.rate_by_life(hits,mask) for k,mask in masks.items()}}


def one(job):
    cfg,rules,net,seeds,regime,noise,trials=job
    t,times,affected=draw_tape(seeds,trials,regime)
    p=alg.Arbiter(ring_spikes.Config(**cfg),alg.Gates(**rules),net,seeds)
    runs=world.rollout(volatile_ring.OutcomeChannel(p,seeds,trials,noise),t)
    return view(t,times,affected,runs,p.sum_gate/p.decision_count.clip(min=1))


def configs():
    rng=np.random.default_rng(717241)
    out=[alg.Gates(scope="slow"),alg.Gates(scope="fast"),alg.Gates(scope="fixed",prior=0.),alg.Gates()]
    for j in range(44):
        out.append(alg.Gates(fast_rate=float(rng.uniform(.45,1.3)),
            fast_ridge=float(np.exp(rng.uniform(np.log(.012),np.log(.45)))),
            gate_rate=float(rng.uniform(.18,1.)),gain=float(rng.uniform(1.4,7.5)),prior=float(rng.uniform(-2.9,.3)),
            exploration=float(rng.uniform(.03,.7)) if j%2 else 0.))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("select","confirm"),required=True);ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args();DIRECTORY.mkdir(exist_ok=True)
    cfg=json.loads((OUT/"ring-value-search-v1/frozen.json").read_text())["best"]["config"]
    seeds=lambda seed,n:np.random.default_rng(seed).integers(100_000_000,400_000_000,size=n).tolist()
    start=time.perf_counter()
    if args.stage=="select":
        values={};grid=configs();spec=[]
        for i,c in enumerate(grid):
            for regime in ("stable","reverse"):
                spec.append((i,regime,(cfg,asdict(c),781,seeds(128406,48),regime,.14,192)))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[(i,rg,pool.submit(one,item)) for i,rg,item in spec]
            for n,(i,rg,j) in enumerate(jobs,1):
                v=j.result()
                values.setdefault(i,{})[rg]={k:v[k]["pooled"] for k in ("all_changed","unaffected","stable_pre")}
                if n%8==0:print(n,len(jobs),"s",round(time.perf_counter()-start),flush=True)
        benefit=lambda i:(values[i]["stable"]["all_changed"]+values[i]["reverse"]["all_changed"])/2
        shortlist=sorted(range(len(grid)),key=benefit,reverse=True)[:5]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            val={}
            for i in shortlist:
                specs=[(cfg,asdict(grid[i]),g,seeds(3991637,64),regime,noise,192)
                       for g in (857,617) for regime,noise in (("reverse",.12),("stable",.22))]
                results=list(pool.map(one,specs))
                val[i]=float(np.mean([x["all_changed"]["pooled"] for x in results]))
        chosen=max(val,key=val.get)
        write(DIRECTORY/"frozen.json",dict(chosen=chosen,settings=asdict(grid[chosen]),base=cfg,grid=list(map(asdict,grid)),
              train=values,validation=val,criterion="equal weight on affected-key post-switch and stationary matched-key accuracy"))
        print("chosen",chosen,asdict(grid[chosen]),val,flush=True);return
    frozen=json.loads((DIRECTORY/"frozen.json").read_text())
    chosen=alg.Gates(**frozen["settings"])
    r={s:replace(chosen,scope=s) for s in ("slow","fast","fixed","global","local","inverted")}
    r["local_no_alarm"]=replace(chosen,scope="local",exploration=0.)
    root=DIRECTORY/"confirm";root.mkdir(exist_ok=True)
    roster=seeds(32613196,128)
    tasks=[];aliases={}
    for key,rule in r.items():
        exact=next((s for s in aliases if r[s]==rule),None)
        if exact is not None:aliases[key]=exact;continue
        aliases[key]=key
        for graph in (3,19,37):
            for regime in ("stable","reverse","return"):
                for noise in (0.,.16,.30):
                    name=f"{key}-{regime}-n{int(100*noise)}-g{graph}"
                    if not (root/(name+".json")).is_file():
                        tasks.append((name,(cfg,asdict(rule),graph,roster,regime,noise,224)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work=[(k,pool.submit(one,j)) for k,j in tasks]
        for i,(k,future) in enumerate(work,1):
            write(root/(k+".json"),future.result())
            if i%8==0:print(i,len(work),"s",round(time.perf_counter()-start),flush=True)
    write(DIRECTORY/"tested.json",dict(seeds=roster,graphs=[3,19,37],selected=asdict(chosen),aliases=aliases,
                                      tested={k:asdict(v) for k,v in r.items()}))


if __name__=="__main__":
    main()
