"""Run the local contradiction / reset study on four-choice stationary and changing worlds."""

import os
for _var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_var]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import json
import time

import numpy as np

from . import arbitration_run as design, local_unlearning as plastic, ring_spikes,ring_world,volatile_ring
from .motor_search import OUT,write


FOLDER=OUT/"ring-local-unlearning-v1"


def one(job):
    cfg,rule,net,seeds,regime,noise,T=job
    tape,at,affected=design.draw_tape(seeds,T,regime)
    agent=plastic.SurpriseForager(ring_spikes.Config(**cfg),plastic.ResetRule(**rule),net,seeds)
    result=ring_world.rollout(volatile_ring.OutcomeChannel(agent,seeds,T,noise),tape)
    out=design.view(tape,at,affected,result,np.zeros(len(seeds)))
    out["resets"]=agent.injections.tolist();out["erased_l2_sum"]=agent.erased.tolist()
    counts=[0,0,0]
    for tick,life in agent.episodes:
        j=tick//ring_world.TICKS
        bucket=0 if j<at[life] else 1 if affected[life,j] else 2
        counts[bucket]+=1
    out["resets_pre_changed_unaffected"]=counts
    return out


def hypotheses():
    rng=np.random.default_rng(1837019)
    opts=[plastic.ResetRule(scope="unchanged"),plastic.ResetRule()]
    for _ in range(54):
        opts.append(plastic.ResetRule(criterion=float(rng.uniform(.55,3.20)),decay=float(rng.uniform(.63,.995)),
            confidence=float(rng.uniform(.10,.62)),erase=float(rng.uniform(.35,1.15)),
            reopen=float(np.exp(rng.uniform(np.log(.18),np.log(11.)))),
            reset_score=float(rng.uniform(-.12,.40)),trace_damping=float(rng.uniform(.01,.12))))
    return opts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("select","confirm"),required=True)
    p.add_argument("--workers",type=int,default=5);args=p.parse_args()
    FOLDER.mkdir(exist_ok=True);cfg=json.loads((OUT/"ring-value-search-v1/frozen.json").read_text())["best"]["config"]
    make_seeds=lambda seed,n:np.random.default_rng(seed).integers(10_000_000,450_000_000,size=n).tolist()
    clock=time.perf_counter()
    if args.stage=="select":
        opts=hypotheses();path=FOLDER/"search.json";s=json.loads(path.read_text()) if path.exists() else {}
        jobs=[(f"{i}-{regime}",(cfg,asdict(opt),713,make_seeds(936117,48),regime,.15,176))
              for i,opt in enumerate(opts) for regime in ("stable","reverse") if f"{i}-{regime}" not in s]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            work=[(k,pool.submit(one,v)) for k,v in jobs]
            for i,(k,f) in enumerate(work,1):
                v=f.result()
                s[k]={m:v[m]["pooled"] for m in ("all_changed","unaffected","exposure_2_3","exposure_4_7")}
                s[k]["resets"]=float(np.mean(v["resets"]))
                if i%8==0:write(path,s);print(i,len(work),"seconds",round(time.perf_counter()-clock),flush=True)
        write(path,s)
        def rank(r,n):
            return float(np.mean([r[f"{n}-stable"]["all_changed"],r[f"{n}-reverse"]["unaffected"],
                                  .5*(r[f"{n}-reverse"]["exposure_2_3"]+r[f"{n}-reverse"]["exposure_4_7"])]))
        shortlist=sorted(range(len(opts)),key=lambda n:rank(s,n),reverse=True)[:6]+[0]
        rerank={}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[(i,g,rg) for i in shortlist for g in (815,317) for rg in ("stable","reverse")]
            for (i,g,rg),v in zip(jobs,pool.map(one,[(cfg,asdict(opts[i]),g,make_seeds(731955,64),rg,.19,208)
                                                     for i,g,rg in jobs]),strict=True):
                r=rerank.setdefault(g,{})
                r[f"{i}-{rg}"]={k:v[k]["pooled"] for k in ("all_changed","unaffected","exposure_2_3","exposure_4_7")}
        winner=max(shortlist,key=lambda n:np.mean([rank(v,n) for v in rerank.values()]))
        write(FOLDER/"frozen.json",dict(chosen=winner,parameters=asdict(opts[winner]),base=cfg,
              rules=[asdict(v) for v in opts],rerank=rerank,shortlist=shortlist))
        print("winner",winner,opts[winner],flush=True);return
    f=json.loads((FOLDER/"frozen.json").read_text());winner=plastic.ResetRule(**f["parameters"])
    if winner.scope=="unchanged":
        winner=plastic.ResetRule(**f["rules"][next(i for i in f["shortlist"] if i!=0)])
    variants={"cue":winner,"no_reset":replace(winner,scope="unchanged"),
              "single_receptor":replace(winner,scope="single_receptor"),"permuted":replace(winner,scope="permuted"),
              "global":replace(winner,scope="whole"),"reopen_only":replace(winner,erase=0.),
              "erase_only":replace(winner,reopen=0.)}
    seeds=make_seeds(69390871,128)
    root=FOLDER/"confirm";root.mkdir(exist_ok=True)
    combos=[(name,g,rg,noise) for name in variants for g in (11,31,73) for rg in ("stable","reverse","return")
            for noise in (0.,.16,.30)]
    jobs=[(root/f"{n}-g{g}-{rg}-n{int(noise*100)}.json",(cfg,asdict(variants[n]),g,seeds,rg,noise,224))
          for n,g,rg,noise in combos]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work=[(path,pool.submit(one,j)) for path,j in jobs if not path.exists()]
        for i,(path,future) in enumerate(work,1):
            write(path,future.result())
            if i%8==0:print(i,len(work),"seconds",round(time.perf_counter()-clock),flush=True)
    write(FOLDER/"confirmation.json",{"seeds":seeds,"graphs":[11,31,73],"variants":{k:asdict(o) for k,o in variants.items()}})


if __name__=="__main__":
    main()
