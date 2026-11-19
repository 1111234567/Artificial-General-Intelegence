"""Candidate search and frozen evaluations for the four-food LIF forager."""

import os
for _var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_var]="1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from . import ring_spikes,ring_world
from .motor_search import write, OUT


@lru_cache(maxsize=3)
def tape_for(lives,trials,seed):
    seeds=np.random.default_rng(seed).integers(30_000_000,450_000_000,size=lives).tolist()
    return ring_world.make_tape(seeds,trials)


def measures(v):
    first=v.first_correct
    return {"early":float(first[:,:8].mean()),"late":float(first[:,-32:].mean()),
            "late_solved":float(v.solved[:,-32:].mean()),"contact":float((v.first>=0).mean()),
            "all":float(first.mean()),"mean_bites":float(v.bites.mean()),
            "latency_or_timeout":float(np.where(v.first_tick>=0,v.first_tick,ring_world.TICKS).mean())}


def evaluate(job):
    cod,graphs,lives,trials,seed=job
    t=tape_for(lives,trials,seed)
    cfg=ring_spikes.Config(**cod)
    values=[]
    for net in graphs:
        p=ring_spikes.Forager(cfg,net,t.seeds)
        values.append(measures(ring_world.rollout(p,t)))
    return {k:float(np.mean([d[k] for d in values])) for k in values[0]}


def candidates(count):
    rng=np.random.default_rng(811940)
    configs=[]
    # Within each base draw, evaluate matched delta/RLS and relative-value/no-inhibition.
    for i in range(count):
        base=ring_spikes.Config(bite_bias=float(rng.uniform(.55,1.32)),
              motor_gain=float(np.exp(rng.uniform(np.log(.45),np.log(3.6)))),
              exploration=float(rng.uniform(.001,.12)),learning_rate=float(rng.uniform(.55,1.15)),
              regularization=float(np.exp(rng.uniform(np.log(.015),np.log(.7)))),
              history=float(rng.uniform(.35,.90)),latent=int(rng.choice([12,24,40])),
              hunger_rate=float(rng.uniform(0.,.042)) if i%2 else 0.)
        level=float(rng.uniform(.45,1.22))
        for method in ("delta","rls"):
            for comp in (0.,level):
                cfg=replace(base,method=method,competition=comp)
                configs.append(asdict(cfg))
    return configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws",type=int,default=36)
    parser.add_argument("--workers",type=int,default=7)
    args=parser.parse_args()
    folder=OUT/"ring-value-search-v1";folder.mkdir(parents=True,exist_ok=True)
    configs=candidates(args.draws)
    path=folder/"discovery.json"
    if path.exists():
        rows=json.loads(path.read_text())
        if [x["config"] for x in rows]!=configs[:len(rows)]:raise ValueError("Search resume settings differ")
    else: rows=[]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[(c,(132,),36,80,111371) for c in configs[len(rows):]]
        for config,score in zip(configs[len(rows):],pool.map(evaluate,jobs,chunksize=1),strict=True):
            rows.append({"config":config,"score":score})
            if len(rows)%8==0:
                write(path,rows)
                print(len(rows),"/",len(configs),"best",round(max(x["score"]["late"] for x in rows),4),
                      "sec",round(time.perf_counter()-start,1),flush=True)
        write(path,rows)
        finalists=[]
        for method in ("delta","rls"):
            for competition in (False,True):
                group=[row for row in rows if row["config"]["method"]==method and bool(row["config"]["competition"])==competition]
                finalists.extend(sorted(group,key=lambda r:r["score"]["late"],reverse=True)[:3])
        finalist_scores=list(pool.map(evaluate,[(r["config"],(18,55,71),64,96,389113) for r in finalists]))
    for row,m in zip(finalists,finalist_scores,strict=True):row["validation"]=m
    best=max(finalists,key=lambda r:r["validation"]["late"])
    import hashlib
    hashfile=lambda filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    write(folder/"frozen.json",{"best":best,"finalists":finalists,"discovery_world_rng":111371,
          "validation_world_rng":389113,"discovery_graph":132,"validation_graphs":[18,55,71],
          "policy_sha256":hashfile(ring_spikes.__file__),"world_sha256":hashfile(ring_world.__file__)})
    print("frozen",best,flush=True)


if __name__=="__main__":
    main()
