"""Evolve local update and spiking dynamics from game returns using CPU world batches."""

# Set BLAS pool limits before importing numpy in each spawned Windows worker.
import os
for _env in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_env]="1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from .motor_lif import MotorConfig, MotorLife
from . import world


# Bounds describe the search, not immutable neuron physiology. Rates and
# scales span a log interval while dimensionless motor voltages are linear.
PARAMS = (
    ("feature_leak",.15,.93,False), ("feature_scale",.15,1.45,True),
    ("recurrence",.025,.9,True), ("pre_decay",.15,.91,False),
    ("background_rate",.001,.15,True), ("motor_leak",.45,.96,False),
    ("tonic",.15,.78,False), ("self_excitation",.3,1.6,False),
    ("cross_inhibition",.35,1.9,False), ("motor_noise",.035,.52,True),
    ("motor_gain",.12,2.,True), ("eligibility_decay",.42,.98,False),
    ("learning_rate",.015,1.6,True),
)
OUT=Path(__file__).resolve().parents[2]/"data/pulse-game"


def config(vector):
    values={}
    for u,(name,lo,hi,log) in zip(vector,PARAMS,strict=True):
        values[name]=float(lo*(hi/lo)**u if log else lo+(hi-lo)*u)
    return MotorConfig(**values)


@lru_cache(maxsize=3)
def scenario(seed, lives):
    seq=np.random.default_rng(seed)
    seeds=tuple(map(int,seq.integers(1,470_000_000,size=lives)))
    return world.make_tape(seeds)


def evaluate(job):
    vector,seed,graphs,lives,drop=job
    cfg=config(vector)
    tape=scenario(seed,lives)
    scores=[]
    for net in graphs:
        policy=MotorLife(cfg,int(net),tape.seeds)
        if drop is not None:
            from .sensation import BriefCue
            policy=BriefCue(policy,drop)
        result=world.rollout(policy,tape)
        scores.append((result.correct[:,32:].mean(),result.correct[:,:8].mean(),result.contact_rate))
    return np.asarray(scores).mean(0).tolist()


def write(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,allow_nan=False,separators=(",",":")),encoding="utf-8")
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name",default="motor-direct-search-01")
    p.add_argument("--seed",type=int,default=11881)
    p.add_argument("--population",type=int,default=64)
    p.add_argument("--generations",type=int,default=32)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--lives",type=int,default=48)
    p.add_argument("--graphs",type=int,default=2)
    p.add_argument("--cue-ticks",type=int)
    args=p.parse_args()
    out=OUT/args.name;out.mkdir(parents=True,exist_ok=True)
    state_path=out/"checkpoint.json"
    rng=np.random.default_rng(args.seed)
    protocol={k:v for k,v in vars(args).items() if k not in ("workers","generations")}
    if state_path.exists():
        state=json.loads(state_path.read_text()); hist=state["history"]
        if state["protocol"]!=protocol:raise ValueError("Incompatible search checkpoint")
        rng.bit_generator.state=state["rng"];pop=np.array(state["next_population"])
        best=state["best"]
    else:
        hist=[];best=None
        pop=rng.random((args.population,len(PARAMS)))
        for k,(field,lo,hi,log) in enumerate(PARAMS):
            x=getattr(MotorConfig(),field)
            pop[0,k]=np.log(x/lo)/np.log(hi/lo) if log else (x-lo)/(hi-lo)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i in range(len(hist),args.generations):
            graphs=tuple(map(int,rng.integers(1,500_000,size=args.graphs)))
            seed=int(rng.integers(1,400_000_000))
            jobs=[(v,seed,graphs,args.lives,args.cue_ticks) for v in pop]
            stats=np.array(list(executor.map(evaluate,jobs,chunksize=2)))
            order=np.argsort(-stats[:,0],kind="stable")
            keep=max(3,len(pop)//8);elites=pop[order[:keep]]
            # Use 2 distinct development graph births and unseen worlds for selection.
            val=np.array(list(executor.map(evaluate,[(v,882419,(811,817),96,args.cue_ticks)
                                                     for v in elites[:3]])))
            idx=int(np.argmax(val[:,0])); vscore=float(val[idx,0])
            if best is None or vscore>best["validation"]:
                best=dict(validation=vscore,generation=i+1,vector=elites[idx].tolist(),
                          config=asdict(config(elites[idx])),metrics=val[idx].tolist())
            hist.append(dict(generation=i+1,seed=seed,graphs=list(graphs),best_train=stats[order[0]].tolist(),
                             mean=stats.mean(0).tolist(),validation=val.tolist()))
            candidates=[elites[0],np.array(best["vector"])]
            while len(candidates)<len(pop):
                if rng.random()<.12:
                    child=rng.random(len(PARAMS))
                else:
                    primary,donor=elites[rng.integers(len(elites),size=2)]
                    mix=rng.random(len(PARAMS))<.20
                    child=np.where(mix,donor,primary)
                    child=np.clip(child+rng.normal(0,rng.choice([.035,.09,.19]),len(PARAMS)),0,1)
                candidates.append(child)
            pop=np.array(candidates)
            write(state_path,dict(protocol=protocol,history=hist,best=best,
                                  rng=rng.bit_generator.state,next_population=pop.tolist()))
            print(i+1,"validation",round(vscore,4),"best",round(best["validation"],4),
                  "mean_late",round(float(stats[:,0].mean()),4),"elapsed_s",round(time.perf_counter()-start,1),flush=True)


if __name__=="__main__":
    main()
