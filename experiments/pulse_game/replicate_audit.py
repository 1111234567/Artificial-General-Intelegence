"""Evaluate the exported always-on update on every registered independent training birth."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from . import independent_births as births,unified_world as task,unified_confirm as score,unified_audit as config
from . import longrun_plastic as update,unified_physical as body,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def raw_roots(seed,n):
    # Same per-batch consumption as unified_train.raw, including the public
    # environment parameters chosen after each set of independent world seeds.
    rng=np.random.default_rng(seed);out=[]
    for i in range(0,n,80):
        out.extend(rng.integers(10_000_000,470_000_000,size=min(n-i,80)).tolist())
        initial=int(rng.choice([8,10,12,14,16]));end=24-initial
        rng.choice(np.arange(1,end//2),size=2,replace=False)
        rng.uniform(0.,.14);rng.uniform(.02,.13);rng.choice(["discrete","smooth","nearby"])
    return set(out)


def audit_seeds(folder,report,ex):
    meta=json.loads((folder.parent/"registered.json").read_text())
    rng=np.random.default_rng(report["recipe_seed"])
    *_,training,valid=rng.integers(1,400_000_000,6).tolist()
    check=next(folder.glob("*--full-*/checkpoint.json"));history=json.loads(check.read_text())["history"]
    used=set(s for row in history for s in row["roots"])
    selection=set()
    for i in range(4):selection.update(np.random.default_rng(36478591+i).integers(3_000_000,475_000_000,size=160).tolist())
    pools=dict(outer_train=raw_roots(training,meta["training"]["lives"]),outer_validation=raw_roots(valid,800),
               policy_train=used,policy_validation=selection)
    return {k:dict(unique=len(v),intersection=sorted(set(ex)&v)) for k,v in pools.items()}


def evaluate(folder,alpha):
    path=folder/"external.json";old=json.loads(path.read_text());seed=old["recipe_seed"]
    front,route,motion=births.resources(folder);policy=next(folder.glob("*--full-*/frozen.json"))
    z=json.loads(policy.read_text());gene=np.asarray(z["best"]["genome"])
    initial=body.params(front,route,motion);hard=body.phenotype(initial,gene,"full")[0]
    roots=old["roots"];fam=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],len(roots),axis=0)
    cases={}
    for key in ("standard","similar_habitats","long_competing_run"):
        e=task.make(roots*4,force=fam,nonce=9500131,**config.SETTINGS[key])
        c=update.Cohort(e,front,route,motion)
        actual=c.adjustments(hard,alpha,detail=True)[0]
        cases[key]=score.data(e,actual,len(roots))
    out=dict(seed=seed,short_worlds=roots,always_on=cases,
             training_seed_disjoint=audit_seeds(folder,old,roots+old["long_roots"]),
             sensory=audit.fingerprint(folder/"receptor.pt"),mapping=audit.fingerprint(next(folder.glob("rls-learned-*.pt"))),
             model=audit.fingerprint(policy))
    _write(folder/"always_on.json",out);print("audited birth",seed,flush=True)


def main():
    torch.set_num_threads(1);root=OUT/"independent-lifelong-births-v1"
    reg=json.loads((root/"registered.json").read_text());seeds=reg["fixed_birth_seeds"]
    done=set()
    while len(done)<len(seeds):
        for s in seeds:
            folder=root/f"birth-{s}"
            if s in done or not (folder/"external.json").exists():continue
            if not (folder/"always_on.json").exists():evaluate(folder,reg["reopening_rule"])
            done.add(s)
        if len(done)<len(seeds):time.sleep(6)
    _write(root/"whole_recipe_verified.json",dict(seeds=seeds,worlds_per_birth=256,long_worlds=96,
                                                  params=reg["reopening_rule"]))


if __name__=="__main__":main()

