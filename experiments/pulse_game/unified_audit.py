"""Evaluate one unchanged phenotype on prior sharing, contrary evidence and old-habitat return."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import numpy as np
import torch

from . import unified_world as game, habitat_world as habitat, habitat_deploy as neural, habitat_meta as meta
from . import habitat_audit, mixture_flash as flash, symmetry_audit as prior, symmetry_world as history
from . import symmetry_grounded as sensory, symmetry_policy as physical, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


SETTINGS={
 "standard":dict(initial=12,depths=(0,2,4,8,12),span=2,appearance="smooth",noise=.04,jitter=.08),
 "named_channels":dict(initial=12,depths=(0,2,4,8,12),span=2,appearance="discrete",noise=.04,jitter=.08),
 "similar_habitats":dict(initial=12,depths=(0,2,4,8,12),span=2,appearance="nearby",noise=.04,jitter=.08),
 "long_competing_run":dict(initial=12,depths=(0,4,12,24,48),span=4,appearance="smooth",noise=.09,jitter=.12),
}


def resources(folder,name):
    receptor=sensory.load(OUT/"symmetry-grounded-v1/public-delta-s7243-before.pt")
    route=meta.load(folder/(name+".pt"))
    ref=json.loads((OUT/"symmetry-policy-v1/public-learned-s7243--motor-e8719/frozen.json").read_text())
    body=physical.materialize(sensory.load(OUT/"symmetry-grounded-v1/public-learned-s7243.pt"),ref["best"]["genome"],"motor")
    return receptor,route,body["motion"]


def one(job):
    name,which,seeds,folder,dest=job
    path=dest/(name+"-"+which+".json")
    if path.exists():return
    torch.set_num_threads(1)
    n=len(seeds);source=folder/(name+".pt")
    front,route,motion=resources(folder,name)
    if which=="shared_game":
        w=prior.joint(seeds,pairs=12,probes=2,span=4,noise=.04,jitter=.08,matched=True)
        expanded=seeds+seeds
        actor=neural.Forager(front,route,expanded,start=w.exposed,motor=motion)
        actor.labels=w.menu.body.ctx
        res=flash.rollout(actor,w.menu)
        m=history.metrics(w,res)
        rows={"true":{k:v.reshape(2,n).tolist() for k,v in m.items()}}
        _write(path,dict(model=name,checkpoint=audit.fingerprint(source),case=which,seeds=seeds,rows=rows));return
    families=np.repeat(np.array([[1,1],[1,-1],[-1,1],[-1,-1]]),n,axis=0)
    expanded=seeds*4
    if which=="conflict_game":
        w=habitat.make(expanded,pairs=(12,12),span=4,noise=.04,jitter=.08,relations=families,appearance="smooth",
                       novel_seed=71625391)
        actor=neural.Forager(front,route,expanded,start=w.start_test,motor=motion);actor.labels=w.menu.body.ctx
        res=habitat.rollout(actor,w,silent_probes=True)
        rows={"true":{k:v.reshape(4,n).tolist() for k,v in habitat.metrics(w,res).items()}}
        _write(path,dict(model=name,checkpoint=audit.fingerprint(source),case=which,seeds=seeds,rows=rows));return
    e=game.make(expanded,force=families,nonce=71625391,**SETTINGS[which]);w=e.arena
    tags=["true"]
    if which=="standard" and name.startswith(("rls-learned","delta-learned")):
        tags+=["no_write","other_context_read","permute_write"]
        if route["updater"]=="rls":tags+=["same_step_length","diagonal_geometry"]
    evidence=w.menu.realized.copy()
    evidence[w.probe]=0
    for k in (1,2,3):
        if not np.array_equal(evidence[:n],evidence[k*n:(k+1)*n]):raise RuntimeError("Time series of signed bouts differs")
    rows={}
    for tag in tags:
        a=neural.Forager(front,route,expanded,start=w.start_test,intervention=tag,appearance=w.tags,probes=w.probe,motor=motion)
        a.labels=w.menu.body.ctx
        r=habitat.rollout(a,w,silent_probes=True)
        if (r.observed_reward[w.probe]!=0).any():raise ValueError("A held-out choice gave the animal a score")
        m=game.measures(e,r)
        early=m["B_initial_side"].reshape(4,n)
        if not np.array_equal(early[0],early[1]) or not np.array_equal(early[2],early[3]):
            raise ValueError("Unobserved B rule altered the animal before its first B action")
        rows[tag]={k:v.reshape(4,n).tolist() for k,v in m.items()}
    _write(path,dict(model=name,checkpoint=audit.fingerprint(source),case=which,seeds=seeds,rows=rows,
          protocol=SETTINGS[which],order=["++","+-","-+","--"],B_initial_sides_identical_for_fixed_A=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worlds",type=int,default=128)
    p.add_argument("--workers",type=int,default=4);args=p.parse_args()
    folder=OUT/"unified-transfer-v1";dest=audit.OUT/"unified-transfer-sealed";dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(74912651).integers(10_000_000,470_000_000,args.worlds).tolist()
    names=[f"{u}-{r}-s{s}" for s in (1373,8539,6547) for u in ("rls","delta")
            for r in ("learned","single","public","shared_public","random")]
    done=set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(done)<len(names):
            ready=[n for n in names if n not in done and (folder/(n+"-history.json")).exists() and
                   json.loads((folder/(n+"-history.json")).read_text())[-1]["step"]==2200]
            jobs=[(n,cond,seeds,folder,dest) for n in ready for cond in (*SETTINGS,"shared_game","conflict_game")]
            for i,f in enumerate(as_completed([pool.submit(one,j) for j in jobs]),1):
                f.result()
                if i%12==0:print(i,len(jobs),flush=True)
            done.update(ready)
            if ready:print("audited",*ready,flush=True)
            if len(done)<len(names):time.sleep(8)
    _write(dest/"complete.json",dict(models=names,lives=len(seeds)))


if __name__=="__main__":main()
