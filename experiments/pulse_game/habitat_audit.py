"""Independent physical recall tests for two private, sometimes conflicting, habitat rules."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time

import numpy as np
import torch
from . import habitat_world as game, habitat_meta as memory, habitat_deploy as neural, symmetry_grounded as sensory
from . import symmetry_policy as tuning, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def sources(root,name,route,actuator):
    model=root/(name+".pt")
    front=OUT/"symmetry-grounded-v1/public-delta-s7243-before.pt"
    p=sensory.load(front);r=memory.load(model)
    if actuator=="raw":return p,r,None
    # One standard actuator for all systems, optimized on separate single-rule
    # games before this comparison. Locally learned memory rule is untouched.
    path=OUT/"symmetry-policy-v1/public-learned-s7243--motor-e8719/frozen.json"
    setting=json.loads(path.read_text());parent=OUT/"symmetry-grounded-v1/public-learned-s7243.pt"
    body=tuning.materialize(sensory.load(parent),setting["best"]["genome"],"motor")["motion"]
    return p,r,body


def test(spec):
    name,setting,seeds,root,dest,controls=spec
    path=dest/(name+"-"+setting+".json")
    if path.exists():return
    torch.set_num_threads(1)
    variants={
      "discrete":((12,12),"discrete",True,"tuned"),"continuous":((12,12),"smooth",True,"tuned"),
      "nearby":((12,12),"nearby",True,"tuned"),"long_gap":((12,48),"smooth",True,"tuned"),
      "one_only":((12,12),"smooth",False,"tuned"),"native_motor":((12,12),"smooth",True,"raw")}
    pairs,appearance,teach,actuator=variants[setting]
    families=np.repeat(np.array([[1,1],[1,-1],[-1,1],[-1,-1]]),len(seeds),axis=0)
    expanded=seeds*4
    w=game.make(expanded,pairs=pairs,span=4,visits=3,noise=.04,jitter=.10,relations=families,
                appearance=appearance,teach_first=teach,novel_seed=11470877)
    n=len(seeds);tr=w.start_test
    for j in (1,2,3):
        sl=slice(n*j,n*(j+1))
        if not np.array_equal(w.menu.realized[:n,:tr],w.menu.realized[sl,:tr]):raise ValueError("A/B teaching signs differ by family")
        if not np.array_equal(w.menu.chemicals[:n,tr:],w.menu.chemicals[sl,tr:]):raise ValueError("Private query leaks in public odors")
    front,route,motion=sources(root,name,name.split("-")[0],actuator)
    tags=["true"]
    if controls:
        tags+=["erase_rule","erase_events","permute_write","other_context_read","erase_A_direction","erase_B_direction","no_write"]
    if controls and route.get("updater")=="rls":tags+=["same_step_length","diagonal_geometry"]
    data={}
    for tag in tags:
        policy=neural.Forager(front,route,expanded,start=tr,intervention=tag,appearance=w.tags,probes=w.probe,motor=motion)
        # Habitat identities are visible to the evaluators for causal ablations
        # and logging only. The intact controller sees their two sensory currents.
        policy.labels=w.menu.body.ctx
        result=game.rollout(policy,w,silent_probes=True)
        m=game.metrics(w,result)
        c=policy.at_test["codes"]
        geom=(c[:,0]*c[:,1]).sum(1)/(np.linalg.norm(c[:,0],axis=1)*np.linalg.norm(c[:,1],axis=1)).clip(min=1e-6)
        m.update(context_cosine=geom,old_gain_A=(policy.at_test["g"]*c[:,0]).sum(1),
                 old_gain_B=(policy.at_test["g"]*c[:,1]).sum(1))
        data[tag]={k:v.reshape(4,n).tolist() for k,v in m.items()}
    _write(path,dict(model=name,setting=setting,training=audit.fingerprint(root/(name+".pt")),seeds=seeds,
                     order=["++","+-","-+","--"],all_probe_feedback_hidden=True,
                     first_opportunities_per_root=8,config=variants[setting],rows=data))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--worlds",type=int,default=128)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--folder",default="habitat-meta-v1");parser.add_argument("--dest",default="habitat-meta-sealed")
    args=parser.parse_args();root=OUT/args.folder;dest=audit.OUT/args.dest;dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(210761399).integers(10_000_000,470_000_000,size=args.worlds).tolist()
    candidates=[f"{r}-s{s}" for s in (2791,4679,9341) for r in ("learned","single","public","random")]
    done=set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(done)<12:
            ready=[n for n in candidates if n not in done and (root/(n+"-history.json")).is_file()
                   and json.loads((root/(n+"-history.json")).read_text())[-1]["step"]==1400]
            jobs=[(n,setting,seeds,root,dest, n.startswith("learned") and setting=="continuous")
                  for n in ready for setting in ("discrete","continuous","nearby","long_gap","one_only","native_motor")]
            for i,f in enumerate(as_completed([pool.submit(test,j) for j in jobs]),1):
                f.result()
                if i%10==0:print(i,len(jobs),flush=True)
            done.update(ready)
            if len(done)<12:time.sleep(5)
    _write(dest/"complete.json",dict(candidates=sorted(done),seeds=seeds))


if __name__=="__main__":main()

