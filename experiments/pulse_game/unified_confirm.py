"""Freeze one policy and score transfer, revision and old-habitat recall on the same untouched life."""
import argparse
import json
import time
import numpy as np
import torch
from . import unified_world as game,unified_audit as baseline,unified_physical as physics
from . import habitat_deploy as native,habitat_world as habitat,symmetry_audit as old,symmetry_world as first_game
from . import mixture_flash as flash,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


EXTERNAL={
 **baseline.SETTINGS,
 "noisy":dict(initial=16,depths=(0,2,4,8,16),span=2,appearance="smooth",noise=.20,jitter=.15),
 "identical_context_sensation":baseline.SETTINGS["standard"],
}


def make(seeds,case):
    n=len(seeds);family=np.repeat(np.array([[1,1],[1,-1],[-1,1],[-1,-1]]),n,axis=0)
    e=game.make(seeds*4,force=family,nonce=19176531,**EXTERNAL[case])
    if case=="identical_context_sensation":
        e.arena.appearance[:]=np.sqrt(.5);e.arena.tags[:]=np.sqrt(.5)
    return e


def data(e,out,n):
    raw=game.measures(e,out)
    a=raw["B_initial_side"].reshape(4,n)
    assert np.array_equal(a[0],a[1]) and np.array_equal(a[2],a[3])
    return {k:v.reshape(4,n).tolist() for k,v in raw.items()}


def run(file,seeds,dest):
    frozen=json.loads(file.read_text());s=frozen["settings"]
    model=s["model"];mode=s["mode"];n=len(seeds)
    source=OUT/"unified-transfer-v1"/(model+".pt")
    if s["model_sha"]!=audit.fingerprint(source):raise ValueError("A frozen parent changed")
    f,r,m=baseline.resources(OUT/"unified-transfer-v1",model)
    genes=np.asarray(frozen["best"]["genome"])
    variants={"selected":(genes,0),"parent":(np.zeros_like(genes),0)}
    result={};parity={}
    for case in EXTERNAL:
        ep=make(seeds,case);co=physics.Cohort(ep,f,r,m)
        states=dict(variants)
        if case in ("standard","similar_habitats","long_competing_run"):
            states.update(no_write=(genes,1),misaddress=(genes,2),equal_step_length=(genes,3))
        result[case]={}
        for key,(v,flag) in states.items():
            real,_=co.run(v,mode,detail=True,variant=flag)[0]
            result[case][key]=data(ep,real,n)
        if case in ("standard","long_competing_run"):
            sample=seeds[:32];w=make(sample,case);cc=physics.Cohort(w,f,r,m)
            real,_=cc.run(genes,mode,detail=True)[0];ff,rr,mm=physics.materialize(f,r,m,genes,mode)
            actor=native.Forager(ff,rr,sample*4,start=w.arena.start_test,motor=mm);actor.labels=w.arena.menu.body.ctx
            expected=habitat.rollout(actor,w.arena,silent_probes=True)
            parity[case]=dict(actions=float(np.mean(real.contact_side==expected.contact_side)),
                      ticks=float(np.mean(real.contact_tick==expected.contact_tick)),rewards=float(np.mean(real.reward==expected.reward)),
                      events=expected.reward.size)
    # Same genotype, circuit and read/write rules. Only the public game changes.
    ssub=seeds
    for label,v in (("selected",genes),("parent",np.zeros_like(genes))):
        ff,rr,mm=physics.materialize(f,r,m,v,mode)
        w=old.joint(ssub,pairs=12,probes=2,span=4,noise=.04,jitter=.08,matched=True)
        brain=native.Forager(ff,rr,ssub+ssub,start=w.exposed,motor=mm);brain.labels=w.menu.body.ctx
        out=flash.rollout(brain,w.menu)
        scored=first_game.metrics(w,out);hi,lo=np.split(scored["first"],2)
        result.setdefault("shared_transfer_game",{})[label]={"first":np.stack((hi,lo)).tolist(),
                "both":np.stack((hi*lo,hi*lo)).tolist()}
        family=np.repeat(np.array([[1,1],[1,-1],[-1,1],[-1,-1]]),len(ssub),axis=0)
        rew=habitat.make(ssub*4,pairs=(12,12),visits=3,span=4,noise=.04,jitter=.08,relations=family,
                         appearance="smooth",novel_seed=8193371)
        brain=native.Forager(ff,rr,ssub*4,start=rew.start_test,motor=mm);brain.labels=rew.menu.body.ctx
        rrout=habitat.rollout(brain,rew,silent_probes=True)
        result.setdefault("double_habitat_game",{})[label]={k:y.reshape(4,-1).tolist() for k,y in habitat.metrics(rew,rrout).items()}
    _write(dest/(file.parent.name+".json"),dict(settings=s,seeds=seeds,gen=frozen["best"]["generation"],
           order=["++","+-","-+","--"],cases=result,parity=parity,source=audit.fingerprint(file)))
    print(file.parent.name,parity,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worlds",type=int,default=512)
    args=p.parse_args();torch.set_num_threads(1)
    root=OUT/"unified-physical-evolution-v1";dest=audit.OUT/"unified-physical-sealed";dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(614988513).integers(5_000_000,480_000_000,args.worlds).tolist()
    if len(set(seeds))!=len(seeds):raise RuntimeError("Repeated root")
    while True:
        files=sorted(root.glob("*/frozen.json"))
        for f in files:
            if not (dest/(f.parent.name+".json")).exists():run(f,seeds,dest)
        if len(files)==6:break
        time.sleep(8)
    _write(dest/"complete.json",dict(roots=seeds,candidates=6))


if __name__=="__main__":main()
