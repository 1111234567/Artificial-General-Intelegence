"""New physical roots for the two-by-two context-geometry and absence-duration study."""
import json
import time
import numpy as np
import torch
from . import unified_physical as physics,unified_audit as origin,unified_world as game,unified_confirm as sealed
from . import habitat_meta as route,habitat_world as body,habitat_deploy as native,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


CONDITIONS={**origin.SETTINGS,
 "very_long_B":dict(initial=12,depths=(0,4,16,32,96),span=4,appearance="smooth",noise=.07,jitter=.12),
 "nearby_long_B":dict(initial=12,depths=(0,4,12,24,48),span=4,appearance="nearby",noise=.07,jitter=.12)}


def check(path,seeds,dest):
    root=path.parent.parent;frozen=json.loads(path.read_text());s=frozen["settings"]
    kind,curr,seed=s["model"].split("-");seed=int(seed[1:])
    weight=root/f"bottleneck-{seed}.pt" if kind=="bottleneck" else OUT/f"unified-transfer-v1/rls-learned-s{seed}.pt"
    if audit.fingerprint(weight)!=s["model_sha"]:raise ValueError("Frozen input changed")
    front,_,motion=origin.resources(OUT/"unified-transfer-v1",f"rls-learned-s{seed}")
    r=route.load(weight);v=np.asarray(frozen["best"]["genome"])
    n=len(seeds);family=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],n,axis=0)
    rows={};compat={}
    for tag,spec in CONDITIONS.items():
        e=game.make(seeds*4,force=family,nonce=47770931,**spec)
        co=physics.Cohort(e,front,r,motion)
        q,_=co.run(v,"full",detail=True)[0]
        rows[tag]=sealed.data(e,q,n)
        if tag=="long_competing_run":
            sub=seeds[:16];s_family=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],len(sub),axis=0)
            ee=game.make(sub*4,force=s_family,nonce=47770931,**spec)
            actual,_=physics.Cohort(ee,front,r,motion).run(v,"full",detail=True)[0]
            ff,rr,mm=physics.materialize(front,r,motion,v,"full")
            actor=native.Forager(ff,rr,sub*4,start=-1,motor=mm)
            truth=body.rollout(actor,ee.arena,silent_probes=True)
            compat=dict(tick=float(np.mean(truth.contact_tick==actual.contact_tick)),
                       decision=float(np.mean(truth.contact_side==actual.contact_side)),opportunities=truth.reward.size)
    log=json.loads((path.parent/"checkpoint.json").read_text())["history"]
    events=sum((2*(x["environment"]["initial"]+x["environment"]["depths"][-1]+len(x["environment"]["depths"])+1))*len(x["roots"]) for x in log)
    standard_equivalent=sum(x["environment"].get("short_equivalent_requests",0) if "short_equivalent_requests" in x["environment"] else
        (2*(x["environment"]["initial"]+x["environment"]["depths"][-1]+len(x["environment"]["depths"])+1))*len(x["roots"]) for x in log)
    _write(dest/(path.parent.name+".json"),dict(settings=s,kind=kind,curriculum=curr,seed=seed,roots=seeds,rows=rows,
           actual_per_candidate_events_summed_across_generations=events,short_equivalent=standard_equivalent,
           parity=compat,source=audit.fingerprint(path)))
    print(path.parent.name,compat,flush=True)


def main():
    torch.set_num_threads(1);root=OUT/"old-habitat-repair-v1";dest=audit.OUT/"old-habitat-repair-sealed";dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(98673137).integers(7_000_000,475_000_000,384).tolist()
    while True:
        files=sorted(root.glob("*/frozen.json"))
        for file in files:
            if not (dest/(file.parent.name+".json")).is_file():check(file,seeds,dest)
        if len(files)==12:break
        time.sleep(8)
    _write(dest/"complete.json",dict(runs=len(files),roots=seeds))


if __name__=="__main__":main()
