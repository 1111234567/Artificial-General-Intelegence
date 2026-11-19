"""Use an unchanged cohort of twelve independent learned bodies to compare address locality."""
import json
import time
import numpy as np
import torch
from . import adaptive_local as native,adaptive_local_gpu as quick,longrun_plastic as distributed
from . import independent_births as birth,unified_world as task,unified_audit as settings,unified_confirm as scoring
from . import unified_physical as physical,habitat_world as real,longrun_world as longgame,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


SCENES={
 "standard":("standard",0.),"absent_A":("long_competing_run",0.),
 "near_contexts":("similar_habitats",0.),"noisy_contexts":("standard",.09),
 "near_and_noisy":("similar_habitats",.09),"same_receptors":("standard",0.),
}


def evaluate(folder,alpha,seeds,longroots,dest):
    fp=next(folder.glob("*--full-*/frozen.json"));fit=json.loads(fp.read_text())
    seed=int(folder.name.split("-")[-1]);f,r,m=birth.resources(folder)
    genes=np.asarray(fit["best"]["genome"]);body=physical.phenotype(physical.params(f,r,m),genes,"full")[0]
    ff,rr,mm=physical.materialize(f,r,m,genes,"full")
    n=len(seeds);cases={};parity={}
    for name,(key,noise) in SCENES.items():
        fam=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],n,axis=0);all_seeds=seeds*4
        e=task.make(all_seeds,force=fam,nonce=9175033,**settings.SETTINGS[key])
        if name=="same_receptors":e.arena.tags[:]=np.sqrt(.5);e.arena.appearance[:]=np.sqrt(.5)
        native.add_sensor_noise(e.arena,all_seeds,noise)
        ctx=quick.Cohort(e,f,r,m);s1=ctx.play(body,alpha);original=distributed.Cohort(e,f,r,m)
        s0=original.adjustments(body,alpha,detail=True)[0]
        cases[name]={"local":scoring.data(e,s1,n),"distributed":scoring.data(e,s0,n)}
        cases[name]["allocated"]=dict(mean=float(np.mean((ctx.grid.counts>0).sum(1))),overflow=float(np.mean(ctx.grid.overflows>0)))
    # A full native NumPy reference with the same public pulses and real outcomes.
    ssub=seeds[:16];nn=len(ssub);family=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],nn,axis=0)
    e=task.make(ssub*4,force=family,nonce=9175033,**settings.SETTINGS["long_competing_run"])
    native.add_sensor_noise(e.arena,ssub*4,.06)
    co=quick.Cohort(e,f,r,m);ans=co.play(body,alpha)
    learner=native.LocalAdaptive(ff,{**rr,"release":alpha},ssub*4,motor=mm)
    actual=real.rollout(learner,e.arena,silent_probes=True)
    parity=dict(action=float(np.mean(ans.contact_side==actual.contact_side)),tick=float(np.mean(ans.contact_tick==actual.contact_tick)),
                reward=float(np.mean(ans.reward==actual.reward)),requests=actual.reward.size,
                same_prototype_count=bool(np.array_equal(learner.places.counts,co.grid.counts)))
    if parity["action"]<.999:raise ValueError(parity)
    long={}
    for i in range(0,len(longroots),24):
        ss=longroots[i:i+24]
        w=longgame.make(ss,blocks=192,mode="reverse_return",noise=.06,nonce=18537761)
        original=distributed.Cohort(w,f,r,m)
        c=quick.Cohort(w,f,r,m)
        for tag,ans in (("distributed",original.adjustments(body,alpha,detail=True)[0]),
                        ("local",c.play(body,alpha)),("local_no_reopening",c.play(body,[0.,0.,0.,0.,0.]))):
            count=longgame.count_regions(w,ans);store=long.setdefault(tag,{})
            for region,fields in count.items():
                for k,vals in fields.items():store.setdefault(region,{}).setdefault(k,[]).extend(vals)
    output=dest/(folder.name+".json")
    _write(output,dict(birth=seed,roots=seeds,long_roots=longroots,source=audit.fingerprint(fp),
                       parameters=dict(slots=6,radius_radians=.09,inherit=True,release=alpha),
                       cases=cases,long=long,parity=parity))
    print("completed",seed,parity,flush=True)


def main():
    torch.set_num_threads(1);root=OUT/"independent-lifelong-births-v1";reg=json.loads((root/"registered.json").read_text())
    dest=audit.OUT/"self-organized-twelve-births";dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(671903951).integers(8_000_000,470_000_000,256).tolist()
    longest=np.random.default_rng(23607543).integers(8_000_000,470_000_000,96).tolist()
    for b in reg["fixed_birth_seeds"]:
        folder=root/f"birth-{b}"
        if not (dest/(folder.name+".json")).exists():evaluate(folder,reg["reopening_rule"],seeds,longest,dest)
    _write(dest/"complete.json",dict(roots=seeds,births=reg["fixed_birth_seeds"],long_roots=longest))


if __name__=="__main__":main()

