"""Can the same frozen circuit survive long teaching, hidden change and return to the old habitat?"""
import argparse
import json
import time
import numpy as np
import torch

from . import longrun_world as world,longrun_plastic as learning,unified_physical as policies,unified_audit as base
from . import habitat_world as bodies,habitat_deploy as native,symmetry_physics as gpu,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def origin(root):
    fit=json.loads((root/"frozen.json").read_text())
    s=fit["settings"];m=s["model"];mode=s["mode"]
    f,r,motion=base.resources(OUT/"unified-transfer-v1",m)
    v=np.asarray(fit["best"]["genome"])
    bod=policies.phenotype(policies.params(f,r,motion),v,mode)[0]
    ff,rr,mm=policies.materialize(f,r,motion,v,mode)
    return f,r,motion,bod,(ff,rr,mm),fit


def discover(root,folder):
    f,r,motion,body,_,fit=origin(root)
    proposals=learning.choices()
    scores=[]
    seeds={}
    cases=[(36,"stable",.05),(36,"reverse_return",.05),(64,"stable",.16),(64,"reverse_return",.16),
           (96,"reverse_return",.07),(96,"stable",.07)]
    for j,(blocks,mode,noise) in enumerate(cases):
        ss=np.random.default_rng(38490651+j//2).integers(8_000_000,470_000_000,64).tolist()
        seeds[j]=ss
        w=world.make(ss,blocks=blocks,mode=mode,noise=noise,order="iid" if j<2 else "clustered",nonce=31855081)
        cohort=learning.Cohort(w,f,r,motion)
        v=cohort.adjustments(body,proposals,batch=32)
        native_base=cohort.run(fit["best"]["genome"],fit["settings"]["mode"])
        if np.max(abs(v[0]-native_base[0]))>.00001:raise ValueError("No-release kernel differs")
        scores.append(v[:,4])
    scores=np.asarray(scores);wins={}
    for name,code in learning.METHOD.items():
        ix=np.flatnonzero(proposals[:,0]==code);j=int(ix[np.argmax(scores[:,ix].mean(0))])
        wins[name]=dict(parameters=proposals[j].tolist(),validation=scores[:,j].tolist(),index=j)
    overall=int(scores.mean(0).argmax())
    frozen=dict(parent=root.name,source=audit.fingerprint(root/"frozen.json"),settings=wins,candidates=proposals.tolist(),
                scores=scores.tolist(),cases=cases,worlds=seeds,global_validation_winner=overall,
                objective="mean actual positive first physical choices on all pre/post probes")
    _write(folder/(root.name+".json"),frozen)
    return frozen


def parity(root,config,folder):
    f,r,motion,body,mat,fit=origin(root);ss=list(range(612733,612749))
    w=world.make(ss,blocks=24,mode="reverse_return",nonce=991843)
    co=learning.Cohort(w,f,r,motion);table=[]
    for name in ("rigid","global","local","uncertainty_direction","surprise_uncertainty"):
        p=np.array([0.,0.,0.,0.,0.] if name=="rigid" else config["settings"][name]["parameters"],np.float32)
        out=co.adjustments(body,p,detail=True)[0]
        ff,rr,mm=mat
        brain=native.Forager(ff,{**rr,"release":p},ss,start=0,motor=mm)
        actual=bodies.rollout(brain,w.arena,silent_probes=True)
        eig=np.linalg.eigvalsh(.5*(brain.cov+brain.cov.transpose(0,2,1)))
        table.append(dict(mode=name,action=float(np.mean(actual.contact_side==out.contact_side)),
                         tick=float(np.mean(actual.contact_tick==out.contact_tick)),reward=float(np.mean(actual.reward==out.reward)),
                         min_eigen=float(eig.min()),max_eigen=float(eig.max()),max_abs_weights=float(np.abs(brain.g).max())))
    _write(folder/(root.name+"-parity.json"),table)
    if min(v["action"] for v in table)<.998:raise ValueError(table)
    return table


CONDITIONS={
  "stable":(192,"stable",.05,"clustered","smooth"),
  "reverse_return":(192,"reverse_return",.05,"clustered","smooth"),
  "noise_stable":(192,"stable",.19,"clustered","smooth"),
  "noise_reverse_return":(192,"reverse_return",.19,"clustered","smooth"),
  "similar_habitats":(192,"reverse_return",.08,"iid","nearby"),
  "long_life":(768,"reverse_return",.07,"clustered","smooth"),
}


def confirm(root,frozen,dest,seeds):
    f,r,motion,body,_,fit=origin(root)
    for name,(blocks,mode,noise,order,appearance) in CONDITIONS.items():
        path=dest/(root.name+"-"+name+".json")
        if path.exists():continue
        selected=seeds[:64] if name=="long_life" else seeds
        arms={"rigid":[0.,0.,0.,0.,0.],**{k:v["parameters"] for k,v in frozen["settings"].items()}}
        if name in ("reverse_return","long_life"):
            best=learning.choices()[frozen["global_validation_winner"]].tolist()
            arms["selected_no_new_B_evidence"]=best
        merged={};boundary=[];start=time.perf_counter()
        size=16 if name=="long_life" else 32
        for j in range(0,len(selected),size):
            ss=selected[j:j+size];w=world.make(ss,blocks=blocks,mode=mode,noise=noise,order=order,
                                               appearance=appearance,nonce=8365331)
            boundary.extend(w.boundary.tolist())
            co=learning.Cohort(w,f,r,motion)
            normal=[k for k in arms if k!="selected_no_new_B_evidence"]
            def record(key,out):
                data=world.count_regions(w,out)
                target=merged.setdefault(key,{})
                for reg,row in data.items():
                    for metric,v in row.items():target.setdefault(reg,{}).setdefault(metric,[]).extend(v)
            for k,o in zip(normal,co.adjustments(body,np.asarray([arms[k] for k in normal]),detail=True,batch=2),strict=True):
                record(k,o)
            if "selected_no_new_B_evidence" in arms:
                original=w.arena.probe
                block=np.arange(blocks)[None,:]
                absent=(block>=w.boundary[:,0,None])&(w.block_context==1)
                mask=original|np.repeat(absent,w.arena.menu.body.trials//blocks,1)
                co.inputs[3]=gpu.tensor(mask.astype(np.uint8))
                # Physical scores continue only for the predetermined free choices.
                key="selected_no_new_B_evidence";record(key,co.adjustments(body,arms[key],detail=True)[0])
        _write(path,dict(parent=root.name,condition=name,blocks=blocks,requests=blocks*12,ticks=blocks*12*16,
               root_seeds=selected,boundaries=boundary,parameters=arms,rows=merged,seconds=time.perf_counter()-start))
        print("long-life",root.name,name,blocks*12,round(time.perf_counter()-start,1),flush=True)


def main():
    torch.set_num_threads(1);root=OUT/"unified-physical-evolution-v1";folder=OUT/"long-life-plasticity-v1"
    folder.mkdir(exist_ok=True);dest=audit.OUT/"long-life-sealed";dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(467294701).integers(8_000_000,470_000_000,192).tolist()
    parents=sorted({p.parent for p in root.glob("*--full-*/frozen.json")})
    if len(parents)!=3:raise RuntimeError("Need three independent physically trained parents")
    for p in parents:
        path=folder/(p.name+".json");sel=json.loads(path.read_text()) if path.is_file() else discover(p,folder)
        parity(p,sel,folder);confirm(p,sel,dest,seeds)
    _write(dest/"complete.json",dict(parents=[p.name for p in parents],worlds=len(seeds),long_life_worlds=64))


if __name__=="__main__":main()
