"""Fit a three-coefficient sensory-role decision from actual choices, then withhold all semantic ports."""
import argparse
from dataclasses import replace
import json
import time
import numpy as np
import torch
from . import untyped_senses as sensory,adaptive_local,adaptive_local_gpu as gpu,independent_births
from . import unified_world as game,unified_audit,unified_physical,unified_confirm,habitat_world,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def organism(fld):
    f,r,m=independent_births.resources(fld);save=next(fld.glob("*--full-*/frozen.json"))
    d=json.loads(save.read_text());gene=d["best"]["genome"]
    value=unified_physical.phenotype(unified_physical.params(f,r,m),gene,"full")[0]
    return f,r,m,value,unified_physical.materialize(f,r,m,gene,"full")


def episode(ss,tag,nonce):
    n=len(ss);setting="long_competing_run" if tag=="prolonged" else "standard"
    fam=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],n,axis=0)
    w=game.make(ss*4,force=fam,nonce=nonce,**unified_audit.SETTINGS[setting])
    packets=sensory.present(w,noise=.09 if tag=="noise" else .025,informative=tag!="unidentifiable")
    return w,packets


def correct_actual(result,ep):
    y=result.reward
    a=ep.checks["return_A"];b=ep.checks[f"B_after_{ep.steps[-1]}"]
    q=list(ep.checks.values())
    return float(.5*((y[:,a]>0)&(y[:,b]>0)).mean()+.5*(y[:,q]>0).mean())


def discover(folder,organisms,alpha):
    rng=np.random.default_rng(51328519)
    weights=np.concatenate((np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[-1,0,0]],float),
                 rng.normal(0.,1.,size=(43,3))),axis=0)
    scores=[]
    for j,case in enumerate(("standard","noise")):
        seeds=np.random.default_rng(8417391+j).integers(6_000_000,475_000_000,48).tolist()
        e,raw=episode(seeds,case,6519733);f,r,m,body,_=organisms[j]
        vals=[]
        for w in weights:
            typed,_,_=sensory.prepare(e,raw,w)
            out=gpu.Cohort(typed,f,r,m).play(body,alpha)
            vals.append(correct_actual(out,e))
        scores.append(vals)
    ix=np.argsort(-np.mean(scores,axis=0),kind="stable")[:6]
    rerank=[]
    for j,case in enumerate(("standard","prolonged","noise")):
        seeds=np.random.default_rng(41897731+j).integers(6_000_000,475_000_000,64).tolist()
        e,raw=episode(seeds,case,2411731);f,r,m,body,_=organisms[j]
        vals=[]
        for which in ix:
            typed,_,_=sensory.prepare(e,raw,weights[which])
            vals.append(correct_actual(gpu.Cohort(typed,f,r,m).play(body,alpha),e))
        rerank.append(vals)
    best=int(ix[np.asarray(rerank).mean(0).argmax()])
    result=dict(selected=weights[best].tolist(),index=best,candidates=weights.tolist(),search=scores,shortlist=ix.tolist(),
                validation=rerank,criterion="actual positive physical probe outcomes, with joint B and A recovery")
    _write(folder/"selected.json",result);return result


def study():
    torch.set_num_threads(1);folder=OUT/"untyped-senses-v1";folder.mkdir(exist_ok=True)
    dest=audit.OUT/"untyped-senses-sealed";dest.mkdir(exist_ok=True)
    rnd=OUT/"independent-lifelong-births-v1";reg=json.loads((rnd/"registered.json").read_text())
    seeds=reg["fixed_birth_seeds"][:3];alpha=reg["reopening_rule"]
    models=[organism(rnd/f"birth-{s}") for s in seeds]
    sel=json.loads((folder/"selected.json").read_text()) if (folder/"selected.json").exists() else discover(folder,models,alpha)
    roots=np.random.default_rng(959038217).integers(6_000_000,475_000_000,192).tolist()
    methods={"learned":(sel["selected"],"true"),"frequency_only":([1,0,0],"true"),
             "no_history":(sel["selected"],"no_history"),"shuffled_frequency":(sel["selected"],"shuffled_density"),
             "always_first":([0,0,1],"true"),"bilateral_only":([0,1,0],"true"),"known_roles":([0,0,0],"oracle")}
    for seed,(front,route,motion,body,material) in zip(seeds,models,strict=True):
        for name in ("standard","prolonged","noise","unidentifiable"):
            path=dest/f"{seed}-{name}.json"
            if path.exists():continue
            e,p=episode(roots,name,69194807);cases={};n=len(roots);native_check=None
            for tag,(weight,op) in methods.items():
                view,roles,res=sensory.prepare(e,p,weight,ablate=op)
                out=gpu.Cohort(view,front,route,motion).play(body,alpha)
                sc=unified_confirm.data(e,out,n)
                sc["role_recovery"]=(roles==p.truth).mean(1).reshape(4,n).tolist()
                sc["first_eight_roles"]=(roles[:,:8]==p.truth[:,:8]).mean(1).reshape(4,n).tolist()
                sc["late_roles"]=(roles[:,-16:]==p.truth[:,-16:]).mean(1).reshape(4,n).tolist()
                cases[tag]=sc
            if name=="standard":
                sub=roots[:16];w,pk=episode(sub,name,69194807)
                view,_,_=sensory.prepare(w,pk,sel["selected"])
                exp=gpu.Cohort(view,front,route,motion).play(body,alpha)
                f,rr,mm=material
                underlying=adaptive_local.LocalAdaptive(f,{**rr,"release":alpha},sub*4,motor=mm)
                brain=sensory.Live(underlying,sel["selected"]);brain.reset(len(sub)*4)
                world=sensory.RawBody(w,pk);obs=world.observe()
                while obs is not None:
                    if obs[:,:2].any():raise ValueError("A labeled context slipped into the physical interface")
                    obs=world.step(brain.act(obs))
                actual=world.result()
                native_check=dict(choices=float(np.mean(actual.contact_side==exp.contact_side)),
                    ticks=float(np.mean(actual.contact_tick==exp.contact_tick)),signed=float(np.mean(actual.reward==exp.reward)))
                if native_check["choices"]<.999:raise ValueError(native_check)
            _write(path,dict(model=seed,case=name,seeds=roots,weights=sel["selected"],rows=cases,live_generic_port=native_check))
            print(seed,name,{k:round(float(np.mean(np.asarray(v["return_A"])[[1,2]])),3) for k,v in cases.items()},flush=True)
    _write(dest/"complete.json",dict(births=seeds,worlds=len(roots)))


if __name__=="__main__":study()
