"""Refine body control and the circuit's local plasticity from real game outcomes."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import symmetry_audit as task, symmetry_world as game, symmetry_policy as policy, symmetry_grounded as loader
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def games(seeds,pairs,span,gap,noise,jitter,probes=3):
    w=task.joint(seeds,pairs=pairs,probes=probes,span=span,noise=noise,jitter=jitter,matched=True)
    return game.with_distractors(w,gap)


def utility(rows):
    return .75*rows[...,0]+.15*rows[...,1]+.1*rows[...,2]


def evalseeds(seed,n):
    return np.random.default_rng(seed).integers(10_000_000,470_000_000,size=n).tolist()


def optimize(folder,path,seed,mode,args):
    name=f"{path.stem}--{mode}-e{seed}"
    dest=folder/name;dest.mkdir(exist_ok=True);checkpoint=dest/"checkpoint.json"
    params=loader.load(path);dim=policy.dimensions(params,mode)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    settings=dict(parent=path.stem,parent_sha=sha,seed=seed,mode=mode,width=args.width,
                  generations=args.generations,lives_per_family=args.worlds,dimension=dim)
    rng=np.random.default_rng(seed)
    n_islands=4;width=args.width//n_islands
    if width%2 or width<16:raise ValueError("Use four even populations")
    if checkpoint.exists():
        s=json.loads(checkpoint.read_text())
        if s["settings"]!=settings:raise ValueError("Incompatible continuation")
        mu,sig,elite=[np.asarray(s[k],np.float64) for k in ("mu","sig","elite")]
        hist=s["history"];best=s["best"];first=len(hist);rng.bit_generator.state=s["rng"]
    else:
        mu=np.zeros((n_islands,dim));sig=np.ones_like(mu)*np.array([.34,.7,1.25,1.75])[:,None];elite=mu.copy()
        best=None;hist=[];first=0
    # Same validation lifetimes for motor-only and full mutants for each birth.
    protocol=((12,4,0,.04,.08),(8,4,2,.06,.12),(12,6,4,.08,.15),(12,4,8,.04,.10))
    check=[policy.Cohort(params,games(evalseeds(37570915+31*i,96),*condition)) for i,condition in enumerate(protocol)]
    initial=np.zeros((1,dim))
    if first==0:
        original=np.stack([c.run(initial,mode)[0] for c in check])
        best=dict(gen=0,utility=float(utility(original.mean(0))),native=original.tolist(),genome=initial[0].tolist())
        _write(dest/"parent.json",best)
    start=time.perf_counter()
    for step in range(first+1,args.generations+1):
        # Paired radius, body outcomes, random environment and intervention order
        # are shared between the motor-only and full-plasticity search. Separate
        # random streams prevent the genotype dimension changing the world draw.
        env=np.random.default_rng(np.random.SeedSequence([seed,step,851]))
        pairs=int(env.choice([8,12,16]))
        span=int(env.choice([2,4,4,6,8]))
        gap=int(env.choice([0,0,1,2,4,6,8]))
        nseeds=env.integers(10_000_000,470_000_000,size=args.worlds).tolist()
        episode=games(nseeds,pairs,span,gap,float(env.uniform(.0,.12)),float(env.uniform(.035,.16)))
        group=policy.Cohort(params,episode)
        children=[]
        for i in range(n_islands):
            direction=rng.normal(size=((width-2)//2,dim))
            population=np.concatenate((elite[i:i+1],mu[i:i+1],mu[i]+sig[i]*direction,mu[i]-sig[i]*direction))
            # Reflect finite coordinates, then decode phenotype parameters into
            # stable physical ranges. Separate broad islands retain exploration.
            bound=6.;population=bound-np.abs((population+bound)%(4*bound)-2*bound)
            children.append(population)
        children=np.stack(children)
        metrics=group.run(children.reshape(-1,dim),mode).reshape(n_islands,width,4)
        score=utility(metrics);order=np.argsort(-score,axis=1,kind="stable")
        top=order[:,:max(8,width//12)]
        selected=np.take_along_axis(children,top[:,:,None],axis=1)
        weights=np.log(top.shape[1]+.5)-np.log(np.arange(1,top.shape[1]+1))
        weights/=weights.sum()
        center=(selected*weights[None,:,None]).sum(1)
        spread=np.sqrt((((selected-center[:,None])**2)*weights[None,:,None]).sum(1))
        mu=.50*mu+.50*center
        sig=np.maximum(np.array([.07,.12,.20,.32])[:,None],.91*sig+.09*spread)
        elite=selected[:,0].copy()
        entry=dict(generation=step,gap=gap,span=span,pairs=pairs,worlds=nseeds,
                   train_top=metrics.reshape(-1,4)[score.argmax()].tolist(),mean=metrics.mean((0,1)).tolist(),
                   elapsed_seconds=time.perf_counter()-start)
        if step==1 or step%8==0 or step==args.generations:
            candidates=np.concatenate((elite,mu,np.asarray(best["genome"])[None,:]))
            validates=np.stack([c.run(candidates,mode) for c in check])
            u=utility(validates.mean(0));ix=int(u.argmax())
            if u[ix]>best["utility"]:
                best=dict(gen=step,utility=float(u[ix]),native=validates[:,ix].tolist(),genome=candidates[ix].tolist())
            entry["validation"]=u.tolist()
            if step%24==0:
                # One mild migration retains the repeatedly validated behavior;
                # broadest island keeps its own radius.
                elite[step//24%n_islands]=best["genome"]
            print(name,step,"validation",round(best["utility"],4),"elapsed",round(time.perf_counter()-start),flush=True)
        hist.append(entry)
        if step%4==0 or step==args.generations:
            _write(checkpoint,dict(settings=settings,mu=mu.tolist(),sig=sig.tolist(),elite=elite.tolist(),
                                   history=hist,rng=rng.bit_generator.state,best=best))
    _write(dest/"frozen.json",dict(settings=settings,best=best,protocol=protocol,parent=path.as_posix()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--folder",default="symmetry-policy-v1")
    p.add_argument("--parents",nargs="+",default=["public-learned-s7243","public-learned-s4187","public-learned-s9679"])
    p.add_argument("--seeds",nargs="+",type=int,default=[8719,9173,5879])
    p.add_argument("--modes",nargs="+",default=["motor","full"])
    p.add_argument("--width",type=int,default=1024);p.add_argument("--generations",type=int,default=160)
    p.add_argument("--worlds",type=int,default=48)
    args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    done=[]
    for name,seed in zip(args.parents,args.seeds,strict=True):
        root=OUT/"symmetry-grounded-v1"
        hist=root/(name+"-history.json")
        while not hist.exists() or json.loads(hist.read_text())[-1]["step"]<750:time.sleep(6)
        for mode in args.modes:
            _write(folder/"status.json",dict(active=[name,mode],done=done))
            optimize(folder,root/(name+".pt"),seed,mode,args)
            done.append([name,mode])
    _write(folder/"status.json",dict(active=None,done=done))


if __name__=="__main__":main()
