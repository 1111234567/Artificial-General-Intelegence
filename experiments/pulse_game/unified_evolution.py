"""Optimize the single continuous phenotype on its actual free choices and return to A."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import numpy as np
import torch

from . import unified_physical as run, unified_world as game, unified_audit as auditgame, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def env_at(seed,g,lives):
    rng=np.random.default_rng(np.random.SeedSequence([seed,g,739901]))
    roots=rng.integers(3_000_000,475_000_000,lives).tolist()
    initial=int(rng.choice([8,12,16]));end=int(rng.choice([8,12,16,20]))
    mid=sorted(rng.choice(np.arange(1,end//2),size=2,replace=False)*2)
    spec=dict(initial=initial,depths=(0,*map(int,mid),end),span=2,noise=float(rng.uniform(.015,.15)),
              jitter=float(rng.uniform(.015,.15)),appearance=str(rng.choice(["discrete","smooth","smooth","nearby"])),
              same_prior=.77,nonce=98918417)
    return game.make(roots,**spec),spec,roots


def objective(m):return .45*m[...,3]+.40*m[...,4]+.15*m[...,0]


def optimize(folder,model,seed,mode,args,*,frozen_inputs=None,model_path=None,environment_stream=None):
    source=OUT/"unified-transfer-v1"
    front,route,motion=auditgame.resources(source,model) if frozen_inputs is None else frozen_inputs
    model_path=source/(model+".pt") if model_path is None else model_path
    n=7 if mode=="motor" else 19
    root=folder/(model+"--"+mode+f"-e{seed}");root.mkdir(exist_ok=True)
    if (root/"frozen.json").exists():return
    details=dict(model=model,model_sha=audit.fingerprint(model_path),mode=mode,seed=seed,
                 width=args.width,gens=args.gens,lives_per_gen=args.lives)
    make_episode=env_at if environment_stream is None else environment_stream
    if environment_stream is not None:details["curriculum"]=environment_stream.__name__
    rng=np.random.default_rng(seed)
    path=root/"checkpoint.json"
    if path.exists():
        d=json.loads(path.read_text())
        if d["settings"]!=details:raise RuntimeError("Checkpoint settings changed")
        centers=np.asarray(d["centers"]);std=np.asarray(d["std"]);elite=np.asarray(d["elite"])
        best=d["best"];hist=d["history"];rng.bit_generator.state=d["rng"]
    else:
        centers=np.zeros((4,n))
        std=np.broadcast_to(np.array([.35,.65,1.05,1.45])[:,None],(4,n)).copy()
        elite=centers.copy();best=None;hist=[]
    validation=[]
    for i,setting in enumerate(("standard","similar_habitats","named_channels","long_competing_run")):
        ss=np.random.default_rng(36478591+i).integers(3_000_000,475_000_000,size=160).tolist()
        w=game.make(ss,**auditgame.SETTINGS[setting],same_prior=.77,nonce=35179263)
        validation.append(run.Cohort(w,front,route,motion))
    if best is None:
        scores=np.stack([v.run(np.zeros(n),mode)[0] for v in validation])
        best=dict(generation=0,score=float(objective(scores.mean(0))),genome=np.zeros(n).tolist(),validation=scores.tolist())
        _write(root/"parent.json",best)
    width=args.width//4
    if args.width%8:raise ValueError("Expected four equal even-sized groups")
    st=time.perf_counter()
    # Worlds and bodily RNGs depend only on the step, so preparing the next
    # environment while the current GPU batch runs preserves candidate histories.
    with ThreadPoolExecutor(max_workers=1) as cpu:
        future=cpu.submit(make_episode,seed,len(hist)+1,args.lives)
        for g in range(len(hist)+1,args.gens+1):
            episode,env,roots=future.result()
            if g<args.gens:future=cpu.submit(make_episode,seed,g+1,args.lives)
            cohort=run.Cohort(episode,front,route,motion)
            children=[]
            for i in range(4):
                d=rng.normal(size=((width-2)//2,n))
                x=np.concatenate((elite[i:i+1],centers[i:i+1],centers[i]+std[i]*d,centers[i]-std[i]*d))
                x=7.-np.abs((x+7.)%28.-14.);children.append(x)
            children=np.stack(children)
            perf=cohort.run(children.reshape(-1,n),mode).reshape(4,width,6)
            score=objective(perf);order=np.argsort(-score,axis=1,kind="stable")[:,:max(8,width//12)]
            top=np.take_along_axis(children,order[:,:,None],axis=1)
            elite=top[:,0].copy()
            weight=np.log(order.shape[1]+.5)-np.log(np.arange(1,order.shape[1]+1));weight/=weight.sum()
            pos=(top*weight[None,:,None]).sum(1)
            spread=np.sqrt(((top-pos[:,None])**2*weight[None,:,None]).sum(1))
            centers=.55*centers+.45*pos
            std=np.maximum(np.array([.06,.10,.18,.30])[:,None],.90*std+.10*spread)
            entry=dict(generation=g,environment=env,roots=roots,mean=perf.mean((0,1)).tolist(),
                       winner=perf.reshape(-1,6)[score.argmax()].tolist(),elapsed=time.perf_counter()-st)
            if g==1 or g%8==0 or g==args.gens:
                population=np.concatenate((elite,centers,np.asarray(best["genome"])[None]))
                val=np.stack([v.run(population,mode) for v in validation])
                ranks=objective(val.mean(0));j=int(ranks.argmax())
                if ranks[j]>best["score"]:
                    best=dict(generation=g,score=float(ranks[j]),genome=population[j].tolist(),validation=val[:,j].tolist())
                entry["validation"]=ranks.tolist()
                if g%32==0:elite[g//32%4]=best["genome"]
                print(model,mode,g,"retention and learning",round(best["score"],4),"s",round(time.perf_counter()-st),flush=True)
            hist.append(entry)
            if g%4==0 or g==args.gens:
                _write(path,dict(settings=details,history=hist,centers=centers.tolist(),std=std.tolist(),elite=elite.tolist(),
                                 best=best,rng=rng.bit_generator.state))
    _write(root/"frozen.json",dict(settings=details,best=best,
           phenotype=run.phenotype(cohort.initial,np.asarray(best["genome"]),mode)[0].tolist(),
           seconds=time.perf_counter()-st))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--folder",default="unified-physical-evolution-v1")
    p.add_argument("--width",type=int,default=1024);p.add_argument("--gens",type=int,default=224)
    p.add_argument("--lives",type=int,default=80);args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    done=[]
    for s,seed in ((1373,19781),(8539,47893),(6547,81913)):
        model=f"rls-learned-s{s}"
        for mode in ("motor","full"):
            _write(folder/"progress.json",dict(active=[model,mode],done=done))
            optimize(folder,model,seed,mode,args);done.append([model,mode])
    _write(folder/"progress.json",dict(active=None,done=done))


if __name__=="__main__":main()


