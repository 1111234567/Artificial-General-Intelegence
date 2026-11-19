"""Learn which existing synaptic address to reinforce using only actual retest outcomes."""
import argparse
import json
import time
import numpy as np
import torch

from . import recurring_memory as plastic, recurring_world as task, symmetry_grounded as sensory
from . import symmetry_policy as controller, symmetry_policy_train as original, research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def parent(name):
    path=OUT/"symmetry-policy-v1"/name/"frozen.json"
    source=json.loads(path.read_text())
    fp=OUT/"symmetry-grounded-v1"/(source["settings"]["parent"]+".pt")
    if audit.fingerprint(fp)!=source["settings"]["parent_sha"]:raise RuntimeError("Parent checkpoint changed")
    params=sensory.load(fp);vector=source["best"]["genome"]
    pheno=controller.decoded(params,vector,source["settings"]["mode"])[0]
    return params,pheno,vector,source


def cohort(params,seeds,gap,distinct,recurring,query,spread,noise,conflict=False,probes=2):
    w,_=task.make(seeds,gap=gap,distinct=distinct,recurring=recurring,query=query,spread=spread,noise=noise,
                  conflicts=conflict,probes=probes)
    return plastic.Cohort(params,w)


VALIDATION=((32,8,True,"anchor",.07,.10),(32,8,True,"random",.07,.12),
            (40,16,True,"random",.15,.12),(24,12,False,"random",.07,.10))


def search(name,seed,mode,folder,args):
    params,body,_,source=parent(name)
    dest=folder/(name+f"-writer{mode}");dest.mkdir(exist_ok=True)
    frozen=dest/"frozen.json"
    if frozen.exists():return
    rng=np.random.default_rng(seed)
    val=[cohort(params,original.evalseeds(8519303+i,72),*cfg) for i,cfg in enumerate(VALIDATION)]
    center=np.zeros(7);sigma=np.ones(7)*1.35;best=np.zeros(7)
    best_score=-np.inf
    hist=[];start=time.perf_counter()
    for gen in range(args.gens):
        e=np.random.default_rng(np.random.SeedSequence([seed,gen,16419]))
        gap=int(e.choice([20,24,32,40,48]))
        distinct=int(e.choice([4,6,8,12,16,20]))
        recur=bool(e.random()<.76);query=e.choice(["random","random","anchor","rare"])
        seeds=e.integers(10_000_000,470_000_000,size=args.worlds).tolist()
        c=cohort(params,seeds,gap,distinct,recur,query,float(e.uniform(.02,.18)),float(e.uniform(.02,.19)),e.random()<.2)
        perturb=rng.normal(size=((args.width-4)//2,7))
        population=np.concatenate(([center,best,np.zeros(7),rng.normal(0,2,7)],center+sigma*perturb,center-sigma*perturb))
        result=c.replacements(body,population,mode=mode)
        f=original.utility(result);rank=np.argsort(-f,kind="stable")[:max(8,args.width//10)]
        parents=population[rank]
        center=.45*center+.55*np.mean(parents,0)
        sigma=np.clip(.89*sigma+.11*parents.std(0),.18,2.1)
        line=dict(generation=gen+1,train_mean=result.mean(0).tolist(),top=result[rank[0]].tolist(),
                  gap=gap,distinct=distinct,query=query,recurring=recur,worlds=seeds)
        if gen==0 or (gen+1)%4==0:
            candidates=np.concatenate(([best,center],parents[:5]))
            results=np.stack([v.replacements(body,candidates,mode=mode) for v in val])
            score=original.utility(results.mean(0));ix=int(score.argmax())
            if score[ix]>best_score:best_score=float(score[ix]);best=candidates[ix].copy()
            line["validation"]=score.tolist()
        hist.append(line)
        if (gen+1)%8==0:
            _write(dest/"progress.json",dict(history=hist,best=best.tolist(),best_score=best_score,elapsed=time.perf_counter()-start))
            print(name,mode,gen+1,round(best_score,4),round(time.perf_counter()-start,1),flush=True)
    _write(frozen,dict(parent=name,seed=seed,mode=mode,width=args.width,generations=args.gens,worlds_per_gen=args.worlds,
                       genes=best.tolist(),phenotype=plastic.traits(best)[0].tolist(),score=best_score,
                       parent_source=source["settings"]["parent_sha"],elapsed=time.perf_counter()-start))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--folder",default="recurring-write-v1")
    p.add_argument("--width",type=int,default=512);p.add_argument("--gens",type=int,default=64)
    p.add_argument("--worlds",type=int,default=32);args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;folder.mkdir(exist_ok=True)
    roots=[("public-learned-s7243--full-e8719",11843),("public-learned-s4187--full-e9173",9817),
           ("public-learned-s9679--full-e5879",6217)]
    for par,seed in roots:
        src=OUT/"symmetry-policy-v1"/par/"frozen.json"
        while not src.is_file():time.sleep(8)
        for mode in (1,2):search(par,seed,mode,folder,args)
    _write(folder/"complete.json",dict(parents=roots,width=args.width,generations=args.gens))


if __name__=="__main__":main()
