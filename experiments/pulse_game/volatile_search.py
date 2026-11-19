"""Select on mixed stationarity, then re-evaluate local plasticity under independent changes."""

import os
for _var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_var]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import json
from pathlib import Path
import time

import numpy as np

from . import ring_spikes,volatile_ring as vr
from .motor_search import write,OUT


FOLDER=OUT/"ring-volatile-v1"


def trial_job(args):
    cfg,r,net,seeds,noise,T,flip,restore=args
    return vr.evaluation(ring_spikes.Config(**cfg),vr.Release(**r),net,seeds,noise,
                         trials=T,flip=flip,restore=restore)


def settings():
    rng=np.random.default_rng(45033)
    out=[vr.Release()]
    for i in range(47):
        out.append(vr.Release(amount=float(np.exp(rng.uniform(np.log(.12),np.log(16)))),
                              threshold=float(rng.uniform(.05,.75)),
                              hysteresis=float(rng.uniform(.08,.65)) if i%2 else 0.,
                              exploration=float(rng.uniform(.05,.55)) if (i//2)%2 else 0.,
                              surprise_rate=float(rng.uniform(.21,.75))))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("select","confirm"),required=True)
    ap.add_argument("--workers",type=int,default=6)
    args=ap.parse_args()
    cfg=json.loads((OUT/"ring-value-search-v1/frozen.json").read_text())["best"]["config"]
    FOLDER.mkdir(parents=True,exist_ok=True)
    def seeds(seed,lives):return np.random.default_rng(seed).integers(10_000_000,490_000_000,size=lives).tolist()
    start=time.perf_counter()
    if args.stage=="select":
        settings_list=settings()
        path=FOLDER/"development.json"
        state=json.loads(path.read_text()) if path.exists() else {}
        jobs=[]
        for i,r in enumerate(settings_list):
            for flip in (True,False):
                key=f"{i}-{int(flip)}"
                if key not in state:
                    jobs.append((key,(cfg,asdict(r),417,seeds(73771,48),.12,176,flip,False)))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks=[(key,pool.submit(trial_job,job)) for key,job in jobs]
            for n,(key,job) in enumerate(tasks,1):
                res=job.result()
                state[key]={"overall":float(np.mean(res["first_rate"])),
                            **{k:res[k]["pooled"] for k in ("post_affected","post_untouched","late_affected","pre_last")}}
                if n%8==0:write(path,state);print(n,len(tasks),round(time.perf_counter()-start),flush=True)
        write(path,state)
        value=lambda i:np.mean([state[f"{i}-{j}"]["overall"] for j in (0,1)])
        ranking=sorted(range(len(settings_list)),key=value,reverse=True)
        retained=ranking[:4]+[0]
        # Rerank on new development worlds and two graph births.
        ret={}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            specs=[(i,b,f) for i in retained for b in (671,449) for f in (True,False)]
            work=[(cfg,asdict(settings_list[i]),b,seeds(127390,64),.16,208,f,False) for i,b,f in specs]
            for (i,b,f),v in zip(specs,pool.map(trial_job,work),strict=True):
                ret.setdefault(i,[]).append(float(np.mean(v["first_rate"])))
        winner=max(ret,key=lambda i:np.mean(ret[i]))
        best=asdict(settings_list[winner])
        write(FOLDER/"frozen.json",{"winner":winner,"selected":best,"validation_overall":ret,
              "search_grid":[asdict(r) for r in settings_list],"criterion":"equal mean first-bite accuracy across stable and reversing noisy lives",
              "graph_seeds_train":[417,671,449],"base":cfg})
        print("selected",winner,best,ret,flush=True)
        return
    frozen=json.loads((FOLDER/"frozen.json").read_text())
    r=vr.Release(**frozen["selected"])
    # If unmodified baseline won, also test the best nonzero release variant,
    # explicitly keeping the failed selection rather than replacing it.
    if not r.amount:
        ix=max((int(k) for k in frozen["validation_overall"] if int(k)!=0),key=lambda k:np.mean(frozen["validation_overall"][str(k)]))
        altered=vr.Release(**frozen["search_grid"][ix])
    else:altered=r
    variants={"base":vr.Release(),"local":altered,
              "uniform":replace(altered,scope="uniform"),
              "permuted":replace(altered,scope="permuted"),
              "explore_only":replace(altered,amount=0.),
              "release_only":replace(altered,exploration=0.),
              "instant":replace(altered,hysteresis=0.)}
    unique={};aliases={}
    for label,value in variants.items():
        # Inactive code paths have no observable settings. Reuse a matched
        # measurement instead of simulating byte-for-byte equivalent variants.
        if not value.amount and not value.exploration:value=vr.Release()
        canonical=unique.setdefault(value,label)
        aliases[label]=canonical
    jobs=[];root=FOLDER/"confirm";root.mkdir(exist_ok=True)
    roster=seeds(70249118,128)
    for b in (3,19,37):
        for noise in (0.,.18,.32):
            for regime,(flip,restore) in {"stationary":(False,False),"reversal":(True,False),"return":(True,True)}.items():
                for opt,label in unique.items():
                    name=f"b{b}-n{int(noise*100):02d}-{regime}-{label}"
                    if not (root/(name+".json")).exists():
                        jobs.append((name,(cfg,asdict(opt),b,roster,noise,224,flip,restore)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[(k,pool.submit(trial_job,v)) for k,v in jobs]
        for n,(name,pending) in enumerate(tasks,1):
            write(root/(name+".json"),pending.result())
            if n%8==0:print(n,len(jobs),round(time.perf_counter()-start),flush=True)
    write(FOLDER/"confirmation-metadata.json",{"worlds":roster,"graph_seeds":[3,19,37],
          "variants":{k:asdict(v) for k,v in variants.items()},"aliases":aliases,"frozen":frozen["selected"]})


if __name__=="__main__":
    main()
