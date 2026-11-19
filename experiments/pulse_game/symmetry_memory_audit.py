"""Independent causal behavior probes of sign, geometry and finite event memory."""
import argparse
import json
import time
import numpy as np
import torch

from . import symmetry_audit as design, symmetry_physics as gpu, symmetry_grounded as grounded
from . import symmetry_memory_train as memory, symmetry_world as game, research_audit as audit
from .attractor_search import OUT
from .paired_choice import first_contact_pairs
from experiments.seed_evolution.large_runner import _write


def rows(w,res,master):
    met=game.metrics(w,res);n=len(w.family)//2
    a,b=met["first"][:n],met["first"][n:]
    j=w.exposed
    return dict(first=(a+b)*.5,both=a*b,even=a,odd=b,**first_contact_pairs(w,res),
                all=(met["all"][:n]+met["all"][n:])*.5,
                contact=(met["contact"][:n]+met["contact"][n:])*.5,
                gain_even=master[:n,j],gain_odd=master[n:,j],anchor=w.anchor_sign[:n,j+1].astype(float))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="symmetry-grounded-v1")
    p.add_argument("--kind",choices=("ground","memory"),default="ground")
    p.add_argument("--worlds",type=int,default=512);p.add_argument("--tag",default="external")
    p.add_argument("--models",nargs="+");p.add_argument("--steps",type=int,default=750)
    p.add_argument("--partial",action="store_true")
    args=p.parse_args();torch.set_num_threads(1)
    root=OUT/args.folder
    dest=audit.OUT/(args.folder+"-"+args.tag);dest.mkdir(exist_ok=True)
    seeds=np.random.default_rng(318175243).integers(5_000_000,470_000_000,size=args.worlds).tolist()
    if len(set(seeds))!=len(seeds):raise ValueError("Repeated held-out seed")
    if args.models:
        expected=args.models
    elif args.kind=="ground":
        expected=[f"{init}-{rule}-s{seed}" for seed in (7243,4187,9679) for init in ("public","unrelated","random")
                  for rule in ("delta","learned","fixed")]
    else:
        expected=[f"m20-{rule}-aux{aux}-s{seed}" for seed in (9643,3137,8273) for aux in (30,0) for rule in ("delta","learned","fixed")]
    conditions={
      "standard":dict(pairs=12,probes=4,span=4,noise=.04,jitter=.08,matched=True),
      "long_interleaving":dict(pairs=12,probes=4,span=12,noise=.04,jitter=.15,matched=True),
      "two_pairs":dict(pairs=2,probes=4,span=2,noise=.04,jitter=.08,matched=True),
      "twentyfour_pairs":dict(pairs=24,probes=4,span=6,noise=.12,jitter=.12,matched=True),
      "same_history":dict(pairs=12,probes=4,span=4,noise=.04,jitter=.08,matched=True,mask=True)}
    mats={};worlds={}
    for name,condition in conditions.items():
        w=design.joint(seeds,**condition);worlds[name]=w;mats[name]=gpu.Physical(w)
    checked=set();start=time.perf_counter()
    while len(checked)<len(expected):
        avail=[]
        for name in expected:
            h=root/(name+"-history.json")
            if name in checked or not h.exists():continue
            if json.loads(h.read_text())[-1]["step"]>=args.steps:avail.append(name)
        for name in avail:
            for checkpoint in (["","-before"] if args.kind=="ground" else [""]):
                path=root/(name+checkpoint+".pt")
                save=dest/(name+checkpoint+".json")
                if save.exists():continue
                loader=grounded.load if args.kind=="ground" else memory.load
                params=loader(path)
                cases={}
                for label,sim in mats.items():
                    w=worlds[label]
                    arm=gpu.CHANGES if label=="standard" else ("true",)
                    data=sim.run(params,changes=arm)
                    if label=="standard":
                        data["cut_cross"]=sim.run(loader(path,cut="cross"))["true"]
                        data["cut_all"]=sim.run(loader(path,cut="all"))["true"]
                    cases[label]={}
                    for k,(result,master) in data.items():
                        m=rows(w,result,master)
                        if label=="same_history" and (np.any(m["both"]) or not np.array_equal(m["first_side_even"],m["first_side_odd"])):
                            raise RuntimeError("Hidden-goal null violated")
                        cases[label][k]={metric:v.tolist() for metric,v in m.items()}
                _write(save,dict(model=name,source=audit.fingerprint(path),checkpoint=checkpoint or "selected",
                        seeds=seeds,protocols=conditions,rows=cases,
                        pair_choice_protocol="both_first_contacts_required"))
            checked.add(name)
            print("audited",name,round(time.perf_counter()-start,1),flush=True)
        if args.partial:break
        if len(checked)<len(expected):time.sleep(8)
    _write(dest/"status.json",dict(checked=sorted(checked),expected=expected,n=len(seeds)))


if __name__=="__main__":main()
