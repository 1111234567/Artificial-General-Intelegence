"""Forward-only internal review using archived actual outcomes or the circuit's earlier rule estimates."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import argparse
import json
import time
import numpy as np
import torch
from . import unified_audit as recipe,unified_world as worlds,unified_physical as engine,habitat_world as body
from . import habitat_deploy as simple,old_experience as study,research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def evaluate(job):
    parent,condition,options,seeds,nonce=job;torch.set_num_threads(1)
    src=OUT/"unified-physical-evolution-v1"/parent/"frozen.json";opt=json.loads(src.read_text())
    s=opt["settings"];f,r,m=recipe.resources(OUT/"unified-transfer-v1",s["model"])
    f,r,m=engine.materialize(f,r,m,np.asarray(opt["best"]["genome"]),s["mode"])
    cfg=recipe.SETTINGS[condition];n=len(seeds);fam=np.repeat([[1,1],[1,-1],[-1,1],[-1,-1]],n,axis=0)
    e=worlds.make(seeds*4,force=fam,nonce=nonce,**cfg)
    if options is None:a=simple.Forager(f,r,seeds*4,start=e.arena.start_test,motor=m)
    else:a=study.Agent(f,r,seeds*4,start=e.arena.start_test,motor=m,options=study.Options(**options))
    out=body.rollout(a,e.arena,silent_probes=True)
    d=worlds.measures(e,out);last=f"B_after_{e.steps[-1]}"
    d["last_B"]=d[last]
    d["internal_updates"]=a.replays if options else np.zeros(len(seeds)*4)
    return {k:v.reshape(4,n).tolist() for k,v in d.items()}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--workers",type=int,default=6)
    args=ap.parse_args();folder=OUT/"old-experience-v1";folder.mkdir(exist_ok=True);dest=audit.OUT/"old-experience-sealed";dest.mkdir(exist_ok=True)
    models=sorted(p.parent.name for p in (OUT/"unified-physical-evolution-v1").glob("*--full-*/frozen.json"))
    roots=np.random.default_rng(89256313).integers(7_000_000,475_000_000,40).tolist()
    held=np.random.default_rng(96781931).integers(7_000_000,475_000_000,144).tolist()
    proposals=[study.Options(storage=st,choose=ch,target=t,step=lr)
       for st in ("reservoir","fifo") for ch in ("uniform","interference") for t in ("actual","self") for lr in (.018,.07,.26)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in models:
            freeze=folder/(name+".json")
            if not freeze.exists():
                queue={(i,c):pool.submit(evaluate,(name,c,asdict(o),roots,69312413)) for i,o in enumerate(proposals)
                       for c in ("standard","long_competing_run")}
                scores={}
                for (i,c),task in queue.items():
                    v=task.result()
                    scores.setdefault(i,[]).append(float(.50*np.mean(v["both"])+.25*np.mean(v["return_A"])+.25*np.mean(v["last_B"])))
                rank={i:float(np.mean(x)) for i,x in scores.items()}
                selected={}
                for which in ("actual","self"):
                    pool_idx=[i for i,o in enumerate(proposals) if o.target==which and o.storage=="reservoir" and o.choose=="interference"]
                    k=max(pool_idx,key=rank.get);selected[which]=asdict(proposals[k])
                _write(freeze,dict(selected=selected,development=roots,options=[asdict(o) for o in proposals],scores=rank))
            config=json.loads(freeze.read_text())["selected"]
            for case in ("standard","similar_habitats","long_competing_run"):
                path=dest/(name+"-"+case+".json")
                if path.exists():continue
                variants={"native":None}
                for ty in ("actual","self"):
                    o=study.Options(**config[ty]);variants[ty]=o
                    variants[ty+"_shuffled"]=replace(o,binding="wrong")
                    variants[ty+"_uniform_matched"]=replace(o,choose="matched_uniform")
                    variants[ty+"_FIFO"]=replace(o,storage="fifo")
                variants["actual_only_after_A"]=replace(variants["actual"],start_after=True)
                tasks={k:pool.submit(evaluate,(name,case,asdict(o) if o else None,held,48650917)) for k,o in variants.items()}
                cases={k:t.result() for k,t in tasks.items()}
                _write(path,dict(parent=name,case=case,seeds=held,choices=config,rows=cases))
                print(name,case,{k:round(float(np.asarray(v["return_A"])[[1,2]].mean()),3) for k,v in cases.items()},flush=True)
    _write(dest/"complete.json",dict(models=models,worlds=len(held)))


if __name__=="__main__":main()
