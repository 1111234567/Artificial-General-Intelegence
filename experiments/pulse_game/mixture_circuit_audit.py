"""Paired behavior and implementation checks for a continuously carried sensory graph."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import mixture_circuit as circuit, mixture_circuit_train as training, mixture_circuit_deploy as body
from . import mixture_flash as flash, mixture_metric, mixture_world as game, research_audit as audit
from .attractor_search import OUT


def parity(folder):
    raw=flash.public_examples(folder,51623871,6,13)
    batch=training.torch_data(raw,"cpu")
    rng=np.random.default_rng(9183)
    t=torch.tensor(rng.integers(0,16,size=raw["touch"].shape))
    parity_rows=[]
    for typ in ("ridge","delta","local"):
        torch.manual_seed(7193)
        cfg=circuit.Config(learner=typ)
        model=circuit.TasteCircuit(cfg).double()
        dat={k:v.double() if v.is_floating_point() else v for k,v in batch.items()}
        with torch.no_grad():
            ref=model.states(dat,t)
            a=model.matrix()
            state=torch.zeros((len(t),2,cfg.nodes),dtype=torch.float64)
            recorded=torch.zeros_like(ref)
            wprev=torch.zeros((len(t),cfg.latent),dtype=torch.float64)
            param=model.export()
            actual=body.FlashForager(param,7,list(range(len(t))))
            actual.reset(len(t))
            # Independently move each side through every tick from birth.
            for j in range(t.shape[1]):
                for tick in range(16):
                    x=dat["odor"][:,j]*(dat["ot"][:,j]==tick)[:,None,None]
                    ctx=dat["ctx"][:,j]*(dat["ct"][:,j]==tick)[:,None]
                    state=state@a + x@model.sensors[:4] + (ctx@model.sensors[4:])[:,None]
                    take=t[:,j]==tick
                    recorded[take,j]=state[take]
            err=float((recorded-ref).abs().max())
            h=8
            pred=model.value(dat,h,t)
            at=model.states(dat,batch["touch"].long())
            ix=torch.arange(len(t))
            if typ=="ridge":p=torch.eye(cfg.latent,dtype=torch.float64).expand(len(t),-1,-1).clone()/param["ridge"]
            for j in range(h):
                k,w,c=model.features(at[ix,j,batch["side"][:,j].long()])
                q=(wprev*k).sum(1)
                y=dat["reward"][:,j]
                if typ=="ridge":
                    z=(p@k.unsqueeze(2)).squeeze(2)
                    gain=z/(1+(z*k).sum(1,keepdim=True))
                    p=p-gain.unsqueeze(-1)*z.unsqueeze(1)
                    wprev=wprev+gain*(y-q)[:,None]
                else:wprev=wprev+w*(c[:,0]*y+c[:,1]*q+c[:,2])[:,None]
            ks=model.features(recorded[ix[:,None],torch.arange(h,t.shape[1])[None,:],batch["side"][:,h:].long()])[0]
            loop=(ks*wprev[:,None]).sum(2)
            parity_rows.append(dict(learner=typ,max_state=float(err),max_prediction=float((loop-pred).abs().max())))
    if any(max(r["max_state"],r["max_prediction"])>1e-9 for r in parity_rows):raise RuntimeError(parity_rows)
    _write(folder/"scan-parity.json",parity_rows)


def one(spec):
    modelpath, family, distribution, full_controls, seeds, dest = spec
    path=dest/f"{modelpath.stem}-{family}-{distribution}.json"
    if path.exists():return
    torch.set_num_threads(1)
    m=game.make(seeds,128,rule=family,distribution=distribution)
    def agent(change=None):
        if modelpath.stem=="visible-compartment":
            params=mixture_metric.load(OUT/"mixture-meta-v1/compartment-s3361.pt")
            return mixture_metric.LearnedForager(params,17,seeds,learn=change!="no_write")
        params=training.load(modelpath,cut=change if change in ("cross","all") else None)
        return body.FlashForager(params,17,seeds,writes=change!="no_write",disrupt=change)
    actual=flash.rollout(agent(),m)
    rows={"true":game.metrics(m,actual),"silent_second":game.metrics(m,flash.rollout(agent(),m,hidden_after=m.counter))}
    if full_controls:
        for tag in ("no_write","cross","all","silence"):
            rows[tag]=game.metrics(m,flash.rollout(agent(tag),m))
        rows["other_life"]=game.metrics(m,flash.rollout(audit.ContactMatched(agent(),actual.reward[np.arange(len(seeds))^1]),m))
    _write(path,dict(model=modelpath.stem,source=audit.fingerprint(modelpath) if modelpath.is_file() else None,
           family=family,distribution=distribution,seeds=seeds,
           rows={k:{metric:v.tolist() for metric,v in values.items()} for k,values in rows.items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",default="mixture-circuit-v1");p.add_argument("--workers",type=int,default=5)
    p.add_argument("--worlds",type=int,default=128)
    args=p.parse_args();torch.set_num_threads(1)
    folder=OUT/args.folder;dest=audit.OUT/(args.folder+"-external");dest.mkdir(exist_ok=True)
    parity(folder)
    seeds=[int(np.random.SeedSequence([s,9180021]).generate_state(1)[0]) for s in audit.split("hard_task")[:args.worlds]]
    expected=[f"{g}-{l}-s{s}" for s in (3187,6947) for g in ("rotations","diagonal","stateless")
              for l in ("local","delta","ridge")]
    done=set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(done)<len(expected)+1:
            ready=[]
            for n in expected:
                hist=folder/(n+"-history.json")
                if n not in done and hist.is_file() and json.loads(hist.read_text())[-1]["step"]>=900:ready.append(n)
            if "visible-compartment" not in done:ready.append("visible-compartment")
            jobs=[]
            for n in ready:
                for f in ("linear","interaction","mixed"):
                    for d in ("continuous","disjoint_octants","components"):
                        controls=d=="continuous" and n.startswith("rotations") and n.endswith("3187")
                        jobs.append((folder/(n+".pt"),f,d,controls,seeds,dest))
            futures=[pool.submit(one,j) for j in jobs]
            for task in as_completed(futures):task.result()
            done.update(ready)
            if ready:print("checked",*ready,flush=True)
            if len(done)<len(expected)+1:time.sleep(5)
    _write(dest/"complete.json",dict(models=sorted(done),lives=len(seeds)))


if __name__=="__main__":main()
