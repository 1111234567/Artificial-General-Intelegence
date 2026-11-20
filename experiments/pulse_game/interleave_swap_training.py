"""Swap the training world streams of the two large paired-transfer outliers and a neighbor."""

from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_revisit_train as study, interleave_capacity as cap
from .interleave_interference import DATA,EVIDENCE,ROOT


def run(cohort,out):
    torch.set_num_threads(1)
    independent=cohort=="confirmation"
    src=DATA/("retention-objective-confirmation" if independent else "retention-objective")
    warm=DATA/"replication-births" if independent else ROOT/"data/local-trace-20261119/local"
    expected_manifest=json.loads((src/"run.json").read_text())
    cfg=study.Config(**expected_manifest["config"])
    births=[814,815,816] if independent else [734,735,736]
    tape_ids=[births[1],births[0],births[2]]
    dest=out/cohort;dest.mkdir(parents=True,exist_ok=True)
    plan=dict(cohort=cohort,births=births,tape_births=tape_ids,config=asdict(cfg),source=str(src),
              schedule="Only the training data RNG keys of the first two births are exchanged. Third root retains "
                       "its original position, initialization, data and optimizer; frozen weights must match exactly. "
                       "Targeted diagnosis on the two previously observed transfer failures; not a population-wide estimate.",
              sources={str(b):sha256(study.old_fit.checkpoint_path(warm,"odor","active",b,8000).read_bytes()).hexdigest() for b in births})
    m=dest/"run.json"
    if m.exists() and json.loads(m.read_text())!=plan:raise ValueError("Swap protocol changed")
    study._write_json(m,plan)
    for arm in study.ARMS:
        graph,rule=study._warm_graph(warm,births)
        optimizer=torch.optim.Adam((graph.weights,graph.bias,*rule.parameters()),lr=cfg.lr)
        folder=dest/arm;folder.mkdir(exist_ok=True)
        resume=folder/"resume.pt"
        step,trace=0,[]
        if resume.exists():
            d=torch.load(resume,weights_only=True,map_location="cpu")
            with torch.no_grad():
                graph.weights.copy_(d["w"]);graph.bias.copy_(d["b"])
            rule.load_state_dict(d["rule"]);optimizer.load_state_dict(d["optimizer"])
            step,trace=d["step"],d["trace"].tolist()
        if step==cfg.updates:continue
        started=perf_counter();first=step
        for step in range(first+1,cfg.updates+1):
            tape=study.make_tape(cfg,tape_ids,step)
            metrics=study._fit_step(graph,rule,optimizer,tape,arm,cfg)
            trace.append(metrics.tolist())
            if step%500==0 or step==cfg.updates:
                obj=dict(step=step,w=graph.weights.detach(),b=graph.bias.detach(),rule=rule.state_dict(),
                         trace=torch.as_tensor(trace),optimizer=optimizer.state_dict())
                study._atomic(resume,lambda p:torch.save(obj,p))
            if step%1000==0 or step in (1,cfg.updates):
                print(cohort,arm,step,round((perf_counter()-started)/(step-first),3),flush=True)
            if step in (1000,4000,8000):
                for i,b in enumerate(births):
                    study.old_fit.save_checkpoint(dest,"odor",arm,b,step,graph,rule,i)
    results={};checks={}
    seed=9124431 if independent else 871831
    tape=cap.make_tape("odor",seed,192,16)
    for arm in study.ARMS:
        cells={}
        for i,b in enumerate(births):
            g,r=study.old_fit.load_checkpoint(dest,"odor",arm,b,8000)
            original,original_rule=study.old_fit.load_checkpoint(src,"odor",arm,b,8000)
            same=all(torch.equal(getattr(g,k),getattr(original,k)) for k in cap.GRAPH_FIELDS)
            same=bool(same and all(torch.equal(v,original_rule.state_dict()[k]) for k,v in r.state_dict().items()))
            if i==2 and not same:raise AssertionError(f"The untouched control birth diverged: {b}/{arm}")
            comparison={}
            for name,body,writer in (("original",original,original_rule),("swapped",g,r)):
                raw=cap.evaluate(body,writer,tape,cap.CAPACITIES,snapshot_worlds=0)
                comparison[name]={str(k):cap._metrics(raw,0,j,k)[0] for j,k in enumerate(cap.CAPACITIES)}
                p=dest/f"{arm}-{b}-{name}.npz"
                study._write_raw(p,side=raw["side"],logit=raw["logit"],truth=raw["right_value"],law=raw["law"])
                checks[p.name]=sha256(p.read_bytes()).hexdigest()
            cells[b]=dict(same_parameters_as_original=same,rates=comparison)
        results[arm]=cells
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    study._write_json(EVIDENCE/f"training-stream-swap-{cohort}.json",dict(plan=plan,results=results,raw_sha256=checks))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","confirmation"),required=True)
    p.add_argument("--out",type=Path,default=DATA/"training-stream-swap")
    a=p.parse_args();run(a.cohort,a.out)
