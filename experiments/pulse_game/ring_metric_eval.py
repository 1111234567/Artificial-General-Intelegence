"""Frozen four-choice outcome-predictor audits using unseen whole-map permutations."""

import os
for _k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_k]="1"

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write
from . import research_audit as audit,ring_metric as m,ring_world as world
from .ring_sensors import ChemicalInterface
from .ring_metric_train import get_seeds,map_parity
from .ring_audit import life_metrics,AtBite


class Features:
    def __init__(self,base,skip,silent):
        self.base,self.skip,self.silent=base,skip,silent
    def reset(self,n):self.base.reset(n)
    def step(self,*args):
        features=self.base.step(*args)
        if self.silent=="raw":
            features[:,:self.skip]=0
        elif self.silent=="spikes":
            features[:,self.skip:]=0
        return features


def load(path):
    d=torch.load(path,map_location="cpu",weights_only=True)
    model=m.Metric(m.Settings(**d["settings"]));model.load_state_dict(d["model"])
    return model.eval()


def run(cfg,params,seeds,style,exposure,appetite,*,detailed):
    t=world.make_tape(seeds)
    def factory(learn=True,cut=False,no_fast=False,mask=None):
        p=m.MetricForager(cfg,params,seeds,appetite=appetite,learn=learn,
                          cut_recurrence=cut,cut_fast=no_fast)
        if mask is not None:p.sense=Features(p.sense,world.OBS if cfg.raw_skip else 0,mask)
        return p if style=="onehot" else ChemicalInterface(p,seeds,style)
    def play(p,**kw):return world.rollout(p,t,cue_ticks=exposure,**kw)
    primary=play(factory())
    swap=np.arange(t.n)^1
    donor=primary.positive[swap]-primary.negative[swap]
    data={"true":life_metrics(t,primary),"no_write":life_metrics(t,play(factory(learn=False))),
          "yoke_at_bite":life_metrics(t,play(AtBite(factory(),donor)))}
    if detailed:
        data.update(no_read=life_metrics(t,play(factory(no_fast=True))),
            no_feedback=life_metrics(t,play(factory(),feedback="none")),
            rec_cut=life_metrics(t,play(factory(cut=True))),
            raw_muted=life_metrics(t,play(factory(mask="raw"))),
            spikes_muted=life_metrics(t,play(factory(mask="spikes"))),
            retry=life_metrics(t,play(factory(),retry=True)))
        random_goal=np.stack([np.random.default_rng(np.random.SeedSequence([s,68811])).integers(
            4,size=t.trials,dtype=np.int8) for s in seeds])
        sham=replace(t,goal=random_goal)
        data["unpredictable"]=life_metrics(sham,world.rollout(factory(),sham,cue_ticks=exposure))
    summary={k:{metric:audit.estimates(vals,clusters=2) for metric,vals in v.items()} for k,v in data.items()}
    contrasts={k:audit.estimates(np.subtract(data["true"]["late"],v["late"]),clusters=2) for k,v in data.items() if k!="true"}
    return {"summary":summary,"true_minus_late":contrasts,"per_world":data}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint",type=Path);ap.add_argument("--hash",required=True);ap.add_argument("--name",required=True)
    ap.add_argument("--worlds",type=int,default=128);ap.add_argument("--seed",type=int,default=1262227)
    ap.add_argument("--exposures",type=int,nargs="+",default=[40]);ap.add_argument("--styles",nargs="+",default=["onehot","anchored","novel","noisy"])
    ap.add_argument("--detailed",action="store_true")
    ap.add_argument("--appetite",nargs=4,type=float,metavar=("THRESH","GAIN","RANDOM","URGENCY"))
    args=ap.parse_args();torch.set_num_threads(1)
    if audit.fingerprint(args.checkpoint)!=args.hash:raise ValueError("Weights differ from frozen file")
    model=load(args.checkpoint);params=model.export()
    seeds=get_seeds(np.random.default_rng(args.seed),args.worlds,1)
    if not (map_parity(world.make_tape(seeds,trials=1))==1).all():raise ValueError("Map leakage")
    appetite=m.Appetite(*args.appetite) if args.appetite else m.Appetite()
    report={"model":str(args.checkpoint),"sha256":args.hash,"seeds":seeds,"A_map_parity":1,"train_A_map_parity":0,
            "appetite":vars(appetite),"conditions":{}}
    for style in args.styles:
        for exp in args.exposures:
            values=run(model.cfg,params,seeds,style,exp,appetite,detailed=args.detailed)
            name=f"{style}-{exp}"
            report["conditions"][name]=values
            _write(audit.OUT/(args.name+".json"),report)
            print(name,{k:round(v["late"]["mean"],3) for k,v in values["summary"].items()},flush=True)


if __name__=="__main__":
    main()
