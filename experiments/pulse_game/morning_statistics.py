"""Summarize held-out life cohorts, retaining paired-world uncertainty and individual seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
from experiments.seed_evolution.large_runner import _write
from . import research_audit as audit
from .attractor_search import OUT


def read(path):return json.loads(path.read_text())


def interval(x,idx=None):
    x=np.asarray(x,dtype=np.float64)
    if idx is None:idx=np.random.default_rng(91331).integers(len(x),size=(2000,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[idx].mean(1),[.025,.975]).tolist(),n=len(x))


def mixture(path,out):
    records=defaultdict(list)
    for file in sorted(path.glob("*.json")):
        if file.stem in ("complete","summary"):continue
        data=read(file)
        model=data.get("candidate",data.get("model"))
        name=re.sub(r"-s\d+$","",model)
        f=data.get("family",data.get("rule"))
        d=data["distribution"]
        for branch,rows in data["rows"].items():
            for k,v in rows.items():
                ident=model+(f"-g{data['graph']}" if "graph" in data else "")
                records[(f,d,name,branch,k)].append((ident,data["seeds"],np.array(v)))
    arrays={}
    result={}
    for key,values in records.items():
        names,seeds,rates=zip(*values,strict=True)
        if any(seeds[0]!=s for s in seeds):raise ValueError(key)
        # Model training seeds and graph births are repeated measurements on
        # these same worlds; they do not multiply the statistical sample size.
        a=np.mean(rates,axis=0);arrays[key]=a
        result["/".join(key)]={**interval(a),"fixed_models":dict(zip(names,[float(v.mean()) for v in rates],strict=True))}
    delta={}
    for (f,d,name,branch,k),x in arrays.items():
        if branch!="true":continue
        for ck in ("no_write","cut_recurrence","cross","all","other_life","silence","context_hidden","silent_second"):
            match=arrays.get((f,d,name,ck,k))
            if match is not None:
                subset=records[(f,d,name,ck,k)]
                keys={i for i,_,_ in subset}
                pair=[a for i,_,a in records[(f,d,name,"true",k)] if i in keys]
                delta["/".join((f,d,name,"true-minus-"+ck,k))]=interval(np.mean(pair,axis=0)-match)
    _write(out,dict(estimator="equal-life mean after within-world averaging of fixed births and outer training seeds",
                    observations=result,within_model_differences=delta))


def ratio(x,n,idx):
    return (x[idx].sum(1)/n[idx].sum(1).clip(min=1))


def reversals():
    for directory,base in (("ring-self-evaluation-v1","slow"),("ring-local-unlearning-v1","no_reset")):
        folder=OUT/directory
        rate=defaultdict(list)
        for file in sorted((folder/"confirm").glob("*.json")):
            m=re.fullmatch(r"(.+)-g(\d+)-(stable|return|reverse)-n(\d+)",file.stem)
            if m is None:
                m=re.fullmatch(r"(.+)-(stable|return|reverse)-n(\d+)-g(\d+)",file.stem)
                agent,reg,noise,birth=m.groups()
            else:agent,birth,reg,noise=m.groups()
            data=read(file)
            for k,v in data.items():
                if not isinstance(v,dict) or "hit" not in v:continue
                rate[(agent,reg,noise,k)].append((int(birth),np.asarray(v["hit"]),np.asarray(v["count"])))
        table={};paired={}
        d={}
        for key,rs in rate.items():
            births,hits,counts=zip(*rs,strict=True)
            h,n=np.mean(hits,axis=0),np.mean(counts,axis=0)
            idx=np.random.default_rng(925).integers(len(n),size=(2000,len(n)))
            distribution=ratio(h,n,idx)
            d[key]=(float(h.sum()/n.sum()),distribution)
            table["/".join(key)]=dict(mean=d[key][0],ci95=np.quantile(distribution,[.025,.975]).tolist(),
                  graph_means={str(b):float(hh.sum()/nn.sum()) for b,hh,nn in rs},lives=len(n))
        for (a,reg,noise,k),(p,ci) in d.items():
            b=d[(base,reg,noise,k)]
            paired["/".join((a,reg,noise,k))]=dict(difference=p-b[0],ci95=np.quantile(ci-b[1],[.025,.975]).tolist())
        _write(audit.OUT/(directory+"-summary.json"),dict(native=table,paired_vs=base,differences=paired))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("which",choices=["volatility","mixture","flash"])
    args=ap.parse_args()
    if args.which=="volatility":reversals();return
    name="mixture-visible-independent" if args.which=="mixture" else "mixture-circuit-v1-external"
    mixture(audit.OUT/name,audit.OUT/(name+"-summary.json"))


if __name__=="__main__":main()
