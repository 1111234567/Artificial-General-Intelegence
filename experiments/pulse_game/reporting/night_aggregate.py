"""Combine fixed phenotypes within a held-out world before bootstrapping independent worlds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
from .. import research_audit as evidence
from experiments.seed_evolution.large_runner import _write


class Book:
    def __init__(self):self.data=defaultdict(dict)
    def add(self,key,birth,values):
        v=np.asarray(values,dtype=float)
        if v.ndim!=1 or not np.isfinite(v).all():raise ValueError((key,v.shape))
        name="/".join(key)
        if birth in self.data[name]:raise ValueError((name,birth))
        self.data[name][birth]=v
    def summary(self):
        buckets=defaultdict(list);out={}
        for key,models in self.data.items():
            arrays=np.stack(list(models.values()))
            buckets[arrays.shape[1]].append((key,arrays.mean(0)))
            out[key]=dict(mean=float(arrays.mean()),fixed_model_means={k:float(v.mean()) for k,v in models.items()},
                          independent_roots=arrays.shape[1])
        for size,seq in buckets.items():
            rng=np.random.default_rng(519841)
            weights=rng.multinomial(size,np.full(size,1/size),size=2048).astype(np.float64)/size
            for j in range(0,len(seq),160):
                rows=seq[j:j+160];vals=weights@np.stack([v for _,v in rows]).T
                ci=np.quantile(vals,[.025,.975],axis=0)
                for i,(key,_) in enumerate(rows):out[key]["ci95"]=ci[:,i].tolist()
        return out


def rows(flat,birth,groups,record,what,reference=None):
    keys=[k for k in what if k in next(iter(record.values()))]
    for arm,data in record.items():
        for m in keys:
            x=np.asarray(data[m],float)
            if x.ndim==1:
                flat.add((*groups,arm,m),birth,x)
                if reference is not None and arm!=reference:
                    flat.add((*groups,arm+"-minus-"+reference,m),birth,x-np.asarray(record[reference][m]))
            else:
                if x.shape[0]==4:
                    partitions={"same":.5*(x[0]+x[3]),"conflict":.5*(x[1]+x[2]),"balanced":x.mean(0),
                                "prior77":.385*(x[0]+x[3])+.115*(x[1]+x[2])}
                elif x.shape[0]==2:partitions={"balanced":x.mean(0)}
                else:raise ValueError(x.shape)
                for condition,v in partitions.items():
                    flat.add((*groups,arm,condition,m),birth,v)
                if reference is not None and arm!=reference:
                    diff=x-np.asarray(record[reference][m])
                    flat.add((*groups,arm+"-minus-"+reference,"balanced",m),birth,diff.mean(0))
                    if len(diff)==4:
                        flat.add((*groups,arm+"-minus-"+reference,"conflict",m),birth,(diff[1]+diff[2])*.5)


def collect(base):
    book=Book();seeds_per_study={};files=[]
    def listing(directory,pattern="*.json"):
        return sorted(p for p in (base/directory).glob(pattern) if p.stem not in ("complete","done","status"))
    def load(p,study):
        files.append(str(p.relative_to(base)))
        d=json.loads(p.read_text())
        seeds=d.get("seeds",d.get("roots",d.get("seed",[])))
        if study not in seeds_per_study:seeds_per_study[study]=seeds
        elif seeds and seeds_per_study[study]!=seeds:raise ValueError("Unpaired life groups: "+study)
        return d
    for p in listing("symmetry-policy-v1-sealed"):
        d=load(p,"symmetry_policy");model=d["settings"]["parent"];mode=d["settings"]["mode"]
        for case,arms in d["cases"].items():
            rows(book,model,("symmetry_policy",mode,case),arms,("first","both","contact","gain_even","gain_odd"),"parent")
    for p in listing("recurring-write-sealed"):
        d=load(p,"recurring_write");model=d["model"]["parent"];mode=str(d["model"]["mode"])
        for case,arms in d["cases"].items():
            rows(book,model,("recurring_write",mode,case),arms,("first","both","contact"),"one_per_contact")
    for p in listing("recurring-read-write-sealed"):
        d=load(p,"read_write")
        for case,arms in d["cases"].items():
            rows(book,d["parent"],("read_write",case),arms,("first","both","contact"),"parent_FIFO")
    for p in listing("recurring-identical-prefix"):
        d=load(p,"prefix_repair");case=f"d{d['protocol']['distinct']}-recur{d['protocol']['recurring']}"
        rows(book,d["parent"],("prefix_repair",str(d["writer"]),case),d["cases"],
             ("success","anchor_present","rule_direction","prior_rule","anchor_sign_intact","comparison_right"),"true")
    for tag,directory in (("habitat_delta","habitat-meta-sealed"),("habitat_rls","habitat-rls-sealed"),
                          ("unified","unified-transfer-sealed")):
        for p in listing(directory):
            d=load(p,tag)
            name=d["model"];kind,seed=name.rsplit("-s",1)
            case=d.get("setting",d.get("case"))
            criteria=("first","both","earlier_habitat","recent_habitat","A","B","contact","context_cosine","return_A","coverage",
                      "B_after_0","B_after_2","B_after_4","B_after_8","B_after_12","B_after_24","B_after_48")
            rows(book,seed,(tag,kind,case),d["rows"],criteria,"true")
    for p in listing("unified-physical-sealed"):
        d=load(p,"unified_physical");name=d["settings"]["model"];method=d["settings"]["mode"]
        for case,arms in d["cases"].items():
            rows(book,name,("unified_physical",method,case),arms,("first","both","return_A","coverage","B_after_0","B_after_2",
                "B_after_4","B_after_8","B_after_12","B_after_24","B_after_48","earlier_habitat"),"parent")
    return dict(method="Fixed birth/training seeds averaged within physical root, paired root multinomial bootstrap",
                coverage={k:len(v) for k,v in seeds_per_study.items()},rows=book.summary(),source_files=files)


def main():
    with_arg=argparse.ArgumentParser(description=__doc__);with_arg.parse_args()
    base=evidence.OUT;_write(base/"life-learning-summary.json",collect(base))


if __name__=="__main__":main()
