"""Report initialization and world uncertainty, with all preregistered births in the denominator."""
import json
from pathlib import Path
import numpy as np
from . import research_audit as audit
from .attractor_search import OUT
from experiments.seed_evolution.large_runner import _write


def crossed(matrix):
    x=np.asarray(matrix,np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError(x.shape)
    n,m=x.shape;rng=np.random.default_rng(735913)
    births=rng.multinomial(n,np.full(n,1/n),size=4096).astype(np.float64)/n
    worlds=rng.multinomial(m,np.full(m,1/m),size=4096).astype(np.float64)/m
    ci=np.quantile(((births@x)*worlds).sum(1),[.025,.975]).tolist()
    val=x.mean(1)
    return dict(mean=float(val.mean()),ci95=ci,per_birth=val.tolist(),birth_range=[float(val.min()),float(val.max())],
                births=n,world_roots=m)


def summarize_replicas(root):
    reg=json.loads((root/"registered.json").read_text());seedlist=reg["fixed_birth_seeds"];sets={};collisions={}
    def series(key,line):sets.setdefault(key,[]).append(line)
    for seed in seedlist:
        folder=root/f"birth-{seed}"
        data=json.loads((folder/"external.json").read_text());supp=json.loads((folder/"always_on.json").read_text())
        cols={k:x["intersection"] for k,x in supp["training_seed_disjoint"].items() if x["intersection"]}
        if cols:collisions[str(seed)]=cols
        for setting,cohort in data["tests"].items():
            for arm,v in {**cohort,"always_on":supp["always_on"][setting]}.items():
                for k in ["B_after_0","B_after_2","B_after_4","B_after_12","B_after_48","return_A","both","coverage"]:
                    if k not in v:continue
                    x=np.asarray(v[k],float)
                    for typ,idx in (("same",[0,3]),("conflict",[1,2])):
                        series(f"{setting}/{arm}/{typ}/{k}",x[idx].mean(0))
            old=cohort["pre_action_training"];now=supp["always_on"][setting]
            for k in ("return_A","both"):
                delta=np.asarray(now[k])-np.asarray(old[k])
                series(f"{setting}/always_minus_initial/conflict/{k}",delta[[1,2]].mean(0))
        long=data["continuous"]
        for region in ("old_B","stable_A","change_1_1","change_3_4","change_9_16","return_3_4","whole_after_warmup"):
            for step in ("pre","post"):
                rates={}
                for arm,metrics in long.items():
                    count=np.asarray(metrics[region]["count"]);hits=np.asarray(metrics[region][step])
                    if np.any(count==0):raise ValueError(("Missing opportunities",region))
                    rates[arm]=hits/count;series(f"change_stream/{arm}/{region}/{step}",rates[arm])
                series(f"change_stream/reopen_minus_fixed/{region}/{step}",rates["reopen"]-rates["fixed_RLS"])
    return dict(seeds=seedlist,procedure=reg,training_evaluation_intersections=collisions,
                estimator="crossed bootstrap resampling 12 developmental starts and independent environmental roots",
                rows={k:crossed(v) for k,v in sets.items()})


def main():
    dest=OUT/"morning-20261119"
    _write(dest/"independent-birth-statistics.json",summarize_replicas(OUT/"independent-lifelong-births-v1"))


if __name__=="__main__":main()
