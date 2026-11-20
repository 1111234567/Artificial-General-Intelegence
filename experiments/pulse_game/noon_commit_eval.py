"""Measure voluntary plasticity against matched commits in the same tasted life."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import noon_commit as study,noon_market_stats as stats


@torch.no_grad()
def measure(g,r,gate,t,arm):
    view,e,_,h=study.prefix(g,r,t)
    p=study.probability(gate,arm,e,h)
    open_gate=t["gate_uniform"]<p
    branches=study.endings(view,e,t,cold=True)
    pay=branches["pay"].bool()
    correct=branches["cold"]==t["new_law"][:,None,:]*t["query_orientation"][:,None,:]
    n=len(p);i=torch.arange(n);review=t["order"][:,2];other=1-review
    rec=branches["receipt"]
    # Reversing the event changes its taste and subsequent state, but the actual
    # previously chosen side of this contact must precede that observation.
    twin=i^1
    torch.testing.assert_close(h[:,-1,0],h[twin,-1,0],rtol=0,atol=0)
    if arm=="always" and not open_gate.all():raise ValueError("Always-write arm skipped an update")
    weighted=t["weight"].reshape(-1,4,2)
    mixed=(open_gate.float().reshape_as(weighted)*weighted).sum(2,keepdim=True)/weighted.sum(2,keepdim=True)
    fraction=mixed.expand_as(weighted).reshape(-1)
    selections={"native":open_gate.float(),"conditional_expectation":p,"always":torch.ones_like(p),
                "skip":torch.zeros_like(p),"same_count":fraction,"opposite_receipt":open_gate[twin].float()}
    behavior=dict(serial=pay.all(2),serial_items=pay.float().mean(2),both=correct.all(2),
                  reviewed=correct[i[:,None],torch.arange(2)[None],review[:,None]],
                  unreviewed=correct[i[:,None],torch.arange(2)[None],other[:,None]])
    vals=dict(commit=open_gate.float(),probability=p,signed_confirmation=h[:,-1,0]*h[:,-1,1]*(h[:,-1,2]/2).tanh(),
              actual_weight_write=open_gate.float()*rec.weight_delta.abs().flatten(1).sum(1),
              actual_bias_write=open_gate.float()*rec.bias_delta.abs().sum(1),
              births=open_gate.float()*(rec.birth>=0),retirements=open_gate.float()*(rec.retired>=0))
    for k,out in behavior.items():
        z=out.float();delta=z[:,1]-z[:,0]
        for tag,choice in selections.items():vals[f"{tag}/{k}"]=z[:,0]+choice*delta
        vals["write_minus_skip/"+k]=delta
        for tag in ("always","skip","same_count","opposite_receipt"):
            vals[f"native_minus_{tag}/{k}"]=(selections["native"]-selections[tag])*delta
    flip=t["reversed"];conflict=t["old_law"][:,0]!=t["old_law"][:,1]
    subsets={"all":torch.ones_like(flip),"reversal":flip,"stable":~flip,
             "initial_opposed":conflict,"stable_opposed":~flip&conflict,"stable_agree":~flip&~conflict,
             "new_opposed":flip&~conflict,"resolved_conflict":flip&conflict}
    metrics={}
    for tag,mask in subsets.items():
        w=(t["weight"]*mask).reshape(-1,8)
        if not torch.all(w.sum(1)>0):raise ValueError("Missing commit stratum")
        for k,v in vals.items():metrics[tag+"/"+k]=((v.reshape_as(w)*w).sum(1)/w.sum(1)).numpy()
    raw=dict(study=h.numpy(),gate_probability=p.numpy(),gate=open_gate.numpy(),serial=branches["serial_actions"].numpy().astype(np.int8),
             cold=branches["cold"].numpy().astype(np.int8),cold_truth=(t["new_law"]*t["query_orientation"]).numpy().astype(np.int8),
             order=t["order"].numpy().astype(np.int8),target=t["value"].numpy().astype(np.int8),weight=t["weight"].numpy(),
             write_size=rec.weight_delta.abs().flatten(1).sum(1).numpy())
    return raw,metrics


def run(cohort,step,arms,worlds,label=None,offset=0,stress=False):
    torch.set_num_threads(1);ids=study.base.births(cohort);seed=(9582711 if cohort=="original" else 9582749)+offset
    label=label or (("stress-" if stress else "")+cohort)
    out=study.base.DATA/"commit-eval"/label/str(step);out.mkdir(parents=True,exist_ok=True)
    settings=[(theta,wait) for theta in ((0,5,30,60,90) if stress else (15,45,75)) for wait in ((4,64) if stress else (4,))]
    sums={};hashes={};code=sha256(Path(__file__).read_bytes()).hexdigest()
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    for arm in arms:
        entries={};gf=study.OUT/cohort/arm/f"gate-{step}.npz"
        with np.load(gf) as z:gate=torch.from_numpy(z["gate"])
        gd=digest(gf);hashes[gf.relative_to(study.base.DATA).as_posix()]=gd
        for i,b in enumerate(ids):
            path=study.base.old.checkpoint_path(study.OUT/cohort,"odor",arm,b,step);bd=digest(path)
            g,r=study.base.old.load_checkpoint(study.OUT/cohort,"odor",arm,b,step);hashes[path.relative_to(study.base.DATA).as_posix()]=bd
            f=out/f"{arm}-{b}.npz"
            if not f.exists():
                raw={};all_metrics={}
                for angle,delay in settings:
                    t=study.tape(seed,0,0,worlds,domain=778871,theta=angle,delay=delay)
                    data=[];measured=[]
                    for j in range(0,worlds,32):
                        a,v=measure(g,r,gate[i:i+1],{k:z[8*j:8*(j+32)] for k,z in t.items()},arm)
                        data.append(a);measured.append(v)
                    raw.update({f"{angle}/{delay}/{k}":np.concatenate([a[k] for a in data]) for k in data[0]})
                    all_metrics.update({f"{angle}/{delay}/{k}":np.concatenate([v[k] for v in measured]) for k in measured[0]})
                if not stress:
                    for k in measured[0]:all_metrics["pooled/"+k]=np.mean([all_metrics[f"{ang}/4/{k}"] for ang in (15,45,75)],0)
                study.base.storage._write_raw(f,**raw,**{"metric/"+k:v for k,v in all_metrics.items()},body=bd,gate=gd,evaluator=code,worlds=worlds,seed=seed)
            with np.load(f) as z:
                if (str(z["body"]),str(z["gate"]),str(z["evaluator"]),int(z["worlds"]),int(z["seed"]))!=(bd,gd,code,worlds,seed):
                    raise ValueError("Commit cache identity mismatch")
                for k in z:
                    if k.startswith("metric/"):entries.setdefault(k[7:],[]).append(z[k])
            hashes[f.relative_to(study.base.DATA).as_posix()]=digest(f);print(cohort,step,arm,b,flush=True)
        sums[arm]={k:np.stack(v) for k,v in entries.items()}
    report=dict(births=ids,seed=seed,worlds=worlds,step=step,hashes=hashes,
                results=stats.summarize(sums,worlds,ids,seed+17,[("state","constant"),("confirmation","constant"),
                                                        ("state","always"),("confirmation","always"),("constant","always")]),
                note="Paired weight .25 for a reversal, .75 for stable. Forced choices use the same truly sampled review "
                     "action and acquired taste in that branch. Native minus count-matched gate shuffles only within "
                     "each geometry and the exact initial A/B private assignment, using the correct reversal probabilities. "
                     "Cold two-question probes fork one state; serial tests perform two actions with no intervening taste.")
    study.base.storage._write_json(study.base.EVIDENCE/f"commit-{label}-{step}.json",report)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--step",type=int,required=True);p.add_argument("--arms",nargs="+",choices=study.ARMS,default=study.ARMS)
    p.add_argument("--worlds",type=int,default=192);p.add_argument("--label");p.add_argument("--offset",type=int,default=0)
    p.add_argument("--stress",action="store_true");run(**vars(p.parse_args()))
