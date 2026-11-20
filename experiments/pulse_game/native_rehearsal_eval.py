"""Held-out real actions under paired interventions on a single continuous graph."""

import argparse
from dataclasses import fields
import json

import numpy as np
import torch

from . import native_rehearsal_fit as fit
from . import native_rehearsal_graph as live
from . import noon_market_stats as stats


ROOTS=384
SEEDS=(81921671,81921701)
MODES=("normal","held","no_quiet_write","quiet_hidden_only","quiet_weights_only",
       "donor_hidden","donor_weights","donor_both","cut_return")


def load(kind,birth):
    path=fit.DATA/kind/f"seed-{birth}.pt"
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if saved["step"]!=fit.STEPS or saved["signature"]["graph"]!=fit.digest(live.__file__):
        raise ValueError(f"Incorrect final checkpoint: {path}")
    model=live.NativeGraph(birth,saved["layout"])
    model.load_state_dict(saved["state"])
    return model.requires_grad_(False),path


def subset(e,start,end):
    return live.Episode(**{f.name:getattr(e,f.name)[start:end] for f in fields(e)})


@torch.no_grad()
def evaluate(kind,birth):
    torch.set_num_threads(1)
    model,path=load(kind,birth)
    for seed in SEEDS:
        whole=live.scenes(seed,ROOTS,evaluation=True)
        rows=[]
        for i in range(0,len(whole.laws),256):
            e=subset(whole,i,i+256)
            n=len(e.laws);twin=torch.arange(n)^4
            assert torch.equal(e.laws[twin,0],-e.laws[:,0]) and torch.equal(e.laws[twin,1:],e.laws[:,1:])
            _,begin,_,_=live.support(model,e)
            end,_,_=live.silence(model,begin)
            cases={"normal":end,"held":begin,
                   "quiet_hidden_only":live.State(end.h,begin.fast,end.trace),
                   "quiet_weights_only":live.State(begin.h,end.fast,begin.trace),
                   "donor_hidden":live.State(end.h[twin],end.fast,end.trace[twin]),
                   "donor_weights":live.State(end.h,end.fast[twin],end.trace),
                   "donor_both":live.State(end.h[twin],end.fast[twin],end.trace[twin])}
            frozen=begin;zero=begin.h.new_zeros((n,10))
            for _ in range(12):
                advanced,_=model.tick(frozen,zero)
                frozen=live.State(advanced.h,begin.fast,advanced.trace)
            cases["no_quiet_write"]=frozen
            cases["cut_return"]=live.silence(model,begin,cut=True)[0] if model.width==40 else end
            group={}
            for tag,state in cases.items():
                choice,_,pay,margin=live.queries(model,state,e)
                group[tag+"/action"]=choice.to(torch.int8).numpy()
                group[tag+"/pay"]=pay.to(torch.int8).numpy()
                group[tag+"/signed_margin"]=margin.numpy()
            group["quiet_l1"]=(end.fast-begin.fast).abs().sum((1,2)).numpy()
            group["quiet_h"]=(end.h-begin.h).abs().mean(1).numpy()
            rows.append(group)
        raw={k:np.concatenate([row[k] for row in rows]) for k in rows[0]}
        # Reconstruct the paid result from private world signs and the two actual choices.
        order=whole.first_query
        angles=whole.angle.numpy()
        identity=whole.laws[:,[0,2]].numpy()
        orient=whole.orientation[:,3:].gather(1,torch.stack((order,1-order),1)).numpy()
        for tag in MODES:np.testing.assert_array_equal(raw[tag+"/pay"],raw[tag+"/action"]*identity*orient)
        active=int(model.mask.sum())
        fit.storage._write_raw(fit.DATA/"eval"/str(seed)/f"{kind}-{birth}.npz",**raw,
                               law=identity,orientation=orient,angle=angles,first=order.numpy(),mask_edges=active,
                               # Each active address supplies a conductance and a learned modulation coefficient.
                               active_trainable=active*2+model.width*5,stored_trainable=sum(p.numel() for p in model.parameters()),
                               seed=seed,kind=kind,birth=birth,fit=fit.digest(path),source=fit.digest(__file__))


def summarize():
    out={}
    for seed in SEEDS:
        banks={};files={};edges={};mutable={}
        for kind in fit.KINDS:
            records={p:[] for p in MODES}
            for birth in fit.BIRTHS:
                path=fit.DATA/"eval"/str(seed)/f"{kind}-{birth}.npz";files[path.name]=fit.digest(path)
                with np.load(path) as z:
                    if str(z["source"])!=fit.digest(__file__) or str(z["fit"])!=fit.digest(fit.DATA/kind/f"seed-{birth}.pt"):
                        raise ValueError(f"Changed native evidence in {path}")
                    edges.setdefault(kind,[]).append(int(z["mask_edges"]))
                    mutable.setdefault(kind,[]).append(float(z["quiet_l1"].mean()))
                    angle=z["angle"].reshape(ROOTS,16)[:,0]
                    agree=(z["law"][:,0]==z["law"][:,1]).reshape(ROOTS,16)
                    for tag in MODES:
                        success=z[tag+"/pay"].reshape(ROOTS,16,2)>0
                        data={}
                        for relation,mask in (("conflict",~agree),("agree",agree),("all",np.ones_like(agree))):
                            a,b=np.moveaxis(success[mask].reshape(ROOTS,-1,2),2,0)
                            for name,cell in (("A",a),("B",b),("both",a&b)):
                                data[relation+"/"+name]=cell.mean(1)
                        records[tag].append(data)
            for tag,roots in records.items():
                banks[kind+"/"+tag]={key:np.stack([r[key] for r in roots]) for key in roots[0]}
        compare=[("copy/normal","single/normal"),("copy/normal","permuted/normal"),
                 ("guide_a/normal","copy/normal"),("guide_a/normal","guide_c/normal")]
        compare += [(kind+"/normal",kind+"/"+tag) for kind in fit.KINDS for tag in MODES if tag!="normal"]
        suites={}
        for name,mask in (("close",np.isin(angle,(15,30))),("all",np.ones_like(angle,dtype=bool)),("far",angle==90)):
            data={k:{field:v[:,mask] for field,v in val.items()} for k,val in banks.items()}
            suites[name]=stats.summarize(data,int(mask.sum()),fit.BIRTHS,seed+19,compare)
        out[str(seed)]=dict(results=suites,hashes=files,edges=edges,quiet_write_l1=mutable,
                            close_worlds=int(np.isin(angle,(15,30)).sum()),roots=ROOTS)
    fit.storage._write_json(fit.EVIDENCE/"behavior.json",out)
    for seed in SEEDS:
        z=out[str(seed)]["results"]["close"]
        print(seed)
        for name in [k+"/normal" for k in fit.KINDS]+[a+" minus "+b for a,b in compare[:4]]:
            print(name,[round(100*z[name]["conflict/"+k]["mean"],1) for k in ("A","B","both")])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kind",choices=fit.KINDS)
    p.add_argument("--birth",type=int,choices=fit.BIRTHS)
    a=p.parse_args()
    if a.kind:evaluate(a.kind,a.birth)
    else:summarize()
