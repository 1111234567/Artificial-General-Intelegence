"""Spatially bind the same sensed familiarity to a physical action without changing graph size."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import noon_coverage as cov


OUT=cov.ticket.DATA/"option-binding";PLAN=cov.ticket.EVIDENCE/"option-binding-plan.json"
ARMS=("local","opposite","detached")


def instrument(t,value,gain,role):
    if role=="bilateral":return cov.instrument(t,value,gain)
    marked_right=t["offer_right"][:,0]>t["offer_left"][:,0]
    right=marked_right if role=="local" else ~marked_right if role=="opposite" else t["coin"]
    signal=(2.**.5)*value*gain
    left_add=torch.zeros_like(t["offer_left"]);right_add=torch.zeros_like(left_add)
    left_add[:,1]=signal*(~right);right_add[:,1]=signal*right
    return {**t,"offer_left":t["offer_left"]+left_add,"offer_right":t["offer_right"]+right_add}


def update(g,r,k,opt,t,role):
    params=(g.weights,g.bias,*r.parameters(),k)
    for precise in (False,True):
        opt.zero_grad(set_to_none=True)
        if precise:
            m=cov.ticket.net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double();coupling=torch.nn.Parameter(k.detach().double())
        else:m,rule,coupling=g,r,k
        view,s,prefix,trace,_=cov.physical_prefix(m,rule,t,"evidence")
        signal=cov.projection(trace,t["preview"])
        u=instrument(t,signal,coupling.flatten().repeat_interleave(len(signal)//len(g.bias)),role)
        res=cov.ticket.finish(view,s,u,"native")
        pre=cov.ticket.centered(res["utility"],len(g.bias),4);post=cov.ticket.centered(res["gross"],len(g.bias),4)
        loss=-((prefix+res["route_score"]).reshape_as(pre)*pre+res["following_score"].reshape_as(post)*post).mean((1,2)).sum()
        if not loss.isfinite():continue
        loss.backward();src=(m.weights,m.bias,*rule.parameters(),coupling)
        norm=cov.ticket.storage._gradients(src,2.)
        if norm is None:continue
        if precise:
            for dest,source in zip(params,src,strict=True):dest.grad=None if source.grad is None else source.grad.to(dest.dtype)
        opt.step()
        stats=[res[name].float().reshape(len(g.bias),-1).mean(1) for name in ("gross","utility","buy")]
        return torch.stack((*stats,norm,k.detach().flatten(),torch.full_like(norm,precise)),1).detach()
    raise FloatingPointError("Nonfinite bound-sensor gradient")


def fit(cohort,role):
    torch.set_num_threads(1);ids=cov.ticket.births(cohort);seed=270551 if cohort=="original" else 270571
    path=OUT/cohort;folder=path/role;folder.mkdir(parents=True,exist_ok=True)
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    meta=dict(births=ids,role=role,seed=seed,protocol=digest(PLAN),steps=8000,lr=.003,training_worlds=4,
              # Same siblings, tapes, clocks and published parents as the bilateral trial.
              parents={str(b):digest(cov.ticket.old.checkpoint_path(cov.ticket.DATA/"tickets"/cohort,"odor","native",b,8000)) for b in ids},
              source={Path(p).name:digest(p) for p in (__file__,cov.__file__,cov.ticket.__file__,cov.ticket.net.__file__)})
    manifest=folder/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=meta:raise ValueError("Bound sensory source changed")
    if not manifest.exists():cov.ticket.storage._write_json(manifest,meta)
    key=digest(manifest);g,r=cov.inherited(cohort);gain=torch.nn.Parameter(torch.zeros(len(ids),1))
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters(),gain),lr=.003)
    resume=folder/"resume.pt";step=0;trace=[]
    if resume.exists():
        obj=torch.load(resume,weights_only=True,map_location="cpu")
        if obj["plan"]!=key:raise ValueError("Wrong bound sensory optimizer")
        with torch.no_grad():g.weights.copy_(obj["weights"]);g.bias.copy_(obj["bias"]);gain.copy_(obj["gain"])
        r.load_state_dict(obj["rule"]);opt.load_state_dict(obj["optimizer"]);step=obj["step"];trace=obj["trace"].tolist()
    def save():
        if step in cov.STAGES:
            for i,b in enumerate(ids):cov.ticket.old.save_checkpoint(path,"odor",role,b,step,g,r,i)
            cov.ticket.storage._write_raw(folder/f"gain-{step}.npz",gain=gain.detach().numpy(),births=ids)
        obj=dict(plan=key,step=step,weights=g.weights.detach(),bias=g.bias.detach(),present=g.present,rule=r.state_dict(),
                 gain=gain.detach(),optimizer=opt.state_dict(),trace=torch.tensor(trace))
        cov.ticket.storage._atomic(resume,lambda f:torch.save(obj,f))
    if not resume.exists():save()
    first=step;t0=perf_counter()
    for step in range(first+1,8001):
        t=cov.ticket.join([cov.ticket.tape(seed,b,step,4,domain=274111) for b in ids])
        trace.append(update(g,r,gain,opt,t,role).tolist())
        if step%250==0:
            save();print(cohort,role,step,np.array(trace[-100:])[:,:,:3].mean((0,1)).round(3),
                         round((perf_counter()-t0)/(step-first),4),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--role",choices=ARMS,required=True);a=p.parse_args();fit(a.cohort,a.role)
