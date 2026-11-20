"""Open one sensory coactivation measurement to the native recurrent decision circuit."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import noon_ticket as ticket


OUT=ticket.DATA/"coverage";PLAN=ticket.EVIDENCE/"coverage-plan.json"
ARMS=("evidence","exposure","dose","plain")
STAGES=(0,2000,4000,8000)


def moments(v):
    x,y,z=v.unbind(-1)
    return torch.stack((x*x,y*y,z*z,x*y,x*z,y*z),-1)


def record(history,stimuli,*,every_exposure=False):
    selected=stimuli*history[:,:,0,None]
    unit=torch.nn.functional.normalize(selected,dim=2)
    strength=torch.ones_like(history[:,:,1]) if every_exposure else history[:,:,1].abs()
    return (moments(unit)*strength[:,:,None]).sum(1)


def projection(trace,query):
    p=moments(torch.nn.functional.normalize(query,dim=1))
    return (p[:,:3]*trace[:,:3]).sum(1)+2*(p[:,3:]*trace[:,3:]).sum(1)


def cue(trace,query,arm):
    if arm=="plain":return torch.zeros_like(query[:,0])
    if arm=="dose":return trace[:,:3].sum(1)/3
    return projection(trace,query)


def instrument(t,value,strength):
    offset=torch.zeros_like(t["offer_left"])
    offset[:,1]=value.to(offset)*strength.to(offset)
    return {**t,"offer_left":t["offer_left"]+offset,"offer_right":t["offer_right"]+offset}


def physical_prefix(g,r,t,arm):
    lanes=len(t["price"]);view=r.expand(lanes//len(g.bias));state=ticket.net.start(g,lanes//len(g.bias))
    trace=g.bias.new_zeros((lanes,6));score=0.;visits=[]
    study_gaps=t.get("study_gaps",t["gaps"]);study_draws=t.get("study_draws",t["draws"])
    for i in range(t["study"].shape[1]):
        state=ticket.net.wait(view,state,study_gaps[:,i],max_ticks=6)
        stimulus=t["study"][:,i]
        state,part,side,taste=ticket.encounter(view,state,stimulus,t["study_value"][:,i],study_draws[:,i])
        selected=stimulus*side[:,None]
        feature=moments(torch.nn.functional.normalize(selected,dim=1))
        amplitude=torch.ones_like(taste) if arm=="exposure" else taste.abs()
        trace=trace+feature*amplitude[:,None]
        score=score+part;visits.append(torch.stack((side,taste),1))
    state=ticket.net.wait(view,state,t["gaps"][:,-2],max_ticks=6)
    x=t["preview"]
    state=ticket.net._advance(view,state,ticket.net.native.sensory(state.graph,-x,x),ticket.net.REACTION_TICKS)
    return view,state,score,trace,torch.stack(visits,1)


def inherited(cohort):
    ids=ticket.births(cohort)
    models=[ticket.old.load_checkpoint(ticket.DATA/"tickets"/cohort,"odor","native",b,8000) for b in ids]
    values=[torch.cat([getattr(g,k) for g,_ in models]) for k in ticket.capacity.GRAPH_FIELDS]
    g=ticket.net.Graph(torch.nn.Parameter(values[0]),torch.nn.Parameter(values[1]),values[2])
    r=ticket.net.Rule(ids)
    r.load_state_dict({name:torch.cat([rr.state_dict()[name] for _,rr in models]) for name in r.state_dict()})
    return g,r


def objective(g,r,gain,t,worlds,arm):
    view,state,score,trace,_=physical_prefix(g,r,t,arm)
    sig=cue(trace,t["preview"],arm)
    augmented=instrument(t,sig,gain.flatten().repeat_interleave(len(sig)//len(g.bias)))
    result=ticket.finish(view,state,augmented,"native")
    n=len(g.bias)
    before=ticket.centered(result["utility"],n,worlds)
    after=ticket.centered(result["gross"],n,worlds)
    loss=-((score+result["route_score"]).reshape_as(before)*before+
           result["following_score"].reshape_as(after)*after).mean((1,2)).sum()
    metrics=torch.stack([result[k].float().reshape(n,-1).mean(1) for k in ("gross","utility","buy")],1)
    return loss,metrics


def update(g,r,gain,opt,t,worlds,arm):
    params=(g.weights,g.bias,*r.parameters(),gain)
    for precise in (False,True):
        opt.zero_grad(set_to_none=True)
        if precise:
            m=ticket.net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double();k=torch.nn.Parameter(gain.detach().double())
        else:m,rule,k=g,r,gain
        loss,stats=objective(m,rule,k,t,worlds,arm)
        if not loss.isfinite():continue
        loss.backward()
        grads=(m.weights,m.bias,*rule.parameters(),k)
        norm=ticket.storage._gradients(grads,2.)
        if norm is None:continue
        if precise:
            for a,b in zip(params,grads,strict=True):a.grad=None if b.grad is None else b.grad.to(a.dtype)
        opt.step()
        return torch.cat((stats.detach(),norm[:,None],torch.full_like(norm[:,None],precise),gain.detach().reshape(-1,1)),1)
    raise FloatingPointError("Nonfinite sensory covariance update")


def fit(cohort,arm):
    torch.set_num_threads(1);ids=ticket.births(cohort);seed=270551 if cohort=="original" else 270571
    folder=OUT/cohort;d=folder/arm;d.mkdir(parents=True,exist_ok=True)
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    plan=dict(cohort=cohort,arm=arm,seed=seed,births=ids,lr=.003,worlds=4,stages=list(STAGES),plan=digest(PLAN),
              source={Path(f).name:digest(f) for f in (__file__,ticket.__file__,ticket.net.__file__,ticket.net.native.__file__)},
              parents={str(b):digest(ticket.old.checkpoint_path(ticket.DATA/"tickets"/cohort,"odor","native",b,8000)) for b in ids})
    info=d/"run.json"
    if info.exists() and json.loads(info.read_text())!=plan:raise ValueError("Coverage experiment changed")
    if not info.exists():ticket.storage._write_json(info,plan)
    key=digest(info);g,r=inherited(cohort);gain=torch.nn.Parameter(torch.zeros(len(ids),1))
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters(),gain),lr=.003)
    step=0;history=[];file=d/"resume.pt"
    if file.exists():
        z=torch.load(file,weights_only=True,map_location="cpu")
        if z["plan"]!=key:raise ValueError("Invalid coverage optimizer")
        with torch.no_grad():
            g.weights.copy_(z["weights"]);g.bias.copy_(z["bias"]);gain.copy_(z["gain"])
        r.load_state_dict(z["rule"]);opt.load_state_dict(z["optimizer"]);step=z["step"];history=z["trace"].tolist()
    def save():
        if step in STAGES:
            for i,b in enumerate(ids):ticket.old.save_checkpoint(folder,"odor",arm,b,step,g,r,i)
            ticket.storage._write_raw(d/f"gain-{step}.npz",gain=gain.detach().numpy(),births=ids)
        z=dict(plan=key,step=step,weights=g.weights.detach(),bias=g.bias.detach(),present=g.present,rule=r.state_dict(),
               gain=gain.detach(),optimizer=opt.state_dict(),trace=torch.tensor(history))
        ticket.storage._atomic(file,lambda f:torch.save(z,f))
    if not file.exists():save()
    first=step;start=perf_counter()
    for step in range(first+1,8001):
        t=ticket.join([ticket.tape(seed,b,step,4,domain=274111) for b in ids])
        history.append(update(g,r,gain,opt,t,4,arm).tolist())
        if step%250==0:
            save();print(cohort,arm,step,np.array(history[-100:])[:,:,:3].mean((0,1)).round(3),
                         gain.detach().flatten().numpy().round(2),round((perf_counter()-start)/(step-first),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--arm",choices=ARMS,required=True);fit(**vars(p.parse_args()))
