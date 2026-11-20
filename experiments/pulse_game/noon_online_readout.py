"""Learn a shared review readout on fresh experiences with one or two actually taken routes."""

from argparse import ArgumentParser
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import noon_readback as bank, noon_readback_fit as initial, noon_ticket as parents
from . import noon_paired_lesson as lessons, noon_review_world as physical
from . import local_trace_graph as net, revisit_choice_graph as motor


SEEDS=(643793,643799,643801)
ARMS=("live","sampled_pair","forced_pair")
OUT=physical.DATA/"online-readout";PLAN=physical.EVIDENCE/"online-readout-plan.json"


def make(seed,step,worlds):
    items=[]
    for birth in physical.COHORTS["original"]:
        entropy=np.random.SeedSequence([518663,seed,birth,step]).generate_state(1)[0]
        items.append(bank.tape(int(entropy),worlds,4))
    return (physical.Tape(**{f.name:torch.cat([getattr(i[0],f.name) for i in items]) for f in fields(physical.Tape)}),
            torch.cat([i[1] for i in items]))


@torch.no_grad()
def observe(g,r,t,first):
    view=r.expand(t.lanes//len(g.bias));s=net.start(g,t.lanes//len(g.bias))
    state,_=physical.introduce(view,s,t)
    state,z=bank.scan(view,state,t,first)
    a=physical.select(view,state,*physical.candidate_menu(t),t.decision_draw)
    return a,z


@torch.no_grad()
def execute(g,r,t,a,side):
    # Exactly one physical continuation when side.shape[1] == 1.
    lateral=side.shape[1]
    if lateral==1:
        view=r.expand(t.lanes//len(g.bias))
        state,contact=physical.review(view,a.state,t,side[:,0])
        ix=physical._rows(t);obj=t.final_item;orient=t.final_orientation[ix,obj]
        left,right=physical.sensory_for(t,obj,orient)
        action=motor.act(state,left,right,t.final_draw[ix,obj])
        return ((action.side*t.law[ix,obj]*orient)>0).float()[:,None]
    result=lessons.branches(g,r,t,a,side)
    return (result["reward"]>0).float()


def step(g,r,model,opt,seed,index,arm,worlds):
    t,first=make(seed,index,worlds)
    action,z=observe(g,r,t,first)
    p=model(z.double(),action.motor_logit.double())
    m=2 if arm!="live" else 1
    if arm=="forced_pair":
        side=torch.stack((-torch.ones_like(p),torch.ones_like(p)),1)
    else:
        # Independent fair motor draws per actual life, not one logged treatment per public geometry.
        draws=np.random.default_rng([685277,seed,index]).random((t.lanes,m))
        side=torch.where(torch.from_numpy(draws)<p.detach()[:,None],1.,-1.)
    pay=execute(g,r,t,action,side.to(g.bias.dtype)).double()
    if arm=="forced_pair":
        expected=pay[:,0]*(1-p)+pay[:,1]*p
        loss=-expected.mean()
        realized=expected.detach().mean()
    else:
        groups=pay.reshape(len(g.bias),worlds,8,m)
        world_means=groups.mean((2,3),keepdim=True)
        baseline=(world_means.sum(1,keepdim=True)-world_means)/(worlds-1)
        adv=(groups-baseline).reshape_as(pay)
        likelihood=torch.where(side>0,p[:,None],1-p[:,None]).clamp_min(1e-14).log()
        loss=-(adv*likelihood).mean()
        realized=pay.mean()
    tiny=1e-12
    kl=(p*(2*p).clamp_min(tiny).log()+(1-p)*(2*(1-p)).clamp_min(tiny).log()).mean()
    penalty=torch.cat([v.flatten() for v in model.parameters()]).square().mean()
    loss=loss+.002*kl+.0001*penalty
    opt.zero_grad(set_to_none=True);loss.backward()
    if not all(torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError("Nonfinite readout update")
    opt.step()
    return [float(realized),float(p.detach().mean()),float(kl.detach())]


def fit(seed,arm,stop=4000):
    torch.set_num_threads(1);worlds=8 if arm=="live" else 4
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    root=OUT/str(seed)/arm;root.mkdir(parents=True,exist_ok=True)
    info=dict(seed=seed,arm=arm,updates=4000,worlds_per_birth_per_update=worlds,trajectories_per_birth_per_update=64,
              learning_rate=.01,source_sha256={Path(f).name:digest(f) for f in
              (__file__,bank.__file__,initial.__file__,lessons.__file__,physical.__file__,net.__file__)},plan=digest(PLAN))
    meta=root/"run.json"
    if meta.exists() and json.loads(meta.read_text())!=info:raise ValueError("Online readout protocol changed")
    if not meta.exists():lessons.storage._write_json(meta,info)
    model=initial.Readout("logged");opt=torch.optim.Adam(model.parameters(),lr=.01)
    g,r=parents.inherited("original");g=g.detach();r.requires_grad_(False)
    file=root/"resume.pt";start=0;trace=[];manifest=digest(meta)
    if file.exists():
        saved=torch.load(file,map_location="cpu",weights_only=True)
        if saved["manifest"]!=manifest:raise ValueError("Mismatched online optimizer")
        model.load_state_dict(saved["weights"]);opt.load_state_dict(saved["optimizer"])
        start=saved["step"];trace=saved["trace"].tolist()
    t0=perf_counter()
    for i in range(start+1,stop+1):
        trace.append(step(g,r,model,opt,seed,i,arm,worlds))
        if i%200==0:
            saved=dict(step=i,weights=model.state_dict(),optimizer=opt.state_dict(),trace=torch.tensor(trace),manifest=manifest)
            if i in (1000,2000,4000):lessons.storage._atomic(root/f"step-{i}.pt",lambda f:torch.save(saved,f))
            lessons.storage._atomic(file,lambda f:torch.save(saved,f))
            print(seed,arm,i,np.array(trace[-100:]).mean(0).round(4),round((perf_counter()-t0)/(i-start),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--seed",choices=SEEDS,type=int,required=True)
    p.add_argument("--arm",choices=ARMS,required=True);a=p.parse_args();fit(a.seed,a.arm)
