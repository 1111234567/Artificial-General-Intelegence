"""Assign credit for a binary review action from two equal-budget experienced continuations."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import fields,replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from . import noon_review_world as world, noon_ticket as families
from . import interleave_revisit_train as storage, local_trace_graph as net, local_trace_train as old
from . import revisit_choice_graph as motor


ROOT=families.DATA/"paired-lesson"
PLAN=families.EVIDENCE/"paired-lesson-plan.json"
STAGES=(0,2000,6000,12000)


def expand(t,idx):
    return world.Tape(**{k.name:getattr(t,k.name)[idx] for k in fields(t)})


def prepare(seed,births,step,worlds,count,*,omissions=None,heldout=False):
    base=world.make_tape(seed,worlds,count) if heldout else world.make_tape(seed,worlds,count,births=births,update=step)
    ids=torch.arange(base.lanes).repeat_interleave(2)
    t=expand(base,ids)
    rng=np.random.default_rng([804107,seed,step,count])
    n=t.lanes//4
    has=rng.random(n)<.75
    if omissions is not None:has[:]=omissions
    miss=torch.arange(t.lanes)%2
    missing=t.candidates[torch.arange(t.lanes),miss]
    missing=torch.where(torch.from_numpy(np.repeat(has,4)),missing,torch.full_like(missing,-1))
    which=torch.from_numpy(np.repeat(rng.integers(0,2,n),4))
    item=t.candidates[torch.arange(t.lanes),which]
    return replace(t,final_item=item),missing


def prefix(g,r,t,missing,*,blank=False):
    n=t.lanes
    state=net.start(g,n//len(g.bias));view=r.expand(n//len(g.bias))
    score=0.;physical=[]
    for i in range(t.count):
        sign=t.study_orientation[:,i]; left=t.odor[:,i]*sign[:,None]
        value=torch.where(missing==i,0.,t.law[:,i]*sign)
        if blank:value=torch.zeros_like(value)
        contact=world.collect(view,state,left,-left,value,t.study_draw[:,i])
        state=contact.state
        score=score+contact.motor.motor_log_prob+contact.edit.edit_log_prob
        physical.append(torch.stack((contact.motor.side,contact.earned),1))
    choice=world.select(view,state,*world.candidate_menu(t),t.decision_draw)
    return choice,score,torch.stack(physical,1)


def branches(g,r,t,choice,sides,*,write=True,content=None):
    count=sides.shape[1];idx=torch.arange(t.lanes).repeat_interleave(count)
    twin=expand(t,idx);state=choice.state.subset(idx);view=r.expand(twin.lanes//len(g.bias))
    executed=sides.flatten()
    after,event=world.review(view,state,twin,executed,content_side=None if content is None else content.flatten(),write=write)
    rows=torch.arange(twin.lanes);item=twin.final_item
    sign=twin.final_orientation[rows,item]
    left,right=world.sensory_for(twin,item,sign)
    a=motor.act(after,left,right,twin.final_draw[rows,item])
    earned=a.side*twin.law[rows,item]*sign
    score=event.motor.motor_log_prob+event.edit.edit_log_prob+a.motor_log_prob
    return dict(score=score.reshape(t.lanes,count),reward=earned.reshape(t.lanes,count).detach(),
                answer=a.side.reshape(t.lanes,count),truth=(twin.law[rows,item]*sign).reshape(t.lanes,count),
                side=event.motor.side.reshape(t.lanes,count),taste=event.earned.reshape(t.lanes,count),
                state=after,view=view,tape=twin)


def objective(g,r,t,missing,arm,worlds):
    choice,previous,_=prefix(g,r,t,missing)
    p=choice.motor_logit.sigmoid()
    if arm=="paired":
        sides=torch.stack((-torch.ones_like(p),torch.ones_like(p)),1)
        weight=torch.stack((1-p.detach(),p.detach()),1)
    else:
        u=torch.stack((t.decision_draw,1-t.decision_draw),1)
        sides=2*(u<p.detach()[:,None]).to(p.dtype)-1
        weight=torch.full_like(sides,.5)
    result=branches(g,r,t,choice,sides)
    rewards=result["reward"]; value=(weight*rewards).sum(1)
    means=value.reshape(len(g.bias),worlds,4).mean(2,keepdim=True)
    baseline=((means.sum(1,keepdim=True)-means)/(worlds-1)).expand(-1,-1,4).reshape(-1)
    credit=(previous*(value-baseline) +
            (weight*result["score"]*(rewards-baseline[:,None])).sum(1))
    if arm=="paired":
        credit=credit+p*(rewards[:,1]-rewards[:,0])
    else:
        credit=credit+(F.logsigmoid(sides*choice.motor_logit[:,None])*(rewards-baseline[:,None])).mean(1)
    loss=-credit.reshape(len(g.bias),-1).mean(1).sum()
    natural=world.selected_item(t,choice.side)
    under=(natural==missing)&(missing>=0)
    mask=(missing>=0).float().reshape(len(g.bias),-1)
    metrics=torch.stack(((value.reshape(len(g.bias),-1).mean(1)+1)/2,
                         under.float().reshape_as(mask).sum(1)/mask.sum(1).clamp_min(1),
                         (choice.side>0).float().reshape_as(mask).mean(1)),1)
    return loss,metrics


def update(g,r,opt,t,missing,arm,worlds):
    params=(g.weights,g.bias,*r.parameters())
    for precise in (False,True):
        opt.zero_grad(set_to_none=True)
        if precise:
            body=net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double()
        else:body,rule=g,r
        loss,metrics=objective(body,rule,t,missing,arm,worlds)
        if not loss.isfinite():continue
        loss.backward()
        origin=(body.weights,body.bias,*rule.parameters())
        norm=storage._gradients(origin,2.)
        if norm is None:continue
        if precise:
            for a,b in zip(params,origin,strict=True):a.grad=None if b.grad is None else b.grad.to(a.dtype)
        opt.step()
        return torch.cat((metrics.detach(),norm[:,None],torch.full_like(norm[:,None],precise)),1)
    raise FloatingPointError("Nonfinite paired route fit")


def fit(cohort,arm,stop=12000):
    torch.set_num_threads(1)
    roots=families.births(cohort);seed=462391 if cohort=="original" else 462409
    path=ROOT/cohort;directory=path/arm;directory.mkdir(parents=True,exist_ok=True)
    digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    info=dict(cohort=cohort,arm=arm,births=roots,seed=seed,worlds=4,lr=.002,stages=list(STAGES),
              parent_sha256={str(b):digest(old.checkpoint_path(families.source(cohort),"odor","uniform",b,8000)) for b in roots},
              source_sha256={Path(f).name:digest(f) for f in (__file__,families.__file__,world.__file__,net.__file__,net.native.__file__,motor.__file__)},
              plan_sha256=digest(PLAN))
    m=directory/"run.json"
    if m.exists() and json.loads(m.read_text())!=info:raise ValueError("Paired study setup changed")
    if not m.exists():storage._write_json(m,info)
    g,r=families.inherited(cohort);opt=torch.optim.Adam((g.weights,g.bias,*r.parameters()),lr=.002)
    manifest=digest(m);resume=directory/"resume.pt";step=0;trace=[]
    if resume.exists():
        z=torch.load(resume,map_location="cpu",weights_only=True)
        if z["plan"]!=manifest:raise ValueError("Unexpected optimizer")
        with torch.no_grad():
            g.weights.copy_(z["w"]);g.bias.copy_(z["b"])
        r.load_state_dict(z["rule"]);opt.load_state_dict(z["optimizer"])
        step,trace=z["step"],z["trace"].tolist()
    def save():
        if step in STAGES:
            for i,b in enumerate(roots):old.save_checkpoint(path,"odor",arm,b,step,g,r,i)
        z=dict(plan=manifest,step=step,w=g.weights.detach(),b=g.bias.detach(),present=g.present,rule=r.state_dict(),
               optimizer=opt.state_dict(),trace=torch.tensor(trace))
        storage._atomic(resume,lambda f:torch.save(z,f))
    if not resume.exists():save()
    initial=step;t0=perf_counter()
    for step in range(initial+1,stop+1):
        count=2 if step%2 else 4
        t,missing=prepare(seed,roots,step,4,count)
        trace.append(update(g,r,opt,t,missing,arm,4).tolist())
        if step%250==0:
            save()
            print(cohort,arm,step,np.array(trace[-100:])[:,:,:3].mean((0,1)).round(3),
                  round((perf_counter()-t0)/(step-initial),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--arm",choices=("paired","sampled"),required=True)
    fit(**vars(p.parse_args()))
