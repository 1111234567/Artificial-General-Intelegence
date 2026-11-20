"""Expose object identity and food polarity as separately observable aspects of the same scene."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import noon_paired_lesson as lesson, noon_review_world as world, noon_ticket as families
from . import local_trace_graph as net,local_trace_train as old,revisit_choice_graph as motor


OUT=world.DATA/"identity";PLAN=world.EVIDENCE/"identity-plan.json"
ARMS=("tag","flavor","unbound")
STAGES=(0,2000,4000,8000)


def context(seed,worlds,count,births=(),step=0):
    identities=[];unbound=[]
    for b in (births if births else (0,)):
        rng=np.random.default_rng([364799,seed,b,step,count])
        def draw(shape):
            a=rng.normal(size=shape)
            a=(.8*a/np.linalg.norm(a,axis=-1,keepdims=True)).astype(np.float32)
            return torch.from_numpy(np.repeat(a,4,axis=0))
        identities.append(draw((worlds,count,3)));unbound.append(draw((worlds,2,3)))
    return torch.cat(identities),torch.cat(unbound)


def prefix(g,r,t,missing,tags):
    n=t.lanes;view=r.expand(n//len(g.bias));s=net.start(g,n//len(g.bias));score=0.;trace=[]
    for i in range(t.count):
        sign=t.study_orientation[:,i];axis=t.odor[:,i]*sign[:,None];cue=tags[:,i]
        factor=torch.where(missing==i,0.,t.law[:,i]*sign)
        ev=world.collect(view,s,cue+axis,cue-axis,factor,t.study_draw[:,i])
        s=ev.state;score=score+ev.motor.motor_log_prob+ev.edit.edit_log_prob
        trace.append(torch.stack((ev.motor.side,ev.earned),1))
    return view,s,score,torch.stack(trace,1)


def choose(view,state,t,tags,unbound,arm,*,swapped=False,flipped=False):
    if arm=="tag":menu=tags[world._rows(t)[:,None],t.candidates]
    elif arm=="unbound":menu=unbound
    else:
        menu=t.odor[world._rows(t)[:,None],t.candidates]*(.8/1.35)*t.menu_orientation[...,None]
    if swapped:menu=menu.flip(1)
    if flipped:menu=-menu
    return world.select(view,state,menu[:,0],menu[:,1],t.decision_draw)


def recollect(view,s,t,tags,side,*,write=True):
    s=world.echo_choice(view,s,side);row=world._rows(t)
    ids=world.selected_item(t,side);sign=t.review_orientation
    vec=t.odor[row,ids]*sign[:,None];cue=tags[row,ids]
    return world.collect(view,s,cue+vec,cue-vec,t.law[row,ids]*sign,t.review_draw,write=write)


def probe(state,t,tags,ids):
    row=world._rows(t);sign=t.final_orientation[row,ids]
    axis=t.odor[row,ids]*sign[:,None];cue=tags[row,ids]
    a=motor.act(state,cue+axis,cue-axis,t.final_draw[row,ids])
    return a,t.law[row,ids]*sign


def update(g,r,opt,t,mask,tags,unbound,worlds,arm):
    params=(g.weights,g.bias,*r.parameters())
    for precise in (False,True):
        opt.zero_grad(set_to_none=True)
        if precise:
            body=net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double()
        else:body,rule=g,r
        view,s,score,_=prefix(body,rule,t,mask,tags)
        a=choose(view,s,t,tags,unbound,arm)
        after=recollect(view,a.state,t,tags,a.side)
        final,goal=probe(after.state,t,tags,t.final_item)
        score=score+a.motor_log_prob+after.motor.motor_log_prob+after.edit.edit_log_prob+final.motor_log_prob
        pay=(final.side*goal).detach().reshape(len(g.bias),worlds,4)
        means=pay.mean(2,keepdim=True)
        baseline=(means.sum(1,keepdim=True)-means)/(worlds-1)
        loss=-(score.reshape_as(pay)*(pay-baseline)).mean((1,2)).sum()
        if not loss.isfinite():continue
        loss.backward()
        src=(body.weights,body.bias,*rule.parameters())
        norm=lesson.storage._gradients(src,2.)
        if norm is None:continue
        if precise:
            for a0,b0 in zip(params,src,strict=True):a0.grad=None if b0.grad is None else b0.grad.to(a0.dtype)
        opt.step()
        chose=world.selected_item(t,a.side)==mask
        n=len(g.bias)
        under=chose.reshape(n,-1).sum(1)/(mask>=0).reshape(n,-1).sum(1).clamp_min(1)
        return torch.stack(((pay>0).float().mean((1,2)),under,norm,torch.full_like(norm,precise)),1).detach()
    raise FloatingPointError("Nonfinite identity-gradient estimate")


def fit(cohort,arm):
    torch.set_num_threads(1)
    ids=families.births(cohort);seed=951481 if cohort=="original" else 951487
    out=OUT/cohort;folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
    digest=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    plan=dict(seed=seed,births=ids,arm=arm,lr=.002,worlds=8,steps=8000,
              plan_sha256=digest(PLAN),parents={str(b):digest(old.checkpoint_path(families.source(cohort),"odor","uniform",b,8000)) for b in ids},
              sources={Path(f).name:digest(f) for f in (__file__,lesson.__file__,world.__file__,net.__file__,net.native.__file__)})
    manifest=folder/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError("Identity protocol changed")
    if not manifest.exists():lesson.storage._write_json(manifest,plan)
    key=digest(manifest);body,rule=families.inherited(cohort)
    opt=torch.optim.Adam((body.weights,body.bias,*rule.parameters()),lr=.002)
    resume=folder/"resume.pt";history=[];step=0
    if resume.exists():
        z=torch.load(resume,weights_only=True,map_location="cpu")
        if z["digest"]!=key:raise ValueError("Different identity optimizer")
        with torch.no_grad():body.weights.copy_(z["weights"]);body.bias.copy_(z["bias"])
        rule.load_state_dict(z["rule"]);opt.load_state_dict(z["optimizer"]);step=z["step"];history=z["trace"].tolist()
    def save():
        if step in STAGES:
            for i,b in enumerate(ids):old.save_checkpoint(out,"odor",arm,b,step,body,rule,i)
        obj=dict(step=step,digest=key,weights=body.weights.detach(),bias=body.bias.detach(),present=body.present,
                 rule=rule.state_dict(),optimizer=opt.state_dict(),trace=torch.tensor(history))
        lesson.storage._atomic(resume,lambda f:torch.save(obj,f))
    if not resume.exists():save()
    first=step;t0=perf_counter()
    for step in range(first+1,8001):
        count=2 if step%2 else 4
        t,mask=lesson.prepare(seed,ids,step,8,count);tags,unbound=context(seed,8,count,ids,step)
        history.append(update(body,rule,opt,t,mask,tags,unbound,8,arm).tolist())
        if step%250==0:
            save();print(cohort,arm,step,np.array(history[-100:])[:,:,:2].mean((0,1)).round(3),
                         round((perf_counter()-t0)/(step-first),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=world.COHORTS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True);a=p.parse_args();fit(a.cohort,a.arm)
