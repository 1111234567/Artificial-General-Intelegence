"""Learn when to pay for another experience using only the graph's native motor and taste ports."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_capacity as capacity, interleave_revisit_train as storage
from . import local_trace_graph as net, local_trace_train as old, revisit_choice_graph as motor


ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/active-review-20261120"
EVIDENCE=ROOT/"docs/evidence/active-review-20261120"
PLAN=EVIDENCE/"ticket-plan.json"
PREVIOUS=ROOT/"data/interleaved-learning-20261120"
PRICES=(.05,.25,.55)
STAGES=(0,2000,4000,8000)


def births(cohort):
    return list(range(731,737)) if cohort=="original" else list(range(811,817))


def source(cohort):
    return PREVIOUS/("retention-objective" if cohort=="original" else "retention-objective-confirmation")


def inherited(cohort):
    ids=births(cohort)
    models=[old.load_checkpoint(source(cohort),"odor","uniform",b,8000) for b in ids]
    values=[torch.cat([getattr(g,k) for g,_ in models]) for k in capacity.GRAPH_FIELDS]
    g=net.Graph(torch.nn.Parameter(values[0]),torch.nn.Parameter(values[1]),values[2])
    r=net.Rule(ids)
    r.load_state_dict({name:torch.cat([rule.state_dict()[name] for _,rule in models]) for name in r.state_dict()})
    return g,r


def tape(seed,birth,step,worlds,*,domain=197177,price=None,alternative=None):
    rng=np.random.default_rng([domain,seed,birth,step])
    directions=rng.normal(size=(worlds,3,3))
    directions=(1.35*directions/np.linalg.norm(directions,axis=2,keepdims=True)).astype(np.float32)
    orient=rng.choice(np.array([-1.,1.],np.float32),(worlds,6))
    law_x=np.tile(np.array([-1.,-1.,1.,1.,-1.,-1.,1.,1.],np.float32),worlds)
    law_b=np.tile(np.array([-1.,1.,-1.,1.,-1.,1.,-1.,1.],np.float32),worlds)
    evidenced=np.tile(np.array([True]*4+[False]*4),worlds)
    draw_price=rng.choice(PRICES,worlds)
    omit=rng.integers(0,2,worlds).astype(bool)
    order=rng.integers(0,2,worlds)
    def expand(value):return np.repeat(value,8,axis=0)
    if price is not None:draw_price[:]=price
    if alternative is not None:omit[:]=alternative=="omitted"
    x,y,b=np.moveaxis(expand(directions),1,0)
    omit=expand(omit)
    directions0=np.where((evidenced|omit)[:,None],x,y)
    y_sign=expand(rng.choice(np.array([-1.,1.],np.float32),worlds))
    sgn0=np.where(evidenced,law_x,np.where(omit,0.,y_sign))
    order=expand(order)
    first=np.where(order[:,None]==0,directions0,b)
    second=np.where(order[:,None]==0,b,directions0)
    side=expand(orient)
    previous=np.stack((first,second),axis=1)*side[:,:2,None]
    values=np.stack((np.where(order==0,sgn0,law_b),np.where(order==0,law_b,sgn0)),1)*side[:,:2]
    price_lane=expand(draw_price).astype(np.float32)
    offer_side=side[:,3]
    mark=np.array([1.25,0.,0.],np.float32)
    center=price_lane[:,None]*np.array([0.,0.,2.],np.float32)
    public=dict(study=previous,study_value=values,preview=x*side[:,2,None],
                offer_left=center-offer_side[:,None]*mark,offer_right=center+offer_side[:,None]*mark,
                review=x*side[:,4,None],review_value=law_x*side[:,4],
                final=x*side[:,5,None],final_value=law_x*side[:,5],
                offer_side=offer_side,price=price_lane,evidenced=evidenced,omitted=omit,
                coin=expand(rng.random(worlds)<.5),
                draws=expand(rng.random((worlds,5,5),dtype=np.float32)),
                gaps=expand(rng.integers(0,7,(worlds,4))))
    return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in public.items()}


def join(parts):return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


def encounter(rule,state,direction,factor,draw,intervention="active",matched_side=None):
    action=net.act(rule,state,-direction,direction,draw[:,0],side_override=matched_side)
    taste=action.side*factor
    contact=net.observe(rule,action,-direction,direction,taste)
    edit=net.commit(rule,contact,draw[:,1:],intervention=intervention)
    return edit.state,action.motor_log_prob+edit.edit_log_prob,action.side,taste


def prefix(g,r,t,*,history="true",preview=True):
    lanes=len(t["price"])
    view=r.expand(lanes//len(g.bias))
    state=net.start(g,lanes//len(g.bias))
    score=0.;steps=[]
    for i in range(2):
        state=net.wait(view,state,t["gaps"][:,i],max_ticks=6)
        factor=t["study_value"][:,i] if history!="blank" else torch.zeros_like(t["price"])
        state,part,side,taste=encounter(view,state,t["study"][:,i],factor,t["draws"][:,i],
                                        "sham" if history=="sham" else "active")
        score=score+part;steps.append(torch.stack((side,taste),1))
    state=net.wait(view,state,t["gaps"][:,2],max_ticks=6)
    x=t["preview"] if preview else torch.zeros_like(t["preview"])
    state=net._advance(view,state,net.native.sensory(state.graph,-x,x),net.REACTION_TICKS)
    return view,state,score,torch.stack(steps,1)


def finish(view,state,t,mode,*,decision=None,deliver="true",posted=None):
    l,r=t["offer_left"],t["offer_right"]
    if posted is not None:
        shift=torch.zeros_like(l);shift[:,2]=2*(posted-t["price"])
        l,r=l+shift,r+shift
    u=t["draws"]
    override=None if decision is None else torch.where(decision,t["offer_side"],-t["offer_side"])
    choice=net.act(view,state,l,r,u[:,2,0],side_override=override)
    chose_ticket=choice.side==t["offer_side"]
    gets_ticket=t["coin"] if mode=="coin" else chose_ticket
    fee=t["price"]*gets_ticket
    entry=net.observe(view,choice,l,r,-fee)
    written=net.commit(view,entry,u[:,2,1:])
    after=written.state
    factor=t["review_value"]*gets_ticket
    if deliver=="zero":factor=torch.zeros_like(factor)
    if deliver=="reversed":factor=-factor
    after,score,side,taste=encounter(view,after,t["review"],factor,u[:,3])
    after=net.wait(view,after,t["gaps"][:,3],max_ticks=6)
    a=motor.act(after,-t["final"],t["final"],u[:,4,0])
    gross=(a.side*t["final_value"]>0).float()
    return dict(choice=choice.side,buy=gets_ticket,logit=choice.motor_logit,review_side=side,
                taste=taste,final=a.side,gross=gross,utility=gross-fee,
                route_score=choice.motor_log_prob,following_score=written.edit_log_prob+score+a.motor_log_prob)


def centered(values,roots,worlds):
    x=values.detach().reshape(roots,worlds,8)
    avg=x.mean(2,keepdim=True)
    return x-(avg.sum(1,keepdim=True)-avg)/(worlds-1)


def update(g,r,optimizer,t,worlds,arm):
    params=(g.weights,g.bias,*r.parameters())
    for precise in (False,True):
        optimizer.zero_grad(set_to_none=True)
        if precise:
            body=net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double()
        else:body,rule=g,r
        view,state,score,_=prefix(body,rule,t)
        res=finish(view,state,t,arm)
        before=centered(res["utility"],len(g.bias),worlds)
        after=centered(res["gross"],len(g.bias),worlds)
        loss=-((score+res["route_score"]).reshape_as(before)*before+
               res["following_score"].reshape_as(after)*after).mean((1,2)).sum()
        if not loss.isfinite():continue
        loss.backward()
        src=(body.weights,body.bias,*rule.parameters())
        norms=storage._gradients(src,2.)
        if norms is None:continue
        if precise:
            for p,q in zip(params,src,strict=True):p.grad=None if q.grad is None else q.grad.to(p.dtype)
        optimizer.step()
        cols=(res["gross"],res["utility"],res["buy"].float())
        return torch.stack((*[v.reshape(len(g.bias),worlds,8).mean((1,2)) for v in cols],norms,
                            torch.full_like(norms,precise)),1).detach()
    raise FloatingPointError("Nonfinite ticket update after precise replay")


def fit(cohort,arm,stop=8000):
    torch.set_num_threads(1)
    ids=births(cohort);seed=762741 if cohort=="original" else 762769
    folder=DATA/"tickets"/cohort;out=folder/arm;out.mkdir(parents=True,exist_ok=True)
    sha=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    plan=dict(cohort=cohort,arm=arm,births=ids,steps=8000,worlds=4,lr=.003,seed=seed,
              plan_sha256=sha(PLAN),sources={Path(s).name:sha(s) for s in
                (__file__,net.__file__,net.native.__file__,motor.__file__,storage.__file__)},
              parents={str(b):sha(old.checkpoint_path(source(cohort),"odor","uniform",b,8000)) for b in ids})
    m=out/"run.json"
    if m.exists() and json.loads(m.read_text())!=plan:raise ValueError("Changed ticket protocol")
    if not m.exists():storage._write_json(m,plan)
    key=sha(m);g,r=inherited(cohort);opt=torch.optim.Adam((g.weights,g.bias,*r.parameters()),lr=.003)
    resume=out/"resume.pt";step=0;trace=[]
    if resume.exists():
        prev=torch.load(resume,map_location="cpu",weights_only=True)
        if prev["plan"]!=key:raise ValueError("Unrelated optimizer")
        with torch.no_grad():
            for field in capacity.GRAPH_FIELDS:getattr(g,field).copy_(prev["graph"][field])
        r.load_state_dict(prev["rule"]);opt.load_state_dict(prev["optimizer"])
        step=prev["step"];trace=prev["trace"].tolist()
    def save():
        if step in STAGES:
            for i,b in enumerate(ids):old.save_checkpoint(folder,"odor",arm,b,step,g,r,i)
        state=dict(step=step,plan=key,trace=torch.tensor(trace),graph={f:getattr(g,f).detach() for f in capacity.GRAPH_FIELDS},
                   rule=r.state_dict(),optimizer=opt.state_dict())
        storage._atomic(resume,lambda p:torch.save(state,p))
    if not resume.exists():save()
    initial=step;start=perf_counter()
    for step in range(initial+1,stop+1):
        t=join([tape(seed,b,step,4) for b in ids])
        trace.append(update(g,r,opt,t,4,arm).tolist())
        if step%250==0:
            save()
            print(cohort,arm,step,np.array(trace[-100:])[:,:,:3].mean((0,1)).round(3),
                  round((perf_counter()-start)/(step-initial),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",required=True,choices=("original","independent"))
    p.add_argument("--arm",required=True,choices=("native","coin"))
    fit(**vars(p.parse_args()))
