"""Teach two coexisting memory skills using a single real two-choice terminal payout."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_cross_skill as previous, interleave_cross_skill_eval as env
from . import interleave_triangle_train as tri, local_trace_graph as graph, revisit_choice_graph as motor
from . import interleave_checkpoint_publish as publish
from .interleave_interference import DATA,EVIDENCE


ARM=("blocked","interspersed")
STEPS=8000
DEST=DATA/"one-life"
PLAN=EVIDENCE/"one-life-plan.json"


def ticket(base_seed,birth,step,n,arm,*,domain=781037,all_laws=False):
    rng=np.random.default_rng([domain,base_seed,birth,step])
    t=env.make(int(rng.integers(2**31)),n)
    indices=np.arange(n)
    blocks=rng.integers(2,size=n)
    selected=t["orders"][indices,blocks if arm=="blocked" else np.full(n,2)]
    draw=rng.random((n,2,5),dtype=np.float32)
    gap=rng.integers(0,9,size=(n,2))
    which=np.stack((rng.integers(0,3,n),rng.integers(3,6,n)),axis=1)
    which=rng.permuted(which,axis=1)
    picks=np.stack([rng.choice(32,4,replace=False) for _ in range(n)])
    picks=np.concatenate((picks,picks^63),axis=1)
    if all_laws:picks=np.broadcast_to(np.arange(64),(n,64))
    laws=env.LAWS[picks]
    def unfold(x):return torch.from_numpy(np.repeat(x,picks.shape[1],axis=0))
    return dict(left=unfold(t["left"][indices[:,None],selected]),right=unfold(t["right"][indices[:,None],selected]),
                support_order=unfold(selected),orientation=unfold(t["study_orientation"][indices[:,None],selected]),
                su=unfold(t["draws"][indices[:,None],selected]),wait=unfold(t["waits"][indices[:,None],selected]),
                query=unfold(which),ql=unfold(t["query_left"][indices[:,None],which]),
                qr=unfold(t["query_right"][indices[:,None],which]),qsign=unfold(t["query_orientation"][indices[:,None],which]),
                qu=unfold(draw),qwait=unfold(gap),law=torch.from_numpy(laws.reshape(-1,6).astype(np.float32)))


def perform(g,r,t):
    n=len(t["law"]);count=n//len(g.bias);row=torch.arange(n)
    rule=r.expand(count)
    state=graph.start(g,count)
    score=0
    history=[]
    for i in range(7):
        state=graph.wait(rule,state,t["wait"][:,i],max_ticks=8)
        l,right,u=t["left"][:,i],t["right"][:,i],t["su"][:,i]
        a=graph.act(rule,state,l,right,u[:,0])
        obj=t["support_order"][:,i]
        factor=t["law"][row,obj.clamp_max(5)]*t["orientation"][:,i]
        reward=torch.where(obj>=5,factor,a.side*factor)
        observed=graph.observe(rule,a,l,right,reward)
        edited=graph.commit(rule,observed,u[:,1:])
        state=edited.state
        score=score+a.motor_log_prob+edited.edit_log_prob
        history.append(torch.stack((a.side,reward),dim=1).detach())
    sides=[];earned=[]
    for i in range(2):
        state=graph.wait(rule,state,t["qwait"][:,i],max_ticks=8)
        l,right,u=t["ql"][:,i],t["qr"][:,i],t["qu"][:,i]
        a=graph.act(rule,state,l,right,u[:,0]) if i==0 else motor.act(state,l,right,u[:,0])
        value=a.side*t["qsign"][:,i]*t["law"][row,t["query"][:,i]]
        earned.append(value.detach());sides.append(a.side.detach())
        score=score+a.motor_log_prob
        if i==0:
            e=graph.observe(rule,a,l,right,torch.zeros_like(value))
            edit=graph.commit(rule,e,u[:,1:])
            state=edit.state;score=score+edit.edit_log_prob
    return score,torch.stack(earned,1),torch.stack(sides,1),torch.stack(history,1)


def update(g,r,opt,t,worlds):
    params=(g.weights,g.bias,*r.parameters())
    for safer in (False,True):
        opt.zero_grad(set_to_none=True)
        if safer:
            body=graph.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double()
        else:body,rule=g,r
        score,values,_,_=perform(body,rule,t)
        positive=(values>0).all(1).float().reshape(len(g.bias),worlds,-1)
        mean=positive.mean(2,keepdim=True)
        adv=positive-(mean.sum(1,keepdim=True)-mean)/(worlds-1)
        loss=-(score.reshape_as(positive)*adv).mean((1,2)).sum()
        if not loss.isfinite():continue
        loss.backward()
        origin=(body.weights,body.bias,*rule.parameters())
        norm=tri.revisit._gradients(origin,2.)
        if norm is None:continue
        if safer:
            for p,q in zip(params,origin,strict=True):p.grad=None if q.grad is None else q.grad.to(p.dtype)
        opt.step()
        return torch.stack((positive.mean((1,2)),(values[:,0]>0).float().reshape_as(positive).mean((1,2)),
                            (values[:,1]>0).float().reshape_as(positive).mean((1,2)),norm,
                            torch.full_like(norm,safer)),1).detach()
    raise FloatingPointError("Nonfinite same-life gradient")


def train(cohort,arm):
    torch.set_num_threads(1)
    original=cohort=="original"
    births=list(range(731,737)) if original else list(range(811,817))
    seed=93071 if original else 93083
    path=DEST/cohort;folder=path/arm
    folder.mkdir(parents=True,exist_ok=True)
    meta=dict(seed=seed,arm=arm,births=births,worlds=4,updates=STEPS,lr=.003,
              source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=sha256(PLAN.read_bytes()).hexdigest())
    source,tag=previous.geometry.sources(cohort,"hard",64000)
    meta["parents"]={str(b):sha256(tri.old_fit.checkpoint_path(source,"odor",tag,b,64000).read_bytes()).hexdigest()
                     for b in births}
    filename=folder/"run.json"
    sha=publish.locked_plan(filename,meta,Path(__file__).stem,"source_sha256")
    g,r=previous.born(cohort,births)
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters()),lr=meta["lr"])
    step=0;trace=[];resume=folder/"resume.pt"
    if resume.exists():
        d=torch.load(resume,weights_only=True,map_location="cpu")
        if d["manifest"]!=sha:raise ValueError("One-life resume belongs to a different study")
        with torch.no_grad():
            for k in tri.revisit.capacity.GRAPH_FIELDS:getattr(g,k).copy_(d["graph"][k])
        r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"]);step=d["step"];trace=d["trace"].tolist()
    def save():
        d=dict(manifest=sha,step=step,trace=torch.as_tensor(trace),graph={k:getattr(g,k).detach() for k in tri.revisit.capacity.GRAPH_FIELDS},
               rule=r.state_dict(),optimizer=opt.state_dict())
        if step in (0,2000,4000,6000,8000):
            for i,b in enumerate(births):tri.old_fit.save_checkpoint(path,"odor",arm,b,step,g,r,i)
        tri.revisit._atomic(resume,lambda f:torch.save(d,f))
    if not resume.exists() or publish.needs_export(path,"odor",arm,births,step,(0,2000,4000,6000,8000)):save()
    first=step;started=perf_counter()
    for step in range(first+1,STEPS+1):
        batches=[ticket(seed,b,step,meta["worlds"],arm) for b in births]
        t={k:torch.cat([v[k] for v in batches]) for k in batches[0]}
        trace.append(update(g,r,opt,t,meta["worlds"]).tolist())
        if step%250==0:
            save()
            print(cohort,arm,step,round(np.array(trace[-100:])[:,:,0].mean(),3),
                  round((perf_counter()-started)/(step-first),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--arm",choices=ARM,required=True)
    a=p.parse_args()
    train(a.cohort,a.arm)
