"""Let an animal decide whether an experienced review should actually modify its graph."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import noon_ticket as base
from . import local_trace_graph as net,revisit_choice_graph as motor


OUT=base.DATA/"commit";PLAN=base.EVIDENCE/"commit-plan.json"
ARMS=("always","constant","state","confirmation");STAGES=(0,2000,5000,10000)
LAWS=np.array(list(product((-1.,1.),repeat=3)),np.float32)


def tape(seed,birth,step,n,*,domain=603211,theta=None,delay=None):
    rng=np.random.default_rng([domain,seed,birth,step])
    a=rng.normal(size=(n,3));a/=np.linalg.norm(a,axis=1,keepdims=True)
    b=rng.normal(size=(n,3));b-=(a*b).sum(1,keepdims=True)*a;b/=np.linalg.norm(b,axis=1,keepdims=True)
    deg=rng.choice((15.,45.,75.),n)
    if theta is not None:deg[:]=theta
    angle=np.deg2rad(deg)[:,None]
    xyz=(1.35*np.stack((a,a*np.cos(angle)+b*np.sin(angle)),1)).astype(np.float32)
    first=rng.integers(2,size=n);review=rng.integers(2,size=n);final=rng.integers(2,size=n)
    order=np.stack((first,1-first,review,final,1-final),1)
    orient=rng.choice(np.array([-1.,1.],np.float32),order.shape)
    pattern=np.tile(LAWS,(n,1));old=pattern[:,:2];new=old.copy()
    row=np.arange(n*8);rep=lambda v:np.repeat(v,8,axis=0)
    order=rep(order);orientation=rep(orient);new[row,order[:,2]]*=pattern[:,2]
    vals=np.column_stack([old[row,order[:,i]] if i<2 else new[row,order[:,i]] for i in range(5)])*orientation
    wait=rng.integers(0,7,(n,5))
    if delay is not None:wait[:,3]=delay
    out=dict(stimulus=rep(xyz)[row[:,None],order]*orientation[:,:,None],value=vals,
             order=order,orient=orientation,old_law=old,new_law=new,reversed=pattern[:,2]<0,
             weight=np.where(pattern[:,2]<0,.25,.75).astype(np.float32)/4,
             warp=rep(deg).astype(np.float32),gate_uniform=rep(rng.random(n,dtype=np.float32)),
             wait=rep(wait),draw=rep(rng.random((n,5,5),dtype=np.float32)),
             query_u=rep(rng.random((n,2),dtype=np.float32)),query_orientation=rep(rng.choice(np.array([-1.,1.],np.float32),(n,2))),
             objects=rep(xyz))
    return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in out.items()}


def prefix(g,r,t):
    n=len(t["value"]);view=r.expand(n//len(g.bias));s=net.start(g,n//len(g.bias));score=0.;history=[]
    for i in range(3):
        s=net.wait(view,s,t["wait"][:,i],max_ticks=6)
        x=t["stimulus"][:,i];u=t["draw"][:,i];a=net.act(view,s,-x,x,u[:,0])
        taste=a.side*t["value"][:,i]
        e=net.observe(view,a,-x,x,taste)
        score=score+a.motor_log_prob
        history.append(torch.stack((a.side,taste,a.motor_logit),1))
        if i<2:
            written=net.commit(view,e,u[:,1:])
            s=written.state;score=score+written.edit_log_prob
    return view,e,score,torch.stack(history,1)


def probability(gate,arm,experience,history):
    n=len(history);w=gate.repeat_interleave(n//len(gate),0)
    if arm=="always":return torch.ones_like(history[:,0,0])
    l=w[:,0]
    if arm=="state":l=l+(w[:,1:]*experience.state.hidden).sum(1)
    if arm=="confirmation":
        side,taste,z=history[:,-1].unbind(1)
        l=l+w[:,1]*side*taste*(z*.5).tanh()
    return l.sigmoid()


def endings(view,experience,t,*,cold=False):
    u=t["draw"][:,2,1:]
    written=net.commit(view,experience,u)
    quiet=net.wait(view,experience.state,net.PRE_WRITE_TICKS+net.POST_WRITE_TICKS)
    values=[];scores=[];ends=[];decisions=[]
    for mode,s in enumerate((quiet,written.state)):
        downstream=written.edit_log_prob if mode else torch.zeros_like(written.edit_log_prob)
        outcomes=[];acts=[]
        for j in (3,4):
            s=net.wait(view,s,t["wait"][:,j],max_ticks=int(t["wait"][:,j].max()))
            if j==3 and cold:
                choices=[]
                for obj in (0,1):
                    sign=t["query_orientation"][:,obj];q=t["objects"][:,obj]*sign[:,None]
                    choices.append(motor.act(s,-q,q,t["query_u"][:,obj]).side)
                ends.append(torch.stack(choices,1))
            x=t["stimulus"][:,j];u=t["draw"][:,j]
            a=net.act(view,s,-x,x,u[:,0]) if j==3 else motor.act(s,-x,x,u[:,0])
            downstream=downstream+a.motor_log_prob;outcomes.append((a.side*t["value"][:,j]>0).float());acts.append(a.side)
            if j==3:
                e=net.observe(view,a,-x,x,torch.zeros_like(a.side))
                edit=net.commit(view,e,u[:,1:]);s=edit.state;downstream=downstream+edit.edit_log_prob
        values.append(torch.stack(outcomes,1));scores.append(downstream);decisions.append(torch.stack(acts,1))
    return dict(pay=torch.stack(values,1),score=torch.stack(scores,1),serial_actions=torch.stack(decisions,1),
                cold=torch.stack(ends,1) if cold else None,receipt=written.receipt)


def update(g,r,gate,opt,t,arm,worlds):
    params=(g.weights,g.bias,*r.parameters(),gate)
    for precise in (False,True):
        opt.zero_grad(set_to_none=True)
        if precise:
            m=net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double();w=torch.nn.Parameter(gate.detach().double())
        else:m,rule,w=g,r,gate
        view,e,previous,h=prefix(m,rule,t);p=probability(w,arm,e,h)
        out=endings(view,e,t);realized=out["pay"].detach().prod(2)
        mix=torch.stack((1-p.detach(),p.detach()),1);expected=(mix*realized).sum(1)
        prior=t["weight"].reshape(len(g.bias),worlds,8);av=(prior*expected.reshape_as(prior)).sum(2,keepdim=True)
        baseline=((av.sum(1,keepdim=True)-av)/(worlds-1)).expand_as(prior).reshape(-1)
        terms=previous*(expected-baseline)+(mix*out["score"]*(realized-baseline[:,None])).sum(1)
        if arm!="always":terms=terms+p*(realized[:,1]-realized[:,0])
        loss=-(prior*terms.reshape_as(prior)).sum(2).mean(1).sum()
        if not loss.isfinite():continue
        loss.backward();origin=(m.weights,m.bias,*rule.parameters(),w)
        norm=base.storage._gradients(origin,2.)
        if norm is None:continue
        if precise:
            for a,b in zip(params,origin,strict=True):a.grad=None if b.grad is None else b.grad.to(a.dtype)
        opt.step()
        means=[(prior*v.reshape_as(prior)).sum(2).mean(1) for v in (expected,p.detach(),realized[:,1]-realized[:,0])]
        return torch.stack((*means,norm,torch.full_like(norm,precise)),1).detach()
    raise FloatingPointError("Nonfinite post-receipt commit gradient")


def fit(cohort,arm):
    torch.set_num_threads(1);ids=base.births(cohort);seed=857393 if cohort=="original" else 857401
    path=OUT/cohort;folder=path/arm;folder.mkdir(parents=True,exist_ok=True)
    digest=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    meta=dict(cohort=cohort,arm=arm,seed=seed,births=ids,worlds=4,steps=10000,plan=digest(PLAN),
              parents={str(b):digest(base.old.checkpoint_path(base.source(cohort),"odor","uniform",b,8000)) for b in ids},
              sources={Path(f).name:digest(f) for f in (__file__,base.__file__,net.__file__,net.native.__file__)})
    mf=folder/"run.json"
    if mf.exists() and json.loads(mf.read_text())!=meta:raise ValueError("Commit-gate settings changed")
    if not mf.exists():base.storage._write_json(mf,meta)
    key=digest(mf);g,r=base.inherited(cohort)
    width={"state":21,"confirmation":2,"constant":1,"always":1}[arm]
    gate=torch.nn.Parameter(torch.zeros(len(ids),width),requires_grad=arm!="always")
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters(),gate),lr=.002)
    step=0;log=[];resume=folder/"resume.pt"
    if resume.exists():
        saved=torch.load(resume,map_location="cpu",weights_only=True)
        if saved["meta"]!=key:raise ValueError("Wrong write-gate optimizer")
        with torch.no_grad():g.weights.copy_(saved["w"]);g.bias.copy_(saved["b"]);gate.copy_(saved["gate"])
        r.load_state_dict(saved["rule"]);opt.load_state_dict(saved["optimizer"]);step=saved["step"];log=saved["trace"].tolist()
    def save():
        if step in STAGES:
            for i,b in enumerate(ids):base.old.save_checkpoint(path,"odor",arm,b,step,g,r,i)
            base.storage._write_raw(folder/f"gate-{step}.npz",gate=gate.detach().numpy(),births=ids)
        state=dict(step=step,w=g.weights.detach(),b=g.bias.detach(),present=g.present,gate=gate.detach(),rule=r.state_dict(),
                   optimizer=opt.state_dict(),trace=torch.tensor(log),meta=key)
        base.storage._atomic(resume,lambda f:torch.save(state,f))
    if not resume.exists():save()
    start=step;t0=perf_counter()
    for step in range(start+1,10001):
        t=base.join([tape(seed,b,step,4) for b in ids])
        log.append(update(g,r,gate,opt,t,arm,4).tolist())
        if step%250==0:
            save();print(cohort,arm,step,np.array(log[-100:])[:,:,:3].mean((0,1)).round(3),
                         round((perf_counter()-t0)/(step-start),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--arm",choices=ARMS,required=True);a=p.parse_args();fit(a.cohort,a.arm)
