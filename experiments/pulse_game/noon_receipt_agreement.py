"""A local update gated by agreement between the graph's previous action and the received taste."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import noon_coverage as cov, noon_option_binding as binding, noon_ticket as ticket


OUT=ticket.DATA/"receipt-agreement";PLAN=ticket.EVIDENCE/"receipt-agreement-plan.json"
ARMS=("count","agreement","scrambled");STAGES=(0,4000,8000,12000)
PATTERNS=np.array(list(product((-1.,1.),repeat=4)),np.float32)


def tape(seed,birth,step,worlds,*,domain=181921,price=None,slip=.2):
    rng=np.random.default_rng([domain,seed,birth,step])
    points=rng.normal(size=(worlds,2,3))
    points=(1.35*points/np.linalg.norm(points,axis=2,keepdims=True)).astype(np.float32)
    ori=rng.choice(np.array([-1.,1.],np.float32),(worlds,7))
    bpos=rng.integers(3,size=worlds)
    idx=np.tile(np.arange(3),(worlds,1))
    event=np.where(idx==bpos[:,None],2,np.where(idx<bpos[:,None],idx,idx-1))
    t_sign=np.tile(PATTERNS,(worlds,1))
    def rep(arr):return np.repeat(arr,16,axis=0)
    worldx=rep(points[:,0]);worldb=rep(points[:,1])
    order=rep(event);signs=rep(ori)
    x1=t_sign[:,0]*t_sign[:,2];x2=t_sign[:,0]*t_sign[:,3]
    vals=np.stack((x1,x2,t_sign[:,1]),1)
    values=np.take_along_axis(vals,order,axis=1)*signs[:,:3]
    study=np.where(order[:,:,None]==2,worldb[:,None],worldx[:,None])*signs[:,:3,None]
    pr=rng.choice(np.array(ticket.PRICES,np.float32),worlds)
    if price is not None:pr[:]=price
    pr=rep(pr)
    center=pr[:,None]*np.array([0.,0.,2.],np.float32)
    side=signs[:,4];mark=np.array([1.25,0.,0.],np.float32)
    weight=(np.where(t_sign[:,2]>0,1-slip,slip)*np.where(t_sign[:,3]>0,1-slip,slip)/4).astype(np.float32)
    data=dict(study=study,study_value=values,study_draws=rep(rng.random((worlds,3,5),dtype=np.float32)),
              study_gaps=rep(rng.integers(0,7,size=(worlds,3))),scramble=rep(rng.choice(np.array([-1.,1.],np.float32),(worlds,3))),
              preview=worldx*signs[:,3,None],offer_left=center-side[:,None]*mark,offer_right=center+side[:,None]*mark,
              offer_side=side,price=pr,review=worldx*signs[:,5,None],review_value=t_sign[:,0]*signs[:,5],
              final=worldx*signs[:,6,None],final_value=t_sign[:,0]*signs[:,6],coin=rep(rng.random(worlds)<.5),
              gaps=rep(rng.integers(0,7,size=(worlds,4))),draws=rep(rng.random((worlds,5,5),dtype=np.float32)),
              weight=weight,pattern=t_sign,order=order)
    return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in data.items()}


def prefix(g,r,t,eta,arm):
    n=len(t["price"]);view=r.expand(n//len(g.bias));s=ticket.net.start(g,n//len(g.bias))
    k=eta.reshape(-1).repeat_interleave(n//len(g.bias))
    trace=g.bias.new_zeros((n,6));baseline_trace=torch.zeros_like(trace)
    score=0.;history=[]
    for i in range(t["study"].shape[1]):
        s=ticket.net.wait(view,s,t["study_gaps"][:,i],max_ticks=6)
        x=t["study"][:,i];u=t["study_draws"][:,i]
        a=ticket.net.act(view,s,-x,x,u[:,0])
        taste=a.side*t["study_value"][:,i]
        tendency=a.side*(a.motor_logit*.5).tanh()
        confirmation=taste*tendency
        if arm=="scrambled":confirmation=confirmation*t["scramble"][:,i]
        amount=taste.abs() if arm=="count" else taste.abs()+k*confirmation
        ux=torch.nn.functional.normalize(a.side[:,None]*x,dim=1)
        features=cov.moments(ux)
        trace=trace+amount[:,None]*features
        baseline_trace=baseline_trace+taste.abs()[:,None]*features
        e=ticket.net.observe(view,a,-x,x,taste);written=ticket.net.commit(view,e,u[:,1:])
        score=score+a.motor_log_prob+written.edit_log_prob;s=written.state
        history.append(torch.stack((a.side,taste,tendency,amount),1))
    s=ticket.net.wait(view,s,t["gaps"][:,2],max_ticks=6)
    q=t["preview"];s=ticket.net._advance(view,s,ticket.net.native.sensory(s.graph,-q,q),ticket.net.REACTION_TICKS)
    return view,s,score,trace,baseline_trace,torch.stack(history,1)


def sensed(t,trace,gain,layout,roots):
    value=cov.projection(trace,t["preview"])
    return binding.instrument(t,value,gain.flatten().repeat_interleave(len(value)//roots),layout)


def update(g,r,gain,eta,opt,t,arm,layout,worlds):
    params=(g.weights,g.bias,*r.parameters(),gain,eta)
    for precise in (False,True):
        opt.zero_grad(set_to_none=True)
        if precise:
            m=ticket.net.Graph(torch.nn.Parameter(g.weights.detach().double()),torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double();k=torch.nn.Parameter(gain.detach().double());j=torch.nn.Parameter(eta.detach().double())
        else:m,rule,k,j=g,r,gain,eta
        view,s,prior,trace,_,_=prefix(m,rule,t,j,arm)
        aug=sensed(t,trace,k,layout,len(g.bias));z=ticket.finish(view,s,aug,"native")
        w=t["weight"].reshape(len(g.bias),worlds,16)
        def term(score,earned):
            p=earned.detach().reshape_as(w)
            avg=(p*w).sum(2,keepdim=True)
            baseline=(avg.sum(1,keepdim=True)-avg)/(worlds-1)
            return (w*score.reshape_as(w)*(p-baseline)).sum(2).mean(1)
        loss=-(term(prior+z["route_score"],z["utility"])+term(z["following_score"],z["gross"])).sum()
        if not loss.isfinite():continue
        loss.backward();origin=(m.weights,m.bias,*rule.parameters(),k,j)
        norm=ticket.storage._gradients(origin,2.)
        if norm is None:continue
        if precise:
            for dest,source in zip(params,origin,strict=True):dest.grad=None if source.grad is None else source.grad.to(dest.dtype)
        opt.step()
        stats=[(z[key].float().reshape_as(w)*w).sum(2).mean(1) for key in ("gross","utility","buy")]
        return torch.stack((*stats,norm,gain.detach().flatten(),eta.detach().flatten(),torch.full_like(norm,precise)),1).detach()
    raise FloatingPointError("Nonfinite agreement-dependent fast update")


def parent(cohort,layout):
    return (cov.OUT,"evidence") if layout=="bilateral" else (binding.OUT,"local")


def fit(cohort,arm,layout):
    torch.set_num_threads(1);ids=ticket.births(cohort);seed=703099 if cohort=="original" else 703127
    path=OUT/layout/cohort;directory=path/arm;directory.mkdir(parents=True,exist_ok=True)
    base,tag=parent(cohort,layout);src=base/cohort
    parents=[ticket.old.load_checkpoint(src,"odor",tag,b,8000) for b in ids]
    w,b,p=(torch.cat([getattr(g,k) for g,r in parents]) for k in ticket.capacity.GRAPH_FIELDS)
    g=ticket.net.Graph(torch.nn.Parameter(w),torch.nn.Parameter(b),p);r=ticket.net.Rule(ids)
    r.load_state_dict({k:torch.cat([rule.state_dict()[k] for _,rule in parents]) for k in r.state_dict()})
    gainfile=src/tag/"gain-8000.npz"
    with np.load(gainfile) as zz:gain=torch.nn.Parameter(torch.tensor(zz["gain"],dtype=g.bias.dtype).reshape(len(ids),1))
    eta=torch.nn.Parameter(torch.zeros_like(gain),requires_grad=arm!="count")
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters(),gain,eta),lr=.002)
    digest=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    cfg=dict(cohort=cohort,arm=arm,layout=layout,seed=seed,worlds=3,lr=.002,stages=list(STAGES),
             plan=digest(PLAN),sources={Path(f).name:digest(f) for f in (__file__,ticket.__file__,cov.__file__,binding.__file__,ticket.net.__file__)},
             parents={str(b):digest(ticket.old.checkpoint_path(src,"odor",tag,b,8000)) for b in ids},gain=digest(gainfile))
    mf=directory/"run.json"
    if mf.exists() and json.loads(mf.read_text())!=cfg:raise ValueError("Agreement study changed")
    if not mf.exists():ticket.storage._write_json(mf,cfg)
    key=digest(mf);resume=directory/"resume.pt";step=0;record=[]
    if resume.exists():
        obj=torch.load(resume,weights_only=True,map_location="cpu")
        if obj["manifest"]!=key:raise ValueError("Incompatible agreement checkpoint")
        with torch.no_grad():g.weights.copy_(obj["w"]);g.bias.copy_(obj["b"]);gain.copy_(obj["gain"]);eta.copy_(obj["eta"])
        r.load_state_dict(obj["rule"]);opt.load_state_dict(obj["optimizer"]);step=obj["step"];record=obj["trace"].tolist()
    def save():
        if step in STAGES:
            for i,b in enumerate(ids):ticket.old.save_checkpoint(path,"odor",arm,b,step,g,r,i)
            ticket.storage._write_raw(directory/f"coupling-{step}.npz",gain=gain.detach().numpy(),eta=eta.detach().numpy(),births=ids)
        obj=dict(manifest=key,step=step,w=g.weights.detach(),b=g.bias.detach(),present=g.present,
                 gain=gain.detach(),eta=eta.detach(),rule=r.state_dict(),optimizer=opt.state_dict(),trace=torch.tensor(record))
        ticket.storage._atomic(resume,lambda f:torch.save(obj,f))
    if not resume.exists():save()
    first=step;t0=perf_counter()
    for step in range(first+1,12001):
        t=ticket.join([tape(seed,b,step,3) for b in ids])
        record.append(update(g,r,gain,eta,opt,t,arm,layout,3).tolist())
        if step%250==0:
            save();print(cohort,arm,layout,step,np.array(record[-100:])[:,:,:3].mean((0,1)).round(3),
                         round((perf_counter()-t0)/(step-first),3),flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__);parser.add_argument("--cohort",choices=("original","independent"),required=True)
    parser.add_argument("--arm",choices=ARMS,required=True);parser.add_argument("--layout",choices=("bilateral","local"),required=True)
    a=parser.parse_args();fit(a.cohort,a.arm,a.layout)
