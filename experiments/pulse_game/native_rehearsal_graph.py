"""One recurrent graph. The environment supplies sensory pulses and later asks the motor ports to act."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import edge_site_train as parent


@dataclass
class State:
    h: torch.Tensor
    fast: torch.Tensor
    trace: torch.Tensor


class NativeGraph(nn.Module):
    """Both populations run the same equation on every tick, including silence."""
    def __init__(self, birth, layout):
        super().__init__()
        g,_=parent.load_parent(birth)
        n=20 if layout=="single" else 40
        w=torch.zeros(n,n)
        mask=torch.zeros(n,n,dtype=torch.bool)
        bias=torch.zeros(n)
        w[:20,:20]=g.weights[0]
        mask[:20,:20]=g.present[0]
        bias[:20]=g.bias[0]
        generator=torch.Generator().manual_seed(37109+birth)
        if n==40:
            order=torch.randperm(20,generator=generator) if layout=="permuted" else torch.arange(20)
            w[20:,20:]=g.weights[0][:,order]
            mask[20:,20:]=g.present[0][:,order]
            bias[20:]=g.bias[0]
            # The same bridges in the matched 40-node starts. Each population has four afferents
            # per node from the other. Recurrent currents can flow in both directions on every tick.
            bridge_rng=torch.Generator().manual_seed(9007+birth)
            rows=torch.arange(20)
            for target,source in ((20,0),(0,20)):
                for offset in (0,3,8,13):
                    cols=(rows+offset)%20
                    mask[rows+target,cols+source]=True
                    w[rows+target,cols+source]=.22*torch.randn(20,generator=bridge_rng)
        self.width=n
        self.register_buffer("mask",mask)
        self.weight=nn.Parameter(w.detach())
        self.bias=nn.Parameter(bias.detach())
        self.gate=nn.Parameter(torch.zeros(n,n))
        with torch.no_grad():
            # Opposed taste ports already exist on the primary body.
            self.gate[:,8]=.6
            self.gate[:,9]=-.6
        self.keep_logit=nn.Parameter(torch.full((n,),.7))
        self.edit_rate=nn.Parameter(torch.full((n,),-2.8))
        self.post_bias=nn.Parameter(torch.zeros(n))
        self.leak_logit=nn.Parameter(torch.full((n,),.85))

    def start(self,batch):
        h=self.bias.new_zeros((batch,self.width))
        return State(h,h.new_zeros((batch,self.width,self.width)),h)

    def tick(self,s,x,*,feedback_override=None,disable_feedback=False):
        # Gates read the same available recurrent edges as the main synaptic current.
        weights=self.weight*self.mask
        current=F.linear(s.h,weights)+torch.bmm(s.fast,s.h.unsqueeze(2)).squeeze(2)
        back=None
        if self.width==40:
            back=(F.linear(s.h[:,20:],weights[:20,20:])
                  +torch.bmm(s.fast[:,:20,20:],s.h[:,20:,None]).squeeze(2))
            if feedback_override is not None or disable_feedback:
                replacement=torch.zeros_like(back) if feedback_override is None else feedback_override
                current=torch.cat((current[:,:20]-back+replacement,current[:,20:]),1)
        x=F.pad(x,(0,self.width-10))
        h=s.h+self.leak_logit.sigmoid()*(torch.tanh(current+self.bias+x)-s.h)
        trace=s.trace+self.keep_logit.neg().sigmoid()*(s.h-s.trace)
        mod=F.linear(h,self.gate*self.mask).tanh()
        post=(h+self.post_bias).tanh()*mod*self.edit_rate.sigmoid()
        delta=post[:,:,None]*trace[:,None,:]*self.mask
        return State(h,(s.fast+delta).clamp(-1.15,1.15),trace),back

    def advance(self,s,x,ticks):
        for _ in range(ticks):
            s,_=self.tick(s,x)
        return s

    def logit(self,s):
        return 2.5*(s.h[:,11]-s.h[:,10])

    def contact(self,s,right,payoff,draw,*,terminal=False):
        z=right.new_zeros((len(right),4))
        menu=torch.cat((-right,right,z),1)
        s=self.advance(s,menu,3)
        logit=self.logit(s)
        side=torch.where(draw<logit.detach().sigmoid(),1.,-1.)
        earned=side*payoff
        score=F.logsigmoid(side*logit)
        touch=menu.clone()
        touch[:,6]=(side<0).to(right.dtype)
        touch[:,7]=(side>0).to(right.dtype)
        s=self.advance(s,touch,2)
        taste=touch.clone()
        if not terminal:
            taste[:,8]=earned.relu()
            taste[:,9]=(-earned).relu()
        return self.advance(s,taste,2),side,earned,score,logit


@dataclass
class Episode:
    vectors: torch.Tensor
    laws: torch.Tensor
    orientation: torch.Tensor
    draws: torch.Tensor
    first_old: torch.Tensor
    first_query: torch.Tensor
    angle: torch.Tensor


def scenes(seed,count,*,evaluation=False,device="cpu"):
    """Evaluation uses eight-law cells and two mirrored older cues at fixed B and C."""
    rng=np.random.default_rng(seed)
    b=rng.normal(size=(count,3))
    b/=np.linalg.norm(b,axis=1,keepdims=True)
    u=rng.normal(size=(count,3));u-=(u*b).sum(1,keepdims=True)*b
    u/=np.linalg.norm(u,axis=1,keepdims=True)
    c=np.cross(b,u)
    angle=rng.choice((15,30,45,60,75,90),count)
    repeats=16 if evaluation else 1
    def repeat(x):
        return np.repeat(x,repeats,axis=0)
    if evaluation:
        law=np.tile(np.array([[a,cc,bb] for a in (-1,1) for cc in (-1,1) for bb in (-1,1)]*2),(count,1))
        mirror=np.tile(np.repeat([1,-1],8),count)
    else:
        law=rng.choice([-1,1],size=(count,3));mirror=rng.choice([-1,1],count)
    theta=np.deg2rad(repeat(angle))[:,None]
    a=repeat(b)*np.cos(theta)+repeat(u)*np.sin(theta)*mirror[:,None]
    vectors=1.35*np.stack((a,repeat(c),repeat(b)),1)
    def tensor(x,dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(x),dtype=dtype,device=device)
    return Episode(tensor(vectors),tensor(law),tensor(repeat(rng.choice([-1,1],size=(count,5)))),
                   tensor(repeat(rng.random(size=(count,5)))),tensor(repeat(rng.integers(2,size=count)),torch.long),
                   tensor(repeat(rng.integers(2,size=count)),torch.long),tensor(repeat(angle),torch.long))


def support(model,e):
    rows=torch.arange(len(e.laws),device=e.laws.device)
    s=model.start(len(rows));scores=[];visits=[]
    for t,obj in enumerate((e.first_old,1-e.first_old,torch.full_like(e.first_old,2))):
        s=model.advance(s,e.vectors.new_zeros((len(rows),10)),2)
        before=s
        right=e.vectors[rows,obj]*e.orientation[:,t,None]
        payoff=e.laws[rows,obj]*e.orientation[:,t]
        s,side,earned,score,_=model.contact(s,right,payoff,e.draws[:,t])
        # The graph never receives this log. It reconstructs a food vector only for a training loss or audit.
        visits.append((obj,right,side,earned))
        scores.append(score)
    return before,s,torch.stack(scores,1),visits


def silence(model,s,*,ticks=12,override=None,cut=False):
    currents=[];activities=[]
    zero=s.h.new_zeros((len(s.h),10))
    for t in range(ticks):
        s,back=model.tick(s,zero,feedback_override=None if override is None else override[t],
                          disable_feedback=cut)
        if back is not None:currents.append(back)
        activities.append(s.h)
    return s,torch.stack(activities),torch.stack(currents) if currents else None


def queries(model,s,e):
    rows=torch.arange(len(e.laws),device=e.laws.device)
    choices=[];scores=[];pays=[];logits=[]
    for t,rank in enumerate((e.first_query,1-e.first_query),3):
        obj=rank*2
        right=e.vectors[rows,obj]*e.orientation[:,t,None]
        payoff=e.laws[rows,obj]*e.orientation[:,t]
        s,side,earned,score,z=model.contact(s,right,payoff,e.draws[:,t],terminal=True)
        choices.append(side);scores.append(score);pays.append(earned);logits.append(z*payoff)
    order=torch.stack((e.first_query,1-e.first_query),1)
    def canonical(values):
        return torch.stack(values,1).gather(1,order)
    return canonical(choices),canonical(scores),canonical(pays),canonical(logits)


def recorded_favorable(visits,object_id):
    candidates=[right*(side*earned)[:,None] for _,right,side,earned in visits]
    return sum(vec*(obj==object_id)[:,None] for (obj,*_),vec in zip(visits,candidates,strict=True))
