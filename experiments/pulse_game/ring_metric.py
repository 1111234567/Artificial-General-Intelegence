"""Learn a sensory similarity for plastic synapses from experienced one-bit bite outcomes.

The simulation acts at every tick. At a physical bite it retains the currently
available neural features as eligibility, and updates its fast weights when the
next observation reports the outcome. No private target, option value for an
unsampled fruit, cue index lookup, nor a navigation teacher is passed to the model.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ring_world as ring


@dataclass(frozen=True)
class Settings:
    neurons:int=48
    latent:int=24
    hidden:int=64
    graph_seed:int=1441
    # A variant can retain the all-sensory activity at the public start event.
    onset_trace:bool=False
    raw_skip:bool=True

    @property
    def inputs(self):
        return (ring.OBS if self.raw_skip else 0)+self.neurons*(2+int(self.onset_trace))


class Perception:
    """Sparse directed LIF graph with fast and slow population spike traces."""
    def __init__(self,cfg,cut=False):
        self.cfg=cfg
        rng=np.random.default_rng(cfg.graph_seed)
        n=cfg.neurons
        self.inputs=(rng.normal(.03,.46,(ring.OBS,n))*(rng.random((ring.OBS,n))<.60)).astype(np.float32)
        edges=rng.random((n,n))<.14
        np.fill_diagonal(edges,False)
        edges[np.arange(n),(np.arange(n)+1)%n]=True
        signs=np.where(rng.random(n)<.27,-1.7,1.)
        self.rec=(edges*rng.uniform(.025,.23,(n,n))*signs[:,None]*(not cut)).astype(np.float32)
        self.back=(rng.normal(0,.053,(3,n))*(rng.random((3,n))<.23)*(not cut)).astype(np.float32)
        self.leak=rng.uniform(.42,.91,n).astype(np.float32)
        self.bias=rng.uniform(.04,.14,n).astype(np.float32)
        self.threshold=rng.uniform(.65,1.12,n).astype(np.float32)

    def reset(self,n):
        self.h=np.zeros((n,self.cfg.neurons),np.float32)
        self.spikes=np.zeros_like(self.h)
        self.short=np.zeros_like(self.h);self.long=np.zeros_like(self.h)
        self.onset=np.zeros_like(self.h)

    def step(self,obs,prior_motor):
        self.h=np.maximum(self.leak*self.h+obs@self.inputs+self.spikes@self.rec+
                          prior_motor@self.back+self.bias,-2.)
        self.spikes=(self.h>=self.threshold).astype(np.float32)
        self.h*=1.-self.spikes
        self.short+=.36*(self.spikes-self.short)
        self.long+=.055*(self.spikes-self.long)
        parts=[]
        if self.cfg.raw_skip:parts.append(obs)
        parts.extend([self.short,self.long])
        if self.cfg.onset_trace:
            # All channels were available to the spiking graph, and none of
            # their semantic identities is selected by the sample-and-hold.
            if np.all(obs[:,ring.NEW]):
                self.onset[:]=self.spikes+self.h
            parts.append(self.onset)
        return np.concatenate(parts,axis=1)


class Metric(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.address=nn.Sequential(nn.Linear(cfg.inputs,cfg.hidden),nn.Tanh(),
                                   nn.Linear(cfg.hidden,cfg.latent))
        self.rate=nn.Parameter(torch.tensor(0.25))
        self.prior=nn.Parameter(torch.tensor(-.5))
        self.logit_gain=nn.Parameter(torch.tensor(1.4))

    def keys(self,x):
        return F.normalize(self.address(x),dim=-1,eps=1e-7)

    def predict_events(self,x,y,valid,*,parallel=True):
        keys=self.keys(x)
        step=1.8*self.rate.sigmoid()
        if not parallel:
            fast=keys.new_zeros(x.size(0),self.cfg.latent)
            values=[]
            for k,r,mask in zip(keys.unbind(1),y.unbind(1),valid.unbind(1),strict=True):
                q=self.prior+(fast*k).sum(1)
                values.append(q)
                fast=fast+(step*(r-q)*mask)[:,None]*k
            return torch.stack(values,1)
        # m_j = step * sum_{i<j} e_i k_i for present events.
        # (I + step * strictly_lower(K K^T) * mask_i) e = y - prior.
        gram=(keys@keys.transpose(1,2)).tril(-1)*step*valid[:,None,:]
        identity=torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
        errors=torch.linalg.solve_triangular(gram+identity,(y-self.prior).unsqueeze(2),
                                               upper=False,unitriangular=True)
        return self.prior+(gram@errors).squeeze(-1)

    def export(self):
        a,b=self.address[0],self.address[2]
        def cpu(t):return t.detach().cpu().numpy().copy()
        return {"first":cpu(a.weight.T),"b1":cpu(a.bias),"second":cpu(b.weight.T),"b2":cpu(b.bias),
                "rate":float((1.8*self.rate.sigmoid()).detach().cpu()),
                "prior":float(self.prior.detach().cpu())}


@dataclass(frozen=True)
class Appetite:
    # These values control exploration/action timing, not fruit identity.
    threshold:float=.10
    gain:float=1.1
    random_bite:float=.04
    urgency:float=.018


class MetricForager:
    def __init__(self,cfg,params,seeds,*,appetite=Appetite(),learn=True,explorer=False,
                 cut_recurrence=False,cut_fast=False,record=False):
        self.cfg,self.params,self.seeds=cfg,params,tuple(map(int,seeds))
        self.sense=Perception(cfg,cut_recurrence)
        self.appetite,self.learn,self.explorer,self.cut_fast=appetite,learn,explorer,cut_fast
        self.record=record

    def reset(self,n):
        if n!=len(self.seeds):raise ValueError("Need independent seeds per policy life")
        self.sense.reset(n)
        self.fast=np.zeros((n,self.cfg.latent),np.float32)
        self.last_features=np.zeros((n,self.cfg.inputs),np.float32)
        self.last_key=np.zeros((n,self.cfg.latent),np.float32)
        self.last_pred=np.zeros(n,np.float32)
        self.last_bite=np.zeros(n,bool)
        self.mv=np.zeros((n,3),np.float32)
        self.motor=np.zeros_like(self.mv)
        self.hunger=np.zeros(n,np.float32)
        self.rng=[np.random.default_rng(np.random.SeedSequence([s,0x331904])) for s in self.seeds]
        self.noise=np.empty((0,n),np.float32); self.index=0
        self.age=-1
        self.events=[[] for _ in range(n)] if self.record else None

    def coin(self):
        if self.index==len(self.noise):
            self.noise=np.stack([r.random(128,dtype=np.float32) for r in self.rng],axis=1);self.index=0
        value=self.noise[self.index];self.index+=1
        return value

    def act(self,obs):
        if np.all(obs[:,ring.NEW]):self.age+=1
        reward=obs[:,ring.POSITIVE]-obs[:,ring.NEGATIVE]
        event=(reward!=0)&self.last_bite
        if self.record and np.any(event):
            for i in np.flatnonzero(event):
                self.events[i].append((self.last_features[i].copy(),float(reward[i]),self.age))
        if self.learn and self.params is not None:
            self.fast[event]+=(self.params["rate"]*(reward[event]-self.last_pred[event]))[:,None]*self.last_key[event]
        features=self.sense.step(obs,self.motor)
        if self.params is not None:
            p=self.params
            key=np.tanh(features@p["first"]+p["b1"])@p["second"]+p["b2"]
            key/=np.maximum(np.linalg.norm(key,axis=-1,keepdims=True),1e-7)
            pred=p["prior"]+(0.*self.last_pred if self.cut_fast else (self.fast*key).sum(1))
        else:
            key=self.last_key
            pred=np.full_like(self.last_pred,-.5)
        reaches=obs[:,ring.REACH];active=obs[:,ring.ACTIVE]
        self.hunger=np.clip(.985*self.hunger+self.appetite.urgency*active-1.0*np.abs(reward),0.,.8)
        toss=self.coin()
        self.mv*=.3
        self.mv[:,1]+=1.12*active-.35*(1.-active)
        if self.explorer:
            drive=np.where(toss<self.appetite.random_bite,1.5,0.).astype(np.float32)
        else:
            drive=.85+self.appetite.gain*(pred-self.appetite.threshold)+self.hunger
            drive+=1.9*(toss<self.appetite.random_bite)
        self.mv[:,2]+=drive*reaches-.7*(1.-reaches)
        np.maximum(self.mv,-.6,out=self.mv)
        self.motor=(self.mv>=1.).astype(np.float32)
        self.mv*=1.-self.motor
        self.last_bite=(self.motor[:,2]>0)&(reaches>0)&(active>0)
        self.last_key=np.asarray(key,dtype=np.float32)
        self.last_pred=pred
        if self.record:self.last_features=features
        return self.motor


def event_batch(events,bound):
    """Chronological action/outcome samples with padding excluded from all updates and losses."""
    n=len(events)
    first=next(v[0][0] for v in events if len(v))
    x=np.zeros((n,bound,len(first)),np.float32)
    y=np.zeros((n,bound),np.float32)
    valid=np.zeros_like(y)
    for i,items in enumerate(events):
        for j,(features,reward,_) in enumerate(items[:bound]):
            x[i,j]=features;y[i,j]=reward;valid[i,j]=1.
    return x,y,valid
