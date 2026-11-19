"""Learn a sensory relation and how signed past examples should affect later tastes.

A fixed number of recent physical contacts are stored in each life. A single
modifiable connection transforms the recalled signed taste into a new predicted
taste. Its update has access to the residual on the currently touched item.
Neither the odor transform nor the meaning of its relationship is a model input.
"""
from dataclasses import dataclass
import torch
from torch import nn
from . import mixture_circuit as base


@dataclass(frozen=True)
class Config:
    nodes:int=24
    width:int=48
    latent:int=16
    capacity:int=20
    graph_seed:int=60673
    graph:str="rotations"
    change:str="delta"
    sensory_loss:float=.30


def affine_history(a,b):
    """State before each event for scalar affine updates from birth, with state initially zero."""
    aa,bb=a,b
    shift=1
    while shift<a.shape[1]:
        bb=torch.cat([bb[:,:shift],bb[:,shift:]+aa[:,shift:]*bb[:,:-shift]],1)
        aa=torch.cat([aa[:,:shift],aa[:,shift:]*aa[:,:-shift]],1)
        shift*=2
    return torch.cat([torch.zeros_like(b[:,:1]),bb[:,:-1]],1)


class Memory(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.sensor=base.TasteCircuit(base.Config(nodes=cfg.nodes,graph=cfg.graph,graph_seed=cfg.graph_seed))
        for name in ("branch_a","branch_b","read","write","coefficients","log_ridge"):delattr(self.sensor,name)
        self.a=nn.Linear(cfg.nodes,cfg.width)
        self.b=nn.Linear(cfg.nodes,cfg.width)
        self.read=nn.Linear(cfg.width,cfg.latent)
        self.write=nn.Linear(cfg.width,cfg.latent)
        self.log_temperature=nn.Parameter(torch.tensor(1.2))
        self.null=nn.Parameter(torch.tensor(.45))
        self.log_rate=nn.Parameter(torch.tensor(-.15))
        self.log_hold=nn.Parameter(torch.tensor(2.4))
        self.log_scale=nn.Parameter(torch.tensor(.20))
        # A discovered two-input local rule may also use the signed product
        # of the recalled taste and outcome as ordinary pre/post coactivation.
        if cfg.change=="learned":
            self.rule=nn.Sequential(nn.Linear(4,12),nn.Tanh(),nn.Linear(12,2))
            nn.init.normal_(self.rule[-1].weight,std=.08)
            nn.init.zeros_(self.rule[-1].bias)
        if cfg.change not in ("learned","delta","fixed"):raise ValueError(cfg.change)

    def heads(self,states):
        h=(self.a(states)*(1.+self.b(states).tanh())).tanh()
        return nn.functional.normalize(self.read(h),dim=-1),nn.functional.normalize(self.write(h),dim=-1)

    def scalars(self):
        return (2.+22.*self.log_temperature.sigmoid(),.30+.59*self.null.sigmoid(),
                .045+1.4*self.log_rate.sigmoid(),.75+.249*self.log_hold.sigmoid(),
                .45+2.7*self.log_scale.sigmoid())

    def retrieved(self,read,write,signed):
        temp,null,_,_,_=self.scalars()
        similarity=read@write.transpose(1,2)
        n=read.shape[1];t=torch.arange(n,device=read.device)
        gap=t[:,None]-t[None,:]
        legal=(gap>0)&(gap<=self.cfg.capacity)
        logit=(temp*similarity).masked_fill(~legal,-1e4)
        scores=torch.cat((logit,(temp*null).expand(*logit.shape[:2],1)),2)
        probs=scores.softmax(2)
        value=(probs[:,:,:-1]*signed[:,None,:]).sum(2)
        # Evidence that a previously experienced stimulus predicts this public
        # sensory pattern; subtracting count controls different history sizes.
        count=legal.sum(1).clamp(min=1)
        evidence=torch.logsumexp(scores,2)-count.log()
        return value, evidence, 1.-probs[:,:,-1]

    def local(self,recalled,sign,confidence):
        _,_,rate,hold,scale=self.scalars()
        if self.cfg.change=="fixed":return torch.ones_like(recalled),torch.ones_like(recalled)
        if self.cfg.change=="delta":
            b=rate*sign*recalled
            a=hold-rate*recalled.square()
        else:
            inp=torch.stack((sign,recalled,sign*recalled,confidence),-1)
            params=self.rule(inp)
            a=.55+.449*params[:,:,0].sigmoid()
            b=.9*params[:,:,1].tanh()
        c=affine_history(a,b)
        return (scale*c).tanh(),c

    def forward(self,batch,times,shuffled=None):
        target=batch["reward"].float()
        side=batch["side"].long()[:,:,None,None].expand(-1,-1,1,self.cfg.nodes)
        touch=self.sensor.states(batch,batch["touch"].long()).gather(2,side).squeeze(2)
        early=self.sensor.states(batch,times).gather(2,side).squeeze(2)
        qr,kw=self.heads(touch)
        qearly,_=self.heads(early)
        source,pos,conf=self.retrieved(qr,kw,target)
        forecast,_,_=self.retrieved(qearly,kw,target)
        gain,internal=self.local(source,target,conf)
        if shuffled is None:
            sensory=pos.sum()*0.
        else:
            other=qr[shuffled,torch.arange(qr.shape[1],device=qr.device)[None,:]]
            _,negative,_=self.retrieved(other,kw,target)
            # A contrast between two genuinely observed future sensory samples.
            # No pair identity, rule, side label or alternative taste is supplied.
            sensory=nn.functional.softplus(negative[:,2:]-pos[:,2:]).mean()
        return gain*forecast,sensory,internal,forecast

    def export(self,cut=None):
        def arr(x):return x.detach().cpu().numpy().copy()
        def layer(m):return arr(m.weight.T),arr(m.bias)
        return dict(matrix=arr(self.sensor.matrix(cut)),sensors=arr(self.sensor.sensors),
                    a=layer(self.a),b=layer(self.b),read=layer(self.read),write=layer(self.write),
                    scalars=[float(v.detach()) for v in self.scalars()],latent=self.cfg.latent,capacity=self.cfg.capacity,
                    change=self.cfg.change,
                    rule=([layer(self.rule[0]),layer(self.rule[2])] if self.cfg.change=="learned" else None))
