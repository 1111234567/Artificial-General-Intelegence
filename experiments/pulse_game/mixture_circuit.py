"""A stable directed recurrent sensory graph and plastic addresses learned from sampled tastes."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    nodes: int = 32
    width: int = 64
    latent: int = 24
    graph: str = "rotations"
    learner: str = "local"
    graph_seed: int = 19391


class TasteCircuit(nn.Module):
    """Rotations on fixed irregular matchings produce a sparse multi-loop transition matrix."""
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        n=cfg.nodes
        if n%2 or cfg.graph not in ("rotations","diagonal","stateless"):
            raise ValueError("Need an even graph or a supported null")
        rng=np.random.default_rng(cfg.graph_seed)
        self.register_buffer("matchings",torch.tensor(np.stack([rng.permutation(n) for _ in range(3)]),dtype=torch.long))
        self.angles=nn.Parameter(.33*torch.randn(3,n//2))
        self.log_decay=nn.Parameter(torch.linspace(-.8,1.65,n))
        self.sensors=nn.Parameter(torch.randn(6,n)*.33)
        self.branch_a=nn.Linear(n,cfg.width)
        self.branch_b=nn.Linear(n,cfg.width)
        self.read=nn.Linear(cfg.width,cfg.latent)
        self.write=nn.Linear(cfg.width,cfg.latent)
        self.coefficients=nn.Linear(cfg.width,3)
        nn.init.normal_(self.coefficients.weight,std=.025)
        self.coefficients.bias.data.copy_(torch.tensor([.40,-.20,0.]))
        self.log_ridge=nn.Parameter(torch.tensor(-.45))
        if cfg.learner not in ("local","ridge","delta"):
            raise ValueError(cfg.learner)

    def matrix(self,cut=None):
        n=self.cfg.nodes
        identity=torch.eye(n,device=self.sensors.device,dtype=self.sensors.dtype)
        if self.cfg.graph=="stateless" or cut=="all":return identity*0
        decay=.28+.70*self.log_decay.sigmoid()
        if self.cfg.graph=="diagonal":return torch.diag(decay)
        q=identity
        for angles, perm in zip(self.angles,self.matchings,strict=True):
            pair=q.index_select(1,perm).reshape(n,-1,2)
            a,b=pair.unbind(-1)
            c,s=angles.cos(),angles.sin()
            pair=torch.stack((a*c-b*s,a*s+b*c),dim=-1)
            q=pair.reshape(n,n).index_select(1,perm.argsort())
        m=q*decay[None,:]
        if cut=="cross":m=torch.diag(m.diag())
        return m

    def powers(self,cut=None):
        m=self.matrix(cut)
        identity=torch.eye(self.cfg.nodes,device=m.device,dtype=m.dtype)
        out=[identity,m]
        for i in range(2,17):out.append(out[i//2]@out[i-i//2])
        return torch.stack(out)

    def states(self,batch,times,cut=None):
        """Exact birth-to-present recurrence at arbitrary ticks in a whole-life tensor."""
        powers=self.powers(cut)
        n,l=times.shape
        odor=batch["odor"]
        ctx=batch["ctx"]
        ot,ct=batch["ot"].long(),batch["ct"].long()
        # For each possible lag, precompute the propagated sensory projection.
        # This avoids materializing a full per-life/per-trial graph matrix.
        kernel=self.sensors[None,:,:]@powers
        def pulses_at(t):
            od=t-ot
            cd=t-ct
            ol=kernel[od.clamp(min=0),:4]
            cl=kernel[cd.clamp(min=0),4:]
            smell=(odor@ol)*(od>=0)[:,:,None,None]
            context=(ctx[:,:,None,:]@cl)*(cd>=0)[:,:,None,None]
            return smell+context
        state=pulses_at(torch.full_like(times,15))
        propagator=powers[16]
        start=1
        while start<l:
            state=torch.cat((state[:,:start],state[:,start:]+state[:,:-start]@propagator),1)
            propagator=propagator@propagator
            start*=2
        prior=torch.cat((torch.zeros_like(state[:,:1]),state[:,:-1]),1)
        # Only 16 possible time indices. Group multiplication by time avoids
        # storing a massive per-trial tensor when lives and graphs grow.
        carried=prior@powers[(times+1).long()]
        return carried+pulses_at(times)

    def features(self,state):
        u=self.branch_a(state)
        z=(u*(1.+self.branch_b(state).tanh())).tanh()
        key=self.read(z)
        normal=lambda k:k/torch.sqrt(.16+k.square().sum(-1,keepdim=True))
        k=normal(key)
        if self.cfg.learner=="ridge":
            return k,k,None
        if self.cfg.learner=="delta":
            a=(1.8*self.coefficients.bias[0].sigmoid()).expand_as(k[...,:1])
            return k,k,torch.cat((a,-a,a*0.),-1)
        w=normal(key+.30*self.write(z))
        c=self.coefficients(z)
        params=torch.stack((1.65*c[...,0].tanh(),-1.65*c[...,1].sigmoid(),.30*c[...,2].tanh()),-1)
        return k,w,params

    def value(self,batch,n_history,query_times):
        contact_time=batch["touch"].long()
        chosen=batch["side"].long()[:,:,None,None].expand(-1,-1,1,self.cfg.nodes)
        address=self.states(batch,contact_time).gather(2,chosen).squeeze(2)
        rq,wq,p=self.features(address)
        obs=self.states(batch,query_times).gather(2,chosen).squeeze(2)
        query,_,_=self.features(obs)
        y=batch["reward"].to(self.sensors.dtype)
        h=n_history
        if self.cfg.learner=="ridge":
            x=rq[:,:h]
            eye=torch.eye(self.cfg.latent,device=x.device,dtype=x.dtype)
            fast=torch.linalg.solve(x.transpose(1,2)@x+(.04+self.log_ridge.exp())*eye,x.transpose(1,2)@y[:,:h,None])
            return (query[:,h:]@fast).squeeze(-1)
        # Each real experience gives a vector update w_j*(a_j*r_j+b_j*q_j+c_j).
        # q_j = sum_{i<j} <rkey_j,wkey_i> update_i, so a triangular solve
        # gives the exact same continuous fast memory as sequential updates.
        keys, writes, p=rq[:,:h], wq[:,:h],p[:,:h]
        gram=(keys@writes.transpose(1,2)).tril(-1)
        matrix=torch.eye(h,device=rq.device,dtype=rq.dtype)-p[:,:,1,None]*gram
        rhs=p[:,:,0]*y[:,:h]+p[:,:,2]
        updates=torch.linalg.solve_triangular(matrix,rhs.unsqueeze(-1),upper=False,unitriangular=True)
        fast=writes.transpose(1,2)@updates
        return (query[:,h:]@fast).squeeze(-1)

    def export(self,cut=None):
        def array(v):return v.detach().cpu().numpy().copy()
        def layer(m):return array(m.weight.T),array(m.bias)
        return dict(graph=array(self.matrix(cut)),input=array(self.sensors),a=layer(self.branch_a),b=layer(self.branch_b),
                    read=layer(self.read),write=layer(self.write),coeff=layer(self.coefficients),
                    ridge=float((.04+self.log_ridge.exp()).detach().cpu()),learner=self.cfg.learner,
                    nodes=self.cfg.nodes,latent=self.cfg.latent)
