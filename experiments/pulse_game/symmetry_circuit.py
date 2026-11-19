"""Competing sets of local plastic synapses learn which odor relation a life obeys."""
from dataclasses import dataclass
import torch
from torch import nn
from . import mixture_circuit as base


@dataclass(frozen=True)
class Spec:
    mechanism:str="evidence"
    experts:int=2
    nodes:int=24
    latent:int=20
    width:int=56
    graph:str="rotations"
    graph_seed:int=70691


class Circuit(nn.Module):
    def __init__(self,spec):
        super().__init__()
        self.spec=spec
        if spec.mechanism not in ("evidence","uniform","single"):raise ValueError(spec.mechanism)
        self.e=1 if spec.mechanism=="single" else spec.experts
        self.sense=base.TasteCircuit(base.Config(nodes=spec.nodes,width=spec.width,latent=spec.latent,
                                                  graph=spec.graph,graph_seed=spec.graph_seed))
        # Use only the stable continuous graph from the sensory module.
        for name in ("branch_a","branch_b","read","write","coefficients","log_ridge"):delattr(self.sense,name)
        self.a=nn.Linear(spec.nodes,spec.width*self.e)
        self.b=nn.Linear(spec.nodes,spec.width*self.e)
        self.out=nn.Parameter(torch.randn(self.e,spec.width,spec.latent)*.14)
        self.offset=nn.Parameter(torch.zeros(self.e,spec.latent))
        self.rate=nn.Parameter(torch.zeros(self.e))
        self.retention=nn.Parameter(torch.linspace(.7,1.5,self.e))
        self.temperature=nn.Parameter(torch.tensor(1.15))
        self.confidence_retention=nn.Parameter(torch.tensor(1.9))
        self.prior=nn.Parameter(torch.zeros(self.e))

    def dynamics(self):
        return (.12+1.5*self.rate.sigmoid(),.10+.899*self.retention.sigmoid(),
                .50+.499*self.confidence_retention.sigmoid(),.25+8.*self.temperature.sigmoid())

    def keys(self,state):
        dims=(*state.shape[:-1],self.e,self.spec.width)
        hidden=(self.a(state).reshape(dims)*(1.+self.b(state).reshape(dims).tanh())).tanh()
        raw=torch.einsum("...eh,ehk->...ek",hidden,self.out)+self.offset
        return raw/torch.sqrt(.13+raw.square().sum(-1,keepdim=True))

    def sequence(self,write_key,read_key,outcome):
        # Inputs (batch,time,experts,key). The same physical contact is offered
        # to each population; they can predict it differently from earlier signs.
        x=write_key.permute(0,2,1,3)
        r=read_key.permute(0,2,1,3)
        y=outcome[:,None,:,None]
        rate,forget,hold,temp=self.dynamics()
        n=x.shape[2]
        t=torch.arange(n,device=x.device)
        lag=t[:,None]-t[None,:]-1
        memory=forget[:,None,None].pow(lag.clamp(min=0))*rate[:,None,None]*(lag>=0)
        gram=(x@x.transpose(-1,-2))*memory
        identity=torch.eye(n,device=x.device)
        errors=torch.linalg.solve_triangular(gram+identity,y.expand(-1,self.e,-1,-1),
                                              upper=False,unitriangular=True)
        contact_values=(gram@errors).squeeze(-1)
        values=((r@x.transpose(-1,-2)*memory)@errors).squeeze(-1)
        if self.e>1 and self.spec.mechanism=="evidence":
            # Confidence for event j uses only earlier actual prediction
            # errors. Centering across populations removes common losses.
            differences=(contact_values-y.squeeze(-1)).square()
            differences=differences-differences.mean(1,keepdim=True)
            confidence=hold.pow(lag.clamp(min=0))*(lag>=0)
            logits=-temp*(differences@confidence.T)+self.prior[None,:,None]
            prob=logits.softmax(1)
        else:
            prob=torch.ones_like(values)/self.e
        return (prob*values).sum(1),values,prob,contact_values

    def forward(self,batch,times):
        side=batch["side"].long()[:,:,None,None].expand(-1,-1,1,self.spec.nodes)
        full=self.sense.states(batch,batch["touch"].long()).gather(2,side).squeeze(2)
        early=self.sense.states(batch,times).gather(2,side).squeeze(2)
        return self.sequence(self.keys(full),self.keys(early),batch["reward"].float())

    def export(self,cut=None):
        def arr(v):return v.detach().cpu().numpy().copy()
        def layer(q):return arr(q.weight.T),arr(q.bias)
        rate,forget,hold,temp=self.dynamics()
        return dict(matrix=arr(self.sense.matrix(cut)),sensors=arr(self.sense.sensors),a=layer(self.a),b=layer(self.b),
                    out=arr(self.out),offset=arr(self.offset),rate=arr(rate),forget=arr(forget),
                    hold=float(hold.detach()),temp=float(temp.detach()),prior=arr(self.prior),
                    mechanism=self.spec.mechanism,experts=self.e,width=self.spec.width,latent=self.spec.latent)
