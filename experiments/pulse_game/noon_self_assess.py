"""Forecast an actual attempted answer, then learn whether self-confidence helps select a revisit."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
from torch import nn

from . import noon_online_readout as online, noon_readback as bank
from . import noon_review_world as w, local_trace_graph as net, revisit_choice_graph as motor


SEEDS=(1723721,1723727,1723741);ARMS=("forecast","uncredited","shuffled")
OUT=w.DATA/"self-assessment";PLAN=w.EVIDENCE/"self-assessment-plan.json"


class Assessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecast=nn.Parameter(torch.tensor([.0,.12,-.07],dtype=torch.double))
        self.choice=nn.Parameter(torch.zeros(2,dtype=torch.double))

    def forward(self,z,side,menu):
        # The committed choice and native probability refer to the SECOND
        # orientation; the first orientation has the opposite left/right meaning.
        response=self.forecast[1]*z[:,:,1]/5-self.forecast[2]*z[:,:,0]/5
        actual=self.forecast[0]+side*response
        prob=z[:,:,1].sigmoid()
        estimated=prob*(self.forecast[0]+response).sigmoid()+(1-prob)*(self.forecast[0]-response).sigmoid()
        gate=self.choice[0]*(estimated[:,1]-estimated[:,0])*5+self.choice[1]*menu
        return actual,estimated,gate


@torch.no_grad()
def observe(g,r,t,first,draws):
    n=t.lanes;view=r.expand(n//len(g.bias));state,_=w.introduce(view,net.start(g,n//len(g.bias)),t)
    row=w._rows(t);z=t.odor.new_empty((n,2,2));side=t.odor.new_empty((n,2));pay=t.odor.new_empty((n,2))
    for k in range(2):
        slot=first if k==0 else 1-first
        item=t.candidates[row,slot]
        sign=t.menu_orientation[row,slot]
        for j in range(2):
            orientation=sign if j==0 else -sign
            l,rv=w.sensory_for(t,item,orientation)
            state=net._advance(view,state,net.native.sensory(state.graph,l,rv),net.REACTION_TICKS)
            z[row,slot,j]=2.5*(state.hidden[:,11]-state.hidden[:,10])
        tentative=(draws[row,slot]<z[row,slot,1].sigmoid()).to(z.dtype)*2-1
        side[row,slot]=tentative
        pay[row,slot]=(tentative*t.law[row,item]*(-sign)>0).float()
        state=w.echo_choice(view,state,tentative)
    route=w.select(view,state,*w.candidate_menu(t),t.decision_draw)
    return route,z,side,pay


def random_inputs(seed,step,lanes):
    rng=np.random.default_rng([643107,seed,step])
    return (torch.from_numpy(rng.random((lanes,2),dtype=np.float32)),torch.from_numpy(rng.random(lanes)),
            torch.from_numpy(rng.integers(2,size=lanes)))


def update(g,r,model,opt,seed,step,arm):
    worlds=4;t,first=online.make(seed,step,worlds)
    tentative_draw,route_draw,assessed=random_inputs(seed,step,t.lanes)
    choice,z,side,prepaid=observe(g,r,t,first,tentative_draw)
    q,c,gate=model(z.double(),side.double(),choice.motor_logit.double());p=gate.sigmoid()
    executed=torch.where(route_draw<p.detach(),1.,-1.)[:,None]
    actual=online.execute(g,r,t,choice,executed)[:,0]
    earned=actual.double().reshape(len(g.bias),worlds,8)
    means=earned.mean(2,keepdim=True)
    advantage=earned-(means.sum(1,keepdim=True)-means)/(worlds-1)
    policy=-(torch.nn.functional.logsigmoid(executed[:,0]*gate)*advantage.flatten()).mean()
    row=torch.arange(t.lanes);prediction=q[row,assessed]
    target=prepaid[row,assessed].double()
    if arm=="shuffled":label=target.reshape(len(g.bias),worlds,8).roll(1,1).roll(3,2).reshape(-1)
    else:label=target
    forecast=torch.nn.functional.binary_cross_entropy_with_logits(prediction,label)
    kl=(p*(2*p).clamp_min(1e-12).log()+(1-p)*(2*(1-p)).clamp_min(1e-12).log()).mean()
    reg=torch.cat([v.flatten() for v in model.parameters()]).square().mean()
    loss=policy+.002*kl+.0001*reg
    if arm!="uncredited":loss=loss+forecast
    opt.zero_grad(set_to_none=True);loss.backward()
    if not all(torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError("Nonfinite self-forecast gradient")
    opt.step()
    return [actual.mean().item(),target.mean().item(),((prediction.detach().sigmoid()-target)**2).mean().item(),
            p.detach().mean().item(),model.choice[0].detach().item()]


def fit(seed,arm):
    torch.set_num_threads(1);out=OUT/str(seed)/arm;out.mkdir(parents=True,exist_ok=True)
    sha=lambda f:sha256(Path(f).read_bytes()).hexdigest()
    spec=dict(seed=seed,arm=arm,steps=4000,worlds=4,lr=.015,plan=sha(PLAN),
              sources={Path(f).name:sha(f) for f in (__file__,online.__file__,bank.__file__,w.__file__,net.__file__)})
    meta=out/"run.json"
    if meta.exists() and json.loads(meta.read_text())!=spec:raise ValueError("Self-forecast protocol changed")
    if not meta.exists():bank.lesson.storage._write_json(meta,spec)
    body,rule=online.parents.inherited("original");body=body.detach();rule.requires_grad_(False)
    model=Assessor();opt=torch.optim.Adam(model.parameters(),lr=.015)
    step=0;record=[];key=sha(meta);save=out/"resume.pt"
    if save.exists():
        old=torch.load(save,weights_only=True,map_location="cpu")
        if old["protocol"]!=key:raise ValueError("Self-forecast optimizer mismatch")
        model.load_state_dict(old["model"]);opt.load_state_dict(old["optimizer"]);step=old["step"];record=old["trace"].tolist()
    first=step;t0=perf_counter()
    for step in range(first+1,4001):
        record.append(update(body,rule,model,opt,seed,step,arm))
        if step%200==0:
            data=dict(step=step,protocol=key,model=model.state_dict(),optimizer=opt.state_dict(),trace=torch.tensor(record))
            if step in (1000,2000,4000):bank.lesson.storage._atomic(out/f"step-{step}.pt",lambda f:torch.save(data,f))
            bank.lesson.storage._atomic(save,lambda f:torch.save(data,f))
            print(seed,arm,step,np.mean(record[-100:],0).round(3),round((perf_counter()-t0)/(step-first),3),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    a=p.parse_args();fit(a.seed,a.arm)
