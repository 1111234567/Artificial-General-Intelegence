"""Learn both a hidden cross-object regularity and physically experienced exceptions."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_age_train as age, interleave_credit_train as credit
from . import interleave_train as common, local_trace_graph as net, interleave_susceptibility as susceptibility
from . import local_trace_train as old
from .interleave_interference import DATA,EVIDENCE


ARMS=("structured","unrelated","mixed")
DOMAIN=175817


@dataclass(frozen=True)
class Config:
    updates:int=8000
    worlds:int=4
    seed:int=661759
    lr:float=.002
    cut:int=4
    save_every:int=500


def count(cfg,step):
    return int(np.random.default_rng([DOMAIN,cfg.seed,step,77]).choice((2,4,8,16)))


def tape(cfg,birth,step,k,arm,domain):
    rng=np.random.default_rng([domain,cfg.seed,birth,step])
    n=cfg.worlds
    xyz=age.directions(rng,n,k+2)
    direction=age.directions(rng,n)[:,None]
    relation=np.where((xyz*direction).sum(2)>=0,1.,-1.).astype(np.float32)
    independent=age.signs(rng,(n,k+2))
    flags=rng.random(n)<.5
    structured=np.ones(n,dtype=bool) if arm=="structured" else flags if arm=="mixed" else np.zeros(n,dtype=bool)
    base=np.where(structured[:,None],relation,independent)
    # At half the worlds, a small fraction of prior items have lawful individual exceptions.
    chance=np.where(rng.random(n)<.5,.2,0.)
    exception=np.where(rng.random((n,k))<chance[:,None],-1.,1.).astype(np.float32)
    base[:,:k]*=exception
    crossed=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)],dtype=np.float32)
    signs=np.repeat(base[:,None],8,axis=1)*crossed[None,:,0,None]
    signs[:,:,k]*=crossed[None,:,1]  # X may agree with or contradict the global rule.
    signs[:,:,k+1]=np.where(structured[:,None],signs[:,:,k+1],
                             base[:,None,k+1]*crossed[None,:,2])
    query=k+rng.integers(2,size=n)  # Randomly return to known X or approach never-seen Y.
    ids=np.concatenate((np.tile(np.arange(k+1),(n,1)),query[:,None]),1)
    orient=age.signs(rng,ids.shape)
    def shared(v):return np.repeat(v,8,axis=0)
    return dict(right=shared(xyz[np.arange(n)[:,None],ids]*orient[...,None]),
                factor=(np.take_along_axis(signs,ids[:,None],2)*orient[:,None]).reshape(-1,k+2),
                u=shared(rng.random((n,k+2,5),dtype=np.float32)),
                waits=shared(rng.integers(0,9,size=(n,k+2))),kind=shared(query-k),
                structured=shared(structured),exception=np.tile(crossed[:,1]<0,n))


def fit(out,source,births,arm,cfg,stop):
    torch.set_num_threads(1)
    g,r=susceptibility.parents(source,births)
    g=net.Graph(torch.nn.Parameter(g.weights),torch.nn.Parameter(g.bias),g.present);r.requires_grad_(True)
    params=(g.weights,g.bias,*r.parameters())
    opt=torch.optim.Adam(params,lr=cfg.lr)
    folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
    files=sorted(folder.glob("step-*.pt"));step,history,events=0,[],0
    if files:
        d=torch.load(files[-1],weights_only=True,map_location="cpu")
        if (d["arm"],d["births"],d["config"])!=(arm,list(births),asdict(cfg)):raise ValueError("Wrong checkpoint")
        with torch.no_grad():g.weights.copy_(d["graph"]["weights"]);g.bias.copy_(d["graph"]["bias"])
        r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"])
        step,history,events=d["step"],d["trace"].tolist(),d["contacts"]
    def save():
        payload=dict(config=asdict(cfg),step=step,arm=arm,births=list(births),trace=torch.tensor(history),
                     contacts=events,graph=dict(weights=g.weights.detach(),bias=g.bias.detach(),present=g.present),
                     rule=r.state_dict(),optimizer=opt.state_dict())
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(payload,f))
    if not files:save()
    first=step;started=perf_counter()
    for step in range(first+1,stop+1):
        k=count(cfg,step)
        worlds=[tape(cfg,b,step,k,arm,DOMAIN) for b in births]
        data={key:torch.from_numpy(np.concatenate([w[key] for w in worlds])) for key in worlds[0]}
        opt.zero_grad(set_to_none=True)
        state=net.start(g,8*cfg.worlds);rule=r.expand(8*cfg.worlds);score=0.
        for t in range(k+2):
            if t and t%cfg.cut==0:state=credit.boundary(state,"all")
            if t:state=net.wait(rule,state,data["waits"][:,t],max_ticks=8)
            x,u=data["right"][:,t],data["u"][:,t]
            a=net.act(rule,state,-x,x,u[:,0])
            earned=(a.side*data["factor"][:,t]).detach()
            score=score+a.motor_log_prob
            if t<=k:
                obs=net.observe(rule,a,-x,x,earned)
                result=net.commit(rule,obs,u[:,1:]);state=result.state;score=score+result.edit_log_prob
        reward=earned.reshape(len(births),cfg.worlds,8)
        m=reward.mean(2,keepdim=True)
        advantage=reward-(m.sum(1,keepdim=True)-m)/(cfg.worlds-1)
        objective=-(score.reshape_as(reward)*advantage).mean((1,2)).sum()
        objective.backward()
        with torch.no_grad():
            norm=sum(p.grad.flatten(1).double().square().sum(1) for p in params if p.grad is not None).sqrt()
            if not torch.isfinite(norm).all():raise FloatingPointError(f"{arm}/{step}")
            scale=(2/norm.clamp_min(1e-12)).clamp_max(1)
            for p in params:
                if p.grad is not None:p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)).to(p.dtype))
            kind=data["kind"].reshape_as(reward)
            growth=[(reward>0).float().mean((1,2))]
            for q in (0,1):
                growth.extend((((reward>0)&(kind==q)).sum((1,2)),(kind==q).sum((1,2))))
            stat=torch.stack((*growth,norm.float()),1)
        opt.step();history.append(stat.tolist());events+=8*cfg.worlds*(k+2)
        if step%cfg.save_every==0 or step==stop:save()
        if step%500==0 or step in (1,stop):
            vals=np.array(history[-100:]);print(arm,step,vals[:,:,0].mean().round(3),
                          (vals[:,:,1].sum()/vals[:,:,2].sum()).round(3),(vals[:,:,3].sum()/vals[:,:,4].sum()).round(3),
                          round((perf_counter()-started)/(step-first),3),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DATA/"induction-fit")
    p.add_argument("--source",type=Path,default=DATA/"retention-objective")
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--births",nargs="+",type=int,default=list(old.BIRTHS))
    p.add_argument("--updates",type=int,default=8000);p.add_argument("--stop",type=int)
    p.add_argument("--worlds",type=int,default=4);p.add_argument("--seed",type=int,default=661759)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args();cfg=Config(updates=a.updates,worlds=a.worlds,seed=a.seed);stop=a.stop or a.updates
    if not 1<=stop<=a.updates or a.worlds<2:raise ValueError("Invalid induction budget")
    plan=dict(config=asdict(cfg),births=a.births,arms=a.arms,
              parent_sha256={str(b):sha256(old.checkpoint_path(a.source,"odor","uniform",b,8000)
                                          .read_bytes()).hexdigest() for b in a.births},
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              note="Prior 2/4/8/16 novel objects; then a previously unseen X with independent typical/exception status "
                   "is experienced once; final physical choice randomly asks for X or first-ever Y. Structured world has "
                   "a new hidden direction each life. Unrelated has independent values; Y's law is crossed independently "
                   "of the whole prefix. Mixed draws between these regimes without an indicator. Paid only by actual "
                   "chosen final gain/loss. Physical states fully carried, full stochastic return credit, pathwise "
                   "derivative cut every four contacts. All models start from the same frozen uniform-query graph.")
    a.out.mkdir(parents=True,exist_ok=True)
    f=a.out/"run.json"
    if f.exists() and json.loads(f.read_text())!=plan:raise ValueError("Frozen protocol changed")
    common._atomic_save(f,lambda q:q.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    if a.workers==1:
        for arm in a.arms:fit(a.out,a.source,a.births,arm,cfg,stop)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for j in [pool.submit(fit,a.out,a.source,a.births,arm,cfg,stop) for arm in a.arms]:j.result()


if __name__=="__main__":main()
