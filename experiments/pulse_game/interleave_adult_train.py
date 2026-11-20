"""Teach fresh two-object learning on states reached after many genuine prior writes."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_age_train as age, interleave_train as common, local_trace_graph as net
from . import local_trace_train as previous, self_param_writer as fw
from .interleave_interference import DATA


ARMS=("aged","sham_age")
DOMAIN=513431


@dataclass(frozen=True)
class Config:
    updates:int=8000
    pairs:int=8
    world_seed:int=871423
    lr:float=.0025
    max_wait:int=8
    save_every:int=500


def past_contacts(config,step):
    options=(0,1,4) if step<=1000 else (0,4,16) if step<=2500 else (0,4,16,64) if step<=4500 else (0,4,16,64,128)
    return int(np.random.default_rng([DOMAIN,config.world_seed,step,50]).choice(options))


def tape(config,birth,step,past,domain=DOMAIN):
    rng=np.random.default_rng([domain,config.world_seed,birth,step])
    n=config.pairs
    # Equal-age prefixes: some worlds repeatedly visit a small pool and others see a fresh item every time.
    pool=age.directions(rng,n,max(past,4)+2)
    value=age.signs(rng,(n,max(past,4)+2))
    repeat=rng.random(n)<.5
    items=np.broadcast_to(np.arange(past),(n,past)).copy()
    items[repeat]=rng.integers(4,size=(int(repeat.sum()),past))
    fresh=pool.shape[1]-2
    visit1=rng.integers(2,size=n)
    review=rng.integers(2,size=n)
    query=rng.integers(2,size=n)
    items=np.concatenate((items,fresh+np.stack((visit1,1-visit1,review,query),1)),1)
    rows=np.arange(n)[:,None]
    orientation=age.signs(rng,items.shape)
    right=pool[rows,items]*orientation[...,None]
    factor=value[rows,items]*orientation
    def share(v):return np.repeat(v[:,None],2,axis=1).reshape(2*n,*v.shape[1:])
    return dict(right=share(right),law=np.stack((factor,-factor),1).reshape(2*n,-1),
                wait=share(rng.integers(config.max_wait+1,size=items.shape)),
                u=share(rng.random((n,past+4,5),dtype=np.float32)),
                past_repeat=share(repeat),recent_final=share(review==query))


def batch(config,births,step,past,domain=DOMAIN):
    tapes=[tape(config,b,step,past,domain) for b in births]
    return {k:torch.from_numpy(np.concatenate([v[k] for v in tapes])) for k in tapes[0]}


def rollout(g,rule,data,cfg,past,mode):
    state=net.start(g,cfg.pairs*2)
    view=rule.expand(cfg.pairs*2)
    # Burn-in is regenerated from the current policy on every update. No previous
    # learner supplies an aged graph and no state is cleared at the boundary.
    if past:
        with torch.no_grad():
            for t in range(past):
                if t:state=net.wait(view,state,data["wait"][:,t],max_ticks=cfg.max_wait)
                v,u=data["right"][:,t],data["u"][:,t]
                a=net.act(view,state,-v,v,u[:,0])
                exp=net.observe(view,a,-v,v,a.side*data["law"][:,t])
                state=net.commit(view,exp,u[:,1:],intervention="sham" if mode=="sham_age" else "active").state
        state=state.detach()
    logp=g.weights.new_zeros(len(state.hidden))
    for t in range(past,past+4):
        if t:state=net.wait(view,state,data["wait"][:,t],max_ticks=cfg.max_wait)
        v,u=data["right"][:,t],data["u"][:,t]
        a=net.act(view,state,-v,v,u[:,0])
        logp=logp+a.motor_log_prob
        final=(a.side*data["law"][:,t]).detach()
        if t<past+3:
            exp=net.observe(view,a,-v,v,final)
            w=net.commit(view,exp,u[:,1:])
            logp=logp+w.edit_log_prob
            state=w.state
    r=final.reshape(len(g.bias),cfg.pairs,2)
    adv=previous.leave_pair_out_advantage(r)
    loss=-(logp.reshape_as(r)*adv).mean((1,2)).sum()
    def totals(mask):
        return ((final>0)&mask).reshape_as(r).sum((1,2)),mask.reshape_as(r).sum((1,2))
    young,ny=totals(data["recent_final"]);older,no=totals(~data["recent_final"])
    stats=torch.stack(((r>0).float().mean((1,2)),young,ny,older,no),1)
    return loss,stats


def parent(source,births):
    gs,rs=zip(*(previous.load_checkpoint(source,"odor","uniform",b,8000) for b in births),strict=True)
    g=fw.Graph(nn.Parameter(torch.cat([v.weights for v in gs])),nn.Parameter(torch.cat([v.bias for v in gs])),
               torch.cat([v.present for v in gs]))
    r=net.Rule(births)
    r.load_state_dict({k:torch.cat([s.state_dict()[k] for s in rs]) for k in r.state_dict()})
    return g,r


def fit(out,source,births,arm,cfg,stop):
    torch.set_num_threads(1)
    g,rule=parent(source,births)
    params=(g.weights,g.bias,*rule.parameters())
    opt=torch.optim.Adam(params,lr=cfg.lr)
    folder=out/arm; folder.mkdir(parents=True,exist_ok=True)
    snaps=sorted(folder.glob("step-*.pt"))
    step,trace,contacts=0,[],0
    if snaps:
        d=torch.load(snaps[-1],weights_only=True,map_location="cpu")
        if (d["arm"],d["births"],d["config"])!=(arm,list(births),asdict(cfg)):raise ValueError("Wrong checkpoint")
        with torch.no_grad():g.weights.copy_(d["graph"]["weights"]);g.bias.copy_(d["graph"]["bias"])
        rule.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"])
        step,trace,contacts=d["step"],d["trace"].tolist(),d["contacts"]
    def save():
        d=dict(arm=arm,births=list(births),config=asdict(cfg),step=step,trace=torch.as_tensor(trace),contacts=contacts,
               graph=dict(weights=g.weights.detach(),bias=g.bias.detach(),present=g.present),
               rule=rule.state_dict(),optimizer=opt.state_dict())
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(d,f))
    if not snaps:save()
    initial=step;started=perf_counter()
    for step in range(initial+1,stop+1):
        p=past_contacts(cfg,step)
        data=batch(cfg,births,step,p)
        opt.zero_grad(set_to_none=True)
        objective,stats=rollout(g,rule,data,cfg,p,arm)
        objective.backward()
        valid=[x for x in params if x.grad is not None]
        with torch.no_grad():
            norm=sum(x.grad.flatten(1).double().square().sum(1) for x in valid).sqrt()
            if not torch.isfinite(norm).all():raise FloatingPointError(f"{arm}/{step}")
            sc=(2/norm.clamp_min(1e-12)).clamp_max(1)
            for x in valid:x.grad.mul_(sc.reshape((-1,)+(1,)*(x.ndim-1)).to(x.dtype))
        opt.step()
        contacts+=cfg.pairs*2*(p+4)
        trace.append(torch.cat((stats,norm[:,None].float(),torch.full_like(stats[:,:1],p)),1).tolist())
        if step%cfg.save_every==0 or step==stop:save()
        if step%250==0 or step in (1,stop):
            print(arm,step,np.array(trace[-100:])[:,:,0].mean(0).round(3),
                  round((perf_counter()-started)/(step-initial),3),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DATA/"adult-fit")
    p.add_argument("--source",type=Path,default=DATA/"retention-objective")
    p.add_argument("--births",nargs="+",type=int,default=list(previous.BIRTHS))
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--updates",type=int,default=8000)
    p.add_argument("--pairs",type=int,default=8)
    p.add_argument("--world-seed",type=int,default=871423)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--stop",type=int)
    a=p.parse_args()
    cfg=Config(updates=a.updates,pairs=a.pairs,world_seed=a.world_seed)
    stop=a.stop or a.updates
    if a.pairs<2 or not 1<=stop<=a.updates:raise ValueError("Invalid adult fit budget")
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(config=asdict(cfg),arms=a.arms,births=a.births,source=str(a.source),
              source_sha256={str(b):sha256(previous.checkpoint_path(a.source,"odor","uniform",b,8000)
                                           .read_bytes()).hexdigest() for b in a.births},
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              objective="Random real unknown-item history, then two unseen objects, one real review and an "
                        "unpredictable earned terminal quiz. Carry h, traces and graph. Detach the finite burn-in "
                        "and fit on real suffix resources only. The sham-age arm receives matched sensory contacts "
                        "and clocks during the prefix but writes no parameter there; both make all suffix writes. "
                        "When age > 0 neither arm differentiates through the preceding history or its discrete scores. "
                        "This trains usability of reached states and is not the exact full-lifetime gradient.")
    manifest=a.out/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError("Protocol mismatch")
    common._atomic_save(manifest,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in [pool.submit(fit,a.out,a.source,a.births,arm,cfg,stop) for arm in a.arms]:f.result()


if __name__=="__main__":main()
