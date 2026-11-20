"""Preserve entire lifetimes while comparing pathwise credit across four-contact boundaries."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_age_train as age, interleave_train as common
from . import local_trace_graph as net, local_trace_train as old
from .self_param_writer import Graph


MODES=("full","hidden","graph","all")


def boundary(state,mode):
    if mode not in MODES:raise ValueError(f"Unknown boundary {mode}")
    if mode in ("hidden","all"):
        state=replace(state,hidden=state.hidden.detach(), sender_trace=state.sender_trace.detach(),
                      receiver_trace=state.receiver_trace.detach())
    if mode in ("graph","all"):
        state=replace(state,graph=state.graph.detach())
    return state


def rollout(g,rule,tape,mode,config):
    state=net.start(g,2*config.pairs)
    view=rule.expand(2*config.pairs)
    rewards,motor,edit=[],[],[]
    for t in range(tape.left.shape[1]):
        if t and t%4==0:
            state=boundary(state,mode)
        if t:state=net.wait(view,state,tape.wait[:,t],max_ticks=config.max_wait)
        u=tape.draws[:,t]
        left,right=tape.left[:,t],tape.right[:,t]
        a=net.act(view,state,left,right,u[:,0])
        earned=(a.side*tape.factor[:,t]+tape.fixed[:,t]).detach()
        e=net.observe(view,a,left,right,earned)
        w=net.commit(view,e,u[:,1:])
        state=w.state
        rewards.append(earned);motor.append(a.motor_log_prob);edit.append(w.edit_log_prob)
    def pack(rows):return torch.stack(rows,-1).reshape(len(g.bias),config.pairs,2,-1)
    return pack(rewards),pack(motor),pack(edit),state


def loss(g,rule,tape,mode,cfg):
    r,m,e,state=rollout(g,rule,tape,mode,cfg)
    nroot,p,tw,length=r.shape
    weights=(torch.arange(length)%4>=2).float()*(2/length)
    adv=common.causal_advantage(r,weights.expand_as(r))
    lm=-(m*adv).sum(-1).mean((1,2))
    le=-(e[...,:-1]*adv[...,1:]).sum(-1).mean((1,2))
    with torch.no_grad():
        good=(r>0).reshape(nroot,p,tw,-1,4)[...,2:].float()
        scores=torch.stack((good.mean((1,2,3,4)),good[:,:,:,0].mean((1,2,3)),
                            good[:,:,:,-1].mean((1,2,3)),good.prod(-1).mean((1,2,3))),1)
    return (lm+le).sum(),scores


def fit(out,cfg,births,mode,stop,warm_root):
    torch.set_num_threads(1)
    gs,rs=zip(*(old.load_checkpoint(warm_root,"odor","active",s,8000) for s in births),strict=True)
    w,b=(nn.Parameter(torch.cat([getattr(x,k) for x in gs])) for k in ("weights","bias"))
    g=Graph(w,b,torch.cat([x.present for x in gs]))
    rule=net.Rule(births)
    rule.load_state_dict({k:torch.cat([r.state_dict()[k] for r in rs]) for k in rs[0].state_dict()})
    params=(w,b,*rule.parameters())
    opt=torch.optim.Adam(params,lr=cfg.lr)
    folder=out/mode
    folder.mkdir(exist_ok=True,parents=True)
    snaps=sorted(folder.glob("step-*.pt"))
    step,contacts,trace=0,0,[]
    if snaps:
        d=torch.load(snaps[-1],weights_only=True,map_location="cpu")
        if (d["births"],d["arm"],d["config"])!=(list(births),mode,asdict(cfg)):
            raise ValueError("Incompatible credit checkpoint")
        with torch.no_grad():w.copy_(d["graph"]["weights"]);b.copy_(d["graph"]["bias"])
        rule.load_state_dict(d["rule"]); opt.load_state_dict(d["optimizer"])
        step,contacts,trace=d["step"],d["contacts"],d["trace"].tolist()
    def save():
        payload=dict(config=asdict(cfg),arm=mode,births=list(births),step=step,contacts=contacts,
                     graph=dict(weights=w.detach(),bias=b.detach(),present=g.present),rule=rule.state_dict(),
                     optimizer=opt.state_dict(),trace=torch.tensor(trace))
        common._atomic_save(age.ckpath(out,mode,step),lambda p:torch.save(payload,p))
    if not snaps:save()
    first=step;started=perf_counter()
    for step in range(first+1,stop+1):
        blocks=age.life_blocks(cfg,step)
        tape=age.make_tape(cfg,births,step,blocks)
        opt.zero_grad(set_to_none=True)
        objective,stats=loss(g,rule,tape,mode,cfg)
        objective.backward()
        with torch.no_grad():
            norm=sum(p.grad.flatten(1).double().square().sum(1) for p in params).sqrt()
            if not torch.isfinite(norm).all():raise FloatingPointError(f"Nonfinite {mode}/{step}")
            scale=(cfg.grad_clip/norm.clamp_min(1e-12)).clamp_max(1)
            for p in params:p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)).to(p.dtype))
        opt.step()
        contacts+=blocks*4*2*cfg.pairs
        trace.append(torch.cat((stats,norm[:,None].float(),torch.full_like(stats[:,:1],blocks)),1).tolist())
        if step%cfg.checkpoint_every==0 or step==stop:save()
        if step%250==0 or step in (1,stop):
            recent=np.array(trace[-100:])
            print(mode,step,recent[:,:,0].mean(0).round(3),np.median(recent[:,:,-2]).round(2),
                  round((perf_counter()-started)/(step-first),3),"s/step",flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=age.OUTPUT.parent/"credit-fit")
    p.add_argument("--arms",choices=MODES,nargs="+",default=("hidden","graph","all"))
    p.add_argument("--births",nargs="+",type=int,default=list(old.BIRTHS))
    p.add_argument("--warm-root",type=Path,default=old.DEFAULT_OUTPUT/"local")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--stop",type=int)
    cfg=age.Config()
    for f in fields(cfg):
        v=getattr(cfg,f.name)
        p.add_argument("--"+f.name.replace("_","-"),type=type(v),default=v)
    a=p.parse_args()
    cfg=age.Config(**{f.name:getattr(a,f.name) for f in fields(cfg)})
    stop=a.stop or cfg.updates
    if not 1<=stop<=cfg.updates or cfg.pairs<2 or cfg.warm_source!="odor":raise ValueError("Invalid credit fit")
    out=a.out
    out.mkdir(parents=True,exist_ok=True)
    plan=dict(config=asdict(cfg),births=a.births,arms=list(a.arms),warm_root=str(a.warm_root),
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              design="Physical state and every reward and sample are unchanged at a boundary. Every discrete action "
                     "still receives its complete earned future reward score. Only pathwise derivatives crossing "
                     "four-contact boundaries are detached for the named state components. Truncation biases the full "
                     "gradient of the continuous components. The full comparator uses the same tapes and loss in lifetime-fit.")
    path=out/"run.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Manifest changed")
    common._atomic_save(path,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in [pool.submit(fit,out,cfg,a.births,arm,stop,a.warm_root) for arm in a.arms]:
            future.result()


if __name__=="__main__":
    main()
