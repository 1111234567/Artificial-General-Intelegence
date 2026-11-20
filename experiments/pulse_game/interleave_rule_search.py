"""Refine an inherited local write rule by paired physical outcomes or score gradients."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_age_train as age, interleave_credit_train as credit
from . import interleave_train as common, local_trace_graph as net, local_trace_train as previous
from .interleave_interference import DATA
from .self_param_writer import Graph


MODES=("directional","score_full","score_short")
DOMAIN=653197


def load_parent(path,births):
    gs,rs=zip(*(previous.load_checkpoint(path,"odor","uniform",b,8000) for b in births),strict=True)
    g=Graph(*(torch.cat([getattr(x,k) for x in gs]) for k in ("weights","bias","present")))
    names=list(rs[0].state_dict())
    shapes={k:rs[0].state_dict()[k].shape[1:] for k in names}
    theta=torch.cat([torch.cat([r.state_dict()[k] for r in rs]).flatten(1) for k in names],1)
    return g,theta,shapes


def unpack(theta,shapes):
    pieces=theta.split([s.numel() for s in shapes.values()],dim=-1)
    return {k:piece.reshape(*theta.shape[:-1],*shape)
            for (k,shape),piece in zip(shapes.items(),pieces,strict=True)}


def lives(base,cfg,births,step,blocks,replicate_draws):
    # [birth, independent geometry, perturbation/replica, law, contact, ...]
    shared={}
    for name in age.Tape.__annotations__:
        arr=getattr(base,name)
        rebatch=arr.reshape(len(births),cfg.pairs,2,*arr.shape[1:])
        shared[name]=rebatch.unsqueeze(2).expand(-1,-1,2,*([-1]*(rebatch.ndim-2))).clone()
    if not replicate_draws:
        other=np.stack([np.random.default_rng([DOMAIN,cfg.seed,b,step,815]).random(
            (cfg.pairs,blocks*4,5),dtype=np.float32) for b in births])
        shared["draws"][:,:,1]=torch.from_numpy(other)[:,:,None]
    return age.Tape(*(v.flatten(0,3) for v in shared.values()))


def experience(g,theta,shapes,tape,mode,cfg):
    roots,directions=theta.shape[:2]
    flat_theta=theta.flatten(0,2).repeat_interleave(2,dim=0)
    coeff=net._Coefficients(**unpack(flat_theta,shapes))
    state=net.start(g,directions*4)
    r,ms,es=[],[],[]
    for t in range(tape.left.shape[1]):
        if t and t%4==0 and mode=="score_short":
            state=credit.boundary(state,"all")
        if t:state=net.wait(coeff,state,tape.wait[:,t],max_ticks=cfg.max_wait)
        l,right,u=tape.left[:,t],tape.right[:,t],tape.draws[:,t]
        a=net.act(coeff,state,l,right,u[:,0])
        earned=(a.side*tape.factor[:,t]+tape.fixed[:,t]).detach()
        exp=net.observe(coeff,a,l,right,earned)
        w=net.commit(coeff,exp,u[:,1:])
        state=w.state
        r.append(earned)
        ms.append(a.motor_log_prob)
        es.append(w.edit_log_prob)
    def group(vals):return torch.stack(vals,-1).reshape(roots,directions,2,2,-1)
    return group(r),group(ms),group(es)


def direction_gradient(pair_return,eps,sigma):
    delta=pair_return[:,:,0]-pair_return[:,:,1]
    return (delta[:,:,None]*eps).mean(1)/(2*sigma)


def fit(out,source,births,mode,cfg,sigma,stop):
    torch.set_num_threads(1)
    g,initial,shapes=load_parent(source,births)
    theta=nn.Parameter(initial)
    opt=torch.optim.Adam([theta],lr=cfg.lr)
    folder=out/mode
    folder.mkdir(parents=True,exist_ok=True)
    saved=sorted(folder.glob("step-*.pt"))
    step,trace,contacts=0,[],0
    if saved:
        d=torch.load(saved[-1],weights_only=True,map_location="cpu")
        if (d["births"],d["config"],d["arm"])!=(list(births),asdict(cfg),mode):raise ValueError("Wrong output run")
        with torch.no_grad():theta.copy_(d["theta"])
        opt.load_state_dict(d["optimizer"])
        step,trace,contacts=d["step"],d["trace"].tolist(),d["contacts"]
    def save():
        payload=dict(births=list(births),config=asdict(cfg),arm=mode,step=step,theta=theta.detach(),
                     optimizer=opt.state_dict(),trace=torch.tensor(trace),contacts=contacts,
                     graph={k:getattr(g,k) for k in ("weights","bias","present")},rule=unpack(theta.detach(),shapes))
        common._atomic_save(age.ckpath(out,mode,step),lambda f:torch.save(payload,f))
    if not saved:save()
    first=step;started=perf_counter()
    for step in range(first+1,stop+1):
        blocks=age.life_blocks(cfg,step)
        base=age.make_tape(cfg,births,step,blocks,domain=DOMAIN)
        tape=lives(base,cfg,births,step,blocks,replicate_draws=mode=="directional")
        eps=torch.as_tensor(np.stack([age.signs(np.random.default_rng([DOMAIN,cfg.seed,b,step,445]),
                                                (cfg.pairs,theta.shape[-1])) for b in births]))
        opt.zero_grad(set_to_none=True)
        direction=mode=="directional"
        with torch.set_grad_enabled(not direction):
            centers=theta[:,None,None,:].expand(-1,cfg.pairs,2,-1)
            variants=centers+sigma*eps[:,:,None,:]*theta.new_tensor([1,-1])[None,None,:,None] if direction else centers
            r,m,e=experience(g,variants,shapes,tape,mode,cfg)
            query=torch.arange(r.shape[-1])%4>=2
            pair_return=r[...,query].mean((3,4))
            if direction:
                theta.grad=-direction_gradient(pair_return,eps,sigma)
            else:
                reward=r*query*(2/r.shape[-1])
                marginal=reward.mean((2,3),keepdim=True)
                baseline=(marginal.sum(1,keepdim=True)-marginal)/(cfg.pairs-1)
                adv=(reward-baseline).flip(-1).cumsum(-1).flip(-1)
                loss=-(m*adv).sum(-1).mean((1,2,3))-(e[...,:-1]*adv[...,1:]).sum(-1).mean((1,2,3))
                loss.sum().backward()
        with torch.no_grad():
            norm=theta.grad.double().norm(dim=1)
            if not torch.isfinite(norm).all():raise FloatingPointError(f"{mode}: {step}")
            theta.grad.mul_((cfg.grad_clip/norm.clamp_min(1e-12)).clamp_max(1).float()[:,None])
            gap=(pair_return[:,:,0]-pair_return[:,:,1]).abs()
            score=(r[...,query]>0).float()
            row=torch.stack((score.mean((1,2,3,4)),score[...,:2].mean((1,2,3,4)),
                             score[...,-2:].mean((1,2,3,4)),(gap!=0).float().mean(1),
                             gap.mean(1),norm.float()),1)
        opt.step()
        trace.append(row.tolist())
        contacts+=cfg.pairs*4*4*blocks
        if step%cfg.checkpoint_every==0 or step==stop:save()
        if step%250==0 or step in (1,stop):
            print(mode,step,np.mean(trace[-100:],axis=(0,1)).round(3),round((perf_counter()-started)/(step-first),3),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=DATA/"rule-search")
    p.add_argument("--source",type=Path,default=DATA/"retention-objective")
    p.add_argument("--births",type=int,nargs="+",default=list(previous.BIRTHS))
    p.add_argument("--arms",nargs="+",choices=MODES,default=MODES)
    p.add_argument("--updates",type=int,default=8000)
    p.add_argument("--stop",type=int)
    p.add_argument("--directions",type=int,default=4)
    p.add_argument("--lr",type=float,default=.003)
    p.add_argument("--sigma",type=float,default=.065)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    stop=a.stop or a.updates
    if not 1<=stop<=a.updates or a.directions<2 or a.sigma<=0:raise ValueError("Invalid comparison")
    cfg=age.Config(updates=a.updates,pairs=a.directions,lr=a.lr,seed=982571)
    a.out.mkdir(parents=True,exist_ok=True)
    parents={b:sha256(previous.checkpoint_path(a.source,"odor","uniform",b,8000).read_bytes()).hexdigest()
             for b in a.births}
    plan=dict(config=asdict(cfg),arms=a.arms,births=a.births,sigma=a.sigma,parents={str(b):h for b,h in parents.items()},
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              design="Only shared inherited write-rule coefficients undergo outer optimization. All body weights, "
                     "biases and graph topology can be changed during each independent lifetime. Four independent "
                     "world groups per step, two replicas per world each crossed with opposite law. Directional "
                     "group uses opposite Rademacher perturbations of the entire rule and identical event/motor uniforms. "
                     "Score controls use the center actor with independent replica motor/edit uniforms. Same world count, "
                     "physical contacts, optimizer count and nominal Adam learning rate; exact full derivatives vs "
                     "truncated continuous derivatives in score_short. Only the actual two quiz resources in each block score.")
    f=a.out/"run.json"
    if f.exists() and json.loads(f.read_text())!=plan:raise ValueError("Protocol mismatch")
    common._atomic_save(f,lambda path:path.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in [pool.submit(fit,a.out,a.source,a.births,arm,cfg,a.sigma,stop) for arm in a.arms]:
            future.result()


if __name__=="__main__":main()
