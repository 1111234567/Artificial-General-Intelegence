"""Learn two contextual values for one object using actual resource and a visible scene cue."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import interleave_age_train as age, interleave_train as common, local_trace_graph as net
from . import local_trace_train as previous, self_param_writer as fw
from .interleave_interference import DATA, EVIDENCE


ARMS=("full_direct","full_morph","preview_direct","preview_morph","blank_morph")
DOMAIN=188953


@dataclass
class Config:
    updates:int=8000
    worlds:int=12
    seed:int=584608
    max_wait:int=8
    lr:float=.002
    clip:float=2.
    save_every:int=500


def make(cfg,birth,step,*,domain=DOMAIN,angular="direct",forced_angle=None,laws=None):
    rng=np.random.default_rng([domain,cfg.seed,birth,step])
    n=cfg.worlds
    a=age.directions(rng,n)/1.35
    b=age.directions(rng,n)/1.35
    b-=(a*b).sum(1,keepdims=True)*a
    b/=np.linalg.norm(b,axis=1,keepdims=True)
    phases=rng.random((n,2))
    if forced_angle is not None:
        deg=np.full(n,forced_angle)
    elif angular=="direct" or step>3000:
        deg=np.zeros(n)
    elif step<=1000:
        deg=np.where(phases[:,0]<.15,0.,45+45*phases[:,1])
    else:
        deg=np.where(phases[:,0]<.40,0.,90*phases[:,1])
    theta=np.deg2rad(deg)[:,None]
    b=(np.cos(theta)*a+np.sin(theta)*b).astype(np.float32)
    v=np.stack((1.35*a,1.35*b),1).astype(np.float32)
    ctx_order=rng.permuted(np.tile(np.arange(3),(n,1)),axis=1)
    cues=common.CUE_BANK[ctx_order[:,:2]]
    if laws is None:laws=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=np.float32)
    m=len(laws)
    def share(x):return torch.from_numpy(np.repeat(x,m,axis=0))
    order=rng.permuted(np.tile(np.arange(2),(n,1)),axis=1)
    query=rng.integers(2,size=(n,1))
    return dict(odor=share(v),cue=share(cues),unused_cue=share(common.CUE_BANK[ctx_order[:,2]]),
                law=torch.from_numpy(np.tile(laws,(n,1))),order=share(np.concatenate((order,query),1)),
                sign=share(age.signs(rng,(n,3))),u=share(rng.random((n,3,5),dtype=np.float32)),
                gap=share(rng.integers(cfg.max_wait+1,size=(n,3))),degrees=share(deg))


def combine(batch):
    return {k:torch.cat([v[k] for v in batch]) for k in batch[0]}


def choose(view,state,odor,cue,draw,timing,*,early_cue=None):
    left,right=cue-odor,cue+odor
    if timing=="full":
        return net.act(view,state,left,right,draw)
    if timing not in ("preview","blank"):
        raise ValueError("Unknown sensory timing")
    first=(cue if early_cue is None else early_cue) if timing=="preview" else torch.zeros_like(cue)
    state=net._advance(view,state,fw.sensory(state.graph,first,first),2)
    state=net._advance(view,state,fw.sensory(state.graph,left,right),3)
    logit=2.5*(state.hidden[:,11]-state.hidden[:,10])
    side=(draw<logit.detach().sigmoid()).float()*2-1
    return net.Action(state,side,logit,F.logsigmoid(side*logit))


def life(g,rule,tape,timing,cfg,*,intervention="active",read_both=False,wrong_preview=False):
    n=len(tape["law"])
    lanes=n//len(g.bias)
    view=rule.expand(lanes)
    state=net.start(g,lanes)
    row=torch.arange(n)
    score=0.
    for t in range(2):
        if t:state=net.wait(view,state,tape["gap"][:,t],max_ticks=cfg.max_wait)
        item=tape["order"][:,t]
        cue=tape["cue"][row,item]
        orient=tape["sign"][:,t]
        odor=tape["odor"][row,item]*orient[:,None]
        u=tape["u"][:,t]
        act=choose(view,state,odor,cue,u[:,0],timing,
                   early_cue=tape["unused_cue"] if wrong_preview else None)
        earned=act.side*orient*tape["law"][row,item]
        obs=net.observe(view,act,cue-odor,cue+odor,earned)
        w=net.commit(view,obs,u[:,1:],intervention=intervention)
        score=score+act.motor_log_prob+w.edit_log_prob
        state=w.state
    state=net.wait(view,state,tape["gap"][:,2],max_ticks=cfg.max_wait)
    selections=(torch.zeros(n,dtype=torch.long), torch.ones(n,dtype=torch.long)) if read_both else (tape["order"][:,2],)
    out=[]
    for item in selections:
        cue=tape["cue"][row,item]
        orient=tape["sign"][:,2]
        odor=tape["odor"][row,item]*orient[:,None]
        a=choose(view,state,odor,cue,tape["u"][:,2,0],timing,
                 early_cue=tape["unused_cue"] if wrong_preview else None)
        reward=(a.side*orient*tape["law"][row,item]).detach()
        out.append((a,reward))
    return score,out


def warm(path,step,births):
    gs,rs=zip(*(previous.load_checkpoint(path,"odor","uniform",b,step) for b in births),strict=True)
    graph=fw.Graph(nn.Parameter(torch.cat([g.weights for g in gs])),nn.Parameter(torch.cat([g.bias for g in gs])),
                   torch.cat([g.present for g in gs]))
    rule=net.Rule(births)
    rule.load_state_dict({k:torch.cat([r.state_dict()[k] for r in rs]) for k in rs[0].state_dict()})
    return graph,rule


def fit(out,source,source_step,cfg,births,arm,stop):
    torch.set_num_threads(1)
    timing,angular=arm.split("_")
    graph,rule=warm(source,source_step,births)
    params=(graph.weights,graph.bias,*rule.parameters())
    optimizer=torch.optim.Adam(params,lr=cfg.lr)
    folder=out/arm
    folder.mkdir(parents=True,exist_ok=True)
    snaps=sorted(folder.glob("step-*.pt"))
    step,trace=0,[]
    if snaps:
        saved=torch.load(snaps[-1],weights_only=True,map_location="cpu")
        if saved["arm"]!=arm or saved["births"]!=list(births) or saved["config"]!=asdict(cfg):
            raise ValueError("Incompatible checkpoint")
        with torch.no_grad():
            graph.weights.copy_(saved["graph"]["weights"])
            graph.bias.copy_(saved["graph"]["bias"])
        rule.load_state_dict(saved["rule"])
        optimizer.load_state_dict(saved["optimizer"])
        step,trace=saved["step"],saved["trace"].tolist()
    def save():
        obj=dict(config=asdict(cfg),births=list(births),arm=arm,step=step,
                 graph={k:getattr(graph,k).detach() for k in ("weights","bias","present")},
                 rule=rule.state_dict(),optimizer=optimizer.state_dict(),trace=torch.as_tensor(trace))
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(obj,f))
    if not snaps:save()
    begin=step
    started=perf_counter()
    for step in range(begin+1,stop+1):
        tape=combine([make(cfg,b,step,angular=angular) for b in births])
        optimizer.zero_grad(set_to_none=True)
        score,[(last,reward)]=life(graph,rule,tape,timing,cfg)
        r=reward.reshape(len(births),cfg.worlds,4)
        block=r.mean(2,keepdim=True)
        baseline=(block.sum(1,keepdim=True)-block)/(cfg.worlds-1)
        loss=-((score+last.motor_log_prob).reshape_as(r)*(r-baseline)).mean((1,2)).sum()
        loss.backward()
        with torch.no_grad():
            norm=sum(p.grad.flatten(1).double().square().sum(1) for p in params).sqrt()
            if not torch.isfinite(norm).all():raise FloatingPointError(f"{arm} gradient at {step}")
            scale=(cfg.clip/norm.clamp_min(1e-10)).clamp_max(1)
            for p in params:p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)).to(p.dtype))
            good=(r>0).float()
            stat=torch.stack((good.mean((1,2)),good[:,:,[1,2]].mean((1,2)),good[:,:,[0,3]].mean((1,2)),norm.float()),1)
        optimizer.step()
        trace.append(stat.tolist())
        if step%cfg.save_every==0 or step==stop:save()
        if step%500==0 or step in (1,stop):
            print(arm,step,np.mean(trace[-100:],axis=(0,1)).round(3),round((perf_counter()-started)/(step-begin),3),flush=True)


@torch.no_grad()
def evaluate(out,cfg,births,arms,step,seed,worlds,filename):
    torch.set_num_threads(1)
    cfg=Config(**{**asdict(cfg),"worlds":worlds,"seed":seed})
    results={"seed":seed,"worlds_per_birth":worlds,"step":step,"births":births,"arms":{},"sha256":{}}
    folder=out/"validation"/filename
    folder.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        timing=arm.split("_")[0]
        by_birth={}
        for b in births:
            g,r=age.load_checkpoint(out,arm,b,step)
            metrics,raw={},{}
            for deg in (0,30,90):
                tape=make(cfg,b,1,domain=DOMAIN+5,forced_angle=deg)
                laws=tape["law"].numpy()
                for control in ("active","sham","reverse","swap_final_cue","wrong_preview"):
                    if control=="wrong_preview" and timing!="preview":continue
                    sample=tape
                    if control=="reverse":
                        sample={**tape,"law":-tape["law"]}
                    if control=="swap_final_cue":
                        # Swap both learned ambient codes at readout only by replaying from the same prefix.
                        # For identical objects this is captured exactly by swapping the two readout actions.
                        if deg!=0:continue
                    _,queries=life(g,r,sample,timing,cfg,read_both=True,
                                   intervention="sham" if control=="sham" else "active",
                                   wrong_preview=control=="wrong_preview")
                    chosen=np.stack([(a.side*tape["sign"][:,2]).numpy() for a,_ in queries],1)
                    if control=="swap_final_cue":chosen=chosen[:,::-1]
                    correct=chosen==laws
                    conflicts=laws[:,0]!=laws[:,1]
                    key=f"{deg}/{control}"
                    raw[key]=chosen.astype(np.int8)
                    metrics[key]={"all":float(correct.mean()),"joint":float(correct.all(1).mean()),
                                  "conflicting":float(correct[conflicts].mean()),
                                  "conflict_both":float(correct[conflicts].all(1).mean()),
                                  "agree_both":float(correct[~conflicts].all(1).mean()),
                                  "different_on_conflict":float((chosen[conflicts,0]!=chosen[conflicts,1]).mean()),
                                  "by_assignment":correct.reshape(-1,4,2).mean(0).tolist()}
            f=folder/f"{arm}-{b}.npz"
            np.savez_compressed(f,**raw)
            results["sha256"][f.name]=sha256(f.read_bytes()).hexdigest()
            by_birth[b]=metrics
        results["arms"][arm]=by_birth
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"context-timing-{filename}.json").write_text(json.dumps(results,indent=1)+"\n",encoding="utf8")


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("fit","eval"),required=True)
    p.add_argument("--out",type=Path,default=DATA/"context-timing")
    p.add_argument("--source",type=Path,default=DATA/"retention-objective")
    p.add_argument("--source-step",type=int,default=8000)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--births",nargs="+",type=int,default=list(previous.BIRTHS))
    p.add_argument("--updates",type=int,default=8000)
    p.add_argument("--worlds",type=int,default=12)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--stop",type=int)
    p.add_argument("--eval-worlds",type=int,default=192)
    p.add_argument("--eval-seed",type=int,default=1927893)
    p.add_argument("--name",default="validation")
    a=p.parse_args()
    cfg=Config(updates=a.updates,worlds=a.worlds)
    stop=a.stop or a.updates
    if a.mode=="eval":
        evaluate(a.out,cfg,a.births,a.arms,stop,a.eval_seed,a.eval_worlds,a.name)
        return
    if not 1<=stop<=a.updates or a.worlds<2:raise ValueError("Invalid context fit size")
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(config=asdict(cfg),births=a.births,arms=a.arms,source=str(a.source),source_step=a.source_step,
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              note="Same randomized ambient vector already present in left+right. Full uses full menu for five "
                   "ticks. Preview presents just scene for first two ticks then full menu for three. Blank has "
                   "same three menu ticks with zero initial input. Identical remaining contact and receipt clocks, "
                   "four private A/B assignments crossed. Direct always shares odor. Morph initially separates "
                   "odor directions and reaches identical test distribution after 3000. One physically earned "
                   "terminal resource trains all sampled decisions, leaving other independent worlds as baseline.")
    path=a.out/"run.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Protocol mismatch")
    common._atomic_save(path,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in [pool.submit(fit,a.out,a.source,a.source_step,cfg,a.births,arm,stop) for arm in a.arms]:
            f.result()


if __name__=="__main__":main()
