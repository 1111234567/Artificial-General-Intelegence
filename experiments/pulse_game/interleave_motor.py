"""Test whether one native product of two node activations improves contextual motor readout."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import interleave_age_train as age, interleave_context_time as ctx, interleave_train as common
from . import local_trace_graph as net, local_trace_train as old
from .interleave_interference import DATA,EVIDENCE


ARMS=("native","linear","product")
DOMAIN=916053


def action(view,state,left,right,u,gain,mode):
    # All versions observe the same inputs for the same five physical ticks. The
    # extra motor currents draw on existing, writable nodes, not a new hidden layer.
    plain=net.act(view,state,left,right,u)
    h=plain.state.hidden
    if mode=="product":
        extra=h[:,14]*h[:,15]-h[:,12]*h[:,13]
    elif mode=="linear":
        extra=(h[:,14]+h[:,15]-h[:,12]-h[:,13])*.5
    elif mode=="native":
        extra=torch.zeros_like(u)
        gain=gain*0
    else:raise ValueError(mode)
    blend=gain.tanh()
    logit=(1-blend.abs())*plain.motor_logit+2.5*blend*extra
    side=(u<logit.detach().sigmoid()).float()*2-1
    return net.Action(plain.state,side,logit,F.logsigmoid(side*logit))


def run_life(g,rule,tape,gain,mode,*,both=False,intervention="active"):
    n=len(tape["law"]);lanes=n//len(g.bias)
    state=net.start(g,lanes)
    view=rule.expand(lanes);row=torch.arange(n)
    gate=gain.repeat_interleave(lanes)
    total=g.weights.new_zeros(n)
    for t in range(2):
        if t:state=net.wait(view,state,tape["gap"][:,t],max_ticks=8)
        item=tape["order"][:,t];cue=tape["cue"][row,item];s=tape["sign"][:,t]
        v=tape["odor"][row,item]*s[:,None];u=tape["u"][:,t]
        a=action(view,state,cue-v,cue+v,u[:,0],gate,mode)
        reward=a.side*tape["law"][row,item]*s
        exp=net.observe(view,a,cue-v,cue+v,reward)
        w=net.commit(view,exp,u[:,1:],intervention=intervention)
        state=w.state;total=total+a.motor_log_prob+w.edit_log_prob
    state=net.wait(view,state,tape["gap"][:,2],max_ticks=8)
    ids=(torch.zeros(n,dtype=torch.long),torch.ones(n,dtype=torch.long)) if both else (tape["order"][:,2],)
    actions=[]
    for item in ids:
        cue=tape["cue"][row,item];s=tape["sign"][:,2];v=tape["odor"][row,item]*s[:,None]
        a=action(view,state,cue-v,cue+v,tape["u"][:,2,0],gate,mode)
        actions.append((a,(a.side*tape["law"][row,item]*s).detach()))
    return total,actions


def parents(source,source_arm,births):
    gs,rs=zip(*(age.load_checkpoint(source,source_arm,b,8000) for b in births),strict=True)
    g=net.Graph(nn.Parameter(torch.cat([v.weights for v in gs])),nn.Parameter(torch.cat([v.bias for v in gs])),
                torch.cat([v.present for v in gs]))
    rule=net.Rule(births)
    rule.load_state_dict({k:torch.cat([v.state_dict()[k] for v in rs]) for k in rs[0].state_dict()})
    return g,rule


def fit(out,source,source_arm,arm,births,cfg,stop):
    torch.set_num_threads(1)
    g,r=parents(source,source_arm,births)
    gain=nn.Parameter(torch.zeros(len(births),1))
    params=(g.weights,g.bias,*r.parameters(),gain)
    opt=torch.optim.Adam(params,lr=cfg.lr)
    folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
    files=sorted(folder.glob("step-*.pt"))
    step,log=0,[]
    if files:
        d=torch.load(files[-1],weights_only=True,map_location="cpu")
        if (d["arm"],d["births"],d["config"])!=(arm,list(births),asdict(cfg)):raise ValueError("Checkpoint mismatch")
        with torch.no_grad():
            g.weights.copy_(d["graph"]["weights"]);g.bias.copy_(d["graph"]["bias"]);gain.copy_(d["gain"])
        r.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"])
        step,log=d["step"],d["trace"].tolist()
    def save():
        payload=dict(step=step,births=list(births),arm=arm,config=asdict(cfg),gain=gain.detach(),
                     graph=dict(weights=g.weights.detach(),bias=g.bias.detach(),present=g.present),
                     rule=r.state_dict(),optimizer=opt.state_dict(),trace=torch.as_tensor(log))
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(payload,f))
    if not files:save()
    first=step;started=perf_counter()
    for step in range(first+1,stop+1):
        tape=ctx.combine([ctx.make(cfg,b,step,domain=DOMAIN,angular="direct") for b in births])
        opt.zero_grad(set_to_none=True)
        score,[(last,rew)]=run_life(g,r,tape,gain[:,0],arm)
        reward=rew.reshape(len(births),cfg.worlds,4)
        groups=reward.mean(2,keepdim=True)
        advantage=reward-(groups.sum(1,keepdim=True)-groups)/(cfg.worlds-1)
        loss=-((score+last.motor_log_prob).reshape_as(reward)*advantage).mean((1,2)).sum()
        loss.backward()
        norm=old.clip_birth_gradients(params,cfg.clip)
        opt.step()
        with torch.no_grad():
            good=reward>0
            metrics=torch.stack((good.float().mean((1,2)),good[:,:,[1,2]].float().mean((1,2)),
                                 gain[:,0].tanh(),norm),1)
        log.append(metrics.tolist())
        if step%cfg.save_every==0 or step==stop:save()
        if step%500==0 or step in (1,stop):
            print(arm,step,np.mean(log[-100:],axis=(0,1)).round(3),
                  round((perf_counter()-started)/(step-first),3),flush=True)


@torch.no_grad()
def evaluate(out,arms,births,step,worlds,seed,label):
    torch.set_num_threads(1)
    dest=out/"validation"/label;dest.mkdir(parents=True,exist_ok=True)
    result=dict(births=births,step=step,worlds=worlds,seed=seed,roots={},sha256={})
    cfg=ctx.Config(worlds=worlds,seed=seed)
    for arm in arms:
        data=torch.load(age.ckpath(out,arm,step),weights_only=True,map_location="cpu")
        entries={}
        for b in births:
            i=data["births"].index(b);gain=data["gain"][i:i+1,0]
            g,r=age.load_checkpoint(out,arm,b,step)
            raw={};cells={}
            for degrees in (0,45,90):
                tape=ctx.make(cfg,b,1,domain=DOMAIN+71,forced_angle=degrees)
                laws=tape["law"]
                conflict=laws[:,0]!=laws[:,1]
                old=tape["order"][:,0];row=torch.arange(len(old))
                for branch in ("native_live","extra_removed","sham_writes","reversed"):
                    seen={**tape,"law":-laws} if branch=="reversed" else tape
                    use="native" if branch=="extra_removed" else arm
                    _,choices=run_life(g,r,seen,gain,use,both=True,
                                       intervention="sham" if branch=="sham_writes" else "active")
                    responses=torch.stack([a.side*tape["sign"][:,2] for a,_ in choices],1)
                    good=responses==laws
                    key=f"{degrees}/{branch}"
                    raw[key]=responses.numpy().astype(np.int8)
                    cells[key]=dict(correct=float(good.float().mean()),
                                    conflict=float(good[conflict].float().mean()),
                                    conflict_both=float(good[conflict].all(1).float().mean()),
                                    agree_both=float(good[~conflict].all(1).float().mean()),
                                    conflict_earlier=float(good[row,old][conflict].float().mean()),
                                    conflict_recent=float(good[row,1-old][conflict].float().mean()),
                                    all_four_histories=float(good.reshape(worlds,4,2).all(2).all(1).float().mean()))
            p=dest/f"{arm}-{b}.npz";np.savez_compressed(p,**raw)
            entries[b]=dict(gain=float(gain.tanh().item()),scores=cells)
            result["sha256"][p.name]=sha256(p.read_bytes()).hexdigest()
        result["roots"][arm]=entries
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/f"motor-{label}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("fit","eval"),required=True)
    p.add_argument("--source",type=Path,default=DATA/"context-timing")
    p.add_argument("--source-arm",default="full_direct")
    p.add_argument("--out",type=Path,default=DATA/"motor-fit")
    p.add_argument("--births",nargs="+",type=int,default=list(old.BIRTHS))
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--updates",type=int,default=8000)
    p.add_argument("--stop",type=int)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--worlds",type=int,default=12)
    p.add_argument("--seed",type=int,default=281079)
    p.add_argument("--eval-worlds",type=int,default=192)
    p.add_argument("--eval-seed",type=int,default=7712417)
    p.add_argument("--label",default="final")
    a=p.parse_args();step=a.stop or a.updates
    if a.mode=="eval":
        evaluate(a.out,a.arms,a.births,step,a.eval_worlds,a.eval_seed,a.label)
        return
    cfg=ctx.Config(updates=a.updates,worlds=a.worlds,seed=a.seed)
    if not 1<=step<=a.updates or a.worlds<2:raise ValueError("Invalid motor study")
    a.out.mkdir(parents=True,exist_ok=True)
    parent=age.ckpath(a.source,a.source_arm,8000)
    plan=dict(config=asdict(cfg),births=a.births,arms=a.arms,parent_sha256=sha256(parent.read_bytes()).hexdigest(),
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              note="One inherited scalar starts at zero. Interpolate the old two motor outputs with an additional "
                   "linear or multiplicative current from four existing mutable neurons; absolute blending weights sum "
                   "to one and logit range remains [-5,5]. No extra state, node, new edge or private context label. "
                   "All graph and shared-rule numbers continue outer training along with this scalar. Independent "
                   "two-context meanings fully crossed; only actual final earned resources weight the action scores.")
    manifest=a.out/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError("Frozen plan mismatch")
    common._atomic_save(manifest,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in [pool.submit(fit,a.out,a.source,a.source_arm,arm,a.births,cfg,step) for arm in a.arms]:f.result()


if __name__=="__main__":main()
