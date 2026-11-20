"""Can a receiver use its existing incoming weights to regulate another write?"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import interleave_age_train as age, interleave_train as common, local_trace_graph as net
from .self_param_writer import Graph


ARMS=("own","permuted","zero")


@dataclass
class View:
    base: object
    gain: torch.Tensor
    arm: str


class ReadRule(nn.Module):
    def __init__(self,births,arm):
        super().__init__()
        self.base=net.Rule(births)
        self.gain=nn.Parameter(torch.zeros((len(births),net.TRACE_RANK)))
        self.arm=arm

    def expand(self,n):
        return View(self.base.expand(n),self.gain.tanh().repeat_interleave(n,dim=0),self.arm)


def commit(view,experience,draws,permutation,*,intervention="active"):
    """Read the input current the present synapses would transmit for each sender trace."""
    s=experience.state
    graph=s.graph
    incoming=torch.bmm(graph.weights*graph.present,s.sender_trace)
    count=graph.present.sum(2).clamp_min(1).to(incoming.dtype).sqrt()
    received=(incoming/count[:,:,None]).tanh()
    if view.arm=="permuted":
        received=received.gather(1,permutation[:,:,None].expand_as(received))
    gain=view.gain if view.arm!="zero" else view.gain*0
    # The readback is available to the write calculation only. Only the original
    # four node traces, physical graph and recurrent activity persist into the future.
    with_read=replace(s,receiver_trace=s.receiver_trace+gain[:,None,:]*received)
    edited=net.commit(view.base,replace(experience,state=with_read),draws,intervention=intervention)
    return replace(edited,state=replace(edited.state,receiver_trace=s.receiver_trace))


def permutations(births,pairs,length,seed,step,domain):
    # Each pair shares its permutations; no recipient samples from a different life.
    p=[np.random.default_rng([domain,seed,b,step,952]).random((pairs,length,net.NODES))
       .argsort(-1).repeat(2,axis=0) for b in births]
    return torch.as_tensor(np.concatenate(p),dtype=torch.long)


def rollout(graph,rule,tape,shuffle,pairs,max_wait):
    state=net.start(graph,pairs*2)
    view=rule.expand(pairs*2)
    rewards,motor,edit=[],[],[]
    for t in range(tape.left.shape[1]):
        if t:
            state=net.wait(view.base,state,tape.wait[:,t],max_ticks=max_wait)
        left,right,u=tape.left[:,t],tape.right[:,t],tape.draws[:,t]
        a=net.act(view.base,state,left,right,u[:,0])
        r=(a.side*tape.factor[:,t]+tape.fixed[:,t]).detach()
        e=net.observe(view.base,a,left,right,r)
        w=commit(view,e,u[:,1:],shuffle[:,t])
        state=w.state
        rewards.append(r)
        motor.append(a.motor_log_prob)
        edit.append(w.edit_log_prob)
    def pack(xs):return torch.stack(xs,-1).reshape(len(graph.bias),pairs,2,-1)
    return pack(rewards),pack(motor),pack(edit)


def loss_and_stats(graph,rule,tape,shuffles,config):
    reward,motor,edits=rollout(graph,rule,tape,shuffles,config.pairs,config.max_wait)
    length=reward.shape[-1]
    query=(torch.arange(length)%4>=2).float()*2/length
    adv=common.causal_advantage(reward,query.expand_as(reward))
    loss=-(motor*adv).sum(-1).mean((1,2))-(edits[...,:-1]*adv[...,1:]).sum(-1).mean((1,2))
    with torch.no_grad():
        good=(reward>0).reshape(len(graph.bias),config.pairs,2,-1,4)[...,2:].float()
        stats=torch.stack((good.mean((1,2,3,4)),good.prod(-1).mean((1,2,3)),
                           good[:,:,:,-1].mean((1,2,3))),dim=1)
    return loss.sum(),stats


def load_checkpoint(path,arm,birth,step):
    d=torch.load(age.ckpath(path,arm,step),weights_only=True,map_location="cpu")
    row=d["births"].index(birth)
    if (d["arm"],d["step"])!=(arm,step):raise ValueError("Readback checkpoint mismatch")
    g=Graph(*(d["graph"][k][row:row+1] for k in ("weights","bias","present")))
    rule=ReadRule([birth],arm)
    rule.load_state_dict({k:v[row:row+1] for k,v in d["rule"].items()})
    return g,rule.eval().requires_grad_(False)


def fit(out,source,source_step,cfg,arm,births,stop):
    torch.set_num_threads(1)
    base,base_rules=zip(*(age.load_checkpoint(source,"episodic",b,source_step) for b in births),strict=True)
    weight=nn.Parameter(torch.cat([g.weights for g in base]))
    bias=nn.Parameter(torch.cat([g.bias for g in base]))
    g=Graph(weight,bias,torch.cat([g.present for g in base]))
    rule=ReadRule(births,arm)
    rule.base.load_state_dict({key:torch.cat([r.state_dict()[key] for r in base_rules])
                               for key in base_rules[0].state_dict()})
    parameters=(weight,bias,*rule.parameters())
    opt=torch.optim.Adam(parameters,lr=cfg.lr)
    folder=out/arm
    folder.mkdir(parents=True,exist_ok=True)
    saved=sorted(folder.glob("step-*.pt"))
    step,contacts,history=0,0,[]
    if saved:
        d=torch.load(saved[-1],weights_only=True,map_location="cpu")
        if (d["arm"],d["births"],d["config"])!=(arm,list(births),asdict(cfg)):
            raise ValueError("Incompatible readback resume")
        with torch.no_grad():
            weight.copy_(d["graph"]["weights"]);bias.copy_(d["graph"]["bias"])
        rule.load_state_dict(d["rule"]);opt.load_state_dict(d["optimizer"])
        step,contacts,history=d["step"],d["contacts"],d["trace"].tolist()
    def save():
        payload=dict(config=asdict(cfg),arm=arm,births=list(births),step=step,contacts=contacts,
                     graph=dict(weights=weight.detach(),bias=bias.detach(),present=g.present),
                     rule=rule.state_dict(),optimizer=opt.state_dict(),trace=torch.as_tensor(history))
        common._atomic_save(age.ckpath(out,arm,step),lambda f:torch.save(payload,f))
    if not saved:save()
    initial=step
    started=perf_counter()
    for step in range(initial+1,stop+1):
        choices=(2,) if step<=1000 else (2,4) if step<=3000 else (2,4,8)
        blocks=int(np.random.default_rng([882013,cfg.seed,step]).choice(choices))
        tape=age.make_tape(cfg,births,step,blocks,domain=882019)
        perm=permutations(births,cfg.pairs,blocks*4,cfg.seed,step,882023)
        opt.zero_grad(set_to_none=True)
        loss,stats=loss_and_stats(g,rule,tape,perm,cfg)
        loss.backward()
        with torch.no_grad():
            norms=sum(p.grad.flatten(1).double().square().sum(1) for p in parameters).sqrt()
            if not torch.isfinite(norms).all():raise FloatingPointError(f"Nonfinite gradient {step}/{arm}")
            scale=(cfg.grad_clip/norms.clamp_min(1e-12)).clamp_max(1)
            for p in parameters:p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)).to(p.dtype))
        opt.step()
        contacts+=blocks*4*2*cfg.pairs
        history.append(torch.cat((stats,rule.gain.detach().tanh(),norms[:,None].float()),1).tolist())
        if step%cfg.checkpoint_every==0 or step==stop:save()
        if step%200==0 or step in (1,stop):
            avg=np.mean(history[-100:],axis=0)
            print(arm,step,avg[:,0].round(3),avg[:,3:5].mean(0).round(3),
                  round((perf_counter()-started)/(step-initial),3),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=age.OUTPUT.parent/"readback-fit")
    p.add_argument("--source",type=Path,default=age.OUTPUT)
    p.add_argument("--source-step",type=int,default=4000)
    p.add_argument("--births",type=int,nargs="+",default=list(range(731,737)))
    p.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    p.add_argument("--updates",type=int,default=8000)
    p.add_argument("--stop",type=int)
    p.add_argument("--pairs",type=int,default=8)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--lr",type=float,default=.0015)
    a=p.parse_args()
    cfg=age.Config(updates=a.updates,pairs=a.pairs,lr=a.lr,seed=681219,checkpoint_every=500)
    if len(a.births)!=len(set(a.births)) or cfg.pairs<2 or not 1<=(a.stop or a.updates)<=a.updates:
        raise ValueError("Invalid readback fit budget")
    src=age.ckpath(a.source,"episodic",a.source_step)
    plan=dict(config=asdict(cfg),births=a.births,arms=a.arms,source_step=a.source_step,
              source_sha256=sha256(src.read_bytes()).hexdigest(),code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              formula="Two inherited scalar gains, initialized to zero, add tanh((W times sender_trace)/sqrt(in_degree)) "
                      "to the receiver trace used in this commit only. Tagged neurons, physical actions, clocks and number "
                      "of stored trace values remain as in the 20-node graph. Wrong-receiver control permutes rows within "
                      "each life independently of taste and of motor/edit draws. Zero-gain group shares the same curriculum.",
              schedule="Start all arms from the same young episodic checkpoint, reset Adam equally. 2 continuous blocks "
                       "through step 1000, 2 or 4 through step 3000, then 2/4/8 with equal probability. Actual rewards only.")
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/"run.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Frozen protocol mismatch")
    common._atomic_save(path,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in [pool.submit(fit,a.out,a.source,a.source_step,cfg,arm,a.births,a.stop or a.updates)
                  for arm in a.arms]:f.result()


if __name__=="__main__":
    main()
