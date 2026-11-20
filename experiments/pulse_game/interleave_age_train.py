"""Train a plastic graph to keep learning after previous writes in the same life."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import local_trace_graph as net, local_trace_train as old
from . import interleave_train as common
from .self_param_writer import Graph


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interleaved-learning-20261120/lifetime-fit"
ARMS = ("carry", "episodic")
ODOR, CONTEXT, FEEDER = range(3)
TRAIN = 730519
DEV = 730523


@dataclass(frozen=True)
class Config:
    updates: int = 10000
    pairs: int = 8
    lr: float = .0015
    grad_clip: float = 2.
    max_wait: int = 8
    seed: int = 171095
    checkpoint_every: int = 500
    warm_source: str = "odor"
    max_blocks: int = 8


@dataclass
class Tape:
    left: torch.Tensor
    right: torch.Tensor
    factor: torch.Tensor
    fixed: torch.Tensor
    draws: torch.Tensor
    wait: torch.Tensor
    kind: torch.Tensor
    obj: torch.Tensor
    laws: torch.Tensor


def signs(rng, shape):
    return (rng.integers(0, 2, size=shape, dtype=np.int8)*2-1).astype(np.float32)


def directions(rng, *shape):
    x = rng.normal(size=(*shape,3)).astype(np.float32)
    return 1.35*x / np.linalg.norm(x,axis=-1,keepdims=True)


def life_blocks(config, step):
    # Length is sampled before any trajectory. Histories for the two arms are matched exactly.
    rng=np.random.default_rng([TRAIN,config.seed,step,9])
    upper = min(config.max_blocks, 4) if step <= 1000 else config.max_blocks
    return int(rng.choice([n for n in (2,4,8,16) if n<=upper]))


def single_tape(config, birth, step, blocks, *, domain=TRAIN):
    rng=np.random.default_rng([domain,config.seed,birth,step,1])
    n=config.pairs
    rows=np.arange(n)[:,None]
    # IDs are environmental bookkeeping. The graph receives the presented vectors only.
    vectors=directions(rng,n,blocks+3)
    laws=signs(rng,(n,blocks+3))
    context_law=signs(rng,(n,2))
    cues=common.CUE_BANK[rng.permuted(np.tile(np.arange(3),(n,1)),axis=1)[:,:2]]
    mark_law=signs(rng,n)
    novel_mark=directions(rng,n)
    use_fixed=rng.random(n)<.65
    mark=np.where(use_fixed[:,None], common.MARK_AXIS, novel_mark)
    l,r,side_factor,constant,kinds,objects,hidden_laws=[],[],[],[],[],[],[]
    last_b=np.full(n,-1)
    for block in range(blocks):
        kind=rng.choice((ODOR,CONTEXT,FEEDER),size=n,p=(.5,.25,.25))
        # Each block contains two exposures followed by two unpredictable returns. The
        # current fresh candidate can be replaced by a previously experienced object.
        new_b=np.full(n,block+1)
        if block:
            reuse=(last_b>=0)&(rng.random(n)<.35)
            new_b=np.where(reuse, last_b, new_b)
        last_b=np.where(kind==ODOR,new_b,last_b)
        anchor_first=rng.integers(2,size=n)
        anchor_probe=rng.integers(2,size=n)
        object_slot=np.stack((anchor_first,1-anchor_first,anchor_probe,1-anchor_probe),axis=1)
        ids=np.where(object_slot==0, 0, new_b[:,None])
        odor=vectors[rows,ids]
        pref=laws[rows,ids]
        is_ctx=kind==CONTEXT
        odor[is_ctx]=np.broadcast_to(vectors[is_ctx,0,None],(int(is_ctx.sum()),4,3))
        pref[is_ctx]=context_law[rows,object_slot][is_ctx]
        ambient=cues[rows,object_slot]*is_ctx[:,None,None]
        orient=signs(rng,(n,4))
        left=ambient-odor*orient[...,None]
        right=ambient+odor*orient[...,None]
        signed=orient*pref
        fixed=np.zeros_like(signed)
        marks=kind==FEEDER
        orient[:,1]=-orient[:,0]
        mark_view=mark[:,None]*orient[...,None]
        right[marks]=mark_view[marks]
        left[marks]=mark_view[marks]
        left[marks,2:]*=-1
        signed[marks,:2]=0
        signed[marks,2:]=(orient[:,2:]*mark_law[:,None])[marks]
        fixed[marks,:2]=(orient[:,:2]*mark_law[:,None])[marks]
        pref[marks]=mark_law[marks,None]
        ids=np.where(is_ctx[:,None],-1-object_slot,ids)
        ids[marks]=-3
        for dest,value in zip((l,r,side_factor,constant,kinds,objects,hidden_laws),
                              (left,right,signed,fixed,np.repeat(kind[:,None],4,1),ids,pref),strict=True):
            dest.append(value)
    t=blocks*4
    def public(parts):
        return np.repeat(np.concatenate(parts,1)[:,None],2,axis=1)
    def hidden(parts):
        v=np.concatenate(parts,1)
        return np.stack((v,-v),axis=1)
    return (public(l),public(r),hidden(side_factor),hidden(constant),
            np.repeat(rng.random((n,1,t,5),dtype=np.float32),2,axis=1),
            np.repeat(rng.integers(0,config.max_wait+1,size=(n,1,t)),2,axis=1),
            public(kinds),public(objects),hidden(hidden_laws))


def make_tape(config,births,step,blocks,*,domain=TRAIN):
    samples=[single_tape(config,s,step,blocks,domain=domain) for s in births]
    return Tape(*(torch.from_numpy(np.concatenate(v,axis=0).reshape(-1,*v[0].shape[2:]))
                  for v in zip(*samples,strict=True)))


def rollout(born,rule,tape,arm,pairs,max_wait,*,intervention="active"):
    nroot=len(born.bias)
    cold=net.start(born,2*pairs)
    coeff=rule.expand(2*pairs)
    state=cold
    rewards,motor,edit=[],[],[]
    for t in range(tape.left.shape[1]):
        if arm=="episodic" and t%4==0:
            state=cold
        if t>0:
            state=net.wait(coeff,state,tape.wait[:,t],max_ticks=max_wait)
        u=tape.draws[:,t]
        left,right=tape.left[:,t],tape.right[:,t]
        a=net.act(coeff,state,left,right,u[:,0])
        reward=(a.side*tape.factor[:,t]+tape.fixed[:,t]).detach()
        e=net.observe(coeff,a,left,right,reward)
        commit=net.commit(coeff,e,u[:,1:],intervention=intervention)
        state=commit.state
        rewards.append(reward)
        motor.append(a.motor_log_prob)
        edit.append(commit.edit_log_prob)
    def collect(v):return torch.stack(v,-1).reshape(nroot,pairs,2,-1)
    return collect(rewards),collect(motor),collect(edit)


def objective(born,rule,tape,arm,pairs,max_wait):
    reward,motor,edit=rollout(born,rule,tape,arm,pairs,max_wait)
    nroot,_,_,t=reward.shape
    # No early target supplied. Both physical quiz resources contribute identically.
    weighted=reward.detach().clone()
    weighted[...,::4]=0
    weighted[...,1::4]=0
    weighted/=t/2
    other=weighted.mean(2,keepdim=True)
    baseline=(other.sum(1,keepdim=True)-other)/(pairs-1)
    adv=(weighted-baseline).flip(-1).cumsum(-1).flip(-1)
    motor_loss=-(motor*adv).sum(-1).mean((1,2))
    edit_loss=-(edit[...,:-1]*adv[...,1:]).sum(-1).mean((1,2))
    with torch.no_grad():
        good=(reward>0).reshape(nroot,pairs,2,-1,4)[...,2:].float()
        kind=tape.kind.reshape(nroot,pairs,2,-1,4)[...,2:]
        # A count of zero is retained explicitly; no accuracy is imputed for that type.
        metrics=[good.mean((1,2,3,4)),good[:,:,:,0].mean((1,2,3)),
                 good[:,:,:,-1].mean((1,2,3)),good.prod(-1).mean((1,2,3))]
        for shape in (ODOR,CONTEXT,FEEDER):
            chosen=kind==shape
            metrics.extend(((good*chosen).sum((1,2,3,4)),chosen.sum((1,2,3,4))))
        values=torch.stack(metrics,1).numpy()
    return (motor_loss+edit_loss).sum(), values


def init(births, source):
    if source=="cold":
        g,r=net.spawn(births),net.Rule(births)
    else:
        gs,rs=zip(*(old.load_checkpoint(ROOT/"data/local-trace-20261119/local",source,"active",b,8000)
                    for b in births),strict=True)
        g=Graph(*(torch.cat([getattr(s,name) for s in gs]) for name in ("weights","bias","present")))
        r=net.Rule(births)
        r.load_state_dict({k:torch.cat([s.state_dict()[k] for s in rs]) for k in r.state_dict()})
    return Graph(nn.Parameter(g.weights),nn.Parameter(g.bias),g.present),r


def ckpath(path,arm,step):
    return Path(path)/arm/f"step-{step:06d}.pt"


def load_checkpoint(path,arm,birth,step):
    data=torch.load(ckpath(path,arm,step),weights_only=True,map_location="cpu")
    if data["arm"]!=arm or data["step"]!=step:
        raise ValueError("Age checkpoint identity mismatch")
    at=data["births"].index(birth)
    graph=Graph(*(data["graph"][k][at:at+1] for k in ("weights","bias","present")))
    rule=net.Rule(birth)
    rule.load_state_dict({k:v[at:at+1] for k,v in data["rule"].items()})
    return graph,rule.eval().requires_grad_(False)


def fit(path,cfg,births,arm,stop):
    torch.set_num_threads(1)
    graph,rule=init(births,cfg.warm_source)
    params=(graph.weights,graph.bias,*rule.parameters())
    opt=torch.optim.Adam(params,lr=cfg.lr)
    folder=Path(path)/arm
    folder.mkdir(parents=True,exist_ok=True)
    files=sorted(folder.glob("step-*.pt"))
    step,n_contact,log=0,0,[]
    if files:
        saved=torch.load(files[-1],weights_only=True,map_location="cpu")
        if saved["births"]!=list(births) or saved["config"]!=asdict(cfg) or saved["arm"]!=arm:
            raise ValueError("Run differs from resume checkpoint")
        with torch.no_grad():
            graph.weights.copy_(saved["graph"]["weights"]);graph.bias.copy_(saved["graph"]["bias"])
        rule.load_state_dict(saved["rule"]); opt.load_state_dict(saved["optimizer"])
        step,n_contact,log=saved["step"],saved["contacts"],saved["trace"].tolist()
    def save():
        obj=dict(births=list(births),step=step,arm=arm,config=asdict(cfg),contacts=n_contact,
                 graph={f.name:getattr(graph,f.name).detach() for f in fields(Graph)},
                 rule=rule.state_dict(),optimizer=opt.state_dict(),trace=torch.tensor(log))
        common._atomic_save(ckpath(path,arm,step),lambda f:torch.save(obj,f))
    if step==0:
        save()
    started=perf_counter()
    initial_step=step
    for step in range(step+1,stop+1):
        blocks=life_blocks(cfg,step)
        tape=make_tape(cfg,births,step,blocks)
        opt.zero_grad(set_to_none=True)
        loss,val=objective(graph,rule,tape,arm,cfg.pairs,cfg.max_wait)
        loss.backward()
        with torch.no_grad():
            norms=sum(p.grad.flatten(1).double().square().sum(1) for p in params).sqrt()
            if not torch.isfinite(norms).all():
                raise FloatingPointError(f"Nonfinite gradient at {arm} step {step}")
            shrink=(cfg.grad_clip/norms.clamp_min(1e-12)).clamp_max(1)
            for p in params:
                p.grad.mul_(shrink.reshape((-1,)+(1,)*(p.ndim-1)).to(p.dtype))
        opt.step()
        n_contact+=cfg.pairs*2*4*blocks
        log.append(np.column_stack((val,norms.numpy(),np.full(len(births),blocks))).tolist())
        if step%cfg.checkpoint_every==0 or step==stop:
            save()
        if step==1 or step%100==0 or step==stop:
            recent=np.array(log[-100:])[:,:,0].mean(0)
            print(arm,step, cfg.updates, np.round(recent,3),
                  round((perf_counter()-started)/max(step-initial_step,1),3),"s/step",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUTPUT)
    p.add_argument("--births",type=int,nargs="+",default=list(old.BIRTHS))
    p.add_argument("--arms",choices=ARMS,nargs="+",default=ARMS)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--stop",type=int)
    for k,v in asdict(Config()).items():
        p.add_argument("--"+k.replace("_","-"),type=type(v),default=v)
    a=p.parse_args()
    cfg=Config(**{f.name:getattr(a,f.name) for f in fields(Config)})
    if cfg.pairs<2 or cfg.max_blocks not in (2,4,8,16) or cfg.updates<1 or cfg.checkpoint_every<1:
        raise ValueError("Invalid training budget")
    stop=a.stop or cfg.updates
    if not 1<=stop<=cfg.updates:raise ValueError("Requested stop outside training budget")
    a.out.mkdir(parents=True,exist_ok=True)
    plan=dict(config=asdict(cfg),births=a.births,arms=list(a.arms),training_domain=TRAIN,
              source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              metrics=["quiz_accuracy","first_block","last_block","both_quizzes","odor_correct",
                       "odor_count","context_correct","context_count","feeder_correct","feeder_count",
                       "grad_norm","blocks"],
              comparison="Same worlds, randomness, learning opportunities, elapsed ticks and optimizer count. "
              "carry is continuous across every block; episodic resets body activity, graph and traces to birth at each block. "
              "Only the earned rewards of each block's two revisits are externally scored. Edits and motor actions receive "
              "their appropriate future sampled credit. No private law enters the graph.")
    manifest=a.out/"run.json"
    if manifest.exists() and json.loads(manifest.read_text())!=plan:
        raise ValueError("Output directory belongs to a different protocol")
    common._atomic_save(manifest,lambda f:f.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8"))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in [pool.submit(fit,a.out,cfg,a.births,arm,stop) for arm in a.arms]:
            future.result()


if __name__=="__main__":
    main()
