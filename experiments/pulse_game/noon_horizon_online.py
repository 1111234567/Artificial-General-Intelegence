"""Learn four physical route coefficients from one actual sampled path per training life."""

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from . import noon_horizon as env, noon_horizon_anchor as anchor, noon_horizon_anchor_preview as preview


PLAN = env.EVIDENCE / "horizon-online-plan.json"
ROOT = env.DATA / "online"
ARMS = ("online", "paired")
WIDTH = preview.FEATURES
LENGTH = preview.LENGTH
TRACE_EVERY = 128


def settings():
    plan=json.loads(PLAN.read_text(encoding="utf8"))
    parent=env.EVIDENCE/"horizon-anchor-seal.json"
    if env.digest(parent)!=plan["parent_anchor_seal_sha256"]:
        raise ValueError("Frozen physical preview protocol changed")
    return plan,json.loads(parent.read_text(encoding="utf8"))["body_unchanged"]


def inherit(cohort, signatures):
    births=env.COHORTS[cohort]
    for b in births:
        p=env.checkpoint(cohort,b)
        if env.digest(p)!=signatures[p.relative_to(env.world.ROOT).as_posix()]:
            raise ValueError("Frozen physical body changed")
    loaded=[env.load(cohort,b) for b in births]
    group=env.net.Graph(*(torch.cat([getattr(g,k) for g,_ in loaded]) for k in ("weights","bias","present")))
    rule=env.net.Rule(births)
    rule.load_state_dict({k:torch.cat([r.state_dict()[k] for _,r in loaded]) for k in rule.state_dict()})
    return group,rule.eval().requires_grad_(False)


def decision_draw(seed,births,step,worlds):
    values=[]
    for b in births:
        rng=np.random.default_rng([0x4f4e454c,seed,b,step])
        v=rng.random((worlds,2),dtype=np.float32)
        values.append(np.repeat(v,4,axis=0))
    return torch.from_numpy(np.concatenate(values))


@torch.no_grad()
def prepare(graph,rule,tape,missing):
    state,_=env.initial(graph,rule,tape,missing)
    coeff=rule.expand(tape.lanes//len(graph.bias))
    state,score=preview.previews(coeff,state,tape)
    return coeff,state,score[:,1]-score[:,0]


@torch.no_grad()
def online_path(coeff,state,tape,future,logit,uniform,horizon):
    side=2*(uniform<logit.sigmoid()).to(logit.dtype)-1
    events,_=anchor.execute(coeff,state,tape,future,side,horizon)
    return side,events


def update(graph,rule,parameters,optimizer,mean_past,tape,missing,future,draws,arm,baseline_rate):
    size=len(graph.bias)
    coeff,state,difference=prepare(graph,rule,tape,missing)
    world_per_root=tape.lanes//size
    losses, paid, actions, probability, logits, gains=[],[],[],[],[],[]
    for i,horizon in enumerate((0,LENGTH)):
        with torch.no_grad():
            cue=state.hidden.new_full((tape.lanes,),2*i-1)
            origin,features=preview.route_basis(coeff,state,tape,cue,difference)
        signal=origin.motor_logit.reshape(size,world_per_root)
        x=features.reshape(size,world_per_root,WIDTH)
        l=signal+(x*parameters[:,None]).sum(2)
        p=l.detach().sigmoid()
        sampled=draws[:,i]
        if arm=="online":
            taken,experience=online_path(coeff,origin.state,tape,future,l.detach().flatten(),sampled,horizon)
            grade=experience[...,1].sum(1).reshape(size,world_per_root)
            logprob=F.logsigmoid(taken.reshape_as(l)*l)
            losses.append(((grade-mean_past[:,i,None])*logprob).mean(1))
            paid.append(experience.reshape(size,world_per_root,LENGTH+1,2).to(torch.int8))
            actions.append(taken.reshape_as(l).to(torch.int8))
            gains.append(grade.mean(1))
        else:
            siblings,_=env.branches(graph,rule,origin.state,tape,future,horizon)
            grade=siblings[...,1].sum(-1).reshape(size,world_per_root,2)
            losses.append((l.sigmoid()*(grade[:,:,1]-grade[:,:,0])).mean(1))
            both=siblings.reshape(size,world_per_root,2,LENGTH+1,2).to(torch.int8)
            paid.append(both)
            actions.append(l.new_tensor((-1.,1.)).repeat(size,world_per_root,1).to(torch.int8))
            gains.append(((1-p)*grade[:,:,0]+p*grade[:,:,1]).mean(1))
        probability.append(p)
        logits.append(l.detach())
    optimizer.zero_grad(set_to_none=True)
    loss=-torch.stack(losses).mean(0).sum()
    if not loss.isfinite():raise FloatingPointError("Nonfinite accumulated route credit")
    loss.backward()
    if not parameters.grad.isfinite().all():raise FloatingPointError("Nonfinite route gradient")
    optimizer.step()
    realized=torch.stack(gains,dim=1).detach()
    previous=mean_past.clone()
    if arm=="online":
        mean_past.lerp_(realized,baseline_rate)
    return {"physical_action":torch.stack(actions,dim=1),"physical_events":torch.stack(paid,dim=1),
            "route_probability":torch.stack(probability,dim=1),"route_logit":torch.stack(logits,dim=1),
            "observed_or_weighted_resource":realized,"baseline_before":previous}


def train(cohort,stream,arm,benchmark=None):
    plan,parents=settings()
    roots=env.COHORTS[cohort]
    seed=plan["training_seeds"][cohort][stream-1]
    worlds=plan[f"{arm}_geometries_per_update"]
    file=ROOT/"training"/cohort/f"stream-{stream}"/arm
    signature={Path(p).name:env.digest(p) for p in (__file__,preview.__file__,anchor.__file__,env.__file__,env.net.__file__)}
    protocol=dict(cohort=cohort,stream=stream,arm=arm,births=list(roots),seed=seed,worlds=worlds,
                  plan_sha256=env.digest(PLAN),code=signature,
                  parent={str(b):env.digest(env.checkpoint(cohort,b)) for b in roots})
    graph,rule=inherit(cohort,parents)
    parameters=torch.nn.Parameter(torch.zeros((len(roots),WIDTH)))
    optimizer=torch.optim.Adam((parameters,),lr=plan["lr"])
    ex_ante=torch.zeros((len(roots),2))
    history={}
    done=0
    if benchmark is None:
        file.mkdir(parents=True,exist_ok=True)
        info=file/"run.json"
        if info.exists() and json.loads(info.read_text(encoding="utf8"))!=protocol:
            raise ValueError("Training provenance changed")
        if not info.exists():env.storage._write_json(info,protocol)
        resume=file/"resume.pt"
        if resume.exists():
            contents=torch.load(resume,map_location="cpu",weights_only=True)
            if contents["protocol"]!=protocol:raise ValueError("Wrong training resume")
            with torch.no_grad():parameters.copy_(contents["parameters"])
            optimizer.load_state_dict(contents["optimizer"])
            done=contents["step"]
            ex_ante=contents["baseline"]
            history={k:list(t.unbind(0)) for k,t in contents["trace"].items()}
    started=perf_counter()
    target=plan["updates"] if benchmark is None else benchmark
    source_seed=seed if benchmark is None else seed+777
    cutoff=datetime.fromisoformat(plan["compute_deadline_beijing"]+"+08:00")
    for iteration in range(done+1,target+1):
        tape,missing=env.lesson.prepare(source_seed,roots,iteration,worlds,2,omissions=True)
        future=env.opportunities(source_seed,roots,iteration,worlds,LENGTH)
        draw=decision_draw(source_seed,roots,iteration,worlds)
        record=update(graph,rule,parameters,optimizer,ex_ante,tape,missing,future,draw,arm,plan["online_baseline_rate"])
        if benchmark is not None:continue
        for key,val in record.items():history.setdefault(key,[]).append(val)
        if iteration%TRACE_EVERY!=0 and iteration!=target:continue
        tracks={k:torch.stack(v) for k,v in history.items()}
        contents=dict(protocol=protocol,step=iteration,parameters=parameters.detach(),optimizer=optimizer.state_dict(),
                      baseline=ex_ante,trace=tracks)
        env.storage._atomic(resume,lambda handle:torch.save(contents,handle))
        recent=tracks["observed_or_weighted_resource"][-TRACE_EVERY:].mean((0,1))
        print(cohort,stream,arm,iteration,round((perf_counter()-started)/(iteration-done),3),recent.tolist(),flush=True)
        if iteration<target and datetime.now(cutoff.tzinfo)>=cutoff:
            raise TimeoutError("Registered compute stop reached before endpoint")
    if benchmark is not None:
        print(cohort,stream,arm,benchmark,round((perf_counter()-started)/benchmark,3),flush=True)
        return
    if not history:raise ValueError("Missing registered training ledgers")
    traces={k:torch.stack(v).numpy() for k,v in history.items()}
    env.storage._write_raw(file/"paid-training-ledger.npz",**traces,step=target,plan_sha256=protocol["plan_sha256"],seed=seed)
    env.storage._write_raw(file/"route-residual.npz",parameters=parameters.detach().numpy(),step=target,
                           plan_sha256=protocol["plan_sha256"],parent_hashes=np.asarray(list(protocol["parent"].values())),
                           paid_ledger_sha256=env.digest(file/"paid-training-ledger.npz"))


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--cohort",choices=env.COHORTS,required=True)
    cli.add_argument("--stream",type=int,choices=(1,2),required=True)
    cli.add_argument("--arm",choices=ARMS)
    cli.add_argument("--benchmark",type=int)
    args=cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for arm in ARMS if args.arm is None else (args.arm,):
        train(args.cohort,args.stream,arm,args.benchmark)
