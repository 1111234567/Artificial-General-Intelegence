"""Test four route coefficients fed by two physically continuous candidate previews."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import noon_horizon as env, noon_horizon_anchor as anchor, noon_horizon_eval as metrics


PLAN = env.EVIDENCE / "horizon-anchor-preview-plan.json"
ROOT = env.DATA / "preview_anchored"
MODES = ("informative","uninformative")
FEATURES = 4
LENGTH = 4


def previews(rule,state,tape):
    rows = torch.arange(tape.lanes)
    first = (tape.external_draw > .5).long()
    output = torch.zeros((tape.lanes,2),dtype=state.hidden.dtype)
    for slot in (first,1-first):
        item = tape.candidates[rows,slot]
        sign = tape.final_orientation[rows,item]
        left,right = env.world.sensory_for(tape,item,sign)
        read = env.net.act(rule,state,left,right,stochastic=False)
        state = read.state
        output[rows,slot] = read.motor_logit.abs()
    return state,output


def route_basis(rule,state,tape,cue,delta):
    action, read = anchor.observe_route(rule,state,tape,cue)
    pulse_response = read[:,anchor.WIDTH]
    basis = torch.column_stack((torch.ones_like(delta),delta,pulse_response,delta*pulse_response))
    return action,basis


def deploy(graph,rule,tape,missing,future,cue,horizon,head=None,*,quiet=False,known=False):
    state,_ = env.initial(graph,rule,tape,missing)
    coeff = rule.expand(tape.lanes)
    if quiet:
        state=env.net.wait(coeff,state,2*env.net.REACTION_TICKS)
        difference=torch.zeros_like(cue)
    else:
        state,tally=previews(coeff,state,tape)
        difference=tally[:,1]-tally[:,0]
    origin,features = route_basis(coeff,state,tape,cue,difference)
    logit = origin.motor_logit if head is None else origin.motor_logit + (features*head).sum(1)
    side = 2*(tape.decision_draw<logit.sigmoid()).float()-1
    if known:
        side=torch.where(tape.candidates[:,1]==missing,-1.,1.)
        logit=side*float("inf")
    events,_=anchor.execute(coeff,origin.state,tape,future,side,horizon)
    return side,logit,events


@torch.no_grad()
def bank(graph,rule,tape,missing,future,*,quiet=False):
    state,history=env.initial(graph,rule,tape,missing)
    coeff=rule.expand(tape.lanes)
    if quiet:
        state=env.net.wait(coeff,state,2*env.net.REACTION_TICKS)
        scores=torch.zeros((tape.lanes,2),dtype=state.hidden.dtype)
    else:
        state,scores=previews(coeff,state,tape)
    choice = scores[:,1]-scores[:,0]
    indices=torch.arange(tape.lanes).repeat_interleave(2)
    twin=env.lesson.expand(tape,indices)
    symbol=state.hidden.new_tensor((-1.,1.)).repeat(tape.lanes)
    action,basis=route_basis(rule.expand(twin.lanes),state.subset(indices),twin,symbol,choice[indices])
    result,_=env.branches(graph,rule,action.state,twin,future.subset(indices),LENGTH)
    return dict(basis=basis.reshape(tape.lanes,2,FEATURES).numpy(),native=action.motor_logit.reshape(tape.lanes,2).numpy(),
                events=result.reshape(tape.lanes,2,2,LENGTH+1,2).to(torch.int8).numpy(),phase=future.cue_noise.numpy(),
                no_taste=(tape.candidates[:,1]==missing).long().numpy(),candidate=tape.candidates.numpy(),
                known_history=history.to(torch.int8).numpy(),preview=scores.numpy(),queries=future.target.numpy())


def fit(raw,plan,birth):
    n=len(raw["events"])
    rows=torch.arange(n)[:,None]
    horizon=torch.arange(2)[None].expand(n,-1)
    phase=torch.from_numpy(raw["phase"])[:,None]>0
    signals=(horizon,torch.where(phase,horizon,1-horizon))
    y=torch.from_numpy(raw["events"][...,1]).float()
    delta=(y[:,:,1]-y[:,:,0]).cumsum(-1)
    basis=torch.stack([torch.from_numpy(raw["basis"])[rows,s].reshape(-1,8,FEATURES) for s in signals])
    original=torch.stack([torch.from_numpy(raw["native"])[rows,s].reshape(-1,8) for s in signals])
    value=torch.stack([delta[rows,s,horizon*LENGTH].reshape(-1,8) for s in signals])
    head=torch.nn.Parameter(torch.zeros(len(MODES),FEATURES))
    optim=torch.optim.Adam((head,),lr=plan["lr"])
    draw=torch.Generator().manual_seed(plan["optimizer_seed"]+birth)
    trace=[]
    for step in range(1,plan["updates"]+1):
        sample=torch.randint(basis.shape[1],(plan["geometries_per_update"],),generator=draw)
        revised=original[:,sample]+torch.einsum("atnf,af->atn",basis[:,sample],head)
        score=(value[:,sample]*revised.sigmoid()).mean((1,2))
        optim.zero_grad(set_to_none=True)
        (-score.sum()).backward()
        optim.step()
        if step%250==0:trace.append(score.detach())
    if not head.isfinite().all():raise FloatingPointError("Nonfinite preview route residual")
    return {arm:values for arm,values in zip(MODES,head.detach().numpy(),strict=True)},torch.stack(trace).numpy()


@torch.no_grad()
def tests(graph,rule,tape,missing,future,heads):
    active=bank(graph,rule,tape,missing,future)
    quiet=bank(graph,rule,tape,missing,future,quiet=True)
    output={**active,"quiet_events":quiet["events"]}
    for label,values in ((*heads.items(),("preview_native",None),("quiet_native",None),("always_known",None))):
        sides,logs,earned=[],[],[]
        bypass=label=="quiet_native"
        potential=quiet["events"] if bypass else active["events"]
        for u,horizon in enumerate((0,LENGTH)):
            signal=tape.decision_draw.new_full((tape.lanes,),2*u-1)
            chosen,logit,events=deploy(graph,rule,tape,missing,future,signal,horizon,values,quiet=bypass,known=label=="always_known")
            recorded=events.to(torch.int8).numpy()
            branch=potential[np.arange(tape.lanes),u,(chosen>0).long().numpy()]
            np.testing.assert_array_equal(recorded[:,:horizon+1],branch[:,:horizon+1])
            if horizon == 0:assert np.count_nonzero(recorded[:,1:])==0
            sides.append(chosen.to(torch.int8).numpy());logs.append(logit.numpy());earned.append(recorded)
        for name,values in (("action",sides),("motor_logit",logs),("direct_events",earned)):
            output[f"{label}/{name}"]=np.stack(values,axis=1)
    return output


def measures(raw,worlds):
    by_arm={}
    for arm in (*MODES,"preview_native","quiet_native","always_known"):
        target={key:raw[key] for key in ("candidate","no_taste")}
        target.update({key:raw[f"{arm}/{key}"] for key in ("action","motor_logit")})
        target["events"]=raw["quiet_events" if arm=="quiet_native" else "events"]
        reading=metrics.measure(target,worlds)
        real=raw[f"{arm}/direct_events"][...,1].sum(-1).reshape(worlds,4,2).mean(1)
        for j,length in enumerate(("short","long")):
            np.testing.assert_array_equal(reading[f"{length}/native/total"],real[:,j])
        by_arm[arm]=reading
    result={f"{arm}/{key}":value for arm,entry in by_arm.items() for key,value in entry.items()}
    for alternate in ("uninformative","preview_native","quiet_native","always_known"):
        for key,value in by_arm["informative"].items():
            result[f"informative_minus_{alternate}/{key}"]=value-by_arm[alternate][key]
    result["observed_lower_preview_is_unknown"]=(raw["preview"].argmin(1)==raw["no_taste"]).reshape(worlds,4).mean(1)
    return result


def run(cohort):
    plan=json.loads(PLAN.read_text(encoding="utf8"))
    _,parents=anchor.settings()
    plan_sha=env.digest(PLAN)
    code={Path(p).name:env.digest(p) for p in (__file__,anchor.__file__,env.__file__,metrics.__file__,env.net.__file__)}
    checkpoints, files, rows = {}, {}, []
    train,hold=(plan[f"{key}_seeds"][cohort] for key in ("training","holdout"))
    births=env.COHORTS[cohort]
    worlds,test_count=plan["worlds_per_bank"],plan["holdout_worlds"]
    folder=ROOT/cohort;folder.mkdir(parents=True,exist_ok=True)
    for birth in births:
        start=perf_counter()
        body=env.checkpoint(cohort,birth)
        identity=env.digest(body)
        address=body.relative_to(env.world.ROOT).as_posix()
        if identity!=parents[address]:raise ValueError("Frozen body changed")
        checkpoints[address]=identity
        graph,rule=env.load(cohort,birth)
        signature=repr((plan_sha,code,identity,birth))
        experience=folder/f"bank-{birth}.npz"
        weights=folder/f"head-{birth}.npz"
        test=folder/f"holdout-{birth}.npz"
        if not experience.exists():env.storage._write_raw(experience,**anchor.replay(graph,rule,train,worlds,bank),signature=signature)
        if not weights.exists():
            with np.load(experience,allow_pickle=False) as saved:
                if str(saved["signature"])!=signature:raise ValueError("Wrong preview training bank")
                learned,trace=fit(saved,plan,birth)
            env.storage._write_raw(weights,**learned,trace=trace,signature=signature,body_sha256=identity,
                                    reference_sha256=env.digest(experience),updates=plan["updates"],plan_sha256=plan_sha)
        with np.load(weights,allow_pickle=False) as saved:
            if str(saved["signature"])!=signature:raise ValueError("Wrong preview route residual")
            head={mode:torch.from_numpy(saved[mode]) for mode in MODES}
            if any(x.shape!=(FEATURES,) or not x.isfinite().all() for x in head.values()):raise ValueError("Invalid preview coefficients")
        if not test.exists():
            record=anchor.replay(graph,rule,hold,test_count,lambda *args:tests(*args,head))
            env.storage._write_raw(test,**record,signature=signature,head_sha256=env.digest(weights))
        with np.load(test,allow_pickle=False) as saved:
            if str(saved["signature"])!=signature or str(saved["head_sha256"])!=env.digest(weights):raise ValueError("Wrong preview holdout")
            rows.append(measures(saved,test_count))
        files.update({p.relative_to(env.world.ROOT).as_posix():env.digest(p) for p in (experience,weights,test)})
        print(cohort,birth,round(perf_counter()-start,2),flush=True)
    batch={key:np.stack([x[key] for x in rows]) for key in rows[0]}
    summary=env.intervals.PairedCrossed(len(births),test_count,hold+331).summary(batch)
    result=dict(cohort=cohort,births=list(births),training_worlds=worlds,holdout_worlds=test_count,train_seed=train,holdout_seed=hold,
                files=files,body_sources=checkpoints,plan_sha256=plan_sha,code=code,metrics=summary)
    env.storage._write_json(env.EVIDENCE/f"horizon-anchor-preview-{cohort}.json",result)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--cohort",choices=env.COHORTS,required=True)
    args=cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(args.cohort)
