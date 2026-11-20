"""Verify the exact paid-life budgets and sampled routes used to train both controllers."""

import json
from pathlib import Path

import numpy as np
import torch

from . import noon_horizon_online as train, noon_horizon as env


def ledger(cohort, stream, arm, plan, graph, rule):
    births=env.COHORTS[cohort]
    seed=plan["training_seeds"][cohort][stream-1]
    updates=plan["updates"]
    worlds=plan[f"{arm}_geometries_per_update"]
    lanes=4*worlds
    path=train.ROOT/"training"/cohort/f"stream-{stream}"/arm
    prefix=(updates,len(births),2,lanes)
    branches=() if arm=="online" else (2,)
    with np.load(path/"paid-training-ledger.npz",allow_pickle=False) as raw:
        events=torch.from_numpy(raw["physical_events"])
        directions=torch.from_numpy(raw["physical_action"])
        prob=torch.from_numpy(raw["route_probability"])
        logit=torch.from_numpy(raw["route_logit"])
        earned=torch.from_numpy(raw["observed_or_weighted_resource"])
        prior=torch.from_numpy(raw["baseline_before"])
        assert int(raw["step"])==updates and int(raw["seed"])==seed
        assert str(raw["plan_sha256"])==env.digest(train.PLAN)
    assert events.shape==prefix+branches+(train.LENGTH+1,2)
    assert directions.shape==prefix+branches
    assert (events[:,:,0,...,0,:].abs()==1).all()
    assert torch.count_nonzero(events[:,:,0,...,1:,:])==0
    assert (events[:,:,1].abs()==1).all()
    torch.testing.assert_close(logit.sigmoid(),prob)
    assert torch.count_nonzero(prior[0])==0
    total=events[...,1].sum(-1).float()
    if arm=="online":
        draws=torch.stack([train.decision_draw(seed,births,step,worlds).reshape(len(births),lanes,2).transpose(1,2)
                           for step in range(1,updates+1)])
        assert torch.equal(directions>0,draws<prob)
        torch.testing.assert_close(total.mean(-1),earned,rtol=0,atol=0)
        next_baseline=torch.stack([x.clone().lerp_(y,plan["online_baseline_rate"])
                                   for x,y in zip(prior[:-1],earned[:-1],strict=True)])
        torch.testing.assert_close(next_baseline,prior[1:],rtol=0,atol=0)
    else:
        assert (directions[...,0]==-1).all() and (directions[...,1]==1).all()
        assert torch.count_nonzero(prior)==0
        want=((1-prob)*total[...,0]+prob*total[...,1]).mean(-1)
        torch.testing.assert_close(want,earned,rtol=0,atol=0)
    explicitly=(1,updates//2,updates)
    for k in explicitly:
        tape,missing=env.lesson.prepare(seed,births,k,worlds,2,omissions=True)
        future=env.opportunities(seed,births,k,worlds,train.LENGTH)
        coeff,state,difference=train.prepare(graph,rule,tape,missing)
        for i,horizon in enumerate((0,train.LENGTH)):
            with torch.no_grad():
                sense=state.hidden.new_full((tape.lanes,),2*i-1)
                origin,_=train.preview.route_basis(coeff,state,tape,sense,difference)
                if arm=="online":
                    z=directions[k-1,:,i].flatten().float()
                    replay,_=train.anchor.execute(coeff,origin.state,tape,future,z,horizon)
                else:
                    replay,_=env.branches(graph,rule,origin.state,tape,future,horizon)
            reference=events[k-1,:,i]
            torch.testing.assert_close(replay.reshape_as(reference).to(torch.int8),reference,rtol=0,atol=0)
    with np.load(path/"route-residual.npz",allow_pickle=False) as head:
        assert int(head["step"])==updates and head["parameters"].shape==(len(births),train.WIDTH)
        assert str(head["paid_ledger_sha256"])==env.digest(path/"paid-training-ledger.npz")
    physical_paths=int((events[:,0,...,0,1]!=0).sum())
    payouts=int(torch.count_nonzero(events[:,0,...,1]))
    assert physical_paths==32768 and payouts==98304
    files={p.relative_to(env.world.ROOT).as_posix():env.digest(p)
           for p in (path/"run.json",path/"resume.pt",path/"paid-training-ledger.npz",path/"route-residual.npz")}
    return dict(cohort=cohort,stream=stream,arm=arm,births=list(births),seed=seed,
                paid_paths_per_birth=physical_paths,paid_choices_per_birth=payouts,
                sampled_steps_replayed=explicitly,full_directions_and_prior_verified=True,files=files)


def run():
    plan,parents=train.settings()
    checks=[]
    for cohort in env.COHORTS:
        graph,rule=train.inherit(cohort,parents)
        for stream in (1,2):
            for arm in train.ARMS:
                checks.append(ledger(cohort,stream,arm,plan,graph,rule))
                print(cohort,stream,arm,flush=True)
    env.storage._write_json(env.EVIDENCE/"horizon-online-ledger.json",
                            dict(plan_sha256=env.digest(train.PLAN),source=env.digest(Path(__file__)),checks=checks))


if __name__=="__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run()
