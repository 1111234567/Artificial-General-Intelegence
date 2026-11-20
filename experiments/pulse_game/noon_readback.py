"""A compact reward-fitted motor readout of continuous, bidirectional candidate previews."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from . import noon_paired_lesson as lesson, noon_review_world as world
from . import local_trace_graph as net,local_trace_train as old,revisit_choice_graph as motor


DATA=world.DATA/"readback"
PLAN=world.EVIDENCE/"readback-plan.json"
SPLITS={"train":(1628301,768),"test-original":(1628357,384),"test-independent":(1628369,384)}


def tape(seed,worlds,count):
    t=world.make_tape(seed,worlds,count)
    t=lesson.expand(t,torch.arange(0,t.lanes,2).repeat_interleave(8))
    row=torch.arange(t.lanes)
    bits=torch.tensor([[-1.,-1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[-1.,1.,1.],
                       [1.,-1.,-1.],[1.,-1.,1.],[1.,1.,-1.],[1.,1.,1.]]).repeat(worlds,1)
    law=t.law*bits[:,2,None]
    law[row,t.candidates[:,0]]=bits[:,0]
    law[row,t.candidates[:,1]]=bits[:,1]
    rng=np.random.default_rng([947191,seed,count])
    first=torch.from_numpy(np.repeat(rng.integers(2,size=worlds),8))
    assigned=torch.from_numpy(np.repeat(rng.integers(2,size=worlds),8))
    return replace(t,law=law),first,assigned


def scan(view,state,t,first):
    row=torch.arange(t.lanes)
    observations=torch.empty(t.lanes,2,2,dtype=state.hidden.dtype)
    for j in range(2):
        slot=first if j==0 else 1-first
        item=t.candidates[row,slot]
        sign=t.menu_orientation[row,slot]
        for k in range(2):
            left,right=world.sensory_for(t,item,sign if k==0 else -sign)
            state=net._advance(view,state,net.native.sensory(state.graph,left,right),net.REACTION_TICKS)
            observations[row,slot,k]=2.5*(state.hidden[:,11]-state.hidden[:,10])
    return state,observations


@torch.no_grad()
def trials(g,r,t,first,assigned,check=False):
    view=r.expand(t.lanes);s,contacts=world.introduce(view,net.start(g,t.lanes),t)
    preview,z=scan(view,s,t,first)
    act=world.select(view,preview,*world.candidate_menu(t),t.decision_draw)
    u=t.decision_draw;side=torch.stack((-torch.ones_like(u),torch.ones_like(u)),1)
    raw=lesson.branches(g,r,t,act,side)
    twin=raw["tape"];end=raw["state"];rows=torch.arange(twin.lanes)
    terminal=[];expected=[]
    for i in range(t.count):
        sign=twin.final_orientation[:,i];left=twin.odor[:,i]*sign[:,None]
        a=motor.act(end,left,-left,twin.final_draw[:,i])
        goal=twin.law[:,i]*sign
        terminal.append(a.side==goal)
        expected.append((a.motor_logit*goal).sigmoid())
    correct=torch.stack(terminal,1).reshape(t.lanes,2,t.count)
    target=t.final_item
    paid=correct[torch.arange(t.lanes)[:,None],torch.arange(2)[None],target[:,None]]
    if not torch.equal(paid,raw["reward"]>0):raise ValueError("Terminal payout disagreed with the visited item")
    if check:
        chosen=torch.where(assigned>0,torch.ones_like(u),-torch.ones_like(u))
        # Independent ordinary execution of the logged treatment, without branch expansion.
        single,ev=world.review(view,act.state,t,chosen)
        a,goal=world.final_probe(view,single,t,t.final_item)
        np.testing.assert_array_equal((a.side==goal).numpy(),paid[torch.arange(t.lanes),assigned].numpy())
        np.testing.assert_array_equal(ev.earned.numpy(),raw["taste"][torch.arange(t.lanes),assigned].numpy())
    return dict(z=z.numpy(),menu=act.motor_logit.numpy(),correct=correct.numpy(),
                future_expected=torch.stack(expected,1).reshape(t.lanes,2,t.count).numpy(),
                paid=paid.numpy(),chosen=assigned.numpy(),u=u.numpy(),query=target.numpy(),
                candidates=t.candidates.numpy(),law=t.law.numpy(),offer_orientation=t.menu_orientation.numpy(),
                study=torch.stack([torch.stack((c.motor.side,c.earned),1) for c in contacts],1).numpy().astype(np.int8),
                review_taste=raw["taste"].numpy().astype(np.int8))


def bank(split,family,count):
    torch.set_num_threads(1)
    seed,n=SPLITS[split];cohort="independent" if split=="test-independent" else "original"
    ids=world.COHORTS[cohort]
    t,first,assigned=tape(seed,n,count)
    dest=DATA/split/family/str(count);dest.mkdir(parents=True,exist_ok=True)
    source=sha256(Path(__file__).read_bytes()).hexdigest()
    manifest=dict(seed=seed,worlds=n,count=count,births=list(ids),family=family,source=source,
                  plan=sha256(PLAN.read_bytes()).hexdigest(),hashes={})
    for b in ids:
        if family=="uniform":
            path,arm,step=world.parent_path(cohort),"uniform",8000
        else:
            path=world.PARENT_DATA/"one-life-repair"/cohort;arm="mixed";step=2000
        checkpoint=old.checkpoint_path(path,"odor",arm,b,step)
        identity=sha256(checkpoint.read_bytes()).hexdigest()
        file=dest/f"{b}.npz"
        if file.exists():
            with np.load(file) as z:
                if str(z["checkpoint"])!=identity or str(z["source"])!=source or int(z["seed"])!=seed:
                    raise ValueError("Different readout data at the same cache path")
        else:
            body,rule=old.load_checkpoint(path,"odor",arm,b,step)
            parts=[]
            for j in range(0,n,24):
                rows=torch.arange(8*j,min(8*(j+24),t.lanes))
                parts.append(trials(body,rule,lesson.expand(t,rows),first[rows],assigned[rows],j==0))
            arrays={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
            lesson.storage._write_raw(file,**arrays,checkpoint=identity,source=source,seed=seed)
        manifest["hashes"][str(b)]=dict(policy=identity,trace=sha256(file.read_bytes()).hexdigest())
        print(split,family,count,b,flush=True)
    lesson.storage._write_json(dest/"index.json",manifest)


def jobs(split):
    if split=="train":
        bank("train","uniform",4)
    else:
        for family in ("uniform","repair"):
            for count in (4,2,8):bank(split,family,count)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--split",choices=SPLITS,required=True)
    jobs(p.parse_args().split)
