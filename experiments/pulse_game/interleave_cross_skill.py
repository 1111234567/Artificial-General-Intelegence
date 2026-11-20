"""Alternate two developmental tasks on one continuously writable nonlinear-memory body."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from . import interleave_joint_continue as joint, interleave_geometry_eval as geometry
from . import interleave_triangle_train as tri, interleave_confirm_triangle as confirmation
from . import interleave_checkpoint_publish as publish
from .interleave_interference import DATA, EVIDENCE


STRAND = 4000
SCHEDULES = ("old_then_new","new_then_old","alternating")
DEST = DATA/"cross-skill"
PLAN = EVIDENCE/"cross-skill-plan.json"


def is_old(schedule, step, seed):
    if schedule=="old_then_new":return step<=STRAND
    if schedule=="new_then_old":return step>STRAND
    if schedule!="alternating":raise ValueError(schedule)
    block,slot=divmod(step-1,2)
    return slot==np.random.default_rng([871373,seed,block]).integers(2)


def born(cohort,births):
    path,arm=geometry.sources(cohort,"hard",64000)
    parents=[tri.old_fit.load_checkpoint(path,"odor",arm,b,64000) for b in births]
    w,bias,present=(torch.cat([getattr(g,k) for g,r in parents]) for k in tri.revisit.capacity.GRAPH_FIELDS)
    g=tri.native.Graph(torch.nn.Parameter(w),torch.nn.Parameter(bias),present)
    r=tri.native.Rule(births)
    r.load_state_dict({key:torch.cat([rule.state_dict()[key] for _,rule in parents]) for key in r.state_dict()})
    return g,r


def target_step(g,r,opt,t,mode,cfg):
    params=(g.weights,g.bias,*r.parameters())
    for high_precision in (False,True):
        opt.zero_grad(set_to_none=True)
        if high_precision:
            model=tri.native.Graph(torch.nn.Parameter(g.weights.detach().double()),
                                   torch.nn.Parameter(g.bias.detach().double()),g.present)
            rule=deepcopy(r).double()
        else:model,rule=g,r
        loss,values=joint.training_step(model,rule,t,mode,cfg)
        if not loss.isfinite():continue
        loss.backward()
        origin=(model.weights,model.bias,*rule.parameters())
        norms=tri.revisit._gradients(origin,cfg.grad_clip)
        if norms is None:continue
        if high_precision:
            for p,q in zip(params,origin,strict=True):p.grad=None if q.grad is None else q.grad.to(p.dtype)
        opt.step()
        return torch.stack((values[:,1],values[:,2],values[:,3],norms,torch.full_like(norms,high_precision)),1)
    raise FloatingPointError("Nonfinite target gradient after same-world recovery")


def run(cohort,schedule):
    torch.set_num_threads(1)
    original=cohort=="original"
    births=list(range(731,737)) if original else list(range(811,817))
    base=json.loads((DATA/("joint-continuation" if original else "joint-continuation-replication")/"run.json").read_text())
    args=dict(base["config"]);args["updates"]=STRAND
    cfg=joint.train.Config(**args)
    geometry_seed=tri.TRAIN_SEED if original else confirmation.TRAIN_SEED
    folder=DEST/cohort/schedule
    folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(cohort=cohort,schedule=schedule,births=births,source_updates=STRAND,target_updates=STRAND,
                  config=asdict(cfg),geometry_seed=geometry_seed,plan_sha256=sha256(PLAN.read_bytes()).hexdigest(),
                  code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    source,tag=geometry.sources(cohort,"hard",64000)
    protocol["parents"]={str(b):sha256(tri.old_fit.checkpoint_path(source,"odor",tag,b,64000).read_bytes()).hexdigest()
                         for b in births}
    manifest=folder/"run.json"
    digest=publish.locked_plan(manifest,protocol,Path(__file__).stem,"code_sha256")
    g,r=born(cohort,births)
    opt=torch.optim.Adam((g.weights,g.bias,*r.parameters()),lr=cfg.lr)
    step=old_count=new_count=0;record=[]
    resume=folder/"resume.pt"
    if resume.is_file():
        saved=torch.load(resume,map_location="cpu",weights_only=True)
        if saved["plan_sha256"]!=digest:raise ValueError("Protocol mismatch for restart")
        with torch.no_grad():
            for k in tri.revisit.capacity.GRAPH_FIELDS:getattr(g,k).copy_(saved["graph"][k])
        r.load_state_dict(saved["rule"]);opt.load_state_dict(saved["optimizer"])
        step,old_count,new_count,record=saved["step"],saved["old"],saved["new"],saved["trace"].tolist()
    def save():
        saved=dict(arm=schedule,births=births,step=step,old=old_count,new=new_count,plan_sha256=digest,
                   graph={k:getattr(g,k).detach() for k in tri.revisit.capacity.GRAPH_FIELDS},
                   rule=r.state_dict(),optimizer=opt.state_dict(),trace=torch.as_tensor(record))
        if step in (0,2000,4000,6000,8000):
            for i,b in enumerate(births):tri.old_fit.save_checkpoint(DEST/cohort,"odor",schedule,b,step,g,r,i)
        tri.revisit._atomic(resume,lambda p:torch.save(saved,p))
    if not resume.exists() or publish.needs_export(DEST/cohort,"odor",schedule,births,step,(0,2000,4000,6000,8000)):save()
    initial=step;t0=perf_counter()
    for step in range(initial+1,2*STRAND+1):
        if is_old(schedule,step,cfg.world_seed):
            old_count+=1
            d=[tri.make_tape(geometry_seed,b,4,-.5,update=64000+old_count) for b in births]
            merged={k:torch.cat([t[k] for t in d]) for k in d[0]}
            values=tri._step(g,r,opt,merged,4)[:,[0,1,2,6,7]]
        else:
            new_count+=1
            ticket,mode=joint.world(births,cfg,new_count)
            values=target_step(g,r,opt,ticket,mode,cfg)
        record.append(values.tolist())
        if step%250==0:
            save()
            print(cohort,schedule,step,old_count,new_count,round((perf_counter()-t0)/(step-initial),3),flush=True)
    if old_count!=new_count or new_count!=STRAND:raise ValueError("The two experience budgets differ")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--cohort",choices=("original","independent"),required=True)
    p.add_argument("--schedule",choices=SCHEDULES,required=True)
    a=p.parse_args()
    run(a.cohort,a.schedule)
