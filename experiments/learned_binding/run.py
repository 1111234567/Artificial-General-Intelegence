"""Own-action acquisition, public-outcome PG, and prospective behavior probes."""
import os
for _var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(_var,"1")
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from experiments.learned_binding.common import *
from experiments.learned_binding.interaction import Subject,probe


def returns_and_advantages(rewards,gamma):
    # Both jobs share M.  An action can improve the other job's future reward,
    # so use the complete actor reward with the joint two-action likelihood.
    total=rewards.sum(-1)
    returns=torch.empty_like(total);running=torch.zeros_like(total[0])
    for tick in range(len(rewards)-1,-1,-1):
        running=total[tick]+gamma*running;returns[tick]=running
    if rewards.shape[1]<2:raise ValueError("leave-one-actor-out baseline needs two actors")
    baseline=(returns.sum(1,keepdim=True)-returns)/(rewards.shape[1]-1)
    return returns,returns-baseline


def pg_update(subject,block,optimizer):
    """Likelihood of actual sampled actions; labels/oracle never enter this loss."""
    model=subject.model;model.train()
    logits,end=model(block["obs"],block["start"]["state"])
    error=float((logits.detach()-block["logits"]).abs().max())
    if error>2e-4:raise AssertionError(f"step/sequence policy likelihood changed: {error}")
    returns,adv=returns_and_advantages(block["rewards"],SETTINGS["gamma"])
    logprob=logits.log_softmax(-1).gather(-1,block["actions"][...,None]).squeeze(-1)
    loss=-(logprob.sum(-1)*adv.detach()).mean()
    optimizer.zero_grad(set_to_none=True);loss.backward()
    grad=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
    assert all(p.grad is None for p in model.executor.parameters())
    before=parameter_hash(model.executor);optimizer.step()
    assert parameter_hash(model.executor)==before
    model.eval()
    # The recomputed differentiable state is NOT transplanted into the live stream.
    subject.work["parameter_updates"]=subject.work.get("parameter_updates",0)+1
    return dict(loss=float(loss.detach()),grad_norm=grad,step_sequence_logit_maxabs=error,
        reward_sum=float(block["rewards"].sum()),positive_finishes=int((block["rewards"]>0).sum()),
        rejected=int((block["rewards"]<0).sum()),return_mean=float(returns.mean()),
        advantage_mean=float(adv.mean()),native_executor_unchanged=True)


def save_block(path,block,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:v.numpy() for k,v in block.items() if isinstance(v,torch.Tensor)}
    np.savez_compressed(path.with_suffix(".npz"),**arrays)
    torch.save({k:v for k,v in block.items() if not isinstance(v,torch.Tensor)},path.with_suffix(".pt"))
    info=dict(arrays_sha256=sha(path.with_suffix(".npz")),states_sha256=sha(path.with_suffix(".pt")))
    if extra:info.update(extra)
    write_json(path.with_suffix(".json"),info)
    return info


def save_probe(subject,path,*,write_enabled=True,read_enabled=True):
    if subject.pending_obs is not None:raise AssertionError("close pending explicitly before probe")
    subject.world.set_family_policy(subject.contexts)
    result=probe(subject,SETTINGS["probe_ticks"],write_enabled=write_enabled,read_enabled=read_enabled)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"),**{k:np.asarray(v) for k,v in result["arrays"].items()})
    torch.save(dict(start=result["start"],final=result["final"]),path.with_suffix(".pt"))
    report=result["report"]
    report.update(name=path.name,write_enabled=write_enabled,read_enabled=read_enabled,
        arrays_sha256=sha(path.with_suffix(".npz")),states_sha256=sha(path.with_suffix(".pt")))
    write_json(path.with_suffix(".json"),report)
    print(json.dumps(dict(event="probe",seed=subject.seed,kind=subject.kind,name=path.name,
        groups=report["groups"],idle=[report["idle_actions"],report["idle_opportunities"]])),flush=True)
    return report


def b_phase(subject,folder,learn=False):
    assert subject.pending_obs is None
    subject.world.set_family_policy((subject.contexts[1],))
    optimizer=torch.optim.Adam([p for p in subject.model.parameters() if p.requires_grad],lr=SETTINGS["pg_lr"]) if learn else None
    records=[];initial=subject.snapshot()
    for block_index in range(SETTINGS["pg_blocks"]):
        block=subject.rollout_block(SETTINGS["pg_ticks"],sample=True)
        before=subject.snapshot()
        update=pg_update(subject,block,optimizer) if learn else dict(reward_sum=float(block["rewards"].sum()),
            positive_finishes=int((block["rewards"]>0).sum()),rejected=int((block["rewards"]<0).sum()))
        # Update weights while preserving actual h/z/M, world and the pending observation.
        after=subject.snapshot()
        for name in ("state","world","pending_obs","rng","binder","executor"):
            assert_tree_equal(before[name],after[name],name)
        block["after_update"]=subject.snapshot()
        if learn:block["optimizer_after"]=deepcopy(optimizer.state_dict())
        save_block(folder/f"block_{block_index:03d}",block,dict(index=block_index,learn=learn,update=update))
        records.append(update)
        if (block_index+1)%8==0:print(json.dumps(dict(event="B",seed=subject.seed,kind=subject.kind,learn=learn,block=block_index+1,**update)),flush=True)
    closure=subject.close_pending(sample=True)
    torch.save(dict(initial=initial,final=subject.snapshot(),closure=closure),folder/"boundary.pt")
    write_json(folder/"completion.json",dict(blocks=len(records),learn=learn,records=records,
        counted_extra_closure_ticks=1,phase_ticks=SETTINGS["pg_blocks"]*SETTINGS["pg_ticks"]+1))


def check_freeze():
    check_protocol()
    p=json.loads((OUT/"training-freeze.json").read_text(encoding="utf-8"))
    assert p["all_complete_before_scoring"]
    for name,h in p["checkpoints"].items():assert sha(ROOT/name)==h


def run_kind(seed,kind):
    torch.set_num_threads(1);check_freeze();timer=time.perf_counter()
    folder=OUT/f"seed_{seed:03d}"/"behavior"/kind
    folder.mkdir(parents=True,exist_ok=False)
    # Main experience is self-generated, including the initial AB support.
    subject=Subject(seed,kind,stream_id=100)
    support=subject.support(SETTINGS["support_ticks"],teacher=False,sample=True)
    save_block(folder/"own_support",support,dict(teacher=False,policy="sample_all_11_actions"))
    save_probe(deepcopy(subject),folder/"own_immediate")
    if kind=="learned":
        learner=deepcopy(subject)
        b_phase(learner,folder/"outcome_B",learn=True)
        save_probe(deepcopy(learner),folder/"own_after_B_outcome")
    b_phase(subject,folder/"frozen_B",learn=False)
    save_probe(deepcopy(subject),folder/"own_after_B_frozen")
    # A separate diagnostic asks whether identical demonstrated experiences
    # can be bound.  It is never pooled with the own-action main experiment.
    teacher=Subject(seed,kind,stream_id=100)
    shown=teacher.support(SETTINGS["support_ticks"],teacher=True,sample=False)
    save_block(folder/"teacher_support",shown,dict(teacher=True))
    save_probe(deepcopy(teacher),folder/"teacher_probe")
    if kind=="learned":
        cleared=deepcopy(teacher);cleared.state["memory"].zero_()
        save_probe(cleared,folder/"teacher_clear_memory")
        cleared=deepcopy(teacher);cleared.state["z"].zero_()
        save_probe(cleared,folder/"teacher_clear_encoder")
        save_probe(deepcopy(teacher),folder/"teacher_no_read",read_enabled=False)
    write_json(folder/"completion.json",dict(seed=seed,kind=kind,seconds=time.perf_counter()-timer,
        files={p.name:sha(p) for p in folder.glob("*.json")}))


def worker(job):
    seed,kind=job;path=OUT/"logs"/f"behavior_{seed}_{kind}.log";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f,redirect_stdout(f),redirect_stderr(f):run_kind(seed,kind)
    return dict(seed=seed,kind=kind)


def launch(workers=9):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done in as_completed([pool.submit(worker,(s,k)) for s in range(3) for k in KINDS]):
            print(json.dumps(dict(event="complete",**done.result())),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=("declare","all"));p.add_argument("--workers",type=int,default=9)
    a=p.parse_args()
    if a.command=="declare":declare()
    else:launch(a.workers)
