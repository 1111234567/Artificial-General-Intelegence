"""Native action evaluation plus a read-only remaining-time sidecar."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys

import numpy as np
import torch

from experiments.action_split import evaluate as base
from experiments.action_split.data import sha,read,write_new
from experiments.remaining_time.model import RemainingTimeGRUAgent

ROOT=base.ROOT
OUT=ROOT/"data/remaining-time/eval-v1"
POLICIES=("original","probe","time","action","joint")
EPS=1e-6


def checkpoint(seed,policy):
    if policy=="original":
        return ROOT/f"data/finish-position/train-v1/seed_{seed:03d}/midpoint/B008.pt"
    from experiments.remaining_time.train import checkpoint_path
    return checkpoint_path(seed,policy,256)


def config(seed,condition):
    spec=base.SEEDS[seed]
    return dict(seed=853000+seed,batch_size=32,n_jobs=2,active_jobs=2,
        actor_ids=tuple(range(98000,98032)),family_allowed=(spec["history_cue"],spec["B_cue"]),
        dt=.25,duration_range=base.CONDITIONS[condition],gap_range=(16.,32.))


class RecordedPolicy(RemainingTimeGRUAgent):
    recorded_time=[]
    recorded_logits=[]

    def step(self,obs,state,dt=None,max_step=None):
        logits,h=super().step(obs,state,dt,max_step)
        # This result is archived AFTER native logits exist and never modifies
        # either logits, h, or the actual argmax action.
        self.recorded_time.append(self.time_head(h).cpu().numpy().copy())
        self.recorded_logits.append(logits.cpu().numpy().copy())
        return logits,h


def bind():
    base.OUT=OUT
    base.POLICIES=POLICIES
    base.checkpoint=checkpoint
    base.config=config
    base.SplitGRUAgent=RecordedPolicy


def declare():
    parent=read(ROOT/"data/action-split/eval-v1/protocol.json")
    sources=dict(parent["sources"])
    for path in (Path(__file__),Path(__file__).with_name("eval_test.py"),Path(__file__).with_name("model.py")):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    parent.update(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        policies=POLICIES,actor_ids=list(range(98000,98032)),world_seed="853000+seed",
        primary="Fixedstep256; joint versus action isolates extra core-reaching time gradient, time versus probe tests time-only core update. Native wholewindow behavior remainsprimary.",
        secondary="Duration16..32 only, gap16..32 fixed. Read-only time MAE/RMSE/bias physicalunits, by distance and phase; paired consecutive boundary sign and conditional content diagnostic only.",
        reference="All4 models begin with identical originalcore+native11head. Auxiliary2timevalues never feed nativeactions. probe must remain exactly original policy.",
        sidecar="*_time.npz saves time_prediction [4608,32,2] normalized by16 and native_logits [4608,32,2,11], computedfrom same post-observationh. *_time.json binds exact original rawfacts/checkpoint and fixed zero-sign metrics.",
        limits="Own trajectories can change valid-row exposure and be overdue beyond earliestteachernegativecoverage(-.25,0]. Common teacher histories are a separate diagnostic. No learned threshold or deployment clock.")
    write_new(OUT/"protocol.json",parent)


def freeze():
    from experiments.remaining_time import train
    training=train.OUT/"globalfreeze.json"
    assert training.exists()
    protocol=read(OUT/"protocol.json")
    for p,digest in protocol["sources"].items():assert sha(ROOT/p)==digest,p
    write_new(OUT/"freeze.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/"protocol.json"),training_freeze_sha256=sha(training),
        checkpoints={str(checkpoint(s,p).relative_to(ROOT)):sha(checkpoint(s,p)) for s in range(3) for p in POLICIES}))


def time_statistics(facts,prediction,logits,request_report,warm=512):
    assert prediction.shape==facts["active"].shape
    assert logits.shape==(*prediction.shape,11)
    assert np.isfinite(prediction).all() and np.isfinite(logits).all()
    valid=facts["active"] & facts["started"] & np.isfinite(facts["ready"])
    valid[:warm]=False
    truth=np.zeros_like(facts["ready"],dtype=np.float64)
    np.subtract(facts["ready"],facts["now"][:,None,None],out=truth,where=valid)
    physical=prediction.astype(np.float64)*16
    error=physical-truth
    target=4*facts["version"]+facts["payload"]
    correct=logits[...,3:11].argmax(-1)==target
    groups={"all_valid":valid,"processing":valid & (truth>EPS),
        "legal":valid & (truth<=EPS) & (facts["now"][:,None,None]<=facts["deadline"]+EPS),
        "remaining_over4":valid & (truth>4),"remaining_1to4":valid & (truth>1) & (truth<=4),
        "remaining_0to1":valid & (truth>EPS) & (truth<=1),
        "near_boundary":valid & (np.abs(truth)<=.25+EPS),
        "overdue_beyond_training":valid & (truth<-.25-EPS)}
    stats={}
    for name,mask in groups.items():
        n=int(mask.sum());e=error[mask]
        stats[name]=dict(rows=n,absolute_error_sum=float(np.abs(e).sum()),squared_error_sum=float(np.square(e).sum()),
            signed_error_sum=float(e.sum()),mae_physical=float(np.abs(e).mean()) if n else None,
            rmse_physical=float(np.sqrt(np.square(e).mean())) if n else None,
            bias_physical=float(e.mean()) if n else None,mae_ticks=float(np.abs(e).mean()/.25) if n else None,
            conditional_content_correct=int(correct[mask].sum()),
            predicted_ready=int((physical[mask]<=0).sum()),actual_ready=int((truth[mask]<=EPS).sum()),
            sign_correct=int(((physical[mask]<=0)==(truth[mask]<=EPS)).sum()))
    pairs=[]
    for r in request_report["requests"]:
        lane,job,rid=r["lane"],r["job"],r["rid"]
        match=(facts["rid"][:,lane,job]==rid)&valid[:,lane,job]
        before=np.flatnonzero(match&(truth[:,lane,job]>EPS))
        after=np.flatnonzero(match&(truth[:,lane,job]<=EPS)&(facts["now"]<=facts["deadline"][:,lane,job]+EPS))
        if not len(before) or not len(after) or before[-1]+1!=after[0]:continue
        b,a=int(before[-1]),int(after[0])
        pv=physical[[b,a],lane,job]
        pairs.append(dict(actor=r["actor"],lane=lane,job=job,rid=rid,family=r["family"],conflict=r["conflict"],
            ticks=[b,a],predicted_physical=pv.tolist(),target_physical=truth[[b,a],lane,job].tolist(),
            both_sign_correct=bool(pv[0]>0 and pv[1]<=0),direction_correct=bool(pv[0]>pv[1]),
            direction_tie=bool(pv[0]==pv[1]),false_early=bool(pv[0]<=0),missed_first=bool(pv[1]>0),
            first_content_correct=bool(correct[a,lane,job])))
    pair_groups={}
    for name,subset in (("all",pairs),("A_conflict",[r for r in pairs if r["family"]=="A" and r["conflict"]]),
                        ("B_conflict",[r for r in pairs if r["family"]=="B" and r["conflict"]])):
        pair_groups[name]=dict(pairs=len(subset),**{k:sum(r[k] for r in subset) for k in
            ("both_sign_correct","direction_correct","direction_tie","false_early","missed_first","first_content_correct")})
    return dict(groups=stats,boundary=pair_groups,boundary_rows=pairs,
        mature_request_denominator=len(request_report["requests"]),paired_requests=len(pairs),
        note="Read-only outputs; paired/time-valid row subset never replaces full behavioral denominator.")


def run(seed,policy,condition):
    bind()
    RecordedPolicy.recorded_time=[];RecordedPolicy.recorded_logits=[]
    base.run(seed,policy,condition)
    if policy=="original":return
    path=base.stem(seed,policy,condition)
    with np.load(path.with_suffix(".npz")) as z:facts=dict(z)
    report=read(path.with_suffix(".json"))
    predictions=np.stack(RecordedPolicy.recorded_time)
    logits=np.stack(RecordedPolicy.recorded_logits)
    np.testing.assert_array_equal(logits.argmax(-1),facts["actions"])
    summary=time_statistics(facts,predictions,logits,report)
    side=path.with_name(path.name+"_time")
    with side.with_suffix(".npz").open("xb") as f:
        np.savez_compressed(f,time_prediction=predictions,native_logits=logits)
    summary.update(seed=seed,policy=policy,condition=condition,facts_path=str(path.with_suffix(".npz").relative_to(ROOT)),
        facts_sha256=sha(path.with_suffix(".npz")),checkpoint_path=report["checkpoint_path"],
        checkpoint_sha256=report["checkpoint_sha256"],sidecar_sha256=sha(side.with_suffix(".npz")),
        protocol_sha256=sha(OUT/"protocol.json"),freeze_sha256=sha(OUT/"freeze.json"))
    write_new(side.with_suffix(".json"),summary)


def launch(workers=6):
    bind();logs=OUT/"logs";logs.mkdir(parents=True,exist_ok=True)
    def job(key):
        s,p,c=key
        with (logs/f"{s}_{p}_{c}.log").open("x",encoding="utf8") as handle:
            done=subprocess.run([sys.executable,"-m","experiments.remaining_time.evaluate","run","--seed",str(s),"--policy",p,"--condition",c],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        return dict(seed=s,policy=p,condition=c,exit_code=done.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs=list(pool.map(job,[(s,p,c) for s in range(3) for p in POLICIES for c in base.CONDITIONS]))
    write_new(OUT/"workers.json",dict(runs=runs))
    assert all(r["exit_code"]==0 for r in runs),runs
    pins={}
    for s in range(3):
        for c in base.CONDITIONS:
            reports=[read(base.stem(s,p,c).with_suffix(".json")) for p in POLICIES]
            assert len({r["external_hash"] for r in reports})==1
            ids=[{(r["actor"],r["job"],r["rid"]) for r in v["requests"]} for v in reports]
            assert all(x==ids[0] for x in ids)
            assert reports[0]["groups"]==reports[1]["groups"]
            assert reports[0]["idle_actions"]==reports[1]["idle_actions"]
            with np.load(base.stem(s,"original",c).with_suffix(".npz")) as a,np.load(base.stem(s,"probe",c).with_suffix(".npz")) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            for p in POLICIES:
                path=base.stem(s,p,c).with_suffix(".json");pins[str(path.relative_to(ROOT))]=sha(path)
                if p!="original":
                    side=path.with_name(path.stem+"_time.json");pins[str(side.relative_to(ROOT))]=sha(side)
    write_new(OUT/"completion.json",dict(complete=True,runs=30,sidecars=24,
        same_external_requests=True,probe_original_all_raw_arrays_exact=True,files=pins))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("declare","freeze","run","all"))
    p.add_argument("--seed",type=int,choices=range(3));p.add_argument("--policy",choices=POLICIES)
    p.add_argument("--condition",choices=base.CONDITIONS);p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="freeze":freeze()
    elif a.command=="run":run(a.seed,a.policy,a.condition)
    else:launch(a.workers)
