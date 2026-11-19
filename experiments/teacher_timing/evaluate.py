"""Fresh native free-action cases for actual earliest/midpoint teaching."""
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import subprocess
import sys
import numpy as np

from experiments.remaining_time import evaluate as previous
from experiments.action_split.data import sha,read,write_new

ROOT=previous.ROOT
OUT=ROOT/"data/teacher-timing/eval-v1"
POLICIES=("original","earliest_action","earliest_joint","midpoint_action","midpoint_joint")


def checkpoint(seed,policy):
    if policy=="original":
        return ROOT/f"data/finish-position/train-v1/seed_{seed:03d}/midpoint/B008.pt"
    from experiments.teacher_timing.train import checkpoint_path
    teacher,objective=policy.split("_")
    return checkpoint_path(seed,teacher,objective,256)


def config(seed,condition):
    spec=previous.base.SEEDS[seed]
    return dict(seed=854000+seed,batch_size=32,n_jobs=2,active_jobs=2,
        actor_ids=tuple(range(99000,99032)),family_allowed=(spec["history_cue"],spec["B_cue"]),
        dt=.25,duration_range=previous.base.CONDITIONS[condition],gap_range=(16.,32.))


def bind():
    previous.OUT=OUT
    previous.POLICIES=POLICIES
    previous.checkpoint=checkpoint
    previous.config=config
    previous.bind()


def declare():
    parent=read(ROOT/"data/remaining-time/eval-v1/protocol.json")
    sources=dict(parent["sources"])
    for path in (Path(__file__),ROOT/"experiments/teacher_timing/eval_audit.py",
                 ROOT/"experiments/action_split/eval_audit.py",ROOT/"experiments/remaining_time/eval_audit.py"):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    parent.update(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        policies=POLICIES,world_seed="854000+seed",actor_ids=list(range(99000,99032)),
        primary="Compare midpoint versus earliest actual teacher regime under action-only native CE; both copied original policy and have exact same fixed256update/input/action-row budgets.",
        interaction="Joint-minus-action changes within each teacher, then compare teacher effects. Time-head loss never influences policy through action-only detached features/separateclipping. Time-valid row counts differ with actual teacher execution.",
        secondary="Duration16..32 with samegap16..32, fixedstep256 only. Frozen read-only time sidecars; no learned threshold or rollout selection.",
        reference="Original native policy is untouched, no extra training. Four copied-policy continuations start identical; earliest endpoints reproduce previous experiment bitwise.",
        limits="Actual teacher changes labels AND receipts/state histories, not a label-only intervention. Wholewindow success is primary; auxiliarytime signed/MAE metrics are only diagnostics.")
    write_new(OUT/"protocol.json",parent)


def freeze():
    from experiments.teacher_timing import train
    training=train.OUT/"globalfreeze.json"
    assert training.exists()
    protocol=read(OUT/"protocol.json")
    for path,digest in protocol["sources"].items():assert sha(ROOT/path)==digest,path
    write_new(OUT/"freeze.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/"protocol.json"),training_freeze_sha256=sha(training),
        checkpoints={str(checkpoint(s,p).relative_to(ROOT)):sha(checkpoint(s,p)) for s in range(3) for p in POLICIES}))


def run(seed,policy,condition):
    bind();previous.run(seed,policy,condition)


def launch(workers=6):
    bind();logs=OUT/"logs";logs.mkdir(parents=True,exist_ok=True)
    def job(key):
        s,p,c=key
        with (logs/f"{s}_{p}_{c}.log").open("x",encoding="utf8") as log:
            done=subprocess.run([sys.executable,"-m","experiments.teacher_timing.evaluate","run","--seed",str(s),"--policy",p,"--condition",c],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        return dict(seed=s,policy=p,condition=c,exit_code=done.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs=list(pool.map(job,[(s,p,c) for s in range(3) for p in POLICIES for c in previous.base.CONDITIONS]))
    write_new(OUT/"workers.json",dict(runs=runs))
    assert all(r["exit_code"]==0 for r in runs),runs
    main_files={};sidecars={}
    for seed in range(3):
        for condition in previous.base.CONDITIONS:
            records=[]
            for policy in POLICIES:
                path=previous.base.stem(seed,policy,condition).with_suffix(".json")
                records.append(read(path));main_files[str(path.relative_to(ROOT))]=sha(path)
                if policy!="original":
                    side=path.with_name(path.stem+"_time.json");sidecars[str(side.relative_to(ROOT))]=sha(side)
            assert len({r["external_hash"] for r in records})==1
            ids=[{(r["actor"],r["job"],r["rid"]) for r in q["requests"]} for q in records]
            assert all(x==ids[0] for x in ids)
    write_new(OUT/"completion.json",dict(complete=True,runs=30,sidecars=24,
        same_external_requests=True,files=main_files,time_files=sidecars))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("declare","freeze","run","all"))
    p.add_argument("--seed",type=int,choices=range(3));p.add_argument("--policy",choices=POLICIES)
    p.add_argument("--condition",choices=previous.base.CONDITIONS);p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="freeze":freeze()
    elif a.command=="run":run(a.seed,a.policy,a.condition)
    else:launch(a.workers)
