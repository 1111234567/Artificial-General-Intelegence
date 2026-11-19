"""Fixed fresh-world comparison of learned coverage and withdrawn old rules."""
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import subprocess
import sys

from experiments.remaining_time import evaluate as previous
from experiments.action_split.data import sha,read,write_new

ROOT=previous.ROOT
OUT=ROOT/"data/coverage-retention/eval-v1"
POLICIES=("baseline","control256","wide256","withdraw256")
CONDITIONS={"long":(8.,16.),"longer":(16.,32.),"far":(32.,64.)}


def checkpoint(seed,policy):
    from experiments.coverage_retention import train
    if policy=="baseline":return train.source_checkpoint(seed)
    return train.checkpoint_path(seed,policy[:-3],int(policy[-3:]))


def config(seed,condition):
    spec=previous.base.SEEDS[seed]
    return dict(seed=855000+seed,batch_size=32,n_jobs=2,active_jobs=2,
        actor_ids=tuple(range(100000,100032)),family_allowed=(spec["history_cue"],spec["B_cue"]),
        dt=.25,duration_range=CONDITIONS[condition],gap_range=(16.,32.))


def bind():
    previous.OUT=OUT;previous.POLICIES=POLICIES
    previous.checkpoint=checkpoint;previous.config=config
    previous.base.CONDITIONS=CONDITIONS
    previous.bind()


def declare():
    parent=read(ROOT/"data/teacher-timing/eval-v1/protocol.json")
    sources=dict(parent["sources"])
    for path in (Path(__file__),Path(__file__).with_name("eval_test.py"),
                 Path(__file__).with_name("eval_audit.py"),Path(__file__).with_name("analysis-plan.md")):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    parent.update(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        policies=POLICIES,conditions=CONDITIONS,world_seed="855000+seed",actor_ids=list(range(100000,100032)),
        primary="Fixed localstep256. Coverage: wide versus control. Withdrawal: withdraw versus control and source baseline, A/B conflict separately. 36freshworld rollouts,512ownwarm+4096test.",
        secondary="8..16 oldcoverage,16..32 learnedcoverage for wide ONLY,32..64 unseen by everyarm. No snapshotselection; step064archivednot evaluated.",
        reference="Baseline is teacher-timing midpoint/action step256 with trained diagnostic timehead. Everypolicy retains native11argmax, includingbaseline; everypolicy gets read-only time sidecar.",
        initialization="Independent freshworld origin h0 for standardization, thenfullycontinuous. Separate live-v1 experiment restores complete trainingworld+h without reset.",
        sidecar="All36 *_time.npz contain time_prediction/nativelogits, unused bypolicy. Valid-row exposure can differ, and largerduration extends beyond prior time-targetcoverage.",
        interaction="Threebranches only; no combined wide+B-only condition or replay branch. Same newly issued holdoutcases within eachseed/duration acrossallpolicies.",
        limits="All teachers remain midpoint. Extendedduration is learnedcoverage for wide, not extrapolation. Validrows differ between ownhistories. Legacy metricbucket overdue_beyond_training means only remaining<-.25; midpointtraining already covers part of it.")
    write_new(OUT/"protocol.json",parent)


def freeze():
    from experiments.coverage_retention import train
    training=train.OUT/"globalfreeze.json"
    assert training.exists()
    p=read(OUT/"protocol.json")
    for name,digest in p["sources"].items():assert sha(ROOT/name)==digest,name
    write_new(OUT/"freeze.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/"protocol.json"),training_freeze_sha256=sha(training),
        checkpoints={str(checkpoint(s,p).relative_to(ROOT)):sha(checkpoint(s,p)) for s in range(3) for p in POLICIES}))


def run(seed,policy,condition):
    bind();previous.run(seed,policy,condition)


def launch(workers=8):
    bind();logs=OUT/"logs";logs.mkdir(parents=True,exist_ok=True)
    def job(key):
        seed,policy,condition=key
        with (logs/f"{seed}_{policy}_{condition}.log").open("x",encoding="utf8") as log:
            p=subprocess.run([sys.executable,"-m","experiments.coverage_retention.evaluate","run","--seed",str(seed),"--policy",policy,"--condition",condition],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        return dict(seed=seed,policy=policy,condition=condition,exit_code=p.returncode)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs=list(pool.map(job,[(s,p,c) for s in range(3) for p in POLICIES for c in CONDITIONS]))
    write_new(OUT/"workers.json",dict(runs=runs));assert all(r["exit_code"]==0 for r in runs),runs
    files={};sidecars={}
    for seed in range(3):
        for condition in CONDITIONS:
            rows=[]
            for policy in POLICIES:
                path=previous.base.stem(seed,policy,condition).with_suffix(".json")
                rows.append(read(path));files[str(path.relative_to(ROOT))]=sha(path)
                side=path.with_name(path.stem+"_time.json");sidecars[str(side.relative_to(ROOT))]=sha(side)
            assert len({r["external_hash"] for r in rows})==1
            ids=[{(r["actor"],r["job"],r["rid"]) for r in x["requests"]} for x in rows]
            assert all(x==ids[0] for x in ids)
    write_new(OUT/"completion.json",dict(complete=True,runs=36,sidecars=36,mode="fresh",same_external_requests=True,files=files,time_files=sidecars))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("declare","freeze","run","all"))
    p.add_argument("--seed",type=int,choices=range(3));p.add_argument("--policy",choices=POLICIES)
    p.add_argument("--condition",choices=CONDITIONS);p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="freeze":freeze()
    elif a.command=="run":run(a.seed,a.policy,a.condition)
    else:launch(a.workers)
