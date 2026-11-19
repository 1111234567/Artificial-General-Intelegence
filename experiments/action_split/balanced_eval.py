"""Fresh-case evaluation for the single post-failure mode-normalization control."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path

from experiments.action_split import evaluate as base
from experiments.action_split.data import sha,read,write_new

ROOT=base.ROOT
OUT=ROOT/"data/action-split/balanced-eval-v1"
POLICIES=("original","base_frozen","base_full","balanced_frozen","balanced_full")


def checkpoint(seed,policy):
    if policy=="original":
        return ROOT/f"data/finish-position/train-v1/seed_{seed:03d}/midpoint/B008.pt"
    arm=policy.split("_")[-1]
    folder="balanced-v1" if policy.startswith("balanced_") else "train-v1"
    return ROOT/f"data/action-split/{folder}/seed_{seed:03d}/{arm}/step256.pt"


def config(seed,condition):
    spec=base.SEEDS[seed]
    return dict(seed=852000+seed,batch_size=32,n_jobs=2,active_jobs=2,
        actor_ids=tuple(range(97000,97032)),family_allowed=(spec["history_cue"],spec["B_cue"]),
        dt=.25,duration_range=base.CONDITIONS[condition],gap_range=(16.,32.))


def declare():
    parent=read(ROOT/"data/action-split/eval-v1/protocol.json")
    sources=dict(parent["sources"])
    sources[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    parent.update(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,policies=POLICIES,
        actor_ids=list(range(97000,97032)),world_seed="852000+seed",
        primary="Final step256, compare originalmodeobjective and equal-class-modeobjective frozen/full, on the SAME FRESH external requests for all5policies.",
        secondary="Duration16..32 only, unchangedgap16..32; no model/threshold selection.",
        posthoc="Chosen after the initial full model achieved zero completion and zero teacher-history FINISH recall. One fixed equal-class normalization, not a sweep. All previous failed outcomes retained. New evaluation actors/worldseed were not in those results.",
        original_evaluation_sha256=sha(ROOT/"data/action-split/eval-v1/completion.json"),
        diagnostic_sha256=sha(ROOT/"data/action-split/diagnostic-v1/summary.json"))
    write_new(OUT/"protocol.json",parent)


def freeze():
    protocol=read(OUT/"protocol.json")
    for path,digest in protocol["sources"].items():assert sha(ROOT/path)==digest,path
    a=ROOT/"data/action-split/train-v1/globalfreeze.json"
    b=ROOT/"data/action-split/balanced-v1/globalfreeze.json"
    assert a.exists() and b.exists()
    write_new(OUT/"freeze.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/"protocol.json"),original_training_freeze_sha256=sha(a),balanced_training_freeze_sha256=sha(b),
        checkpoints={str(checkpoint(s,p).relative_to(ROOT)):sha(checkpoint(s,p)) for s in range(3) for p in POLICIES}))


def bind():
    base.OUT=OUT
    base.POLICIES=POLICIES
    base.checkpoint=checkpoint
    base.config=config


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","freeze","run","all"))
    parser.add_argument("--seed",type=int,choices=range(3))
    parser.add_argument("--policy",choices=POLICIES)
    parser.add_argument("--condition",choices=base.CONDITIONS)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    if args.command=="declare":declare()
    elif args.command=="freeze":freeze()
    elif args.command=="run":
        bind();base.run(args.seed,args.policy,args.condition)
    else:
        # Parent launcher hardcodes its module name. Launch this write-isolated
        # wrapper explicitly while reusing the unchanged native scorer.
        import subprocess,sys,os
        from concurrent.futures import ThreadPoolExecutor
        bind(); logs=OUT/"logs";logs.mkdir(parents=True,exist_ok=True)
        def job(key):
            s,p,c=key;logfile=logs/f"{s}_{p}_{c}.log"
            with logfile.open("x",encoding="utf8") as stream:
                done=subprocess.run([sys.executable,"-m","experiments.action_split.balanced_eval","run","--seed",str(s),"--policy",p,"--condition",c],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            return dict(seed=s,policy=p,condition=c,exit_code=done.returncode)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(job,[(s,p,c) for s in range(3) for p in POLICIES for c in base.CONDITIONS]))
        write_new(OUT/"workers.json",dict(results=results))
        assert all(r["exit_code"]==0 for r in results),results
        pins={}
        for s in range(3):
            for c in base.CONDITIONS:
                rows=[read(base.stem(s,p,c).with_suffix(".json")) for p in POLICIES]
                assert len({r["external_hash"] for r in rows})==1
                ids=[{(r["actor"],r["job"],r["rid"]) for r in row["requests"]} for row in rows]
                assert all(x==ids[0] for x in ids)
                for p in POLICIES:
                    path=base.stem(s,p,c).with_suffix(".json");pins[str(path.relative_to(ROOT))]=sha(path)
        write_new(OUT/"completion.json",dict(complete=True,runs=len(results),fresh_cases=True,identical_external_requests=True,files=pins))
