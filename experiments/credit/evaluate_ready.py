"""Run the frozen finite CPU evaluation list as complete training branches arrive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.credit.batch import digest, save


def run(plan_path, status_path, training_status, workers):
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    status=dict(plan=str(plan_path.resolve()), plan_digest=digest(plan), jobs={})
    if status_path.exists():
        status=json.loads(status_path.read_text(encoding="utf-8"))
        if status["plan_digest"] != digest(plan):
            raise ValueError("evaluation command manifest changed")
    for path,expected in plan["file_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"frozen input changed: {path}")
    pending=[job for job in plan["jobs"] if status["jobs"].get(job["id"],{}).get("returncode")!=0]
    for job in plan["jobs"]:
        if job not in pending and not all(Path(p).exists() for p in job["outputs"]):
            raise ValueError("previous successful evaluation missing")
    active={}
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    failed=False
    try:
        while pending or active:
            if training_status.exists():
                upstream=json.loads(training_status.read_text(encoding="utf-8"))
                failed |= any(item.get("returncode") not in (None,0) for item in upstream["jobs"].values())
            for job in tuple(pending):
                if len(active)>=workers or failed:
                    break
                source=Path(job["command"][job["command"].index("--checkpoint")+1])
                marker=source.parent/"launch-complete.json"
                if not marker.exists():
                    continue
                completion=json.loads(marker.read_text(encoding="utf-8"))
                if completion["stage_ticks"]!=completion["target"] or hashlib.sha256(source.read_bytes()).hexdigest()!=completion["checkpoint_sha256"]:
                    raise ValueError("training completion provenance mismatch")
                log_path=Path(job["log"])
                log_path.parent.mkdir(parents=True,exist_ok=True)
                log=log_path.open("ab")
                process=subprocess.Popen(job["command"],cwd=plan["cwd"],env=env,stdout=log,
                    stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                status["jobs"][job["id"]]=dict(started=time.time(),pid=process.pid,returncode=None,
                    command_digest=digest(job["command"]),checkpoint_sha256=completion["checkpoint_sha256"],log=str(log_path))
                active[job["id"]]=(job,process,log)
                pending.remove(job)
                print(json.dumps({"started":job["id"],"pid":process.pid}),flush=True)
                save(status_path,status)
            for identifier,(job,process,log) in tuple(active.items()):
                code=process.poll()
                if code is None:
                    continue
                log.close()
                if code==0 and not all(Path(p).exists() for p in job["outputs"]):
                    code=-1
                status["jobs"][identifier].update(finished=time.time(),returncode=code)
                failed |= code!=0
                del active[identifier]
                print(json.dumps({"finished":identifier,"returncode":code}),flush=True)
                save(status_path,status)
            if failed and not active:
                break
            if pending or active:
                time.sleep(2)
    finally:
        for job,process,log in active.values():
            process.terminate()
            process.wait()
            log.close()
            status["jobs"][job["id"]].update(finished=time.time(),returncode=process.returncode)
        status.update(finished=time.time(),complete=not failed and not pending and not active,
            coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save(status_path,status)
    return 0 if status["complete"] else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--status",type=Path,required=True)
    parser.add_argument("--training-status",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("workers must be positive")
    raise SystemExit(run(args.plan,args.status,args.training_status,args.workers))
