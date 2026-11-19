"""Finite, resumable execution of an explicit experiment command manifest.

This is process orchestration only. Completed jobs are skipped only when the
same manifest and command digest previously exited successfully and all declared
outputs exist. It does not schedule future runs or modify model behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def execute(plan_path, status_path, workers):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_hash = digest(plan)
    if workers < 1 or len({job["id"] for job in plan["jobs"]}) != len(plan["jobs"]):
        raise ValueError("positive workers and unique job IDs required")
    status = {"plan": str(plan_path.resolve()), "plan_digest": plan_hash, "jobs": {}}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["plan_digest"] != plan_hash:
            raise ValueError("execution manifest changed")
    for path, expected in plan.get("file_hashes", {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen source/input hash mismatch: {path}")
    pending = []
    for job in plan["jobs"]:
        previous = status["jobs"].get(job["id"], {})
        if previous.get("returncode") == 0:
            if not all(Path(path).exists() for path in job.get("outputs", [])):
                raise ValueError(f"successful job output missing: {job['id']}")
        else:
            pending.append(job)
    active = {}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    failed = False
    try:
        while pending or active:
            while pending and len(active) < workers and not failed:
                job = pending.pop(0)
                log_path = Path(job["log"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("ab")
                process = subprocess.Popen(job["command"], cwd=plan["cwd"], env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                status["jobs"][job["id"]] = dict(started=time.time(), pid=process.pid,
                    command_digest=digest(job["command"]), log=str(log_path), returncode=None)
                active[job["id"]] = (job, process, log)
                print(json.dumps({"started": job["id"], "pid": process.pid}), flush=True)
                save(status_path, status)
            for identifier, (job, process, log) in tuple(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                missing = [path for path in job.get("outputs", []) if not Path(path).exists()]
                record = status["jobs"][identifier]
                record.update(finished=time.time(), returncode=code, missing_outputs=missing)
                if missing and code == 0:
                    record["returncode"] = -1
                failed |= record["returncode"] != 0
                del active[identifier]
                print(json.dumps({"finished": identifier, "returncode": record["returncode"]}), flush=True)
                save(status_path, status)
            if failed and not active:
                break
            if active:
                time.sleep(1)
    finally:
        # Interrupting this finite coordinator never leaves untracked children.
        for job, process, log in active.values():
            process.terminate()
            process.wait()
            log.close()
            status["jobs"][job["id"]].update(finished=time.time(), returncode=process.returncode)
        status["finished"] = time.time()
        status["complete"] = not failed and not pending and not active
        save(status_path, status)
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(execute(args.plan, args.status, args.workers))
