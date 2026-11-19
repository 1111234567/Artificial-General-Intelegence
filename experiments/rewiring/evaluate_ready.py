"""CPU queue adapter for the frozen two-phase rewiring completion format."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from experiments.credit.batch import digest
from experiments.credit.run import file_sha256
from experiments.credit.windows_runner import save_with_retry as save
from experiments.loops.audit_run import verify_files


def ready(job):
    command = job["command"]
    sentinel = Path(command[command.index("--checkpoint") + 1])
    marker = sentinel.parent / "launch-complete.json"
    if not marker.exists():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("format") != "rewiring-launch-complete-v1" or value.get("branch", {}).get("id") != job["id"]:
        raise ValueError("foreign completion marker")
    for phase in ("B", "A2"):
        entry = value["checkpoints"][phase]
        path = sentinel.parent / f"checkpoint_{phase}.pt"
        if Path(entry["path"]).resolve() != path.resolve() or file_sha256(path) != entry["sha256"]:
            raise ValueError("phase checkpoint provenance mismatch")
    return file_sha256(marker)


def run(plan_path, status_path, training_status, workers):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_files(plan["file_hashes"])
    if workers < 1:
        raise ValueError("workers must be positive")
    state = {"plan": str(plan_path.resolve()), "plan_digest": digest(plan), "jobs": {}}
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8"))
        if state["plan_digest"] != digest(plan):
            raise ValueError("evaluation manifest changed")
    state.pop("finished", None)
    state["complete"] = False
    state["workers"] = workers
    state["coordinator_sha256"] = file_sha256(__file__)
    pending = []
    for job in plan["jobs"]:
        prior = state["jobs"].get(job["id"], {})
        if prior.get("returncode") == 0:
            if not all(Path(p).exists() for p in job["outputs"]):
                raise ValueError("completed evaluation output missing")
        else:
            pending.append(job)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    active, failed = {}, False
    save(status_path, state)
    try:
        while pending or active:
            if training_status.exists():
                upstream = json.loads(training_status.read_text(encoding="utf-8"))
                failed |= any(r.get("returncode") not in (None, 0) for r in upstream["jobs"].values())
            for job in tuple(pending):
                if len(active) >= workers or failed:
                    break
                marker_hash = ready(job)
                if marker_hash is None:
                    continue
                log_path = Path(job["log"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("ab")
                process = subprocess.Popen(job["command"], cwd=plan["cwd"], env=env, stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                state["jobs"][job["id"]] = dict(started=time.time(), pid=process.pid, returncode=None,
                    command_digest=digest(job["command"]), completion_sha256=marker_hash, log=str(log_path))
                active[job["id"]] = job, process, log
                pending.remove(job)
                print(json.dumps({"started": job["id"], "pid": process.pid}), flush=True)
                save(status_path, state)
            for identifier, (job, process, log) in tuple(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                missing = [p for p in job["outputs"] if not Path(p).exists()]
                if code == 0 and missing: code = -1
                state["jobs"][identifier].update(finished=time.time(), returncode=code, missing_outputs=missing)
                failed |= code != 0
                del active[identifier]
                print(json.dumps({"finished": identifier, "returncode": code}), flush=True)
                save(status_path, state)
            if failed and not active: break
            if pending or active: time.sleep(1)
    finally:
        for job, process, log in active.values():
            process.terminate()
            process.wait()
            log.close()
            state["jobs"][job["id"]].update(finished=time.time(), returncode=process.returncode)
        state.update(finished=time.time(), complete=not failed and not pending and not active)
        save(status_path, state)
    return 0 if state["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "status", "training-status"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.status, args.training_status, args.workers))
