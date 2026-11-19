"""Resume the sealed seed search with balanced CPU and hybrid queues."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
import os
import subprocess
from time import monotonic, sleep
import traceback
from unittest.mock import patch

from experiments.seed_evolution import balanced_schedule, execution_runtime as runtime


runner = runtime.runner


def _init_worker(spec, schedule):
    runtime._init_worker(spec)
    if balanced_schedule.load() != schedule:
        raise ValueError("worker routing supplement differs from the scheduler")
    runtime.backend_for = schedule.backend
    runtime.load_amendment = lambda: spec


def _free_vram_mib():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(result.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _parallel(jobs, worker, workers, *, spec, schedule, cpu_workers, hybrid_workers,
              hybrid_start_free_mib=None):
    if not jobs:
        return
    role = {
        "_candidate_job": "candidate", "_train_final": "final", "_test_final": "test",
    }[worker.__name__]
    queues = {"cpu": deque(), "hybrid": deque()}
    for job in jobs:
        mode = runtime._job_mode(job, role, spec)
        lane = "hybrid" if mode == "hybrid" else "cpu"
        queues[lane].append((job, mode))

    search = role == "candidate"
    limits = {
        "cpu": cpu_workers if search else min(workers, 6),
        "hybrid": hybrid_workers if search else min(hybrid_workers, spec["gpu_workers"]),
    }
    pools = {}
    pending = {}

    def submit_next(lane):
        queue = queues[lane]
        if search and lane == "hybrid" and not queue:
            queue = queues["cpu"]
        if queue:
            job, mode = queue.popleft()
            pending[pools[lane].submit(runtime._worker, job, role, mode)] = lane

    def start_pool(lane):
        pools[lane] = ProcessPoolExecutor(
            max_workers=limits[lane], initializer=_init_worker, initargs=(spec, schedule))
        for _ in range(limits[lane]):
            submit_next(lane)

    next_vram_check = 0.0
    waiting = False
    try:
        if queues["cpu"]:
            start_pool("cpu")
        while pending or queues["hybrid"]:
            if queues["hybrid"] and "hybrid" not in pools and monotonic() >= next_vram_check:
                free = _free_vram_mib() if hybrid_start_free_mib is not None else None
                if hybrid_start_free_mib is None or (free is not None and free >= hybrid_start_free_mib):
                    if waiting:
                        print(f"Hybrid queue resuming with {free} MiB free VRAM", flush=True)
                    start_pool("hybrid")
                else:
                    if not waiting:
                        print(f"Hybrid queue waiting for {hybrid_start_free_mib} MiB free VRAM "
                              f"(available: {free} MiB)", flush=True)
                    waiting = True
                    next_vram_check = monotonic() + 10
            if not pending:
                sleep(max(0, next_vram_check - monotonic()))
                continue
            timeout = max(0, next_vram_check - monotonic()) if waiting and "hybrid" not in pools else None
            done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            for future in done:
                lane = pending.pop(future)
                result = future.result()
                submit_next(lane)
                yield result
    except BaseException:
        for pool in pools.values():
            if hasattr(pool, "terminate_workers"):
                pool.terminate_workers()
        raise
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)


def run(cpu_workers=10, hybrid_workers=3, hybrid_start_free_mib=None):
    from experiments.seed_evolution import balanced_audit

    if cpu_workers < 1 or hybrid_workers < 1:
        raise ValueError("worker counts must be positive")
    if hybrid_start_free_mib is not None and hybrid_start_free_mib < 1:
        raise ValueError("hybrid startup VRAM threshold must be positive")
    out = runtime.OUT
    with runner._lock(out / ".pipeline.lock"):
        runner._activate()
        spec = runtime.load_amendment()
        schedule = balanced_schedule.load()
        amendment_sha = runtime.sha(out / runtime.AMENDMENT)
        provenance = dict(execution_amendment_sha256=amendment_sha,
                          balanced_schedule_sha256=schedule.sha256)
        runner._write(out / "execution-process.json", dict(
            pid=os.getpid(), started_utc=runner._utc(), amendment_sha=amendment_sha,
            balanced_schedule_sha256=schedule.sha256,
            cpu_workers=cpu_workers, hybrid_workers=hybrid_workers,
            hybrid_start_free_mib=hybrid_start_free_mib,
        ))
        stage = "search"
        try:
            parallel = partial(_parallel, spec=spec, schedule=schedule,
                               cpu_workers=cpu_workers, hybrid_workers=hybrid_workers,
                               hybrid_start_free_mib=hybrid_start_free_mib)
            with (patch.object(runner, "_parallel", parallel),
                  patch.object(runner, "_committed_candidate", runtime._committed),
                  patch.object(runtime, "backend_for", schedule.backend),
                  patch.object(runtime, "load_amendment", lambda: spec)):
                runner.search(cpu_workers)
                stage = "finals"
                runner.finals(6)
                stage = "test"
                runner.tests(6)
            stage = "audit"
            runner._progress(stage, "running")
            result = balanced_audit.audit_all(out, workers=9)
            if result.get("full_audit_passed") is not True or any(
                    result.get(key) != value for key, value in provenance.items()):
                raise ValueError("balanced execution audit failed")
            runner._write(out / "audit.json", result)
            stage = "results"
            summary = balanced_audit.aggregate(out)
            runner._write(out / "pipeline-completion.json",
                          dict(full_audit_passed=True, result=summary, **provenance))
            budget = runner._unique_budget()
            runner._progress("complete", "complete", completed_candidates=budget,
                             total_candidates=budget)
        except BaseException as error:
            runner._write(out / "execution-failure.json", dict(
                stage=stage, error=repr(error), traceback=traceback.format_exc()))
            runner._progress(stage, "failed", error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-workers", type=int, default=10)
    parser.add_argument("--hybrid-workers", type=int, default=3)
    parser.add_argument("--hybrid-start-free-mib", type=int)
    args = parser.parse_args()
    run(args.cpu_workers, args.hybrid_workers, args.hybrid_start_free_mib)


if __name__ == "__main__":
    main()
