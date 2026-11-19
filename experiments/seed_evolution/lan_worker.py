"""Execute an assigned CPU shard of a sealed plan in a private repository copy."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import re
from typing import NamedTuple

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.seed_evolution import balanced_runtime, balanced_schedule, execution_runtime as runtime
from experiments.seed_evolution.audit import same_tree


runner = runtime.runner
_PARENT = re.compile(r"(?:shared|evolution)_g(\d{2})_i(\d{2})")
_MAX_WORKERS = 8


class Shard(NamedTuple):
    generation: int
    plan_sha256: str
    descriptors: list
    spec: dict
    schedule: balanced_schedule.Schedule


def _preflight_parent(descriptor):
    parent = descriptor["parent_id"]
    if parent is None:
        return
    match = _PARENT.fullmatch(parent)
    if match is None or int(match[1]) >= descriptor["generation"] or int(match[2]) >= runner.CONFIG["population"]:
        raise ValueError(f"invalid parent identity for {descriptor['id']}")
    location = runner._candidate_dir(descriptor["rep"], parent)
    record = runtime.read(location / "candidate.json")
    if record["id"] != parent or record["rep"] != descriptor["rep"]:
        raise ValueError(f"wrong parent record for {descriptor['id']}")
    runner._verify_hash(location / "initial.pt", record["artifacts"]["initial.pt"])


def _load_shard(jobs_path):
    path = Path(jobs_path).resolve(strict=True)
    if not path.is_relative_to(runner.ROOT):
        raise ValueError("jobs manifest must be inside the private repository copy")
    request = runtime.read(path)
    if set(request) != {"generation", "plan_sha256", "jobs"}:
        raise ValueError("invalid shard manifest fields")
    generation = request["generation"]
    spec = runtime.load_amendment()
    schedule = balanced_schedule.load()
    if type(generation) is not int or not spec["start_generation"] <= generation < runner.CONFIG["generations"]:
        raise ValueError("shard must use a generation with an amended CPU allocation")
    if str(torch.__version__) != spec["backend_metadata"]["cpu"]["torch_version"]:
        raise ValueError("worker PyTorch differs from the sealed CPU backend version")
    plan_path = runner._plan_path(generation)
    runner._verify_hash(plan_path, request["plan_sha256"])
    plan = runtime.read(plan_path)
    if type(plan["generation"]) is not int or plan["generation"] != generation:
        raise ValueError("plan generation differs from shard manifest")
    planned = {(row["rep"], row["id"]): row for row in plan["candidates"]}
    if len(planned) != len(plan["candidates"]):
        raise ValueError("issued plan contains duplicate candidate identities")
    descriptors = request["jobs"]
    if not isinstance(descriptors, list) or not 1 <= len(descriptors) <= len(planned):
        raise ValueError("shard must contain a nonempty list of issued candidate jobs")
    assigned = set()
    for descriptor in descriptors:
        ident = (descriptor["rep"], descriptor["id"])
        if ident not in planned or ident in assigned:
            raise ValueError(f"unplanned or repeated candidate in shard: {ident}")
        same_tree(descriptor, planned[ident], "shard descriptor differs from issued plan")
        if (descriptor["generation"] != generation or descriptor["rep"] not in runner.CONFIG["source_seeds"]
                or descriptor["source_seed"] != descriptor["rep"] or schedule.backend(descriptor["id"], spec) != "cpu"):
            raise ValueError(f"candidate does not belong to the sealed CPU shard: {ident}")
        _preflight_parent(descriptor)
        assigned.add(ident)
    runner._verify_hash(plan_path, request["plan_sha256"])
    return Shard(generation, request["plan_sha256"], descriptors, spec, schedule)


def _event(event, **fields):
    print(json.dumps(dict(event=event, **fields), allow_nan=False), flush=True)


def _candidate_job(descriptor):
    return runtime._worker(descriptor, "candidate", "cpu")


def _run_shard(shard, workers):
    jobs = iter(shard.descriptors)
    pending = {}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=balanced_runtime._init_worker, initargs=(shard.spec, shard.schedule))

    def submit(descriptor):
        pending[pool.submit(_candidate_job, descriptor)] = descriptor

    try:
        for _ in range(workers):
            submit(next(jobs))
        completed = 0
        total = len(shard.descriptors)
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            records = []
            for future in finished:
                descriptor = pending.pop(future)
                try:
                    records.append(future.result())
                except Exception as error:
                    raise RuntimeError(f"candidate failed: rep_{descriptor['rep']}/{descriptor['id']}") from error
            for record in records:
                completed += 1
                _event("candidate_complete", rep=record["rep"], id=record["id"], completed=completed, total=total)
                descriptor = next(jobs, None)
                if descriptor is not None:
                    submit(descriptor)
    except BaseException:
        if hasattr(pool, "terminate_workers"):
            pool.terminate_workers()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run(jobs_path, workers=4):
    if type(workers) is not int or not 1 <= workers <= _MAX_WORKERS:
        raise ValueError(f"workers must be an integer from 1 to {_MAX_WORKERS}")
    with runner._lock(runner.OUT / ".pipeline.lock"), runner._lock(runner.OUT / ".run.lock"):
        runner._check()
        shard = _load_shard(jobs_path)
        total = len(shard.descriptors)
        workers = min(workers, total, os.process_cpu_count() or 1)
        _event("started", generation=shard.generation, jobs=total, workers=workers, plan_sha256=shard.plan_sha256)
        _run_shard(shard, workers)
        _event("complete", generation=shard.generation, completed=total, plan_sha256=shard.plan_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="Assigned generation plan shard within the private copy")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.jobs, args.workers)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
