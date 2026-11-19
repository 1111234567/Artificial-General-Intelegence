"""Use a second Windows host for a disjoint portion of each CPU queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from threading import Event
import time
from unittest.mock import patch
from uuid import uuid4

from experiments.seed_evolution import balanced_runtime, execution_runtime as runtime, large_audit, lan_dispatch


runner = runtime.runner
_local_parallel = balanced_runtime._parallel
_ARTIFACTS = ("attempt.json", "evaluation-time.json", *runner.CANDIDATE_ARTIFACTS)


def _select_remote(jobs, spec, schedule, pairs):
    if pairs < 1:
        return []
    matched = {}
    for job in jobs:
        if (schedule.backend(job["id"], spec) == "cpu"
                and not runner._candidate_dir(job["rep"], job["id"]).exists()):
            matched.setdefault((job["rep"], job["index"]), []).append(job)
    chosen = []
    for pair in matched.values():
        if {job["arm"] for job in pair} == {"evolution", "random"}:
            chosen.extend(pair)
            if len(chosen) >= 2 * pairs:
                break
    return chosen


def _import_candidate(source, job, spec):
    marker = source / "candidate.json"
    try:
        marker_bytes = marker.read_bytes()
    except FileNotFoundError:
        return None
    dest = runner._candidate_dir(job["rep"], job["id"])
    record = json.loads(marker_bytes)
    if any(record[key] != value for key, value in job.items()):
        raise ValueError(f"remote candidate differs from plan: {marker}")
    if set(record["artifacts"]) != set(runner.CANDIDATE_ARTIFACTS):
        raise ValueError(f"remote candidate artifact inventory differs: {marker}")
    stage = dest.with_name(f".{dest.name}.{uuid4().hex}.lan")
    stage.mkdir(parents=True)
    try:
        for name in _ARTIFACTS:
            shutil.copyfile(source / name, stage / name)
        if (source / "execution.json").is_file():
            shutil.copyfile(source / "execution.json", stage / "execution.json")
        for name, digest in record["artifacts"].items():
            runner._verify_hash(stage / name, digest)
        runtime._validate_training(stage / "training.json", "cpu", spec)
        large_audit.verify_attempt(stage, record)
        genome = runner._genome(job)
        runtime.common.assert_tree_equal(runner._load(stage / "initial.pt"), genome, "LAN candidate initialization")
        checkpoint = runner._load(stage / "inner64.pt")
        source_score = runtime.read(stage / "score.json")
        if (record["genome_hash"] != runtime.core.genome_hash(genome)
                or checkpoint["source_seed"] != job["rep"]
                or checkpoint["updates"] != runner.CONFIG["inner_steps"]
                or runner.base.model_hash(checkpoint["model"]) != record["trained_hash"]
                or source_score["parameter_hash"] != record["trained_hash"]
                or source_score["fitness"] != record["fitness"]):
            raise ValueError(f"remote candidate result differs from its training or score: {marker}")
        runner._write(stage / "lan-origin.json", dict(
            host="DESKTOP-T3C3GMF", backend="cpu", candidate_sha256=hashlib.sha256(marker_bytes).hexdigest()))
        (stage / "candidate.json").write_bytes(marker_bytes)
        if dest.exists():
            raise FileExistsError(dest)
        runner._replace(stage, dest)
        return record
    finally:
        if stage.is_dir():
            for artifact in stage.iterdir():
                artifact.unlink()
            stage.rmdir()


def _run_remote(transport, jobs, spec, schedule, cancel):
    generation = jobs[0]["generation"]
    remaining = {(job["rep"], job["id"]): job for job in jobs}
    shard = transport.prepare(jobs, runner._plan_path(generation), spec=spec, schedule=schedule)
    prefix = runner.OUT / f"lan-g{generation:02d}-{uuid4().hex[:8]}"
    print(f"LAN generation {generation + 1}: assigned {len(jobs)} CPU candidates", flush=True)
    with prefix.with_suffix(".stdout.log").open("xb") as stdout, prefix.with_suffix(".stderr.log").open("xb") as stderr:
        process = shard.start(stdout, stderr)
        try:
            deadline = time.monotonic() + 8 * 60
            while not cancel.is_set():
                for identity, job in tuple(remaining.items()):
                    if cancel.is_set():
                        return
                    if _import_candidate(shard.result_path(job), job, spec) is not None:
                        del remaining[identity]
                        deadline = time.monotonic() + 8 * 60
                if process.poll() is not None:
                    if process.returncode or remaining:
                        raise RuntimeError(f"LAN worker exited {process.returncode}; {len(remaining)} candidates pending; {prefix}")
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"LAN shard made no progress for 8 minutes: {prefix}")
                cancel.wait(2)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _parallel(jobs, worker, workers, *, transport, pairs, **options):
    if not jobs or worker.__name__ != "_candidate_job":
        yield from _local_parallel(jobs, worker, workers, **options)
        return
    generation = jobs[0]["generation"]
    assignments = runner.OUT / "lan-assignments" / f"generation_{generation:02d}.json"
    plan_sha = runtime.sha(runner._plan_path(generation))
    planned = {(job["rep"], job["id"]): job for job in jobs}
    if assignments.is_file():
        state = runtime.read(assignments)
        if state["plan_sha256"] != plan_sha:
            raise ValueError("Persisted LAN assignments differ from the issued plan")
        remote = [planned[ident] for pair in state["candidate_ids"] if (ident := tuple(pair)) in planned
                  and not runner._candidate_dir(*ident).exists()]
    else:
        remote = _select_remote(jobs, options["spec"], options["schedule"], pairs)
        if remote:
            runner._write(assignments, dict(plan_sha256=plan_sha,
                candidate_ids=[[job["rep"], job["id"]] for job in remote]))
    if not remote:
        yield from _local_parallel(jobs, worker, workers, **options)
        return
    assigned = {(job["rep"], job["id"]) for job in remote}
    local = [job for job in jobs if (job["rep"], job["id"]) not in assigned]
    reported = set()

    def imported_records():
        for job in remote:
            identity = job["rep"], job["id"]
            if identity not in reported and (runner._candidate_dir(*identity) / "candidate.json").is_file():
                record = runtime._committed(job)
                if record is not None:
                    reported.add(identity)
                    yield record

    cancel = Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(_run_remote, transport, remote, options["spec"], options["schedule"], cancel)
        try:
            for record in _local_parallel(local, worker, workers, **options):
                yield record
                yield from imported_records()
            while not result.done():
                yield from imported_records()
                cancel.wait(2)
            try:
                result.result()
            except Exception as error:
                print(f"LAN shard returned incomplete: {error}", flush=True)
        finally:
            cancel.set()
    yield from imported_records()
    remaining = [job for job in remote if (job["rep"], job["id"]) not in reported]
    yield from _local_parallel(remaining, worker, workers, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--known-hosts", required=True, type=Path)
    parser.add_argument("--cpu-workers", type=int, default=11)
    parser.add_argument("--hybrid-workers", type=int, default=4)
    parser.add_argument("--remote-workers", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=28)
    parser.add_argument("--hybrid-start-free-mib", type=int, default=5600)
    args = parser.parse_args()
    ssh = ["ssh.exe", "-n", "-T", "-i", str(args.key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
           "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={args.known_hosts}",
           "-o", "HostKeyAlgorithms=ssh-ed25519", "Jerry@192.168.0.120"]
    remote_command = r"py -3.14 -I -B C:\Users\Jerry\Desktop\agi-lan-probe-20261114\lan-start.py"
    transport = lan_dispatch.ShardTransport(args.remote_root, ssh, remote_command, args.remote_workers)
    with patch.object(balanced_runtime, "_parallel", partial(_parallel, transport=transport, pairs=args.pairs)):
        balanced_runtime.run(args.cpu_workers, args.hybrid_workers, args.hybrid_start_free_mib)


if __name__ == "__main__":
    main()
