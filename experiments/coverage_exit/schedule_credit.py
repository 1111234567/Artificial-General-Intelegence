"""Fixed four-physical-core, four-replica FP64 RTRL scheduling measurement.

All four independent copies of the same learned checkpoint are restored and
warmed before a configuration's wall clock starts. Configurations 1x4, 2x2,
4x1 mean concurrent jobs x execution threads; four jobs are always completed.
The measurement calls complete native train_update with opt-in exact J/K.
Extra tensor comparisons and serialization happen after every job reports done.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import time
import traceback

import psutil
import torch

from experiments.coverage_exit.benchmark_fast_credit import (
    cpu_topology, difference, require_numeric_equivalence, tensor_map_difference,
)
from experiments.coverage_exit.fast_credit import backend_manifest, install_fast_credit
from experiments.undirected.credit import _cpu, _native_process_memory, snapshot_parameters, state_digest
from experiments.undirected.run import NativeTrainer, file_sha256, source_hashes


def _worker(job, cores, threads, checkpoint, updates, commands, results, payload_path):
    try:
        psutil.Process().cpu_affinity(cores)
        trainer = install_fast_credit(NativeTrainer.from_checkpoint(checkpoint, device="cpu"))
        if trainer.config.dtype != "float64" or trainer.config.credit != "rtrl":
            raise ValueError("scheduling requires an actual learned FP64 RTRL checkpoint")
        torch.set_num_threads(threads)
        results.put({"event": "ready", "job": job, "pid": psutil.Process().pid,
                     "memory": _native_process_memory(), "actual_threads": torch.get_num_threads(),
                     "affinity": psutil.Process().cpu_affinity()})
        while True:
            command = commands.get()
            if command == "stop":
                return
            if command == "warm":
                trainer.train_update()
                results.put({"event": "warm", "job": job})
            elif command == "measure":
                started = time.perf_counter()
                for _ in range(updates):
                    trainer.train_update()
                seconds = time.perf_counter() - started
                # No additional tensor hashes, scientific comparison, or output
                # serialization precedes this completion message.
                results.put({"event": "done", "job": job, "seconds": seconds,
                             "updates": updates, "tick": trainer.ticks})
            elif command == "validate":
                memory_before_validation = _native_process_memory()
                payload = {"model": snapshot_parameters(trainer.model),
                    "h": _cpu(trainer.h), "S": _cpu(trainer.credit.sensitivity),
                    "Adam": _cpu(trainer.optimizer.state_dict()["state"]),
                    "action_hash": trainer.action_hash, "data_hash": trainer.data_hash,
                    "world_hash": state_digest(trainer.world.state_dict()),
                    "action_tensor_hash": state_digest(trainer.last_chunk["actions"]),
                    "tick": trainer.ticks, "updates": trainer.updates}
                torch.save(payload, payload_path)
                results.put({"event": "validated", "job": job,
                    "payload": str(payload_path), "payload_sha256": file_sha256(payload_path),
                    "memory_before_extra_validation": memory_before_validation,
                    "payload_content_hash": state_digest(payload)})
            else:
                raise ValueError(f"unknown worker command {command}")
    except BaseException as error:
        results.put({"event": "error", "job": job, "exception": repr(error),
                     "traceback": traceback.format_exc()})


def compare_payload(actual, expected):
    result = {"model": tensor_map_difference(actual["model"], expected["model"]),
        "h": difference(actual["h"], expected["h"]), "S": difference(actual["S"], expected["S"]),
        "Adam": {str(index): tensor_map_difference(actual["Adam"][index], values)
                 for index, values in expected["Adam"].items()},
        "bitwise_invariants": {name: actual[name] == expected[name] for name in
            ("action_hash", "data_hash", "world_hash", "action_tensor_hash", "tick", "updates")}}
    require_numeric_equivalence(result)
    if not all(result["bitwise_invariants"].values()):
        raise AssertionError(f"replica actions/world/clock differ: {result['bitwise_invariants']}")
    return result


def run_configuration(args, context, round_index, concurrent, cores, folder, reference):
    threads = len(cores) // concurrent
    label = f"{concurrent}x{threads}"
    prefix = f"round-{round_index}-{label}"
    available = psutil.virtual_memory().available
    if available < args.min_available_gib * 1024 ** 3:
        raise MemoryError(f"admission needs {args.min_available_gib} GiB available; got {available / 1024**3:.2f}")
    results = context.Queue()
    commands = [context.Queue() for _ in range(4)]
    workers, assignments = [], []
    for job in range(4):
        partition = job % concurrent
        assignment = cores[partition * threads:(partition + 1) * threads]
        assignments.append(assignment)
        workers.append(context.Process(target=_worker, args=(job, assignment, threads,
            args.checkpoint, args.updates, commands[job], results, folder / f"{prefix}-job-{job}.pt")))
    samples, records = [], []
    maximum_rss, minimum_available = 0, available
    def sample_load():
        nonlocal maximum_rss, minimum_available
        rss, cpu_seconds = [], []
        for worker in workers:
            try:
                process = psutil.Process(worker.pid)
                rss.append(process.memory_info().rss)
                times = process.cpu_times()
                cpu_seconds.append(times.user + times.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                rss.append(0)
                cpu_seconds.append(0.)
        memory_available = psutil.virtual_memory().available
        maximum_rss = max(maximum_rss, sum(rss))
        minimum_available = min(minimum_available, memory_available)
        entry = {"monotonic_seconds": time.perf_counter(), "worker_rss_bytes": rss,
                 "worker_cpu_seconds": cpu_seconds,
                 "system_cpu_percent_per_logical_processor": psutil.cpu_percent(interval=None, percpu=True),
                 "available_memory_bytes": memory_available}
        samples.append(entry)
        if memory_available < args.abort_available_gib * 1024 ** 3:
            raise MemoryError("available memory fell below run abort threshold")
        if sum(rss) > args.max_workers_rss_gib * 1024 ** 3:
            raise MemoryError("four-worker RSS exceeded run admission budget")

    def wait_for(event, pending, timed=False):
        pending = set(pending)
        deadline, last_sample = time.perf_counter() + args.timeout, 0.
        received = []
        while pending:
            now = time.perf_counter()
            if now > deadline:
                raise TimeoutError(f"workers timed out waiting for {event}: {pending}")
            if now - last_sample >= .25:
                sample_load()
                last_sample = now
            try:
                message = results.get(timeout=.1)
            except queue.Empty:
                dead = [job for job in pending if workers[job].exitcode is not None]
                if dead:
                    raise RuntimeError(f"workers exited before {event}: {dead}")
                continue
            if message["event"] == "error":
                raise RuntimeError(message["traceback"])
            if message["event"] != event or message["job"] not in pending:
                raise RuntimeError(f"unexpected worker event: {message}")
            pending.remove(message["job"])
            received.append(message)
        records.extend(received)
        return received

    try:
        for worker in workers:
            worker.start()
        ready = wait_for("ready", range(4))
        # Warmups follow the same non-overlapping CPU budget, outside timing.
        for first in range(0, 4, concurrent):
            wave = list(range(first, first + concurrent))
            for job in wave:
                commands[job].put("warm")
            wait_for("warm", wave)
        sample_load()
        first_timed_sample = len(samples) - 1
        started = time.perf_counter()
        done = []
        for first in range(0, 4, concurrent):
            wave = list(range(first, first + concurrent))
            for job in wave:
                commands[job].put("measure")
            done.extend(wait_for("done", wave, timed=True))
        wall_seconds = time.perf_counter() - started
        sample_load()
        last_timed_sample = len(samples)
        for job in range(4):
            commands[job].put("validate")
        validated = wait_for("validated", range(4))
        comparisons = []
        for item in sorted(validated, key=lambda value: value["job"]):
            payload = torch.load(item["payload"], map_location="cpu", weights_only=False)
            if reference is None:
                reference = payload
                torch.save(reference, folder / "validation-reference.pt")
            comparisons.append({"job": item["job"], "payload_sha256": item["payload_sha256"],
                "payload_content_hash": item["payload_content_hash"], "comparison": compare_payload(payload, reference),
                "memory_before_extra_validation": item["memory_before_extra_validation"]})
            # Only these new per-job scratch files are removed, individually.
            scratch = Path(item["payload"]).resolve()
            if scratch.parent != folder.resolve() or scratch.name != f"{prefix}-job-{item['job']}.pt":
                raise ValueError("scratch cleanup escaped the scheduling output directory")
            scratch.unlink()
        timed_samples = samples[first_timed_sample:last_timed_sample]
        return {"round": round_index, "configuration": label, "concurrent_jobs": concurrent,
            "threads_per_job": threads, "total_jobs": 4, "warmup_updates_per_job": 1,
            "measured_updates_per_job": args.updates, "total_measured_updates": 4 * args.updates,
            "core_budget": cores, "job_affinities": assignments, "worker_ready": ready,
            "wall_seconds_four_jobs": wall_seconds, "updates_per_second": 4 * args.updates / wall_seconds,
            "worker_measurements": sorted(done, key=lambda value: value["job"]),
            "available_memory_before_bytes": available,
            "minimum_available_memory_bytes": minimum_available,
            "maximum_sampled_sum_worker_rss_bytes": maximum_rss,
            "timed_load_samples": timed_samples, "validation": comparisons,
            "preload_warmup_and_extra_validation_outside_timing": True}, reference
    finally:
        for channel in commands:
            channel.put("stop")
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=5.)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5.)
        for channel in commands:
            channel.close()
        results.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cores", default="0,2,4,6")
    parser.add_argument("--coordinator-cpu", type=int, default=12)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-available-gib", type=float, default=8.)
    parser.add_argument("--abort-available-gib", type=float, default=2.)
    parser.add_argument("--max-workers-rss-gib", type=float, default=8.)
    parser.add_argument("--timeout", type=float, default=300.)
    parser.add_argument("--environment-note", default="")
    args = parser.parse_args(argv)
    cores = [int(value) for value in args.cores.split(",")]
    if len(cores) != 4 or len(set(cores)) != 4 or args.updates < 3 or args.rounds < 2:
        parser.error("exactly four distinct cores, >=3 measured updates and >=2 rounds required")
    topology = cpu_topology()
    core_ids = [index for cpu in cores for index, core in enumerate(topology["physical_cores"])
                if cpu in core["logical_processors"]]
    if len(core_ids) != 4 or len(set(core_ids)) != 4:
        parser.error("selected logical CPUs must belong to four different verified physical cores")
    psutil.Process().cpu_affinity([args.coordinator_cpu])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = output.with_suffix("")
    folder.mkdir(exist_ok=True)
    for name in ("schedule_credit.py", "fast_credit.py"):
        (folder / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    record = {"format": "coverage-exit-four-core-credit-scheduling-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "runtime": {"torch": torch.__version__, "python": platform.python_version(),
                    "platform": platform.platform(), "dtype": "float64", "AMP": False, "TF32": False},
        "backend": backend_manifest(), "native_source_sha256": source_hashes(),
        "checkpoint_sha256": file_sha256(args.checkpoint), "cpu_topology": topology,
        "scope": "four replicas and four physical cores only; total 12 measured updates per configuration",
        "timing": "restore and all warmups excluded; complete native train_update includes its normal trace bookkeeping; extra validation/serialization excluded",
        "cases": []}
    context, reference = mp.get_context("spawn"), None
    for round_index in range(args.rounds):
        order = (1, 2, 4) if round_index % 2 == 0 else (4, 2, 1)
        for concurrent in order:
            print(f"starting round={round_index} concurrency={concurrent} threads={4 // concurrent}", flush=True)
            result, reference = run_configuration(args, context, round_index, concurrent, cores, folder, reference)
            record["cases"].append(result)
            output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"round": round_index, "configuration": result["configuration"],
                              "wall_seconds": result["wall_seconds_four_jobs"],
                              "updates_per_second": result["updates_per_second"]}), flush=True)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
