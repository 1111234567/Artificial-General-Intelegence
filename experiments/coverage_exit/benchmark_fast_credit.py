"""Matched native vs analytic block-RTRL timings, with actual trainer inputs.

python -m experiments.coverage_exit.benchmark_fast_credit --output \
  data/coverage-exit/perf/block-rtrl-cpu.json --repeats 4

The default is H80/B8/T128 FP64, fixed graph ports, both graph modes, one CPU
thread. A native update calibrates the starter, so timing uses nonzero S and
Adam state. Construction/checkpoint validation are outside timed regions.
Every pair alternates AB/BA; full train_update includes world/live/replay,
hashing/diagnostics, gradient combination, clipping, structure, and Adam.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from experiments.coverage_exit.fast_credit import (
    FastTemporalCreditEngine, backend_manifest, install_fast_credit,
)
from experiments.undirected.credit import (
    TemporalCreditEngine, state_digest, snapshot_parameters, _native_process_memory, _cpu,
)
from experiments.undirected.run import NativeConfig, NativeTrainer, source_hashes, file_sha256


def difference(actual, expected):
    actual, expected = actual.detach().double().cpu(), expected.detach().double().cpu()
    absolute = (actual - expected).abs()
    scale = float(expected.abs().max())
    return {"bitwise_equal": torch.equal(actual, expected),
            "max_abs_error": float(absolute.max()),
            "max_abs_error_over_reference_max": float(absolute.max()) / scale if scale else 0.,
            "allclose_atol_3e-12_rtol_3e-11": torch.allclose(actual, expected, atol=3e-12, rtol=3e-11),
            "all_finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())}


def tensor_map_difference(actual, expected):
    return {name: difference(actual[name], expected[name]) for name in expected}


def require_numeric_equivalence(value):
    if isinstance(value, dict):
        if "allclose_atol_3e-12_rtol_3e-11" in value:
            if not value["allclose_atol_3e-12_rtol_3e-11"] or not value["all_finite"]:
                raise AssertionError(f"FP64 numerical equivalence check failed: {value}")
        for child in value.values():
            require_numeric_equivalence(child)
    elif isinstance(value, list):
        for child in value:
            require_numeric_equivalence(child)


def cpu_topology():
    """Read Windows core masks rather than assuming logical CPUs are cores."""
    if os.name != "nt":
        return {"source": "not_collected_on_this_platform", "physical_cores": []}
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    size = wintypes.DWORD(0)
    kernel.GetLogicalProcessorInformation(None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformation(buffer, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformation failed")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("CPU topology reader requires 64-bit Windows Python")
    raw, cores = buffer.raw, []
    for offset in range(0, size.value, 32):
        relationship = int.from_bytes(raw[offset + 8:offset + 12], "little")
        if relationship == 0:
            mask = int.from_bytes(raw[offset:offset + 8], "little")
            cores.append({"logical_processors": [bit for bit in range(64) if mask & (1 << bit)],
                          "hyperthreaded": bool(raw[offset + 16] & 1)})
    return {"source": "Windows GetLogicalProcessorInformation / RelationProcessorCore",
            "physical_cores": cores}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def paired_times(prepare, execute, *, repeats, warmups, device):
    # prepare may clone/restore native state and is deliberately outside timing.
    for _ in range(warmups):
        for backend in ("native", "analytic"):
            value = prepare(backend)
            execute(value)
            _sync(device)
            del value
    seconds = {"native": [], "analytic": []}
    ordering = []
    for index in range(repeats):
        order = ("native", "analytic") if index % 2 == 0 else ("analytic", "native")
        ordering.append(list(order))
        for backend in order:
            value = prepare(backend)
            gc.collect()
            _sync(device)
            started = time.perf_counter()
            execute(value)
            _sync(device)
            seconds[backend].append(time.perf_counter() - started)
            del value
    medians = {backend: statistics.median(values) for backend, values in seconds.items()}
    return {"seconds": seconds, "median_seconds": medians,
            "native_over_analytic_median_speedup": medians["native"] / medians["analytic"],
            "pair_order": ordering, "warmups_per_backend": warmups,
            "prepare_and_gc_outside_timed_region": True,
            "device_synchronized_end_to_end": str(device).startswith("cuda")}


def benchmark_case(args, interface, mode):
    config = NativeConfig(seed=args.seed, interface=interface, mode=mode, width=args.width,
                          batch_size=args.batch_size, chunk_ticks=args.ticks,
                          device=args.device, dtype="float64", cpu_threads=args.threads)
    if args.checkpoint:
        starter = NativeTrainer.from_checkpoint(args.checkpoint, device=args.device)
        config = starter.config
        if (config.interface, config.mode) != (interface, mode):
            raise ValueError("checkpoint interface/mode differ from selected case")
        origin = {"checkpoint": str(Path(args.checkpoint).resolve()), "updates": starter.updates,
                  "checkpoint_sha256": file_sha256(args.checkpoint), "strict_sources": True}
    else:
        starter = NativeTrainer(config)
        for _ in range(args.calibration_updates):
            starter.train_update()
        origin = {"newborn_seed": args.seed, "native_calibration_updates": args.calibration_updates,
                  "updates": starter.updates}
    torch.set_num_threads(args.threads)
    # Preserve the source checkpoint's original RNG record when migrating a
    # CPU checkpoint to CUDA. Merely initializing CUDA adds CUDA RNG state;
    # resnapshotting before a completed GPU update would change that record.
    state = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)
             if args.checkpoint else starter.state_dict())
    chunk_trainer = NativeTrainer.from_state_dict(deepcopy(state), device=args.device)
    torch.set_num_threads(args.threads)
    snapshot, _, chunk = chunk_trainer._live_chunk()
    del chunk_trainer
    engine_state = starter.credit.state_dict()
    engines = {"native": TemporalCreditEngine(starter.model, starter.credit.config),
               "analytic": FastTemporalCreditEngine(starter.model, starter.credit.config)}
    for engine in engines.values():
        engine.load_state_dict(engine_state)
    record = engines["native"]._record(chunk, snapshot)
    expected_j, expected_k = engines["native"]._block_jacobians(record)
    actual_j, actual_k = engines["analytic"]._block_jacobians(record)
    accuracy = {"J": difference(actual_j, expected_j), "K": difference(actual_k, expected_k)}
    del actual_j, actual_k, expected_j, expected_k

    expected_g, _ = engines["native"].gradients(chunk, snapshot)
    actual_g, _ = engines["analytic"].gradients(chunk, snapshot)
    accuracy["live_gradients"] = tensor_map_difference(actual_g, expected_g)
    accuracy["next_full_S"] = difference(engines["analytic"].sensitivity, engines["native"].sensitivity)
    del actual_g, expected_g

    def trainer_for(backend):
        trainer = NativeTrainer.from_state_dict(deepcopy(state), device=args.device)
        torch.set_num_threads(args.threads)
        return install_fast_credit(trainer) if backend == "analytic" else trainer

    def trajectory_for(backend):
        trainer, capture, trajectory = trainer_for(backend), {}, []
        original_credit = trainer.credit.gradients
        original_aux, original_apply = trainer._auxiliary_gradients, trainer._apply_gradients
        def credit(*args, **kwargs):
            result = original_credit(*args, **kwargs)
            capture["live"] = _cpu(result[0])
            return result
        def auxiliary(*args, **kwargs):
            result = original_aux(*args, **kwargs)
            capture["auxiliary"] = _cpu(result[0])
            return result
        def apply(gradients, *args, **kwargs):
            capture["combined"] = _cpu(gradients)
            return original_apply(gradients, *args, **kwargs)
        trainer.credit.gradients = credit
        trainer._auxiliary_gradients, trainer._apply_gradients = auxiliary, apply
        for _ in range(args.continuation_updates):
            trainer.train_update()
            trajectory.append({"tick": trainer.ticks, "gradients": deepcopy(capture),
                "model": snapshot_parameters(trainer.model), "Adam": _cpu(trainer.optimizer.state_dict()["state"]),
                "hidden": _cpu(trainer.h), "full_S": _cpu(trainer.credit.sensitivity),
                "clip_scale": trainer.trace[-1]["update_metrics"]["clip_scale"],
                "hashes": {name: state_digest(getter(trainer)) for name, getter in {
                    "actions": lambda t: t.last_chunk["actions"], "action_hash": lambda t: t.action_hash,
                    "data_hash": lambda t: t.data_hash, "world": lambda t: t.world.state_dict(),
                    "auxiliary_source": lambda t: t.last_auxiliary, "replay": lambda t: t.replay.state_dict(),
                }.items()}})
        # Break wrapper cycles before another trainer is made resident.
        trainer.credit.gradients = original_credit
        trainer._auxiliary_gradients, trainer._apply_gradients = original_aux, original_apply
        return trajectory

    native_trajectory = trajectory_for("native")
    gc.collect()
    analytic_trajectory = trajectory_for("analytic")
    accuracy["continuation"] = []
    for actual, expected in zip(analytic_trajectory, native_trajectory):
        row = {"tick": actual["tick"],
               "post_Adam_parameters": tensor_map_difference(actual["model"], expected["model"]),
               "Adam_state": {str(index): tensor_map_difference(actual["Adam"][index], values)
                              for index, values in expected["Adam"].items()},
               "live_hidden": difference(actual["hidden"], expected["hidden"]),
               "next_full_S": difference(actual["full_S"], expected["full_S"]),
               "clip_scales": {"native": expected["clip_scale"], "analytic": actual["clip_scale"]},
               "bitwise_invariant": {name: actual["hashes"][name] == value
                                     for name, value in expected["hashes"].items()}}
        for name in ("live", "auxiliary", "combined"):
            row[name + "_gradients"] = tensor_map_difference(actual["gradients"][name], expected["gradients"][name])
        accuracy["continuation"].append(row)
    require_numeric_equivalence(accuracy)
    del native_trajectory, analytic_trajectory
    gc.collect()

    def prepare_credit(backend):
        engine = engines[backend]
        engine.load_state_dict(engine_state)
        return engine

    timings = {}
    if args.workload in ("all", "jacobians"):
        timings["block_jacobians"] = paired_times(lambda backend: engines[backend],
            lambda engine: engine._block_jacobians(record), repeats=args.repeats,
            warmups=args.warmups, device=args.device)
    if args.workload in ("all", "credit"):
        timings["full_credit_gradients_and_S"] = paired_times(prepare_credit,
            lambda engine: engine.gradients(chunk, snapshot), repeats=args.repeats,
            warmups=args.warmups, device=args.device)
    if args.workload in ("all", "trainer"):
        timings["full_native_train_update"] = paired_times(trainer_for,
            lambda trainer: trainer.train_update(), repeats=args.repeats,
            warmups=args.warmups, device=args.device)
    return {"interface": interface, "mode": mode,
            "dimensions": {"H": config.width, "B": config.batch_size, "T": config.chunk_ticks,
                           "P_dynamic": starter.credit.dynamic_parameter_count},
            "origin": origin, "matched_input_record_hash": state_digest(record),
            "starting_full_S_nonzero": starter.credit.sensitivity is not None,
            "accuracy": accuracy, "timings": timings,
            "process_memory_after_case": _native_process_memory()}


def memory_probe(args):
    """Run one selected backend in a fresh process without a resident peer.

    Requires a checkpoint so an analytic probe does not first execute a native
    calibration sweep and contaminate its process-lifetime memory peak.
    """
    if not args.checkpoint:
        raise ValueError("memory-probe requires --checkpoint from a matched source")
    trainer = NativeTrainer.from_checkpoint(args.checkpoint, device=args.device)
    torch.set_num_threads(args.threads)
    if args.probe_backend == "analytic":
        install_fast_credit(trainer)
    before = _native_process_memory()
    cuda_memory = None
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        cuda_memory = {"free_before_bytes": free, "total_bytes": total,
                       "allocated_before_bytes": torch.cuda.memory_allocated(args.device),
                       "reserved_before_bytes": torch.cuda.memory_reserved(args.device)}
    _sync(args.device)
    started = time.perf_counter()
    if args.probe_phase == "trainer":
        trainer.train_update()
    else:
        snapshot, _, chunk = trainer._live_chunk()
        if args.probe_phase == "credit":
            trainer.credit.gradients(chunk, snapshot)
        else:
            record = trainer.credit._record(chunk, snapshot)
            trainer.credit._block_jacobians(record)
    _sync(args.device)
    seconds = time.perf_counter() - started
    if cuda_memory is not None:
        cuda_memory.update(allocated_after_bytes=torch.cuda.memory_allocated(args.device),
                           reserved_after_bytes=torch.cuda.memory_reserved(args.device),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    return {"backend": args.probe_backend, "phase": args.probe_phase,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "seconds_cold_execution": seconds,
            "before": before, "after": _native_process_memory(),
            "cuda_memory": cuda_memory,
            "peak_scope": "fresh process lifetime, including imports and checkpoint validation",
            "timing_scope": "cold memory probe; use warmed alternating benchmark for speed claims"}


def thread_sweep(args):
    """One optimized job at a time, matched real CP, varying CPU execution threads."""
    if not args.checkpoint or args.device != "cpu":
        raise ValueError("thread-sweep requires a real --checkpoint and --device cpu")
    threads = tuple(int(value) for value in args.thread_sweep.split(","))
    if not threads or any(value < 1 for value in threads):
        raise ValueError("thread-sweep needs positive comma-separated counts")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    import psutil
    process = psutil.Process()
    affinity_pool = process.cpu_affinity()
    topology = cpu_topology()
    selected = []
    for core in topology["physical_cores"]:
        candidates = [cpu for cpu in core["logical_processors"] if cpu in affinity_pool]
        if candidates:
            selected.append(candidates[0])
    if not selected:
        selected = affinity_pool
    if max(threads) > len(selected):
        raise ValueError("thread count exceeds distinct allowed physical cores")
    affinities = {str(count): selected[:count] for count in threads}
    reference = None
    accuracy = []
    def prepare(count):
        process.cpu_affinity(affinities[str(count)])
        trainer = install_fast_credit(NativeTrainer.from_state_dict(deepcopy(state), device="cpu"))
        # This changes execution only. The saved mathematical/training config
        # stays source-faithful; the benchmark records this explicit override.
        torch.set_num_threads(count)
        return trainer
    for count in threads:
        for _ in range(args.warmups):
            trainer = prepare(count)
            trainer.train_update()
            current = {"model": snapshot_parameters(trainer.model), "h": trainer.h.detach().clone(),
                       "S": trainer.credit.sensitivity.detach().clone(), "actions": trainer.action_hash}
            if reference is None:
                reference = current
            accuracy.append({"threads": count,
                "model": tensor_map_difference(current["model"], reference["model"]),
                "h": difference(current["h"], reference["h"]),
                "S": difference(current["S"], reference["S"]),
                "action_hash_equal": current["actions"] == reference["actions"]})
            del trainer, current
    require_numeric_equivalence(accuracy)
    times = {str(count): [] for count in threads}
    cpu_usage = {str(count): [] for count in threads}
    orders = []
    for index in range(args.repeats):
        order = threads if index % 2 == 0 else tuple(reversed(threads))
        orders.append(order)
        for count in order:
            trainer = prepare(count)
            gc.collect()
            before_cpu = process.cpu_times()
            psutil.cpu_percent(interval=None, percpu=True)
            started = time.perf_counter()
            trainer.train_update()
            elapsed = time.perf_counter() - started
            after_cpu = process.cpu_times()
            times[str(count)].append(elapsed)
            cpu_usage[str(count)].append({
                "process_cpu_percent": 100. * (after_cpu.user + after_cpu.system - before_cpu.user - before_cpu.system) / elapsed,
                "system_cpu_percent_per_logical_processor": psutil.cpu_percent(interval=None, percpu=True)})
            del trainer
    return {"checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "backend": "analytic", "execution_threads_override": list(threads),
            "saved_config_cpu_threads": state["config"]["cpu_threads"],
            "seconds": times, "median_seconds": {key: statistics.median(value) for key, value in times.items()},
            "cpu_affinity_by_thread_count": affinities, "cpu_topology": topology,
            "cpu_usage_during_timed_update": cpu_usage,
            "pair_order": orders, "warmups_per_count": args.warmups,
            "accuracy_against_first_count": accuracy, "one_job_resident_at_a_time": True,
            "scope": "full native train_update; thread override, no mathematical/training change"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interface", choices=("ports", "core", "both"), default="ports")
    parser.add_argument("--mode", choices=("undirected", "untied", "both"), default="both")
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", type=int, help="Optional process CPU affinity via psutil")
    parser.add_argument("--cpus", help="Optional comma-separated CPU affinity set via psutil")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--calibration-updates", type=int, default=1)
    parser.add_argument("--continuation-updates", type=int, choices=(1, 2, 3), default=1,
                        help="Accuracy-check consecutive actual updates; timing stays one update")
    parser.add_argument("--checkpoint")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--workload", choices=("all", "jacobians", "credit", "trainer"), default="all")
    parser.add_argument("--probe-backend", choices=("native", "analytic"))
    parser.add_argument("--probe-phase", choices=("jacobians", "credit", "trainer"), default="trainer")
    parser.add_argument("--environment-note", default="",
                        help="Record external load/isolation information without inferring it")
    parser.add_argument("--thread-sweep", help="Optimized real-CP CPU one-job thread sweep, e.g. 1,2,4")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.repeats < 1 or args.warmups < 1 or args.calibration_updates < 1:
        parser.error("positive repeats, warmups and calibration-updates required")
    if args.cpu is not None and args.cpus:
        parser.error("choose either --cpu or --cpus")
    affinity = [args.cpu] if args.cpu is not None else ([int(x) for x in args.cpus.split(",")] if args.cpus else None)
    if affinity is not None:
        import psutil
        psutil.Process().cpu_affinity(affinity)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    interfaces = ("ports", "core") if args.interface == "both" else (args.interface,)
    modes = ("undirected", "untied") if args.mode == "both" else (args.mode,)
    result = {"format": "coverage-exit-fast-block-rtrl-benchmark-v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "backend": backend_manifest(), "native_source_sha256": source_hashes(),
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "platform": platform.platform(), "processor": platform.processor(),
                          "pid": os.getpid(), "threads": args.threads, "cpu_affinity_requested": args.cpu,
                          "cpu_affinity_set": affinity,
                          "cpu_topology": cpu_topology(),
                          "device": args.device, "dtype": "float64", "AMP": False, "TF32": False,
                          "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                          "cuda_available_in_runtime": torch.cuda.is_available()},
              "arguments": vars(args), "cases": []}
    if str(args.device).startswith("cuda"):
        free, total = torch.cuda.mem_get_info(args.device)
        result["runtime"].update(cuda_device_name=torch.cuda.get_device_name(args.device),
                                  cuda_free_at_start_bytes=free, cuda_total_bytes=total)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the exact measured implementation available even as research evolves.
    source_copy = output.with_suffix(".fast-credit-source.py")
    source_copy.write_bytes(Path(__file__).with_name("fast_credit.py").read_bytes())
    result["backend"]["measured_source_snapshot"] = str(source_copy.resolve())
    benchmark_copy = output.with_suffix(".benchmark-source.py")
    benchmark_copy.write_bytes(Path(__file__).read_bytes())
    result["benchmark_source_snapshot"] = str(benchmark_copy.resolve())
    result["benchmark_source_sha256"] = file_sha256(benchmark_copy)
    if args.thread_sweep:
        result["thread_sweep"] = thread_sweep(args)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"thread_sweep_median_seconds": result["thread_sweep"]["median_seconds"]}), flush=True)
        return
    if args.probe_backend:
        result["memory_probe"] = memory_probe(args)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result["memory_probe"], allow_nan=False), flush=True)
        return
    for interface in interfaces:
        for mode in modes:
            print(f"benchmark starting: {interface}/{mode} H{args.width} B{args.batch_size} T{args.ticks}", flush=True)
            case = benchmark_case(args, interface, mode)
            result["cases"].append(case)
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"case": f"{interface}/{mode}",
                              "timings": {name: values["median_seconds"] for name, values in case["timings"].items()}},
                             allow_nan=False), flush=True)
    print(str(output.resolve()), flush=True)


if __name__ == "__main__":
    main()
