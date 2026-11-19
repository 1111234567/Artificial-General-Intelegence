"""Benchmark the seed search with the frozen GRU executor on CUDA."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import subprocess
from threading import Event, Thread
import time
from types import MethodType
from unittest.mock import patch

import torch

from experiments.seed_evolution import core, device_benchmark as baseline, large_runner


OUTPUT = core.ROOT / "data/seed-evolution/runtime-2026-11-11/hybrid-benchmark.json"
SHORT_STEPS = 8
MIN_LONG_SPEEDUP = 1.15
VRAM_HEADROOM_MIB = 768
_BARRIER = None


def _executor_forward(self, obs, hidden):
    device = self.head.weight.device
    if device.type == "cpu":
        self.to("cuda")
        device = self.head.weight.device
    logits, next_h = type(self).forward(self, obs.to(device), hidden.to(device))
    return logits.to(obs.device), next_h.to(hidden.device)


def _hybrid_train(genome, steps, checkpoints):
    model_from_genome = core.model_from_genome

    def attach_executor(source_seed, initial):
        model = model_from_genome(source_seed, initial)
        model.executor.forward = MethodType(_executor_forward, model.executor)
        return model

    with patch.object(core, "model_from_genome", attach_executor):
        return core.inner_train(baseline.SOURCE_SEED, genome, steps=steps,
                                checkpoints=checkpoints, device="cpu")


def _measure(genome, steps, checkpoints, hybrid):
    if hybrid:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, _ = torch.cuda.mem_get_info()
    start = time.perf_counter()
    if hybrid:
        result = _hybrid_train(genome, steps, checkpoints)
        torch.cuda.synchronize()
    else:
        result = core.inner_train(baseline.SOURCE_SEED, genome, steps=steps,
                                  checkpoints=checkpoints, device="cpu")
    end = time.perf_counter()
    stats = dict(elapsed_seconds=end - start, updates_per_second=steps / (end - start),
                 log=result["log"], metadata=result["metadata"])
    if hybrid:
        free_after, _ = torch.cuda.mem_get_info()
        stats.update(torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / baseline.MIB,
                     torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / baseline.MIB,
                     cuda_free_before_mib=free_before / baseline.MIB,
                     cuda_free_after_mib=free_after / baseline.MIB)
    return stats, result["checkpoints"], start, end


def _compare(cpu, hybrid, cpu_log, hybrid_log):
    comparison = baseline.compare_checkpoints(cpu, hybrid, cpu_log, hybrid_log)
    if not all(comparison[key] for key in ("config_matches", "optimizer_groups_match", "updates_match")):
        raise ValueError("Checkpoint structure differs across device placement")
    if cpu["source_seed"] != hybrid["source_seed"]:
        raise ValueError("Checkpoint source seed differs across device placement")
    if comparison["model"]["nonzero"] != comparison["trainable"]["nonzero"]:
        raise ValueError("A frozen model tensor changed across device placement")
    if comparison["step"]["nonzero"]:
        raise ValueError("Adam step counters differ across device placement")
    return comparison


def _init_worker(barrier):
    global _BARRIER
    _BARRIER = barrier


def _worker(steps):
    baseline.strict_fp32()
    genome = baseline.load(baseline.REFERENCE / "initial.pt")
    core._training_streams(baseline.SOURCE_SEED)
    torch.cuda.init()
    _BARRIER.wait()
    report, _, start, end = _measure(genome, steps, (0, steps), hybrid=True)
    return report, start, end


def _sample_gpu(stop, samples):
    while not stop.is_set():
        snapshot = baseline.gpu_snapshot()
        if snapshot is not None:
            samples.append(snapshot)
        stop.wait(0.5)


def _parallel(steps, workers):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(workers, timeout=120)
    stop, samples = Event(), []
    monitor = Thread(target=_sample_gpu, args=(stop, samples), daemon=True)
    gpu_before = baseline.gpu_snapshot()
    monitor.start()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker,
                                 initargs=(barrier,)) as pool:
            jobs = [pool.submit(_worker, steps) for _ in range(workers)]
            rows = [job.result() for job in jobs]
    finally:
        stop.set()
        monitor.join()
    wall = max(end for _, _, end in rows) - min(start for _, start, _ in rows)
    return dict(workers=workers, steps_per_worker=steps, overlapping_wall_seconds=wall,
                aggregate_updates_per_second=workers * steps / wall,
                all_workers_overlap_seconds=max(0, min(end for _, _, end in rows) -
                                                max(start for _, start, _ in rows)),
                workers_timing_and_memory=[stats for stats, _, _ in rows],
                sum_peak_reserved_mib=sum(stats["torch_peak_reserved_mib"] for stats, _, _ in rows),
                gpu_before=gpu_before, gpu_samples=len(samples),
                system_vram_sample_peak_mib=max((s["system_used_mib"] for s in samples), default=None),
                gpu_utilization_sample_peak=max((s["utilization_percent"] for s in samples), default=None))


def _background(pipeline_pid):
    state = baseline.read(large_runner.OUT / "progress.json")
    result = {key: state[key] for key in ("phase", "state", "generation", "completed_candidates", "updated_utc")
              if key in state}
    if pipeline_pid is not None:
        command = (
            f"$proc = Get-Process -Id {pipeline_pid} -ErrorAction SilentlyContinue; "
            f"$workers = @(Get-CimInstance Win32_Process -Filter 'ParentProcessId = {pipeline_pid}'); "
            "$cpu = Get-CimInstance Win32_Processor; "
            "$active = $null -ne $proc; "
            "@{pipeline_alive=$active; pipeline_workers=$workers.Count; "
            "cpu_percent_sample=$cpu.LoadPercentage} | ConvertTo-Json -Compress"
        )
        process = subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True,
                                 capture_output=True, text=True, timeout=15)
        result.update(pipeline_pid=pipeline_pid, **json.loads(process.stdout))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-pid", type=int)
    args = parser.parse_args()
    baseline.strict_fp32()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu_before = baseline.gpu_snapshot()
    properties = torch.cuda.get_device_properties(0)
    total_mib = properties.total_memory / baseline.MIB
    genome, archived_cpu, cpu_reference = baseline.reference()
    report = dict(status="running", started_utc=baseline.utc(), torch=str(torch.__version__),
                  cuda=torch.version.cuda, gpu=properties.name, gpu_total_mib=total_mib,
                  gpu_before=gpu_before, torch_threads=torch.get_num_threads(),
                  background_start=_background(args.pipeline_pid), source_seed=baseline.SOURCE_SEED,
                  init_seed=baseline.INIT_SEED, genome_hash=core.genome_hash(genome),
                  path=dict(cpu=["encoder", "heads", "memory_scan", "soft_translate", "loss", "Adam"],
                            cuda=["frozen_executor"], dtype="float32", tf32=False,
                            note="Autograd follows mapped observations and logits across device copies. "
                                 "The unchanged inner trainer's device metadata describes the CPU policy inputs."),
                  cpu_reference=cpu_reference)

    def publish(stage):
        large_runner._write(OUTPUT, report)
        print(json.dumps(dict(stage=stage, output=str(OUTPUT))), flush=True)

    publish("start")
    core._training_streams(baseline.SOURCE_SEED)
    torch.cuda.init()
    checkpoints = tuple(range(SHORT_STEPS + 1))
    cpu, cpu_checkpoints, _, _ = _measure(genome, SHORT_STEPS, checkpoints, hybrid=False)
    hybrid, hybrid_checkpoints, _, _ = _measure(genome, SHORT_STEPS, checkpoints, hybrid=True)
    short_speedup = cpu["elapsed_seconds"] / hybrid["elapsed_seconds"]
    core.common.assert_tree_equal(cpu_checkpoints[0], hybrid_checkpoints[0], "initial checkpoints")
    report["short"] = dict(steps=SHORT_STEPS, cpu=cpu, hybrid=hybrid, speedup=short_speedup,
                           checkpoint_differences={
                               step: _compare(cpu_checkpoints[step], hybrid_checkpoints[step],
                                              [cpu["log"][step - 1]], [hybrid["log"][step - 1]])
                               for step in range(1, SHORT_STEPS + 1)})
    publish("short_complete")

    if short_speedup >= MIN_LONG_SPEEDUP:
        hybrid64, snapshots, _, _ = _measure(genome, 64, (0, 64), hybrid=True)
        if cpu_reference["metadata"] != hybrid64["metadata"]:
            raise ValueError("Hybrid training metadata differs from the archived CPU reference")
        checkpoint = snapshots[64]
        report["hybrid64"] = dict(training=hybrid64,
                                  archived_cpu_speedup=cpu_reference["train_seconds"] / hybrid64["elapsed_seconds"],
                                  comparison=_compare(archived_cpu, checkpoint,
                                                      cpu_reference["log"], hybrid64["log"]))
        publish("hybrid64_complete")
        options = large_runner._score_options("dev")
        if options["stream_ids"] != (1100, 1101):
            raise ValueError("Only development streams 1100 and 1101 are permitted here")
        start = time.perf_counter()
        score = core.score_model(baseline.SOURCE_SEED, checkpoint["model"], **options)
        cpu_score = baseline.read(baseline.REFERENCE / "score.json")
        report["development_fitness"] = dict(
            options=options, scoring_device="cpu", cpu=baseline.dev_metrics(cpu_score),
            hybrid=baseline.dev_metrics(score), hybrid_minus_cpu=score["fitness"] - cpu_score["fitness"],
            hybrid_checkpoint_cpu_eval_seconds=time.perf_counter() - start)
        publish("development_complete")
    else:
        report["hybrid64"] = dict(status="skipped", required_short_speedup=MIN_LONG_SPEEDUP)

    torch.cuda.empty_cache()
    solo, _, _, _ = _measure(genome, SHORT_STEPS, (0, SHORT_STEPS), hybrid=True)
    torch.cuda.empty_cache()
    report["parallel"] = dict(single_worker=solo, runs=[])
    last_increment_per_worker = solo["torch_peak_reserved_mib"] + 384
    for workers in (2, 4):
        snapshot = baseline.gpu_snapshot()
        estimated_mib = snapshot["system_used_mib"] + workers * last_increment_per_worker if snapshot else None
        if estimated_mib is None or estimated_mib + VRAM_HEADROOM_MIB >= total_mib:
            report["parallel"]["runs"].append(dict(workers=workers, status="skipped",
                                                   estimated_system_vram_mib=estimated_mib,
                                                   headroom_mib=VRAM_HEADROOM_MIB))
            break
        result = _parallel(SHORT_STEPS, workers)
        result.update(estimated_system_vram_mib=estimated_mib,
                      throughput_vs_single=result["aggregate_updates_per_second"] / solo["updates_per_second"])
        report["parallel"]["runs"].append(result)
        publish(f"parallel_{workers}_complete")
        if result["system_vram_sample_peak_mib"] is not None:
            measured_increment = (result["system_vram_sample_peak_mib"] -
                                  result["gpu_before"]["system_used_mib"]) / workers
            last_increment_per_worker = max(last_increment_per_worker, measured_increment)

    report.update(status="complete", finished_utc=baseline.utc(), background_end=_background(args.pipeline_pid),
                  gpu_end=baseline.gpu_snapshot(),
                  caveat="Short trials repeat one genome. CPU load and display VRAM usage fluctuate. "
                         "The 64-step CPU timing is archival. Sampled VRAM can miss transient peaks.")
    publish("complete")


if __name__ == "__main__":
    main()
