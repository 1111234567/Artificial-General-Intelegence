"""Compare the sealed seed-0 training path across CPU and strict FP32 CUDA."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import platform
import statistics
import subprocess
import time

import torch

from experiments.learned_binding import common, pretrain
from experiments.learned_binding.frozen_trace import TRAINABLE_NAMES
from experiments.seed_evolution import core, large_runner


SOURCE_SEED = 0
INIT_SEED = 710000
REFERENCE = large_runner.OUT / "search/rep_0/shared_g00_i00"
OUTPUT = core.ROOT / "data/seed-evolution/runtime-2026-11-11/gpu-benchmark.json"
MIB = 1024 ** 2


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def strict_fp32():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
            timeout=5,
        )
        driver, used, utilization = result.stdout.strip().splitlines()[0].split(", ")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return dict(driver_version=driver, system_used_mib=int(used), utilization_percent=int(utilization))


def measure_cuda(genome, steps):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, _ = torch.cuda.mem_get_info()
    start_utc, start = utc(), time.perf_counter()
    result = core.inner_train(SOURCE_SEED, genome, steps=steps, checkpoints=(0, steps), device="cuda")
    torch.cuda.synchronize()
    end = time.perf_counter()
    free_after, _ = torch.cuda.mem_get_info()
    report = dict(started_utc=start_utc, elapsed_seconds=end - start, updates_per_second=steps / (end - start),
                  torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / MIB,
                  torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / MIB,
                  cuda_free_before_mib=free_before / MIB, cuda_free_after_mib=free_after / MIB,
                  metadata=result["metadata"], log=result["log"])
    return report, result["checkpoints"][steps], start, end


def reference():
    candidate = read(REFERENCE / "candidate.json")
    if (candidate["source_seed"], candidate["init_seed"]) != (SOURCE_SEED, INIT_SEED):
        raise ValueError("Unexpected CPU reference genome identity")
    for name, digest in candidate["artifacts"].items():
        if common.sha(REFERENCE / name) != digest:
            raise ValueError(f"CPU reference artifact hash mismatch for {name}")
    genome = load(REFERENCE / "initial.pt")
    if core.genome_hash(genome) != candidate["genome_hash"]:
        raise ValueError("CPU reference genome hash mismatch")
    training = read(REFERENCE / "training.json")
    training.update(reference_path=str(REFERENCE.relative_to(core.ROOT)),
                    training_finished_local=datetime.fromtimestamp(
                        (REFERENCE / "training.json").stat().st_mtime).astimezone().isoformat(),
                    eval_seconds=candidate["eval_seconds"], artifacts=candidate["artifacts"],
                    timing_context="Archived CPU worker timing from an earlier part of the pipeline. "
                                   "Background load, worker cache state and startup differ.")
    return genome, load(REFERENCE / "inner64.pt"), training


def tensor_errors(left, right):
    if left.keys() != right.keys():
        raise ValueError("Tensor keys differ")
    rows = {}
    for name, before in left.items():
        after = right[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Tensor metadata differs for {name}")
        errors = (before.double() - after.double()).abs()
        rows[name] = dict(scalars=errors.numel(), nonzero=int(torch.count_nonzero(errors)),
                          max_abs=float(errors.max()), sum_abs=float(errors.sum()))
    scalars = sum(row["scalars"] for row in rows.values())
    return dict(scalars=scalars, nonzero=sum(row["nonzero"] for row in rows.values()),
                max_abs=max(row["max_abs"] for row in rows.values()),
                mean_abs=sum(row["sum_abs"] for row in rows.values()) / scalars,
                by_tensor={name: dict(scalars=row["scalars"], nonzero=row["nonzero"],
                                     max_abs=row["max_abs"], mean_abs=row["sum_abs"] / row["scalars"])
                           for name, row in rows.items()})


def compare_checkpoints(cpu, gpu, cpu_log, gpu_log):
    cpu_opt, gpu_opt = cpu["optimizer"], gpu["optimizer"]
    if cpu_opt["state"].keys() != gpu_opt["state"].keys():
        raise ValueError("Optimizer parameter indices differ")
    result = dict(model=tensor_errors(cpu["model"], gpu["model"]),
                  trainable=tensor_errors({k: cpu["model"][k] for k in sorted(TRAINABLE_NAMES)},
                                          {k: gpu["model"][k] for k in sorted(TRAINABLE_NAMES)}),
                  config_matches=cpu["model_config"] == gpu["model_config"],
                  optimizer_groups_match=cpu_opt["param_groups"] == gpu_opt["param_groups"],
                  updates_match=cpu["updates"] == gpu["updates"])
    for field in ("step", "exp_avg", "exp_avg_sq"):
        result[field] = tensor_errors({i: row[field] for i, row in cpu_opt["state"].items()},
                                      {i: row[field] for i, row in gpu_opt["state"].items()})
    result["log_cuda_minus_cpu"] = []
    for left, right in zip(cpu_log, gpu_log, strict=True):
        if (left["update"], left["stream"]) != (right["update"], right["stream"]):
            raise ValueError("Training log updates differ")
        result["log_cuda_minus_cpu"].append(dict(update=left["update"], stream=left["stream"],
            **{k: right[k] - left[k] for k in ("loss", "grad_norm", "finish_accuracy")}))
    return result


def dev_metrics(score):
    return dict(fitness=score["fitness"], parameter_hash=score["parameter_hash"],
                streams=[dict(stream_id=row["stream_id"], fitness=row["fitness"],
                              groups={name: row["groups"][name] for name in ("A_conflict", "B_conflict")},
                              support_actions_hash=row["support"]["actions_hash"],
                              probe_actions_hash=row["probe"]["actions_hash"])
                         for row in score["streams"]])


_BARRIER = None


def init_parallel(barrier):
    global _BARRIER
    _BARRIER = barrier


def parallel_worker(steps):
    strict_fp32()
    genome = load(REFERENCE / "initial.pt")
    core._training_streams(SOURCE_SEED)
    torch.cuda.init()
    _BARRIER.wait()
    report, _, start, end = measure_cuda(genome, steps)
    return report, start, end


def concurrency(steps):
    core._training_streams(SOURCE_SEED)
    torch.cuda.empty_cache()
    solo, _, _, _ = measure_cuda(load(REFERENCE / "initial.pt"), steps)
    torch.cuda.empty_cache()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2, timeout=120)
    with ProcessPoolExecutor(max_workers=2, mp_context=context, initializer=init_parallel,
                             initargs=(barrier,)) as pool:
        jobs = [pool.submit(parallel_worker, steps) for _ in range(2)]
        rows = [job.result() for job in jobs]
    elapsed = max(end for _, _, end in rows) - min(start for _, start, _ in rows)
    return dict(single_worker=solo, two_workers=[report for report, _, _ in rows],
                overlap_seconds=max(0, min(end for _, _, end in rows) - max(start for _, start, _ in rows)),
                two_worker_wall_seconds=elapsed, two_worker_updates_per_second=2 * steps / elapsed,
                throughput_ratio_two_to_one=(2 * steps / elapsed) / solo["updates_per_second"],
                peak_reserved_sum_mib=sum(report["torch_peak_reserved_mib"] for report, _, _ in rows),
                caveat="Streams and contexts preloaded. The parent retains its idle CUDA context. "
                       "Summed per-process allocator peaks are an upper bound for their aggregate peak.")


def profile_update(genome, device):
    """Attribute one real forward/backward using module boundaries and the chain rule."""
    model = core.model_from_genome(SOURCE_SEED, genome).to(device).train()
    obs, actions = (value.to(device) for value in core._training_streams(SOURCE_SEED)[0])
    state = model.initial_state(obs.shape[1])
    times, inputs = {}, {}

    def mark(label):
        if device == "cuda":
            torch.cuda.synchronize()
        times[label] = time.perf_counter()

    def capture_executor(_module, args):
        inputs["mapped_obs"] = args[0]
        mark("executor_start")

    handles = [model.encoder.register_forward_pre_hook(lambda *_: mark("encoder_start")),
               model.encoder.register_forward_hook(lambda *_: mark("encoder_end")),
               model.executor.register_forward_pre_hook(capture_executor)]
    try:
        mark("start")
        logits, _ = model(obs, state)
        mark("executor_end")
        loss = pretrain.loss_for(logits, actions)
        mark("ce_end")
        mapped_grad, = torch.autograd.grad(loss, inputs["mapped_obs"])
        mark("executor_backward_end")
        inputs["mapped_obs"].backward(mapped_grad)
        mark("scan_backward_end")
    finally:
        for handle in handles:
            handle.remove()
    intervals = dict(encoder_forward=("encoder_start", "encoder_end"),
                     heads_scan_translate_forward=("encoder_end", "executor_start"),
                     executor_forward=("executor_start", "executor_end"),
                     ce_forward=("executor_end", "ce_end"),
                     ce_executor_backward=("ce_end", "executor_backward_end"),
                     heads_scan_translate_backward=("executor_backward_end", "scan_backward_end"),
                     forward_backward_total=("start", "scan_backward_end"))
    return dict(device=device, obs_shape=list(obs.shape), loss=float(loss.detach()),
                seconds={name: times[end] - times[start] for name, (start, end) in intervals.items()},
                caveat="One forward/backward at step 1, no optimizer update. CUDA syncs at stage "
                       "boundaries include host launches. Backward is split at executor input.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-steps", type=int, choices=(0, 8, 16), default=8)
    args = parser.parse_args()
    strict_fp32()
    background = gpu_snapshot()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment")
    properties = torch.cuda.get_device_properties(0)
    genome, cpu, cpu_report = reference()
    report = dict(status="running", started_utc=utc(), source_seed=SOURCE_SEED, init_seed=INIT_SEED,
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                   platform=platform.platform(), gpu=properties.name,
                                   gpu_total_mib=properties.total_memory / MIB,
                                   cpu_logical_processors=multiprocessing.cpu_count(),
                                   torch_threads=torch.get_num_threads(),
                                   gpu_before_cuda_context=background,
                                   concurrent_cpu_search="Existing pipeline with 14 CPU workers"),
                  cpu_reference=cpu_report)

    def publish(stage):
        large_runner._write(OUTPUT, report)
        print(json.dumps(dict(stage=stage, output=str(OUTPUT))), flush=True)

    publish("start")
    cuda_report, cuda, _, _ = measure_cuda(genome, 64)
    if any(cpu_report["metadata"][k] != v for k, v in cuda_report["metadata"].items() if k != "device"):
        raise ValueError("CPU and CUDA training metadata differ")
    report.update(cuda64=cuda_report, cpu_to_cuda_training_time_ratio=(
                      cpu_report["train_seconds"] / cuda_report["elapsed_seconds"]),
                  checkpoint64_comparison=compare_checkpoints(cpu, cuda, cpu_report["log"], cuda_report["log"]),
                  gpu_after_single_cuda_worker=gpu_snapshot())
    publish("cuda64_complete")

    options = large_runner._score_options("dev")
    if options["stream_ids"] != (1100, 1101):
        raise ValueError("Benchmark must use only dev streams 1100 and 1101")
    cpu_score = read(REFERENCE / "score.json")
    start = time.perf_counter()
    gpu_score = core.score_model(SOURCE_SEED, cuda["model"], **options)
    report["development_fitness"] = dict(options=options, score_device="cpu", cpu=dev_metrics(cpu_score),
                                         cuda=dev_metrics(gpu_score),
                                         cuda_checkpoint_cpu_eval_seconds=time.perf_counter() - start,
                                         cuda_minus_cpu=gpu_score["fitness"] - cpu_score["fitness"])
    publish("dev_complete")

    report["single_update_profile"] = [profile_update(genome, device) for device in ("cpu", "cuda")]
    publish("profile_complete")
    if args.parallel_steps:
        report["cuda_concurrency"] = concurrency(args.parallel_steps)
    report.update(status="complete", finished_utc=utc(), gpu_final=gpu_snapshot())
    publish("complete")


if __name__ == "__main__":
    main()
