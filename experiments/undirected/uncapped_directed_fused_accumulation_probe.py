"""Compare FP16 and FP32 accumulation in the production fused directed rollout."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import io
import json
from pathlib import Path
from statistics import fmean
from threading import RLock
import time

import torch

from experiments.seed_evolution.large_runner import _utc, _write

from . import uncapped_directed_reward as backend
from . import uncapped_directed_reward_run as run
from .evolved_plasticity_world import N_SYMBOLS
from .uncapped_cuda_backend import CUDAGraphUnavailable
from .uncapped_directed_continuation_audit import _check_tapes
from .uncapped_directed_reward_fused_run import _FusedTrainingEvaluation
from .uncapped_island_crossover_run import block_ticks
from .uncapped_ready_signal_run import schedule
from .uncapped_reward_precision_probe import interval


OUT = run.OUT.with_name("2026-11-17-directed-fused-accumulation-probe-v1")
BATCH_SIZE, ACTORS, VISITS = 32, 16, 4
SPEED_SEEDS = range(84_291_000, 84_291_008)
QUALITY_SEEDS = range(84_291_100, 84_291_108)
FORMATS = {"fp16_accum": True, "fp32_accum": False}
T_CRITICAL = 2.364624  # Two-sided 95% Student interval, seven degrees of freedom.
FIELDS = (
    "first_finish_correct_first_return", "first_finish_correct_a_first_return",
    "first_finish_correct_b_first_return", "strict", "strict_both_return", "completion",
    "ready_coverage", "reward_events", "attempted_writes", "changed_writes",
)
_OVERRIDE_LOCK = RLock()


@contextmanager
def _backend_accumulation(device, fp16_accumulation):
    device = torch.device(device)
    with _OVERRIDE_LOCK:
        original = backend._matmul_precision
        if device.type == "cuda":
            if fp16_accumulation and (
                torch.version.cuda is None
                or not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")
                or not torch.cuda.is_available()
                or torch.cuda.get_device_capability(device) < (7, 0)
            ):
                raise CUDAGraphUnavailable(f"{device} lacks full FP16 CUDA accumulation support")

            def accumulation(dtype, default):
                return original(dtype, fp16_accumulation)

            backend._matmul_precision = accumulation
        try:
            yield
        finally:
            backend._matmul_precision = original


class FusedAccumulationEvaluation(_FusedTrainingEvaluation):
    """Use one production evaluator and capture cache per accumulation setting."""

    def __init__(self, schedule, *, fp16_accumulation):
        if type(fp16_accumulation) is not bool:
            raise TypeError("fp16_accumulation must be a bool")
        super().__init__(schedule)
        self._fp16_accumulation = fp16_accumulation

    def __call__(self, *args, **kwargs):
        with _backend_accumulation(kwargs["device"], self._fp16_accumulation):
            return super().__call__(*args, **kwargs)


class _LongAccumulationEvaluation(backend.RewardEvaluation):
    """Keep the same precision switch without the production request-clock shortcut."""

    def __init__(self, schedule, *, fp16_accumulation):
        super().__init__(schedule)
        self._fp16_accumulation = fp16_accumulation

    def __call__(self, *args, **kwargs):
        with _backend_accumulation(kwargs["device"], self._fp16_accumulation):
            return super().__call__(*args, **kwargs)


def load_elites(run_dir):
    committed = [folder for folder in (run_dir / "generations").iterdir()
                 if folder.is_dir() and folder.name.isdecimal() and (folder / "summary.json").is_file()]
    if not committed:
        raise ValueError(f"No committed elite checkpoint in {run_dir}")
    folder = max(committed, key=lambda path: int(path.name))
    checkpoint_path = folder / "elites.pt"
    data = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    pools = checkpoint["islands"]
    per_island = BATCH_SIZE // run.ISLANDS
    if len(pools) != run.ISLANDS or any(len(pool) < per_island for pool in pools):
        raise ValueError(f"Expected at least {per_island} elites per island in {checkpoint_path}")
    selected = [(island, row["id"]) for island, pool in enumerate(pools)
                for row in sorted(pool, key=run.rank, reverse=True)[:per_island]]
    if len({name for _, name in selected}) != BATCH_SIZE:
        raise ValueError(f"Expected {BATCH_SIZE} distinct elite IDs in {checkpoint_path}")
    genomes, candidates = [], []
    for island, name in selected:
        source = checkpoint["genomes"][name]
        if source["node_bias"].numel() != run.NODES:
            raise ValueError(f"Expected {run.NODES} nodes in elite {name}")
        candidates.append({"id": name, "island": island, "source_directed": bool(source["directed"])})
        genomes.append({**source, "directed": torch.tensor(True, dtype=torch.bool)})
    return genomes, {
        "run_dir": str(run_dir), "checkpoint": str(checkpoint_path),
        "generation": int(folder.name), "sha256": hashlib.sha256(data).hexdigest(),
        "candidates": candidates,
    }


def _profile_kernels(operation, device):
    """Collect CUDA kernel names outside the timed comparison, when Kineto supports it."""
    try:
        if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            return {"status": "CUDA profiling unavailable"}
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as profile:
            operation()
            torch.cuda.synchronize(device)
        kernels = {}
        for event in profile.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernel = kernels.setdefault(event.name, {"launches": 0, "total_us": 0.})
                kernel["launches"] += 1
                kernel["total_us"] += event.device_time_total
        return {"status": "ok" if kernels else "no CUDA activity collected", "kernels": kernels}
    except (AttributeError, RuntimeError, OSError) as error:
        return {"status": "CUDA profiling failed", "error": f"{type(error).__name__}: {error}"}


@torch.no_grad()
def check_matmul_path(genomes, device):
    _, birth = backend._stack_directed_genomes(genomes)
    birth = {name: tensor.to(device) for name, tensor in birth.items()}
    policy = backend._DirectedRewardPolicy(birth, ACTORS, device, "full")
    rng = torch.Generator(device=device).manual_seed(84_291_200)
    policy.activity.copy_(torch.randn(policy.activity.shape, device=device, generator=rng).tanh_())
    matrix = policy.matrix.flatten(0, 1)
    activity = policy.activity.flatten(0, 1).unsqueeze(-1)
    values, profiles = {}, {}
    original = backend._matmul_precision
    for name, flag in FORMATS.items():
        with original(torch.float16, flag):
            values[name] = torch.bmm(matrix, activity)
            torch.cuda.synchronize(device)
            profiles[name] = _profile_kernels(partial(torch.bmm, matrix, activity), device)
    half, single = (values[name].float() for name in FORMATS)
    if not (half.isfinite().all() and single.isfinite().all()):
        raise ValueError("Nonfinite results in the exact-shape bmm witness")
    gap = (half - single).abs()
    return {
        "matrix_shape": list(matrix.shape), "matrix_stride": list(matrix.stride()),
        "activity_shape": list(activity.shape), "activity_stride": list(activity.stride()),
        "values": gap.numel(), "changed_values": int(gap.count_nonzero().item()),
        "mean_absolute_difference": gap.mean().item(), "max_absolute_difference": gap.max().item(),
        "kernels": profiles,
    }


def _evaluators(evaluator_type, tape):
    return {name: evaluator_type(tape, fp16_accumulation=flag) for name, flag in FORMATS.items()}


def _call(evaluator, genomes, device, seed):
    return evaluator(genomes, seed=seed, actors=ACTORS, visits=VISITS, device=device)


def _order(index):
    return tuple(FORMATS) if index % 2 == 0 else tuple(reversed(FORMATS))


def speed(evaluators, genomes, device):
    for name in (*FORMATS, *reversed(FORMATS)):
        _call(evaluators[name], genomes, device, SPEED_SEEDS.start)
        torch.cuda.synchronize(device)

    blocks = []
    for index, seed in enumerate(SPEED_SEEDS):
        first, second = _order(index)
        order = (first, second, second, first)
        block = {"seed": seed, "order": order, **{name: [] for name in FORMATS}}
        for name in order:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _call(evaluators[name], genomes, device, seed)
            torch.cuda.synchronize(device)
            block[name].append(time.perf_counter() - start)
        blocks.append(block)
        elapsed = " ".join(f"{name}={fmean(block[name]):.4f}s" for name in FORMATS)
        print(f"speed block {index + 1}/{len(SPEED_SEEDS)} {elapsed}", flush=True)

    means = {name: fmean(seconds for block in blocks for seconds in block[name]) for name in FORMATS}
    ratios = [fmean(block["fp32_accum"]) / fmean(block["fp16_accum"]) for block in blocks]
    return {
        "mean_batch_seconds": means,
        "candidates_per_second": {name: len(genomes) / seconds for name, seconds in means.items()},
        "paired_fp16_accum_speedup": interval(ratios, T_CRITICAL), "blocks": blocks,
    }


def quality(evaluators, genomes, device, condition):
    paired = []
    for index, seed in enumerate(QUALITY_SEEDS):
        order = _order(index)
        pair = {"seed": seed, "order": order}
        for name in order:
            values = _call(evaluators[name], genomes, device, seed)
            metrics = {field: values[field].cpu().tolist() for field in FIELDS}
            for events, coverage, changed, attempted in zip(
                metrics["reward_events"], metrics["ready_coverage"], metrics["changed_writes"],
                metrics["attempted_writes"], strict=True,
            ):
                if events != coverage * (ACTORS * N_SYMBOLS * VISITS) or changed > attempted:
                    raise ValueError(f"Reward telemetry mismatch at seed {seed} for {condition}/{name}")
            pair[name] = metrics
        paired.append(pair)
        print(f"quality {condition} {index + 1}/{len(QUALITY_SEEDS)}", flush=True)

    means = {name: {field: fmean(fmean(pair[name][field]) for pair in paired) for field in FIELDS}
             for name in FORMATS}
    deltas = {field: interval([fmean(a - b for a, b in zip(pair["fp16_accum"][field],
                                                          pair["fp32_accum"][field], strict=True))
                               for pair in paired], T_CRITICAL) for field in FIELDS}
    return {"formats": means, "fp16_minus_fp32_accum": deltas, "paired": paired}


def _replay_kernels(evaluators, genomes, device, witness):
    replays = {}
    for name, evaluator in evaluators.items():
        known = witness["kernels"][name].get("kernels", {})
        replay = _profile_kernels(partial(_call, evaluator, genomes, device, SPEED_SEEDS.start), device)
        kernels = replay.pop("kernels", {})
        replay["total_cuda_events"] = sum(kernel["launches"] for kernel in kernels.values())
        replay["gemm_kernels"] = {kernel: stats for kernel, stats in kernels.items()
                                  if kernel in known or any(word in kernel.lower()
                                                            for word in ("gemm", "gemv", "matmul"))}
        replay["matches_witness"] = sorted(set(known) & set(kernels))
        replays[name] = replay
    return replays


def _progress(run_dir):
    path = run_dir / "progress.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    return {key: state.get(key) for key in ("state", "phase", "generation", "generation_completed", "checkpoint")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run containing committed elite checkpoints.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    run_dir, out = args.run_dir.resolve(), args.out.resolve()
    if not (run_dir / "generations").is_dir():
        parser.error(f"Run has no generations directory: {run_dir}")
    if (out / "protocol.json").exists() or (out / "results.json").exists():
        parser.error("Use a fresh output directory.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required.")
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    genomes, source = load_elites(run_dir)
    print(f"Loaded {len(genomes)} elites from committed generation {source['generation']}.", flush=True)
    with _backend_accumulation(device, True):
        pass

    regular = _evaluators(FusedAccumulationEvaluation, partial(schedule, timing=run.TIMING))
    long = _evaluators(_LongAccumulationEvaluation, partial(schedule, timing=run.DELAYED))
    _check_tapes({
        "regular_native": (regular["fp32_accum"], run.TIMING, block_ticks(run.TIMING)),
        "long": (long["fp32_accum"], run.DELAYED, block_ticks(run.DELAYED)),
    }, QUALITY_SEEDS, ACTORS, VISITS)
    source_paths = (Path(__file__), Path(backend.__file__), *(
        Path(__file__).with_name(f"uncapped_directed_reward_{suffix}")
        for suffix in ("fused_run.py", "fast_run.py", "cuda_fused.py", "cuda_fused.cu", "run.py")
    ), Path(__file__).with_name("uncapped_reward_precision.py"))
    _write(out / "protocol.json", {
        "started_utc": _utc(), "source": source, "candidates": BATCH_SIZE, "actors": ACTORS, "visits": VISITS,
        "nodes": run.NODES, "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "formats": FORMATS, "state_dtype": "torch.float16",
        "allow_fp16_reduced_precision_reduction": False,
        "timing": {"regular": run.TIMING, "long": run.DELAYED},
        "speed_seeds": list(SPEED_SEEDS), "quality_seeds": list(QUALITY_SEEDS),
        "speed": "Production fused training and four world blocks, synchronized warm CUDA Graph replay wall time, "
                 "alternating ABBA and BAAB; both precisions use the same seed in each block.",
        "quality": "Paired heldout world seeds with the same contracts across timings. Regular uses fused static "
                   "receipt skipping; long uses the reference directed evaluator for its different clock.",
        "interval": "Two-sided 95% Student interval over eight paired seed blocks for these fixed elites.",
        "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    })
    result_path = out / "results.json"
    training = _evaluators(FusedAccumulationEvaluation, run.training_schedule)
    timing_before = _progress(run_dir)
    throughput = speed(training, genomes, device)
    throughput.update(training_before=timing_before, training_after=_progress(run_dir))
    result = {"state": "speed_complete", "speed": throughput, "quality": {}}
    _write(result_path, result)
    try:
        witness = check_matmul_path(genomes, device)
        print(f"Exact-shape bmm changed {witness['changed_values']}/{witness['values']} outputs.", flush=True)
        result["accumulation_witness"] = witness
        result["production_replay_kernels"] = _replay_kernels(training, genomes, device, witness)
        _write(result_path, result)
    finally:
        for evaluator in training.values():
            evaluator.clear_cache()

    for condition, evaluators in (("regular", regular), ("long", long)):
        try:
            result["quality"][condition] = quality(evaluators, genomes, device, condition)
            _write(result_path, result)
        finally:
            for evaluator in evaluators.values():
                evaluator.clear_cache()
    result.update(state="complete", completed_utc=_utc())
    _write(result_path, result)
    print(json.dumps({"speed": throughput["candidates_per_second"],
                      "paired_fp16_accum_speedup": throughput["paired_fp16_accum_speedup"]}), flush=True)


if __name__ == "__main__":
    main()
