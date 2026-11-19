"""Paired benchmarks of the detailed and fused physical scores on frozen candidates."""

import argparse
import copy
import ctypes
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

# Each CPU worker is one rollout process with one numerical-library thread.
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
import torch

from experiments.undirected._cuda_nvrtc import compile_kernels
from . import mixture_flash, mixture_world, physical_counts, symmetry_audit, symmetry_grounded, symmetry_memory_deploy
from . import symmetry_physics, symmetry_policy, symmetry_policy_train, world
from . import symmetry_world


DATA = Path(__file__).resolve().parents[2] / "data" / "pulse-game"
CHECKPOINT = DATA / "symmetry-policy-v1" / "public-learned-s7243--full-e8719" / "checkpoint.json"
ORDER_SEED = 46389837
WORLD_SEED = 82140743
POPULATION_SEED = 14238053
INT8 = torch.int8


@dataclass(frozen=True)
class Work:
    seeds: list[int]
    settings: tuple
    coords: np.ndarray


def digest(*values):
    result = hashlib.sha256()
    for value in values:
        if isinstance(value, Path):
            result.update(value.read_bytes())
        elif isinstance(value, str):
            result.update(value.encode())
        else:
            array = np.ascontiguousarray(value)
            result.update(str((array.shape, array.dtype.str)).encode())
            result.update(array.tobytes())
    return result.hexdigest()


def load_inputs(checkpoint):
    fit = json.loads(checkpoint.read_text())
    parent = DATA / "symmetry-grounded-v1" / (fit["settings"]["parent"] + ".pt")
    if digest(parent) != fit["settings"]["parent_sha"]:
        raise ValueError("The frozen genotype does not match the parent checkpoint")
    return fit, parent, symmetry_grounded.load(parent)


def works(fit, width, pair_count, stages):
    """Draw frozen antithetic populations using saved centers and radii and fresh RNG streams."""
    mu, sig, elite = (np.asarray(fit[k]) for k in ("mu", "sig", "elite"))
    islands, dim = mu.shape
    per_island = width // islands
    if width % islands or per_island < 2 or per_island % 2:
        raise ValueError("Population must comprise four equal islands of even size")
    rng = np.random.default_rng(POPULATION_SEED)
    env = np.random.default_rng(WORLD_SEED)
    result = []
    for i in range(stages):
        candidates = []
        for m, s, e in zip(mu, sig, elite, strict=True):
            direction = rng.normal(size=((per_island - 2) // 2, dim))
            pop = np.concatenate((e[None], m[None], m + s * direction, m - s * direction))
            bound = 6.0
            candidates.append(bound - np.abs((pop + bound) % (4 * bound) - 2 * bound))
        count = int(env.choice([8, 12, 16]))
        span = int(env.choice([2, 4, 4, 6, 8]))
        gap = int(env.choice([0, 0, 1, 2, 4, 6, 8]))
        seeds = env.choice(460_000_000, size=pair_count, replace=False) + 10_000_000
        config = (count, span, gap, float(env.uniform(0, .12)), float(env.uniform(.035, .16)))
        if i == 0:
            config = (12, 4, 2, .06, .10)
        result.append(Work(seeds.tolist(), config, np.concatenate(candidates)))
    return result


def prepare(task):
    start = perf_counter()
    episode = symmetry_policy_train.games(task.seeds, *task.settings)
    return episode, perf_counter() - start


def world_fingerprint(episode):
    menu = episode.menu
    return digest(menu.body.seeds, menu.chemicals, menu.body.ctx, menu.body.distances,
                  menu.body.wind, menu.realized, episode.query, [episode.exposed])


def score_numerators(rewards, mask):
    selected = rewards[..., mask]
    first = selected[..., 0] > 0
    half = first.shape[-1] // 2
    components = first[..., :half], first[..., half:]
    return np.stack((components[0].astype(int) + components[1], components[0] & components[1],
                     (selected[..., :half, :] > 0).sum(-1) + (selected[..., half:, :] > 0).sum(-1),
                     (selected[..., :half, :] != 0).sum(-1) + (selected[..., half:, :] != 0).sum(-1)), -1)


class Dispatch:
    """Timers reuse identical device-resident input and output allocations."""

    def __init__(self, cohort, pop, reference_source):
        self.cohort = cohort
        self.decoded = symmetry_physics.tensor(pop)
        self.mask = symmetry_physics.tensor(cohort.world.query[0].astype(np.uint8))
        actors, trials = cohort.shape
        self.shape = len(pop), actors, trials
        self.pairs = actors // 2
        self.base = cohort.features, *cohort.ports.body, *([cohort.empty] * 5)
        self.end = self.pairs, trials, cohort.world.exposed, cohort.rtype, 0, self.decoded
        self.old = symmetry_policy.program(cohort.params["latent"], cohort.params["capacity"], pop.shape[1])["policy_search"]
        self.warp_source = physical_counts.warp_normalize(reference_source)
        self.new = physical_counts.compiled(self.warp_source)
        self.detail = [torch.empty(self.shape, device="cuda", dtype=INT8) for _ in range(3)]
        self.detail.append(torch.empty(self.shape, device="cuda", dtype=torch.float32))
        self.counts = torch.empty((len(pop), self.pairs, 4), device="cuda", dtype=torch.int32)

    def detailed(self):
        self.old((self.pairs, self.shape[0]), (64,), *self.base, *self.detail, *self.end)

    def fused(self):
        self.new((self.pairs, self.shape[0]), (64,), *self.base, self.counts, self.mask, *self.end)

    @staticmethod
    def trace_source(source):
        """Add trace ports without changing the exact string edits used in production fusion."""
        edits = {
            "float* gains,": "float* gains,unsigned char* pulse_trace,float* ends,",
            "    if(active){": "    pulse_trace[oi*16+t]=(unsigned char)(motor[branch][0]+2.f*motor[branch][1]);\n    if(active){",
            "if(lane==0){side_out[oi]": "if(lane==0){ends[oi*2]=position[branch];ends[oi*2+1]=velocity[branch];}\n  if(lane==0){side_out[oi]",
        }
        for old, new in edits.items():
            if source.count(old) != 1:
                raise ValueError(f"Changed diagnostic insertion point {old!r}")
            source = source.replace(old, new)
        return source

    def trace_outputs(self):
        pulse = torch.empty((*self.shape, world.TICKS), device="cuda", dtype=torch.uint8)
        end = torch.empty((*self.shape, 2), device="cuda", dtype=torch.float32)
        return pulse, end

    def trace(self, source):
        traced = compile_kernels(self.trace_source(source), ("policy_search",), device=torch.device("cuda:0"),
                                 options=("--fmad=false",))["policy_search"]
        pulse, end = self.trace_outputs()
        detail = [torch.empty_like(v) for v in self.detail]
        traced((self.pairs, self.shape[0]), (64,), *self.base, *detail, pulse, end, *self.end)
        return tuple(v.cpu().numpy() for v in (*detail, pulse, end))

    def trace_fused(self):
        traced = physical_counts.compiled(self.trace_source(self.warp_source))
        pulse, end = self.trace_outputs()
        counts = torch.empty_like(self.counts)
        traced((self.pairs, self.shape[0]), (64,), *self.base, counts, self.mask, pulse, end, *self.end)
        return tuple(v.cpu().numpy() for v in (counts, pulse, end))


def measured_cuda(action, repeats):
    start, finish = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Allocate CUDA events outside the timed range and ensure no preceding
    # workload on this process's stream contributes to either endpoint.
    start.record()
    finish.record()
    finish.synchronize()
    host_start = perf_counter()
    start.record()
    for _ in range(repeats):
        action()
    finish.record()
    finish.synchronize()
    return {"seconds": start.elapsed_time(finish) / (1000 * repeats),
            "host_seconds": (perf_counter() - host_start) / repeats}


def compare_arrays(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch {actual.shape} != {expected.shape}")
    neq = actual != expected
    return {"different": int(neq.sum()), "elements": int(actual.size),
            "max_abs": float(np.abs(actual.astype(float) - expected.astype(float)).max(initial=0)),
            "first_different": np.argwhere(neq)[:8].tolist(),
            "actual_sha256": digest(actual), "expected_sha256": digest(expected)}


def native_result(params, episode, trace):
    agent = symmetry_memory_deploy.Forager(params, episode.menu.body.seeds, boundary=episode.exposed)
    result = mixture_flash.rollout(agent, episode.menu, record=trace)
    data = (result.contact_side, result.contact_tick, result.reward)
    if trace:
        posvel = result.obs[:, :, -1, [world.POSITION, world.VELOCITY]]
        pulse = result.pulses[..., 0].astype(np.uint8) + 2 * result.pulses[..., 1].astype(np.uint8)
        return (*data, np.stack(agent.record, axis=1), pulse, posvel)
    return score_numerators(result.reward, episode.query[0])


def ordered_blocks(blocks, a, b, measure, seed=ORDER_SEED):
    """One balanced four-period crossover per block; reverse order in half the blocks."""
    if blocks < 2 or blocks % 2:
        raise ValueError("Use an even positive number of paired blocks")
    order = ["ABBA"] * (blocks // 2) + ["BAAB"] * (blocks // 2)
    np.random.default_rng(seed).shuffle(order)
    actions = {"A": a, "B": b}
    return [{"sequence": sequence, "positions": [measure(actions[label]) for label in sequence]}
            for sequence in order]


def summarize(blocks):
    totals = []
    for block in blocks:
        positions = block["positions"]
        result = [np.mean([m["seconds"] for s, m in zip(block["sequence"], positions, strict=True)
                           if s == label]) for label in "AB"]
        totals.append(result)
    seconds = np.asarray(totals)
    log_ratio = np.log(seconds[:, 0]) - np.log(seconds[:, 1])
    grouped = {seq: [v for v, b in zip(log_ratio, blocks, strict=True) if b["sequence"] == seq]
               for seq in sorted({b["sequence"] for b in blocks})}
    rng = np.random.default_rng(10987943)
    boot = np.hstack([rng.choice(values, size=(20_000, len(values))) for values in grouped.values()])
    order_effect = {seq: float(np.exp(np.mean(values))) for seq, values in grouped.items()}
    single = np.asarray([m["seconds"] for b in blocks for m in b["positions"]])
    return {"paired_blocks": len(blocks), "speedup_a_over_b": float(np.exp(log_ratio.mean())),
            "speedup_95_ci": np.exp(np.quantile(boot.mean(1), [.025, .975])).tolist(),
            "speedup_by_order": order_effect,
            "measured_seconds_per_call_range": [float(single.min()), float(single.max())],
            "median_seconds_a_b": np.median(seconds, axis=0).tolist(),
            "mean_seconds_a_b": seconds.mean(axis=0).tolist(),
            "block_mean_range_a_b": np.stack((seconds.min(0), seconds.max(0)), 1).tolist()}


def run_score(dispatch, cohort, pop, arm, batch):
    if arm == "reference":
        return cohort.run_decoded(pop, dispatch.old, batch=batch)
    return physical_counts.score(cohort, pop, dispatch.warp_source, block_candidates=batch)


def measure_score(dispatch, cohort, pop, arm, batch):
    begin = perf_counter()
    result = run_score(dispatch, cohort, pop, arm, batch)
    elapsed = perf_counter() - begin
    return {"seconds": elapsed, "batch": batch, "scores_sha256": digest(result)}


def pipeline(worklist, params, mode, dispatch, arm, prefetch):
    """Inclusive time from queue creation through its final scoring/ranking and drain."""
    start = perf_counter()
    stages = []
    score_rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        buffer = {i: pool.submit(prepare, worklist[i]) for i in range(min(prefetch, len(worklist)))}
        for i, task in enumerate(worklist):
            step_start = perf_counter()
            episode, host_build = buffer.pop(i).result()
            wait = perf_counter() - step_start
            if i + prefetch < len(worklist):
                buffer[i + prefetch] = pool.submit(prepare, worklist[i + prefetch])
            device_start = perf_counter()
            cohort = symmetry_policy.Cohort(params, episode)
            pop = symmetry_policy.decoded(params, task.coords, mode)
            setup = perf_counter() - device_start
            # The production batch sizes differ, but everything up to scoring is shared.
            score_start = perf_counter()
            scores = run_score(dispatch, cohort, pop, arm, 64 if arm == "reference" else 2048)
            ranking = symmetry_policy_train.utility(scores.reshape(4, -1, 4))
            np.argsort(-ranking, axis=1, kind="stable")
            score_rows.append(scores)
            stages.append({"host_world_seconds": host_build, "host_world_wait_seconds": wait,
                           "cohort_and_decode_seconds": setup, "scores_and_rank_seconds": perf_counter() - score_start,
                           "actors": cohort.shape[0], "trials": cohort.shape[1]})
    elapsed = perf_counter() - start
    return {"seconds": elapsed, "generations": stages, "scores_sha256": digest(*score_rows)}


CPU_INIT = None


def init_cpu_worker(params, episode, mode):
    global CPU_INIT
    torch.set_num_threads(1)
    CPU_INIT = params, episode, mode


def cpu_worker(coords):
    params, episode, mode = CPU_INIT
    phenotype = symmetry_policy.materialize(params, coords, mode)
    return os.getpid(), native_result(phenotype, episode, trace=False)


def measure_cpu(iterator, candidates):
    start = perf_counter()
    results = list(iterator(cpu_worker, candidates))
    seconds = perf_counter() - start
    pids, scores = zip(*results, strict=True)
    return {"seconds": seconds, "worker_pids": sorted(set(pids)), "numerators_sha256": digest(np.stack(scores))}


def host_times():
    if platform.system() != "Windows":
        return None
    clocks = tuple(ctypes.c_ulonglong() for _ in range(3))
    if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(c) for c in clocks)):
        raise ctypes.WinError()
    return [c.value for c in clocks]


def monitor_snapshot():
    query = ("name,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total,"
             "clocks.sm,clocks.mem,power.draw")
    props = {}
    for name, fields in (("device", ("--query-gpu=" + query,)),
                         ("compute_processes", ("--query-compute-apps=pid,process_name,used_memory",))):
        try:
            proc = subprocess.run(["nvidia-smi", *fields, "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=4, check=True)
            props[name] = proc.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            props[name] = [str(exc)]
    return {"local_time": datetime.now().astimezone().isoformat(), "nvidia_smi": props,
            "device_fields": query.split(","), "system_cpu_counter_idle_kernel_user": host_times()}


def measured_stage(name, results, method):
    before = monitor_snapshot()
    start = perf_counter()
    value = method()
    elapsed = perf_counter() - start
    after = monitor_snapshot()
    before_cpu, after_cpu = (m["system_cpu_counter_idle_kernel_user"] for m in (before, after))
    delta = np.subtract(after_cpu, before_cpu) if before_cpu is not None else None
    busy = float(1 - delta[0] / delta[1:].sum()) if delta is not None else None
    results[name] = {"elapsed_seconds": elapsed, "whole_system_cpu_busy_fraction": busy,
                     "context_before": before, "context_after": after, **value}
    print(name, f"{elapsed:.2f}s", value.get("summary", ""), flush=True)


def cpu_trials(params, mode, episode, candidates, blocks, workers):
    init_cpu_worker(params, episode, mode)
    solo_warm = measure_cpu(map, candidates)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_cpu_worker,
                             initargs=(params, episode, mode)) as pool:
        pool_warm = measure_cpu(pool.map, candidates)
        raw = ordered_blocks(blocks, map, pool.map, lambda method: measure_cpu(method, candidates))
    same = {m["numerators_sha256"] for block in raw for m in block["positions"]}
    if len(same) != 1 or pool_warm["numerators_sha256"] not in same:
        raise AssertionError("CPU process counts differ")
    return {"summary": summarize(raw), "raw": raw, "workers_parallel": workers,
            "startup_serial": solo_warm, "startup_parallel": pool_warm,
            "score_numerators_sha256": same.pop()}


def gpu_parity(params, mode, task, candidates):
    episode, _ = prepare(task)
    cohort = symmetry_policy.Cohort(params, episode)
    pop = symmetry_policy.decoded(params, candidates, mode)
    source = symmetry_policy.policy_source(params["latent"], params["capacity"], pop.shape[1])
    dispatch = Dispatch(cohort, pop, source)
    ref = dispatch.trace(source)
    opt = dispatch.trace(dispatch.warp_source)
    fused_trace_count, fused_pulse, fused_end = dispatch.trace_fused()
    native = tuple(np.stack(items) for items in zip(*(native_result(symmetry_policy.materialize(params, c, mode),
                                                                  episode, trace=True) for c in candidates), strict=True))
    dispatch.detailed()
    dispatch.fused()
    old = tuple(v.cpu().numpy() for v in dispatch.detail)
    counts = dispatch.counts.cpu().numpy().astype(int)
    mask = episode.query[0]
    numerators = [score_numerators(a[2], mask) for a in (native, ref, opt)]
    scoring = [run_score(dispatch, cohort, pop, arm, 64) for arm in ("reference", "fused")]
    fields = "side", "contact_tick", "reward", "rule_gain_start", "motor_pulse_pair", "final_position_velocity"
    return {"work": {"candidates": len(pop), "independent_pairs": len(task.seeds), "actors": cohort.shape[0],
                     "trials": cohort.shape[1], "queries": int(mask.sum()), "settings": task.settings, "seeds": task.seeds},
            "world_sha256": world_fingerprint(episode),
            "traced_reference_vs_reference": {k: compare_arrays(a, b) for k, a, b in zip(fields[:4], ref[:4], old, strict=True)},
            "warp_vs_reference": {k: compare_arrays(a, b) for k, a, b in zip(fields, opt, ref, strict=True)},
            "fused_trace_vs_reference": {"motor_pulse_pair": compare_arrays(fused_pulse, ref[4]),
                                         "final_position_velocity": compare_arrays(fused_end, ref[5])},
            "numpy_vs_reference": {k: compare_arrays(a, b) for k, a, b in zip(fields, native, ref, strict=True)},
            "fused_traced_vs_uninstrumented_counts": compare_arrays(fused_trace_count.astype(int), counts),
            "raw_fused_counts_vs_reference": compare_arrays(counts, numerators[1]),
            "raw_fused_counts_vs_warp_trace": compare_arrays(counts, numerators[2]),
            "raw_fused_counts_vs_numpy": compare_arrays(counts, numerators[0]),
            "float_score_fused_vs_reference": compare_arrays(scoring[1], scoring[0]),
            "cpu_score_numerators_sha256": digest(numerators[0])}


def matched_trials(params, mode, task, candidates, blocks, workers):
    if blocks % 4:
        raise ValueError("A balanced four-arm crossover requires multiples of four blocks")
    episode, host_setup = prepare(task)
    begin = perf_counter()
    cohort = symmetry_policy.Cohort(params, episode)
    pop = symmetry_policy.decoded(params, candidates, mode)
    source = symmetry_policy.policy_source(params["latent"], params["capacity"], pop.shape[1])
    dispatch = Dispatch(cohort, pop, source)
    torch.cuda.synchronize()
    first_setup = perf_counter() - begin
    init_cpu_worker(params, episode, mode)
    plan = ["ABDC", "BCAD", "CDBA", "DACB"] * (blocks // 4)
    np.random.default_rng(ORDER_SEED).shuffle(plan)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_cpu_worker,
                             initargs=(params, episode, mode)) as pool:
        calls = {"A": lambda: measure_cpu(map, candidates), "B": lambda: measure_cpu(pool.map, candidates),
                 "C": lambda: measure_score(dispatch, cohort, pop, "reference", 64),
                 "D": lambda: measure_score(dispatch, cohort, pop, "fused", 2048)}
        warm = {k: call() for k, call in calls.items()}
        raw = [{"sequence": order, "positions": [calls[k]() for k in order]} for order in plan]
    names = {"A": "numpy_serial", "B": "numpy_four_processes", "C": "gpu_reference", "D": "gpu_fused"}
    repeat_hashes = {name: sorted({entry.get("numerators_sha256", entry.get("scores_sha256"))
                                  for row in raw for arm, entry in zip(row["sequence"], row["positions"], strict=True)
                                  if arm == key}) for key, name in names.items()}
    dispatch.detailed()
    dispatch.fused()
    mask = episode.query[0]
    cpu_counts = np.stack([cpu_worker(c)[1] for c in candidates])
    gpu_counts = score_numerators(dispatch.detail[2].cpu().numpy(), mask)
    fused_counts = dispatch.counts.cpu().numpy().astype(int)
    parity = {"numpy_vs_old": compare_arrays(cpu_counts, gpu_counts), "fused_vs_old": compare_arrays(fused_counts, gpu_counts)}
    if any(len(v) != 1 for v in repeat_hashes.values()) or repeat_hashes[names["A"]] != repeat_hashes[names["B"]]:
        raise AssertionError(f"Matched score repeatability failed: {repeat_hashes}")
    contrast = {}
    for left, right in ("AB", "AC", "AD", "BC", "BD", "CD"):
        pairs = []
        for row in raw:
            group = dict(zip(row["sequence"], row["positions"], strict=True))
            pairs.append({"sequence": "AB" if row["sequence"].index(left) < row["sequence"].index(right) else "BA",
                          "positions": [group[k] for k in row["sequence"] if k in (left, right)]})
        contrast[f"{names[left]}_over_{names[right]}"] = summarize(pairs)
    med = {names[k]: float(np.median([row["positions"][row["sequence"].index(k)]["seconds"] for row in raw])) for k in names}
    return {"summary": med, "contrast": contrast, "raw": raw, "warm": warm, "labels": names,
            "score_parity": parity, "repeat_hashes": repeat_hashes, "world_sha256": world_fingerprint(episode),
            "cpu_workers": workers, "host_world_seconds": host_setup, "first_gpu_setup_seconds": first_setup}


def pipeline_parity(params, mode, worklist):
    rows = []
    for task in worklist:
        episode, _ = prepare(task)
        cohort = symmetry_policy.Cohort(params, episode)
        pop = symmetry_policy.decoded(params, task.coords, mode)
        source = symmetry_policy.policy_source(params["latent"], params["capacity"], pop.shape[1])
        dispatch = Dispatch(cohort, pop, source)
        dispatch.detailed()
        dispatch.fused()
        old = score_numerators(dispatch.detail[2].cpu().numpy(), episode.query[0])
        counts = dispatch.counts.cpu().numpy().astype(int)
        a, b = [run_score(dispatch, cohort, pop, arm, batch) for arm, batch in (("reference", 64), ("fused", 2048))]
        rankings = [np.argsort(-symmetry_policy_train.utility(r.reshape(4, -1, 4)), axis=1, kind="stable") for r in (a, b)]
        queries = int(episode.query[0].sum())
        total = old.sum(1).reshape(4, -1, 4)
        # Integer numerator of the fitness in symmetry_policy_train.utility.
        # The common denominator is 20 * lives * queries.
        utility = total @ np.array([15 * queries, 6 * queries, 2, 0])
        top_count = max(8, rankings[0].shape[1] // 12)
        rows.append({"settings": task.settings, "shape": cohort.shape, "world_sha256": world_fingerprint(episode),
                     "counts": compare_arrays(counts, old),
                     "float_scores": compare_arrays(b, a), "stable_island_rank": compare_arrays(*rankings[::-1]),
                     "island_winners": compare_arrays(rankings[1][:, 0], rankings[0][:, 0]),
                     "winner_exact_utility": compare_arrays(*(np.take_along_axis(utility, r[:, :1], axis=1)
                                                               for r in rankings[::-1])),
                     "top_selection_order": compare_arrays(rankings[1][:, :top_count], rankings[0][:, :top_count]),
                     "top_selection_membership": compare_arrays(*(np.sort(r[:, :top_count], axis=1) for r in rankings[::-1]))})
    return {"cases": rows}


def gpu_trials(name, args, out, params, mode, worklist, cohort, pop, dispatch):
    if name == "resident_device_event":
        actions = dispatch.detailed, dispatch.fused
        measure = lambda fn: measured_cuda(fn, args.device_repeats)
    elif name == "resident_scoring_equal_batch":
        actions = ("reference", args.equal_batch), ("fused", args.equal_batch)
        measure = lambda item: measure_score(dispatch, cohort, pop, *item)
    elif name == "resident_scoring_native_batch":
        actions = ("reference", 64), ("fused", 2048)
        measure = lambda item: measure_score(dispatch, cohort, pop, *item)
    else:
        actions = "reference", "fused"
        measure = lambda arm: pipeline(worklist, params, mode, dispatch, arm, args.prefetch)
    # Both arms have a full untimed warm-up: NVRTC and CUDA initialization
    # happen beforehand, for both the fixed world and the prefetched pipeline.
    a_warm, b_warm = (measure(item) for item in actions)
    def run():
        raw = ordered_blocks(args.blocks, *actions, measure)
        return {"summary": summarize(raw), "raw": raw, "warm_a_b": [a_warm, b_warm]}
    measured_stage(name, out, run)


def select_cpu_tasks(fit, args):
    small = works(fit, max(16, args.cpu_candidates * 2), args.cpu_pairs, 3)
    for i, task in enumerate(small):
        if i == 1:
            task = Work(task.seeds, (16, 8, 8, .11, .15), task.coords)
        if i == 2:
            task = Work(task.seeds, (8, 2, 0, .02, .04), task.coords)
        ii = np.linspace(2, len(task.coords) - 1, args.cpu_candidates, dtype=int)
        candidates = task.coords[ii].copy()
        candidates[0] = fit["best"]["genome"]
        yield task, candidates


def metadata(args, fit, parent, params):
    modules = (symmetry_policy, physical_counts, symmetry_policy_train, symmetry_physics, symmetry_memory_deploy,
               symmetry_world, symmetry_audit, mixture_flash, mixture_world, world)
    sources = [Path(__file__), *(Path(x.__file__) for x in modules),
               Path(symmetry_physics.__file__).with_name("symmetry_physics_gpu.cu")]
    return {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "checkpoint_settings": fit["settings"], "checkpoint_sha256": digest(args.checkpoint),
            "parent_sha256": digest(parent), "source_sha256": {str(p): digest(p) for p in sources},
            "rng": {"order": ORDER_SEED, "world": WORLD_SEED, "candidate": POPULATION_SEED},
            "sizes": {"nodes": int(params["matrix"].shape[0]), "latent": params["latent"], "memory": params["capacity"]},
            "system": {"processor": platform.processor(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
                       "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
                       "cuda": torch.version.cuda, "python_pid": os.getpid(), "blas": np.__config__.CONFIG.get("Build Dependencies", {})}}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["cpu", "gpu", "quick", "parity", "matched"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--world-pairs", type=int, default=48)
    parser.add_argument("--pipeline-generations", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--equal-batch", type=int, default=64)
    parser.add_argument("--device-repeats", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--cpu-blocks", type=int, default=8)
    parser.add_argument("--cpu-pairs", type=int, default=16)
    parser.add_argument("--cpu-candidates", type=int, default=8)
    args = parser.parse_args()
    args.cpu_workers = 4
    if args.cpu_candidates < 4 or args.cpu_candidates % 4 or args.prefetch <= 0:
        parser.error("Use a positive prefetch and at least four CPU candidates in multiples of four")
    torch.set_num_threads(1)
    fit, parent, params = load_inputs(args.checkpoint)
    mode = fit["settings"]["mode"]
    results = {"metadata": metadata(args, fit, parent, params), "stages": {}}
    out = results["stages"]
    if args.part in {"cpu", "quick", "parity", "matched"}:
        cpu_tasks = list(select_cpu_tasks(fit, args))
        results["small_cases"] = [{"seeds": task.seeds, "settings": task.settings, "candidate_sha256": digest(c)}
                                  for task, c in cpu_tasks]
    if args.part == "cpu":
        task, candidates = cpu_tasks[0]
        episode, setup = prepare(task)
        results["host_world_seconds"] = setup
        measured_stage("native_numpy_1_vs_4", out, lambda: cpu_trials(params, mode, episode, candidates,
                                                                     args.cpu_blocks, args.cpu_workers))
    if args.part in {"quick", "parity"}:
        tasks = cpu_tasks[:1] if args.part == "quick" else cpu_tasks
        for i, (task, candidates) in enumerate(tasks):
            measured_stage(f"native_gpu_parity_{i}", out, lambda: gpu_parity(params, mode, task, candidates))
            save(args.out, results)
    if args.part == "matched":
        task, candidates = cpu_tasks[0]
        measured_stage("matched_native_runtime", out, lambda: matched_trials(params, mode, task, candidates,
                                                                              args.cpu_blocks, args.cpu_workers))
    if args.part == "gpu":
        tasks = works(fit, args.width, args.world_pairs, args.pipeline_generations)
        results["pipeline_cases"] = [{"seeds": task.seeds, "settings": task.settings, "candidate_sha256": digest(task.coords)}
                                     for task in tasks]
        episode, _ = prepare(tasks[0])
        cohort = symmetry_policy.Cohort(params, episode)
        results["resident_case"] = {"shape": cohort.shape, "queries": int(episode.query[0].sum())}
        pop = symmetry_policy.decoded(params, tasks[0].coords, mode)
        source = symmetry_policy.policy_source(params["latent"], params["capacity"], pop.shape[1])
        dispatch = Dispatch(cohort, pop, source)
        results["cuda_source_sha256"] = {"reference": digest(source), "warp_pre_fusion": digest(dispatch.warp_source)}
        dispatch.detailed()
        dispatch.fused()
        exact = compare_arrays(dispatch.counts.cpu().numpy().astype(int),
                               score_numerators(dispatch.detail[2].cpu().numpy(), episode.query[0]))
        results["resident_raw_count_parity"] = exact
        old, new = [run_score(dispatch, cohort, pop, arm, args.equal_batch) for arm in ("reference", "fused")]
        results["resident_float_score_parity"] = compare_arrays(new, old)
        for name in ("resident_device_event", "resident_scoring_equal_batch", "resident_scoring_native_batch", "prefetch_pipeline"):
            gpu_trials(name, args, out, params, mode, tasks, cohort, pop, dispatch)
            save(args.out, results)
        measured_stage("pipeline_parity", out, lambda: pipeline_parity(params, mode, tasks))
        if exact["different"]:
            print("WARNING: fused score counts differ", flush=True)
    save(args.out, results)


if __name__ == "__main__":
    main()

