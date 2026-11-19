"""Compare on-chip span layouts using the main search's sealed journals."""

import argparse
from contextlib import contextmanager
import ctypes
from statistics import fmean, stdev
from time import perf_counter

import torch

from experiments.seed_evolution.large_runner import _utc, _write
from . import uncapped_private_symbol_fast_run as fast
from . import uncapped_private_symbol_run as run
from . import uncapped_private_symbol_span as span
from .uncapped_directed_support_cooldown import _SupportCooldownMixin
from .uncapped_directed_support_rebuild import _SupportRebuildMixin
from .uncapped_private_symbol_delta import delta_rollout
from .uncapped_private_symbol_fused_run import FusedEvaluation
from .uncapped_private_symbol_verify_main import read_main_batches, score_rows, verify_tensors


class SharedPolicy(_SupportCooldownMixin, span.SpanPolicy):
    _support_cooldown_threads = 128
    _span_matrix_stride = 104
    _span_lookup_tanh = True


class GlobalPolicy(SharedPolicy):
    _span_shared_deltas = False


class GlobalWidePolicy(GlobalPolicy):
    _span_threads = 256


class RebuildPolicy(_SupportRebuildMixin, GlobalWidePolicy):
    pass


class RebuildWidePolicy(RebuildPolicy):
    _support_cooldown_threads = 256


class RebuildSpan288(RebuildWidePolicy):
    _span_threads = 288


class RebuildSpan320(RebuildWidePolicy):
    _span_threads = 320


class RebuildRows(RebuildWidePolicy):
    _span_fused_rows = True


class RebuildRows288(RebuildRows):
    _span_threads = 288


LAYOUTS = {
    "shared-8": (SharedPolicy, 8),
    "shared-16": (SharedPolicy, 16),
    "global-8": (GlobalPolicy, 8),
    "global-16": (GlobalPolicy, 16),
    "global-32": (GlobalPolicy, 32),
    "global-wide-16": (GlobalWidePolicy, 16),
    "rebuild-128-16": (RebuildPolicy, 16),
    "rebuild-256-16": (RebuildWidePolicy, 16),
    "rebuild-256-32": (RebuildWidePolicy, 32),
    "rebuild-span-288": (RebuildSpan288, 16),
    "rebuild-span-320": (RebuildSpan320, 16),
    "rebuild-fused-rows": (RebuildRows, 16),
    "rebuild-fused-rows-288": (RebuildRows288, 16),
}


def layout_evaluator(policy, **options):
    cls = type("LayoutEvaluation", (span.SpanEvaluation,), {"span_policy": policy})
    return cls(run.world.training_schedule, **options)


@contextmanager
def pause_worker(pid):
    if not pid:
        yield
        return
    active = run.runner.base.read(run.OUT / "process.json")
    if active["pid"] != pid:
        raise ValueError("The supplied PID no longer owns the selected main run")
    kernel, native = ctypes.windll.kernel32, ctypes.windll.ntdll
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    native.NtSuspendProcess.argtypes = native.NtResumeProcess.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x0800, False, pid)  # PROCESS_SUSPEND_RESUME
    if not handle:
        raise OSError("Unable to open the specified main worker")
    try:
        if native.NtSuspendProcess(handle):
            raise OSError("Unable to suspend the specified main worker")
        try:
            yield
        finally:
            if native.NtResumeProcess(handle):
                raise OSError("Unable to resume the specified main worker")
    finally:
        kernel.CloseHandle(handle)


def final_state_rollout(observed):
    def rollout(policy, tape, table):
        metrics, primary = observed(policy, tape, table)
        names = ("w", "g", "mask", "matrix", "last_change", "h", "node_trace", "eligibility", "pending", "tried")
        return {**metrics, "primary_trace": primary, **{name: getattr(policy, name) for name in names}}
    return rollout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolate-pid", type=int)
    parser.add_argument("--output", type=str, default="throughput-main-spans-2026-11-18.json")
    parser.add_argument("--layout", choices=LAYOUTS, action="append")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    cfg, samples, seeds, parents, saved, digest = read_main_batches(run.OUT / "pair-01" / "downhill")
    settings = {key: cfg[key] for key in ("device", "actors", "visits")}
    genomes = [genome for _, batch, _ in samples for genome in batch]
    entries = [entry for batch, _, _ in samples for entry in batch]
    journal = [record for _, _, rows in samples for record in rows]
    layouts_to_check = tuple(args.layout or LAYOUTS)
    variants = {"current": (FusedEvaluation(run.world.training_schedule), 16),
                **{name: (layout_evaluator(LAYOUTS[name][0]), LAYOUTS[name][1]) for name in layouts_to_check}}

    def evaluate(name, seed):
        evaluator, width = variants[name]
        for start in range(0, len(genomes), width):
            yield evaluator(genomes[start:start + width], seed=seed, **settings)

    def sweep(name, seed):
        width = variants[name][1]
        for result in evaluate(name, seed):
            fast.unpack_scores_lists(result, width)

    def signature(name):
        results = list(evaluate(name, seeds[-1]))
        width = variants[name][1]
        rows = [row for start, result in zip(range(0, len(genomes), width), results, strict=True)
                for row in score_rows(result, entries[start:start + width], cfg["edge_pairs"])]
        if rows != journal:
            raise ValueError(f"{name} differs from committed candidate journal")
        host = [result.cpu_fields() for result in results]
        return {key: torch.cat([result[key] for result in host]) for key in host[0]}

    reference = signature("current")
    for name in layouts_to_check:
        verify_tensors(reference, signature(name), name)
        print(f"{name} exact", flush=True)

    def timed(name, seed):
        torch.cuda.synchronize()
        now = perf_counter()
        sweep(name, seed)
        torch.cuda.synchronize()
        return perf_counter() - now

    started = perf_counter()
    with pause_worker(args.isolate_pid):
        layouts = {name: [] for name in variants}
        for index, seed in enumerate(seeds[::2]):
            order = list(variants)[::(1 if index % 2 == 0 else -1)]
            for name in order:
                layouts[name].append(timed(name, seed))
        fastest = min(layouts_to_check, key=lambda name: fmean(layouts[name]))
        paired = []
        for index, seed in enumerate(seeds):
            order = ["current", fastest][::(1 if index % 2 == 0 else -1)]
            for name in order:
                sweep(name, seed)
            times = {name: [] for name in order}
            for name in [*order, *reversed(order)]:
                times[name].append(timed(name, seed))
            ratio = fmean(times["current"]) / fmean(times[fastest])
            paired.append({"seed": seed, "seconds": times, "speedup": ratio})
    timed_seconds = perf_counter() - started

    for evaluator, _ in variants.values():
        evaluator.clear_cache()
    policy, width = LAYOUTS[fastest]
    parent_evaluator = layout_evaluator(policy, rollout=delta_rollout)
    learned = fast.padded_sample_deltas(parent_evaluator, parents, seed=saved["seed"], **settings)
    verify_tensors(saved["deltas"], learned, "stored parent deltas")
    parent_evaluator.clear_cache()

    extremes = []
    for genome, sign in zip(genomes[:2], (1, -1), strict=True):
        rule = genome["rule"].clone()
        rule[1].zero_()
        rule[1, 0] = sign
        extremes.append({**genome, "rule": rule})
    state_evaluators = (
        FusedEvaluation(run.world.training_schedule, rollout=final_state_rollout(run.world.observed_rollout)),
        layout_evaluator(policy, rollout=final_state_rollout(span.observed_rollout)),
    )
    control, actual = (evaluator(extremes, seed=seeds[-1], **settings).cpu_fields() for evaluator in state_evaluators)
    verify_tensors(control, actual, "topology churn, hidden state and raw actions")
    for evaluator in state_evaluators:
        evaluator.clear_cache()

    ratios = [row["speedup"] for row in paired]
    speedup = fmean(ratios)
    radius = 2.364624251 * stdev(ratios) / len(ratios) ** .5
    rates = {name: 60 * len(genomes) / fmean(t for row in paired for t in row["seconds"][name])
             for name in ("current", fastest)}
    report = {"completed_utc": _utc(), "isolated": bool(args.isolate_pid), "timed_window_seconds": timed_seconds,
              "checkpoint_sha256": digest, "exact_fields": len(reference), "exact_journal_rows": len(journal),
              "exact_parent_deltas": len(learned), "exact_raw_actions": control["primary_trace"].numel(),
              "churn_births": control["births"].tolist(), "churn_deaths": control["deaths"].tolist(),
              "screen_seconds": layouts, "fastest": fastest, "width": width, "candidates_per_minute": rates,
              "speedup": speedup, "interval95": [speedup - radius, speedup + radius], "paired": paired}
    _write(run.OUT / "audits" / args.output, report)
    print(f"{fastest} at width {width}: {speedup:.3f}x, isolated={bool(args.isolate_pid)}, rates={rates}", flush=True)


if __name__ == "__main__":
    main()
