"""Audit and publish the two sealed candidate backend allocations."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from experiments.learned_binding import common
from experiments.seed_evolution import (
    audit, balanced_schedule, execution_audit, execution_runtime, large_audit, large_results,
)


@dataclass(frozen=True)
class _BalancedExecution(execution_audit.Execution):
    later: execution_audit.Execution

    def backend(self, cid):
        backend = self.later.backend(cid)
        if backend == "legacy_cpu":
            return super().backend(cid)
        return backend


def _candidate(root, record, config, source_executor, genomes, *, execution, earlier, later):
    counts = earlier if execution.later.backend(record["id"]) == "legacy_cpu" else later
    return execution_audit._audit_candidate(
        root, record, config, source_executor, genomes, execution=execution, counts=counts,
    )


def _backend_counts(counts):
    return {
        str(rep): {
            arm: {backend: counts[rep, arm, backend] for backend in ("cpu", "hybrid")}
            for arm in audit.ARMS
        }
        for rep in range(3)
    }


def _phase(start, stop, indices, counts, *, full):
    if full:
        generations = stop - start
        expected = dict(cpu=generations * (64 - len(indices)), hybrid=generations * len(indices))
        audit.same_tree(counts, {str(rep): {arm: expected for arm in audit.ARMS} for rep in range(3)},
                        "paired backend counts for execution phase")
    return dict(start_generation=start, stop_generation_exclusive=stop,
                hybrid_indices=list(indices), backend_counts=counts)


def _audit_seals(root, schedule):
    return dict(execution_amendment_sha256=common.sha(root / execution_runtime.AMENDMENT),
                balanced_schedule_sha256=schedule.sha256,
                execution_audit_sha256=common.sha(execution_audit.__file__),
                balanced_audit_sha256=common.sha(__file__))


def audit_all(root=large_audit.OUT, workers=9, *, metadata_only=False):
    """Check both allocations, then replay on each candidate's actual training backend."""
    root = Path(root).resolve()
    base = execution_audit._load_execution(root)
    schedule = balanced_schedule.load(root)
    audit.require(base.start_generation < schedule.start_generation < 32,
                  "invalid supplemental execution boundary")
    seals = _audit_seals(root, schedule)
    audit.same_tree(seals["execution_amendment_sha256"], base.sha256, "execution amendment digest")
    later = execution_audit.Execution(schedule.start_generation, schedule.hybrid_indices,
                                      base.metadata, base.sha256)
    execution = _BalancedExecution(base.start_generation, base.hybrid_indices,
                                   base.metadata, base.sha256, later)
    earlier_counts, later_counts = Counter(), Counter()
    candidate = partial(_candidate, execution=execution, earlier=earlier_counts, later=later_counts)
    pool = partial(execution_audit._ReplayPool, execution=execution)
    override = execution_audit._override
    with (override(audit, audit_candidate=candidate, replay_candidate=execution_audit.replay_candidate,
                   audit_final=execution_audit.audit_final),
          override(large_audit, ProcessPoolExecutor=pool)):
        report = large_audit.audit_all(root, workers, metadata_only=metadata_only)

    counts = earlier_counts + later_counts
    audit.require(sum(counts.values()) == report["candidates_provenance_checked"],
                  "execution ledger coverage differs")
    prefix_count = sum(count for (_, _, backend), count in earlier_counts.items()
                       if backend == "legacy_cpu")
    phases = [
        _phase(base.start_generation, schedule.start_generation, base.hybrid_indices,
               _backend_counts(earlier_counts), full=not metadata_only),
        _phase(schedule.start_generation, report["target_generations"], schedule.hybrid_indices,
               _backend_counts(later_counts), full=not metadata_only),
    ]
    if not metadata_only:
        audit.require(sum(replay["execution_backend"] == "hybrid" and replay["independent_ce_updates"] > 0
                          for replay in report["development_replays"]) == 6,
                      "incomplete hybrid development training replay coverage")
    audit.same_tree(_audit_seals(root, balanced_schedule.load(root)), seals,
                    "execution manifests or audit sources changed during audit")
    return dict(report, **seals, execution_start_generation=base.start_generation,
                legacy_cpu_candidates=prefix_count, suffix_backend_counts=_backend_counts(counts),
                execution_phases=phases)


def aggregate(root=large_audit.OUT, dest=large_results.DEST):
    """Publish only a complete audit bound to the original and supplemental allocations."""
    root, dest = Path(root), Path(dest)
    spec = execution_runtime.load_amendment(root)
    schedule = balanced_schedule.load(root)
    audited = large_results.read(root / "audit.json")
    seals = _audit_seals(root, schedule)
    audit.same_tree({key: audited.get(key) for key in seals}, seals, "audited execution seals")
    target_generations = large_results.read(root / "protocol.json")["config"]["generations"]
    phases = audited.get("execution_phases", [])
    audit.require(len(phases) == 2, "publication requires both audited execution phases")
    audit.same_tree(phases, [
        _phase(spec["start_generation"], schedule.start_generation, spec["hybrid_indices"],
               phases[0]["backend_counts"], full=True),
        _phase(schedule.start_generation, target_generations, schedule.hybrid_indices,
               phases[1]["backend_counts"], full=True),
    ], "audited execution phases")
    expected_counts = {
        str(rep): {
            arm: {backend: sum(phase["backend_counts"][str(rep)][arm][backend] for phase in phases)
                  for backend in ("cpu", "hybrid")}
            for arm in audit.ARMS
        }
        for rep in range(3)
    }
    audit.same_tree(audited["suffix_backend_counts"], expected_counts, "audited suffix backend counts")

    result = large_results.aggregate(root, dest)
    report_path = dest / "results.json"
    report = large_results.read(report_path)
    report["execution"] = dict(
        amendment_sha256=seals["execution_amendment_sha256"],
        balanced_schedule_sha256=seals["balanced_schedule_sha256"],
        start_generation=spec["start_generation"],
        hybrid_indices=spec["hybrid_indices"],
        backend_metadata=spec["backend_metadata"],
        suffix_backend_counts=audited["suffix_backend_counts"],
        phases=phases,
        note="Corresponding evolution and random indices share a backend in each phase. "
             "Training uses FP32 throughout. CPU and CUDA arithmetic can round differently.",
    )
    common.write_json(report_path, report)
    with (dest / "report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n第 1–{spec['start_generation']} 代使用 CPU。"
            f"第 {spec['start_generation'] + 1}–{schedule.start_generation} 代沿用原混合分配，"
            f"第 {schedule.start_generation + 1}–{target_generations} 代使用补充混合分配。"
            "各代中，进化组与随机组的相同候选序号使用同一后端。训练全程使用 FP32，"
            "CPU 与 CUDA 的舍入结果可能不同。分段计数及执行记录的 SHA-256 见 results.json。\n"
        )
    return result
