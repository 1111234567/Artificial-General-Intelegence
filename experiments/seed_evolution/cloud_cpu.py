"""Resume the transferred search, train finalists, and publish its CPU audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import json
import multiprocessing
import os
import platform
import traceback
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import torch

from experiments.seed_evolution import audit, cloud_cpu_audit, cloud_cpu_final, core, hybrid_backend
from experiments.seed_evolution import large_results, large_runner as runner
from experiments.seed_evolution.lazy_training_streams import LazyTrainingStreams


_CANDIDATE_JOB = runner._candidate_job
_TRAIN = core.inner_train
_STREAMS = LazyTrainingStreams()


def cpu_train(seed, genome, steps=64, checkpoints=(0, 64), device="cpu"):
    if device != "cpu":
        raise ValueError("The migrated search uses the CPU backend")
    with patch.object(core, "inner_train", _TRAIN), _STREAMS.install():
        result = hybrid_backend.train(seed, genome, steps, checkpoints, backend="cpu", cache=True)
    # The CPU cache and stream loader preserve the original numerical path.
    result["metadata"].pop("execution")
    return result


def candidate_job(descriptor):
    with patch.object(core, "inner_train", cpu_train):
        return _CANDIDATE_JOB(descriptor)


def validate():
    runner._activate()
    protocol = runner._check()
    migration, _ = cloud_cpu_audit._verify_migration(runner.OUT, protocol)
    audit.verify_sources(protocol)
    return migration


def check():
    migration = validate()
    genome = core.new_genome(0, 710000)
    smoke = cpu_train(0, genome, steps=2, checkpoints=(0, 2))
    checkpoint = smoke["checkpoints"][2]
    if checkpoint["updates"] != 2 or not checkpoint["optimizer"]["state"]:
        raise RuntimeError("CPU smoke training did not complete")
    return dict(check="passed", torch=str(torch.__version__),
                archived_candidates=migration["historical_count"],
                sealed_generations=migration["cutover_generation"],
                next_generation=migration["cutover_generation"] + 1,
                smoke_cpu_updates=2)


def run(workers):
    root = runner.OUT
    pool = partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn"))
    with runner._lock(root / ".pipeline.lock"):
        stage = "validate"
        try:
            migration = validate()
            runner._write(root / "cloud-process.json", dict(
                pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                workers=workers, device="cpu", torch=str(torch.__version__),
                platform=platform.platform(), migration_sha256=runner.base.sha(root / "migration.json"),
                archived_candidates=migration["historical_count"]))
            print(f"CPU continuation: {migration['historical_count']} archived candidates, {workers} workers", flush=True)
            with (patch.object(runner, "ProcessPoolExecutor", pool),
                  patch.object(runner, "_candidate_job", candidate_job),
                  patch.object(runner, "_train_final", cloud_cpu_final.train_final)):
                stage = "search"
                runner.search(workers)
                stage = "finals"
                runner.finals(min(6, workers))
                stage = "test"
                runner.tests(min(6, workers))
            stage = "audit"
            runner._progress(stage, "running", completed_candidates=runner._unique_budget(),
                             total_candidates=runner._unique_budget())
            with patch.object(cloud_cpu_audit, "ProcessPoolExecutor", pool):
                result = cloud_cpu_audit.audit_all(root, workers=min(6, workers))
            if result.get("full_audit_passed") is not True:
                raise ValueError("CPU continuation audit failed")
            runner._write(root / "audit.json", result)
            stage = "results"
            summary = large_results.aggregate(root)
            dest = Path(summary["output"])
            result_path = dest / "results.json"
            evidence = runner.base.read(result_path)
            evidence["migration"] = dict(
                manifest_sha256=result["migration_sha256"],
                archived_candidates=result["historical_candidates_provenance_checked"],
                cpu_candidates=result["linux_cpu_candidates_provenance_checked"],
                historical_training_replays=result["historical_training_replays"],
                historical_backends=result["historical_backend_counts"],
                finalist_continuation=result["continuation_disclosure"])
            runner._write(result_path, evidence)
            report = dest / "report.md"
            note = (
                "\n## 云端 CPU 迁移\n\n"
                f"本轮从本机已完成的 {migration['historical_count']:,} 个候选接续。"
                "历史候选核验档案、谱系和评分记录。后续候选在云端使用 CPU，"
                "抽样重放在同一 CPU 环境完成。终训从胜者保存的第 64 步模型与 Adam 状态"
                "继续至第 512 步。\n"
            )
            report_text = report.read_text(encoding="utf-8") + note
            runner._atomic(report, lambda temporary: temporary.write_text(report_text, encoding="utf-8"))
            runner._write(root / "cloud-completion.json", dict(full_audit_passed=True, result=summary,
                migration_sha256=result["migration_sha256"], completed_utc=datetime.now(timezone.utc).isoformat()))
            runner._progress("complete", "complete", completed_candidates=runner._unique_budget(),
                             total_candidates=runner._unique_budget())
        except BaseException as exc:
            runner._write(root / "cloud-failure.json", dict(stage=stage, error=repr(exc), traceback=traceback.format_exc()))
            runner._progress(stage, "failed", error=repr(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.command == "check":
        print(json.dumps(check(), ensure_ascii=False))
    else:
        run(arguments.workers)
