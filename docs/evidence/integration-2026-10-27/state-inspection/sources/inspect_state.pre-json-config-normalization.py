"""Post-hoc CPU inspection of all final integration snapshots, with no rollout.

The recorded plan precedes access to final state tensors. This module loads raw
checkpoint dictionaries, never constructs a controller/world/optimizer, and
never restores RNG state. Terminal actor covariance is descriptive, phase
heterogeneous and rank-limited by sample size; it is not a longitudinal task
representation matrix or evidence of causal loss of plasticity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = "integration-terminal-state-inspection-v1"
PLAN_VERSION = "integration-state-inspection-plan-v1"
TIMING_EPS = 1e-6
SATURATION_THRESHOLD = 0.95
PHASE_NAMES = ("inactive", "active_unstarted_feasible", "active_unstarted_missed_start",
               "active_processing", "active_legal_finish", "active_past_deadline")


class StudyNotReady(ValueError):
    """The entire final cohort must be ready before any checkpoint is loaded."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def process_rng_fingerprints() -> dict[str, str]:
    """Read CPU RNG states; no draw, seeding, restoration or CUDA initialization."""
    numpy_state = np.random.get_state()
    numpy_digest = hashlib.sha256(numpy_state[1].tobytes())
    numpy_digest.update(json.dumps([numpy_state[0], *numpy_state[2:]]).encode())
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": numpy_digest.hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def _finite_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point matrix")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains nonfinite entries")
    return result


@torch.inference_mode()
def matrix_statistics(recurrent: torch.Tensor) -> dict[str, float]:
    weight = _finite_matrix(recurrent, "recurrent W")
    if min(weight.shape) < 1 or weight.shape[0] != weight.shape[1]:
        raise ValueError("recurrent W must be a nonempty square matrix")
    return {"frobenius_norm": float(torch.linalg.vector_norm(weight)),
            "spectral_norm": float(torch.linalg.svdvals(weight)[0])}


def _center_rows_across_actors(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Identical rows subtract to exact zero before any inexact mean reduction.
    differences = values - values[:1]
    mean_difference = differences.mean(0)
    return differences - mean_difference, values[0] + mean_difference


@torch.inference_mode()
def activity_statistics(hidden: torch.Tensor) -> dict[str, Any]:
    saved = _finite_matrix(hidden, "hidden state")
    batch, width = saved.shape
    if batch < 2 or width < 1:
        raise ValueError("activity covariance requires at least two actors and one unit")
    activity = saved.tanh()
    centered, mean = _center_rows_across_actors(activity)
    covariance = centered.T @ centered / (batch - 1)
    raw_eigenvalues = torch.linalg.eigvalsh(covariance)
    largest = max(float(raw_eigenvalues[-1]), 0.)
    tolerance = torch.finfo(torch.float64).eps * max(batch, width) * largest
    if float(raw_eigenvalues[0]) < -tolerance:
        raise FloatingPointError("centred covariance has a materially negative eigenvalue")
    eigenvalues = torch.where(raw_eigenvalues.abs() <= tolerance, 0., raw_eigenvalues).clamp_min(0.)
    total = float(eigenvalues.sum())
    nonzero = eigenvalues[eigenvalues > 0]
    rank = int(nonzero.numel())
    bound = min(width, batch - 1)
    if rank > bound:
        raise FloatingPointError("numerical covariance rank exceeds its centred sample bound")
    if total:
        probability = nonzero / total
        participation = float(1. / probability.square().sum())
        entropy_rank = float(torch.exp(-(probability * probability.log()).sum()))
    else:
        participation = entropy_rank = 0.
    unit_differences = activity - activity[:, :1]
    row_centered = unit_differences - unit_differences.mean(1, keepdim=True)
    row_centered_across_actors, _ = _center_rows_across_actors(row_centered)
    uncentered_energy = float(activity.square().sum(1).mean())
    mean_energy = float(mean.square().sum())
    return {
        "batch_size": batch, "width": width,
        "tanh_mean": float(activity.mean()), "per_unit_tanh_mean": mean.tolist(),
        "mean_vector_rms": float(mean.square().mean().sqrt()),
        "within_actor_variance_mean": float(row_centered.square().mean()),
        "within_actor_centred_batch_variance_mean": float(row_centered_across_actors.square().sum(0).mean() / (batch - 1)),
        "mean_tanh_derivative": float((1. - activity.square()).mean()),
        "saturation_threshold": SATURATION_THRESHOLD,
        "saturation_fraction": float((activity.abs() > SATURATION_THRESHOLD).double().mean()),
        "mean_vector_energy": mean_energy, "uncentred_activity_energy": uncentered_energy,
        "common_mode_energy_fraction": mean_energy / uncentered_energy if uncentered_energy else 0.,
        "covariance": {
            "centering": "across actors per unit", "denominator": batch - 1,
            "eigenvalues_descending": eigenvalues.flip(0).tolist(),
            "mean_diagonal_variance": float(covariance.diag().mean()),
            "trace": float(covariance.trace()), "numerical_zero_tolerance": tolerance,
            "raw_min_eigenvalue": float(raw_eigenvalues[0]), "numerical_rank": rank,
            "centred_sample_rank_bound": bound, "participation_ratio": participation,
            "entropy_effective_rank": entropy_rank,
            "zero_spectrum_convention": "both effective ranks are zero",
        },
    }


@torch.inference_mode()
def phase_statistics(world: dict[str, Any]) -> dict[str, Any]:
    """Read saved flags/times only; the snapshot follows its completed action."""
    spec, tensors = world["spec"], world["tensors"]
    batch, jobs, active_jobs = int(spec["batch_size"]), int(spec["n_jobs"]), int(spec["active_jobs"])
    if min(batch, jobs, active_jobs) < 1 or active_jobs > jobs:
        raise ValueError("invalid saved world dimensions")
    now = float(world["now"])
    if not math.isfinite(now) or world.get("acted") is not True:
        raise ValueError("world snapshot must have finite time and follow an action")
    for key in ("active", "started", "duration", "deadline", "ready_time"):
        if not isinstance(tensors[key], torch.Tensor) or tensors[key].shape != (batch, jobs):
            raise ValueError(f"invalid saved world tensor: {key}")
    active, started = (tensors[key].detach().cpu().bool() for key in ("active", "started"))
    duration, deadline, ready = (tensors[key].detach().to(device="cpu", dtype=torch.float64)
                                 for key in ("duration", "deadline", "ready_time"))
    if not bool(torch.isfinite(duration).all()) or not bool(torch.isfinite(deadline).all()) or bool(torch.isnan(ready).any()):
        raise ValueError("invalid saved world timing values")
    enabled = (torch.arange(jobs)[None, :] < active_jobs).expand(batch, -1)
    active = active & enabled
    past = active & (now > deadline + TIMING_EPS)
    unstarted = active & ~started & ~past
    begun = active & started & ~past
    feasible = now + duration <= deadline + TIMING_EPS
    phases = {
        "inactive": enabled & ~active,
        "active_unstarted_feasible": unstarted & feasible,
        "active_unstarted_missed_start": unstarted & ~feasible,
        "active_processing": begun & (now < ready - TIMING_EPS),
        "active_legal_finish": begun & (now >= ready - TIMING_EPS),
        "active_past_deadline": past,
    }
    counts = {name: int(value.sum()) for name, value in phases.items()}
    total = batch * active_jobs
    if sum(counts.values()) != total:
        raise ValueError("saved world phases do not partition enabled slots")
    active_per_actor = active.sum(1)
    started_per_actor = (active & started).sum(1)
    return {
        "world_tick": int(world["tick"]), "world_now": now, "elapsed": float(world["elapsed"]),
        "event_hash": world["event_hash"], "batch_size": batch, "n_jobs": jobs, "active_jobs": active_jobs,
        "enabled_slots": total, "active_slots": int(active.sum()),
        "raw_started_flag_slots": int((started & enabled).sum()),
        "active_started_slots": int((active & started).sum()),
        "actors_without_active_job": int((active_per_actor == 0).sum()),
        "actors_with_active_job": int((active_per_actor > 0).sum()),
        "actors_with_active_started_job": int((started_per_actor > 0).sum()),
        "active_jobs_per_actor_histogram": torch.bincount(active_per_actor, minlength=active_jobs + 1).tolist(),
        "active_started_jobs_per_actor_histogram": torch.bincount(started_per_actor, minlength=active_jobs + 1).tolist(),
        "phase_counts": counts, "phase_fractions": {name: value / total for name, value in counts.items()},
        "timing_tolerance": TIMING_EPS,
        "boundary_note": "World flags are post-action; h has consumed the last observation but not this action's next-tick receipt.",
    }


def frozen_source_fingerprints(manifest: dict[str, Any]) -> dict[str, str]:
    expected = dict(manifest["source_file_hashes"])
    for name, digest in manifest["source_hashes"]["continuity"].items():
        expected[str(Path("experiments") / "continuity" / name)] = digest
    observed = {}
    for name, digest in expected.items():
        path = ROOT / name.replace("\\", "/")
        actual = file_sha256(path)
        if actual != digest:
            raise ValueError(f"frozen source changed: {name}")
        observed[name] = actual
    return observed


def validate_final_reports(manifest: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Preflight all metadata before any checkpoint tensor is loaded."""
    expected = {(seed, arm) for seed in plan["runs"]["seeds"] for arm in plan["runs"]["arms"]}
    declared = {(run["seed"], run["mode"]): run for run in manifest["runs"]}
    if set(declared) != expected or len(declared) != len(manifest["runs"]) or len(expected) != 36:
        raise ValueError("inspection requires the exact declared 36 final runs")
    for field, key in (("stage_ticks", "required_stage_ticks"), ("batch_size", "batch_size"),
                       ("width", "width"), ("total_parameters", "total_parameters")):
        if manifest[field] != plan["runs"][key]:
            raise ValueError(f"inspection plan differs from main manifest: {field}")
    missing, ready = [], []
    for key, run in sorted(declared.items()):
        folder = Path(run["output"])
        checkpoint, report_path = folder / "checkpoint.pt", folder / "report.json"
        if not checkpoint.is_file() or not report_path.is_file():
            missing.append({"seed": key[0], "arm": key[1], "reason": "final files absent"})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        live = run["integration_config"]["live_ticks"]
        expected_updates = manifest["stage_ticks"] // live
        actual = (report.get("stage"), report.get("stage_ticks"), report.get("stage_updates"),
                  report.get("config", {}).get("seed"), report.get("adaptation_mode"), report.get("integration_config"))
        required = ("B", manifest["stage_ticks"], expected_updates, key[0], key[1], run["integration_config"])
        if actual != required:
            missing.append({"seed": key[0], "arm": key[1], "reason": "final stage/config not reached"})
            continue
        if (report.get("model", {}).get("state_scalars") != manifest["width"]
                or report.get("model", {}).get("total_parameters") != manifest["total_parameters"]
                or report.get("hidden_resets") != 0 or report.get("world_resets") != 0):
            raise ValueError(f"invalid final model/continuity metadata: {key}")
        for namespace, report_key in (("continuity", "source_hashes"), ("integration", "integration_source_hashes")):
            for name, digest in manifest["source_hashes"][namespace].items():
                if report.get(report_key, {}).get(name) != digest:
                    raise ValueError(f"final report source mismatch: {key}/{namespace}/{name}")
        ready.append({"seed": key[0], "arm": key[1], "checkpoint": checkpoint,
                      "report_path": report_path, "report_sha256": file_sha256(report_path), "report": report})
    if missing:
        raise StudyNotReady(json.dumps({"ready": len(ready), "required": 36, "missing": missing}))
    return ready


def inspect_checkpoint(job: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    path, report = Path(job["checkpoint"]), job["report"]
    before = file_sha256(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "integration-trainer-v1":
            raise ValueError("inspection requires a complete integration checkpoint")
        saved = state["continuation"]
        ticks = saved["ticks"] - saved["phase_start_tick"]
        if (saved.get("stage") != "B" or ticks != manifest["stage_ticks"]
                or saved["config"]["seed"] != job["seed"] or saved.get("adaptation_mode") != job["arm"]
                or state["integration_config"] != report["integration_config"]
                or saved["ticks"] != report["ticks"] or saved["updates"] != report["updates"]):
            raise ValueError("checkpoint is not the report's declared final continuation")
        hidden, recurrent = saved["hidden"], saved["model"]["recurrent"]
        if tuple(hidden.shape) != (manifest["batch_size"], manifest["width"]) or tuple(recurrent.shape) != (manifest["width"],) * 2:
            raise ValueError("checkpoint model/actor shapes differ from the plan")
        if sum(value.numel() for value in saved["model"].values()) != manifest["total_parameters"]:
            raise ValueError("checkpoint parameter count differs from the plan")
        for namespace, actual in (("continuity", saved["source_hashes"]), ("integration", state["source_hashes"])):
            if any(actual.get(name) != digest for name, digest in manifest["source_hashes"][namespace].items()):
                raise ValueError("checkpoint source hashes differ from the frozen main manifest")
        if saved["world"]["tick"] != saved["ticks"] - 1:
            raise ValueError("saved world/controller tick boundary differs")
        result = {
            "seed": job["seed"], "arm": job["arm"], "checkpoint": str(path.resolve()),
            "report_sha256": job["report_sha256"], "stage_ticks": ticks,
            "weight": matrix_statistics(recurrent), "activity": activity_statistics(hidden),
            "world_phase": phase_statistics(saved["world"]),
        }
        if file_sha256(job["report_path"]) != job["report_sha256"]:
            raise ValueError("final report changed during inspection")
    finally:
        after = file_sha256(path)
        if after != before:
            raise ValueError("checkpoint bytes changed during read-only inspection")
    result["checkpoint_sha256_before"] = before
    result["checkpoint_sha256_after"] = after
    return result


def _scalar_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {"seed": record["seed"], "arm": record["arm"]}
    for prefix in ("weight", "activity"):
        output.update({f"{prefix}/{key}": value for key, value in record[prefix].items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)})
    output.update({f"covariance/{key}": value for key, value in record["activity"]["covariance"].items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)})
    output.update({f"world/{key}": record["world_phase"][key] for key in
                   ("enabled_slots", "active_slots", "raw_started_flag_slots", "active_started_slots", "actors_without_active_job",
                    "actors_with_active_job", "actors_with_active_started_job")})
    output.update({f"phase/{key}": value for key, value in record["world_phase"]["phase_fractions"].items()})
    return output


def summarize_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scalar = [_scalar_record(record) for record in records]
    groups = defaultdict(list)
    for row in scalar:
        for metric, value in row.items():
            if metric not in {"seed", "arm"}:
                groups[(row["arm"], metric)].append({"seed": row["seed"], "value": value})
    summary = []
    for (arm, metric), rows in sorted(groups.items()):
        values = [row["value"] for row in rows]
        summary.append({"arm": arm, "metric": metric, "n_seeds": len(values),
                        "mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                        "per_seed": rows})
    return scalar, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row if key != "per_seed"))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def inspect_study(plan_path: str | Path, output: str | Path) -> dict[str, Any]:
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan_before = file_sha256(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("version") != PLAN_VERSION:
        raise ValueError("unsupported inspection plan")
    manifest_path = Path(plan["main_manifest"])
    manifest_before = file_sha256(manifest_path)
    if manifest_before != plan["main_manifest_sha256"]:
        raise ValueError("main manifest differs from the pre-inspection plan")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_before = frozen_source_fingerprints(manifest)
    jobs = validate_final_reports(manifest, plan)  # all36 guard; no checkpoint loaded above
    rng_before = process_rng_fingerprints()
    try:
        records = [inspect_checkpoint(job, manifest) for job in jobs]
    finally:
        frozen_after = frozen_source_fingerprints(manifest)
        rng_after = process_rng_fingerprints()
        if frozen_after != frozen_before or file_sha256(manifest_path) != manifest_before or file_sha256(plan_path) != plan_before:
            raise ValueError("frozen sources/manifest/inspection plan changed during inspection")
        if rng_after != rng_before:
            raise ValueError("inspection changed a process CPU RNG state")
    scalar, summary = summarize_records(records)
    result = {
        "format": FORMAT_VERSION, "classification": plan["classification"],
        "plan": str(plan_path), "plan_sha256_before": plan_before, "plan_sha256_after": file_sha256(plan_path),
        "main_manifest": str(manifest_path), "main_manifest_sha256_before": manifest_before,
        "main_manifest_sha256_after": file_sha256(manifest_path),
        "inspector_sources": {str(path.relative_to(ROOT)): file_sha256(path) for path in
                              (Path(__file__), Path(__file__).with_name("test_inspect_state.py"))},
        "runtime": {"device": "cpu", "arithmetic": "float64", "torch": str(torch.__version__),
                    "cuda_used": False, "forward_steps": 0, "optimizer_steps": 0, "random_draws": 0},
        "audit": {"status": "passed", "checkpoints": len(records), "source_hashes_before": frozen_before,
                  "source_hashes_after": frozen_after, "cpu_rng_before": rng_before, "cpu_rng_after": rng_after,
                  "all_checkpoint_bytes_unchanged": all(row["checkpoint_sha256_before"] == row["checkpoint_sha256_after"] for row in records)},
        "definitions": {"covariance": plan["covariance_metrics"], "phase_context": plan["phase_context"],
                        "activity": plan["activity_metrics"]},
        "limitations": plan["summary"]["exploratory_limits"], "per_seed": records, "summary": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_seed.csv", scalar)
    _write_csv(output / "seed_summary.csv", summary)
    atomic_json(output / "summary.json", result)
    atomic_json(output / "audit.json", result["audit"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "docs/evidence/integration-2026-10-27/state-inspection-plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence/integration-2026-10-27/state-inspection")
    args = parser.parse_args(argv)
    try:
        result = inspect_study(args.plan, args.output)
    except StudyNotReady as error:
        print(json.dumps({"status": "not_ready", "detail": str(error)}))
        return 2
    print(json.dumps({"status": result["audit"]["status"], "checkpoints": len(result["per_seed"]),
                      "output": str(args.output.resolve()), "classification": result["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
