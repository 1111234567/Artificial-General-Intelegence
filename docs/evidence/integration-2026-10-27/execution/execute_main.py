"""Finite 36-run main queue, guarded by a separately frozen manifest."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.continuity import run as continuity
from experiments.integration.run import ARMS, IntegrationConfig, source_hashes

FOLDER = Path(__file__).resolve().parent
EXPECTED_MANIFEST_SHA256 = "b6d9bdc3a5cf1f81ff017f8fbd0169e010d6d34ca9d961ec8a5a53a002c56674"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def configs_and_origins(manifest):
    """Use explicit source hashes and arm configs; do not infer a new protocol."""
    seeds, modes = manifest["seeds"], manifest["arms"]
    if seeds != [0, 1, 2] or modes != list(ARMS):
        raise ValueError("main manifest must contain the agreed three seeds and twelve ordered arms")
    origins = manifest["source_checkpoints"]
    if isinstance(origins, list):
        origins = {str(item["seed"]): item for item in origins}
    specs = []
    for seed in seeds:
        entry = origins[str(seed)]
        source = resolve(entry.get("path", entry.get("checkpoint")))
        source_sha = entry.get("sha256", entry.get("checkpoint_sha256"))
        if not source_sha or sha256(source) != source_sha:
            raise ValueError(f"source checkpoint hash mismatch for seed {seed}")
        if sha256(source.with_name("report.json")) != entry["report_sha256"]:
            raise ValueError(f"source report hash mismatch for seed {seed}")
        old_report = json.loads(source.with_name("report.json").read_text(encoding="utf-8"))
        if (old_report["stage"] != "A" or old_report["stage_updates"] != 2048
                or old_report["ticks"] != 131072 or old_report["variant"]["policy"] != "online"
                or old_report["variant"]["width"] != 128 or old_report["config"]["batch_size"] != 64
                or old_report["config"]["train_dts"] != [.25]):
            raise ValueError(f"source A budget/settings mismatch for seed {seed}")
        for mode in modes:
            values = {"mode": mode, "local_lr": manifest["local_lr"], "local_tau": manifest["local_tau"]}
            values.update(manifest.get("arm_configs", {}).get(mode, {}))
            if values["mode"] != mode:
                raise ValueError("arm config changed the named mode")
            config = IntegrationConfig(**values)
            declared = next(item for item in manifest["runs"] if item["seed"] == seed and item["mode"] == mode)
            if (resolve(declared["source_checkpoint"]) != source
                    or declared["source_checkpoint_sha256"] != source_sha
                    or json.loads(json.dumps(asdict(config))) != declared["integration_config"]
                    or resolve(declared["output"]) != FOLDER / f"seed_{seed:03d}" / mode):
                raise ValueError(f"explicit run disagrees with manifest source/config/output: {seed}/{mode}")
            specs.append({"seed": seed, "mode": mode, "source": str(source), "source_sha256": source_sha,
                          "output": str(resolve(declared["output"])), "integration_config": asdict(config)})
    return specs


def matches(expected, actual):
    return all(actual.get(name) == digest for name, digest in expected.items())


def frozen_sources_match(manifest):
    return (matches(manifest["source_hashes"]["integration"], source_hashes())
            and matches(manifest["source_hashes"]["continuity"], continuity.source_hashes())
            and all(sha256(resolve(name)) == digest for name, digest in manifest["source_file_hashes"].items()))


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(item) for item in value)
    return True


def audit_report(spec, ticks, expected_sources):
    folder = Path(spec["output"])
    report_path, checkpoint_path = folder / "report.json", folder / "checkpoint.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = spec["integration_config"]
    # JSON turns the runner's tuple ranges into lists.
    expected_config = json.loads(json.dumps(config))
    if report["integration_config"] != expected_config or report["stage_ticks"] != ticks:
        raise ValueError(f"final config/tick mismatch: {folder}")
    if report["stage_updates"] != ticks // config["live_ticks"]:
        raise ValueError(f"final update budget mismatch: {folder}")
    if not matches(expected_sources, report["integration_source_hashes"]):
        raise ValueError(f"final integration source mismatch: {folder}")
    if report["origin"]["checkpoint_sha256"] != spec["source_sha256"]:
        raise ValueError(f"final origin mismatch: {folder}")
    arm = ARMS[spec["mode"]]
    expected_gradient = report["stage_updates"] * (int(arm.current_gradient) + int(arm.extra_gradient is not None))
    if report["stage_gradient_updates"] != expected_gradient:
        raise ValueError(f"final gradient budget mismatch: {folder}")
    if report["local_updates"] != report["stage_updates"] * int(arm.local_source is not None):
        raise ValueError(f"final local update budget mismatch: {folder}")
    if report["world_resets"] or report["hidden_resets"]:
        raise ValueError(f"unexpected live state reset: {folder}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = state["continuation"]
    if (base["ticks"] - base["phase_start_tick"] != ticks
            or state["origin"] != report["origin"]
            or state["integration_config"] != config
            or state["work"] != report["work"]):
        raise ValueError(f"checkpoint/report mismatch: {folder}")
    finite = {name: finite_tree(base[name]) for name in ("model", "optimizer", "hidden")}
    finite.update({name: finite_tree(state[name]) for name in ("local_live", "local_replay", "update_norms")})
    if not all(finite.values()):
        raise ValueError(f"nonfinite final model/hidden/Adam/local state: {folder}")
    final_hashes = {name: continuity.state_digest(base[name]) for name in
                    ("model", "optimizer", "hidden", "world", "rng", "replay")}
    if final_hashes["replay"] != report["origin"]["replay_hash"]:
        raise ValueError(f"inherited experience bank/RNG changed: {folder}")
    final_hashes.update({name: continuity.state_digest(state[name]) for name in
                       ("local_live", "local_replay", "replay_rng")})
    return {"seed": spec["seed"], "mode": spec["mode"], "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path), "report": str(report_path),
            "report_sha256": sha256(report_path), "stage_ticks": report["stage_ticks"],
            "stage_updates": report["stage_updates"], "stage_gradient_updates": report["stage_gradient_updates"],
            "local_updates": report["local_updates"], "event_hash": report["event_hash"],
            "origin": report["origin"], "integration_config": report["integration_config"],
            "work": report["work"], "update_norms": report["update_norms"],
            "finite": finite, "final_state_hashes": final_hashes}


def command_for(spec, ticks):
    config = spec["integration_config"]
    folder = Path(spec["output"])
    command = [sys.executable, "-m", "experiments.integration.run", "--output", str(folder),
               "--ticks", str(ticks), "--device", "cuda", "--threads", "1", "--checkpoint-every", "256"]
    if folder.exists():
        checkpoint = folder / "checkpoint.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"partial output lacks a complete resumable checkpoint: {folder}")
        return command + ["--resume", str(checkpoint)], True
    command += ["--from-checkpoint", spec["source"], "--mode", config["mode"],
                "--live-ticks", str(config["live_ticks"]), "--gradient-horizon", str(config["gradient_horizon"]),
                "--local-lr", str(config["local_lr"]), "--local-tau", str(config["local_tau"]),
                "--duration", *map(str, config["duration_range"]), "--gap", *map(str, config["gap_range"])]
    return command, False


def evaluation_specs(spec, trained, manifest):
    label = f"seed_{spec['seed']:03d}_{spec['mode']}"
    common = {"seed": spec["seed"], "mode": spec["mode"], "checkpoint": trained["checkpoint"],
              "checkpoint_sha256": trained["checkpoint_sha256"], "steps": manifest["eval_steps"]}
    jobs = [{**common, "kind": "baseline", "conditions": manifest["conditions"], "interventions": ["none"],
             "output": str(resolve(manifest["outputs"]["evaluations"]) / f"{label}.json")}]
    if spec["mode"] in manifest["diagnostic_arms"]:
        for condition, interventions in manifest["diagnostic_interventions"].items():
            jobs.append({**common, "kind": "diagnostic", "conditions": [condition],
                         "interventions": interventions,
                         "output": str(resolve(manifest["outputs"]["diagnostics"]) / f"{label}_{condition}.json")})
    return jobs


def audit_evaluation(job, manifest):
    path = Path(job["output"])
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if wrapper["format"] != "integration-own-trace-v2" or len(wrapper["evaluations"]) != 1:
        raise ValueError(f"unexpected evaluation wrapper: {path}")
    value = wrapper["evaluations"][0]
    expected_pairs = {(condition, intervention) for condition in job["conditions"] for intervention in job["interventions"]}
    actual_pairs = {(item["condition"], item["intervention"]) for item in value["records"]}
    if (value["checkpoint_sha256"] != job["checkpoint_sha256"] or expected_pairs != actual_pairs
            or len(value["records"]) != len(expected_pairs)
            or value["evaluation_args"]["steps"] != job["steps"]
            or value["training_ticks"] != manifest["origin_ticks"] + manifest["stage_ticks"]
            or not matches(manifest["source_hashes"]["diagnose"], value["source_hashes"])):
        raise ValueError(f"evaluation provenance/budget mismatch: {path}")
    if sha256(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
        raise ValueError(f"evaluation modified the final checkpoint: {path}")
    return {"seed": job["seed"], "mode": job["mode"], "kind": job["kind"], "path": str(path),
            "sha256": sha256(path), "checkpoint_sha256": job["checkpoint_sha256"],
            "checkpoint_state_hash": value["checkpoint_state_hash"], "records": len(value["records"]),
            "conditions": job["conditions"], "interventions": job["interventions"]}


def evaluation_command(job, manifest):
    # Thread selection is execution configuration; the frozen observer is imported
    # unchanged and its CLI receives only the manifest's frozen conditions.
    wrapper = "import torch; torch.set_num_threads(1); from experiments.integration.diagnose import main; main()"
    return [sys.executable, "-c", wrapper, "--checkpoint", job["checkpoint"], "--output", job["output"],
            "--conditions", *job["conditions"], "--interventions", *job["interventions"],
            "--steps", str(job["steps"]), "--dt", str(manifest["dt"]), "--device", "cuda"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not 1 <= args.workers <= 8:
        raise ValueError("main queue supports one to eight GPU workers")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest differs from the human-approved frozen manifest")
    if not frozen_sources_match(manifest):
        raise ValueError("manifest runtime sources do not match the current frozen revision")
    expected_sources = manifest["source_hashes"]["integration"]
    expected_base = manifest["source_hashes"]["continuity"]
    ticks = manifest["stage_ticks"]
    if ticks != 65536 or manifest["local_lr"] != .001 or manifest["local_tau"] != 16.:
        raise ValueError("main manifest differs from the agreed fixed training budget/coefficient")
    specs = configs_and_origins(manifest)
    if len(specs) != 36 or len(manifest["runs"]) != 36:
        raise ValueError("main manifest did not expand to thirty-six runs")
    evaluation_jobs_expected = len(specs) + len(manifest["seeds"]) * len(manifest["diagnostic_arms"]) * len(manifest["diagnostic_interventions"])
    evaluation_records_expected = len(specs) * len(manifest["conditions"]) + len(manifest["seeds"]) * len(manifest["diagnostic_arms"]) * sum(len(x) for x in manifest["diagnostic_interventions"].values())
    execution = {"format": "integration-main-execution-v1", "manifest": str(manifest_path),
                 "manifest_sha256": manifest_sha, "orchestrator_sha256": sha256(Path(__file__)),
                 "created_utc": now(), "workers": args.workers, "stage_ticks": ticks,
                 "integration_source_hashes": expected_sources, "continuity_source_hashes": expected_base,
                 "raw_source_hashes": manifest["source_file_hashes"], "specs": specs,
                 "expected_evaluation_jobs": evaluation_jobs_expected,
                 "expected_evaluation_records": evaluation_records_expected,
                 "queue_policy": "prefer pending training; then ready final-checkpoint baselines; then selected diagnostic groups"}
    if args.validate_only:
        print(json.dumps({"event": "main_manifest_validated", "runs": len(specs),
                          "evaluation_jobs": evaluation_jobs_expected, "evaluation_records": evaluation_records_expected,
                          "manifest_sha256": manifest_sha, "orchestrator_sha256": execution["orchestrator_sha256"]}), flush=True)
        return
    plan_path = FOLDER / "execution-plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous["manifest_sha256"] != manifest_sha or previous["orchestrator_sha256"] != execution["orchestrator_sha256"]:
            raise ValueError("existing execution belongs to a different manifest/orchestrator")
    else:
        continuity.atomic_json(plan_path, execution)
    for kind in ("evaluations", "diagnostics"):
        resolve(manifest["outputs"][kind]).mkdir(parents=True, exist_ok=True)
    pending, ready_eval, running, completed, evaluated, errors = deepcopy(specs), [], [], [], [], []

    def queue_evaluations(spec, trained):
        for job in evaluation_specs(spec, trained, manifest):
            if Path(job["output"]).exists():
                evaluated.append(audit_evaluation(job, manifest))
            else:
                ready_eval.append(job)

    for spec in list(pending):
        final_report = Path(spec["output"]) / "report.json"
        if final_report.exists():
            report = json.loads(final_report.read_text(encoding="utf-8"))
            if report.get("stage_ticks") == ticks:
                trained = audit_report(spec, ticks, expected_sources)
                completed.append(trained)
                queue_evaluations(spec, trained)
                pending.remove(spec)
    while pending or ready_eval or running:
        if sha256(manifest_path) != manifest_sha or not frozen_sources_match(manifest):
            if not any(item.get("error") == "frozen manifest/runtime changed during execution" for item in errors):
                errors.append({"error": "frozen manifest/runtime changed during execution"})
            pending.clear()
            ready_eval.clear()
        while (pending or ready_eval) and len(running) < args.workers and not errors:
            if pending:
                payload = pending.pop(0)
                kind = "training"
                command, resumed = command_for(payload, ticks)
                label = f"seed_{payload['seed']:03d}_{payload['mode']}"
                log_folder = FOLDER
            else:
                ready_eval.sort(key=lambda item: item["kind"] != "baseline")
                payload = ready_eval.pop(0)
                kind, resumed = payload["kind"], False
                command = evaluation_command(payload, manifest)
                label = Path(payload["output"]).stem
                log_folder = Path(payload["output"]).parent
            stdout = (log_folder / f"{label}.stdout.log").open("a", encoding="utf-8")
            stderr = (log_folder / f"{label}.stderr.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            running.append((kind, payload, process, stdout, stderr))
            print(json.dumps({"event": "job_started", "kind": kind, "seed": payload["seed"], "mode": payload["mode"],
                              "conditions": payload.get("conditions"), "pid": process.pid, "resumed": resumed}), flush=True)
        for entry in list(running):
            kind, payload, process, stdout, stderr = entry
            code = process.poll()
            if code is None:
                continue
            stdout.close()
            stderr.close()
            running.remove(entry)
            if code:
                errors.append({"kind": kind, "seed": payload["seed"], "mode": payload["mode"],
                               "conditions": payload.get("conditions"), "exit_code": code})
            else:
                try:
                    if kind == "training":
                        result = audit_report(payload, ticks, expected_sources)
                        completed.append(result)
                        queue_evaluations(payload, result)
                        journal = FOLDER / "completed.jsonl"
                    else:
                        result = audit_evaluation(payload, manifest)
                        evaluated.append(result)
                        journal = FOLDER / "evaluated.jsonl"
                    with journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                except Exception as error:
                    errors.append({"kind": kind, "seed": payload["seed"], "mode": payload["mode"],
                                   "conditions": payload.get("conditions"), "audit_error": repr(error)})
            print(json.dumps({"event": "job_finished", "kind": kind, "seed": payload["seed"], "mode": payload["mode"],
                              "conditions": payload.get("conditions"), "exit_code": code,
                              "training_completed": len(completed), "evaluation_jobs_completed": len(evaluated), "errors": len(errors)}), flush=True)
        progress = {"updated_utc": now(), "manifest_sha256": manifest_sha, "completed": completed,
                    "evaluations_completed": evaluated,
                    "running": [{"kind": item[0], "seed": item[1]["seed"], "mode": item[1]["mode"],
                                 "conditions": item[1].get("conditions"), "pid": item[2].pid} for item in running],
                    "pending": [{"seed": item["seed"], "mode": item["mode"]} for item in pending],
                    "evaluations_pending": [{"kind": item["kind"], "seed": item["seed"], "mode": item["mode"],
                                             "conditions": item["conditions"]} for item in ready_eval], "errors": errors}
        continuity.atomic_json(FOLDER / "progress.json", progress)
        if errors and not running:
            break
        if running:
            time.sleep(10)
    unique_sources = {(spec["source"], spec["source_sha256"]) for spec in specs}
    unchanged = (sha256(manifest_path) == manifest_sha and frozen_sources_match(manifest)
                 and all(sha256(path) == digest for path, digest in unique_sources))
    events, origins = {}, {}
    for result in completed:
        origins.setdefault(str(result["seed"]), set()).add(result["origin"]["semantic_hash"])
        if result["mode"] not in {"coverage_long_dense", "coverage_short_sparse"}:
            events.setdefault(str(result["seed"]), set()).add(result["event_hash"])
    event_pairing = {seed: len(values) == 1 for seed, values in events.items()}
    origin_pairing = {seed: len(values) == 1 for seed, values in origins.items()}
    summary = {"format": "integration-main-execution-summary-v1", "completed_utc": now(),
               "execution": execution, "completed": completed, "evaluations_completed": evaluated, "errors": errors,
               "all_36_completed": len(completed) == 36 and not errors,
               "all_evaluations_completed": len(evaluated) == evaluation_jobs_expected and not errors,
               "evaluation_records": sum(item["records"] for item in evaluated),
               "frozen_inputs_unchanged": unchanged, "noncoverage_event_pairing": event_pairing,
               "same_seed_origin_pairing": origin_pairing}
    continuity.atomic_json(FOLDER / "execution-summary.json", summary)
    print(json.dumps({"event": "main_queue_complete", "training_completed": len(completed),
                      "evaluation_jobs_completed": len(evaluated), "errors": len(errors),
                      "frozen_inputs_unchanged": unchanged, "event_pairing": event_pairing,
                      "origin_pairing": origin_pairing}), flush=True)
    if (errors or len(completed) != 36 or len(evaluated) != evaluation_jobs_expected or not unchanged
            or not all(event_pairing.values()) or not all(origin_pairing.values())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
