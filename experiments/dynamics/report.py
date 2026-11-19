"""Audit and aggregate completed dynamics runs, then render static evidence.

Example (after all held-out tests have finished):
  python experiments/dynamics/report.py --runs data/dynamics/main-seed0 \
      data/dynamics/main-seed1 data/dynamics/main-seed2 \
      --output docs/evidence/dynamics-2026-10-26

No model is executed unless --verify-checkpoints or --content-probe is supplied.
Checkpoint verification reexecutes the saved policy in the same simulator; it
does not constitute an independent simulator implementation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
CONDITION_ORDER = ("iid", "clock_0375", "overlap", "conflict", "long_gap", "sparse", "long")
KIND_ORDER = ("reservoir", "learned", "gru")
CONTROL_ORDER = ("baseline", "small_noise", "reset", "half_step", "clock_half", "clock_double")
TRAIN_CONFIG_KEYS = ("steps", "updates", "batch", "width", "tbptt", "train_dts", "max_step",
                     "lr", "weight_decay", "wait_weight", "clip", "deterministic")
EVAL_CONFIG_KEYS = ("eval_steps", "eval_batch", "dt", "unseen_dt", "long_factor", "max_step",
                    "wait_weight", "noise_scale", "eval_conditions", "controls", "deterministic")
WORLD_RATES = {"completion_rate": ("completed_eligible", "eligible"),
               "clean_completion_rate": ("clean_completed_eligible", "eligible"),
               "finish_precision": ("completed_all", "finish_attempted")}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ordered(values, preferred) -> list:
    return [value for value in preferred if value in values] + sorted(set(values) - set(preferred))


def moments(values) -> dict:
    """Unweighted across-seed statistics; a missing denominator stays missing."""
    values = list(values)
    observed = [float(value) for value in values if value is not None]
    require(all(math.isfinite(value) for value in observed), "nonfinite evidence value")
    return {"n": len(observed), "missing": len(values) - len(observed),
            "mean": statistics.mean(observed) if observed else None,
            "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None}


def count(value, location: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{location}: expected a nonnegative integer")
    return value


def check_rate(value, numerator: int, denominator: int, location: str, empty=None) -> None:
    expected = numerator / denominator if denominator else empty
    if expected is None:
        require(value is None, f"{location}: empty denominator must be null")
    else:
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and abs(value - expected) <= 1e-10,
                f"{location}: rate disagrees with its counts")


def check_action_metrics(metrics: dict, location: str) -> None:
    total = count(metrics["count"], location + ".count")
    for name in ("wait", "nonwait", "begin", "finish"):
        group = metrics[name]
        n = count(group["count"], location + "." + name + ".count")
        value = group["accuracy"]
        require((value is None and n == 0) or (n > 0 and value is not None and 0 <= value <= 1),
                f"{location}.{name}: accuracy/denominator inconsistency")
    require(metrics["wait"]["count"] + metrics["nonwait"]["count"] == total,
            f"{location}: WAIT/non-WAIT counts do not sum to all labels")
    require(metrics["begin"]["count"] + metrics["finish"]["count"] == metrics["nonwait"]["count"],
            f"{location}: BEGIN/FINISH counts do not sum to non-WAIT labels")


def check_rollout(row: dict, world_hash: str, location: str) -> None:
    require(row["world_hash"] == world_hash == row["world"]["world_hash"],
            f"{location}: schedule fingerprint mismatch")
    world = row["world"]
    for name, (numerator_key, denominator_key) in WORLD_RATES.items():
        numerator = count(world[numerator_key], location + "." + numerator_key)
        denominator = count(world[denominator_key], location + "." + denominator_key)
        require(numerator <= denominator, f"{location}.{name}: numerator exceeds denominator")
        check_rate(world[name], numerator, denominator, location + "." + name, empty=0.0)
    require(world["clean_completed_eligible"] <= world["completed_eligible"],
            f"{location}: clean completions exceed completions")
    pending = count(row["midpoint_pending"], location + ".midpoint_pending")
    completed = count(row["midpoint_completed"], location + ".midpoint_completed")
    clean = count(row["midpoint_clean_completed"], location + ".midpoint_clean_completed")
    require(clean <= completed <= pending, f"{location}: invalid targeted completion counts")
    check_rate(row["midpoint_completion_rate"], completed, pending, location + ".midpoint_completion_rate")
    check_rate(row["midpoint_clean_completion_rate"], clean, pending,
               location + ".midpoint_clean_completion_rate")
    for name, total in (("midpoint_pending_per_stream", pending),
                        ("midpoint_completed_per_stream", completed)):
        if name in row:
            require(sum(count(value, location + "." + name) for value in row[name]) == total,
                    f"{location}: {name} does not sum to aggregate")
    expected_interventions = int(row["control"] in {"small_noise", "reset"})
    require(row["intervention_count"] == expected_interventions,
            f"{location}: expected exactly {expected_interventions} state interventions")
    if expected_interventions:
        require(row["intervention"] is not None and row["intervention"]["tick"] == row["midpoint_tick"],
                f"{location}: intervention is not at the recorded midpoint")
    require(bool(row["prediction_hash"]) and bool(row["midpoint_target_hash"]),
            f"{location}: prediction/target identity is missing")
    check_action_metrics(row["action_metrics"], location + ".action_metrics")


def read_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        file = path / "summary.json" if path.is_dir() else path
        file = file.resolve()
        def reject_constant(value):
            raise ValueError(f"{file}: invalid JSON constant {value}")
        summary = json.loads(file.read_text(encoding="utf-8"), parse_constant=reject_constant)
        require(summary.get("status") == "complete", f"{file}: run is not complete")
        require(not summary.get("test_skipped", False), f"{file}: held-out test was skipped")
        require(summary.get("test_streams") and summary.get("models"), f"{file}: test evidence is missing")
        seed = summary["config"]["seed"]
        require(isinstance(seed, int) and not isinstance(seed, bool), f"{file}: integer seed missing")
        runs.append({"path": file, "sha256": digest(file), "seed": seed, "summary": summary})
    require(bool(runs), "at least one completed run is required")
    require(len({row["seed"] for row in runs}) == len(runs), "duplicate seeds cannot count as independent runs")
    return sorted(runs, key=lambda row: row["seed"])


def training_config(summary: dict, model: dict) -> dict:
    return model.get("training_config", summary["config"])


def training_identity(model: dict):
    streams = model.get("training", {}).get("streams")
    if streams is None:
        return None
    return [{key: stream[key] for key in ("index", "seed", "dt", "world_hash", "teacher_hash", "consumed_steps")}
            for stream in streams]


def audit_runs(runs: list[dict]) -> dict:
    reference = runs[0]["summary"]
    kinds = ordered(reference["models"], KIND_ORDER)
    first_conditions = {item["condition"] for item in reference["models"][kinds[0]]["test"]}
    conditions = ordered(first_conditions, CONDITION_ORDER)
    source_keys = ("world.py", "model.py", "run.py")
    control_sets, parameter_reports, train_configs = {}, {}, {}
    perturbation_pairs = world_pairs = validation_pairs = trajectory_pairs = checked_rollouts = 0
    validation_names = {item["condition"] for item in reference["models"][kinds[0]]["validation"]}
    for run in runs:
        summary, seed = run["summary"], run["seed"]
        require(set(summary["models"]) == set(kinds), f"seed {seed}: model kinds differ across runs")
        for name in source_keys:
            require(bool(summary["source_hashes"].get(name)), f"seed {seed}: missing {name} source hash")
            require(summary["source_hashes"][name] == reference["source_hashes"][name],
                    f"seed {seed}: incompatible {name} source across runs")
        for key in EVAL_CONFIG_KEYS:
            require(summary["config"].get(key) == reference["config"].get(key),
                    f"seed {seed}: different evaluation configuration for {key}")
        manifest = {item["name"]: item for item in summary["test_streams"]}
        require(len(manifest) == len(summary["test_streams"]), f"seed {seed}: duplicate test stream names")
        require(set(manifest) == set(conditions), f"seed {seed}: test manifest conditions differ")
        paired_training = None
        paired_validation = {}
        for kind in kinds:
            model = summary["models"][kind]
            if kind in parameter_reports:
                require(model["parameters"] == parameter_reports[kind], f"{kind}: parameter reports differ across seeds")
            parameter_reports[kind] = model["parameters"]
            config = {key: training_config(summary, model).get(key) for key in TRAIN_CONFIG_KEYS}
            require(kind not in train_configs or config == train_configs[kind],
                    f"{kind}: training budget/configuration differs across seeds")
            train_configs[kind] = config
            for name in ("world.py", "model.py"):
                require(model.get("training_source_hashes", summary["source_hashes"])[name]
                        == summary["source_hashes"][name],
                        f"seed {seed}/{kind}: trained and evaluated {name} sources differ")
            identity = training_identity(model)
            if identity is not None:
                if paired_training is None:
                    paired_training = identity
                else:
                    require(identity == paired_training, f"seed {seed}: training streams differ across models")
                    trajectory_pairs += 1
            test = {item["condition"]: item for item in model["test"]}
            require(len(test) == len(model["test"]) and set(test) == set(conditions),
                    f"seed {seed}/{kind}: incomplete or duplicate test conditions")
            require(model.get("validation"), f"seed {seed}/{kind}: validation metrics missing")
            validation = {item["condition"]: item for item in model["validation"]}
            require(len(validation) == len(model["validation"]) and set(validation) == validation_names,
                    f"seed {seed}/{kind}: validation conditions differ")
            for condition, item in validation.items():
                identity = (item["world_hash"], item["teacher_forced"]["teacher_hash"])
                require(condition not in paired_validation or paired_validation[condition] == identity,
                        f"seed {seed}/{kind}/{condition}: validation streams differ across models")
                paired_validation[condition] = identity
                require(any(row["control"] == "baseline" for row in item["closed_loop"]),
                        f"seed {seed}/{kind}/{condition}: validation baseline missing")
                for row in item["closed_loop"]:
                    check_rollout(row, item["world_hash"], f"seed {seed}/{kind}/validation/{condition}")
                    checked_rollouts += 1
                check_action_metrics(item["teacher_forced"]["action_metrics"],
                                     f"seed {seed}/{kind}/validation/{condition}/teacher_forced")
                validation_pairs += 1
            for condition, item in test.items():
                world_hash = manifest[condition]["world_hash"]
                require(item["world_hash"] == world_hash, f"seed {seed}/{kind}/{condition}: unpaired world")
                require(item["teacher_forced"]["teacher_hash"] == manifest[condition]["teacher_hash"],
                        f"seed {seed}/{kind}/{condition}: teacher trajectory mismatch")
                world_pairs += 1
                controls = {row["control"]: row for row in item["closed_loop"]}
                require(len(controls) == len(item["closed_loop"]), f"{kind}/{condition}: duplicate controls")
                require({"baseline", "small_noise", "reset"} <= set(controls),
                        f"{kind}/{condition}: baseline/noise/reset evidence is required")
                if condition in control_sets:
                    require(set(controls) == set(control_sets[condition]), f"{condition}: controls differ across runs/models")
                control_sets[condition] = ordered(controls, CONTROL_ORDER)
                baseline = controls["baseline"]
                for control in ("small_noise", "reset"):
                    row = controls[control]
                    require(row["midpoint_target_hash"] == baseline["midpoint_target_hash"],
                            f"seed {seed}/{kind}/{condition}: {control} targets differ from baseline")
                    require(row["midpoint_pending"] == baseline["midpoint_pending"],
                            f"seed {seed}/{kind}/{condition}: {control} target denominator differs")
                    perturbation_pairs += 1
                for control, row in controls.items():
                    check_rollout(row, world_hash, f"seed {seed}/{kind}/{condition}/{control}")
                    checked_rollouts += 1
                check_action_metrics(item["teacher_forced"]["action_metrics"],
                                     f"seed {seed}/{kind}/{condition}/teacher_forced")
    return {"status": "passed", "kinds": kinds, "conditions": conditions, "controls": control_sets,
            "world_fingerprint_checks": world_pairs, "midpoint_pair_checks": perturbation_pairs,
            "validation_fingerprint_checks": validation_pairs,
            "training_trajectory_pair_checks": trajectory_pairs, "checked_rollouts": checked_rollouts,
            "checks": ["Completed held-out tests and unique seeds", "Frozen world/model/runner source hashes",
                       "Equal evaluation configuration and per-model training budget across seeds",
                       "External world and teacher fingerprints match across model kinds",
                       "Baseline/noise/reset midpoint request identities match within each model",
                       "Stored rates agree with counts; interventions occur exactly once",
                       "Training trajectory hashes match across models when present"],
            "limits": ["Hash/count audits do not reconstruct action traces.",
                       "Midpoint target populations can differ across models; pairing is within each model.",
                       "Parameter count, trainable parameters and runtime differ; FLOPs are not matched."]}


def agreement_summary(rows: list[dict]) -> dict:
    return {"closed_loop_finish_accuracy": moments(row["closed_loop"] for row in rows),
            "teacher_forced_finish_accuracy": moments(row["teacher_forced"] for row in rows),
            "seed_results": rows}


def aggregate(runs: list[dict]) -> dict:
    audit = audit_runs(runs)
    grouped, agreements, costs = defaultdict(list), defaultdict(list), defaultdict(list)
    inputs = []
    for run in runs:
        summary, seed = run["summary"], run["seed"]
        metadata = {"seed": seed, "summary": str(run["path"]), "summary_sha256": run["sha256"],
                    "source_hashes": summary["source_hashes"], "config": summary["config"],
                    "environment": summary.get("environment"), "models": {}}
        for kind, model in summary["models"].items():
            training = model.get("training", {})
            cost = {key: training.get(key) for key in ("seconds", "updates", "batch_observations",
                    "action_labels", "batch_observations_per_second", "peak_cuda_bytes")}
            cost["seed"] = seed
            costs[kind].append(cost)
            metadata["models"][kind] = {"parameters": model["parameters"],
                "checkpoint": model.get("checkpoint"), "checkpoint_sha256": model.get("checkpoint_sha256"),
                "training_config": training_config(summary, model),
                "training_source_hashes": model.get("training_source_hashes", summary["source_hashes"]),
                "training_data_digest": canonical_digest(training_identity(model)) if training else None,
                "initial_parameter_hash": training.get("initial_parameter_hash"),
                "final_parameter_hash": training.get("final_parameter_hash", model.get("parameter_hash")),
                "training_cost": cost}
            for split in ("validation", "test"):
                for item in model[split]:
                    condition = item["condition"]
                    baseline = next(row for row in item["closed_loop"] if row["control"] == "baseline")
                    agreements[kind, split, condition].append({"seed": seed,
                        "closed_loop": baseline["action_metrics"]["finish"]["accuracy"],
                        "closed_loop_finish_labels": baseline["action_metrics"]["finish"]["count"],
                        "teacher_forced": item["teacher_forced"]["action_metrics"]["finish"]["accuracy"],
                        "teacher_forced_finish_labels": item["teacher_forced"]["action_metrics"]["finish"]["count"]})
                    if split != "test":
                        continue
                    for row in item["closed_loop"]:
                        record = {"seed": seed, "world_hash": row["world_hash"],
                                  "prediction_hash": row["prediction_hash"],
                                  "midpoint_target_hash": row["midpoint_target_hash"],
                                  "seconds": row["seconds"],
                                  "batch_observations_per_second": row["batch_observations_per_second"],
                                  "solver_control_applicable": row.get("solver_control_applicable"),
                                  "finish_accuracy": row["action_metrics"]["finish"]["accuracy"],
                                  "nonwait_accuracy": row["action_metrics"]["nonwait"]["accuracy"],
                                  "intervention": row["intervention"],
                                  "world_counts": {key: value for key, value in row["world"].items()
                                                   if isinstance(value, int) and not isinstance(value, bool)}}
                        for key in WORLD_RATES:
                            record[key] = row["world"][key]
                        for key in ("midpoint_pending", "midpoint_completed", "midpoint_clean_completed",
                                    "midpoint_completion_rate", "midpoint_clean_completion_rate"):
                            record[key] = row[key]
                        grouped[kind, condition, row["control"]].append(record)
        inputs.append(metadata)
    models = {}
    for kind in audit["kinds"]:
        first_model = runs[0]["summary"]["models"][kind]
        model = {"parameters": first_model["parameters"], "training_cost": {},
                 "conditions": {}, "finish_agreement": {"validation": {}, "test": {}}}
        for key in ("seconds", "updates", "batch_observations", "action_labels",
                    "batch_observations_per_second", "peak_cuda_bytes"):
            model["training_cost"][key] = moments(row[key] for row in costs[kind])
        model["training_cost"]["seed_results"] = costs[kind]
        for condition in audit["conditions"]:
            controls = {}
            for control in audit["controls"][condition]:
                rows = grouped[kind, condition, control]
                pending = sum(row["midpoint_pending"] for row in rows)
                completed = sum(row["midpoint_completed"] for row in rows)
                clean = sum(row["midpoint_clean_completed"] for row in rows)
                world_keys = set.intersection(*(set(row["world_counts"]) for row in rows))
                item = {key: moments(row[key] for row in rows) for key in WORLD_RATES}
                item.update({"n_seeds": len(rows), "world_count_totals": {
                    key: sum(row["world_counts"][key] for row in rows) for key in sorted(world_keys)},
                    "targeted": {"pending": pending, "completed": completed, "clean_completed": clean,
                        "pooled_completion_rate": completed / pending if pending else None,
                        "pooled_clean_completion_rate": clean / pending if pending else None,
                        "completion_rate": moments(row["midpoint_completion_rate"] for row in rows),
                        "clean_completion_rate": moments(row["midpoint_clean_completion_rate"] for row in rows)},
                    "finish_accuracy": moments(row["finish_accuracy"] for row in rows),
                    "nonwait_accuracy": moments(row["nonwait_accuracy"] for row in rows),
                    "seconds": moments(row["seconds"] for row in rows),
                    "batch_observations_per_second": moments(row["batch_observations_per_second"] for row in rows),
                    "seed_results": rows})
                controls[control] = item
            model["conditions"][condition] = {"controls": controls}
        for (agreement_kind, split, condition), rows in agreements.items():
            if agreement_kind == kind:
                model["finish_agreement"][split][condition] = agreement_summary(rows)
        models[kind] = model
    return {"format": "persistent-dynamics-evidence-v1", "status": "complete",
            "seeds": [row["seed"] for row in runs], "n_seeds": len(runs), "audit": audit,
            "report_source_sha256": digest(Path(__file__)),
            "report_environment": {"python": sys.version, "platform": platform.platform()},
            "definitions": {
                "mean": "Arithmetic mean of seed-level rates; each independent training seed has equal weight.",
                "sample_sd": "Sample standard deviation across seeds (n-1 denominator); null for fewer than two observations.",
                "missing": "A seed with no midpoint pending targets or no FINISH labels has a null corresponding rate and is excluded from that rate's mean.",
                "targeted_pooled_rate": "Sum of exact midpoint-request completions divided by sum of midpoint pending requests; distinct from the unweighted seed mean.",
                "clean_completion": "Eligible request completes without a rejected non-WAIT attempt while it was active.",
                "finish_precision": "Accepted FINISH actions / all FINISH attempts; zero if there are no attempts.",
                "targeted_pairing": "Eligible started unfinished requests with no same-job command at the midpoint; request IDs/masks match for baseline/noise/reset within each model and seed.",
                "finish_agreement": "Action agreement with evaluator labels. Closed-loop labels reflect the policy's own world trajectory; teacher-forced labels reflect the oracle trajectory.",
                "clock_0375": "Held-out observation-interval distribution, with separate exogenous stream seed; exact interval is recorded in input config.unseen_dt.",
                "clock_half_double": "Clock-calibration ablations: only the model's dt is scaled; world time, inputs and action opportunities are held fixed.",
                "solver_half": "Numerical control with max solver step halved and identical inputs/action opportunities; GRU has no autonomous solver.",
                "runtime": "Stored wall-clock measurements, including their original evaluator/training overhead; parameter counts and FLOPs are not matched."},
            "inputs": inputs, "models": models,
            "checkpoint_verification": {"status": "not_requested", "checks": []},
            "posthoc_content_probe": {"status": "not_requested"}}


def replay_context(entry: dict, device: str):
    """Lazy runtime setup shared by the two explicitly requested diagnostics."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from experiments.dynamics import run as runner
    config = entry["summary"]["config"]
    chosen_device = device
    if chosen_device == "auto":
        chosen_device = (entry["summary"].get("environment") or {}).get("device", "cpu")
        if str(chosen_device).startswith("cuda") and not torch.cuda.is_available():
            chosen_device = "cpu"
    torch.set_num_threads(config.get("threads", 2))
    runner.seed_all(entry["seed"], config.get("deterministic", True))
    return torch, runner, chosen_device


def load_saved_model(entry: dict, kind: str, device: str, torch, runner):
    record = entry["summary"]["models"][kind]
    require(bool(record.get("checkpoint")), f"{kind}: checkpoint path missing")
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_file():
        checkpoint = entry["path"].parent / kind / "final.pt"
    require(checkpoint.is_file(), f"checkpoint unavailable: {checkpoint}")
    expected_digest = record.get("checkpoint_sha256")
    require(expected_digest is not None and digest(checkpoint) == expected_digest,
            f"checkpoint digest differs: {checkpoint}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    current = runner.source_hashes()
    runner.check_checkpoint_sources(saved, current)
    for name in ("world.py", "model.py"):
        require(current[name] == entry["summary"]["source_hashes"][name],
                f"{kind}: current {name} differs from the recorded test source")
    model = runner.DynamicsModel(**saved["model_config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved, expected_digest


def finish_content_counts(logits, target):
    """Six integer sufficient statistics, scored after the model's forward call."""
    import torch
    require(logits.shape[-1] == 11 and logits.shape[:-1] == target.shape,
            "content probe requires [batch,2,11] logits and [batch,2] labels")
    with torch.no_grad():
        selected = target >= 3
        actual = logits.argmax(-1)
        conditional = logits[..., 3:].argmax(-1) + 3
        return torch.stack((selected.sum(), ((actual == target) & selected).sum(),
                            ((conditional == target) & selected).sum(),
                            ((actual == 0) & selected).sum(),
                            ((actual >= 1) & (actual < 3) & selected).sum(),
                            ((actual >= 3) & selected).sum()))


def content_probe(runs: list[dict], device: str = "auto") -> dict:
    """Opened-test readout analysis; no optimizer, fitting, or model feedback."""
    count_names = ("teacher_finish_labels", "actual_top1_correct", "conditional_finish_content_version_correct",
                   "top1_wait", "top1_begin", "top1_finish")
    ratios = {"actual_top1_accuracy": "actual_top1_correct",
              "conditional_finish_content_version_accuracy": "conditional_finish_content_version_correct",
              "top1_wait_fraction": "top1_wait", "top1_begin_fraction": "top1_begin",
              "top1_finish_fraction": "top1_finish"}
    grouped = defaultdict(list)
    conditions = ("iid", "long_gap")
    for entry in runs:
        torch, runner, chosen_device = replay_context(entry, device)
        summary, config = entry["summary"], entry["summary"]["config"]
        manifests = {row["name"]: row for row in summary["test_streams"]}
        require(set(conditions) <= set(manifests), "content probe requires existing IID and long_gap test streams")
        models = {kind: load_saved_model(entry, kind, chosen_device, torch, runner)
                  for kind in summary["models"]}
        for condition in conditions:
            spec = manifests[condition]
            stream = runner.generate_stream(spec["seed"], config["eval_batch"], spec["steps"],
                                            dt=spec["dt"], condition=spec["condition"])
            require(stream.fingerprint == spec["world_hash"], "content-probe regenerated world differs")
            # The existing teacher routine executes on CPU and transfers x/y once.
            trajectory = runner.teacher_trajectory(stream, chosen_device)
            require(trajectory.fingerprint == spec["teacher_hash"], "content-probe teacher trajectory differs")
            for kind, (model, saved, checkpoint_sha) in models.items():
                expected = next(row for row in summary["models"][kind]["test"] if row["condition"] == condition)
                tally = torch.zeros(len(count_names), dtype=torch.long, device=chosen_device)
                state = model.initial_state(stream.batch_size, chosen_device)
                predictions = []
                finite = torch.ones((), dtype=torch.bool, device=chosen_device)
                with torch.no_grad():
                    for tick in range(stream.steps):
                        # Target labels enter only the downstream scoring call.
                        logits, state = model.step(trajectory.observations[tick], state,
                            trajectory.elapsed[tick], max_step=saved["model_config"]["max_step"])
                        tally += finish_content_counts(logits, trajectory.targets[tick])
                        predictions.append(logits.argmax(-1))
                        finite &= torch.isfinite(logits).all() & torch.isfinite(state).all()
                require(bool(finite), f"seed {entry['seed']}/{kind}/{condition}: nonfinite content-probe readout/state")
                prediction_hash = runner.tensor_digest({"predictions": torch.stack(predictions)})
                require(prediction_hash == expected["teacher_forced"]["prediction_hash"],
                        f"seed {entry['seed']}/{kind}/{condition}: teacher-forced predictions differ")
                counts = dict(zip(count_names, tally.cpu().tolist()))
                require(counts["top1_wait"] + counts["top1_begin"] + counts["top1_finish"]
                        == counts["teacher_finish_labels"], "content-probe action-type partition is invalid")
                denominator = counts["teacher_finish_labels"]
                row = {"seed": entry["seed"], "device": chosen_device, "world_hash": stream.fingerprint,
                       "teacher_hash": trajectory.fingerprint, "prediction_hash": prediction_hash,
                       "checkpoint_sha256": checkpoint_sha, "counts": counts}
                row.update({name: counts[numerator] / denominator if denominator else None
                            for name, numerator in ratios.items()})
                grouped[kind, condition].append(row)
                print(json.dumps({"event": "posthoc_content_probe", "seed": entry["seed"],
                    "kind": kind, "condition": condition, "teacher_finish_labels": denominator,
                    "actual_top1_accuracy": row["actual_top1_accuracy"],
                    "conditional_finish_content_version_accuracy": row["conditional_finish_content_version_accuracy"]}),
                      flush=True)
    models = {}
    for kind in ordered(runs[0]["summary"]["models"], KIND_ORDER):
        models[kind] = {}
        for condition in conditions:
            rows = grouped[kind, condition]
            totals = {key: sum(row["counts"][key] for row in rows) for key in count_names}
            denominator = totals["teacher_finish_labels"]
            models[kind][condition] = {"seed_results": rows,
                "across_seeds": {key: moments(row[key] for row in rows) for key in ratios},
                "pooled": {"counts": totals, **{name: totals[numerator] / denominator if denominator else None
                                                for name, numerator in ratios.items()}}}
    return {"status": "complete", "analysis_class": "post_hoc_opened_test_reanalysis",
            "learning": False, "tuning": False, "new_decoder": False, "conditions": list(conditions),
            "runtime_source_hashes": runner.source_hashes(),
            "method": "Reuse the saved checkpoint and exact existing test teacher trajectory. At target>=3 only, score 11-way top-1, argmax(logits[...,3:])+3 over the eight FINISH content/version classes, and the actual top-1 WAIT/BEGIN/FINISH fractions.",
            "interpretation": "This probes recoverable content/version information in the existing readout at teacher-chosen FINISH moments. It is a teacher-forced diagnostic; it does not establish closed-loop timing competence or isolate every cause of failure.",
            "models": models}


def verify_checkpoints(runs: list[dict], conditions: list[str], device: str = "auto") -> dict:
    """Optional expensive replay; called only by the explicit CLI switch."""
    checks = []
    for entry in runs:
        torch, runner, chosen_device = replay_context(entry, device)
        summary = entry["summary"]
        config = summary["config"]
        manifests = {row["name"]: row for row in summary["test_streams"]}
        for condition in conditions:
            require(condition in manifests, f"verification condition not in run: {condition}")
        for kind, model_record in summary["models"].items():
            model, saved, expected_digest = load_saved_model(entry, kind, chosen_device, torch, runner)
            test = {row["condition"]: row for row in model_record["test"]}
            for condition in conditions:
                spec = manifests[condition]
                stream = runner.generate_stream(spec["seed"], config["eval_batch"], spec["steps"],
                                                dt=spec["dt"], condition=spec["condition"])
                require(stream.fingerprint == spec["world_hash"], "regenerated world fingerprint differs")
                actual = runner.closed_loop_rollout(model, stream, device=chosen_device,
                    max_step=saved["model_config"]["max_step"], wait_weight=config["wait_weight"])
                expected = next(row for row in test[condition]["closed_loop"] if row["control"] == "baseline")
                require(actual["prediction_hash"] == expected["prediction_hash"],
                        f"seed {entry['seed']}/{kind}/{condition}: prediction replay differs")
                require(actual["world"] == expected["world"],
                        f"seed {entry['seed']}/{kind}/{condition}: world metrics replay differs")
                require(actual["midpoint_target_hash"] == expected["midpoint_target_hash"],
                        f"seed {entry['seed']}/{kind}/{condition}: midpoint replay differs")
                checks.append({"seed": entry["seed"], "kind": kind, "condition": condition,
                               "device": chosen_device, "checkpoint_sha256": expected_digest,
                               "prediction_hash": actual["prediction_hash"], "world_metrics_match": True,
                               "midpoint_target_match": True})
                print(json.dumps({"event": "verified", **checks[-1]}), flush=True)
    return {"status": "passed", "checks": checks,
            "method": "Regenerate schedules and rerun saved baseline policies in the frozen simulator; compare full prediction hash, world metrics and midpoint target identity.",
            "limit": "This replay reuses the original simulator; it is not an independent simulator implementation."}


def plots(summary: dict, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "persistent-dynamics-evidence-v1"})
    kinds, conditions = summary["audit"]["kinds"], summary["audit"]["conditions"]
    colors = {"reservoir": "#74818a", "learned": "#186c9b", "gru": "#d18a33"}
    labels = {"reservoir": "Frozen reservoir", "learned": "Learned dynamics", "gru": "GRU"}
    condition_labels = {"iid": "IID", "clock_0375": "Clock 0.375", "overlap": "Overlap",
                        "conflict": "Conflict", "long_gap": "Long wait", "sparse": "Sparse", "long": "Long stream"}

    def draw(ax, groups, series, values, xlabels):
        x = np.arange(len(groups))
        width = .78 / max(1, len(series))
        upper = 1.03
        for index, kind in enumerate(series):
            stats = [values(kind, group) for group in groups]
            means = [value["mean"] if value["mean"] is not None else np.nan for value in stats]
            deviations = [value["sample_sd"] or 0 for value in stats]
            upper = max(upper, max((mean + deviation + .03 for mean, deviation in zip(means, deviations)
                                    if math.isfinite(mean)), default=1.03))
            ax.bar(x + (index - (len(series) - 1) / 2) * width, means, width,
                   color=colors.get(kind, "#555555"), label=labels.get(kind, kind),
                   yerr=deviations, capsize=2, error_kw={"elinewidth": 1})
        ax.set_xticks(x, xlabels)
        ax.set_ylim(0, upper)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [3.8, 2, 1.65]})
    draw(axes[0], conditions, kinds,
         lambda kind, condition: summary["models"][kind]["conditions"][condition]["controls"]["baseline"]["clean_completion_rate"],
         [condition_labels.get(condition, condition) for condition in conditions])
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("(a) Closed-loop clean completion", loc="left", fontsize=11)
    if "iid" in conditions:
        draw(axes[1], ["baseline", "small_noise", "reset"], kinds,
             lambda kind, control: summary["models"][kind]["conditions"]["iid"]["controls"][control]["targeted"]["completion_rate"],
             ["Baseline", "Noise", "Reset"])
        ode_kinds = [kind for kind in kinds if kind != "gru"]
        if "half_step" in summary["audit"]["controls"]["iid"]:
            solver_step = summary["inputs"][0]["config"]["max_step"]
            draw(axes[2], ["baseline", "half_step"], ode_kinds,
                 lambda kind, control: summary["models"][kind]["conditions"]["iid"]["controls"][control]["clean_completion_rate"],
                 [f"Step {solver_step:g}", f"Step {solver_step / 2:g}"])
        else:
            axes[2].text(.5, .5, "No half-step evidence", ha="center", va="center", transform=axes[2].transAxes)
    axes[1].set_title("(b) Existing IID requests", loc="left", fontsize=11)
    axes[1].set_ylabel("Midpoint-request completion")
    axes[2].set_title("(c) ODE solver step (IID)", loc="left", fontsize=11)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(kinds), frameon=False,
               bbox_to_anchor=(.5, 1.01))
    fig.text(.06, .025, f"{summary['n_seeds']} independent training seeds. Bars: seed means; error bars: sample SD. "
             "Midpoint requests are paired within each model; their populations can differ across models.", fontsize=9)
    fig.subplots_adjust(left=.055, right=.99, top=.85, bottom=.23, wspace=.32)
    artifacts = {}
    for suffix in ("png", "svg"):
        path = output / ("overview." + suffix)
        fig.savefig(path, dpi=180, metadata={"Date": None} if suffix == "svg" else None)
        artifacts[path.name] = {"path": str(path), "sha256": digest(path)}
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    groups = [("validation", "closed_loop_finish_accuracy", "Validation: closed loop"),
              ("validation", "teacher_forced_finish_accuracy", "Validation: teacher forced"),
              ("test", "closed_loop_finish_accuracy", "Test IID: closed loop"),
              ("test", "teacher_forced_finish_accuracy", "Test IID: teacher forced")]
    x = np.arange(len(kinds))
    shades = ("#9cb8c7", "#36728f", "#e2bf83", "#ad7324")
    width = .18
    for index, (split, metric, label) in enumerate(groups):
        stats = [summary["models"][kind]["finish_agreement"][split].get("iid", {}).get(metric,
                 {"mean": None, "sample_sd": None}) for kind in kinds]
        ax.bar(x + (index - 1.5) * width,
               [row["mean"] if row["mean"] is not None else np.nan for row in stats], width,
               yerr=[row["sample_sd"] or 0 for row in stats], capsize=2, color=shades[index], label=label)
    ax.set_xticks(x, [labels.get(kind, kind) for kind in kinds])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("FINISH label accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.18), fontsize=9)
    fig.text(.08, .025, "Secondary action agreement. Closed-loop FINISH labels depend on each policy's own actions; "
             "teacher-forced labels follow the oracle.", fontsize=8.5)
    fig.subplots_adjust(left=.08, right=.98, top=.8, bottom=.17)
    for suffix in ("png", "svg"):
        path = output / ("finish-agreement." + suffix)
        fig.savefig(path, dpi=180, metadata={"Date": None} if suffix == "svg" else None)
        artifacts[path.name] = {"path": str(path), "sha256": digest(path)}
    plt.close(fig)
    summary["report_environment"].update(matplotlib=matplotlib.__version__, numpy=np.__version__)
    return artifacts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runs", type=Path, nargs="+", required=True, help="completed run directories or summary.json files")
    result.add_argument("--output", type=Path, default=ROOT / "docs" / "evidence" / "dynamics-2026-10-26")
    result.add_argument("--verify-checkpoints", action="store_true", help="explicitly rerun saved policies and check baseline prediction hashes")
    result.add_argument("--verify-conditions", nargs="+", default=["iid"])
    result.add_argument("--content-probe", action="store_true", help="post hoc readout analysis on existing IID/long_gap teacher trajectories; no training")
    result.add_argument("--verify-device", "--probe-device", default="auto", help="device for either diagnostic; auto uses each run's recorded device")
    return result


def main(argv=None) -> dict:
    args = parser().parse_args(argv)
    runs = read_runs(args.runs)
    summary = aggregate(runs)
    if args.verify_checkpoints:
        summary["checkpoint_verification"] = verify_checkpoints(runs, args.verify_conditions, args.verify_device)
    if args.content_probe:
        summary["posthoc_content_probe"] = content_probe(runs, args.verify_device)
    output = args.output.resolve()
    require(all(output / "summary.json" != entry["path"] for entry in runs),
            "output directory must differ from input run directories")
    output.mkdir(parents=True, exist_ok=True)
    summary["artifacts"] = plots(summary, output)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "summary.json")
    print(json.dumps({"event": "report_complete", "output": str(output), "seeds": summary["seeds"]}), flush=True)
    return summary


if __name__ == "__main__":
    main()
