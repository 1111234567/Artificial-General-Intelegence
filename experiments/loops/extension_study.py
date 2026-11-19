"""Freeze and execute a finite function-preserving recurrent-field comparison."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, state_digest
from experiments.credit.run import file_sha256
from experiments.integration.diagnose import diagnose_fork
from experiments.loops.audit_run import verify_files
from experiments.loops.model import ARMS
from experiments.loops.train import LoopTrainer, main as train_main

ROOT = Path(__file__).resolve().parents[2]


def prepare(folder, raw, audit_summary):
    folder, raw = Path(folder).resolve(), Path(raw).resolve()
    protocol = folder / "extension-protocol-v1.json"
    if protocol.exists():
        raise FileExistsError(protocol)
    audit_summary = Path(audit_summary).resolve()
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    # The existing-model diagnosis must have completed before this next study.
    if (audit.get("complete") is not True or audit.get("audit", {}).get("status") != "passed"
            or audit.get("audit", {}).get("errors") != []
            or any(audit.get("counts", {}).get(key) != value for key, value in
                   {"sources": 9, "jobs": 27, "forks": 540, "structural_summaries": 9}.items())):
        raise ValueError("completed existing-model audit required")
    source_paths = [ROOT / "experiments" / group / name for group, names in {
        "loops": ["__init__.py", "model.py", "train.py", "test_model.py", "test_train.py", "extension_study.py", "audit_run.py", "structure.py", "interventions.py"],
        "continuity": ["model.py", "world.py", "run.py", "evaluate.py"],
        "integration": ["run.py", "diagnose.py"],
        "credit": ["run.py", "local.py", "batch.py", "windows_runner.py", "evaluate_ready.py"],
    }.items() for name in names]
    hashes = {str(path): file_sha256(path) for path in source_paths}
    branches, historical = [], []
    for seed in (0, 1, 2):
        source = ROOT / f"data/continuity/main-20261027-v1/seed_{seed:03d}/online128/checkpoint.pt"
        source_hash = file_sha256(source)
        old_bp = ROOT / f"data/credit/main-v1/seed_{seed:03d}/bp64/checkpoint.pt"
        historical.append({"seed": seed, "path": str(old_bp), "sha256": file_sha256(old_bp)})
        for arm in ARMS:
            branches.append({"id": f"seed_{seed:03d}_{arm}", "seed": seed, "arm": arm,
                "source": str(source), "source_sha256": source_hash,
                "output": str(raw / "training" / f"seed_{seed:03d}" / arm)})
    plan = {"format": "loops-extension-protocol-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": hashes, "completed_existing_audit": str(audit_summary),
        "completed_existing_audit_sha256": file_sha256(audit_summary), "branches": branches,
        "historical_bp64_checkpoints": historical,
        "training_device": "cuda", "evaluation_device": "cpu", "target_ticks": 65536,
        "chunk_ticks": 64, "conditions": ["iid", "long_wait", "longer", "sparse", "long_sparse"],
        "evaluation_steps": 2048, "diagnostic_conditions": ["longer", "long_sparse"],
        "solver_refinement_max_step": .125, "dt": .25,
        "training_time_sampling": {"inherited_train_dts": [.25], "dt_every": 512,
            "note": "Verified in all three bound source checkpoints; no change to inherited clock or dt history."},
        "equations": {"baseline": "f=-h+W r", "source_square": "f=-h+W r+V(r*r)",
            "interaction": "f=-h+W r+r*(V r)", "r": "tanh(h)"},
        "parameters": {"state_scalars": 128, "baseline": 34348, "source_square": 50732, "interaction": 50732},
        "hypothesis": "Recipient/source products improve learning and transfer relative to a source-square branch with equal added parameters and recurrent matrix arithmetic.",
        "initialization": "V exactly zero; full inherited W/U/head, h/world/clocks/old Adam/RNG retained. New V has fresh Adam moments and same optimizer hyperparameters.",
        "learning": "Exact implemented Heun BPTT64; inherited Adam lr, one joint clip and update, external causal teacher labels on own-action history; no replay or local-credit changes.",
        "limits": ["All arms already have complete directed graphs; this tests richer nonlinear field interactions, not newly invented graph cycles.",
            "Two extension arms match added scalars and dense MACs; their features, derivatives and conditioning differ.",
            "Joint clipping includes V, so old parameters can change differently at the first optimizer step.",
            "Squared features and recipient/source products both break the odd symmetry of the original autonomous field.",
            "Longer-horizon timing is not general reasoning or continual-learning mastery.",
            "Setting learned V to zero at evaluation is a counterfactual ablation, not an alternative trained model."]}
    atomic_json(protocol, plan)
    files = {**hashes, str(protocol): file_sha256(protocol), str(audit_summary): file_sha256(audit_summary)}
    train_jobs, eval_jobs = [], []
    for item in branches:
        output = Path(item["output"])
        train_jobs.append({"id": item["id"], "command": [sys.executable, "-m", "experiments.loops.extension_study",
            "train", "--protocol", str(protocol), "--branch-id", item["id"]],
            "outputs": [str(output / "checkpoint.pt"), str(output / "launch-complete.json")],
            "log": str(raw / "logs" / f"train_{item['id']}.log")})
        eval_jobs.append({"id": item["id"], "command": [sys.executable, "-m", "experiments.loops.extension_study",
            "evaluate", "--protocol", str(protocol), "--branch-id", item["id"],
            "--checkpoint", str(output / "checkpoint.pt"), "--output", str(raw / "evaluation" / f"{item['id']}.json")],
            "outputs": [str(raw / "evaluation" / f"{item['id']}.json")],
            "log": str(raw / "logs" / f"evaluate_{item['id']}.log")})
    for name, jobs in (("extension-training-commands-v1.json", train_jobs), ("extension-evaluation-commands-v1.json", eval_jobs)):
        atomic_json(folder / name, {"cwd": str(ROOT), "file_hashes": files, "jobs": jobs})
    print("Frozen 9 training branches and 75 evaluation forks (45 autonomous, 18 solver checks, 12 branch ablations).")


def read_plan(protocol, branch_id):
    protocol = Path(protocol)
    original_hash = file_sha256(protocol)
    plan = json.loads(protocol.read_text(encoding="utf-8"))
    if plan["format"] != "loops-extension-protocol-v1":
        raise ValueError("unknown extension protocol")
    verify_files(plan["sources"])
    if file_sha256(plan["completed_existing_audit"]) != plan["completed_existing_audit_sha256"]:
        raise ValueError("prior audit evidence changed")
    item = next(x for x in plan["branches"] if x["id"] == branch_id)
    if file_sha256(item["source"]) != item["source_sha256"]:
        raise ValueError("A source checkpoint changed")
    for reference in plan.get("historical_bp64_checkpoints", []):
        if file_sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("historical baseline reference changed")
    return plan, item, original_hash


def validate_continuation(trainer, plan, item, *, complete=False):
    origin = trainer.loop_origin
    if (Path(origin.get("checkpoint", "")).resolve() != Path(item["source"]).resolve()
            or origin.get("checkpoint_sha256") != item["source_sha256"]):
        raise ValueError("checkpoint origin source provenance mismatch")
    if trainer.arm != item["arm"] or trainer.config.seed != item["seed"]:
        raise ValueError("checkpoint branch/seed provenance mismatch")
    if trainer.credit_config.chunk_ticks != plan["chunk_ticks"]:
        raise ValueError("checkpoint chunk_ticks differs from protocol")
    sampling = plan.get("training_time_sampling")
    if sampling and (list(trainer.config.train_dts) != sampling["inherited_train_dts"]
                     or trainer.config.dt_every != sampling["dt_every"]):
        raise ValueError("checkpoint time sampling differs from inherited protocol")
    if (trainer.credit_config.mode != "bp64" or tuple(trainer.credit_config.duration_range) != (4., 8.)
            or tuple(trainer.credit_config.gap_range) != (4., 12.)):
        raise ValueError("checkpoint training configuration differs from protocol")
    regime = trainer.world.regime
    if (tuple(regime.duration_range) != (4., 8.) or tuple(regime.gap_range) != (4., 12.)
            or regime.name != "loops-B" or len(trainer.regime_changes) != 1):
        raise ValueError("checkpoint live training regime differs from protocol")
    ticks = trainer.stage_ticks
    if trainer.hidden_resets or ticks < 0 or ticks > plan["target_ticks"] or ticks % plan["chunk_ticks"]:
        raise ValueError("checkpoint is not a valid continuous training boundary")
    if complete and ticks != plan["target_ticks"]:
        raise ValueError("checkpoint has incomplete stage ticks")


def validate_marker(marker, plan, item, protocol_hash, checkpoint_hash):
    if (marker.get("format") != "loops-launch-complete-v1" or marker.get("branch") != item
            or marker.get("protocol_sha256") != protocol_hash
            or marker.get("checkpoint_sha256") != checkpoint_hash
            or marker.get("target") != plan["target_ticks"]
            or marker.get("stage_ticks") != plan["target_ticks"] or marker.get("hidden_resets") != 0):
        raise ValueError("training completion provenance mismatch")


def comparable_semantic_hash(trainer, training_device):
    value = trainer.semantic_state()
    # Device migration is the only intentional semantic metadata difference.
    # Tensor content and all saved RNG states remain part of the digest.
    value["credit"]["continuation"]["config"]["device"] = training_device
    return state_digest(value)


def verify_end(protocol, branch_id, expected_hash):
    if file_sha256(protocol) != expected_hash:
        raise AssertionError("study protocol changed during execution")
    read_plan(protocol, branch_id)  # Recheck sources, historical A and prior audit.


def train(protocol, branch_id):
    plan, item, protocol_hash = read_plan(protocol, branch_id)
    output = Path(item["output"])
    complete = output / "launch-complete.json"
    if complete.exists():
        marker = json.loads(complete.read_text(encoding="utf-8"))
        validate_marker(marker, plan, item, protocol_hash, file_sha256(output / "checkpoint.pt"))
        with _preserve_process_state():
            trainer = LoopTrainer.from_checkpoint(output / "checkpoint.pt", device=plan["training_device"])
            validate_continuation(trainer, plan, item, complete=True)
            if comparable_semantic_hash(trainer, plan["training_device"]) != marker["semantic_hash"]:
                raise ValueError("completed checkpoint semantic provenance mismatch")
        verify_end(protocol, branch_id, protocol_hash)
        return
    started = output.with_name(output.name + ".launch-start.json")
    binding = {"format": "loops-launch-start-v1", "protocol_sha256": protocol_hash, "branch": item}
    if started.exists():
        if json.loads(started.read_text(encoding="utf-8")) != binding:
            raise ValueError("partial training protocol provenance mismatch")
    elif (output / "checkpoint.pt").exists():
        raise ValueError("partial checkpoint has no protocol provenance binding")
    else:
        atomic_json(started, binding)
    command = ["--source", item["source"], "--arm", item["arm"], "--output", str(output),
               "--target-ticks", str(plan["target_ticks"]), "--device", plan["training_device"]]
    if (output / "checkpoint.pt").exists():
        with _preserve_process_state():
            partial = LoopTrainer.from_checkpoint(output / "checkpoint.pt", device=plan["training_device"])
            validate_continuation(partial, plan, item)
        del partial
        command.append("--resume")
    train_main(command)
    trainer = LoopTrainer.from_checkpoint(output / "checkpoint.pt", device=plan["training_device"])
    validate_continuation(trainer, plan, item, complete=True)
    verify_end(protocol, branch_id, protocol_hash)
    atomic_json(complete, {"format": "loops-launch-complete-v1", "protocol_sha256": protocol_hash,
        "branch": item, "stage_ticks": trainer.stage_ticks, "target": plan["target_ticks"],
        "checkpoint_sha256": file_sha256(output / "checkpoint.pt"),
        "semantic_hash": comparable_semantic_hash(trainer, plan["training_device"]), "hidden_resets": trainer.hidden_resets})


def evaluate(protocol, branch_id, checkpoint, output):
    plan, item, protocol_hash = read_plan(protocol, branch_id)
    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    if checkpoint != (Path(item["output"]) / "checkpoint.pt").resolve() or output.exists():
        raise ValueError("checkpoint differs from declared branch or output already exists")
    checkpoint_hash = file_sha256(checkpoint)
    marker = json.loads((checkpoint.parent / "launch-complete.json").read_text(encoding="utf-8"))
    validate_marker(marker, plan, item, protocol_hash, checkpoint_hash)
    with _preserve_process_state():
        trainer = LoopTrainer.from_checkpoint(checkpoint, device=plan["evaluation_device"])
        before = state_digest(trainer.semantic_state())
        validate_continuation(trainer, plan, item, complete=True)
        normalized_hash = comparable_semantic_hash(trainer, plan["training_device"])
        if normalized_hash != marker["semantic_hash"]:
            raise ValueError("checkpoint semantic provenance mismatch")
        records = []
        for condition in plan["conditions"]:
            shared = None
            treatments = ["none"] + (["solver_half"] + (["branch_zero"] if item["arm"] != "baseline" else [])
                if condition in plan["diagnostic_conditions"] else [])
            for treatment in treatments:
                model = deepcopy(trainer.model).eval()
                if treatment == "solver_half":
                    model.max_step = plan["solver_refinement_max_step"]
                elif treatment == "branch_zero":
                    with torch.no_grad():
                        model.branch.zero_()
                record = diagnose_fork(model, trainer.world, trainer.h, condition=condition,
                    steps=plan["evaluation_steps"], dt=plan["dt"], device=plan["evaluation_device"])
                history = {key: record[key] for key in ("start_world_state_hash", "start_hidden_hash",
                    "start_scheduler_hash", "end_scheduler_hash", "end_event_hash", "primary_eligible_issued")}
                history["primary_fingerprint"] = record["primary"]["request_fingerprint"]
                if shared is not None and shared != history:
                    raise AssertionError("diagnostic interventions changed external requests")
                shared = history
                record["extension_intervention"] = treatment
                records.append(record)
        if state_digest(trainer.semantic_state()) != before:
            raise AssertionError("evaluation changed training continuation")
        result = {"format": "loops-extension-evaluation-v1", "protocol_sha256": protocol_hash,
            "branch": item, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "checkpoint_state_hash": before, "source_unchanged": True,
            "training_device_normalized_state_hash": normalized_hash,
            "model": trainer.model.parameter_report(), "branch_metrics": trainer.model.branch_metrics(trainer.h),
            "evaluation_device": plan["evaluation_device"], "records": records}
    verify_end(protocol, branch_id, protocol_hash)
    if file_sha256(checkpoint) != checkpoint_hash:
        raise AssertionError("evaluation input bytes changed")
    atomic_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--branch-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--audit-summary", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare(args.output, args.raw, args.audit_summary)
    elif args.mode == "train":
        train(args.protocol, args.branch_id)
    else:
        evaluate(args.protocol, args.branch_id, args.checkpoint, args.output)
