"""Finite full-state recurrent rewiring study: inherited A -> B -> A return."""
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

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["iid", "long_wait", "longer", "sparse", "long_sparse"]
PHASES = ("B", "A2")


def prepare(folder, raw):
    folder, raw = Path(folder).resolve(), Path(raw).resolve()
    protocol = folder / "study-protocol-v1.json"
    if protocol.exists():
        raise FileExistsError(protocol)
    names = {
        "rewiring": ["__init__.py", "plasticity.py", "run.py", "test_plasticity.py", "test_run.py", "study.py", "test_study.py"],
        "continuity": ["model.py", "world.py", "run.py", "evaluate.py"],
        "integration": ["run.py", "diagnose.py"],
        "credit": ["run.py", "local.py", "batch.py", "windows_runner.py", "evaluate_ready.py"],
    }
    paths = [ROOT / "experiments" / group / name for group, files in names.items() for name in files]
    sources = {str(p): file_sha256(p) for p in paths}
    arms = [{"id": "dense", "arm": "dense", "shrink_step": 0.}]
    arms += [{"id": f"{arm}_{dose}", "arm": arm, "shrink_step": step}
             for dose, step in (("low", 5e-5), ("high", 2e-4))
             for arm in ("retire", "learned", "random")]
    branches, historical = [], []
    for seed in (0, 1, 2):
        source = ROOT / f"data/continuity/main-20261027-v1/seed_{seed:03d}/online128/checkpoint.pt"
        old = ROOT / f"data/credit/main-v1/seed_{seed:03d}/bp64/checkpoint.pt"
        historical.append({"seed": seed, "path": str(old), "sha256": file_sha256(old)})
        for arm in arms:
            branches.append({"id": f"seed_{seed:03d}_{arm['id']}", "seed": seed,
                "condition_id": arm["id"], "arm": arm["arm"], "shrink_step": arm["shrink_step"],
                "structural_seed": 91000 + seed, "source": str(source), "source_sha256": file_sha256(source),
                "output": str(raw / "training" / f"seed_{seed:03d}" / arm["id"])})
    plan = {"format": "rewiring-study-protocol-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources, "branches": branches, "arms": arms, "seeds": [0, 1, 2],
        "phase_ticks": 65536, "target_ticks": 131072, "window_updates": 8, "chunk_ticks": 64,
        "training_device": "cuda", "evaluation_device": "cpu", "evaluation_steps": 2048,
        "dt": .25, "conditions": CONDITIONS, "phases": list(PHASES),
        "historical_bp64_checkpoints": historical,
        "regimes": {"B": {"duration": [4., 8.], "gap": [4., 12.]},
                    "A2": {"duration": [1., 4.], "gap": [.5, 2.]}},
        "primary": "Strict clean completion of mature post-fork final-valid requests; inherited/carryover/censored separate.",
        "hypothesis": "Gradient-directed independent births improve adaptation and return learning relative to the same shrink-only and random-placement rules.",
        "dose_policy": "Both predeclared absolute shrink steps are reported; no dose selected using evaluation outcomes.",
        "controls": "Dense is the maximum-capacity envelope. Retire shares shrink but disallows births. Random relocates its own proposed count and signed magnitudes within each recipient row; later capacity trajectories are not forced equal.",
        "limits": ["Every source starts fully connected; first births reactivate previously retired pairs.",
            "Fixed-sparse or peak-static initialization would change inherited functions; no hard source pruning is introduced.",
            "No extra neurons or allowed edge positions; graph support within 128 by 128 changes.",
            "Adam followed by Euclidean soft threshold is a specified heuristic, not exact proximal Adam for one L1 objective.",
            "Dormant shadow Adam stores candidate evidence, using inherited lr/betas/eps; candidate storage and dense arithmetic are counted separately from active edges.",
            "Structural window and shrink strengths are experimental choices, not learned birth/death rules.",
            "Three training seeds and a synthetic control task do not establish general intelligence or lifelong plasticity."]}
    atomic_json(protocol, plan)
    frozen = {**sources, str(protocol): file_sha256(protocol)}
    train_jobs, eval_jobs = [], []
    for item in branches:
        out = Path(item["output"])
        base = [sys.executable, "-m", "experiments.rewiring.study"]
        tail = ["--protocol", str(protocol), "--branch-id", item["id"]]
        train_jobs.append({"id": item["id"], "command": base + ["train"] + tail,
            "outputs": [str(out / f"checkpoint_{phase}.pt") for phase in PHASES] + [str(out / "launch-complete.json")],
            "log": str(raw / "logs" / f"train_{item['id']}.log")})
        eval_jobs.append({"id": item["id"], "command": base + ["evaluate"] + tail +
            ["--checkpoint", str(out / "checkpoint.pt"), "--output", str(raw / "evaluation" / f"{item['id']}.json")],
            "outputs": [str(raw / "evaluation" / f"{item['id']}.json")],
            "log": str(raw / "logs" / f"evaluate_{item['id']}.log")})
    for label, jobs in (("training", train_jobs), ("evaluation", eval_jobs)):
        atomic_json(folder / f"{label}-commands-v1.json", {"cwd": str(ROOT), "file_hashes": frozen, "jobs": jobs})
    print("Frozen 21 continuations, 42 full phase checkpoints and 210 evaluation forks.")


def read_plan(path, branch_id):
    sha = file_sha256(path)
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["format"] != "rewiring-study-protocol-v1":
        raise ValueError("unknown study protocol")
    verify_files(plan["sources"])
    item = next(row for row in plan["branches"] if row["id"] == branch_id)
    verify_files({item["source"]: item["source_sha256"]})
    return plan, item, sha


def normalized_hash(trainer, device):
    value = trainer.semantic_state()
    value["credit"]["continuation"]["config"]["device"] = device
    return state_digest(value)


def validate(trainer, plan, item, ticks=None):
    # Class-level loader validates structural buffer and Adam invariants.
    config = trainer.rewire_config
    for key in ("arm", "shrink_step", "structural_seed"):
        if getattr(config, key) != item[key]:
            raise ValueError(f"foreign checkpoint {key}")
    if config.window_updates != plan["window_updates"] or trainer.config.seed != item["seed"]:
        raise ValueError("checkpoint window/seed differs")
    origin = trainer.origin
    if (Path(origin["checkpoint"]).resolve() != Path(item["source"]).resolve()
            or origin["checkpoint_sha256"] != item["source_sha256"]):
        raise ValueError("checkpoint source differs")
    b, a = plan["regimes"]["B"], plan["regimes"]["A2"]
    if (tuple(trainer.credit_config.duration_range) != tuple(b["duration"])
            or tuple(trainer.credit_config.gap_range) != tuple(b["gap"])
            or trainer.credit_config.regime_name != "rewiring-B"
            or tuple(trainer.return_regime["duration_range"]) != tuple(a["duration"])
            or tuple(trainer.return_regime["gap_range"]) != tuple(a["gap"])
            or trainer.return_regime["name"] != "rewiring-A2"):
        raise ValueError("checkpoint future schedule differs")
    if (trainer.hidden_resets != 0 or trainer.credit_config.chunk_ticks != plan["chunk_ticks"]
            or trainer.phase_ticks != plan["phase_ticks"] or trainer.stage_ticks % plan["chunk_ticks"]
            or not 0 <= trainer.stage_ticks <= plan["target_ticks"]
            or (ticks is not None and trainer.stage_ticks != ticks)):
        raise ValueError("checkpoint incomplete or discontinuous")
    expected_returned = trainer.stage_ticks > plan["phase_ticks"]
    regime = plan["regimes"]["A2" if expected_returned else "B"]
    if (trainer.phase_returned != expected_returned
            or trainer.world.regime.name != ("rewiring-A2" if expected_returned else "rewiring-B")
            or tuple(trainer.world.regime.duration_range) != tuple(regime["duration"])
            or tuple(trainer.world.regime.gap_range) != tuple(regime["gap"])
            or len(trainer.regime_changes) != (2 if expected_returned else 1)):
        raise ValueError("checkpoint live regime differs")


def verify_end(path, branch_id, sha):
    if file_sha256(path) != sha:
        raise ValueError("protocol changed")
    read_plan(path, branch_id)


def completed(path, plan, item, sha):
    marker = json.loads((Path(item["output"]) / "launch-complete.json").read_text(encoding="utf-8"))
    if marker.get("format") != "rewiring-launch-complete-v1" or marker.get("branch") != item or marker.get("protocol_sha256") != sha:
        raise ValueError("completion identity differs")
    for phase in PHASES:
        cp = Path(item["output"]) / f"checkpoint_{phase}.pt"
        if file_sha256(cp) != marker["checkpoints"][phase]["sha256"]:
            raise ValueError("phase checkpoint bytes differ")
    return marker


def train(path, branch_id):
    from experiments.rewiring.run import RewireTrainer, main as train_main
    plan, item, sha = read_plan(path, branch_id)
    out = Path(item["output"])
    if (out / "launch-complete.json").exists():
        marker = completed(path, plan, item, sha)
        with _preserve_process_state():
            for phase, tick in (("B", plan["phase_ticks"]), ("A2", plan["target_ticks"])):
                trainer = RewireTrainer.from_checkpoint(out / f"checkpoint_{phase}.pt", device=plan["evaluation_device"])
                validate(trainer, plan, item, tick)
                if normalized_hash(trainer, plan["training_device"]) != marker["checkpoints"][phase]["semantic_hash"]:
                    raise ValueError("completed semantic hash differs")
        verify_end(path, branch_id, sha)
        return
    binding = {"format": "rewiring-launch-start-v1", "protocol_sha256": sha, "branch": item}
    start = out.with_name(out.name + ".launch-start.json")
    if start.exists():
        if json.loads(start.read_text(encoding="utf-8")) != binding:
            raise ValueError("partial run protocol differs")
    elif (out / "checkpoint.pt").exists():
        raise ValueError("unbound partial checkpoint")
    else:
        atomic_json(start, binding)
    command = ["--source", item["source"], "--output", str(out), "--arm", item["arm"],
        "--shrink-step", str(item["shrink_step"]), "--structural-seed", str(item["structural_seed"]),
        "--window-updates", str(plan["window_updates"]), "--phase-ticks", str(plan["phase_ticks"]),
        "--target-ticks", str(plan["target_ticks"]), "--device", plan["training_device"]]
    if (out / "checkpoint.pt").exists():
        with _preserve_process_state():
            partial = RewireTrainer.from_checkpoint(out / "checkpoint.pt", device=plan["training_device"])
            validate(partial, plan, item)
        del partial
        command.append("--resume")
    train_main(command)
    checkpoints = {}
    with _preserve_process_state():
        for phase, tick in (("B", plan["phase_ticks"]), ("A2", plan["target_ticks"])):
            cp = out / f"checkpoint_{phase}.pt"
            trainer = RewireTrainer.from_checkpoint(cp, device=plan["training_device"])
            validate(trainer, plan, item, tick)
            checkpoints[phase] = {"path": str(cp), "sha256": file_sha256(cp),
                "semantic_hash": normalized_hash(trainer, plan["training_device"]), "stage_ticks": tick}
    verify_end(path, branch_id, sha)
    atomic_json(out / "launch-complete.json", {"format": "rewiring-launch-complete-v1", "branch": item,
        "protocol_sha256": sha, "checkpoints": checkpoints, "hidden_resets": 0})


def evaluate(path, branch_id, checkpoint, output):
    from experiments.rewiring.run import RewireTrainer
    plan, item, sha = read_plan(path, branch_id)
    out, output = Path(item["output"]), Path(output)
    if Path(checkpoint).resolve() != (out / "checkpoint.pt").resolve() or output.exists():
        raise ValueError("wrong ready path or existing evaluation")
    marker = completed(path, plan, item, sha)
    records, states = [], {}
    with _preserve_process_state():
        for phase, tick in (("B", plan["phase_ticks"]), ("A2", plan["target_ticks"])):
            cp = out / f"checkpoint_{phase}.pt"
            trainer = RewireTrainer.from_checkpoint(cp, device=plan["evaluation_device"])
            validate(trainer, plan, item, tick)
            before = state_digest(trainer.semantic_state())
            normalized = normalized_hash(trainer, plan["training_device"])
            if normalized != marker["checkpoints"][phase]["semantic_hash"]:
                raise ValueError("checkpoint semantic hash differs")
            states[phase] = {"checkpoint": str(cp), "checkpoint_sha256": file_sha256(cp),
                "state_hash": before, "training_device_normalized_state_hash": normalized}
            for condition in plan["conditions"]:
                row = diagnose_fork(trainer.model, trainer.world, trainer.h, condition=condition,
                    steps=plan["evaluation_steps"], dt=plan["dt"], device=plan["evaluation_device"])
                row["learning_phase"] = phase
                records.append(row)
            if state_digest(trainer.semantic_state()) != before:
                raise AssertionError("evaluation changed training state")
    verify_end(path, branch_id, sha)
    completed(path, plan, item, sha)
    atomic_json(output, {"format": "rewiring-evaluation-v1", "branch": item, "protocol_sha256": sha,
        "source_unchanged": True, "states": states, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    for field in ("protocol", "output", "raw", "checkpoint"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--branch-id")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare": prepare(args.output, args.raw)
    elif args.mode == "train": train(args.protocol, args.branch_id)
    else: evaluate(args.protocol, args.branch_id, args.checkpoint, args.output)
