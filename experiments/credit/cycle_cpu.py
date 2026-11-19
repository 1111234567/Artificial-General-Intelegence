"""CPU measurement adapter around the frozen CUDA revisitation runner.

Only the frozen cycle module's evaluation callback is temporarily substituted.
Training and its full CUDA continuation remain on the original trainer. A v1
prefix is read after its workers have stopped; checkpoints remain immutable
references and every retained milestone is measured anew on CPU.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from experiments.continuity.run import atomic_json, state_digest
from experiments.credit import cycle as frozen_cycle
from experiments.credit import evaluate as frozen_evaluate
from experiments.credit.run import CreditTrainer, file_sha256


FORMAT = "credit-cycle-cpu-measurement-v2"


class CPUFacade:
    """Delegate original state and model; change only the observer's device."""
    device = torch.device("cpu")

    def __init__(self, trainer):
        self._trainer = trainer

    def __getattr__(self, name):
        return getattr(self._trainer, name)


def measurement_metadata(revision_path):
    revision_path = Path(revision_path).resolve()
    return {"measurement_backend": "cpu", "measurement_revision": str(revision_path),
            "measurement_revision_sha256": file_sha256(revision_path),
            "measurement_adapter_sha256": file_sha256(__file__)}


def cpu_evaluate(trainer, revision_path, **kwargs):
    before = state_digest(trainer.semantic_state())
    sources = {str(Path(module.__file__).resolve()): file_sha256(module.__file__)
               for module in (frozen_cycle, frozen_evaluate)}
    result = frozen_evaluate.evaluate_trainer(CPUFacade(trainer), **kwargs)
    after = state_digest(trainer.semantic_state())
    if before != after or any(file_sha256(path) != expected for path, expected in sources.items()):
        raise AssertionError("CPU measurement changed CUDA continuation or frozen sources")
    result.update(**measurement_metadata(revision_path), training_device=str(trainer.device),
                  training_state_hash_before=before, training_state_hash_after=after)
    return result


def _validate_revision(protocol_path, revision_path, device, reuse_root):
    revision = json.loads(Path(revision_path).read_text(encoding="utf-8"))
    cycles = json.loads(Path(protocol_path).read_text(encoding="utf-8"))["cycles"]
    expected = {"format": "credit-cycle-measurement-revision-v2", "evaluation_device": "cpu",
                "base_protocol_sha256": file_sha256(protocol_path), "training_device": str(device),
                "eval_conditions": cycles["eval_conditions"], "eval_steps": cycles["eval_steps"], "dt": .25,
                "adapter_source_sha256": file_sha256(__file__),
                "frozen_evaluator_sha256": file_sha256(frozen_evaluate.__file__),
                "frozen_cycle_sha256": file_sha256(frozen_cycle.__file__)}
    for key, value in expected.items():
        if revision.get(key) != value:
            raise ValueError(f"measurement revision mismatch: {key}")
    if Path(revision["reuse_root"]).resolve() != Path(reuse_root).resolve():
        raise ValueError("revision reuse root differs")
    return cycles


def _bootstrap(checkpoint, folder, protocol_path, revision_path, reuse_root, device):
    trainer = CreditTrainer.from_checkpoint(checkpoint, device=device)
    cycles = json.loads(protocol_path.read_text(encoding="utf-8"))["cycles"]
    if trainer.stage_ticks != cycles["start_stage_ticks"]:
        raise ValueError("origin is not the complete predeclared main endpoint")
    if trainer.config.seed not in cycles["seeds"] or trainer.credit_config.mode not in cycles["modes"]:
        raise ValueError("origin not in the preselected revisitation cohort")
    old_branch = reuse_root/f"seed_{trainer.config.seed:03d}"/trainer.credit_config.mode
    old_progress = old_branch/"progress.json"
    old = json.loads(old_progress.read_text(encoding="utf-8")) if old_progress.exists() else None
    original_hash, protocol_hash = file_sha256(checkpoint), file_sha256(protocol_path)
    if old is not None and (old["origin_sha256"] != original_hash or old["protocol_sha256"] != protocol_hash
                            or Path(old["origin"]).resolve() != checkpoint):
        raise ValueError("v1 prefix origin/protocol differs")
    source_progress_hash = file_sha256(old_progress) if old is not None else None
    rows = [] if old is None else old["completed"]
    expected = [(segment, name, offset, cycles["start_stage_ticks"]+segment*cycles["segment_ticks"]+offset)
                for segment, name in enumerate(cycles["sequence"]) for offset in cycles["milestones"]]
    if len(rows) > len(expected):
        raise ValueError("v1 prefix exceeds the predeclared schedule")
    progress_path = folder/"progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
    completed = [] if progress is None else progress["completed"]
    if progress is not None and (progress["origin_sha256"] != original_hash or progress["protocol_sha256"] != protocol_hash):
        raise ValueError("partial v2 bootstrap provenance differs")
    if len(completed) > len(rows):
        raise ValueError("partial bootstrap has more milestones than its immutable v1 prefix")
    references = []
    for index, old_row in enumerate(rows):
        if tuple(old_row[key] for key in ("segment", "regime", "offset", "stage_ticks")) != expected[index]:
            raise ValueError("v1 completed rows are not a contiguous protocol prefix")
        source = Path(old_row["checkpoint"]).resolve()
        if not source.is_relative_to(old_branch.resolve()) or file_sha256(source) != old_row["checkpoint_sha256"]:
            raise ValueError("v1 checkpoint path or bytes differ")
        references.append({"checkpoint": str(source), "checkpoint_sha256": old_row["checkpoint_sha256"],
                           "report": str(source.parent/"report.json")})
        if index < len(completed):
            retained = completed[index]
            if any(retained[key] != old_row[key] for key in ("checkpoint", "checkpoint_sha256", "state_hash", "stage_ticks")):
                raise ValueError("partial v2 prefix changed its immutable checkpoint")
            evaluation = json.loads(Path(retained["evaluation"]).read_text(encoding="utf-8"))
            if any(evaluation.get(key) != value for key, value in measurement_metadata(revision_path).items()):
                raise ValueError("partial v2 measurement revision differs")
            continue
        point = CreditTrainer.from_checkpoint(source, device=device)
        semantic_hash = state_digest(point.semantic_state())
        if semantic_hash != old_row["state_hash"] or point.stage_ticks != old_row["stage_ticks"]:
            raise ValueError("v1 checkpoint semantic state or tick differs")
        destination = folder/f"segment_{old_row['segment']:02d}_{old_row['regime']}"/f"tick_{old_row['offset']:04d}"
        row = deepcopy(old_row)
        row.update(evaluation=str((destination/"evaluation.json").resolve()),
                   source_evaluation=old_row["evaluation"], checkpoint_source="immutable_v1_prefix")
        evaluation = cpu_evaluate(point, revision_path, conditions=cycles["eval_conditions"], steps=cycles["eval_steps"])
        evaluation.update(cycle=row, origin=str(checkpoint), origin_sha256=original_hash, protocol_sha256=protocol_hash)
        atomic_json(destination/"evaluation.json", evaluation)
        if file_sha256(source) != old_row["checkpoint_sha256"]:
            raise AssertionError("CPU prefix measurement changed old checkpoint bytes")
        completed.append(row)
        atomic_json(progress_path, dict(origin=str(checkpoint), origin_sha256=original_hash,
            protocol_sha256=protocol_hash, completed=completed, complete=False))
    if old is not None and file_sha256(old_progress) != source_progress_hash:
        raise AssertionError("v1 progress changed; its workers must be stopped before reuse")
    result = dict(origin=str(checkpoint), origin_sha256=original_hash, protocol_sha256=protocol_hash,
                  source_progress=str(old_progress), source_progress_sha256=source_progress_hash,
                  retained_milestones=len(rows), immutable_checkpoint_references=references,
                  **measurement_metadata(revision_path))
    atomic_json(folder/"bootstrap-complete.json", result)
    return result


def run_cpu_cycles(checkpoint, folder, protocol_path, revision_path, reuse_root, device="cuda"):
    checkpoint, folder, protocol_path, revision_path, reuse_root = [Path(value).resolve()
        for value in (checkpoint, folder, protocol_path, revision_path, reuse_root)]
    if folder.is_relative_to(reuse_root) or reuse_root.is_relative_to(folder):
        raise ValueError("v2 output must be separate from immutable v1 data")
    cycles = _validate_revision(protocol_path, revision_path, device, reuse_root)
    metadata = dict(**measurement_metadata(revision_path), origin=str(checkpoint),
                    origin_sha256=file_sha256(checkpoint), protocol_sha256=file_sha256(protocol_path), training_device=str(device))
    binding = folder/"measurement-binding.json"
    if binding.exists():
        if json.loads(binding.read_text(encoding="utf-8")) != metadata:
            raise ValueError("v2 measurement provenance changed")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("nonempty v2 folder has no measurement provenance")
        atomic_json(binding, metadata)
    bootstrap_path = folder/"bootstrap-complete.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8")) if bootstrap_path.exists() else _bootstrap(
        checkpoint, folder, protocol_path, revision_path, reuse_root, device)
    callback = frozen_cycle.evaluate_trainer
    try:
        frozen_cycle.evaluate_trainer = lambda trainer, **kwargs: cpu_evaluate(trainer, revision_path, **kwargs)
        frozen_cycle.run_cycles(checkpoint, folder, protocol_path, device)
    finally:
        frozen_cycle.evaluate_trainer = callback
    complete_path = folder/"complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected_milestones = len(cycles["sequence"])*len(cycles["milestones"])
    if len(complete["completed"]) != expected_milestones:
        raise AssertionError("v2 cycle did not complete the predeclared schedule")
    for row in complete["completed"]:
        evaluation = json.loads(Path(row["evaluation"]).read_text(encoding="utf-8"))
        if (evaluation.get("measurement_backend") != "cpu"
            or evaluation.get("measurement_revision_sha256") != metadata["measurement_revision_sha256"]
            or evaluation.get("checkpoint_state_hash") != row["state_hash"]):
            raise AssertionError("milestone is not a CPU measurement of its original CUDA semantic state")
    if (file_sha256(checkpoint) != metadata["origin_sha256"] or file_sha256(protocol_path) != metadata["protocol_sha256"]
        or file_sha256(revision_path) != metadata["measurement_revision_sha256"]):
        raise AssertionError("origin/protocol/revision changed")
    final = complete["completed"][-1]
    if (bootstrap["source_progress_sha256"] is not None
        and file_sha256(bootstrap["source_progress"]) != bootstrap["source_progress_sha256"]):
        raise AssertionError("immutable v1 progress changed after bootstrap")
    if any(file_sha256(row["checkpoint"]) != row["checkpoint_sha256"]
           for row in bootstrap["immutable_checkpoint_references"]):
        raise AssertionError("immutable v1 checkpoint changed after bootstrap")
    result = {"format": FORMAT, **metadata, "source_complete_sha256": file_sha256(complete_path),
              "final_training_state_hash": final["state_hash"], "final_checkpoint": final["checkpoint"],
              "final_checkpoint_sha256": final["checkpoint_sha256"], "final_stage_ticks": complete["final_stage_ticks"],
              "completed_milestones": expected_milestones, "measurement_forks": expected_milestones*len(cycles["eval_conditions"]),
              "retained_v1_milestones": bootstrap["retained_milestones"], "bootstrap": str(bootstrap_path), "complete": True}
    atomic_json(folder/"measurement-complete.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "output", "protocol", "revision", "reuse-root"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run_cpu_cycles(args.checkpoint, args.output, args.protocol, args.revision, args.reuse_root, args.device)
