"""Frozen, own-action evaluation of new credit checkpoints using the old observer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, state_digest
from experiments.credit.run import load_checkpoint, file_sha256
from experiments.integration.diagnose import diagnose_fork, CONDITIONS, INTERVENTIONS, COHORTS
from experiments.integration.inspect_state import activity_statistics, matrix_statistics

PRIMARY_CONDITIONS = ("iid", "long_wait", "longer", "sparse", "long_sparse")

def evaluate_trainer(trainer, *, conditions=PRIMARY_CONDITIONS, steps=2048, interventions=()):
    conditions, interventions = tuple(conditions), tuple(interventions)
    if not conditions or len(set(conditions)) != len(conditions) or any(x not in CONDITIONS for x in conditions):
        raise ValueError("distinct known conditions required")
    if len(set(interventions)) != len(interventions) or any(x not in INTERVENTIONS or x == "none" for x in interventions):
        raise ValueError("distinct non-baseline interventions required")
    before = state_digest(trainer.semantic_state())
    records = []
    with _preserve_process_state():
        for condition in conditions:
            baseline = diagnose_fork(trainer.model, trainer.world, trainer.h,
                condition=condition, steps=steps, dt=.25, device=trainer.device)
            records.append(baseline)
            if condition in ("longer", "long_sparse"):
                for intervention in interventions:
                    record = diagnose_fork(trainer.model, trainer.world, trainer.h,
                        condition=condition, steps=steps, dt=.25, device=trainer.device,
                        intervention=intervention)
                    for key in ("start_world_state_hash", "start_hidden_hash", "end_event_hash",
                                "start_scheduler_hash", "end_scheduler_hash"):
                        if record[key] != baseline[key]:
                            raise AssertionError(f"intervention changed external history: {key}")
                    for cohort in COHORTS[:3]:
                        if record[cohort]["request_fingerprint"] != baseline[cohort]["request_fingerprint"]:
                            raise AssertionError(f"intervention changed request cohort: {cohort}")
                    record["paired_baseline_action_hash"] = baseline["action_hash"]
                    records.append(record)
    if before != state_digest(trainer.semantic_state()):
        raise AssertionError("frozen evaluation changed complete continuation")
    return {"format": "credit-evaluation-v1", "seed": trainer.config.seed,
        "mode": trainer.adaptation_mode, "stage_ticks": trainer.stage_ticks,
        "training_ticks": trainer.ticks, "checkpoint_state_hash": before,
        "model": trainer.model.parameter_report(), "records": records,
        "credit_config": asdict(trainer.credit_config),
        "state_snapshot": {"activity": activity_statistics(trainer.h),
                           "recurrent": matrix_statistics(trainer.model.recurrent)},
        "source_sha256": file_sha256(__file__),
        "observer_sha256": file_sha256(Path(__file__).parents[1]/"integration/diagnose.py")}


def evaluate_checkpoint(checkpoint, **kwargs):
    checkpoint = Path(checkpoint).resolve()
    before = file_sha256(checkpoint)
    with _preserve_process_state():
        trainer = load_checkpoint(checkpoint, device=kwargs.pop("device", "cpu"))
        result = evaluate_trainer(trainer, **kwargs)
    if before != file_sha256(checkpoint):
        raise AssertionError("checkpoint bytes changed")
    result.update(checkpoint=str(checkpoint), checkpoint_sha256=before)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(PRIMARY_CONDITIONS))
    parser.add_argument("--interventions", nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_json(args.output, evaluate_checkpoint(args.checkpoint, conditions=args.conditions,
        interventions=args.interventions, steps=args.steps, device=args.device))
