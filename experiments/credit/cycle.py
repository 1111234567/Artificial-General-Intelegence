"""Continue fixed A/B revisitation schedules without resetting any live state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.continuity.run import atomic_json, state_digest
from experiments.credit.run import CreditTrainer, file_sha256, run_to_target
from experiments.credit.evaluate import evaluate_trainer


def run_cycles(checkpoint, folder, protocol_path, device="cuda"):
    checkpoint, folder, protocol_path = map(Path, (checkpoint, folder, protocol_path))
    protocol_hash = file_sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))["cycles"]
    progress_path = folder / "progress.json"
    completed = []
    original_hash = file_sha256(checkpoint)
    current = checkpoint
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["protocol_sha256"] != protocol_hash or progress["origin_sha256"] != original_hash:
            raise ValueError("cycle provenance changed")
        completed = progress["completed"]
        if completed:
            current = Path(completed[-1]["checkpoint"])
            if file_sha256(current) != completed[-1]["checkpoint_sha256"]:
                raise ValueError("cycle resume checkpoint changed")
    trainer = CreditTrainer.from_checkpoint(current, device=device)
    chunk = trainer.credit_config.chunk_ticks
    milestones = protocol["milestones"]
    if (not protocol["sequence"] or not milestones or milestones != sorted(set(milestones))
        or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 or x % chunk for x in milestones)
        or milestones[-1] != protocol["segment_ticks"]
        or protocol["start_stage_ticks"] % chunk
        or any(name not in protocol["regimes"] for name in protocol["sequence"])):
        raise ValueError("cycle requires known regimes and positive aligned milestones ending each segment")
    if trainer.credit_config.mode not in protocol["modes"] or trainer.config.seed not in protocol["seeds"]:
        raise ValueError("checkpoint not in the preselected cycle cohort")
    if not completed and trainer.stage_ticks != protocol["start_stage_ticks"]:
        raise ValueError("cycle requires complete initial B training")
    for segment, name in enumerate(protocol["sequence"]):
        start = protocol["start_stage_ticks"] + segment * protocol["segment_ticks"]
        if trainer.stage_ticks >= start + protocol["segment_ticks"]:
            continue
        if trainer.stage_ticks == start:
            duration, gap = protocol["regimes"][name]
            trainer.change_regime(duration, gap, f"revisit/{segment:02d}/{name}")
        for offset in protocol["milestones"]:
            target = start + offset
            if trainer.stage_ticks >= target:
                continue
            destination = folder / f"segment_{segment:02d}_{name}" / f"tick_{offset:04d}"
            run_to_target(trainer, target, destination, checkpoint_every=0)
            state_hash = state_digest(trainer.semantic_state())
            evaluation = evaluate_trainer(trainer, conditions=protocol["eval_conditions"],
                                          steps=protocol["eval_steps"])
            if state_hash != state_digest(trainer.semantic_state()):
                raise AssertionError("cycle milestone evaluation mutated training")
            saved = destination / "checkpoint.pt"
            record = dict(segment=segment, regime=name, offset=offset, stage_ticks=target,
                          checkpoint=str(saved.resolve()), checkpoint_sha256=file_sha256(saved),
                          state_hash=state_hash, evaluation=str((destination/"evaluation.json").resolve()))
            evaluation.update(cycle=record, origin=str(checkpoint.resolve()), origin_sha256=original_hash,
                              protocol_sha256=protocol_hash)
            atomic_json(destination/"evaluation.json", evaluation)
            completed.append(record)
            atomic_json(progress_path, dict(origin=str(checkpoint.resolve()), origin_sha256=original_hash,
                protocol_sha256=protocol_hash, completed=completed, complete=False))
    if file_sha256(checkpoint) != original_hash:
        raise AssertionError("cycle modified its original checkpoint")
    atomic_json(folder/"complete.json", dict(origin=str(checkpoint.resolve()), origin_sha256=original_hash,
        protocol_sha256=protocol_hash, completed=completed, complete=True,
        final_stage_ticks=trainer.stage_ticks, report=trainer.report(),
        source_sha256=file_sha256(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run_cycles(args.checkpoint, args.output, args.protocol, args.device)
