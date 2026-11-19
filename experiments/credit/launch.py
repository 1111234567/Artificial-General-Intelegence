"""Start or resume one manifest-owned branch from a complete atomic checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from experiments.credit.run import CreditTrainer, CreditConfig, file_sha256, run_to_target
from experiments.continuity.run import atomic_json


def launch(source, output, mode, ticks=65536, device="cuda", checkpoint_every=256):
    source, output = Path(source).resolve(), Path(output).resolve()
    checkpoint = output/"checkpoint.pt"
    config = CreditConfig(mode=mode)
    resumed = checkpoint.exists()
    if resumed:
        trainer = CreditTrainer.from_checkpoint(checkpoint, device=device)
        if (trainer.origin["checkpoint_sha256"] != file_sha256(source)
            or trainer.credit_config != config or trainer.stage_ticks > ticks):
            raise ValueError("existing branch does not match manifest origin/config/target")
        if str(trainer.origin["checkpoint"]) != str(source):
            raise ValueError("existing branch origin path differs")
    else:
        if output.exists() and any(p.name != "checkpoint.pt.tmp" for p in output.iterdir()):
            raise ValueError("nonempty branch has no complete checkpoint; preserve and inspect it")
        trainer = CreditTrainer.from_base_checkpoint(source, config, device=device)
        # Even an interruption before the first regular save has a full origin.
        trainer.save_checkpoint(checkpoint)
    history = output/"launches.jsonl"
    if resumed and (output/"training.jsonl").exists():
        # Keep the abandoned log suffix, but never count it as resumed training.
        index = 1
        while (output/f"training_attempt_{index:03d}.jsonl").exists():
            index += 1
        (output/"training.jsonl").rename(output/f"training_attempt_{index:03d}.jsonl")
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), resumed=resumed,
            checkpoint_sha256=file_sha256(checkpoint), stage_ticks=trainer.stage_ticks,
            target=ticks, device=device)) + "\n")
    run_to_target(trainer, ticks, output, checkpoint_every=checkpoint_every)
    atomic_json(output/"launch-complete.json", dict(target=ticks, stage_ticks=trainer.stage_ticks,
        checkpoint_sha256=file_sha256(checkpoint), origin_sha256=file_sha256(source),
        launcher_sha256=file_sha256(__file__)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--ticks", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    launch(args.from_checkpoint,args.output,args.mode,args.ticks,args.device,args.checkpoint_every)
