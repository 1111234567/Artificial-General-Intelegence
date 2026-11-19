"""Evaluate saved last checkpoints on new, disjoint synthetic worlds.

Development evaluations in training logs are not reused as final holdout data.
No optimization, best-checkpoint selection, or parameter search occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.persistent.run import (aggregate, atomic_json, build_model,
                                        environment, evaluate, source_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=["core", "retention"], default="core")
    parser.add_argument("--seed-offset", type=int, default=2900000)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opts = parser.parse_args()
    source = ROOT / opts.input
    output = ROOT / opts.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Choose a new, empty evaluation output directory")
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if original["sources"] != source_hashes():
        raise ValueError("Training source differs from the current implementation")
    torch.set_num_threads(original["arguments"]["threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cases = (
        [("one_hop",16,32,1), ("two_hop",16,32,2), ("four_hop",16,32,4),
         ("capacity32",32,32,4), ("capacity64",64,32,4), ("eight_hop",16,32,8),
         ("corrupt_memory",16,32,4)]
        if opts.suite == "core" else
        [("two_facts",2,0,1), ("two_gap16",2,16,1), ("two_gap64",2,64,1),
         ("two_gap256",2,256,1), ("two_gap1024",2,1024,1), ("two_gap4096",2,4096,1),
         ("small_gap0",4,0,1), ("small_gap16",4,16,1),
         ("small_gap64",4,64,1), ("small_gap256",4,256,1),
         ("small_gap1024",4,1024,1), ("small_gap4096",4,4096,1),
         ("eight_facts",8,0,1)]
    )
    manifest = dict(training_manifest=original, evaluation=vars(opts),
                    environment=environment(opts.device), cases=cases,
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output / "manifest.json", manifest)
    results = []
    for path in sorted(source.glob("*-seed*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**saved["args"])
        args.device = opts.device
        args.eval_batches = opts.batches
        model = build_model(saved["mode"], args).to(opts.device)
        model.load_state_dict(saved["model"])
        selected_cases = cases.copy()
        if saved["mode"] == "judge" and opts.suite == "core":
            selected_cases += [("feedback_zero",16,32,4), ("feedback_shuffle",16,32,4)]
        values = evaluate(model, args, seed=opts.seed_offset + saved["seed"], cases=selected_cases)
        result = dict(mode=saved["mode"], seed=saved["seed"], evaluations=values,
                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        atomic_json(output / f'{saved["mode"]}-seed{saved["seed"]}.json', result)
        results.append(result)
        atomic_json(output / "summary.json", aggregate(results))
        print(json.dumps(dict(mode=saved["mode"], seed=saved["seed"],
                              scores={v["case"]: v["accuracy"] for v in values})), flush=True)


if __name__ == "__main__":
    main()
