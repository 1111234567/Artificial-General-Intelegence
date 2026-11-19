"""Independent long-stream holdout with fixed weights and continuing state."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.streaming.calibrate_stop import model_from_checkpoint, weights_hash
from experiments.streaming.run_stream import evaluate, hashes
from experiments.persistent.run import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--policy", choices=("mru", "learned"), required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--cycles", type=int, default=64)
    p.add_argument("--noise", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(1)
    out = ROOT / args.output
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty output")
    out.mkdir(parents=True, exist_ok=True)
    source = {**hashes(), "evaluate_long.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(out / "manifest.json", dict(config=vars(args), source=source))
    for seed in map(int, args.seeds.split(",")):
        path = ROOT / args.input / f"{args.policy}-{seed}.pt"
        model, checkpoint = model_from_checkpoint(path, args.device)
        if checkpoint["policy"] != args.policy or checkpoint["seed"] != seed:
            raise ValueError("checkpoint identity mismatch")
        config = SimpleNamespace(**dict(checkpoint["config"], device=args.device, eval_batch=args.batch))
        before = weights_hash(model)
        result = evaluate(model, config, args.policy, 5700000 + seed,
                          cycles=args.cycles, noise=args.noise)
        assert weights_hash(model) == before
        atomic_json(out / f"{args.policy}-{seed}.json", dict(seed=seed, policy=args.policy,
                    config=vars(args), source=source, base_config=checkpoint["config"],
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    parameter_hash=before, heldout=result))
        print(dict(seed=seed, **{k: result["metrics"][k] for k in
                               ("accuracy", "reads", "return/accuracy", "revision/accuracy")}), flush=True)


if __name__ == "__main__":
    main()
