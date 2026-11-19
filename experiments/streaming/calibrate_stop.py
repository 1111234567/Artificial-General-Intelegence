"""Align soft-trained halting with hard deployment using final answers only.

Only the three halting-head parameters change. Neither correct depth nor path
labels are read by the training objective. This is a separately recorded stage,
not a threshold search or a change to the reader/memory algorithm.
"""
import argparse
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.streaming.cache import CacheAgent
from experiments.streaming.stream import cycle
from experiments.streaming.run_stream import rollout, evaluate, hashes
from experiments.persistent.run import atomic_json, environment


def weights_hash(model, exclude_halt=False):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if exclude_halt and name.startswith("halt."):
            continue
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_from_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source = hashes()
    for name in ("cache.py", "stream.py"):
        if checkpoint["source"][name] != source[name]:
            raise ValueError(f"checkpoint source mismatch: {name}")
    c = checkpoint["config"]
    model = CacheAgent(**{k: c[k] for k in ("contexts", "vocab", "slots", "hidden", "max_reads", "width", "read_dim")}).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=.02)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    torch.set_num_threads(1)
    out = ROOT / a.output
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty output")
    out.mkdir(parents=True, exist_ok=True)
    source = {**hashes(), "calibrate_stop.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(out / "manifest.json", dict(config=vars(a), source=source, environment=environment(a.device)))
    for seed in map(int, a.seeds.split(",")):
        initial = ROOT / a.input / f"mru-{seed}.pt"
        model, checkpoint = model_from_checkpoint(initial, a.device)
        if checkpoint["seed"] != seed or checkpoint["policy"] != "mru":
            raise ValueError("checkpoint identity mismatch")
        frozen = copy.deepcopy(model)
        before = weights_hash(model, True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("halt."))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        c = dict(checkpoint["config"], device=a.device)
        config = SimpleNamespace(**c)
        rng = torch.Generator(device=a.device).manual_seed(2300000 + seed)
        state = model.initial(config.batch, a.device)
        curve = []
        start = time.perf_counter()
        for step in range(a.steps + 1):
            if step % 50 == 0 or step == a.steps:
                val = evaluate(model, config, "mru", 2700000 + seed, cycles=2)
                curve.append(dict(step=step, validation=val))
                print(dict(seed=seed, step=step, accuracy=val["metrics"]["accuracy"], reads=val["metrics"]["reads"]), flush=True)
            if step == a.steps:
                break
            model.train()
            events = cycle(config.batch, rng, contexts=config.contexts, vocab=config.vocab,
                           nodes=config.nodes, noise=8, queries=config.queries, device=a.device)
            state, loss, _ = rollout(model, state, events, "mru", generator=rng,
                                     cost=config.cost, halt_loss="expected")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            state = state.detach()
        torch.cuda.synchronize() if a.device.startswith("cuda") else None
        elapsed = time.perf_counter() - start
        assert weights_hash(model, True) == before, "non-halting parameters changed"
        tests = {}
        for name, kwargs in {"stream": {}, "noise32": dict(noise=32), "return_a": dict(return_to="a"),
                             "stay_b": dict(return_to="b"), "half_capacity": dict(slots=config.slots//2),
                             "erase_memory": dict(reset_memory=True), "force1": dict(force_reads=1),
                             "force_max": dict(force_reads=config.max_reads)}.items():
            tests[name] = evaluate(model, config, "mru", 3700000 + seed, cycles=8, **kwargs)
        tests["before_calibration"] = evaluate(frozen, config, "mru", 3700000 + seed, cycles=8)
        for alternate in ("fifo", "random"):
            tests["writer_" + alternate] = evaluate(model, config, alternate, 3700000 + seed, cycles=8)
        saved = out / f"mru-{seed}.pt"
        torch.save(dict(model=model.state_dict(), config=dict(c, halt_loss="expected"), seed=seed,
                        policy="mru", source=hashes(), calibration=dict(steps=a.steps, lr=a.lr)), saved)
        result = dict(seed=seed, source=source, base_config=c, calibration=dict(steps=a.steps, lr=a.lr),
                      parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=3,
                      initial_checkpoint_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
                      checkpoint_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
                      non_halt_hash_before=before, non_halt_hash_after=weights_hash(model, True),
                      seconds=elapsed, validation=curve, heldout=tests)
        atomic_json(out / f"mru-{seed}.json", result)


if __name__ == "__main__":
    main()
