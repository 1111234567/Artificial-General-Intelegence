"""Train on continuing streams; detach graphs, never reset memory per update."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.streaming.cache import CacheAgent, State
from experiments.streaming.stream import cycle
from experiments.persistent.run import atomic_json, environment, seed_all


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).with_name(n) for n in ("cache.py", "stream.py", "run_stream.py"))}


def rollout(model, state, events, policy, *, generator, cost=.02, reset_memory=False,
            force_reads=None, halt_loss="expected"):
    losses, metrics = [], defaultdict(list)
    for event in events:
        if event.kind == "fact":
            state, skipped = model.write(state, event, policy, generator)
            metrics["skip"].append(skipped)
        else:
            state = model.observe(state, event)
            if event.kind != "query":
                continue
            read_state = state
            if reset_memory:
                read_state = State(torch.zeros_like(state.memory), state.hidden, state.clock, state.recent, state.demand_time)
            out = model.read(read_state, event, force_reads)
            if model.training:
                candidate_p = out["candidates"].gather(2, event.target[:, None, None].expand(-1, out["candidates"].shape[1], 1)).squeeze(-1)
                if halt_loss == "mixture":
                    loss = -(out["masses"] * candidate_p).sum(-1).clamp_min(1e-12).log()
                else:
                    loss = (out["masses"] * -candidate_p.clamp_min(1e-12).log()).sum(-1)
                loss = loss + cost * out["expected_reads"]
            else:
                answer_probability = out["hard_prob"].gather(1, event.target[:, None]).squeeze(1)
                loss = -answer_probability.clamp_min(1e-12).log() + cost * out["reads"]
            losses.append(loss.mean())
            correct = (out["answer"] == event.target).float()
            metrics[event.phase + "/accuracy"].append(correct)
            metrics[event.phase + "/reads"].append(out["reads"].float())
            metrics[event.phase + "/depth"].append(event.depth.float())
            metrics[event.phase + "/stop_exact"].append((out["reads"] == event.depth).float())
            metrics[event.phase + "/utility"].append(correct - cost * out["reads"])
            metrics["accuracy"].append(correct)
            metrics["reads"].append(out["reads"].float())
    return state, torch.stack(losses).mean(), metrics


def reduce_metrics(metrics):
    return {k: torch.cat(v).float().mean().item() for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, args, policy, seed, *, cycles=4, noise=8, slots=None,
             return_to="random", reset_memory=False, force_reads=None):
    model.eval()
    old_slots = model.slots
    if slots is not None:
        model.slots = slots
    generator = torch.Generator(device=args.device).manual_seed(seed)
    policy_rng = torch.Generator(device=args.device).manual_seed(seed + 4000000)
    state = model.initial(args.eval_batch, args.device)
    records = []
    for index in range(cycles):
        events = cycle(args.eval_batch, generator, contexts=args.contexts, vocab=args.vocab,
                       nodes=args.nodes, noise=noise, queries=args.queries,
                       return_to=return_to, device=args.device)
        state, loss, metrics = rollout(model, state, events, policy, generator=policy_rng,
                                      cost=args.cost, reset_memory=reset_memory,
                                      force_reads=force_reads)
        records.append(dict(cycle=index, loss=loss.item(), **reduce_metrics(metrics)))
    model.slots = old_slots
    keys = records[0].keys() - {"cycle"}
    return dict(cycles=cycles, batch=args.eval_batch, queries_per_phase=args.queries,
                metrics={k: sum(r[k] for r in records) / cycles for k in keys},
                per_cycle=records)


def train(args, policy, seed, output):
    source = hashes()
    seed_all(seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = CacheAgent(contexts=args.contexts, vocab=args.vocab, slots=args.slots,
                       hidden=args.hidden, max_reads=args.max_reads,
                       width=args.width, read_dim=args.read_dim).to(args.device)
    initial_hash = None
    if args.initial_dir:
        initial = ROOT / args.initial_dir / f"mru-{seed}.pt"
        checkpoint = torch.load(initial, map_location=args.device, weights_only=False)
        required = ("contexts", "vocab", "nodes", "slots", "hidden", "width", "read_dim", "max_reads")
        if checkpoint["policy"] != "mru" or checkpoint["seed"] != seed:
            raise ValueError("initial checkpoint policy/seed mismatch")
        for key in required:
            if checkpoint["config"].get(key) != getattr(args, key):
                raise ValueError(f"initial configuration mismatch: {key}")
        for name in ("cache.py", "stream.py"):
            if checkpoint["source"].get(name) != source[name]:
                raise ValueError(f"initial model/task source mismatch: {name}")
        model.load_state_dict(checkpoint["model"])
        initial_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
    if args.writer_only:
        if policy != "learned":
            raise ValueError("writer-only requires a learned writer")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("retain.", "event_state.")))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = torch.Generator(device=args.device).manual_seed(100000 + seed)
    policy_rng = torch.Generator(device=args.device).manual_seed(4100000 + seed)
    state = model.initial(args.batch, args.device)
    records = []
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step in {0, args.steps} or step % args.log_every == 0:
            dev = evaluate(model, args, policy, 700000 + seed, cycles=2)
            records.append(dict(step=step, validation=dev))
            print(json.dumps(dict(policy=policy, seed=seed, step=step,
                                  **{k: round(v, 4) for k, v in dev["metrics"].items()
                                     if k in {"accuracy", "reads", "return/accuracy", "revision/accuracy"}})), flush=True)
        if step == args.steps:
            break
        model.train()
        # Variable interference length, no phase-dependent state reset.
        noise = int(torch.randint(3, 11, (), generator=rng, device=args.device).item())
        events = cycle(args.batch, rng, contexts=args.contexts, vocab=args.vocab,
                       nodes=args.nodes, noise=noise, queries=args.queries, device=args.device)
        state, loss, metrics = rollout(model, state, events, policy, generator=policy_rng,
                                      cost=args.cost, halt_loss=args.halt_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        state = state.detach()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checkpoint = output / f"{policy}-{seed}.pt"
    if hashes() != source:
        raise RuntimeError("source changed during training; refusing mismatched checkpoint")
    torch.save(dict(model=model.state_dict(), config=vars(args), seed=seed,
                    policy=policy, source=source), checkpoint)
    tests = {}
    if not args.pilot:
        for name, kwargs in {
            "stream": {}, "noise32": dict(noise=32),
            "return_a": dict(return_to="a"), "stay_b": dict(return_to="b"),
            "half_capacity": dict(slots=max(1, args.slots // 2)), "erase_memory": dict(reset_memory=True),
            "force1": dict(force_reads=1), "force_max": dict(force_reads=args.max_reads),
        }.items():
            tests[name] = evaluate(model, args, policy, 1700000 + seed,
                                   cycles=args.eval_cycles, **kwargs)
        if policy == "learned":
            # Same reader and halt head: isolate the learned writer at inference.
            for alternate in ("mru", "fifo", "random"):
                tests["writer_" + alternate] = evaluate(model, args, alternate, 1700000 + seed,
                                                        cycles=args.eval_cycles)
    result = dict(policy=policy, seed=seed, config=vars(args), source=hashes(),
                  initial_checkpoint_sha256=initial_hash,
                  peak_cuda_allocated=(torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None),
                  peak_cuda_reserved=(torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None),
                  seconds=elapsed, parameters=sum(p.numel() for p in model.parameters()),
                  validation=records, heldout=tests,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    atomic_json(output / f"{policy}-{seed}.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--policies", default="learned,mru,fifo")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--eval-cycles", type=int, default=8)
    p.add_argument("--queries", type=int, default=2)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--contexts", type=int, default=8)
    p.add_argument("--vocab", type=int, default=16)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--read-dim", type=int, default=32)
    p.add_argument("--max-reads", type=int, default=6)
    p.add_argument("--initial-dir")
    p.add_argument("--writer-only", action="store_true")
    p.add_argument("--halt-loss", choices=("mixture", "expected"), default="mixture")
    p.add_argument("--cost", type=float, default=.02)
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    output = ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", dict(config=vars(args), source=hashes(), environment=environment(args.device)))
    for policy in args.policies.split(","):
        for seed in map(int, args.seeds.split(",")):
            train(args, policy, seed, output)


if __name__ == "__main__":
    main()
