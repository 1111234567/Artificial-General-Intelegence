"""Train a hard cache writer with binary-feedback REINFORCE, without ST.

The pretrained reader and halt parameters stay fixed. Only retain/event_state
learn. Rewards are actual hard-answer correctness from the environment. There
is no answer CE, path label, depth label, salience label, or learned critic.
Evaluation delegates writing to the frozen CacheAgent implementation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.streaming.cache import CacheAgent, State
from experiments.streaming.diagnose import parameter_hash, file_hash
from experiments.streaming.run_stream import evaluate
from experiments.streaming.stream import cycle, observed
from experiments.persistent.run import environment


def derived_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"rl-writer-v1/{seed}/{name}".encode()).digest()[:8],
                          "little") % (2**62)


def hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in (
        "cache.py", "stream.py", "run_stream.py", "diagnose.py", "reinforce_writer.py"
    )}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


@dataclass
class Action:
    clock: int
    log_probability: torch.Tensor
    entropy: torch.Tensor
    eligible: torch.Tensor


class ReinforceWriter(CacheAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer_actions = []
        self.write_counts = defaultdict(int)

    def write(self, state, event, policy="learned", generator=None):
        # The deployed writer is exactly the base deterministic argmin policy.
        if not self.training or policy != "learned":
            return super().write(state, event, policy, generator)
        if generator is None:
            raise ValueError("Training requires an explicit categorical RNG")
        state = self.observe(state, event)
        raw = observed(event, self.contexts, self.vocab)[:, :self.payload]
        record = torch.cat((raw, raw.new_ones((raw.shape[0], 1)),
                            raw.new_full((raw.shape[0], 1), state.clock)), -1)
        memory = state.memory
        valid = memory[:, :, -2] > .5
        identity = self.contexts + self.vocab
        matches = ((memory[:, :, :identity] - raw[:, None, :identity]).abs().sum(-1) < .1) & valid
        existing = matches.any(-1)
        empty = ~valid
        forced = empty.any(-1) & ~existing
        batch = raw.shape[0]
        candidates = torch.cat((memory, record[:, None]), 1)
        age = torch.log1p((state.clock - candidates[:, :, -1]).clamp_min(0)) / 5
        context = candidates[:, :, :self.contexts].argmax(-1)
        last_requested = state.demand_time.gather(1, context)
        seen = (last_requested >= 0).float()
        demand_age = torch.log1p((state.clock - last_requested).clamp_min(0)) / 5
        features = torch.cat((
            candidates[:, :, :self.payload], torch.stack((age, demand_age, seen), -1),
            state.hidden[:, None].expand(-1, self.slots + 1, -1),
        ), -1)
        score = self.retain(features).squeeze(-1)
        allowed = torch.cat((
            torch.where(existing[:, None], matches, valid),
            torch.ones(batch, 1, dtype=torch.bool, device=raw.device),
        ), -1)
        # Validity is guaranteed by the nonempty incoming option and seeded
        # multinomial. Generic distribution validation would synchronize the
        # GPU at every write; disabling it does not change probabilities.
        distribution = Categorical(logits=(-score).masked_fill(~allowed, -1e9),
                                   validate_args=False)
        # Equivalent categorical sampling, with a separately seeded RNG.
        sampled = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        choice = torch.where(forced, empty.long().argmax(-1), sampled)
        eligible = ~forced
        log_probability = torch.where(eligible, distribution.log_prob(sampled),
                                      torch.zeros_like(score[:, 0]))
        entropy = torch.where(eligible, distribution.entropy(), torch.zeros_like(score[:, 0]))
        self.writer_actions.append(Action(state.clock, log_probability, entropy, eligible))
        # No straight-through expression or differentiable cache payload.
        weights = F.one_hot(choice, self.slots + 1).to(raw.dtype)
        replacement = weights[:, :self.slots, None]
        new_memory = memory * (1 - replacement) + record[:, None] * replacement
        if new_memory.requires_grad:
            raise RuntimeError("Hard cache unexpectedly acquired a pathwise gradient")
        self.write_counts["incoming_records"] += batch
        # Keep counters on-device until logging; per-write int(tensor) would
        # force hundreds of host/device synchronizations per training cycle.
        self.write_counts["forced_empty_fills"] += forced.sum()
        self.write_counts["optional_actions"] += eligible.sum()
        self.write_counts["optional_slot_selections"] += (eligible & (choice < self.slots)).sum()
        self.write_counts["optional_incoming_skips"] += (eligible & (choice == self.slots)).sum()
        self.write_counts["existing_identities"] += existing.sum()
        return State(new_memory, state.hidden, state.clock, state.recent, state.demand_time), weights[:, -1]


def leave_one_out(returns, eligible):
    """Other eligible rows only; an eligible singleton gets baseline zero."""
    selected = eligible.to(returns.dtype)
    count = selected.sum(-1, keepdim=True)
    # Exclude a row before summing, avoiding cancellation through its own
    # return. This also makes finite-precision self-independence testable.
    chosen_returns = returns * selected
    zero = torch.zeros_like(returns[..., :1])
    prefix = torch.cat((zero, chosen_returns.cumsum(-1)[..., :-1]), -1)
    suffix = torch.cat((chosen_returns.flip(-1).cumsum(-1).flip(-1)[..., 1:], zero), -1)
    numerator = prefix + suffix
    baseline = numerator / (count - selected).clamp_min(1)
    return torch.where(eligible, baseline, torch.zeros_like(baseline)).detach()


def policy_objective(actions, query_rewards, entropy_coefficient):
    if not actions:
        raise ValueError("No writer actions in rollout")
    log_probability = torch.stack([action.log_probability for action in actions])
    entropies = torch.stack([action.entropy for action in actions])
    eligible = torch.stack([action.eligible for action in actions])
    action_clock = torch.tensor([action.clock for action in actions], device=log_probability.device)
    query_clock = torch.tensor([clock for clock, _ in query_rewards], device=log_probability.device)
    rewards = torch.stack([reward for _, reward in query_rewards]).detach()
    future = (query_clock[None] > action_clock[:, None]).to(log_probability.dtype)
    future_count = future.sum(-1, keepdim=True)
    returns = ((future @ rewards) / future_count.clamp_min(1)).detach()
    eligible = eligible & (future_count > 0)
    advantage = (returns - leave_one_out(returns, eligible)).detach()
    mask = eligible.to(returns.dtype)
    denominator = mask.sum().clamp_min(1)
    actor_loss = -(advantage * log_probability * mask).sum() / denominator
    entropy = (entropies * mask).sum() / denominator
    objective = actor_loss - entropy_coefficient * entropy
    return objective, {
        "actor_loss": actor_loss.detach(), "entropy": entropy.detach(),
        "credited_optional_actions": mask.sum().detach(),
        "return_mean": ((returns * mask).sum() / denominator).detach(),
        "advantage_abs_mean": ((advantage.abs() * mask).sum() / denominator).detach(),
    }


def reward_rollout(model, state, events, policy_rng):
    query_rewards = []
    metrics = defaultdict(list)
    for event in events:
        visible = replace(event, target=None, depth=None, phase="")
        if event.kind == "fact":
            state, _ = model.write(state, visible, "learned", policy_rng)
            continue
        state = model.observe(state, visible)
        if event.kind != "query":
            continue
        # Only the environment compares the frozen deployed answer with truth.
        with torch.no_grad():
            # Base read's train/eval difference is only early loop exit.
            # Computing all candidates avoids per-hop GPU->host bool sync;
            # hard answer/reads remain identical to deployed threshold halt.
            output = model.read(state, visible)
            reward = (output["answer"] == event.target).float()
            query_rewards.append((state.clock, reward))
            metrics["accuracy"].append(reward)
            metrics[event.phase + "/accuracy"].append(reward)
            metrics[event.phase + "/reads"].append(output["reads"].float())
    return state, query_rewards, {
        name: torch.cat(values).mean().detach() for name, values in metrics.items()
    }


def writer_parameter_change(model, initial):
    result = {}
    for prefix in ("retain.", "event_state."):
        total = sum((value.detach() - initial[name]).square().sum().item()
                    for name, value in model.named_parameters() if name.startswith(prefix))
        result[prefix.rstrip(".") + "_l2_change"] = total ** .5
    return result


def writer_parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("retain.", "event_state.")):
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_reader(directory, reader_seed, device):
    checkpoint_path = directory / f"mru-{reader_seed}.pt"
    checkpoint_sha256 = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint_sha256 != file_hash(checkpoint_path):
        raise RuntimeError("Reader checkpoint changed during loading")
    if checkpoint.get("policy") != "mru" or checkpoint.get("seed") != reader_seed:
        raise ValueError("Reader checkpoint payload policy/seed disagrees with requested identity")
    config = checkpoint["config"]
    current = hashes()
    for name in ("cache.py", "stream.py"):
        if checkpoint["source"].get(name) != current[name]:
            raise ValueError(f"Reader checkpoint has different {name} source")
    model = ReinforceWriter(**{name: config[name] for name in (
        "contexts", "vocab", "slots", "hidden", "max_reads", "width", "read_dim"
    ) if name in config}).to(device)
    model.load_state_dict(checkpoint["model"])
    model.initial_source_checkpoint_sha256 = checkpoint_sha256
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("retain.", "event_state.")))
    return model, config, checkpoint_path


def train(seed, args, directory):
    reader_seed = seed if args.reader_seed is None else args.reader_seed
    model, reader_config, reader_path = load_reader(Path(args.initial_dir), reader_seed, args.device)
    frozen_hash = parameter_hash(model, reader_only=True)
    reader_checkpoint_sha256 = model.initial_source_checkpoint_sha256
    initial_writer_sha256 = writer_parameter_hash(model)
    initial_writer = {name: value.detach().clone() for name, value in model.named_parameters()
                      if value.requires_grad}
    evaluation_args = SimpleNamespace(
        contexts=model.contexts, vocab=model.vocab, nodes=reader_config["nodes"],
        queries=args.queries, device=args.device, eval_batch=args.eval_batch, cost=args.cost,
    )
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    stream_rng = torch.Generator(device=args.device).manual_seed(derived_seed(seed, "train-stream"))
    policy_rng = torch.Generator(device=args.device).manual_seed(derived_seed(seed, "categorical-actions"))
    validation_seed = derived_seed(seed, "validation")
    state = model.initial(args.batch, args.device)
    records = []
    pure_training_seconds = 0.0
    training_cuda_milliseconds = 0.0
    first_validation = evaluate(model, evaluation_args, "learned", validation_seed, cycles=2)
    records.append({"step": 0, "validation": first_validation,
                    "parameter_change": writer_parameter_change(model, initial_writer)})
    print(json.dumps({"event": "writer_initial", "seed": seed, "reader_seed": reader_seed,
                      "validation": first_validation["metrics"]}), flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    segment_start = start
    cuda_start = torch.cuda.Event(enable_timing=True) if args.device.startswith("cuda") else None
    cuda_end = torch.cuda.Event(enable_timing=True) if args.device.startswith("cuda") else None
    if cuda_start is not None:
        cuda_start.record()
    for step in range(1, args.steps + 1):
        model.train()
        model.writer_actions.clear()
        noise = int(torch.randint(3, 11, (), generator=stream_rng, device=args.device))
        events = cycle(
            args.batch, stream_rng, contexts=model.contexts, vocab=model.vocab,
            nodes=reader_config["nodes"], noise=noise, queries=args.queries, device=args.device,
        )
        state, query_rewards, metrics = reward_rollout(model, state, events, policy_rng)
        loss, policy_stats = policy_objective(model.writer_actions, query_rewards, args.entropy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Nonfinite writer policy gradient")
        optimizer.step()
        state = state.detach()
        model.writer_actions.clear()
        if step % args.log_every == 0 or step == args.steps:
            if cuda_end is not None:
                cuda_end.record()
                cuda_end.synchronize()
                training_cuda_milliseconds += cuda_start.elapsed_time(cuda_end)
            pure_training_seconds += time.perf_counter() - segment_start
            validation = evaluate(model, evaluation_args, "learned", validation_seed, cycles=2)
            if parameter_hash(model, reader_only=True) != frozen_hash:
                raise RuntimeError("Frozen reader/halt values changed")
            point = {
                "step": step, "seconds": time.perf_counter() - start,
                "loss": loss.detach().item(), "gradient_norm": float(gradient_norm),
                "training_binary_rewards": {name: float(value) for name, value in metrics.items()},
                "policy": {name: float(value) for name, value in policy_stats.items()},
                "actual_write_counts": {name: int(value) for name, value in model.write_counts.items()},
                "pure_training_seconds": pure_training_seconds,
                "training_cuda_milliseconds": training_cuda_milliseconds,
                "validation": validation,
                "parameter_change": writer_parameter_change(model, initial_writer),
            }
            records.append(point)
            print(json.dumps({"event": "writer_progress", "seed": seed, "reader_seed": reader_seed,
                              "step": step, "seconds": point["seconds"],
                              "validation": {name: validation["metrics"][name] for name in (
                                  "accuracy", "a_acquire/accuracy", "b_acquire/accuracy",
                                  "return/accuracy", "revision/accuracy")},
                              **point["policy"], **point["parameter_change"],
                              "pure_training_seconds": pure_training_seconds,
                              "training_cuda_milliseconds": training_cuda_milliseconds}), flush=True)
            segment_start = time.perf_counter()
            if cuda_start is not None:
                cuda_start.record()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
    if parameter_hash(model, reader_only=True) != frozen_hash:
        raise RuntimeError("Frozen reader/halt values changed")
    if file_hash(reader_path) != reader_checkpoint_sha256:
        raise RuntimeError("Initial checkpoint was replaced during writer training")
    checkpoint_config = {
        **reader_config, "output": str(directory), "policies": "learned",
        "seeds": args.seeds, "steps": args.steps, "batch": args.batch,
        "eval_batch": args.eval_batch, "eval_cycles": args.eval_cycles,
        "queries": args.queries, "cost": args.cost, "device": args.device,
        "writer_only": True, "initial_dir": args.initial_dir, "pilot": args.pilot,
    }
    training_recipe = {
        **vars(args), "reader_seed": reader_seed,
        "train_stream_seed": derived_seed(seed, "train-stream"),
        "categorical_seed": derived_seed(seed, "categorical-actions"),
        "validation_seed": validation_seed,
        "heldout_seed": 3_700_000 + seed,
        "return": "mean strictly subsequent hard-query binary correctness within cycle",
        "baseline": "detached leave-one-out over other nonforced eligible rows",
    }
    checkpoint = directory / f"learned-{seed}.pt"
    with checkpoint.open("xb") as handle:
        torch.save({"model": model.state_dict(), "config": checkpoint_config, "seed": seed,
                    "reader_seed": reader_seed, "policy": "learned", "source": hashes(),
                    "algorithm": "binary-query-reward REINFORCE",
                    "training": training_recipe}, handle)
    heldout = {}
    if not args.pilot:
        for name, options in {
            "stream": {}, "noise32": {"noise": 32}, "return_a": {"return_to": "a"},
            "stay_b": {"return_to": "b"},
            "half_capacity": {"slots": reader_config["slots"] // 2},
            "erase_memory": {"reset_memory": True},
        }.items():
            heldout[name] = evaluate(model, evaluation_args, "learned", 3_700_000 + seed,
                                     cycles=args.eval_cycles, **options)
        for policy in ("mru", "fifo", "random"):
            heldout["writer_" + policy] = evaluate(
                model, evaluation_args, policy, 3_700_000 + seed, cycles=args.eval_cycles)
        # Paired heldout control: same calibrated reader, same initial writer
        # weights, same worlds, with no RL updates. Test is read only after
        # fixed-budget training and is never used to select a checkpoint.
        initial_model, _, _ = load_reader(Path(args.initial_dir), reader_seed, args.device)
        if initial_model.initial_source_checkpoint_sha256 != reader_checkpoint_sha256:
            raise RuntimeError("Paired initial checkpoint differs from training input")
        if parameter_hash(initial_model, reader_only=True) != frozen_hash:
            raise RuntimeError("Paired initial control has a different reader")
        if writer_parameter_hash(initial_model) != initial_writer_sha256:
            raise RuntimeError("Paired initial control has different initial writer weights")
        heldout["initial_learned_stream"] = evaluate(
            initial_model, evaluation_args, "learned", 3_700_000 + seed, cycles=args.eval_cycles)
        del initial_model
    result = {
        "algorithm": "REINFORCE with mean future query binary correctness",
        "seed": seed, "reader_seed": reader_seed, "config": checkpoint_config,
        "training": training_recipe,
        "reader_checkpoint": str(reader_path), "reader_checkpoint_sha256": reader_checkpoint_sha256,
        "initial_writer_parameter_sha256": initial_writer_sha256,
        "paired_initial_writer_verified": not args.pilot,
        "reader_and_halt_unchanged": True, "reader_parameter_sha256": frozen_hash,
        "sources": hashes(), "environment": environment(args.device),
        "seconds": elapsed, "pure_training_seconds": pure_training_seconds,
        "training_cuda_milliseconds": training_cuda_milliseconds,
        "peak_gpu_bytes": peak_memory, "validation": records,
        "heldout": heldout,
        "actual_write_counts": {name: int(value) for name, value in model.write_counts.items()},
        "parameter_change": writer_parameter_change(model, initial_writer),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "selected_checkpoint": "fixed_last",
    }
    write_json(directory / f"learned-{seed}.json", result)
    return result


def self_test():
    model = ReinforceWriter(contexts=3, vocab=8, slots=2, hidden=8, width=8,
                            read_dim=8, max_reads=3)
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("retain.", "event_state.")))
    from experiments.streaming.stream import Event
    state = model.initial(4, "cpu")
    rng = torch.Generator().manual_seed(71)
    for key in (0, 1, 2):
        event = Event("fact", torch.zeros(4, dtype=torch.long),
                      torch.full((4,), key), torch.full((4,), (key + 1) % 8),
                      torch.zeros(4))
        state, _ = model.write(state, event, generator=rng)
    for action in model.writer_actions[:2]:
        assert not action.eligible.any()
        assert action.log_probability.count_nonzero() == 0 and action.entropy.count_nonzero() == 0
    assert model.writer_actions[-1].eligible.all() and not state.memory.requires_grad
    loss, stats = policy_objective(
        model.writer_actions, [(state.clock + 1, torch.tensor([1., 0., 1., 0.]))], .001)
    loss.backward()
    assert stats["credited_optional_actions"] == 4
    assert all(p.grad is None for name, p in model.named_parameters() if not p.requires_grad)
    assert sum(p.grad.abs().sum().item() for p in model.retain.parameters() if p.grad is not None) > 0
    returns = torch.tensor([1., .2, .6, .8])
    eligible = torch.tensor([True, False, True, False])
    baseline = leave_one_out(returns, eligible)
    changed = returns.clone()
    changed[0] = 99
    assert baseline[0] == leave_one_out(changed, eligible)[0]
    assert torch.allclose(baseline, torch.tensor([.6, 0., 1., 0.]))
    model.eval()
    base = CacheAgent(contexts=3, vocab=8, slots=2, hidden=8, width=8, read_dim=8, max_reads=3)
    base.load_state_dict(model.state_dict())
    base.eval()
    state_a, state_b = model.initial(4, "cpu"), base.initial(4, "cpu")
    with torch.no_grad():
        for key in (0, 1, 2, 1):
            event = Event("fact", torch.zeros(4, dtype=torch.long),
                          torch.full((4,), key), torch.full((4,), (key + 1) % 8),
                          torch.zeros(4))
            state_a, _ = model.write(state_a, event)
            state_b, _ = base.write(state_b, event)
            assert torch.equal(state_a.memory, state_b.memory)
            assert torch.equal(state_a.hidden, state_b.hidden)
        query = Event("query", torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))
        model.train()
        full = model.read(state_a, query)
        model.eval()
        hard = model.read(state_a, query)
        assert torch.equal(full["answer"], hard["answer"])
        assert torch.equal(full["hard_prob"], hard["hard_prob"])
        assert torch.equal(full["reads"], hard["reads"])
    print("self-test passed: forced actions zero, hard cache, binary-reward gradients, LOO independence, frozen reader, deterministic base eval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-dir", default="data/streaming/large-stopping-v1")
    parser.add_argument("--reader-seed", type=int, help="Override matching reader seed for pilot only")
    parser.add_argument("--output", default="data/streaming/reinforce-writer-v1")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--eval-cycles", type=int, default=8)
    parser.add_argument("--queries", type=int, default=2)
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--entropy", type=float, default=.001)
    parser.add_argument("--cost", type=float, default=.02, help="Evaluation utility coefficient only, not RL reward")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.self_test:
        self_test()
        return
    if args.batch < 2 or min(args.steps, args.log_every, args.eval_batch, args.eval_cycles, args.queries) < 1:
        parser.error("batch>=2 and positive budgets required")
    if args.entropy < 0 or args.lr <= 0:
        parser.error("nonnegative entropy and positive learning rate required")
    if args.reader_seed is not None and not args.pilot:
        parser.error("Reader seed override is pilot-only; formal runs use each seed's calibrated reader")
    directory = Path(args.output)
    directory = (directory if directory.is_absolute() else ROOT / directory).resolve()
    initial_dir = Path(args.initial_dir)
    args.initial_dir = str((initial_dir if initial_dir.is_absolute() else ROOT / initial_dir).resolve())
    args.output = str(directory)
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)) or not seeds:
        parser.error("distinct nonempty seeds required")
    for seed in seeds:
        reader_seed = seed if args.reader_seed is None else args.reader_seed
        if not (Path(args.initial_dir) / f"mru-{reader_seed}.pt").is_file():
            parser.error(f"Reader checkpoint is not complete: mru-{reader_seed}.pt")
    directory.mkdir(parents=True, exist_ok=False)
    before = hashes()
    for name in before:
        with (directory / ("source-" + name)).open("xb") as handle:
            handle.write(Path(__file__).with_name(name).read_bytes())
    write_json(directory / "manifest.json", {
        "configuration": vars(args), "sources": before, "environment": environment(args.device),
        "protocol": [
            "Only retain/event_state train; pretrained reader and halt values remain frozen.",
            "Hard categorical eviction, including dropping incoming; no straight-through cache gradient.",
            "Forced empty fills have exactly zero log_probability/entropy and no policy-gradient credit.",
            "Within-cycle return is the mean of strictly subsequent hard-query binary correctness.",
            "Leave-one-out baseline uses other nonforced eligible rows only; detached, no learned critic.",
            "Entropy coefficient is an engineering hyperparameter, not a task/salience teacher.",
            "No answer CE, path/depth/phase inputs, correct-write labels, or salience labels train the policy.",
            "Memory and hidden numbers carry across cycles; policy graphs detach at each cycle boundary.",
            "Evaluation writer delegates to base deterministic argmin; no sampled policy at deployment.",
            "Fixed last checkpoint; pilot has no heldout evaluations; no test-based model selection.",
            "Formal initial-learned and final-learned stream controls use identical seed3700000+seed worlds.",
            "Formal budget chosen from val-only pilot-reinforce-writer-v1; fixed last 200 steps, not the best intermediate.",
            "Evaluation read cost is reported as utility only, not included in binary RL training reward.",
        ],
    })
    results = [train(seed, args, directory) for seed in seeds]
    if before != hashes():
        raise RuntimeError("Source changed during writer training")
    write_json(directory / "summary.json", {
        "seeds": seeds, "fixed_last": True, "pilot": args.pilot,
        "seconds": sum(result["seconds"] for result in results),
        "final_validation": {str(result["seed"]): result["validation"][-1]["validation"]["metrics"]
                             for result in results},
        "heldout": {str(result["seed"]): result["heldout"] for result in results},
        "reader_and_halt_unchanged": all(result["reader_and_halt_unchanged"] for result in results),
    })
    write_json(directory / "artifact_hashes.json", {
        path.name: file_hash(path) for path in sorted(directory.iterdir()) if path.is_file()
    })


if __name__ == "__main__":
    main()
