"""Controlled activity-related learning on an observed-context A -> B -> A task.

This is a rate-based, single-softmax mechanism experiment, not a LIF replica.
The supplied context interactions make the task linearly solvable; learning a
representation, delayed credit assignment, and structural changes are not tested.

Local delta and one-layer cross-entropy BP are mathematically the same update.
Reward eligibility is an analytic policy-gradient estimator using binary
environment feedback, not a gradient-free rule or an internal self-evaluator.
Frequency homeostasis is only one activity-only rule, not all Hebbian learning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("frozen", "frequency", "local_delta", "reward_local", "bp")
SOURCES = {
    "oja": "https://doi.org/10.1007/BF00275687",
    "reward_stdp": "https://izhikevich.org/publications/dastdp.pdf",
    "eprop": "https://doi.org/10.1038/s41467-020-17236-y",
    "set": "https://doi.org/10.1038/s41467-018-04316-3",
    "rigl": "https://arxiv.org/abs/1911.11134",
    "forward_forward_distinct": "https://arxiv.org/abs/2212.13345",
    "forward_gradient_distinct": "https://arxiv.org/abs/2202.08587",
}
LIMITATIONS = [
    "Fixed engineered [1,x,c,context_scale*c*x] features already express both tasks.",
    "Single-layer local delta is mathematically equivalent to this BP baseline.",
    "Frequency homeostasis changes output biases without labels; it is one rule.",
    "Reward-local receives binary correctness, while BP/local-delta receive full labels.",
    "Reward-local uses an analytic policy-gradient estimator with immediate feedback.",
    "Signed rate features are not biological spike counts or a LIF simulation.",
    "No learned self-evaluation head, topology change, replay, EWC, or protected weights.",
    "Mixed-context training is a separate feasibility control, not sequential replay.",
    "Finite small-task seeds do not establish general continual-learning performance.",
]


def derived_seed(seed: int, name: str) -> int:
    payload = f"local-v1/{seed}/{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class Split:
    x: Tensor
    labels_a: Tensor


def features_for_context(x: Tensor, context: int | Tensor,
                         context_scale: float = 0.5) -> Tensor:
    """Return [1, x, c, scale*c*x], with c=+1 for A and -1 for B."""
    context = torch.as_tensor(context, dtype=torch.long, device=x.device)
    if context.ndim == 0:
        context = context.expand(x.shape[0])
    if context.shape != (x.shape[0],) or ((context < 0) | (context > 1)).any():
        raise ValueError("context must be 0/1, scalar or one entry per example")
    sign = (1 - 2 * context).to(x.dtype).unsqueeze(1)
    return torch.cat((torch.ones_like(sign), x, sign, context_scale * sign * x), 1)


@dataclass(frozen=True)
class ContextTask:
    splits: dict[str, Split]
    teacher: Tensor
    classes: int
    context_scale: float

    @property
    def feature_dim(self) -> int:
        return 2 * self.teacher.shape[1] + 2

    def batch(self, split: str, indices: Tensor | None = None,
              context: int | Tensor = 0) -> tuple[Tensor, Tensor]:
        data = self.splits[split]
        x, labels = data.x, data.labels_a
        if indices is not None:
            x, labels = x[indices], labels[indices]
        ctx = torch.as_tensor(context, dtype=torch.long, device=x.device)
        features = features_for_context(x, ctx, self.context_scale)
        return features, (labels + ctx) % self.classes


def make_task(seed: int, *, input_dim: int = 32, classes: int = 3,
              n_train: int = 4096, n_val: int = 1024, n_test: int = 2048,
              context_scale: float = 0.5, device: str = "cpu") -> ContextTask:
    """Independent splits, paired A/B raw inputs, and no use of global RNG."""
    if min(input_dim, n_train, n_val, n_test) < 1 or classes < 2:
        raise ValueError("positive dimensions/split sizes and >=2 classes required")
    if context_scale <= 0:
        raise ValueError("context_scale must be positive")
    generator = torch.Generator(device="cpu").manual_seed(derived_seed(seed, "teacher"))
    teacher = torch.randn(classes, input_dim, generator=generator)
    teacher = F.normalize(teacher, dim=1)
    splits = {}
    for name, size in (("train", n_train), ("val", n_val), ("test", n_test)):
        generator = torch.Generator(device="cpu").manual_seed(derived_seed(seed, name))
        x = torch.randn(size, input_dim, generator=generator)
        labels = (x @ teacher.T).argmax(1)
        splits[name] = Split(x.to(device), labels.to(device))
    return ContextTask(splits, teacher.to(device), classes, context_scale)


def sample_batch(task: ContextTask, split: str, context: int | Tensor,
                 batch_size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Sample only using the supplied RNG, including for CUDA task tensors."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = torch.randint(len(task.splits[split].x), (batch_size,),
                            generator=generator, device=generator.device)
    indices = indices.to(task.splits[split].x.device)
    return task.batch(split, indices, context)


class LocalClassifier(nn.Module):
    """Identical parameterization for every learning rule; column zero is bias."""

    def __init__(self, feature_dim: int, classes: int, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(derived_seed(seed, "weights"))
        self.weight = nn.Parameter(0.02 * torch.randn(classes, feature_dim,
                                                    generator=generator))

    def forward(self, features: Tensor) -> Tensor:
        return features @ self.weight.T


@dataclass
class ActivityState:
    rate_ema: Tensor
    reward_baseline: float = 0.0

    @classmethod
    def new(cls, classes: int) -> ActivityState:
        return cls(torch.full((classes,), 1.0 / classes))


@torch.no_grad()
def local_delta_update(model: LocalClassifier, features: Tensor, targets: Tensor,
                       lr: float) -> None:
    probability = model(features).softmax(1)
    residual = F.one_hot(targets, model.weight.shape[0]).to(features.dtype) - probability
    model.weight.add_(residual.T @ features, alpha=lr / features.shape[0])


@torch.no_grad()
def frequency_homeostasis_update(model: LocalClassifier, actions: Tensor,
                                 state: ActivityState, lr: float,
                                 rate_decay: float = 0.95) -> None:
    """Only output activity is observed; no target/reward argument exists."""
    classes = model.weight.shape[0]
    activity = F.one_hot(actions, classes).to(model.weight.dtype).mean(0)
    state.rate_ema.mul_(rate_decay).add_(activity, alpha=1 - rate_decay)
    model.weight[:, 0].add_(1.0 / classes - state.rate_ema, alpha=lr)


@torch.no_grad()
def reward_eligibility_update(model: LocalClassifier, features: Tensor,
                              probability: Tensor, actions: Tensor,
                              rewards: Tensor, state: ActivityState, lr: float,
                              baseline_decay: float = 0.99) -> None:
    """Three factors: pre * (sampled post - expected post) * binary reward RPE.

    The current trial's eligibility is consumed immediately, with no temporal
    trace claim. No labels are available to this update, only 0/1 rewards.
    """
    if ((rewards != 0) & (rewards != 1)).any():
        raise ValueError("environment rewards must be binary")
    innovation = F.one_hot(actions, model.weight.shape[0]).to(features.dtype) - probability
    modulation = rewards.to(features.dtype) - state.reward_baseline
    model.weight.add_((modulation[:, None] * innovation).T @ features,
                      alpha=lr / features.shape[0])
    # Use the pre-update baseline for credit assignment, then update it.
    state.reward_baseline = (baseline_decay * state.reward_baseline
                             + (1 - baseline_decay) * rewards.float().mean().item())


def learning_step(method: str, model: LocalClassifier, features: Tensor,
                  targets: Tensor, state: ActivityState, lr: float,
                  generator: torch.Generator,
                  optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    with torch.no_grad():
        probability = model(features).softmax(1)
        actions = torch.multinomial(probability, 1, generator=generator).squeeze(1)
        rewards = (actions == targets).to(features.dtype)  # independent environment
        stats = {
            "sampled_reward": rewards.mean().item(),
            "sampled_output_rate": F.one_hot(actions, probability.shape[1]).float().mean(0).tolist(),
            "mean_output_probability": probability.mean(0).tolist(),
        }
    if method == "frozen":
        pass  # Bias and all auxiliary state are frozen too.
    elif method == "frequency":
        frequency_homeostasis_update(model, actions, state, lr)
    elif method == "local_delta":
        local_delta_update(model, features, targets, lr)
    elif method == "reward_local":
        reward_eligibility_update(model, features, probability, actions, rewards, state, lr)
    elif method == "bp":
        if optimizer is None:
            raise ValueError("BP requires an optimizer")
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()
    else:
        raise ValueError(method)
    return stats


@torch.no_grad()
def evaluate(model: LocalClassifier, task: ContextTask, split: str) -> dict[str, Any]:
    result = {}
    for context, name in ((0, "A"), (1, "B")):
        features, targets = task.batch(split, context=context)
        logits = model(features)
        probability = logits.softmax(1)
        prediction = logits.argmax(1)
        result[name] = {
            "accuracy": (prediction == targets).float().mean().item(),
            "nll": F.cross_entropy(logits, targets).item(),
            "mean_output_rate": probability.mean(0).tolist(),
            "greedy_output_rate": F.one_hot(prediction, task.classes).float().mean(0).tolist(),
            "target_rate": F.one_hot(targets, task.classes).float().mean(0).tolist(),
            "count": targets.numel(),
        }
    return result


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    input_dim: int = 32
    classes: int = 3
    context_scale: float = 0.5
    n_train: int = 4096
    n_val: int = 1024
    n_test: int = 2048
    phase_steps: int = 800
    batch_size: int = 128
    eval_every: int = 50
    lr: float = 0.2
    threads: int = 1
    pilot: bool = False


def tensor_hash(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("local_learning.py", "test_local_learning.py")
    }


def write_json(path: Path, value: Any) -> None:
    # Never overwrite an artifact, including partial output from an earlier run.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def save_checkpoint(path: Path, model: LocalClassifier, state: ActivityState,
                    metadata: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        torch.save({"model": model.state_dict(), "rate_ema": state.rate_ema,
                    "reward_baseline": state.reward_baseline, **metadata}, handle)


def stream_indices(seed: int, phase: str, count: int, config: Config) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(derived_seed(seed, f"stream/{phase}"))
    return torch.randint(config.n_train, (count, config.batch_size), generator=generator)


def run_regime(method: str, regime: str, seed: int, task: ContextTask,
               config: Config, directory: Path) -> dict[str, Any]:
    model = LocalClassifier(task.feature_dim, task.classes, seed)
    state = ActivityState.new(task.classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr) if method == "bp" else None
    generator = torch.Generator(device="cpu").manual_seed(derived_seed(seed, f"actions/{regime}"))
    if regime == "sequential":
        phases = (("A1", 0, config.phase_steps), ("B", 1, config.phase_steps),
                  ("A2", 0, config.phase_steps))
    elif regime == "fresh_A":
        phases = (("A2", 0, config.phase_steps),)
    elif regime == "mixed":
        phases = (("mixed", None, 3 * config.phase_steps),)
    else:
        raise ValueError(regime)
    result: dict[str, Any] = {"method": method, "regime": regime, "seed": seed,
                              "initial_weight_hash": tensor_hash(model.weight),
                              "parameters": model.weight.numel(), "phases": []}
    result["initial_val"] = evaluate(model, task, "val")
    if not config.pilot:
        result["initial_test"] = evaluate(model, task, "test")
    total_start = time.perf_counter()
    for phase, context, steps in phases:
        indices = stream_indices(seed, phase, steps, config)
        contexts = None
        if context is None:
            context_rng = torch.Generator().manual_seed(derived_seed(seed, "mixed/contexts"))
            contexts = torch.randint(2, indices.shape, generator=context_rng)
        curve = [{"step": 0, "val": evaluate(model, task, "val"), "seconds": 0.0}]
        start = time.perf_counter()
        update_seconds = 0.0
        for step, index in enumerate(indices, 1):
            ctx = context if contexts is None else contexts[step - 1]
            features, targets = task.batch("train", index, ctx)
            update_start = time.perf_counter()
            statistics_step = learning_step(method, model, features, targets, state,
                                            config.lr, generator, optimizer)
            update_seconds += time.perf_counter() - update_start
            if step % config.eval_every == 0 or step == steps:
                curve.append({"step": step, "val": evaluate(model, task, "val"),
                              "seconds": time.perf_counter() - start,
                              "last_batch": statistics_step,
                              "rate_ema": state.rate_ema.tolist(),
                              "reward_baseline": state.reward_baseline})
        phase_result: dict[str, Any] = {
            "phase": phase, "context": context, "steps": steps,
            "examples": steps * config.batch_size, "curve": curve,
            "seconds": time.perf_counter() - start, "update_seconds": update_seconds,
            "final_train": evaluate(model, task, "train"),
            "final_val": evaluate(model, task, "val"),
        }
        if not config.pilot:
            phase_result["test"] = evaluate(model, task, "test")
        filename = f"{method}-{regime}-seed{seed}-{phase}.pt"
        save_checkpoint(directory / filename, model, state,
                        {"method": method, "regime": regime, "seed": seed, "phase": phase})
        phase_result["checkpoint"] = filename
        phase_result["final_weight_hash"] = tensor_hash(model.weight)
        result["phases"].append(phase_result)
        print(json.dumps({"event": "phase_complete", "seed": seed, "method": method,
                          "regime": regime, "phase": phase,
                          "val": {name: phase_result["final_val"][name]["accuracy"]
                                  for name in ("A", "B")},
                          "seconds": phase_result["seconds"]}), flush=True)
    result["seconds"] = time.perf_counter() - total_start
    return result


def first_reached(curve: list[dict[str, Any]], context: str,
                  threshold: float) -> int | None:
    return next((point["step"] for point in curve
                 if point["val"][context]["accuracy"] >= threshold), None)


def summarize_seed(runs: dict[str, dict[str, Any]], pilot: bool) -> dict[str, Any]:
    result = {}
    split = "final_val" if pilot else "test"
    for method in METHODS:
        sequential = runs[f"{method}/sequential"]
        first, second, third = sequential["phases"]
        fresh = runs[f"{method}/fresh_A"]["phases"][0]
        mixed = runs[f"{method}/mixed"]["phases"][0]
        original_accuracy = first["final_val"]["A"]["accuracy"]
        result[method] = {
            "initial_A": sequential["initial_val" if pilot else "initial_test"]["A"]["accuracy"],
            "A_after_A1": first[split]["A"]["accuracy"],
            "A_after_B": second[split]["A"]["accuracy"],
            "B_after_B": second[split]["B"]["accuracy"],
            "A_after_A2": third[split]["A"]["accuracy"],
            "B_after_A2": third[split]["B"]["accuracy"],
            "A_forgetting_after_B": first[split]["A"]["accuracy"] - second[split]["A"]["accuracy"],
            "B_forgetting_after_A2": second[split]["B"]["accuracy"] - third[split]["B"]["accuracy"],
            "mixed_A": mixed[split]["A"]["accuracy"],
            "mixed_B": mixed[split]["B"]["accuracy"],
            "A1_steps_to_90_val": first_reached(first["curve"], "A", 0.90),
            "A2_steps_to_90_val": first_reached(third["curve"], "A", 0.90),
            "fresh_A_steps_to_90_val": first_reached(fresh["curve"], "A", 0.90),
            "A2_steps_to_A1_terminal_val": first_reached(third["curve"], "A", original_accuracy),
            "fresh_A_steps_to_A1_terminal_val": first_reached(fresh["curve"], "A", original_accuracy),
            "sequential_seconds": sequential["seconds"],
        }
    return result


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for method in METHODS:
        result[method] = {}
        for metric in summaries[0][method]:
            values = [summary[method][metric] for summary in summaries]
            observed = [value for value in values if value is not None]
            result[method][metric] = {
                "values": values, "reached_count": len(observed), "seeds": len(values),
                "mean": statistics.mean(observed) if observed else None,
                "std": statistics.stdev(observed) if len(observed) > 1 else 0.0 if observed else None,
            }
    return result


def run_experiment(config: Config, output: str | Path) -> dict[str, Any]:
    if min(config.phase_steps, config.batch_size, config.eval_every, config.threads) < 1:
        raise ValueError("positive steps, batch size, evaluation interval, and threads required")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds) or config.lr <= 0:
        raise ValueError("unique nonempty seeds and positive learning rate required")
    directory = Path(output)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    sources_before = source_hashes()
    manifest = {
        "config": asdict(config), "sources": sources_before, "citations": SOURCES,
        "limitations": LIMITATIONS, "methods": list(METHODS),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "torch": torch.__version__, "platform": platform.platform(),
                        "device": "cpu", "threads": torch.get_num_threads()},
        "protocol": {
            "test_used_for_selection": False, "pilot_test_metrics_disabled": config.pilot,
            "selection": "fixed budgets, no checkpoint selection",
            "relearning": "first validation checkpoint reaching threshold; checkpoint resolution eval_every",
            "fresh_control": "same initial weights and A2 minibatch indices; newly initialized rule state",
            "same_regime_data_across_methods": True,
            "compute": "equal examples/updates, not equal FLOPs; report update and wall times",
        },
    }
    write_json(directory / "manifest.json", manifest)
    all_start = time.perf_counter()
    summaries = []
    for seed in config.seeds:
        task = make_task(seed, input_dim=config.input_dim, classes=config.classes,
                         n_train=config.n_train, n_val=config.n_val, n_test=config.n_test,
                         context_scale=config.context_scale)
        dataset_hashes = {"teacher": tensor_hash(task.teacher)}
        for name, data in task.splits.items():
            dataset_hashes[name] = {"x": tensor_hash(data.x), "labels_a": tensor_hash(data.labels_a)}
        write_json(directory / f"dataset-seed{seed}.json", dataset_hashes)
        with (directory / f"dataset-seed{seed}.pt").open("xb") as handle:
            torch.save({"teacher": task.teacher, "splits": {
                name: {"x": data.x, "labels_a": data.labels_a}
                for name, data in task.splits.items()}}, handle)
        runs = {}
        for method in METHODS:
            for regime in ("sequential", "fresh_A", "mixed"):
                run = run_regime(method, regime, seed, task, config, directory)
                runs[f"{method}/{regime}"] = run
                write_json(directory / f"{method}-{regime}-seed{seed}.json", run)
        summary = summarize_seed(runs, config.pilot)
        summaries.append(summary)
        write_json(directory / f"summary-seed{seed}.json", summary)
    summary = {"config": asdict(config), "metrics_split": "val" if config.pilot else "test",
               "metrics": aggregate(summaries), "seconds": time.perf_counter() - all_start,
               "sources_unchanged": source_hashes() == sources_before}
    if not summary["sources_unchanged"]:
        raise RuntimeError("Source changed during the run; artifacts are not a frozen-code result")
    write_json(directory / "summary.json", summary)
    write_json(directory / "artifact_hashes.json", {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir()) if path.is_file()
    })
    print(json.dumps({"event": "finished", "output": str(directory),
                      "seconds": summary["seconds"]}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/streaming/local-v1")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--phase-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = Config(seeds=tuple(int(value) for value in args.seeds.split(",")),
                    phase_steps=args.phase_steps, batch_size=args.batch_size,
                    eval_every=args.eval_every, lr=args.lr, threads=args.threads,
                    pilot=args.pilot)
    run_experiment(config, args.output)


if __name__ == "__main__":
    main()
