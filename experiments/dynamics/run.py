"""Reproducible persistent-stream training and closed-loop evaluation.

The learner receives only WorldBatch.observe(). Oracle state is used to make
supervised labels and score policies; it is never passed to DynamicsModel.
TBPTT detaches the state graph and preserves its numerical value. Only a new
independent stream resets the state, apart from the named reset intervention.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dynamics.model import DynamicsModel
from experiments.dynamics.world import StreamBatch, WorldBatch, generate_stream

KINDS = ("reservoir", "learned", "gru")
CONDITIONS = ("iid", "clock_0375", "overlap", "conflict", "long_gap", "sparse", "long")
CONTROLS = ("baseline", "small_noise", "reset", "half_step", "clock_half", "clock_double")
FORMAT_VERSION = "persistent-dynamics-v2"


def seed_for(seed: int, split: str, index: int = 0, condition: str = "train") -> int:
    """Stable split namespaces, independent of Python's randomized hash()."""
    value = f"{FORMAT_VERSION}/{seed}/{split}/{condition}/{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def seed_all(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def synchronize(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}/{value.dtype}/{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    folder = Path(__file__).parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("world.py", "model.py", "run.py", "test_run.py")}


def check_checkpoint_sources(saved: dict, current: dict) -> None:
    for name in ("world.py", "model.py"):
        if saved.get("source_hashes", {}).get(name) != current[name]:
            raise ValueError(f"checkpoint source mismatch for {name}; evaluate with its frozen source snapshot")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def environment(device: str) -> dict:
    record = {"python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": device, "threads": torch.get_num_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if torch.device(device).type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(gpu=properties.name, gpu_memory_bytes=properties.total_memory,
                      capability=list(torch.cuda.get_device_capability(device)))
    return record


def elapsed_times(stream: StreamBatch) -> tuple[float, ...]:
    times = stream.times.detach().cpu().tolist()
    # There is no unobserved prehistory before the first observation pulse.
    return (0.0,) + tuple(b - a for a, b in zip(times, times[1:]))


@dataclass
class TeacherTrajectory:
    observations: torch.Tensor
    targets: torch.Tensor
    elapsed: tuple[float, ...]
    stream: StreamBatch
    oracle_metrics: dict
    fingerprint: str


def teacher_action(world: WorldBatch) -> torch.Tensor:
    """A causal demonstration policy with timing margin inside the legal window.

    BEGIN immediately. FINISH halfway between actual readiness and deadline,
    rounded to the next observation time. This changes demonstration choice,
    not environment legality. Early-boundary v1 pilots are archived separately.
    The evaluator-only queue/timers are never features passed to the learner.
    """
    actions = world.oracle()
    center = .5 * (world.ready_time + world.deadline)
    return torch.where((actions >= 3) & (world.now < center - 1e-6), 0, actions)


@torch.no_grad()
def teacher_trajectory(stream: StreamBatch, device: str | torch.device = "cpu") -> TeacherTrajectory:
    # This small evaluator is faster on CPU than launching many tiny kernels.
    # Transfer the completed trajectory once, keeping the learner on its device.
    world = WorldBatch(stream, device="cpu")
    observations, targets = [], []
    for tick in range(stream.steps):
        observations.append(world.observe(tick))
        target = teacher_action(world)
        targets.append(target)
        world.act(target)
    x, y = torch.stack(observations), torch.stack(targets)
    fingerprint = tensor_digest({"observations": x, "targets": y})
    return TeacherTrajectory(x.to(device), y.to(device), elapsed_times(stream), stream,
                             world.metrics(), fingerprint)


class ActionStats:
    """Agreement diagnostics with separate WAIT/BEGIN/FINISH denominators."""

    def __init__(self, n_actions: int, device: str | torch.device, wait_weight: float = .1):
        self.n_actions, self.wait_weight = n_actions, wait_weight
        self.confusion = torch.zeros(n_actions, n_actions, dtype=torch.long, device=device)
        self.nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.weighted_sum = self.nll_sum.clone()
        self.weight_sum = self.nll_sum.clone()

    @torch.no_grad()
    def add(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.detach().argmax(-1)
        labels, predicted = target.reshape(-1), prediction.reshape(-1)
        self.confusion += torch.bincount(labels * self.n_actions + predicted,
                                        minlength=self.n_actions**2).view(self.n_actions, self.n_actions)
        losses = F.cross_entropy(logits.detach().reshape(-1, self.n_actions), labels,
                                 reduction="none")
        weights = torch.where(labels == 0, self.wait_weight, 1.0)
        self.nll_sum += losses.double().sum()
        self.weighted_sum += (losses * weights).double().sum()
        self.weight_sum += weights.double().sum()

    def result(self) -> dict:
        matrix = self.confusion.cpu()
        counts, correct = matrix.sum(1), matrix.diag()
        total = int(counts.sum())
        def accuracy(start, end):
            count = int(counts[start:end].sum())
            return {"count": count,
                    "accuracy": int(correct[start:end].sum()) / count if count else None}
        return {"count": total, "accuracy": int(correct.sum()) / total if total else None,
                "nll": float(self.nll_sum) / total if total else None,
                "weighted_ce": float(self.weighted_sum / self.weight_sum) if total else None,
                "wait": accuracy(0, 1), "nonwait": accuracy(1, self.n_actions),
                "begin": accuracy(1, 3), "finish": accuracy(3, self.n_actions),
                "target_counts": counts.tolist(), "prediction_counts": matrix.sum(0).tolist(),
                "confusion_target_by_prediction": matrix.tolist()}


def forward_chunk(model, trajectory: TeacherTrajectory, state: torch.Tensor,
                  start: int, end: int, max_step: float | None = None):
    logits = []
    for tick in range(start, end):
        output, state = model.step(trajectory.observations[tick], state,
                                   trajectory.elapsed[tick], max_step=max_step)
        logits.append(output)
    return torch.stack(logits), state


def make_model(kind: str, args) -> DynamicsModel:
    return DynamicsModel(width=args.width, kind=kind, max_step=args.max_step).to(args.device)


def train_one(kind: str, args) -> tuple[DynamicsModel, dict, dict]:
    # Encoder/head and the continuous recurrent matrices share initialization.
    seed_all(seed_for(args.seed, "initialization"), args.deterministic)
    model = make_model(kind, args)
    initial_hash = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    weights = torch.ones(model.n_actions, device=args.device)
    weights[0] = args.wait_weight
    stats = ActionStats(model.n_actions, args.device, args.wait_weight)
    curves, streams = [], []
    updates = observed_steps = stream_index = 0
    model.train()
    synchronize(args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    while updates < args.updates:
        dt = args.train_dts[stream_index % len(args.train_dts)]
        data_seed = seed_for(args.seed, "train", stream_index)
        stream = generate_stream(data_seed, args.batch, args.steps, dt=dt)
        trajectory = teacher_trajectory(stream, args.device)
        entry = {"index": stream_index, "seed": data_seed, "dt": dt,
                 "world_hash": stream.fingerprint, "teacher_hash": trajectory.fingerprint,
                 "case_ids": list(stream.case_ids), "generated_steps": stream.steps,
                 "consumed_steps": 0, "oracle_reference": trajectory.oracle_metrics}
        streams.append(entry)
        state = model.initial_state(stream.batch_size, args.device)
        for start in range(0, stream.steps, args.tbptt):
            if updates >= args.updates:
                break
            end = min(start + args.tbptt, stream.steps)
            optimizer.zero_grad(set_to_none=True)
            logits, state = forward_chunk(model, trajectory, state, start, end)
            target = trajectory.targets[start:end]
            loss = F.cross_entropy(logits.reshape(-1, model.n_actions), target.reshape(-1),
                                   weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss in {kind}, update {updates}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip,
                                                       error_if_nonfinite=True)
            optimizer.step()
            # Preserve the actual state across every optimizer/TBPTT boundary.
            state = state.detach()
            stats.add(logits, target)
            updates += 1
            observed_steps += end - start
            entry["consumed_steps"] = end
            if updates == 1 or updates % args.log_every == 0 or updates == args.updates:
                synchronize(args.device)
                point = {"update": updates, "stream": stream_index, "stream_step": end,
                         "dt": dt, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                         "seconds": time.perf_counter() - started}
                curves.append(point)
                print(json.dumps({"event": "train", "kind": kind, **point}), flush=True)
        stream_index += 1
    synchronize(args.device)
    seconds = time.perf_counter() - started
    report = {"updates": updates, "observations_per_stream_lane": observed_steps,
              "batch_observations": observed_steps * args.batch,
              "action_labels": observed_steps * args.batch * 2,
              "independent_stream_batches": len(streams), "seconds": seconds,
              "batch_observations_per_second": observed_steps * args.batch / seconds,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(args.device)
              if torch.device(args.device).type == "cuda" else None,
              "initial_parameter_hash": initial_hash, "final_parameter_hash": tensor_digest(model.state_dict()),
              "action_metrics": stats.result(), "curve": curves, "streams": streams,
              "final_state_hash": tensor_digest({"state": state})}
    return model, report, optimizer.state_dict()


def intervene(state: torch.Tensor, control: str, noise_scale: float,
              generator: torch.Generator) -> tuple[torch.Tensor, dict]:
    before = state
    if control == "reset":
        state = torch.zeros_like(state)
    elif control == "small_noise":
        # Relative to the batch's state RMS; no global RNG consumption.
        scale = before.square().mean().sqrt().clamp_min(1e-3)
        noise = torch.randn(state.shape, device=state.device, dtype=state.dtype,
                            generator=generator)
        state = state + noise_scale * scale * noise
    else:
        raise ValueError(control)
    return state, {"state_rms_before": float(before.square().mean().sqrt()),
                   "change_rms": float((state - before).square().mean().sqrt())}


@torch.no_grad()
def closed_loop_rollout(model, stream: StreamBatch, *, device="cpu", control="baseline",
                        max_step=.25, wait_weight=.1, noise_scale=.1, noise_seed=0) -> dict:
    """Actual model actions affect receipts, completion timing, and future input."""
    if control not in CONTROLS:
        raise ValueError(control)
    model.eval()
    world = WorldBatch(stream, device=device)
    state = model.initial_state(stream.batch_size, device)
    elapsed = elapsed_times(stream)
    stats = ActionStats(model.n_actions, device, wait_weight)
    post = ActionStats(model.n_actions, device, wait_weight)
    rng = torch.Generator(device=device).manual_seed(noise_seed)
    multiplier = {"clock_half": .5, "clock_double": 2.0}.get(control, 1.0)
    solver_step = max_step / 2 if control == "half_step" else max_step
    predictions, intervention = [], None
    midpoint_mask = midpoint_ids = midpoint_initially_clean = None
    midpoint_completed = torch.zeros_like(world.active)
    midpoint_clean_completed = torch.zeros_like(world.active)
    finite = torch.ones((), dtype=torch.bool, device=device)
    synchronize(device)
    started = time.perf_counter()
    for tick in range(stream.steps):
        obs = world.observe(tick)
        label = teacher_action(world)  # Diagnostic only; never part of model.step().
        if tick == stream.steps // 2:
            # Score content that is already being retained, excluding commands
            # at this tick and requests whose processing has already finished.
            midpoint_mask = (world.active & world.current_eligible & world.started
                             & (world.stream.kind[tick] == 0) & (world.ready_time > world.now))
            midpoint_ids = world.request_id.clone()
            midpoint_initially_clean = midpoint_mask & ~world.had_error
        if tick == stream.steps // 2 and control in {"small_noise", "reset"}:
            state, intervention = intervene(state, control, noise_scale, rng)
            intervention.update(tick=tick, time=float(stream.times[tick]))
        logits, state = model.step(obs, state, elapsed[tick] * multiplier, max_step=solver_step)
        prediction = logits.argmax(-1)
        stats.add(logits, label)
        if tick >= stream.steps // 2:
            post.add(logits, label)
        finite &= torch.isfinite(logits).all() & torch.isfinite(state).all()
        predictions.append(prediction)
        world.act(prediction)
        if midpoint_mask is not None:
            completed = (midpoint_mask & (world.request_id == midpoint_ids)
                         & (prediction >= 3) & (world.last_result == 1))
            midpoint_completed |= completed
            midpoint_clean_completed |= completed & ~world.had_error
    synchronize(device)
    seconds = time.perf_counter() - started
    if not bool(finite):
        raise FloatingPointError("nonfinite model state/logits during evaluation")
    pending_count = int(midpoint_mask.sum())
    completed_count = int(midpoint_completed.sum())
    clean_count = int(midpoint_clean_completed.sum())
    return {"mode": "closed_loop", "control": control,
            "solver_control_applicable": getattr(model, "kind", None) != "gru",
            "world_hash": stream.fingerprint, "world": world.metrics(per_stream=True),
            "action_metrics": stats.result(), "post_midpoint_action_metrics": post.result(),
            "prediction_hash": tensor_digest({"predictions": torch.stack(predictions)}),
            "final_state_hash": tensor_digest({"state": state}),
            "final_state_rms": float(state.square().mean().sqrt()),
            "intervention": intervention, "intervention_count": int(intervention is not None),
            "midpoint_tick": stream.steps // 2,
            "midpoint_pending": pending_count, "midpoint_completed": completed_count,
            "midpoint_completion_rate": completed_count / pending_count if pending_count else None,
            "midpoint_initially_clean": int(midpoint_initially_clean.sum()),
            "midpoint_clean_completed": clean_count,
            "midpoint_clean_completion_rate": clean_count / pending_count if pending_count else None,
            "midpoint_target_hash": tensor_digest({"mask": midpoint_mask,
                "request_ids": torch.where(midpoint_mask, midpoint_ids, -1)}),
            "midpoint_pending_per_stream": midpoint_mask.sum(1).cpu().tolist(),
            "midpoint_completed_per_stream": midpoint_completed.sum(1).cpu().tolist(),
            "model_clock_multiplier": multiplier, "max_solver_step": solver_step,
            "seconds": seconds, "batch_observations_per_second": stream.steps * stream.batch_size / seconds}


@torch.no_grad()
def teacher_forced_rollout(model, trajectory: TeacherTrajectory, *, device="cpu",
                           max_step=.25, wait_weight=.1) -> dict:
    model.eval()
    state = model.initial_state(trajectory.stream.batch_size, device)
    stats = ActionStats(model.n_actions, device, wait_weight)
    predictions = []
    synchronize(device)
    started = time.perf_counter()
    for tick in range(trajectory.stream.steps):
        logits, state = model.step(trajectory.observations[tick], state,
                                   trajectory.elapsed[tick], max_step=max_step)
        stats.add(logits, trajectory.targets[tick])
        predictions.append(logits.argmax(-1))
    synchronize(device)
    return {"mode": "teacher_forced", "action_metrics": stats.result(),
            "teacher_hash": trajectory.fingerprint,
            "prediction_hash": tensor_digest({"predictions": torch.stack(predictions)}),
            "seconds": time.perf_counter() - started}


def evaluation_specs(args, split: str) -> list[dict]:
    names = ["iid"] if split == "validation" else args.eval_conditions
    specs = []
    for name in names:
        condition = "train" if name in {"iid", "clock_0375"} else name
        steps = args.eval_steps * (args.long_factor if name == "long" else 1)
        dt = args.unseen_dt if name == "clock_0375" else args.dt
        specs.append({"name": name, "condition": condition, "steps": steps, "dt": dt,
                      "seed": seed_for(args.seed, split, condition=name)})
    return specs


def evaluate_suite(models: dict, args, split: str) -> tuple[dict, list[dict]]:
    results = {kind: [] for kind in models}
    manifest = []
    for spec in evaluation_specs(args, split):
        stream = generate_stream(spec["seed"], args.eval_batch, spec["steps"],
                                 dt=spec["dt"], condition=spec["condition"])
        trajectory = teacher_trajectory(stream, args.device)
        metadata = {**spec, "world_hash": stream.fingerprint,
                    "teacher_hash": trajectory.fingerprint, "case_ids": list(stream.case_ids),
                    "world_config": asdict(stream.config), "oracle_reference": trajectory.oracle_metrics}
        manifest.append(metadata)
        for kind, model in models.items():
            controls = ["baseline"] if split == "validation" else args.controls
            # Clock ablations are paired on exactly the IID schedule only.
            controls = [c for c in controls if spec["name"] == "iid" or not c.startswith("clock_")]
            rollouts = [closed_loop_rollout(model, stream, device=args.device, control=control,
                                           max_step=args.max_step, wait_weight=args.wait_weight,
                                           noise_scale=args.noise_scale,
                                           noise_seed=seed_for(args.seed, split, condition="noise-" + spec["name"]))
                        for control in controls]
            item = {"condition": spec["name"], "world_hash": stream.fingerprint,
                    "closed_loop": rollouts,
                    "teacher_forced": teacher_forced_rollout(model, trajectory, device=args.device,
                                                             max_step=args.max_step, wait_weight=args.wait_weight)}
            results[kind].append(item)
            baseline = next((r for r in rollouts if r["control"] == "baseline"), None)
            print(json.dumps({"event": "evaluation", "split": split, "kind": kind,
                              "condition": spec["name"],
                              "clean_completion_rate": baseline["world"].get("clean_completion_rate")
                              if baseline else None,
                              "completion_rate": baseline["world"]["completion_rate"] if baseline else None}),
                  flush=True)
    return results, manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=ROOT / "data" / "dynamics" / "pilot")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--kinds", nargs="+", default=list(KINDS))
    result.add_argument("--checkpoint", type=Path, nargs="+", help="evaluate saved final.pt files without training")
    result.add_argument("--steps", type=int, default=512, help="ticks per independent training stream")
    result.add_argument("--updates", type=int, default=256, help="optimizer/TBPTT chunks per model")
    result.add_argument("--batch", type=int, default=64)
    result.add_argument("--width", type=int, default=128)
    result.add_argument("--tbptt", type=int, default=64)
    result.add_argument("--dt", type=float, default=.25, help="standard evaluation observation interval")
    result.add_argument("--train-dts", type=float, nargs="+", default=[.125, .25, .5])
    result.add_argument("--unseen-dt", type=float, default=.375)
    result.add_argument("--max-step", type=float, default=.25)
    result.add_argument("--lr", type=float, default=.001)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--wait-weight", type=float, default=.1)
    result.add_argument("--clip", type=float, default=1.0)
    result.add_argument("--eval-steps", type=int, default=512)
    result.add_argument("--eval-batch", type=int, default=32)
    result.add_argument("--eval-conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    result.add_argument("--controls", nargs="+", choices=CONTROLS, default=list(CONTROLS))
    result.add_argument("--long-factor", type=int, default=4)
    result.add_argument("--noise-scale", type=float, default=.1)
    result.add_argument("--skip-test", action="store_true", help="pilot: train and validation only")
    result.add_argument("--device", default="auto")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--log-every", type=int, default=32)
    result.add_argument("--no-deterministic", dest="deterministic", action="store_false", default=True)
    return result


def parse_args(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    args.kinds = [part for value in args.kinds for part in value.split(",")]
    if not args.kinds or len(set(args.kinds)) != len(args.kinds) or any(k not in KINDS for k in args.kinds):
        arg_parser.error(f"--kinds must contain distinct values from {KINDS}")
    for name in ("steps", "updates", "batch", "width", "tbptt", "eval_steps", "eval_batch",
                 "long_factor", "threads", "log_every"):
        if getattr(args, name) < 1:
            arg_parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("dt", "unseen_dt", "max_step", "lr", "wait_weight", "clip", "noise_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            arg_parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        arg_parser.error("--weight-decay must be finite and nonnegative")
    if not args.train_dts or any(not math.isfinite(v) or v <= 0 for v in args.train_dts):
        arg_parser.error("--train-dts must be finite and positive")
    if len(set(args.eval_conditions)) != len(args.eval_conditions) or len(set(args.controls)) != len(args.controls):
        arg_parser.error("evaluation conditions and controls must be distinct")
    if "baseline" not in args.controls:
        arg_parser.error("--controls must include baseline")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output = args.output.resolve()
    return args


def config_dict(args) -> dict:
    return {name: str(value) if isinstance(value, Path) else
            [str(p) for p in value] if name == "checkpoint" and value else value
            for name, value in vars(args).items()}


def main(argv=None) -> dict:
    args = parse_args(argv)
    if (args.output / "summary.json").exists() or (args.output / "config.json").exists():
        raise FileExistsError(f"use a new output directory: {args.output}")
    torch.set_num_threads(args.threads)
    seed_all(args.seed, args.deterministic)
    hashes = source_hashes()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = args.output / "source"
    sources.mkdir(exist_ok=True)
    for name in hashes:
        (sources / name).write_bytes((Path(__file__).parent / name).read_bytes())
    config = config_dict(args)
    atomic_json(args.output / "config.json", config)
    summary = {"format": FORMAT_VERSION, "config": config, "source_hashes": hashes,
               "environment": environment(args.device), "status": "running", "models": {},
               "comparison": "State width and external streams are paired. Parameter counts and compute differ; no equal-FLOP claim.",
               "evaluation": "Closed-loop world outcomes are primary; teacher-forced action agreement is secondary."}
    models, checkpoints = {}, {}
    if args.checkpoint:
        for path in args.checkpoint:
            saved = torch.load(path, map_location=args.device, weights_only=False)
            check_checkpoint_sources(saved, hashes)
            spec = saved["model_config"]
            kind = spec["kind"]
            if kind in models:
                raise ValueError(f"duplicate checkpoint kind: {kind}")
            model = DynamicsModel(**spec).to(args.device)
            model.load_state_dict(saved["state_dict"])
            if spec["max_step"] != args.max_step:
                raise ValueError("pass the saved --max-step when evaluating a checkpoint")
            models[kind] = model
            summary["models"][kind] = {"parameters": model.parameter_report(),
                                       "checkpoint": str(path.resolve()),
                                       "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "training_source_hashes": saved["source_hashes"],
                                       "training_config": saved["config"],
                                       "parameter_hash": tensor_digest(model.state_dict())}
    else:
        for kind in args.kinds:
            model, training, optimizer = train_one(kind, args)
            models[kind] = model
            checkpoint_path = args.output / kind / "final.pt"
            record = {"parameters": model.parameter_report(), "training": training,
                      "checkpoint": str(checkpoint_path)}
            summary["models"][kind] = record
            checkpoints[kind] = {"format": FORMAT_VERSION, "model_config": {
                "input_size": model.input_size, "width": model.width, "n_actions": model.n_actions,
                "kind": kind, "max_step": model.max_step}, "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer, "config": config, "source_hashes": hashes,
                "training": training, "parameters": model.parameter_report()}
            atomic_checkpoint(checkpoint_path, checkpoints[kind])
            atomic_json(args.output / "summary.json", summary)
    for split in (["validation"] if args.skip_test else ["validation", "test"]):
        evaluation, manifest = evaluate_suite(models, args, split)
        summary[split + "_streams"] = manifest
        for kind in models:
            summary["models"][kind][split] = evaluation[kind]
            if kind in checkpoints:
                checkpoints[kind][split] = evaluation[kind]
                atomic_checkpoint(args.output / kind / "final.pt", checkpoints[kind])
        atomic_json(args.output / "summary.json", summary)
    for kind, checkpoint in checkpoints.items():
        path = args.output / kind / "final.pt"
        summary["models"][kind]["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["status"] = "complete"
    summary["test_skipped"] = args.skip_test
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)
    return summary


if __name__ == "__main__":
    main()
