"""Train one persistent controller on one unbounded stream.

Every observation advances the same world and hidden state. TBPTT truncates
the graph after an update without changing the hidden value. Checkpoints are
taken at these optimizer boundaries and contain the complete continuation,
including Adam, the world's scheduler, the finite experience reservoir and all
random generators. No training or adaptation boundary creates a new world.

Examples (run from the repository root)::

    .venv/Scripts/python -m experiments.continuity.run --variants c128 online128
    .venv/Scripts/python -m experiments.continuity.run --resume RUN/checkpoint.pt
    .venv/Scripts/python -m experiments.continuity.run --adapt-from RUN/checkpoint.pt

Adaptation compares no updates, one current-data update, two current-data
updates, and one current plus one past-data update. Reprocessing either kind
of data never replaces the live controller's state or advances its world.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.continuity.model import AgentModel
from experiments.continuity.world import OnlineWorld

FORMAT_VERSION = "continuity-trainer-v1"
ADAPTATION_MODES = ("frozen", "current", "current_repeat", "replay")


def seed_for(seed: int, namespace: str, index: int = 0) -> int:
    value = f"{FORMAT_VERSION}/{seed}/{namespace}/{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def seed_all(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
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


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return deepcopy(value)


def state_digest(value: Any) -> str:
    """Canonical content hash, independent of device and pickle storage IDs."""
    digest = hashlib.sha256()

    def put(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor/{tensor.dtype}/{tuple(tensor.shape)}/".encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(f"ndarray/{array.dtype}/{array.shape}/".encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                put(key)
                put(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"list[" if isinstance(item, list) else b"tuple[")
            for part in item:
                put(part)
            digest.update(b"]")
        elif isinstance(item, float):
            digest.update(b"float/" + struct.pack("!d", item))
        elif isinstance(item, (str, int, bool)) or item is None:
            digest.update(type(item).__name__.encode() + b"/")
            digest.update(json.dumps(item, ensure_ascii=False).encode() + b";")
        else:
            raise TypeError(f"unsupported checkpoint hash value: {type(item).__name__}")

    put(value)
    return digest.hexdigest()


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    return state_digest(tensors)


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []}


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint includes CUDA RNG state but CUDA is unavailable")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def source_snapshots() -> dict[str, str]:
    folder = Path(__file__).parent
    return {name: (folder / name).read_text(encoding="utf-8")
            for name in ("world.py", "model.py", "run.py", "test_run.py")
            if (folder / name).is_file()}


def source_hashes(sources: dict[str, str] | None = None) -> dict[str, str]:
    return {name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in (source_snapshots() if sources is None else sources).items()}


def runtime_record(device: str | torch.device) -> dict[str, Any]:
    record = {"python": sys.version, "platform": platform.platform(),
              "machine": platform.machine(), "processor": platform.processor(),
              "torch": str(torch.__version__), "numpy": str(np.__version__),
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "device": str(device), "cpu_threads": torch.get_num_threads(),
              "cpu_interop_threads": torch.get_num_interop_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    if torch.device(device).type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(gpu=properties.name, gpu_memory_bytes=properties.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(device)))
    return record


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str = "learned"
    width: int = 128
    policy: str = "teacher"
    reset_every: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"learned", "reservoir", "gru"}:
            raise ValueError("unknown model kind")
        if self.policy not in {"teacher", "online"}:
            raise ValueError("policy must be teacher or online")
        if self.width < 1 or self.reset_every < 0:
            raise ValueError("invalid width or reset interval")


VARIANTS = {variant.name: variant for variant in (
    Variant("c64", width=64), Variant("c128"), Variant("c256", width=256),
    Variant("reset128", reset_every=512), Variant("reservoir128", kind="reservoir"),
    Variant("gru128", kind="gru"), Variant("online128", policy="online"),
)}


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 64
    tbptt: int = 64
    updates: int = 2048
    device: str = "cpu"
    world_device: str = "cpu"
    n_jobs: int = 4
    train_dts: tuple[float, ...] = (0.125, 0.25, 0.5)
    dt_every: int = 512
    duration_range: tuple[float, float] = (1.0, 4.0)
    gap_range: tuple[float, float] = (0.5, 2.0)
    window_range: tuple[float, float] = (1.5, 2.5)
    cancel: float = 0.20
    replace: float = 0.30
    max_step: float = 0.25
    lr: float = 0.001
    weight_decay: float = 0.0
    wait_weight: float = 0.1
    clip: float = 1.0
    replay_capacity: int = 32
    replay_block_ticks: int = 128
    replay_burnin: int = 64
    log_every: int = 32
    deterministic: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_size", "tbptt", "updates", "n_jobs", "dt_every",
                     "replay_capacity", "replay_block_ticks", "log_every"):
            if isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        self.train_dts = tuple(map(float, self.train_dts))
        self.duration_range = tuple(self.duration_range)
        self.gap_range = tuple(self.gap_range)
        self.window_range = tuple(self.window_range)
        if not self.train_dts or any(not math.isfinite(dt) or dt <= 0 for dt in self.train_dts):
            raise ValueError("train_dts must be finite and positive")
        if not 0 <= self.replay_burnin < self.replay_block_ticks:
            raise ValueError("replay burn-in must leave at least one supervised tick")
        if self.replay_block_ticks - self.replay_burnin != self.tbptt:
            raise ValueError("past-data loss must cover the same tick budget as current-data loss")
        if self.lr <= 0 or self.clip <= 0 or self.wait_weight <= 0 or self.weight_decay < 0:
            raise ValueError("invalid learning rate, clipping, class weight or weight decay")


class ExperienceReservoir:
    """Uniform finite reservoir of consecutive, non-overlapping live A blocks."""

    def __init__(self, capacity: int, block_ticks: int, seed: int):
        self.capacity, self.block_ticks = capacity, block_ticks
        self.rng = random.Random(seed)
        self.blocks_seen = 0
        self.blocks: list[dict[str, Any]] = []
        self.partial: dict[str, Any] | None = None

    def add_tick(self, observation: torch.Tensor, target: torch.Tensor,
                 elapsed: float, start_h: torch.Tensor, tick: int, reset: bool) -> None:
        if self.partial is None:
            self.partial = {"start_tick": tick, "start_h": cpu_tree(start_h),
                            "observations": [], "targets": [], "elapsed": [], "resets": []}
        partial = self.partial
        partial["observations"].append(cpu_tree(observation))
        partial["targets"].append(cpu_tree(target))
        partial["elapsed"].append(float(elapsed))
        partial["resets"].append(bool(reset))
        if len(partial["elapsed"]) == self.block_ticks:
            partial["observations"] = torch.stack(partial["observations"])
            partial["targets"] = torch.stack(partial["targets"])
            partial["elapsed"] = tuple(partial["elapsed"])
            partial["resets"] = tuple(partial["resets"])
            partial["end_tick"] = tick + 1
            self.blocks_seen += 1
            if len(self.blocks) < self.capacity:
                self.blocks.append(partial)
            else:
                index = self.rng.randrange(self.blocks_seen)
                if index < self.capacity:
                    self.blocks[index] = partial
            self.partial = None

    def sample(self) -> dict[str, Any]:
        if not self.blocks:
            raise ValueError("past-data replay requires at least one complete A block")
        return self.blocks[self.rng.randrange(len(self.blocks))]

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "block_ticks": self.block_ticks,
                "blocks_seen": self.blocks_seen, "blocks": cpu_tree(self.blocks),
                "partial": cpu_tree(self.partial), "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["capacity"] != self.capacity or state["block_ticks"] != self.block_ticks:
            raise ValueError("experience-reservoir dimensions differ from checkpoint")
        self.blocks_seen = state["blocks_seen"]
        self.blocks = cpu_tree(state["blocks"])
        self.partial = cpu_tree(state["partial"])
        self.rng.setstate(state["rng"])

    def summary(self) -> dict[str, Any]:
        return {"capacity_blocks": self.capacity, "block_ticks": self.block_ticks,
                "complete_blocks_seen": self.blocks_seen, "stored_blocks": len(self.blocks),
                "partial_ticks": 0 if self.partial is None else len(self.partial["elapsed"]),
                "stored_tick_ranges": [[block["start_tick"], block["end_tick"]]
                                       for block in self.blocks],
                "selection": "uniform reservoir over completed non-overlapping live A blocks",
                "source": "live observations and own-world causal oracle labels; no evaluation data"}


class ActionStats:
    def __init__(self, n_actions: int, device: str | torch.device, wait_weight: float):
        self.n_actions, self.wait_weight = n_actions, wait_weight
        self.confusion = torch.zeros(n_actions, n_actions, dtype=torch.long, device=device)
        self.nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.weighted_sum = self.nll_sum.clone()
        self.weight_sum = self.nll_sum.clone()

    @torch.no_grad()
    def add(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        labels, prediction = target.reshape(-1), logits.detach().argmax(-1).reshape(-1)
        self.confusion += torch.bincount(labels * self.n_actions + prediction,
                                        minlength=self.n_actions**2).view(self.n_actions, self.n_actions)
        loss = F.cross_entropy(logits.detach().reshape(-1, self.n_actions), labels,
                               reduction="none")
        weights = torch.where(labels == 0, self.wait_weight, 1.0)
        self.nll_sum += loss.double().sum()
        self.weighted_sum += (loss * weights).double().sum()
        self.weight_sum += weights.double().sum()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: cpu_tree(getattr(self, name))
                for name in ("confusion", "nll_sum", "weighted_sum", "weight_sum")}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, value in state.items():
            getattr(self, name).copy_(value.to(self.confusion.device))

    def result(self) -> dict[str, Any]:
        matrix = self.confusion.cpu()
        counts, correct = matrix.sum(1), matrix.diag()
        total = int(counts.sum())

        def accuracy(start: int, end: int) -> dict[str, Any]:
            count = int(counts[start:end].sum())
            return {"count": count,
                    "accuracy": int(correct[start:end].sum()) / count if count else None}

        weight_sum = float(self.weight_sum)
        return {"count": total, "accuracy": int(correct.sum()) / total if total else None,
                "nll": float(self.nll_sum) / total if total else None,
                "weighted_ce": float(self.weighted_sum) / weight_sum if weight_sum else None,
                "wait": accuracy(0, 1), "begin": accuracy(1, 3),
                "finish": accuracy(3, self.n_actions), "nonwait": accuracy(1, self.n_actions),
                "target_counts": counts.tolist(), "prediction_counts": matrix.sum(0).tolist(),
                "confusion_target_by_prediction": matrix.tolist()}


class Trainer:
    """A complete continuation, serializable after any train_update()."""

    def __init__(self, variant: str | Variant, config: TrainConfig):
        self.variant = VARIANTS[variant] if isinstance(variant, str) else variant
        self.config = deepcopy(config)
        self.device = torch.device(config.device)
        self.initialization_seed = seed_for(config.seed, "initialization", self.variant.width)
        self.data_seed = seed_for(config.seed, "live-data")
        seed_all(self.initialization_seed, config.deterministic)
        self.model = AgentModel(input_size=config.n_jobs * 24, n_jobs=config.n_jobs,
                                n_actions=11, width=self.variant.width, kind=self.variant.kind,
                                max_step=config.max_step).to(self.device)
        self.optimizer = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad),
                                          lr=config.lr, weight_decay=config.weight_decay)
        self.world = OnlineWorld(seed=self.data_seed, batch_size=config.batch_size,
                                 n_jobs=config.n_jobs, dt=config.train_dts[0],
                                 duration_range=config.duration_range, gap_range=config.gap_range,
                                 window_range=config.window_range, cancel=config.cancel,
                                 replace=config.replace, device=config.world_device)
        self.h = self.model.initial_state(config.batch_size, self.device)
        self.replay = ExperienceReservoir(config.replay_capacity, config.replay_block_ticks,
                                          seed_for(config.seed, "A-reservoir"))
        self.stats = ActionStats(self.model.n_actions, self.device, config.wait_weight)
        self.loss_weights = torch.ones(self.model.n_actions, device=self.device)
        self.loss_weights[0] = config.wait_weight
        self.updates = self.ticks = self.gradient_updates = self.hidden_resets = 0
        self.stage, self.adaptation_mode = "A", None
        self.phase_start_update = self.phase_start_tick = self.phase_start_gradient = 0
        self.adaptation_origin: dict[str, Any] | None = None
        self.dt_index = 0
        self.dt_history = [{"tick": 0, "dt": config.train_dts[0]}]
        self.initial_parameter_hash = tensor_digest(self.model.state_dict())
        zero_hash = hashlib.sha256(f"{FORMAT_VERSION}/{self.data_seed}".encode()).hexdigest()
        self.action_hash = self.data_hash = zero_hash
        self.trace: list[dict[str, Any]] = []
        self.curve: list[dict[str, Any]] = []
        self.sources = source_snapshots()
        self.source_hashes = source_hashes(self.sources)
        self.runtime = runtime_record(self.device)
        self.runtime_history = [deepcopy(self.runtime)]
        self.seconds = 0.0
        self._at_boundary = True

    @property
    def stage_updates(self) -> int:
        return self.updates - self.phase_start_update

    @property
    def policy(self) -> str:
        return self.variant.policy if self.stage == "A" else "online"

    def _before_observe(self) -> tuple[int, bool]:
        tick = self.ticks
        index = (tick // self.config.dt_every) % len(self.config.train_dts)
        if index != self.dt_index:
            self.dt_index = index
            dt = self.config.train_dts[index]
            self.world.set_dt(dt)
            self.dt_history.append({"tick": tick, "dt": dt})
        reset = bool(self.variant.reset_every and tick and tick % self.variant.reset_every == 0)
        return tick, reset

    def _forward_sequence(self, observations: torch.Tensor, start_h: torch.Tensor,
                          elapsed: tuple[float, ...], resets: tuple[bool, ...],
                          burnin: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        h, logits = start_h, []
        for index, (observation, dt, reset) in enumerate(zip(observations, elapsed, resets)):
            if reset:
                h = self.model.initial_state(self.config.batch_size, self.device)
            if index < burnin:
                with torch.no_grad():
                    _, h = self.model.step(observation, h, dt)
                h = h.detach()
            else:
                output, h = self.model.step(observation, h, dt)
                logits.append(output)
        return torch.stack(logits), h

    def _live_teacher_chunk(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        observations, targets, elapsed, resets = [], [], [], []
        start_tick = self.ticks
        for _ in range(self.config.tbptt):
            _, reset = self._before_observe()
            observation = self.world.observe()
            target = self.world.oracle("midpoint")
            observations.append(cpu_tree(observation))
            targets.append(cpu_tree(target))
            elapsed.append(float(self.world.elapsed))
            resets.append(reset)
            self.world.act(target)
            self.ticks += 1
        x_cpu, y_cpu = torch.stack(observations), torch.stack(targets)
        x, y = x_cpu.to(self.device), y_cpu.to(self.device)
        logits = []
        for index, (dt, reset) in enumerate(zip(elapsed, resets)):
            if reset:
                self.h = self.model.initial_state(self.config.batch_size, self.device)
                self.hidden_resets += 1
            before_h = self.h
            output, self.h = self.model.step(x[index], self.h, dt)
            logits.append(output)
            self.replay.add_tick(x_cpu[index], y_cpu[index], dt, before_h,
                                 start_tick + index, reset)
        return torch.stack(logits), y, {
            "observations": x_cpu, "targets": y_cpu, "actions": y_cpu.clone(),
            "elapsed": tuple(elapsed), "resets": tuple(resets), "start_tick": start_tick}

    def _live_online_chunk(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        observations, targets, actions, elapsed, resets, logits = [], [], [], [], [], []
        start_tick = self.ticks
        for _ in range(self.config.tbptt):
            tick, reset = self._before_observe()
            observation = cpu_tree(self.world.observe())
            # This label belongs to the learner's current world before its action.
            target = cpu_tree(self.world.oracle("midpoint"))
            dt = float(self.world.elapsed)
            if reset:
                self.h = self.model.initial_state(self.config.batch_size, self.device)
                self.hidden_resets += 1
            before_h = self.h
            output, self.h = self.model.step(observation.to(self.device), self.h, dt)
            action = output.detach().argmax(-1).cpu()
            self.world.act(action)
            observations.append(observation)
            targets.append(target)
            actions.append(action)
            elapsed.append(dt)
            resets.append(reset)
            logits.append(output)
            if self.stage == "A":
                self.replay.add_tick(observation, target, dt, before_h, tick, reset)
            self.ticks += 1
        return torch.stack(logits), torch.stack(targets).to(self.device), {
            "observations": torch.stack(observations), "targets": torch.stack(targets),
            "actions": torch.stack(actions), "elapsed": tuple(elapsed),
            "resets": tuple(resets), "start_tick": start_tick}

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, self.model.n_actions), targets.reshape(-1),
                               weight=self.loss_weights)

    def _optimize(self, loss: torch.Tensor) -> float:
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at update {self.updates}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip,
                                                   error_if_nonfinite=True)
        self.optimizer.step()
        self.gradient_updates += 1
        return float(grad_norm)

    def train_update(self) -> dict[str, Any]:
        if not self._at_boundary:
            raise RuntimeError("trainer is not at a complete update boundary")
        self._at_boundary = False
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        start_h = self.h.detach().clone()
        frozen = self.stage == "B" and self.adaptation_mode == "frozen"
        context = torch.no_grad() if frozen else nullcontext()
        with context:
            if self.policy == "teacher":
                logits, target, chunk = self._live_teacher_chunk()
            else:
                logits, target, chunk = self._live_online_chunk()
            loss = self._loss(logits, target)
        self.stats.add(logits, target)
        gradient_before = self.gradient_updates
        grad_norm = None if frozen else self._optimize(loss)
        # The live hidden value was produced by the live trajectory above.
        # Extra optimization passes below have separate local hidden variables.
        self.h = self.h.detach()
        extra_loss, replay_start_tick = None, None
        if self.stage == "B" and self.adaptation_mode in {"current_repeat", "replay"}:
            self.optimizer.zero_grad(set_to_none=True)
            if self.adaptation_mode == "current_repeat":
                extra_logits, _ = self._forward_sequence(
                    chunk["observations"].to(self.device), start_h,
                    chunk["elapsed"], chunk["resets"])
                extra_target = target
            else:
                block = self.replay.sample()
                replay_start_tick = block["start_tick"]
                extra_logits, _ = self._forward_sequence(
                    block["observations"].to(self.device), block["start_h"].to(self.device),
                    block["elapsed"], block["resets"], burnin=self.config.replay_burnin)
                extra_target = block["targets"][self.config.replay_burnin:].to(self.device)
            extra = self._loss(extra_logits, extra_target)
            self._optimize(extra)
            extra_loss = float(extra.detach())
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        action_chunk_hash = tensor_digest({"actions": chunk["actions"]})
        data_chunk_hash = state_digest({key: chunk[key] for key in
                                       ("observations", "targets", "elapsed", "resets", "start_tick")})
        self.action_hash = hashlib.sha256(bytes.fromhex(self.action_hash)
                                          + bytes.fromhex(action_chunk_hash)).hexdigest()
        self.data_hash = hashlib.sha256(bytes.fromhex(self.data_hash)
                                        + bytes.fromhex(data_chunk_hash)).hexdigest()
        point = {"update": self.updates, "stage_update": self.stage_updates, "stage": self.stage,
                 "tick": self.ticks, "world_tick": self.world.tick, "now": float(self.world.now),
                 "dt": float(self.world.dt), "elapsed_min": min(chunk["elapsed"]),
                 "elapsed_max": max(chunk["elapsed"]), "loss": float(loss.detach()),
                 "extra_loss": extra_loss, "grad_norm": grad_norm,
                 "gradient_updates": self.gradient_updates,
                 "gradient_updates_this_chunk": self.gradient_updates - gradient_before,
                 "hidden_resets": self.hidden_resets, "replay_start_tick": replay_start_tick,
                 "action_chunk_hash": action_chunk_hash, "data_chunk_hash": data_chunk_hash,
                 "action_hash": self.action_hash, "data_hash": self.data_hash,
                 "event_hash": self.world.event_hash,
                 "actual_action_counts": torch.bincount(chunk["actions"].reshape(-1),
                                                         minlength=self.model.n_actions).tolist(),
                 "target_action_counts": torch.bincount(chunk["targets"].reshape(-1),
                                                         minlength=self.model.n_actions).tolist()}
        self.trace.append(deepcopy(point))
        self._at_boundary = True
        synchronize(self.device)
        self.seconds += time.perf_counter() - started
        if self.stage_updates == 1 or self.stage_updates % self.config.log_every == 0:
            point = {**point, "seconds": self.seconds, "world_metrics": self.world.metrics(),
                     "agreement": self.stats.result(), "replay": self.replay.summary()}
            self.curve.append(deepcopy(point))
        return point

    def run_updates(self, count: int) -> list[dict[str, Any]]:
        if count < 0:
            raise ValueError("update count must be nonnegative")
        return [self.train_update() for _ in range(count)]

    def begin_adaptation(self, mode: str, duration_range: tuple[float, float] = (4.0, 8.0)) -> None:
        if not self._at_boundary or self.stage != "A":
            raise ValueError("adaptation must branch from a completed A update boundary")
        if self.policy != "online":
            raise ValueError("adaptation branches must start from an own-action A controller")
        if mode not in ADAPTATION_MODES:
            raise ValueError("unknown adaptation mode")
        if mode == "replay" and not self.replay.blocks:
            raise ValueError("no complete A blocks are available for replay")
        self.adaptation_origin = {
            "model_hash": tensor_digest(self.model.state_dict()),
            "optimizer_hash": state_digest(self.optimizer.state_dict()),
            "hidden_hash": tensor_digest({"h": self.h}),
            "world_hash": state_digest(self.world.state_dict()),
            "replay_hash": state_digest(self.replay.state_dict()),
            "action_hash": self.action_hash, "data_hash": self.data_hash,
            "tick": self.ticks, "update": self.updates,
            "duration_range_before": list(self.world.regime.duration_range),
            "duration_range_after": list(duration_range)}
        self.stage, self.adaptation_mode = "B", mode
        self.phase_start_update, self.phase_start_tick = self.updates, self.ticks
        self.phase_start_gradient = self.gradient_updates
        self.world.set_regime(duration_range=duration_range, name="adaptation")
        self.stats = ActionStats(self.model.n_actions, self.device, self.config.wait_weight)

    def semantic_state(self) -> dict[str, Any]:
        """All continuation data; excludes hardware, sources and elapsed wall time."""
        if not self._at_boundary:
            raise RuntimeError("checkpoint only after a complete TBPTT/optimizer update")
        return {"format": FORMAT_VERSION, "config": asdict(self.config),
                "variant": asdict(self.variant), "model": cpu_tree(self.model.state_dict()),
                "optimizer": cpu_tree(self.optimizer.state_dict()), "hidden": cpu_tree(self.h),
                "world": cpu_tree(self.world.state_dict()), "replay": self.replay.state_dict(),
                "rng": rng_state(), "stats": self.stats.state_dict(),
                "updates": self.updates, "ticks": self.ticks, "gradient_updates": self.gradient_updates,
                "hidden_resets": self.hidden_resets, "stage": self.stage,
                "adaptation_mode": self.adaptation_mode,
                "phase_start_update": self.phase_start_update,
                "phase_start_tick": self.phase_start_tick,
                "phase_start_gradient": self.phase_start_gradient,
                "adaptation_origin": deepcopy(self.adaptation_origin),
                "dt_index": self.dt_index, "dt_history": deepcopy(self.dt_history),
                "initialization_seed": self.initialization_seed, "data_seed": self.data_seed,
                "initial_parameter_hash": self.initial_parameter_hash,
                "action_hash": self.action_hash, "data_hash": self.data_hash,
                "trace": deepcopy(self.trace)}

    def state_dict(self) -> dict[str, Any]:
        return {**self.semantic_state(), "sources": deepcopy(self.sources),
                "source_hashes": deepcopy(self.source_hashes), "runtime": deepcopy(self.runtime),
                "runtime_history": deepcopy(self.runtime_history), "seconds": self.seconds,
                "curve": deepcopy(self.curve)}

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None,
                        strict_sources: bool = True) -> "Trainer":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported continuity checkpoint format")
        if strict_sources:
            current = source_hashes()
            for name in ("world.py", "model.py", "run.py"):
                if state["source_hashes"].get(name) != current.get(name):
                    raise ValueError(f"checkpoint source mismatch for {name}; use its saved source snapshot")
        config = TrainConfig(**state["config"])
        if device is not None:
            config.device = device
        trainer = cls(Variant(**state["variant"]), config)
        trainer.model.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.h = state["hidden"].to(trainer.device).detach()
        trainer.world.load_state_dict(state["world"])
        trainer.replay.load_state_dict(state["replay"])
        trainer.stats.load_state_dict(state["stats"])
        for name in ("updates", "ticks", "gradient_updates", "hidden_resets", "stage",
                     "adaptation_mode", "phase_start_update", "phase_start_tick",
                     "phase_start_gradient", "adaptation_origin", "dt_index", "dt_history",
                     "initialization_seed", "data_seed", "initial_parameter_hash",
                     "action_hash", "data_hash", "trace", "seconds", "curve", "sources", "source_hashes"):
            setattr(trainer, name, deepcopy(state[name]))
        trainer.runtime_history = deepcopy(state["runtime_history"]) + [deepcopy(trainer.runtime)]
        restore_rng_state(state["rng"])
        return trainer

    def report(self) -> dict[str, Any]:
        return {"format": FORMAT_VERSION, "variant": asdict(self.variant),
                "config": asdict(self.config), "model": self.model.parameter_report(),
                "stage": self.stage, "adaptation_mode": self.adaptation_mode,
                "updates": self.updates, "stage_updates": self.stage_updates,
                "ticks": self.ticks, "stage_ticks": self.ticks - self.phase_start_tick,
                "batch_observations": self.ticks * self.config.batch_size,
                "action_labels": self.ticks * self.config.batch_size * self.config.n_jobs,
                "gradient_updates": self.gradient_updates,
                "stage_gradient_updates": self.gradient_updates - self.phase_start_gradient,
                "hidden_resets": self.hidden_resets, "world_resets": 0,
                "hidden_hash": tensor_digest({"h": self.h}),
                "world_hash": state_digest(self.world.state_dict()),
                "initial_parameter_hash": self.initial_parameter_hash,
                "final_parameter_hash": tensor_digest(self.model.state_dict()),
                "initialization_seed": self.initialization_seed, "data_seed": self.data_seed,
                "action_hash": self.action_hash, "data_hash": self.data_hash,
                "event_hash": self.world.event_hash, "dt_history": deepcopy(self.dt_history),
                "world_metrics": self.world.metrics(), "action_metrics": self.stats.result(),
                "replay": self.replay.summary(), "adaptation_origin": self.adaptation_origin,
                "seconds": self.seconds, "curve": deepcopy(self.curve),
                "trace": deepcopy(self.trace), "runtime": self.runtime,
                "runtime_history": self.runtime_history, "source_hashes": self.source_hashes,
                "continuity": "one live hidden state and world; detach only after each live TBPTT chunk",
                "learning": "Adam weighted causal-oracle cross entropy; no protected units or freezing gates",
                "checkpoint_boundary": "after a complete TBPTT chunk and all its optimizer updates",
                "budget_note": "frozen=0, current=1, current_repeat=2, replay=2 updates per live chunk; replay also uses burn-in"}


def write_run_artifacts(trainer: Trainer, folder: Path) -> None:
    trainer.save_checkpoint(folder / "checkpoint.pt")
    atomic_json(folder / "report.json", trainer.report())
    snapshot_folder = folder / "sources"
    snapshot_folder.mkdir(parents=True, exist_ok=True)
    for name, source in trainer.sources.items():
        (snapshot_folder / name).write_text(source, encoding="utf-8")


def run_to_target(trainer: Trainer, target_stage_updates: int, folder: Path,
                  checkpoint_every: int = 0, stop_after_updates: int | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    completed = 0
    log_path = folder / "training.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        while trainer.stage_updates < target_stage_updates:
            if stop_after_updates is not None and completed >= stop_after_updates:
                break
            point = trainer.train_update()
            completed += 1
            is_last = trainer.stage_updates == target_stage_updates
            should_log = (trainer.stage_updates == 1 or
                          trainer.stage_updates % trainer.config.log_every == 0 or is_last)
            if should_log:
                if "world_metrics" not in point:
                    point = {**point, "seconds": trainer.seconds,
                             "world_metrics": trainer.world.metrics(),
                             "agreement": trainer.stats.result(), "replay": trainer.replay.summary()}
                    trainer.curve.append(deepcopy(point))
                entry = {"event": "train", "variant": trainer.variant.name,
                         "mode": trainer.adaptation_mode, **point}
                encoded = json.dumps(entry, ensure_ascii=False, allow_nan=False)
                log.write(encoded + "\n")
                log.flush()
                print(encoded, flush=True)
            if checkpoint_every and trainer.stage_updates % checkpoint_every == 0:
                trainer.save_checkpoint(folder / f"checkpoint_{trainer.stage_updates:06d}.pt")
                trainer.save_checkpoint(folder / "checkpoint.pt")
    write_run_artifacts(trainer, folder)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--updates", type=int, default=2048, help="total Stage A TBPTT updates")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tbptt", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--world-device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--train-dts", nargs="+", type=float, default=[0.125, 0.25, 0.5])
    parser.add_argument("--dt-every", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--wait-weight", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=0.25)
    parser.add_argument("--replay-capacity", type=int, default=32)
    parser.add_argument("--replay-block-ticks", type=int, default=128)
    parser.add_argument("--replay-burnin", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="retain complete checkpoints every N stage updates, plus the final checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="resume one complete checkpoint to --updates or --adapt-updates")
    parser.add_argument("--stop-after-updates", type=int,
                        help="stop this invocation after N additional complete updates, preserving continuation")
    parser.add_argument("--adapt-from", type=Path, help="branch Stage B modes from the same online Stage A checkpoint")
    parser.add_argument("--adapt-modes", nargs="+", choices=ADAPTATION_MODES, default=list(ADAPTATION_MODES))
    parser.add_argument("--adapt-updates", type=int, default=512)
    parser.add_argument("--adapt-durations", nargs=2, type=float, default=[4.0, 8.0])
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.adapt_from:
        parser.error("--resume and --adapt-from are mutually exclusive")
    if args.threads < 1 or args.checkpoint_every < 0:
        parser.error("threads must be positive and checkpoint interval nonnegative")
    if args.stop_after_updates is not None and args.stop_after_updates < 0:
        parser.error("stop-after-updates must be nonnegative")
    torch.set_num_threads(args.threads)
    if args.resume:
        trainer = Trainer.from_checkpoint(args.resume, device=args.device)
        target = args.updates if trainer.stage == "A" else args.adapt_updates
        folder = args.output or args.resume.parent
        run_to_target(trainer, target, folder, args.checkpoint_every, args.stop_after_updates)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or ROOT / "data" / "continuity" / stamp
    if args.adapt_from:
        for mode in args.adapt_modes:
            trainer = Trainer.from_checkpoint(args.adapt_from, device=args.device)
            trainer.begin_adaptation(mode, tuple(args.adapt_durations))
            folder = output / f"seed_{trainer.config.seed:03d}" / f"B_{mode}"
            if folder.exists():
                raise FileExistsError(f"run folder already exists: {folder}; resume its checkpoint explicitly")
            run_to_target(trainer, args.adapt_updates, folder,
                          args.checkpoint_every, args.stop_after_updates)
        return
    for seed in args.seeds or [args.seed]:
        for name in args.variants:
            config = TrainConfig(seed=seed, batch_size=args.batch, tbptt=args.tbptt,
                                 updates=args.updates, device=args.device, world_device=args.world_device,
                                 train_dts=tuple(args.train_dts), dt_every=args.dt_every,
                                 lr=args.lr, weight_decay=args.weight_decay,
                                 wait_weight=args.wait_weight, clip=args.clip, max_step=args.max_step,
                                 replay_capacity=args.replay_capacity,
                                 replay_block_ticks=args.replay_block_ticks, replay_burnin=args.replay_burnin,
                                 log_every=args.log_every)
            folder = output / f"seed_{seed:03d}" / name
            if folder.exists():
                raise FileExistsError(f"run folder already exists: {folder}; resume its checkpoint explicitly")
            trainer = Trainer(name, config)
            run_to_target(trainer, args.updates, folder,
                          args.checkpoint_every, args.stop_after_updates)


if __name__ == "__main__":
    main()
