"""Native, persistent contextual learning from one random birth.

The live controller acts with all eleven logits.  Each optimizer boundary uses
one live TBPTT gradient and one equally weighted auxiliary gradient, both under
the same pre-update parameters.  Auxiliary history has a detached warmup and a
one-chunk differentiable suffix; it never replaces the live state or world.
The old numerical model and structural heuristics are imported unchanged.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from experiments.continuity import run as continuity
from experiments.continuity.model import AgentModel
from experiments.factors.structure import FactorConfig, FactorOperator
from experiments.rewiring.plasticity import RewireConfig, RewireOperator
from experiments.lifelong.replay import (
    HistoryReservoir, block_hash, combine_blocks, copied_block, validate_block,
)
from experiments.lifelong.schedule import CHECKPOINT_TICKS, Schedule, ScheduleConfig
from experiments.lifelong.world import ContextWorld, INPUT_SIZE, TRAINING_COVERAGE


FORMAT_VERSION = "lifelong-native-trainer-v1"
ARCHITECTURES = ("fixed_shrink", "grow_shrink", "open_shrink")
LEARNING_METHODS = ("recent_repeat", "history_replay")
RUNTIME_SOURCES = ("run.py", "world.py", "schedule.py", "replay.py")
DEPENDENCIES = (
    "continuity/run.py", "continuity/model.py", "continuity/world.py",
    "factors/structure.py", "rewiring/plasticity.py",
)
DATA_HASH_FIELDS = ("observations", "targets", "elapsed", "start_tick", "end_tick", "family_context")
_SEMANTIC_FIELDS = {
    "format", "config", "model_config", "model", "optimizer", "optimizer_parameter_names",
    "hidden", "hidden_time", "world", "schedule", "schedule_history", "replay", "operator",
    "rng", "stats", "auxiliary_stats", "work", "ticks", "updates", "gradient_updates",
    "hidden_resets", "birth", "action_hash", "data_hash", "trace", "last_chunk", "last_auxiliary",
    "last_live_family_tick", "last_aux_family_tick", "live_supervised_family_positions",
}
_RECORD_FIELDS = {
    "sources", "source_hashes", "dependencies", "dependency_hashes", "runtime", "runtime_history",
    "seconds", "state_hashes",
}


def _integer(name: str, value: Any, *, lower: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return value


def _number(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"invalid {name}")
    return float(value)


@dataclass(frozen=True)
class NativeConfig:
    seed: int = 0
    architecture: str = "fixed_shrink"
    schedule: str = "blocked"
    learning: str = "recent_repeat"
    coverage: str = "narrow"
    width: int = 128
    batch_size: int = 32
    chunk_ticks: int = 128
    device: str = "cpu"
    world_device: str = "cpu"
    n_jobs: int = 4
    dt: float = .25
    max_step: float = .25
    lr: float = .001
    wait_weight: float = .1
    clip: float = 1.
    shrink_step: float = 2e-4
    initial_fan_in: int = 32
    structural_window_updates: int = 8
    replay_capacity: int = 32
    main_ticks: int = 131072
    macrocycle_ticks: int = 32768
    block_ticks: int = 8192
    withdrawal_ticks: int = 32768
    return_ticks: int = 8192
    window_range: tuple[float, float] = (1.5, 2.5)
    cancel: float = .20
    replace: float = .30
    deterministic: bool = True

    def __post_init__(self):
        _integer("seed", self.seed)
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unknown native architecture")
        if self.learning not in LEARNING_METHODS:
            raise ValueError("unknown native learning method")
        if self.coverage not in TRAINING_COVERAGE:
            raise ValueError("unknown time coverage")
        for name in ("width", "batch_size", "chunk_ticks", "initial_fan_in", "structural_window_updates"):
            _integer(name, getattr(self, name), lower=1)
        _integer("n_jobs", self.n_jobs, lower=1)
        _integer("replay_capacity", self.replay_capacity, lower=1)
        if self.n_jobs != 4:
            raise ValueError("the contextual native task has four jobs and 112 inputs")
        for name in ("dt", "max_step", "lr", "clip"):
            _number(name, getattr(self, name), positive=True)
        for name in ("wait_weight", "shrink_step", "cancel", "replace"):
            _number(name, getattr(self, name))
        if self.wait_weight <= 0 or self.cancel + self.replace > 1:
            raise ValueError("positive WAIT weight and valid modification probabilities required")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        bounds = tuple(float(value) for value in self.window_range)
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("invalid public finish window")
        object.__setattr__(self, "window_range", bounds)
        self.schedule_config()
        torch.device(self.device)
        torch.device(self.world_device)

    def schedule_config(self) -> ScheduleConfig:
        return ScheduleConfig(mode=self.schedule, seed=self.seed, **{name: getattr(self, name) for name in (
            "main_ticks", "withdrawal_ticks", "return_ticks", "macrocycle_ticks", "block_ticks", "chunk_ticks")})

    @property
    def total_ticks(self):
        return self.main_ticks + self.withdrawal_ticks + self.return_ticks

    @property
    def total_updates(self):
        return self.total_ticks // self.chunk_ticks

    @property
    def effective_fan_in(self):
        return min(self.initial_fan_in, self.width)


# An explicit native name is primary; these aliases ease generic launch code.
TrainConfig = LifelongConfig = NativeConfig


def source_snapshots() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: (root / name).read_text(encoding="utf-8") for name in RUNTIME_SOURCES}


def source_hashes(sources=None):
    return {name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in (source_snapshots() if sources is None else sources).items()}


def dependency_snapshots():
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text(encoding="utf-8") for name in DEPENDENCIES}


def _model(config):
    return AgentModel(input_size=INPUT_SIZE, n_jobs=config.n_jobs, n_actions=11,
                      width=config.width, kind="learned", max_step=config.max_step)


def _optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.)


def _configure_determinism(config):
    """Restore numerical runtime switches without reseeding any generator."""
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _validate_rng(state):
    if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("incomplete native global RNG state")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        generator = torch.Generator(device="cpu")
        generator.set_state(state["torch_cpu"].cpu())
        for value in state["torch_cuda"]:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("invalid CUDA RNG state")
    except (TypeError, KeyError, RuntimeError, AttributeError) as error:
        raise ValueError("invalid native global RNG state") from error


def _validate_optimizer_state(model, optimizer, state, updates):
    expected = optimizer.state_dict()
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError("incomplete native Adam state")
    if not isinstance(state["param_groups"], list) or len(state["param_groups"]) != 1:
        raise ValueError("native Adam must have exactly one parameter group")
    if state["param_groups"][0] != expected["param_groups"][0]:
        raise ValueError("native Adam parameter order/hyperparameters mismatch")
    slots = state["state"]
    if not isinstance(slots, Mapping):
        raise ValueError("invalid native Adam slots")
    parameters = list(model.parameters())
    expected_keys = set(range(len(parameters))) if updates else set()
    if set(slots) != expected_keys:
        raise ValueError("native Adam moments do not match its age")
    for index, parameter in enumerate(parameters):
        if not updates:
            break
        slot = slots[index]
        if not isinstance(slot, Mapping) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incomplete native Adam parameter state")
        step = slot["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != updates:
            raise ValueError("native Adam global age mismatch")
        for name in ("exp_avg", "exp_avg_sq"):
            value = slot[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())
                    or (name == "exp_avg_sq" and bool(value.lt(0).any()))):
                raise ValueError(f"invalid native Adam {name}")


def _empty_auxiliary_stats():
    return {"calls": 0, "startup": 0, "fallback_recent": 0, "recent_repeat": 0, "history_replay": 0,
            "source_family_counts": {family: 0 for family in range(4)},
            "suffix_family_positions": {family: 0 for family in range(4)},
            "source_age_ticks_sum": 0, "source_age_ticks_max": 0,
            "source_ticks": 0, "warmup_ticks": 0, "loss_ticks": 0}


def _empty_work():
    return {name: 0 for name in (
        "live_forward_ticks", "aux_forward_ticks", "aux_warmup_ticks", "live_loss_ticks", "aux_loss_ticks",
        "live_forward_actor_ticks", "aux_forward_actor_ticks", "aux_warmup_actor_ticks",
        "live_loss_actor_ticks", "aux_loss_actor_ticks", "live_logit_positions", "aux_logit_positions",
        "live_loss_positions", "aux_loss_positions", "backward_calls", "bp_graph_ticks",
        "bp_graph_actor_ticks", "optimizer_steps", "real_ticks_admitted",
    )}


class NativeTrainer:
    """A full continuation born from random weights, never from a legacy B run."""

    def __init__(self, config: NativeConfig):
        if not isinstance(config, NativeConfig):
            raise ValueError("NativeTrainer requires NativeConfig")
        self.config = deepcopy(config)
        self.device = torch.device(config.device)
        initialization_seed = continuity.seed_for(config.seed, "lifelong-initialization", config.width)
        mask_seed = continuity.seed_for(config.seed, "lifelong-initial-mask", config.width)
        data_seed = continuity.seed_for(config.seed, "lifelong-live-data")
        continuity.seed_all(initialization_seed, config.deterministic)
        self.model = _model(config)
        dense_hash = continuity.state_digest(self.model.recurrent)
        generator = torch.Generator(device="cpu").manual_seed(mask_seed)
        initial_mask = torch.zeros(config.width, config.width, dtype=torch.bool)
        for row in range(config.width):
            initial_mask[row, torch.randperm(config.width, generator=generator)[:config.effective_fan_in]] = True
        scale = math.sqrt(config.width / config.effective_fan_in)
        with torch.no_grad():
            self.model.recurrent.mul_(initial_mask).mul_(scale)
        self.model.to(self.device)
        self.optimizer = _optimizer(self.model, config)
        inherited = RewireOperator(self.model.recurrent, RewireConfig(
            arm="learned" if config.architecture == "grow_shrink" else "retire",
            shrink_step=config.shrink_step, window_updates=config.structural_window_updates,
            structural_seed=continuity.seed_for(config.seed, "lifelong-structure")))
        mode = {"fixed_shrink": "fixed_shrink", "grow_shrink": "grow", "open_shrink": "dense_shrink"}
        self.operator = FactorOperator(self.model.recurrent,
            FactorConfig(mode=mode[config.architecture], idle_lambda=0., shrink_step=config.shrink_step),
            inherited_operator=inherited)
        self.schedule = Schedule(config.schedule_config())
        policy = self.schedule.mode_for_tick(0)
        self.world = ContextWorld(seed=data_seed, batch_size=config.batch_size,
            n_jobs=config.n_jobs, dt=config.dt, **TRAINING_COVERAGE[config.coverage],
            window_range=config.window_range, cancel=config.cancel, replace=config.replace,
            device=config.world_device, family_allowed=policy["future_family_allowed"],
            regime_name=f"lifelong-{config.coverage}")
        self.schedule_history = [{"tick": 0, **deepcopy(policy)}]
        self.h = self.model.initial_state(config.batch_size, self.device)
        self.hidden_time = 0.
        self.replay = HistoryReservoir(config.replay_capacity, 2 * config.chunk_ticks,
                                      continuity.seed_for(config.seed, "lifelong-history"))
        self.stats = continuity.ActionStats(11, self.device, config.wait_weight)
        self.loss_weights = torch.ones(11, device=self.device)
        self.loss_weights[0] = config.wait_weight
        self.auxiliary_stats, self.work = _empty_auxiliary_stats(), _empty_work()
        self.last_live_family_tick, self.last_aux_family_tick = [-1] * 4, [-1] * 4
        self.live_supervised_family_positions = [0] * 4
        self.ticks = self.updates = self.gradient_updates = self.hidden_resets = 0
        self.birth = {"source": "from_scratch", "initialization_seed": initialization_seed,
            "data_seed": data_seed, "mask_seed": mask_seed,
            "initial_fan_in": config.effective_fan_in, "initial_recurrent_rescale": scale,
            "dense_gaussian_recurrent_hash": dense_hash,
            "initial_mask_hash": continuity.state_digest(initial_mask),
            "mask_generator_state": generator.get_state().clone(),
            "initial_parameter_hash": continuity.state_digest(self.model.state_dict()),
            "initial_optimizer_hash": continuity.state_digest(self.optimizer.state_dict())}
        zero_hash = hashlib.sha256(f"{FORMAT_VERSION}/{data_seed}".encode()).hexdigest()
        self.action_hash = self.data_hash = zero_hash
        self.trace, self.last_chunk, self.last_auxiliary = [], None, None
        self.sources, self.dependencies = source_snapshots(), dependency_snapshots()
        self.source_hashes, self.dependency_hashes = source_hashes(self.sources), source_hashes(self.dependencies)
        self.runtime = continuity.runtime_record(self.device)
        self.runtime_history = [deepcopy(self.runtime)]
        self.seconds, self._at_boundary = 0., True

    @property
    def initial_parameter_hash(self):
        return self.birth["initial_parameter_hash"]

    def _loss(self, logits, targets):
        return F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1), weight=self.loss_weights)

    def _before_observe(self):
        policy = self.schedule.mode_for_tick(self.ticks)
        previous = self.schedule_history[-1]
        if any(policy[key] != previous[key] for key in policy):
            if tuple(self.world.family_allowed) != policy["future_family_allowed"]:
                self.world.set_family_policy(policy["future_family_allowed"])
            self.schedule_history.append({"tick": self.ticks, **deepcopy(policy)})
        return policy

    def _gradients(self, loss, graph_ticks):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite native learning loss")
        parameters = dict(self.model.named_parameters())
        values = torch.autograd.grad(loss, tuple(parameters.values()))
        result = {name: value.detach().clone() for name, value in zip(parameters, values)}
        self.work["backward_calls"] += 1
        self.work["bp_graph_ticks"] += graph_ticks
        self.work["bp_graph_actor_ticks"] += graph_ticks * self.config.batch_size
        return result

    def _live_chunk(self):
        observations, targets, actions, elapsed, outputs, contexts = [], [], [], [], [], []
        start_tick, start_h = self.ticks, continuity.cpu_tree(self.h)
        before = self.world.family_exposures()["commands"]
        for _ in range(self.config.chunk_ticks):
            self._before_observe()
            observation = continuity.cpu_tree(self.world.observe())
            target = continuity.cpu_tree(self.world.oracle("midpoint"))
            family_context = continuity.cpu_tree(self.world.current_family)
            contexts.append(family_context)
            for family in range(4):
                positions = int(family_context.eq(family).sum())
                self.live_supervised_family_positions[family] += positions
                if positions:
                    self.last_live_family_tick[family] = self.ticks
            dt = float(self.world.elapsed)
            output, self.h = self.model.step(observation.to(self.device), self.h, dt)
            action = output.detach().argmax(-1).cpu()
            self.world.act(action)
            observations.append(observation)
            targets.append(target)
            actions.append(action)
            elapsed.append(dt)
            outputs.append(output)
            self.ticks += 1
            self.hidden_time += dt
        logits, target = torch.stack(outputs), torch.stack(targets).to(self.device)
        loss = self._loss(logits, target)
        self.stats.add(logits, target)
        gradients = self._gradients(loss, self.config.chunk_ticks)
        self.h = self.h.detach()
        after = self.world.family_exposures()["commands"]
        chunk = copied_block({"observations": torch.stack(observations), "targets": torch.stack(targets),
            "actions": torch.stack(actions), "elapsed": tuple(elapsed), "start_h": start_h,
            "start_tick": start_tick, "end_tick": self.ticks,
            "family_context": torch.stack(contexts),
            "family_counts": {family: after[family] - before[family] for family in range(4)}})
        ticks, actors, jobs = self.config.chunk_ticks, self.config.batch_size, self.config.n_jobs
        for name in ("live_forward_ticks", "live_loss_ticks"):
            self.work[name] += ticks
        for name in ("live_forward_actor_ticks", "live_loss_actor_ticks"):
            self.work[name] += ticks * actors
        for name in ("live_logit_positions", "live_loss_positions"):
            self.work[name] += ticks * actors * jobs
        return gradients, float(loss.detach()), chunk

    def _auxiliary_block(self, current):
        recent = current if self.last_chunk is None else combine_blocks(self.last_chunk, current)
        if self.last_chunk is None:
            block, source = copied_block(current), "startup"
        elif self.config.learning == "history_replay" and self.replay.eligible_count(current["start_tick"]):
            block, source = self.replay.sample(current["start_tick"]), "history_replay"
        else:
            block = copied_block(recent)
            source = "fallback_recent" if self.config.learning == "history_replay" else "recent_repeat"
        length = block["end_tick"] - block["start_tick"]
        warmup = length - self.config.chunk_ticks
        if warmup not in (0, self.config.chunk_ticks):
            raise AssertionError("auxiliary suffix must have exactly one differentiable chunk")
        if source == "history_replay" and block["end_tick"] > current["start_tick"]:
            raise AssertionError("historical source overlaps the current live chunk")
        return block, source, warmup

    def _auxiliary_gradients(self, current):
        block, source, warmup = self._auxiliary_block(current)
        # Actual stored h is the initial condition. Only the warmup is detached;
        # the suffix uses the current pre-update parameters and its full graph.
        hidden, outputs = block["start_h"].to(self.device).detach().clone(), []
        for index, (observation, dt) in enumerate(zip(block["observations"], block["elapsed"])):
            if index < warmup:
                with torch.no_grad():
                    _, hidden = self.model.step(observation.to(self.device), hidden, dt)
                hidden = hidden.detach()
            else:
                output, hidden = self.model.step(observation.to(self.device), hidden, dt)
                outputs.append(output)
        target = block["targets"][warmup:].to(self.device)
        loss = self._loss(torch.stack(outputs), target)
        gradients = self._gradients(loss, self.config.chunk_ticks)
        ticks, actors, jobs = len(block["elapsed"]), self.config.batch_size, self.config.n_jobs
        loss_ticks = ticks - warmup
        for name, count in (("aux_forward_ticks", ticks), ("aux_warmup_ticks", warmup),
                            ("aux_loss_ticks", loss_ticks)):
            self.work[name] += count
            self.work[name.replace("_ticks", "_actor_ticks")] += count * actors
        self.work["aux_logit_positions"] += ticks * actors * jobs
        self.work["aux_loss_positions"] += loss_ticks * actors * jobs
        age = max(0, current["start_tick"] - block["end_tick"])
        stats = self.auxiliary_stats
        stats["calls"] += 1
        stats[source] += 1
        stats["source_age_ticks_sum"] += age
        stats["source_age_ticks_max"] = max(stats["source_age_ticks_max"], age)
        stats["source_ticks"] += ticks
        stats["warmup_ticks"] += warmup
        stats["loss_ticks"] += loss_ticks
        for family, count in block["family_counts"].items():
            stats["source_family_counts"][family] += count
        suffix_positions = {family: int(block["family_context"][warmup:].eq(family).sum()) for family in range(4)}
        for family, count in suffix_positions.items():
            stats["suffix_family_positions"][family] += count
            if count:
                self.last_aux_family_tick[family] = self.ticks
        info = {"source": source, "source_id": block["source_id"], "source_hash": block_hash(block),
            "start_tick": block["start_tick"], "end_tick": block["end_tick"], "age_ticks": age,
            "forward_ticks": ticks, "warmup_ticks": warmup, "loss_ticks": loss_ticks,
            "family_counts": deepcopy(block["family_counts"]), "suffix_family_positions": suffix_positions,
            "loss": float(loss.detach())}
        self.last_auxiliary = {"info": deepcopy(info), "block": copied_block(block)}
        return gradients, info

    def _apply_gradients(self, gradients):
        parameters = dict(self.model.named_parameters())
        if set(gradients) != set(parameters):
            raise ValueError("native BPTT must cover every model parameter")
        raw = gradients["recurrent"]
        eligible = {**gradients, "recurrent": self.operator.before_gradient(raw)}
        before = {name: value.detach().double().clone() for name, value in parameters.items()}
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():
            value = eligible[name]
            if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"invalid native gradient for {name}")
            parameter.grad = value.detach().clone()
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip, error_if_nonfinite=True))
        clip_scale = min(1., self.config.clip / (norm + 1e-6))
        self.optimizer.step()
        group = self.optimizer.param_groups[0]
        structure = self.operator.apply_after_optimizer(self.model.recurrent,
            self.optimizer.state[self.model.recurrent], raw, clip_scale, group["lr"], group["betas"], group["eps"])
        self.optimizer.zero_grad(set_to_none=True)
        self.gradient_updates += 1
        self.work["optimizer_steps"] += 1
        squares = {name: (value.detach().double() - before[name]).square().sum() for name, value in parameters.items()}
        structure.update(architecture=self.config.architecture,
            density=float(self.operator.support_mask.sum()) / self.model.recurrent.numel(),
            eligible_density=float(self.operator.allowed_mask.sum()) / self.model.recurrent.numel(),
            turnover=structure["births"] + structure["deaths"],
            cumulative_turnover=structure["cumulative_births"] + structure["cumulative_deaths"],
            global_adam_age_rule="unchanged; one shared scalar Adam age per parameter")
        return {"gradient_norm": norm, "clip_scale": clip_scale,
            "raw_recurrent_gradient_norm": float(raw.double().norm()),
            "eligible_recurrent_gradient_norm": float(eligible["recurrent"].double().norm()),
            "recurrent_delta": float(squares["recurrent"].sqrt()),
            "all_delta": float(sum(squares.values()).sqrt()), "structure": structure}

    def train_update(self):
        if not self._at_boundary:
            raise RuntimeError("native trainer is not at a completed optimizer boundary")
        if self.ticks + self.config.chunk_ticks > self.config.total_ticks:
            raise ValueError("native finite protocol is complete")
        self._at_boundary = False
        started = time.perf_counter()
        self.model.train()
        live, loss, chunk = self._live_chunk()
        live_h_hash = continuity.state_digest(self.h)
        live_world_hash = continuity.state_digest(self.world.state_dict())
        old, auxiliary = self._auxiliary_gradients(chunk)
        if continuity.state_digest(self.h) != live_h_hash or continuity.state_digest(self.world.state_dict()) != live_world_hash:
            raise AssertionError("auxiliary learning changed the live hidden state or world")
        combined = {name: .5 * value + .5 * old[name] for name, value in live.items()}
        update = self._apply_gradients(combined)
        # Admission happens only after the update; current experience can never
        # enter this update's historical auxiliary source.
        completed = self.replay.add_chunk(chunk)
        self.work["real_ticks_admitted"] += self.config.chunk_ticks
        self.last_chunk = copied_block(chunk)
        self.updates += 1
        action_chunk_hash = continuity.state_digest(chunk["actions"])
        data_chunk_hash = continuity.state_digest({key: chunk[key] for key in DATA_HASH_FIELDS})
        self.action_hash = hashlib.sha256(bytes.fromhex(self.action_hash) + bytes.fromhex(action_chunk_hash)).hexdigest()
        self.data_hash = hashlib.sha256(bytes.fromhex(self.data_hash) + bytes.fromhex(data_chunk_hash)).hexdigest()
        point = {"tick": self.ticks, "update": self.updates, "phase": self.schedule.mode_for_tick(chunk["start_tick"])["phase"],
            "world_tick": self.world.tick, "now": float(self.world.now), "hidden_time": self.hidden_time,
            "live_loss": loss, "auxiliary": auxiliary, "combined_loss": .5 * (loss + auxiliary["loss"]),
            "update_metrics": update, "gradient_updates": self.gradient_updates,
            "hidden_resets": self.hidden_resets, "history_block_admitted": completed,
            "action_chunk_hash": action_chunk_hash, "data_chunk_hash": data_chunk_hash,
            "action_hash": self.action_hash, "data_hash": self.data_hash,
            "event_hash": self.world.event_hash, "live_hidden_hash": live_h_hash,
            "live_world_hash": live_world_hash, "reservoir_hash": self.replay.retention_fingerprint(),
            "reservoir_hash_kind": "compositional-full-content-v1",
            "operator_hash": continuity.state_digest(self.operator.state_dict()),
            "rng_hash": continuity.state_digest(continuity.rng_state()),
            "actual_family_exposures": self.world.family_exposures(),
            "family_recency": self.family_recency(),
            "actual_action_counts": torch.bincount(chunk["actions"].reshape(-1), minlength=11).tolist(),
            "target_action_counts": torch.bincount(chunk["targets"].reshape(-1), minlength=11).tolist()}
        self.trace.append(deepcopy(point))
        self._at_boundary = True
        continuity.synchronize(self.device)
        self.seconds += time.perf_counter() - started
        return point

    def run_updates(self, count):
        _integer("count", count)
        if count > self.config.total_updates - self.updates:
            raise ValueError("requested updates exceed the finite protocol")
        return [self.train_update() for _ in range(count)]

    def semantic_state(self):
        if not self._at_boundary:
            raise RuntimeError("checkpoint only at a completed native optimizer boundary")
        return {"format": FORMAT_VERSION, "config": asdict(self.config), "model_config": self.model.config(),
            "model": continuity.cpu_tree(self.model.state_dict()), "optimizer": continuity.cpu_tree(self.optimizer.state_dict()),
            "optimizer_parameter_names": [name for name, _ in self.model.named_parameters()],
            "hidden": continuity.cpu_tree(self.h), "hidden_time": self.hidden_time,
            "world": continuity.cpu_tree(self.world.state_dict()), "schedule": self.schedule.state_dict(),
            "schedule_history": deepcopy(self.schedule_history), "replay": self.replay.state_dict(),
            "operator": self.operator.state_dict(), "rng": continuity.rng_state(), "stats": self.stats.state_dict(),
            "auxiliary_stats": deepcopy(self.auxiliary_stats), "work": deepcopy(self.work),
            "ticks": self.ticks, "updates": self.updates, "gradient_updates": self.gradient_updates,
            "hidden_resets": self.hidden_resets, "birth": deepcopy(self.birth),
            "last_live_family_tick": list(self.last_live_family_tick),
            "last_aux_family_tick": list(self.last_aux_family_tick),
            "live_supervised_family_positions": list(self.live_supervised_family_positions),
            "action_hash": self.action_hash, "data_hash": self.data_hash, "trace": deepcopy(self.trace),
            "last_chunk": continuity.cpu_tree(self.last_chunk), "last_auxiliary": continuity.cpu_tree(self.last_auxiliary)}

    def state_dict(self):
        semantic = self.semantic_state()
        hashes = {key: continuity.state_digest(semantic[key]) for key in
                  ("model", "optimizer", "hidden", "world", "schedule", "replay", "operator", "rng", "last_chunk", "last_auxiliary")}
        hashes["semantic"] = continuity.state_digest(semantic)
        return {**semantic, "sources": deepcopy(self.sources), "source_hashes": deepcopy(self.source_hashes),
            "dependencies": deepcopy(self.dependencies), "dependency_hashes": deepcopy(self.dependency_hashes),
            "runtime": deepcopy(self.runtime), "runtime_history": deepcopy(self.runtime_history), "seconds": self.seconds,
            "state_hashes": hashes}

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, device=None, strict_sources=True):
        return cls.from_state_dict(torch.load(Path(path), map_location="cpu", weights_only=False),
                                   device=device, strict_sources=strict_sources)

    @classmethod
    def from_state_dict(cls, state, *, device=None, strict_sources=True):
        """Restore native objects without replaying birth or consuming live RNG."""
        if not isinstance(state, Mapping) or state.get("format") != FORMAT_VERSION or set(state) != _SEMANTIC_FIELDS | _RECORD_FIELDS:
            raise ValueError("unsupported or incomplete native trainer state")
        if not isinstance(state["config"], Mapping) or set(state["config"]) != set(NativeConfig.__dataclass_fields__):
            raise ValueError("incomplete native configuration")
        hash_fields = {"model", "optimizer", "hidden", "world", "schedule", "replay", "operator", "rng", "last_chunk", "last_auxiliary"}
        if not isinstance(state["state_hashes"], Mapping) or set(state["state_hashes"]) != hash_fields | {"semantic"}:
            raise ValueError("incomplete native continuation digests")
        if any(state["state_hashes"][key] != continuity.state_digest(state[key]) for key in hash_fields):
            raise ValueError("native continuation digest mismatch")
        if state["state_hashes"]["semantic"] != continuity.state_digest({key: state[key] for key in _SEMANTIC_FIELDS}):
            raise ValueError("native complete continuation digest mismatch")
        for key, expected in (("sources", RUNTIME_SOURCES), ("dependencies", DEPENDENCIES)):
            if not isinstance(state[key], Mapping) or set(state[key]) != set(expected):
                raise ValueError(f"incomplete native {key}")
            hashes = state["source_hashes" if key == "sources" else "dependency_hashes"]
            if hashes != source_hashes(state[key]):
                raise ValueError(f"native {key} snapshot digest mismatch")
        if strict_sources:
            for key, current in (("source_hashes", source_hashes()),
                                 ("dependency_hashes", source_hashes(dependency_snapshots()))):
                if state[key] != current:
                    raise ValueError(f"native checkpoint {key} mismatch; use saved runtime sources")
        config = NativeConfig(**state["config"])
        if device is not None:
            config = replace(config, device=str(device))
        _validate_rng(state["rng"])
        _configure_determinism(config)
        result = cls.__new__(cls)
        result.config, result.device = config, torch.device(config.device)
        # Model construction is shape-aware and isolated from the saved stream.
        # NativeTrainer.__init__/seed_all are deliberately never called here.
        with torch.random.fork_rng(devices=[]):
            result.model = _model(config).to(result.device)
        if state["model_config"] != result.model.config():
            raise ValueError("native model dimensions/solver configuration mismatch")
        expected_parameters = result.model.state_dict()
        if not isinstance(state["model"], Mapping) or set(state["model"]) != set(expected_parameters):
            raise ValueError("incomplete native model parameters")
        for name, value in state["model"].items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected_parameters[name].shape
                    or value.dtype != expected_parameters[name].dtype or not bool(torch.isfinite(value).all())):
                raise ValueError(f"invalid native parameter {name}")
        result.model.load_state_dict(state["model"], strict=True)
        if state["optimizer_parameter_names"] != [name for name, _ in result.model.named_parameters()]:
            raise ValueError("native optimizer parameter order mismatch")
        result.optimizer = _optimizer(result.model, config)
        _validate_optimizer_state(result.model, result.optimizer, state["optimizer"], state["updates"])
        result.optimizer.load_state_dict(state["optimizer"])
        hidden = state["hidden"]
        if (not isinstance(hidden, torch.Tensor) or hidden.shape != (config.batch_size, config.width)
                or hidden.dtype != result.model.encoder.weight.dtype or not bool(torch.isfinite(hidden).all())):
            raise ValueError("invalid live native hidden state")
        result.h = hidden.detach().to(result.device).clone()
        result.world = ContextWorld.from_state_dict(state["world"], device=config.world_device)
        if (result.world.batch_size, result.world.n_jobs, result.world.input_size) != (config.batch_size, config.n_jobs, INPUT_SIZE):
            raise ValueError("native world dimensions mismatch")
        coverage = TRAINING_COVERAGE[config.coverage]
        if (result.world.seed != state["birth"]["data_seed"] or result.world.dt != config.dt
                or result.world.active_jobs != config.n_jobs or tuple(result.world.actor_ids) != tuple(range(config.batch_size))
                or result.world.regime.duration_range != coverage["duration_range"]
                or result.world.regime.gap_range != coverage["gap_range"]
                or result.world.regime.window_range != config.window_range
                or result.world.regime.cancel != config.cancel or result.world.regime.replace != config.replace):
            raise ValueError("native world source/time coverage mismatch")
        result.schedule = Schedule.from_state_dict(state["schedule"])
        if result.schedule.config != config.schedule_config():
            raise ValueError("native schedule/config mismatch")
        result.replay = HistoryReservoir(config.replay_capacity, 2 * config.chunk_ticks,
                                        continuity.seed_for(config.seed, "lifelong-history"))
        result.replay.load_state_dict(state["replay"])
        result.operator = FactorOperator.from_state(result.model.recurrent, state["operator"])
        expected_mode = {"fixed_shrink": "fixed_shrink", "grow_shrink": "grow", "open_shrink": "dense_shrink"}[config.architecture]
        if result.operator.config != FactorConfig(expected_mode, 0., config.shrink_step):
            raise ValueError("native structural configuration mismatch")
        result.stats = continuity.ActionStats(11, result.device, config.wait_weight)
        expected_stats = result.stats.state_dict()
        if not isinstance(state["stats"], Mapping) or set(state["stats"]) != set(expected_stats):
            raise ValueError("incomplete native action statistics")
        for name, value in state["stats"].items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected_stats[name].shape
                    or value.dtype != expected_stats[name].dtype or not bool(torch.isfinite(value).all())
                    or bool(value.lt(0).any())):
                raise ValueError(f"invalid native action statistic {name}")
        result.stats.load_state_dict(state["stats"])
        result.loss_weights = torch.ones(11, device=result.device)
        result.loss_weights[0] = config.wait_weight
        for name in ("hidden_time", "schedule_history", "auxiliary_stats", "work", "ticks", "updates",
                     "gradient_updates", "hidden_resets", "birth", "action_hash", "data_hash", "trace",
                     "last_live_family_tick", "last_aux_family_tick", "live_supervised_family_positions",
                     "last_chunk", "last_auxiliary", "sources", "source_hashes", "dependencies", "dependency_hashes",
                     "runtime_history", "seconds"):
            setattr(result, name, continuity.cpu_tree(state[name]))
        result._validate_continuation()
        result.runtime = continuity.runtime_record(result.device)
        if not result.runtime_history or result.runtime_history[-1] != result.runtime:
            result.runtime_history.append(deepcopy(result.runtime))
        result._at_boundary = True
        continuity.restore_rng_state(state["rng"])
        return result

    def _validate_continuation(self):
        config = self.config
        for name in ("ticks", "updates", "gradient_updates", "hidden_resets"):
            _integer(name, getattr(self, name))
        if (self.ticks != self.updates * config.chunk_ticks or self.updates > config.total_updates
                or self.gradient_updates != self.updates or self.hidden_resets != 0
                or self.operator.updates != self.updates or self.operator.factor_start_updates != 0):
            raise ValueError("inconsistent native clocks or structural age")
        if self.world.tick != self.ticks - 1:
            raise ValueError("native world clock does not match its uninterrupted history")
        if not math.isfinite(self.hidden_time) or not math.isclose(self.hidden_time, float(self.world.now), abs_tol=1e-9):
            raise ValueError("native hidden/world physical clocks differ")
        if self.replay.ticks_seen != self.ticks or self.replay.chunks_seen != self.updates:
            raise ValueError("native reservoir has omitted or duplicated real history")
        if set(self.work) != set(_empty_work()) or set(self.auxiliary_stats) != set(_empty_auxiliary_stats()):
            raise ValueError("incomplete native compute/auxiliary statistics")
        for name in ("last_live_family_tick", "last_aux_family_tick", "live_supervised_family_positions"):
            values = getattr(self, name)
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"invalid native family vector {name}")
            for value in values:
                _integer(name, value, lower=0 if name == "live_supervised_family_positions" else -1)
        for family in range(4):
            live, auxiliary = self.last_live_family_tick[family], self.last_aux_family_tick[family]
            live_count = self.live_supervised_family_positions[family]
            auxiliary_count = self.auxiliary_stats["suffix_family_positions"][family]
            if (live >= self.ticks or auxiliary > self.ticks
                    or (live == -1) != (live_count == 0) or (auxiliary == -1) != (auxiliary_count == 0)
                    or live_count > self.ticks * config.batch_size * config.n_jobs):
                raise ValueError("inconsistent native family feedback recency")
        if (self.work["optimizer_steps"] != self.updates or self.work["backward_calls"] != 2 * self.updates
                or self.work["live_forward_ticks"] != self.ticks or self.work["live_loss_ticks"] != self.ticks
                or self.work["aux_loss_ticks"] != self.ticks or self.work["bp_graph_ticks"] != 2 * self.ticks
                or self.auxiliary_stats["calls"] != self.updates):
            raise ValueError("native compute budget and continuation age disagree")
        if len(self.trace) != self.updates:
            raise ValueError("native trace does not cover every update")
        if not self.schedule_history or self.schedule_history[0] != {"tick": 0, **self.schedule.mode_for_tick(0)}:
            raise ValueError("missing native birth schedule")
        expected_policy = self.schedule.mode_for_tick(max(0, self.ticks - 1))["future_family_allowed"]
        if tuple(self.world.family_allowed) != expected_policy:
            raise ValueError("future family policy was changed before its live boundary")
        if self.birth.get("source") != "from_scratch":
            raise ValueError("native continuation must originate from random birth")
        if self.updates == 0:
            if self.last_chunk is not None or self.last_auxiliary is not None or len(self.optimizer.state):
                raise ValueError("newborn native controller cannot contain trained state")
        else:
            validate_block(self.last_chunk, expected_ticks=config.chunk_ticks)
            if self.last_chunk["end_tick"] != self.ticks or self.last_chunk["start_tick"] != self.ticks - config.chunk_ticks:
                raise ValueError("last native chunk does not reach the live boundary")
            if not isinstance(self.last_auxiliary, Mapping) or set(self.last_auxiliary) != {"info", "block"}:
                raise ValueError("incomplete last native auxiliary source")
            validate_block(self.last_auxiliary["block"])
            if self.last_auxiliary["info"]["source_hash"] != block_hash(self.last_auxiliary["block"]):
                raise ValueError("last native auxiliary source digest mismatch")
            for parameter in self.model.parameters():
                step = self.optimizer.state[parameter].get("step")
                if step is None or float(step) != self.updates:
                    raise ValueError("native Adam global age differs from update age")
            retained = [*self.replay.blocks, self.last_chunk, self.last_auxiliary["block"]]
            if self.replay.partial is not None:
                retained.append(self.replay.partial)
            for block in retained:
                self._validate_original_history(block)

    def _validate_original_history(self, block):
        """Bind retained context metadata to its original live data digest."""
        chunk_ticks = self.config.chunk_ticks
        if ("family_context" not in block or block["start_tick"] % chunk_ticks
                or (block["end_tick"] - block["start_tick"]) % chunk_ticks):
            raise ValueError("retained native history must contain original aligned context-labelled chunks")
        for offset, tick in enumerate(range(block["start_tick"], block["end_tick"], chunk_ticks)):
            start, end = offset * chunk_ticks, (offset + 1) * chunk_ticks
            index = tick // chunk_ticks
            if index >= len(self.trace):
                raise ValueError("retained native history lies beyond its live trace")
            original = self.trace[index]
            causal_data = {"observations": block["observations"][start:end],
                "targets": block["targets"][start:end], "elapsed": tuple(block["elapsed"][start:end]),
                "start_tick": tick, "end_tick": tick + chunk_ticks,
                "family_context": block["family_context"][start:end]}
            if (original["tick"] != tick + chunk_ticks
                    or continuity.state_digest(causal_data) != original["data_chunk_hash"]
                    or continuity.state_digest(block["actions"][start:end]) != original["action_chunk_hash"]):
                raise ValueError("retained native history differs from its original causal live digest")

    def family_recency(self):
        return {**self.world.family_recency(),
            "last_live_family_tick": list(self.last_live_family_tick),
            "last_aux_family_tick": list(self.last_aux_family_tick),
            "live_supervised_family_positions": list(self.live_supervised_family_positions),
            "aux_supervised_family_positions": [self.auxiliary_stats["suffix_family_positions"][family] for family in range(4)],
            "clock_note": "Real cues and live supervision use observed zero-based world ticks; "
                "auxiliary supervision uses the optimizer boundary tick count after its live chunk.",
            "supervision_note": "Position counts use retained pre-action context, including WAIT supervision; "
                "real command recency changes only on actual NEW/REPLACE, never on replay."}

    def report(self):
        birth = deepcopy(self.birth)
        birth["mask_generator_state_hash"] = continuity.state_digest(birth.pop("mask_generator_state"))
        auxiliary_stats = deepcopy(self.auxiliary_stats)
        auxiliary_stats["source_family_counts_note"] = "Commands in the entire stored source window, including detached warmup."
        auxiliary_stats["suffix_family_positions_note"] = "Differentiable suffix positions by persistent current context; idle -1 excluded."
        return {"format": FORMAT_VERSION, "source": "random birth", "config": asdict(self.config),
            "ticks": self.ticks, "updates": self.updates, "total_ticks": self.config.total_ticks,
            "complete": self.ticks == self.config.total_ticks, "birth": birth,
            "hidden_resets": self.hidden_resets, "hidden_time": self.hidden_time,
            "model": self.model.parameter_report(), "schedule": self.schedule.summary(),
            "schedule_history": deepcopy(self.schedule_history), "world_metrics": self.world.metrics(),
            "actual_family_exposures": self.world.family_exposures(), "agreement": self.stats.result(),
            "family_recency": self.family_recency(),
            "auxiliary_stats": auxiliary_stats, "work": deepcopy(self.work),
            "replay": self.replay.summary(), "structure": {**self.operator.summary(),
                "architecture": self.config.architecture,
                "density": float(self.operator.support_mask.sum()) / self.model.recurrent.numel(),
                "eligible_density": float(self.operator.allowed_mask.sum()) / self.model.recurrent.numel(),
                "cumulative_turnover": self.operator.rewire.cumulative_births + self.operator.rewire.cumulative_deaths},
            "state_hashes": {"model": continuity.state_digest(self.model.state_dict()),
                "optimizer": continuity.state_digest(self.optimizer.state_dict()), "hidden": continuity.state_digest(self.h),
                "world": continuity.state_digest(self.world.state_dict()), "replay": continuity.state_digest(self.replay.state_dict()),
                "operator": continuity.state_digest(self.operator.state_dict()), "rng": continuity.state_digest(continuity.rng_state())},
            "source_hashes": deepcopy(self.source_hashes), "dependency_hashes": deepcopy(self.dependency_hashes),
            "runtime": deepcopy(self.runtime), "seconds": self.seconds,
            "optimizer_note": "One Adam step after averaging live and auxiliary gradients; eligibility then joint clip. "
                "Old dormant shadow rules and global parameter Adam age are unchanged.",
            "replay_note": "Stored own-trajectory observations, original causal oracle labels, actions and actual start_h; "
                "detached warmup followed by one differentiable chunk; no live state replacement."}


Trainer = NativeTrainer


def write_run(folder, trainer):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # The mask generator tensor is a checkpoint detail, not JSON metadata.
    report = trainer.report()
    report["birth"].pop("mask_generator_state", None)
    continuity.atomic_json(folder / "report.json", report)
    (folder / "trace.jsonl").write_text("".join(json.dumps(point, ensure_ascii=False, allow_nan=False) + "\n"
                                               for point in trainer.trace), encoding="utf-8")
    for name, text in trainer.sources.items():
        path = folder / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for name, text in trainer.dependencies.items():
        path = folder / "dependencies" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="fixed_shrink")
    parser.add_argument("--schedule", choices=("blocked", "mixed"), default="blocked")
    parser.add_argument("--learning", choices=LEARNING_METHODS, default="recent_repeat")
    parser.add_argument("--coverage", choices=tuple(TRAINING_COVERAGE), default="narrow")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-at", type=int, help="Stop at this completed live tick; defaults to the full protocol.")
    args = parser.parse_args(argv)
    if not args.resume and args.output.exists():
        parser.error("a new run requires a new output directory; use --resume to continue an existing run")
    if args.output.exists() and not args.output.is_dir():
        parser.error("--output must be a directory")
    torch.set_num_threads(1)
    trainer = (NativeTrainer.from_checkpoint(args.resume, device=args.device) if args.resume else
               NativeTrainer(NativeConfig(seed=args.seed, architecture=args.architecture, schedule=args.schedule,
                   learning=args.learning, coverage=args.coverage, device=args.device)))
    stop = trainer.config.total_ticks if args.stop_at is None else args.stop_at
    if stop < trainer.ticks or stop > trainer.config.total_ticks or stop % trainer.config.chunk_ticks:
        parser.error("--stop-at must be a future aligned tick within the finite protocol")
    checkpoints = args.output / "checkpoints"
    try:
        if trainer.ticks in CHECKPOINT_TICKS:
            trainer.save_checkpoint(checkpoints / f"tick_{trainer.ticks:06d}.pt")
        trainer.save_checkpoint(args.output / "latest.pt")
        write_run(args.output, trainer)
        print(f"[lifelong] start update={trainer.updates}/{trainer.config.total_updates} "
              f"tick={trainer.ticks}/{trainer.config.total_ticks} seed={trainer.config.seed} "
              f"{trainer.config.architecture} {trainer.config.schedule} {trainer.config.learning} "
              f"{trainer.config.coverage} device={trainer.device}", flush=True)
        while trainer.ticks < stop:
            point = trainer.train_update()
            if trainer.ticks in CHECKPOINT_TICKS:
                trainer.save_checkpoint(checkpoints / f"tick_{trainer.ticks:06d}.pt")
            if trainer.updates % 128 == 0:
                print(f"[lifelong] update={trainer.updates}/{trainer.config.total_updates} "
                      f"tick={trainer.ticks}/{trainer.config.total_ticks} phase={point['phase']} "
                      f"live={point['live_loss']:.5f} aux={point['auxiliary']['loss']:.5f} "
                      f"density={point['update_metrics']['structure']['density']:.4f} "
                      f"train_seconds={trainer.seconds:.1f}", flush=True)
            if trainer.updates % 256 == 0:
                trainer.save_checkpoint(args.output / "latest.pt")
                write_run(args.output, trainer)
    finally:
        if trainer._at_boundary:
            trainer.save_checkpoint(args.output / "latest.pt")
            write_run(args.output, trainer)


if __name__ == "__main__":
    main()
