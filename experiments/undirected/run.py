"""A native continuous individual with recorded-trajectory temporal credit.

Live actions/world transitions run once under no_grad.  The credit engine
reconstructs that realized parameter trajectory; auxiliary history uses the
current parameters with a detached warmup and one differentiable suffix.
Only their equally weighted gradients enter one ordinary Adam update.
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
import traceback

import numpy as np
import torch

from experiments.continuity import run as continuity
from experiments.lifelong.replay import HistoryReservoir, block_hash, combine_blocks, copied_block, validate_block
from experiments.lifelong.schedule import Schedule, ScheduleConfig
from experiments.lifelong.world import ContextWorld, TRAINING_COVERAGE
from experiments.undirected.model import UndirectedModel, PortGraphModel, _tensor_digest
from experiments.undirected.credit import CREDIT_METHODS, CreditConfig, TemporalCreditEngine, snapshot_parameters, weighted_loss, state_digest as credit_state_digest
from experiments.undirected.plasticity import PairConfig, PairOperator


FORMAT_VERSION = "undirected-native-trainer-v1"
INTERFACES = ("core", "ports")
MODES = ("undirected", "untied")
COURSES = ("wide", "wide-only", "warmup")
STRUCTURES = ("fixed", "fixed_probe", "grow", "retire", "both")
RUNTIME_SOURCES = ("run.py", "model.py", "credit.py", "plasticity.py")
DEPENDENCIES = ("continuity/run.py", "continuity/model.py", "continuity/world.py",
                "lifelong/world.py", "lifelong/schedule.py", "lifelong/replay.py")
SEMANTIC_FIELDS = {"format", "config", "model_config", "model", "optimizer", "parameter_layout",
    "hidden", "hidden_time", "world", "schedule", "schedule_history", "course_history", "replay",
    "operator", "credit", "rng", "stats", "auxiliary_stats", "work", "ticks", "updates",
    "hidden_resets", "birth", "action_hash", "data_hash", "trace", "last_chunk", "last_auxiliary",
    "last_live_family_tick", "last_aux_family_tick", "live_supervised_family_positions"}
RECORD_FIELDS = {"sources", "source_hashes", "dependencies", "dependency_hashes", "runtime",
                 "runtime_history", "seconds", "state_hashes"}
HASH_FIELDS = ("model", "optimizer", "hidden", "world", "schedule", "replay", "operator", "credit",
               "rng", "last_chunk", "last_auxiliary")
DATA_HASH_FIELDS = ("observations", "targets", "elapsed", "start_tick", "end_tick", "family_context")


def _integer(name, value, lower=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return value


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _require_finite(value, label):
    """Classify only an observed numerical value, never an exception string."""
    finite = bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else math.isfinite(float(value))
    if not finite:
        raise FloatingPointError(f"nonfinite {label}")


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("progress reports cannot contain full tensors")
        return _json_safe(value.detach().cpu().item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"unsupported or nonfinite progress value: {type(value).__name__}")


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_trace(path, trace):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for point in trace:
            stream.write(json.dumps(_json_safe(point), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for(seed, namespace):
    return int.from_bytes(hashlib.sha256(f"{FORMAT_VERSION}/{seed}/{namespace}".encode()).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class NativeConfig:
    seed: int = 0
    interface: str = "ports"
    mode: str = "undirected"
    credit: str = "rtrl"
    course: str = "wide"
    structure: str = "fixed"
    width: int = 80
    batch_size: int = 8
    n_jobs: int = 2
    chunk_ticks: int = 128
    dtype: str = "float64"
    device: str = "cpu"
    world_device: str = "cpu"
    cpu_threads: int = 1
    initial_degree: int = 16
    birth_gain: float = .6
    pulse_gain: float = .25
    dt: float = .25
    max_step: float = .25
    lr: float = .001
    clip: float = 1.
    wait_weight: float = .1
    replay_capacity: int = 32
    main_ticks: int = 196608
    withdrawal_ticks: int = 32768
    return_ticks: int = 8192
    warmup_ticks: int = 65536
    macrocycle_ticks: int = 32768
    block_ticks: int = 8192
    jacrev_chunk_size: int = 8
    structural_window_updates: int = 8
    structural_candidates: int = 4
    window_range: tuple[float, float] = (1.5, 2.5)
    cancel: float = .2
    replace: float = .3
    deterministic: bool = True

    def __post_init__(self):
        for name in ("seed", "warmup_ticks"):
            _integer(name, getattr(self, name))
        for name in ("width", "batch_size", "n_jobs", "chunk_ticks", "initial_degree", "replay_capacity", "jacrev_chunk_size", "structural_window_updates", "structural_candidates", "cpu_threads"):
            _integer(name, getattr(self, name), 1)
        if self.interface not in INTERFACES or self.mode not in MODES or self.credit not in CREDIT_METHODS or self.course not in COURSES or self.structure not in STRUCTURES:
            raise ValueError("unknown interface/mode/credit/course/structure")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.mode != "undirected" and self.structure != "fixed":
            raise ValueError("adaptive pair selection currently requires undirected mode")
        if self.width < 39 * self.n_jobs:
            raise ValueError("port-matched birth needs 28 input and 11 output nodes per job")
        if self.initial_degree >= self.width or self.width * self.initial_degree % 2 or (self.initial_degree == 1 and self.width != 2):
            raise ValueError("invalid connected regular birth graph")
        for name in ("birth_gain", "pulse_gain", "dt", "max_step", "lr", "clip", "wait_weight"):
            _positive(name, getattr(self, name))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in (self.cancel, self.replace)) or self.cancel + self.replace > 1:
            raise ValueError("invalid cancellation/replacement probabilities")
        bounds = tuple(float(value) for value in self.window_range)
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("invalid finish window")
        object.__setattr__(self, "window_range", bounds)
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        self.schedule_config()
        self.pair_config()
        if self.warmup_ticks % self.chunk_ticks or self.warmup_ticks > self.main_ticks:
            raise ValueError("warmup must be aligned and finish within main training")
        torch.device(self.device)
        torch.device(self.world_device)

    def schedule_config(self):
        return ScheduleConfig(mode="mixed", seed=seed_for(self.seed, "schedule"), **{name: getattr(self, name) for name in
            ("main_ticks", "withdrawal_ticks", "return_ticks", "macrocycle_ticks", "block_ticks", "chunk_ticks")})

    @property
    def total_ticks(self):
        return self.main_ticks + self.withdrawal_ticks + self.return_ticks

    @property
    def total_updates(self):
        return self.total_ticks // self.chunk_ticks

    @property
    def input_size(self):
        return self.n_jobs * 28

    def coverage_at(self, tick):
        return "narrow" if self.course == "warmup" and tick < self.warmup_ticks else "wide"

    def snapshot_ticks(self):
        middle = self.warmup_ticks + (self.main_ticks - self.warmup_ticks) // (2 * self.chunk_ticks) * self.chunk_ticks
        return tuple(sorted({0, min(self.warmup_ticks, self.main_ticks), middle, self.main_ticks,
                             self.main_ticks + self.withdrawal_ticks, self.total_ticks}))

    def pair_config(self):
        return PairConfig(mode=self.structure, window_updates=self.structural_window_updates,
                          candidates=self.structural_candidates, seed=seed_for(self.seed, "structure"))


TrainConfig = NativeConfig


def source_snapshots():
    return {name: (Path(__file__).parent / name).read_text(encoding="utf-8") for name in RUNTIME_SOURCES}


def dependency_snapshots():
    return {name: (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8") for name in DEPENDENCIES}


def source_hashes(sources=None):
    return {name: hashlib.sha256(source.encode()).hexdigest() for name, source in
            (source_snapshots() if sources is None else sources).items()}


def _configure(config):
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _model(config):
    kind = UndirectedModel if config.interface == "core" else PortGraphModel
    kwargs = dict(input_size=config.input_size, n_jobs=config.n_jobs, n_actions=11,
                  width=config.width, max_step=config.max_step, mode=config.mode)
    if config.interface == "core":
        kwargs["initialization_seed"] = seed_for(config.seed, "interfaces")
    else:
        kwargs["pulse_gain"] = config.pulse_gain
    with torch.random.fork_rng(devices=[]):
        model = kind(**kwargs).to(device=config.device, dtype=getattr(torch, config.dtype))
    if config.interface == "core":
        model.initialize_port_matched_interfaces_(pulse_gain=config.pulse_gain)
    return model


def _layout(model):
    return [(name, tuple(value.shape), str(value.dtype)) for name, value in model.named_parameters()]


def _empty_work():
    return {name: 0 for name in ("live_forward_ticks", "live_forward_actor_ticks", "live_loss_positions",
        "aux_forward_ticks", "aux_warmup_ticks", "aux_loss_ticks", "aux_forward_actor_ticks", "aux_loss_positions",
        "aux_reverse_sweeps", "optimizer_steps", "real_ticks_admitted", "structural_probe_calls",
        "credit_loss_forward_calls", "credit_loss_forward_actor_ticks", "credit_jacobian_forward_calls",
        "credit_jacobian_forward_actor_ticks", "credit_loss_reverse_sweeps", "credit_jacobian_reverse_rows",
        "credit_jacobian_reverse_chunks", "credit_sensitivity_matmul_mac", "structural_probe_forward_ticks",
        "structural_probe_forward_actor_ticks", "structural_probe_loss_ticks", "structural_probe_warmup_ticks")}


class NativeTrainer:
    def __init__(self, config: NativeConfig):
        if not isinstance(config, NativeConfig):
            raise ValueError("NativeTrainer requires NativeConfig")
        self.config, self.device = deepcopy(config), torch.device(config.device)
        _configure(config)
        self.model = _model(config)
        mask, graph_birth = self.model.initialize_birth(mask_seed=seed_for(config.seed, "graph"),
            weight_seed=seed_for(config.seed, "weights"), degree=config.initial_degree, gain=config.birth_gain)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=0.)
        self.operator = PairOperator(self.model, mask, config.pair_config())
        self.credit = TemporalCreditEngine(self.model, CreditConfig(method=config.credit,
            update_ticks=config.chunk_ticks, wait_weight=config.wait_weight, jacrev_chunk_size=config.jacrev_chunk_size))
        self.schedule = Schedule(config.schedule_config())
        policy = self.schedule.mode_for_tick(0)
        coverage = config.coverage_at(0)
        self.world = ContextWorld(seed=seed_for(config.seed, "data"), batch_size=config.batch_size, n_jobs=config.n_jobs,
            dt=config.dt, **TRAINING_COVERAGE[coverage], window_range=config.window_range, cancel=config.cancel,
            replace=config.replace, device=config.world_device, family_allowed=policy["future_family_allowed"],
            regime_name=f"undirected-{coverage}")
        self.h = self.model.initial_state(config.batch_size, self.device)
        self.hidden_time = 0.
        self.replay = HistoryReservoir(config.replay_capacity, 2 * config.chunk_ticks, seed_for(config.seed, "replay"))
        self.stats = continuity.ActionStats(11, self.device, config.wait_weight)
        self.auxiliary_stats = {"calls": 0, "startup": 0, "fallback_recent": 0, "history_replay": 0,
            "source_age_ticks_sum": 0, "source_age_ticks_max": 0, "suffix_family_positions": [0] * 4}
        self.work = _empty_work()
        self.schedule_history = [{"tick": 0, **deepcopy(policy)}]
        self.course_history = [{"tick": 0, "coverage": coverage}]
        self.ticks = self.updates = self.hidden_resets = 0
        self.last_live_family_tick, self.last_aux_family_tick = [-1] * 4, [-1] * 4
        self.live_supervised_family_positions = [0] * 4
        self.birth = {"source": "from_scratch", **graph_birth, "interface": config.interface,
            "interface_birth": "port_matched", "data_seed": seed_for(config.seed, "data"),
            "replay_seed": seed_for(config.seed, "replay"), "initial_parameter_hash": continuity.state_digest(self.model.state_dict()),
            "initial_optimizer_hash": continuity.state_digest(self.optimizer.state_dict()),
            "initial_hidden_hash": continuity.state_digest(self.h), "initial_world_hash": continuity.state_digest(self.world.state_dict())}
        self.action_hash = self.data_hash = hashlib.sha256(f"{FORMAT_VERSION}/{self.birth['data_seed']}".encode()).hexdigest()
        self.trace, self.last_chunk, self.last_auxiliary = [], None, None
        self.sources, self.dependencies = source_snapshots(), dependency_snapshots()
        self.source_hashes, self.dependency_hashes = source_hashes(self.sources), source_hashes(self.dependencies)
        self.runtime = continuity.runtime_record(self.device)
        self.runtime_history = [deepcopy(self.runtime)]
        self.seconds, self._at_boundary = 0., True

    def _before_observe(self):
        policy = self.schedule.mode_for_tick(self.ticks)
        if policy != {key: value for key, value in self.schedule_history[-1].items() if key != "tick"}:
            if tuple(self.world.family_allowed) != policy["future_family_allowed"]:
                self.world.set_family_policy(policy["future_family_allowed"])
            self.schedule_history.append({"tick": self.ticks, **deepcopy(policy)})
        coverage = self.config.coverage_at(self.ticks)
        if coverage != self.course_history[-1]["coverage"]:
            self.world.set_regime(**TRAINING_COVERAGE[coverage], name=f"undirected-{coverage}")
            self.course_history.append({"tick": self.ticks, "coverage": coverage})

    def _live_chunk(self):
        self.operator.project_and_validate()
        parameters = snapshot_parameters(self.model)
        start_tick, start_h = self.ticks, continuity.cpu_tree(self.h)
        before = self.world.family_exposures()["commands"]
        observations, targets, actions, contexts, elapsed, outputs = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(self.config.chunk_ticks):
                self._before_observe()
                observation = self.world.observe().to(dtype=getattr(torch, self.config.dtype)).cpu()
                if not bool(torch.isfinite(observation).all()):
                    raise ValueError("nonfinite live observation data")
                target = self.world.oracle("midpoint").cpu()
                context = self.world.current_family.cpu().clone()
                dt = float(self.world.elapsed)
                output, self.h = self.model.step(observation.to(self.device), self.h, dt)
                _require_finite(self.h, "live hidden state")
                _require_finite(output, "live logits")
                action = output.argmax(-1).cpu()
                self.world.act(action)
                for family in range(4):
                    count = int(context.eq(family).sum())
                    self.live_supervised_family_positions[family] += count
                    if count:
                        self.last_live_family_tick[family] = self.ticks
                observations.append(observation)
                targets.append(target)
                actions.append(action)
                contexts.append(context)
                elapsed.append(dt)
                outputs.append(output)
                self.ticks += 1
                self.hidden_time += dt
        logits, target = torch.stack(outputs), torch.stack(targets).to(self.device)
        self.stats.add(logits, target)
        loss = float(weighted_loss(logits, target, self.config.wait_weight)[0])
        _require_finite(loss, "live loss")
        after = self.world.family_exposures()["commands"]
        chunk = copied_block({"observations": torch.stack(observations), "targets": torch.stack(targets),
            "actions": torch.stack(actions), "elapsed": tuple(elapsed), "start_h": start_h,
            "end_h": continuity.cpu_tree(self.h), "start_tick": start_tick, "end_tick": self.ticks,
            "family_context": torch.stack(contexts), "family_counts": {f: after[f] - before[f] for f in range(4)}})
        self.work["live_forward_ticks"] += self.config.chunk_ticks
        self.work["live_forward_actor_ticks"] += self.config.chunk_ticks * self.config.batch_size
        self.work["live_loss_positions"] += target.numel()
        return parameters, loss, chunk

    def _auxiliary_block(self, current):
        if self.last_chunk is None:
            return copied_block(current), "startup", 0
        if self.replay.eligible_count(current["start_tick"]):
            block, source = self.replay.sample(current["start_tick"]), "history_replay"
        else:
            block, source = combine_blocks(self.last_chunk, current), "fallback_recent"
        warmup = block["end_tick"] - block["start_tick"] - self.config.chunk_ticks
        if warmup != self.config.chunk_ticks:
            raise ValueError("auxiliary history must have exactly one warmup and one loss block")
        return block, source, warmup

    def _replay_loss(self, block, warmup, *, differentiable=True):
        hidden, outputs = block["start_h"].to(self.device).detach().clone(), []
        for index, (observation, dt) in enumerate(zip(block["observations"], block["elapsed"])):
            if not bool(torch.isfinite(observation).all()):
                raise ValueError("nonfinite retained observation data")
            with torch.set_grad_enabled(differentiable and index >= warmup):
                output, hidden = self.model.step(observation.to(self.device), hidden, dt)
            _require_finite(hidden, "replayed hidden state")
            _require_finite(output, "replayed logits")
            if index < warmup:
                hidden = hidden.detach()
            else:
                outputs.append(output)
        loss = weighted_loss(torch.stack(outputs), block["targets"][warmup:].to(self.device), self.config.wait_weight)[0]
        _require_finite(loss, "replayed loss")
        return loss

    def _auxiliary_gradients(self, current):
        block, source, warmup = self._auxiliary_block(current)
        loss = self._replay_loss(block, warmup)
        parameters = dict(self.model.named_parameters())
        values = torch.autograd.grad(loss, tuple(parameters.values()))
        gradients = {name: value.detach().clone() for name, value in zip(parameters, values)}
        for name, value in gradients.items():
            _require_finite(value, f"auxiliary gradient for {name}")
        age = current["start_tick"] - block["end_tick"] if source == "history_replay" else 0
        info = {"source": source, "source_hash": block_hash(block), "start_tick": block["start_tick"],
            "end_tick": block["end_tick"], "age_ticks": age, "forward_ticks": len(block["elapsed"]),
            "warmup_ticks": warmup, "loss_ticks": self.config.chunk_ticks, "loss": float(loss.detach()),
            "auxiliary_credit_hard_cap_ticks": self.config.chunk_ticks}
        self.last_auxiliary = {"info": deepcopy(info), "block": copied_block(block)}
        self.auxiliary_stats["calls"] += 1
        self.auxiliary_stats[source] += 1
        self.auxiliary_stats["source_age_ticks_sum"] += age
        self.auxiliary_stats["source_age_ticks_max"] = max(self.auxiliary_stats["source_age_ticks_max"], age)
        for family in range(4):
            positions = int(block["family_context"][warmup:].eq(family).sum())
            self.auxiliary_stats["suffix_family_positions"][family] += positions
            if positions:
                self.last_aux_family_tick[family] = self.ticks
        self.work["aux_forward_ticks"] += len(block["elapsed"])
        self.work["aux_warmup_ticks"] += warmup
        self.work["aux_loss_ticks"] += self.config.chunk_ticks
        self.work["aux_forward_actor_ticks"] += len(block["elapsed"]) * self.config.batch_size
        self.work["aux_loss_positions"] += self.config.chunk_ticks * self.config.batch_size * self.config.n_jobs
        self.work["aux_reverse_sweeps"] += 1
        return gradients, info

    def _structural_probe(self, current):
        auxiliary = self.last_auxiliary
        live = float(self._replay_loss(current, 0, differentiable=False))
        history = float(self._replay_loss(auxiliary["block"], auxiliary["info"]["warmup_ticks"], differentiable=False))
        self.work["structural_probe_calls"] += 1
        self.work["structural_probe_forward_ticks"] += self.config.chunk_ticks + auxiliary["info"]["forward_ticks"]
        self.work["structural_probe_forward_actor_ticks"] += (self.config.chunk_ticks + auxiliary["info"]["forward_ticks"]) * self.config.batch_size
        self.work["structural_probe_loss_ticks"] += 2 * self.config.chunk_ticks
        self.work["structural_probe_warmup_ticks"] += auxiliary["info"]["warmup_ticks"]
        return {"live": live, "history": history, "mixed": .5 * (live + history)}

    def _apply_gradients(self, gradients, probe=None):
        parameters = dict(self.model.named_parameters())
        if set(gradients) != set(parameters):
            raise ValueError("learning must cover every actual model parameter")
        before = snapshot_parameters(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():
            value = gradients[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"invalid gradient layout for {name}")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"nonfinite gradient for {name}")
            parameter.grad = value.detach().clone()
        # Scale before squaring: finite RTRL gradients near 1e200 must not
        # overflow merely because their ordinary sum of squares does.
        maximum = max(float(parameter.grad.detach().abs().max()) for parameter in parameters.values())
        norm = (maximum * math.sqrt(math.fsum(float((parameter.grad.detach().double() / maximum).square().sum())
                for parameter in parameters.values())) if maximum else 0.)
        _require_finite(norm, "true joint gradient norm exceeds float64 range")
        clip_scale = min(1., self.config.clip / (norm + 1e-6))
        for parameter in parameters.values():
            parameter.grad.copy_((parameter.grad.double() * clip_scale).to(parameter.grad.dtype))
        self.optimizer.step()
        for name, parameter in parameters.items():
            _require_finite(parameter, f"post-Adam parameter {name}")
        self.operator.project_and_validate()
        structure = self.operator.after_optimizer(self.optimizer, gradients, clip_scale, probe=probe)
        if any(not bool(torch.isfinite(value).all()) for value in parameters.values()):
            raise FloatingPointError("nonfinite post-update model parameters")
        self.optimizer.zero_grad(set_to_none=True)
        self.work["optimizer_steps"] += 1
        delta = sum(float((parameter.detach().cpu() - before[name]).double().square().sum()) for name, parameter in parameters.items())
        return {"gradient_norm": norm, "clip_scale": clip_scale,
                "all_delta": math.sqrt(delta), "structure": structure,
                "optimizer_rule": "all potential coordinates enter clipping and Adam; inactive values project to zero; moments retained"}

    def _matrix_diagnostics(self):
        matrix = self.model.recurrent_matrix().detach().cpu().double()
        if not bool(torch.isfinite(matrix).all()):
            raise FloatingPointError("nonfinite physical coupling matrix")
        if self.config.mode == "undirected":
            spectrum = torch.linalg.eigvalsh(matrix)
            radius = norm = float(spectrum.abs().max())
        else:
            radius = float(torch.linalg.eigvals(matrix).abs().max())
            norm = float(torch.linalg.matrix_norm(matrix, ord=2))
        return {"frobenius_norm": float(matrix.norm()), "operator_norm": norm, "spectral_radius": radius,
            "antisymmetric_frobenius_norm": float((matrix - matrix.T).norm()),
            "oriented_nonzero_couplings": int(matrix.ne(0).sum()),
            "scope": "actual post-update matrix; descriptive only, no spectral projection or sensitivity reset"}

    def train_update(self):
        if not self._at_boundary:
            raise RuntimeError("only completed optimizer boundaries may continue")
        if self.ticks + self.config.chunk_ticks > self.config.total_ticks:
            raise ValueError("finite protocol complete")
        self._at_boundary = False
        started = time.perf_counter()
        self.model.train()
        parameters, loss, chunk = self._live_chunk()
        live_h_hash, live_world_hash = continuity.state_digest(self.h), continuity.state_digest(self.world.state_dict())
        if self.credit.sensitivity is not None:
            _require_finite(self.credit.sensitivity, "existing full temporal sensitivity")
        live_gradients, credit_metrics = self.credit.gradients(chunk, parameters)
        if self.credit.sensitivity is not None:
            _require_finite(self.credit.sensitivity, "updated full temporal sensitivity")
        for name, value in live_gradients.items():
            _require_finite(value, f"live gradient for {name}")
        diagnostics = credit_metrics.get("gradient_diagnostics", {})
        for name in ("full_gradient_l2", "next_sensitivity_l2"):
            if diagnostics.get(name) is not None:
                _require_finite(diagnostics[name], f"temporal {name}")
        for name in ("loss_forward_calls", "loss_forward_actor_ticks", "jacobian_forward_calls",
                     "jacobian_forward_actor_ticks", "loss_reverse_sweeps", "jacobian_reverse_rows",
                     "jacobian_reverse_chunks", "sensitivity_matmul_mac"):
            self.work[f"credit_{name}"] += credit_metrics["work"][name]
        auxiliary_gradients, auxiliary = self._auxiliary_gradients(chunk)
        if continuity.state_digest(self.h) != live_h_hash or continuity.state_digest(self.world.state_dict()) != live_world_hash:
            raise AssertionError("credit/replay changed live state/world")
        combined = {name: .5 * value + .5 * auxiliary_gradients[name] for name, value in live_gradients.items()}
        update = self._apply_gradients(combined, probe=lambda: self._structural_probe(chunk))
        if continuity.state_digest(self.h) != live_h_hash or continuity.state_digest(self.world.state_dict()) != live_world_hash:
            raise AssertionError("structural probes changed live state/world")
        completed = self.replay.add_chunk(chunk)
        self.work["real_ticks_admitted"] += self.config.chunk_ticks
        self.last_chunk = copied_block(chunk)
        self.updates += 1
        action_chunk_hash = continuity.state_digest(chunk["actions"])
        data_chunk_hash = continuity.state_digest({key: chunk[key] for key in DATA_HASH_FIELDS})
        self.action_hash = hashlib.sha256(bytes.fromhex(self.action_hash) + bytes.fromhex(action_chunk_hash)).hexdigest()
        self.data_hash = hashlib.sha256(bytes.fromhex(self.data_hash) + bytes.fromhex(data_chunk_hash)).hexdigest()
        point = {"tick": self.ticks, "update": self.updates, "world_tick": self.world.tick,
            "phase": self.schedule.mode_for_tick(chunk["start_tick"])["phase"], "coverage": self.config.coverage_at(chunk["start_tick"]),
            "hidden_time": self.hidden_time, "live_loss": loss, "auxiliary": auxiliary,
            "combined_loss": .5 * (loss + auxiliary["loss"]), "credit": credit_metrics, "update_metrics": update,
            "physical_matrix": self._matrix_diagnostics(),
            "hidden_resets": self.hidden_resets, "history_block_admitted": completed,
            "action_chunk_hash": action_chunk_hash, "data_chunk_hash": data_chunk_hash,
            "action_hash": self.action_hash, "data_hash": self.data_hash, "event_hash": self.world.event_hash,
            "live_hidden_hash": live_h_hash, "live_world_hash": live_world_hash,
            "reservoir_hash": self.replay.retention_fingerprint(), "operator_hash": continuity.state_digest(self.operator.state_dict()),
            "credit_hash": self.credit.state_dict()["state_hash"], "actual_family_exposures": self.world.family_exposures(),
            "actual_action_counts": torch.bincount(chunk["actions"].reshape(-1), minlength=11).tolist(),
            "target_action_counts": torch.bincount(chunk["targets"].reshape(-1), minlength=11).tolist()}
        point["parameter_hash"] = continuity.state_digest(self.model.state_dict())
        point["optimizer_hash"] = continuity.state_digest(self.optimizer.state_dict())
        point["rng_hash"] = continuity.state_digest(continuity.rng_state())
        point["continuity_hash"] = continuity.state_digest({name: point[name] for name in (
            "tick", "parameter_hash", "optimizer_hash", "rng_hash", "live_hidden_hash", "live_world_hash",
            "credit_hash", "operator_hash", "reservoir_hash", "action_hash", "data_hash")})
        self.trace.append(deepcopy(point))
        self._at_boundary = True
        continuity.synchronize(self.device)
        self.seconds += time.perf_counter() - started
        return point

    def run_updates(self, count):
        _integer("count", count)
        if count > self.config.total_updates - self.updates:
            raise ValueError("requested updates exceed finite protocol")
        return [self.train_update() for _ in range(count)]

    def report(self):
        """Small committed-boundary progress; no parameters, S or tape arrays."""
        if not self._at_boundary:
            raise RuntimeError("progress requires a completed optimizer boundary")
        c = self.config
        last = self.trace[-1] if self.trace else None
        boundary_keys = ("tick", "continuity_hash", "parameter_hash", "optimizer_hash", "rng_hash",
            "live_hidden_hash", "live_world_hash", "credit_hash", "operator_hash", "reservoir_hash", "action_hash", "data_hash")
        if last is not None:
            boundary = {name: last[name] for name in boundary_keys}
            physical = last["physical_matrix"]
        else:
            boundary = {"tick": 0, "parameter_hash": self.birth["initial_parameter_hash"],
                "optimizer_hash": self.birth["initial_optimizer_hash"], "live_hidden_hash": self.birth["initial_hidden_hash"],
                "live_world_hash": self.birth["initial_world_hash"], "credit_hash": self.credit.state_dict()["state_hash"],
                "operator_hash": continuity.state_digest(self.operator.state_dict()),
                "reservoir_hash": self.replay.retention_fingerprint(), "rng_hash": continuity.state_digest(continuity.rng_state()),
                "action_hash": self.action_hash, "data_hash": self.data_hash}
            boundary["continuity_hash"] = continuity.state_digest(boundary)
            physical = self._matrix_diagnostics()
        sensitivity = self.credit.sensitivity
        norm = None if self.credit.last_metrics is None else self.credit.last_metrics["gradient_diagnostics"]["next_sensitivity_l2"]
        result = {"format": "undirected-native-progress-v1", "status": "complete" if self.ticks == c.total_ticks else "in_progress",
            "completed": self.ticks == c.total_ticks, "ticks": self.ticks, "updates": self.updates,
            "total_ticks": c.total_ticks, "total_updates": c.total_updates, "progress_fraction": self.ticks / c.total_ticks,
            "snapshot_ticks": list(c.snapshot_ticks()), "config": asdict(c), "model_config": self.model.config(),
            "parameters": self.model.parameter_report(), "parameter_layout": _layout(self.model), "birth": deepcopy(self.birth),
            "physical_matrix": deepcopy(physical), "structure": self.operator.summary(),
            "credit": {"method": c.credit, "update_age": self.credit.updates, "hard_cap_ticks": self.credit.config.credit_ticks,
                "last_metrics": deepcopy(self.credit.last_metrics),
                "sensitivity": {"present": sensitivity is not None, "shape": None if sensitivity is None else list(sensitivity.shape),
                    "dtype": None if sensitivity is None else str(sensitivity.dtype), "l2_norm": norm,
                    "bytes": 0 if sensitivity is None else sensitivity.numel() * sensitivity.element_size()},
                "tape_blocks": len(self.credit.tape), "tape_ranges": [[block["start_tick"], block["end_tick"]] for block in self.credit.tape]},
            "auxiliary_stats": deepcopy(self.auxiliary_stats),
            "last_auxiliary": None if self.last_auxiliary is None else deepcopy(self.last_auxiliary["info"]),
            "replay": self.replay.summary(), "work": deepcopy(self.work), "action_statistics": self.stats.result(),
            "actual_family_exposures": self.world.family_exposures(),
            "recency": {"world_commands": self.world.family_recency(), "last_live_feedback_tick": list(self.last_live_family_tick),
                "last_auxiliary_feedback_tick": list(self.last_aux_family_tick),
                "live_supervised_family_positions": list(self.live_supervised_family_positions)},
            "world_tick": self.world.tick, "hidden_time": self.hidden_time, "hidden_resets": self.hidden_resets,
            "last_policy": self.schedule.mode_for_tick(max(0, self.ticks - 1)), "next_policy": self.schedule.mode_for_tick(self.ticks),
            "last_coverage": c.coverage_at(max(0, self.ticks - 1)), "next_coverage": c.coverage_at(self.ticks),
            "schedule_history": deepcopy(self.schedule_history), "course_history": deepcopy(self.course_history),
            "last_boundary_hashes": boundary, "source_hashes": deepcopy(self.source_hashes),
            "dependency_hashes": deepcopy(self.dependency_hashes), "runtime": deepcopy(self.runtime), "seconds": self.seconds,
            "trace_records": len(self.trace), "note": "Progress and training diagnostics only; no frozen behavioral evaluation score."}
        return _json_safe(result)

    def write_run(self, folder):
        """Publish an atomic report plus complete, JSON-safe update trace."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        report = self.report()
        _atomic_json(folder / "report.json", report)
        _write_trace(folder / "trace.jsonl", self.trace)
        return report

    def _publish_update(self, folder, point):
        # The initial/resume write_run reconstructs the checkpoint's full
        # prefix.  Each following real update appends one line, avoiding an
        # O(lifetime^2) rewrite merely to expose the current tick to readers.
        folder = Path(folder)
        with (folder / "trace.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_json_safe(point), ensure_ascii=False, allow_nan=False) + "\n")
        report = self.report()
        _atomic_json(folder / "report.json", report)
        return report

    def semantic_state(self):
        if not self._at_boundary:
            raise RuntimeError("checkpoint requires completed optimizer boundary")
        return {"format": FORMAT_VERSION, "config": asdict(self.config), "model_config": self.model.config(),
            "model": continuity.cpu_tree(self.model.state_dict()), "optimizer": continuity.cpu_tree(self.optimizer.state_dict()),
            "parameter_layout": _layout(self.model), "hidden": continuity.cpu_tree(self.h), "hidden_time": self.hidden_time,
            "world": continuity.cpu_tree(self.world.state_dict()), "schedule": self.schedule.state_dict(),
            "schedule_history": deepcopy(self.schedule_history), "course_history": deepcopy(self.course_history),
            "replay": self.replay.state_dict(), "operator": self.operator.state_dict(), "credit": self.credit.state_dict(),
            "rng": continuity.rng_state(), "stats": self.stats.state_dict(), "auxiliary_stats": deepcopy(self.auxiliary_stats),
            "work": deepcopy(self.work), "ticks": self.ticks, "updates": self.updates, "hidden_resets": self.hidden_resets,
            "birth": deepcopy(self.birth), "action_hash": self.action_hash, "data_hash": self.data_hash, "trace": deepcopy(self.trace),
            "last_chunk": continuity.cpu_tree(self.last_chunk), "last_auxiliary": continuity.cpu_tree(self.last_auxiliary),
            "last_live_family_tick": list(self.last_live_family_tick), "last_aux_family_tick": list(self.last_aux_family_tick),
            "live_supervised_family_positions": list(self.live_supervised_family_positions)}

    def state_dict(self):
        semantic = self.semantic_state()
        hashes = {name: continuity.state_digest(semantic[name]) for name in HASH_FIELDS}
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
        return cls.from_state_dict(torch.load(Path(path), map_location="cpu", weights_only=False), device=device, strict_sources=strict_sources)

    @classmethod
    def from_state_dict(cls, state, *, device=None, strict_sources=True):
        if not isinstance(state, Mapping) or set(state) != SEMANTIC_FIELDS | RECORD_FIELDS or state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported or incomplete undirected trainer state")
        if not isinstance(state["config"], Mapping) or set(state["config"]) != set(NativeConfig.__dataclass_fields__):
            raise ValueError("incomplete configuration")
        if set(state["state_hashes"]) != set(HASH_FIELDS) | {"semantic"}:
            raise ValueError("incomplete continuation hashes")
        if any(state["state_hashes"][name] != continuity.state_digest(state[name]) for name in HASH_FIELDS):
            raise ValueError("continuation digest mismatch")
        if state["state_hashes"]["semantic"] != continuity.state_digest({name: state[name] for name in SEMANTIC_FIELDS}):
            raise ValueError("complete continuation digest mismatch")
        for key, names in (("sources", RUNTIME_SOURCES), ("dependencies", DEPENDENCIES)):
            if not isinstance(state[key], Mapping) or set(state[key]) != set(names):
                raise ValueError(f"incomplete {key}")
            hashes = "source_hashes" if key == "sources" else "dependency_hashes"
            if state[hashes] != source_hashes(state[key]):
                raise ValueError(f"{key} hash mismatch")
        if strict_sources and (state["source_hashes"] != source_hashes() or state["dependency_hashes"] != source_hashes(dependency_snapshots())):
            raise ValueError("runtime source mismatch; restore using the saved implementation")
        config = NativeConfig(**state["config"])
        if device is not None:
            config = replace(config, device=str(device))
        _validate_rng(state["rng"])
        _configure(config)
        result = cls.__new__(cls)
        result.config, result.device = config, torch.device(config.device)
        result.model = _model(config)  # no graph/weight birth; constructor RNG is isolated
        if result.model.config() != state["model_config"] or _layout(result.model) != state["parameter_layout"]:
            raise ValueError("model configuration or parameter layout mismatch")
        _validate_tensor_map(state["model"], result.model.state_dict(), "model")
        result.model.load_state_dict(state["model"], strict=True)
        result.optimizer = torch.optim.Adam(result.model.parameters(), lr=config.lr, weight_decay=0.)
        _validate_optimizer(state["optimizer"], result.optimizer, result.model, state["updates"])
        result.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        hidden = state["hidden"]
        if not isinstance(hidden, torch.Tensor) or hidden.shape != (config.batch_size, config.width) or hidden.dtype != getattr(torch, config.dtype) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("invalid hidden state dtype/layout")
        result.h = hidden.to(result.device).detach().clone()
        result.world = ContextWorld.from_state_dict(state["world"], device=config.world_device)
        result.schedule = Schedule.from_state_dict(state["schedule"])
        if result.schedule.config != config.schedule_config():
            raise ValueError("schedule configuration mismatch")
        result.replay = HistoryReservoir(config.replay_capacity, 2 * config.chunk_ticks, seed_for(config.seed, "replay"))
        result.replay.load_state_dict(state["replay"])
        result.operator = PairOperator.from_state(result.model, state["operator"])
        if result.operator.config != config.pair_config():
            raise ValueError("pair operator configuration mismatch")
        result.credit = TemporalCreditEngine(result.model, CreditConfig(method=config.credit,
            update_ticks=config.chunk_ticks, wait_weight=config.wait_weight, jacrev_chunk_size=config.jacrev_chunk_size))
        result.credit.load_state_dict(state["credit"])
        result.stats = continuity.ActionStats(11, result.device, config.wait_weight)
        _validate_tensor_map(state["stats"], result.stats.state_dict(), "statistics")
        result.stats.load_state_dict(state["stats"])
        for name in ("hidden_time", "schedule_history", "course_history", "auxiliary_stats", "work", "ticks", "updates", "hidden_resets",
            "birth", "action_hash", "data_hash", "trace", "last_chunk", "last_auxiliary", "last_live_family_tick",
            "last_aux_family_tick", "live_supervised_family_positions", "sources", "source_hashes", "dependencies",
            "dependency_hashes", "runtime_history", "seconds"):
            setattr(result, name, continuity.cpu_tree(state[name]))
        result._at_boundary = True
        result._validate_continuation()
        if result.updates and result.trace[-1]["rng_hash"] != continuity.state_digest(state["rng"]):
            raise ValueError("checkpoint RNG differs from last complete update")
        result.runtime = continuity.runtime_record(result.device)
        if not result.runtime_history or result.runtime_history[-1] != result.runtime:
            result.runtime_history.append(deepcopy(result.runtime))
        continuity.restore_rng_state(state["rng"])
        return result

    def _validate_continuation(self):
        c = self.config
        _integer("ticks", self.ticks)
        _integer("updates", self.updates)
        if self.ticks != self.updates * c.chunk_ticks or self.ticks > c.total_ticks or self.operator.updates != self.updates or self.credit.updates != self.updates or self.hidden_resets != 0:
            raise ValueError("inconsistent clock/credit/operator age")
        if self.world.tick != self.ticks - 1 or not math.isclose(self.hidden_time, float(self.world.now), abs_tol=1e-9):
            raise ValueError("hidden/world clock mismatch")
        if (self.world.batch_size, self.world.n_jobs, self.world.input_size, self.world.seed, self.world.dt) != (c.batch_size, c.n_jobs, c.input_size, seed_for(c.seed, "data"), c.dt):
            raise ValueError("world source/layout mismatch")
        if tuple(self.world.actor_ids) != tuple(range(c.batch_size)) or self.world.active_jobs != c.n_jobs:
            raise ValueError("actor identity mismatch")
        expected_tick = max(0, self.ticks - 1)
        expected_coverage = c.coverage_at(expected_tick)
        if (tuple(self.world.family_allowed) != self.schedule.mode_for_tick(expected_tick)["future_family_allowed"]
                or self.world.regime.duration_range != TRAINING_COVERAGE[expected_coverage]["duration_range"]
                or self.world.regime.gap_range != TRAINING_COVERAGE[expected_coverage]["gap_range"]
                or self.world.regime.window_range != c.window_range or self.world.regime.cancel != c.cancel or self.world.regime.replace != c.replace):
            raise ValueError("future request policy mismatch")
        if self.replay.ticks_seen != self.ticks or self.replay.chunks_seen != self.updates or len(self.trace) != self.updates:
            raise ValueError("history omitted or duplicated")
        action_chain = data_chain = hashlib.sha256(f"{FORMAT_VERSION}/{self.birth['data_seed']}".encode()).hexdigest()
        for index, point in enumerate(self.trace):
            if (point["tick"] != (index + 1) * c.chunk_ticks or point["update"] != index + 1
                    or point["credit"]["update_age"] != index + 1):
                raise ValueError("noncontiguous committed update trace")
            action_chain = hashlib.sha256(bytes.fromhex(action_chain) + bytes.fromhex(point["action_chunk_hash"])).hexdigest()
            data_chain = hashlib.sha256(bytes.fromhex(data_chain) + bytes.fromhex(point["data_chunk_hash"])).hexdigest()
            if point["action_hash"] != action_chain or point["data_hash"] != data_chain:
                raise ValueError("committed live data/action chain mismatch")
            if point["continuity_hash"] != continuity.state_digest({name: point[name] for name in (
                "tick", "parameter_hash", "optimizer_hash", "rng_hash", "live_hidden_hash", "live_world_hash",
                "credit_hash", "operator_hash", "reservoir_hash", "action_hash", "data_hash")}):
                raise ValueError("committed continuity composition mismatch")
        if self.action_hash != action_chain or self.data_hash != data_chain:
            raise ValueError("live data/action endpoint differs from committed history")
        for record in self.credit.tape:
            if record["observations"].dtype != getattr(torch, c.dtype):
                raise ValueError("credit tape observation dtype mismatch")
            index = record["start_tick"] // c.chunk_ticks
            if self.trace[index]["credit"]["record_hash"] != credit_state_digest(record):
                raise ValueError("credit tape differs from its committed live trace")
        if set(self.work) != set(_empty_work()) or self.work["optimizer_steps"] != self.updates or self.work["live_forward_ticks"] != self.ticks or self.work["real_ticks_admitted"] != self.ticks:
            raise ValueError("invalid work ledger")
        if self.auxiliary_stats["calls"] != self.updates or self.work["aux_loss_ticks"] != self.ticks:
            raise ValueError("auxiliary budget mismatch")
        if self.birth.get("source") != "from_scratch" or self.birth.get("data_seed") != seed_for(c.seed, "data"):
            raise ValueError("invalid birth source")
        if (self.birth.get("mask_seed") != seed_for(c.seed, "graph") or self.birth.get("weight_seed") != seed_for(c.seed, "weights")
                or self.birth.get("degree") != c.initial_degree or self.birth.get("width") != c.width
                or self.birth.get("gain") != c.birth_gain or self.birth.get("initial_mask_hash") != _tensor_digest(self.operator.initial_mask)):
            raise ValueError("birth graph/configuration binding mismatch")
        self.model.validate_eligibility(self.operator.mask)
        if self.updates:
            point = self.trace[-1]
            endpoints = {"parameter_hash": continuity.state_digest(self.model.state_dict()),
                "optimizer_hash": continuity.state_digest(self.optimizer.state_dict()),
                "live_hidden_hash": continuity.state_digest(self.h), "live_world_hash": continuity.state_digest(self.world.state_dict()),
                "credit_hash": self.credit.state_dict()["state_hash"],
                "operator_hash": continuity.state_digest(self.operator.state_dict()), "reservoir_hash": self.replay.retention_fingerprint()}
            if any(point[name] != expected for name, expected in endpoints.items()):
                raise ValueError("restored state differs from committed complete boundary")
            validate_block(self.last_chunk, expected_ticks=c.chunk_ticks)
            if self.last_chunk["end_tick"] != self.ticks or self.last_chunk["start_tick"] != self.ticks - c.chunk_ticks or not torch.equal(self.last_chunk["end_h"], self.h.cpu()):
                raise ValueError("last live state boundary mismatch")
            if not torch.equal(self.credit.tape[-1]["end_h"], self.h.cpu()):
                raise ValueError("credit/live state mismatch")
            latest_record = self.credit.tape[-1]
            for name in ("observations", "targets", "elapsed", "start_h", "end_h", "start_tick", "end_tick"):
                if continuity.state_digest(self.last_chunk[name]) != continuity.state_digest(latest_record[name]):
                    raise ValueError("last chunk differs from recorded credit history")
            if not isinstance(self.last_auxiliary, Mapping) or set(self.last_auxiliary) != {"info", "block"} or self.last_auxiliary["info"]["source_hash"] != block_hash(self.last_auxiliary["block"]):
                raise ValueError("invalid auxiliary source binding")
            for block in [self.last_chunk, self.last_auxiliary["block"], *self.replay.blocks] + ([] if self.replay.partial is None else [self.replay.partial]):
                _validate_block_layout(block, c)
                self._validate_original_history(block)
            if self.trace[-1]["tick"] != self.ticks or self.trace[-1]["credit"]["update_age"] != self.updates:
                raise ValueError("trace/credit endpoint mismatch")
        else:
            if self.last_chunk is not None or self.last_auxiliary is not None or self.optimizer.state:
                raise ValueError("newborn cannot contain training state")
            for field, actual in (("initial_parameter_hash", self.model.state_dict()),
                                  ("initial_optimizer_hash", self.optimizer.state_dict()),
                                  ("initial_hidden_hash", self.h), ("initial_world_hash", self.world.state_dict())):
                if self.birth.get(field) != continuity.state_digest(actual):
                    raise ValueError(f"newborn state differs from its actual birth: {field}")

    def _validate_original_history(self, block):
        c = self.config
        start, end = block["start_tick"], block["end_tick"]
        if start % c.chunk_ticks or end % c.chunk_ticks or end > self.ticks:
            raise ValueError("retained history is not aligned original experience")
        expected_h = self.birth["initial_hidden_hash"] if start == 0 else self.trace[start // c.chunk_ticks - 1]["live_hidden_hash"]
        if continuity.state_digest(block["start_h"]) != expected_h:
            raise ValueError("retained initial state differs from its original live boundary")
        before = {f: 0 for f in range(4)} if start == 0 else self.trace[start // c.chunk_ticks - 1]["actual_family_exposures"]["commands"]
        after = self.trace[end // c.chunk_ticks - 1]["actual_family_exposures"]["commands"]
        if block["family_counts"] != {f: after[f] - before[f] for f in range(4)}:
            raise ValueError("retained exposure counts differ from actual live events")
        for tick in range(start, end, c.chunk_ticks):
            offset = tick - start
            point = self.trace[tick // c.chunk_ticks]
            causal = {"observations": block["observations"][offset:offset + c.chunk_ticks],
                "targets": block["targets"][offset:offset + c.chunk_ticks],
                "elapsed": tuple(block["elapsed"][offset:offset + c.chunk_ticks]),
                "start_tick": tick, "end_tick": tick + c.chunk_ticks,
                "family_context": block["family_context"][offset:offset + c.chunk_ticks]}
            if point["data_chunk_hash"] != continuity.state_digest(causal):
                raise ValueError("retained causal data differs from the actual live trace")
            if point["action_chunk_hash"] != continuity.state_digest(block["actions"][offset:offset + c.chunk_ticks]):
                raise ValueError("retained actions differ from the actual live trace")


def _validate_rng(state):
    if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("incomplete RNG state")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
        for value in state["torch_cuda"]:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("invalid CUDA RNG")
    except (TypeError, RuntimeError, KeyError, AttributeError) as error:
        raise ValueError("invalid RNG state") from error


def _validate_tensor_map(actual, expected, label):
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise ValueError(f"incomplete {label}")
    for name, value in actual.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid {label} dtype/layout: {name}")


def _validate_optimizer(state, optimizer, model, updates):
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"} or len(state["param_groups"]) != 1:
        raise ValueError("invalid Adam state")
    group, expected = state["param_groups"][0], optimizer.state_dict()["param_groups"][0]
    if group != expected:
        raise ValueError("Adam configuration/parameter order mismatch")
    ids, parameters = group["params"], list(model.parameters())
    if set(state["state"]) != (set(ids) if updates else set()):
        raise ValueError("missing Adam parameter history")
    for identifier, parameter in zip(ids, parameters):
        if not updates:
            continue
        entry = state["state"][identifier]
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incomplete Adam moments")
        step = entry["step"]
        if not isinstance(step, torch.Tensor) or step.shape != () or step.dtype not in (torch.float32, torch.float64) or float(step) != updates:
            raise ValueError("invalid Adam age")
        for name in ("exp_avg", "exp_avg_sq"):
            value = entry[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()) or (name == "exp_avg_sq" and bool(value.lt(0).any())):
                raise ValueError("invalid Adam moment dtype/layout")


def _validate_block_layout(block, config):
    if "family_context" not in block or "family_counts" not in block:
        raise ValueError("retained contextual experience is incomplete")
    validate_block(block)
    if (block["observations"].shape[1:] != (config.batch_size, config.input_size) or block["observations"].dtype != getattr(torch, config.dtype)
            or block["targets"].shape[1:] != (config.batch_size, config.n_jobs) or bool(block["targets"].ge(11).any())
            or bool(block["actions"].ge(11).any()) or block["start_h"].shape != (config.batch_size, config.width)
            or block["start_h"].dtype != getattr(torch, config.dtype)):
        raise ValueError("retained experience dtype/layout mismatch")


def train(config, output, *, resume=None, stop_at=None, on_update=None):
    """Continue learning; an optional read-only callback publishes audit data.

    The callback sees a completed boundary before checkpoint/progress writes.
    A resumed non-newborn boundary is offered again so an idempotent audit
    writer can repair a prior I/O failure.  It must not mutate learner state.
    No ledger implementation is imported by the native runner.
    """
    if on_update is not None and not callable(on_update):
        raise TypeError("on_update must be callable or None")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    if resume is None:
        if any(folder.iterdir()):
            raise ValueError("output is not empty; explicitly resume a source-bound checkpoint")
        trainer = NativeTrainer(config)
    else:
        trainer = NativeTrainer.from_checkpoint(resume, device=config.device)
        if trainer.config != config:
            raise ValueError("resume configuration differs from requested protocol")
    target = config.total_ticks if stop_at is None else _integer("stop_at", stop_at)
    if target < trainer.ticks or target > config.total_ticks or target % config.chunk_ticks:
        raise ValueError("stop_at must be an aligned boundary within the declared lifetime")
    last_complete = trainer.state_dict()
    last_report = None

    def failed(error, stage):
        last_path = folder / "checkpoint_last_complete.pt"
        temporary = last_path.with_suffix(".pt.tmp")
        torch.save(last_complete, temporary)
        temporary.replace(last_path)
        # Audit writers are outside the learning algorithm.  Their exceptions
        # never create a scientific model-failure outcome.
        numerical = stage == "learning_update" and isinstance(error, FloatingPointError)
        failure = {"format": "undirected-training-failure-v1", "status": "algorithm_numerical_failure" if numerical else "execution_failure",
            "kind": "numerical_failure" if numerical else "execution_failure", "stage": stage,
            "completed": False, "declared_target_tick": target, "declared_lifetime_ticks": config.total_ticks,
            "last_complete_tick": last_complete["ticks"], "failed_tick": trainer.ticks, "observed_tick_when_failed": trainer.ticks,
            "last_complete_checkpoint": str(last_path.resolve()), "exception_type": type(error).__name__,
            "last_complete_checkpoint_sha256": file_sha256(last_path),
            "exception": str(error), "traceback": traceback.format_exc(), "config": asdict(config),
            "recovery_rule": "No reset, sensitivity clipping, seed replacement, or automatic retry; this run failed."}
        _atomic_json(folder / "failure.json", failure)
        report = last_report
        if report is None or report["ticks"] != last_complete["ticks"]:
            # The latest update completed before an audit callback failed.
            # Its full state was already backed up; report that same boundary.
            report = trainer.report()
        _atomic_json(folder / "report.json", {**report, "status": failure["status"], "kind": failure["kind"],
            "completed": False, "failed_tick": trainer.ticks, "last_complete_checkpoint": str(last_path.resolve()),
            "last_complete_checkpoint_sha256": failure["last_complete_checkpoint_sha256"]})
        _write_trace(folder / "trace.jsonl", last_complete["trace"])

    if on_update is not None and trainer.updates:
        try:
            on_update(trainer)
        except BaseException as error:
            failed(error, "audit_recovery_callback")
            raise
    snapshots = set(config.snapshot_ticks())
    if trainer.ticks in snapshots:
        trainer.save_checkpoint(folder / f"checkpoint_tick_{trainer.ticks:06d}.pt")
    trainer.save_checkpoint(folder / "checkpoint_latest.pt")
    last_report = trainer.write_run(folder)
    while trainer.ticks < target:
        stage = "learning_update"
        try:
            point = trainer.train_update()
            # Commit the recovery copy before audit I/O.  A writer failure
            # must preserve this new S/tape/last_chunk, not the previous one.
            last_complete = trainer.state_dict()
            if on_update is not None:
                stage = "audit_update_callback"
                on_update(trainer)
        except BaseException as error:
            failed(error, stage)
            raise
        last_report = trainer._publish_update(folder, point)
        if trainer.ticks in snapshots:
            trainer.save_checkpoint(folder / f"checkpoint_tick_{trainer.ticks:06d}.pt")
        if trainer.ticks in snapshots or trainer.ticks % 32768 == 0 or trainer.ticks == target:
            trainer.save_checkpoint(folder / "checkpoint_latest.pt")
            last_report = trainer.write_run(folder)
            print(json.dumps({"tick": trainer.ticks, "updates": trainer.updates, "loss": point["combined_loss"],
                              "credit": config.credit, "seconds": trainer.seconds}), flush=True)
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-at", type=int)
    for name, choices in (("interface", INTERFACES), ("mode", MODES), ("credit", CREDIT_METHODS), ("course", COURSES), ("structure", STRUCTURES), ("dtype", ("float32", "float64"))):
        p.add_argument(f"--{name}", choices=choices, default=getattr(NativeConfig(), name))
    for name in ("device", "world_device"):
        p.add_argument(f"--{name.replace('_', '-')}", default=getattr(NativeConfig(), name))
    for name in ("seed", "width", "batch_size", "n_jobs", "chunk_ticks", "initial_degree", "main_ticks", "withdrawal_ticks", "return_ticks", "warmup_ticks", "macrocycle_ticks", "block_ticks", "jacrev_chunk_size", "replay_capacity", "structural_window_updates", "structural_candidates", "cpu_threads"):
        p.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(NativeConfig(), name))
    for name in ("birth_gain", "pulse_gain", "dt", "max_step", "lr", "clip", "wait_weight"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(NativeConfig(), name))
    args = vars(parser.parse_args(argv))
    args.pop("command")
    output, resume, stop_at = (args.pop(name) for name in ("output", "resume", "stop_at"))
    train(NativeConfig(**args), output, resume=resume, stop_at=stop_at)


if __name__ == "__main__":
    main()
