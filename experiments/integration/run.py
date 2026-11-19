"""Continue a frozen continuity checkpoint with explicit learning interventions.

The inherited model, Adam, world, live state, random generators and finite old
experience bank remain one continuation. Historical passes have their own
hidden state and eligibility trace. No historical pass replaces the live state.

All learning arms retain causal-oracle supervision on the learner's own live
trajectory. The optional temporal coactivity update has no target input; these
hybrids are supervised experiments, not reinforcement learning.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from experiments.continuity import run as continuity


FORMAT_VERSION = "integration-trainer-v1"


@dataclass(frozen=True)
class Arm:
    current_gradient: bool = True
    extra_gradient: str | None = None
    local_source: str | None = None
    shuffled: bool = False


ARMS = {
    "frozen": Arm(current_gradient=False),
    "bp64": Arm(),
    "repeat64": Arm(extra_gradient="current"),
    "replay64": Arm(extra_gradient="replay"),
    "accum128": Arm(),
    "bp128": Arm(),
    "coact_live": Arm(local_source="live"),
    "coact_mix": Arm(local_source="mix"),
    "coact_mix_shuffled": Arm(local_source="mix", shuffled=True),
    "coact_only_mix": Arm(current_gradient=False, local_source="mix"),
    "coverage_long_dense": Arm(),
    "coverage_short_sparse": Arm(),
}


@dataclass(frozen=True)
class IntegrationConfig:
    mode: str = "bp64"
    live_ticks: int | None = None
    gradient_horizon: int | None = None
    duration_range: tuple[float, float] | None = None
    gap_range: tuple[float, float] | None = None
    local_lr: float = 0.001
    local_tau: float = 16.0
    regime_name: str = "integration"

    def __post_init__(self) -> None:
        if self.mode not in ARMS:
            raise ValueError(f"unknown integration mode: {self.mode}")
        if self.live_ticks is None:
            object.__setattr__(self, "live_ticks", 128 if self.mode in {"accum128", "bp128"} else 64)
        if self.gradient_horizon is None:
            object.__setattr__(self, "gradient_horizon", self.live_ticks // 2 if self.mode == "accum128"
                               else self.live_ticks)
        if self.duration_range is None:
            object.__setattr__(self, "duration_range", (1.0, 4.0) if self.mode == "coverage_short_sparse"
                               else (4.0, 8.0))
        if self.gap_range is None:
            object.__setattr__(self, "gap_range", (0.5, 2.0) if self.mode == "coverage_long_dense"
                               else (4.0, 12.0))
        for name in ("live_ticks", "gradient_horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.gradient_horizon > self.live_ticks or self.live_ticks % self.gradient_horizon:
            raise ValueError("gradient_horizon must be a positive divisor of live_ticks")
        for name in ("duration_range", "gap_range"):
            value = getattr(self, name)
            if value is None:
                continue
            values = tuple(map(float, value))
            if (len(values) != 2 or any(not math.isfinite(x) for x in values)
                    or values[0] <= 0 or values[1] < values[0]):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, values)
        if not math.isfinite(self.local_lr) or self.local_lr < 0:
            raise ValueError("local_lr must be finite and nonnegative")
        if not math.isfinite(self.local_tau) or self.local_tau <= 0:
            raise ValueError("local_tau must be finite and positive")
        if not self.regime_name:
            raise ValueError("regime_name cannot be empty")


def source_snapshots() -> dict[str, str]:
    folder = Path(__file__).parent
    return {name: (folder / name).read_text(encoding="utf-8")
            for name in ("run.py", "local.py", "test_run.py", "test_local.py")
            if (folder / name).is_file()}


def source_hashes(sources: dict[str, str] | None = None) -> dict[str, str]:
    return {name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in (source_snapshots() if sources is None else sources).items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_continuation(state: dict[str, Any], device: str | None,
                          strict_sources: bool) -> continuity.Trainer:
    """Restore the unchanged trainer schema without a second large disk copy."""
    if state.get("format") != continuity.FORMAT_VERSION:
        raise ValueError("unsupported inherited continuation format")
    if strict_sources:
        current = continuity.source_hashes()
        for name in ("world.py", "model.py", "run.py"):
            if state["source_hashes"].get(name) != current.get(name):
                raise ValueError(f"inherited checkpoint source mismatch for {name}")
    config = continuity.TrainConfig(**state["config"])
    if device is not None:
        config.device = device
    base = continuity.Trainer(continuity.Variant(**state["variant"]), config)
    base.model.load_state_dict(state["model"])
    base.optimizer.load_state_dict(state["optimizer"])
    base.h = state["hidden"].to(base.device).detach()
    base.world.load_state_dict(state["world"])
    base.replay.load_state_dict(state["replay"])
    base.stats.load_state_dict(state["stats"])
    payloads = {"format", "config", "variant", "model", "optimizer", "hidden",
                "world", "replay", "rng", "stats"}
    for name in base.semantic_state():
        if name not in payloads:
            setattr(base, name, deepcopy(state[name]))
    for name in ("sources", "source_hashes", "seconds", "curve"):
        setattr(base, name, deepcopy(state[name]))
    base.runtime_history = deepcopy(state["runtime_history"]) + [deepcopy(base.runtime)]
    continuity.restore_rng_state(state["rng"])
    return base


class IntegrationTrainer:
    """A versioned extension around, rather than a modification of, old Trainer."""

    def __init__(self, base: continuity.Trainer, config: IntegrationConfig, *,
                 origin: dict[str, Any] | None = None):
        if not base._at_boundary or base.stage != "A":
            raise ValueError("new integration stage requires an A optimizer boundary")
        if base.variant.policy != "online" or base.variant.kind != "learned" or base.variant.reset_every:
            raise ValueError("integration requires a learned continuous own-action controller")
        self.base = base
        self.integration_config = deepcopy(config)
        self.arm = ARMS[config.mode]
        requires_old = self.arm.extra_gradient == "replay" or self.arm.local_source is not None
        if requires_old and not base.replay.blocks:
            raise ValueError("this arm requires complete inherited A experiences")
        if requires_old and any(any(block["resets"]) for block in base.replay.blocks):
            raise ValueError("integration old experiences must come from a continuous source")
        self.origin = deepcopy(origin or {})
        self.origin.update(semantic_hash=continuity.state_digest(base.semantic_state()),
                           model_hash=continuity.tensor_digest(base.model.state_dict()),
                           optimizer_hash=continuity.state_digest(base.optimizer.state_dict()),
                           hidden_hash=continuity.state_digest(base.h),
                           world_hash=continuity.state_digest(base.world.state_dict()),
                           replay_hash=continuity.state_digest(base.replay.state_dict()),
                           rng_hash=continuity.state_digest(continuity.rng_state()),
                           tick=base.ticks, update=base.updates,
                           inherited_source_hashes=deepcopy(base.source_hashes))
        self.sources = source_snapshots()
        self.source_hashes = source_hashes(self.sources)
        self.replay_rng = random.Random()
        self.replay_rng.setstate(base.replay.rng.getstate())
        self.local_live = self._make_local("live") if self.arm.local_source is not None else None
        self.local_replay = self._make_local("replay") if self.arm.local_source is not None else None
        self.local_updates = 0
        self.update_norms = {name: 0.0 for name in (
            "bp_recurrent_frob_sum", "bp_recurrent_frob_squared_sum", "bp_all_frob_sum",
            "bp_all_frob_squared_sum", "local_recurrent_frob_sum", "local_recurrent_frob_squared_sum")}
        self.work = {name: 0 for name in (
            "live_forward_ticks", "current_reprocess_forward_ticks", "old_forward_ticks",
            "old_burnin_ticks", "supervised_ticks", "local_contributing_ticks",
            "backward_horizon_ticks", "old_samples")}
        # begin_adaptation preserves pending contracts, state, Adam and RNG. A
        # second regime change, when requested, also affects future NEW draws only.
        base.begin_adaptation("frozen" if not self.arm.current_gradient else "current",
                              config.duration_range)
        if config.gap_range is not None:
            base.world.set_regime(gap_range=config.gap_range, name=config.regime_name)
        base.adaptation_mode = config.mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def stage_ticks(self) -> int:
        return self.base.ticks - self.base.phase_start_tick

    def _make_local(self, namespace: str):
        from experiments.integration.local import LocalConfig, TemporalOja

        config = LocalConfig(lr=self.integration_config.local_lr,
                             tau=self.integration_config.local_tau,
                             shuffled=self.arm.shuffled,
                             seed=continuity.seed_for(self.base.config.seed, f"integration-local-{namespace}"))
        return TemporalOja(self.base.config.batch_size, self.base.model.width, config,
                           device=self.base.device, dtype=self.base.h.dtype)

    @classmethod
    def from_base_checkpoint(cls, path: str | Path, config: IntegrationConfig,
                             device: str | None = None, strict_sources: bool = True):
        path = Path(path).resolve()
        base = continuity.Trainer.from_checkpoint(path, device=device, strict_sources=strict_sources)
        return cls(base, config, origin={"checkpoint": str(path), "checkpoint_sha256": _file_sha256(path)})

    def _live_chunk(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        base = self.base
        observations, targets, actions, elapsed, logits = [], [], [], [], []
        start_tick, start_h = base.ticks, continuity.cpu_tree(base.h)
        context = nullcontext() if self.arm.current_gradient else torch.no_grad()
        with context:
            for index in range(self.integration_config.live_ticks):
                if index and index % self.integration_config.gradient_horizon == 0:
                    # Accumulate losses across detached graph segments without
                    # altering any hidden value or updating parameters midway.
                    base.h = base.h.detach()
                _, reset = base._before_observe()
                if reset:
                    raise AssertionError("integration may not reset the live hidden state")
                observation = continuity.cpu_tree(base.world.observe())
                target = continuity.cpu_tree(base.world.oracle("midpoint"))
                dt = float(base.world.elapsed)
                pre_h = base.h
                output, base.h = base.model.step(observation.to(base.device), base.h, dt)
                action = output.detach().argmax(-1).cpu()
                base.world.act(action)
                if self.local_live is not None:
                    self.local_live.observe(pre_h.detach(), base.h.detach(), dt)
                observations.append(observation)
                targets.append(target)
                actions.append(action)
                elapsed.append(dt)
                logits.append(output)
                base.ticks += 1
        chunk = {"observations": torch.stack(observations), "targets": torch.stack(targets),
                 "actions": torch.stack(actions), "elapsed": tuple(elapsed),
                 "resets": (False,) * len(elapsed), "start_tick": start_tick, "start_h": start_h}
        self.work["live_forward_ticks"] += len(elapsed)
        return torch.stack(logits), chunk["targets"].to(base.device), chunk

    def _optimize_audited(self, loss: torch.Tensor) -> tuple[float, dict[str, float]]:
        parameters = dict(self.base.model.named_parameters())
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        grad_norm = self.base._optimize(loss)
        with torch.no_grad():
            squared = {name: (parameter.detach().double() - before[name].double()).square().sum()
                       for name, parameter in parameters.items()}
            recurrent = float(squared["recurrent"].sqrt())
            total = float(sum(squared.values()).sqrt())
        self.update_norms["bp_recurrent_frob_sum"] += recurrent
        self.update_norms["bp_recurrent_frob_squared_sum"] += recurrent**2
        self.update_norms["bp_all_frob_sum"] += total
        self.update_norms["bp_all_frob_squared_sum"] += total**2
        return grad_norm, {"bp_recurrent_delta_frob": recurrent, "bp_all_delta_frob": total}

    def _old_pass(self, *, gradient: bool, local: bool) -> dict[str, Any]:
        base = self.base
        index = self.replay_rng.randrange(len(base.replay.blocks))
        block = base.replay.blocks[index]
        burnin = base.config.replay_burnin
        h, logits = block["start_h"].to(base.device), []
        rule = self.local_replay if local else None
        if rule is not None:
            rule.reset_history_()
        for tick, (observation, dt, reset) in enumerate(zip(
                block["observations"], block["elapsed"], block["resets"])):
            if reset:
                h = base.model.initial_state(base.config.batch_size, base.device)
                if rule is not None:
                    rule.reset_history_()
            pre_h = h
            context = nullcontext() if gradient and tick >= burnin else torch.no_grad()
            with context:
                output, h = base.model.step(observation.to(base.device), h, dt)
            if rule is not None:
                rule.observe(pre_h.detach(), h.detach(), dt, accumulate=tick >= burnin)
            if tick < burnin:
                h = h.detach()
            elif gradient:
                logits.append(output)
        train_ticks = len(block["elapsed"]) - burnin
        extra_loss, grad_norm = None, None
        deltas = {"extra_bp_recurrent_delta_frob": None, "extra_bp_all_delta_frob": None}
        if gradient:
            # These targets were recorded before the original learner's own
            # actions in its own historical world. No present-world relabeling.
            target = block["targets"][burnin:].to(base.device)
            loss = base._loss(torch.stack(logits), target)
            grad_norm, measured = self._optimize_audited(loss)
            deltas = {f"extra_{name}": value for name, value in measured.items()}
            extra_loss = float(loss.detach())
            self.work["supervised_ticks"] += train_ticks
            self.work["backward_horizon_ticks"] += train_ticks
        self.work["old_forward_ticks"] += len(block["elapsed"])
        self.work["old_burnin_ticks"] += burnin
        self.work["old_samples"] += 1
        return {"old_block_index": index, "replay_start_tick": block["start_tick"],
                "extra_loss": extra_loss, "extra_grad_norm": grad_norm, **deltas}

    def train_update(self) -> dict[str, Any]:
        base = self.base
        if not base._at_boundary:
            raise RuntimeError("trainer is not at a complete optimizer boundary")
        base._at_boundary = False
        started = time.perf_counter()
        base.model.train()
        base.optimizer.zero_grad(set_to_none=True)
        gradient_before, local_before = base.gradient_updates, self.local_updates
        logits, targets, chunk = self._live_chunk()
        base.stats.add(logits, targets)
        with torch.no_grad():
            live_loss = float(base._loss(logits.detach(), targets))
        grad_norm, gradient_ticks = None, 0
        deltas = {"bp_recurrent_delta_frob": None, "bp_all_delta_frob": None}
        if self.arm.current_gradient:
            gradient_ticks = self.integration_config.gradient_horizon
            loss = base._loss(logits, targets)
            grad_norm, deltas = self._optimize_audited(loss)
            self.work["supervised_ticks"] += self.integration_config.live_ticks
            self.work["backward_horizon_ticks"] += self.integration_config.live_ticks
        # Parameters may change, but the actual forward trajectory is retained.
        base.h = base.h.detach()
        live_hidden_hash = continuity.state_digest(base.h)
        extra = {"extra_loss": None, "extra_grad_norm": None,
                 "replay_start_tick": None, "old_block_index": None, "local": None,
                 "extra_bp_recurrent_delta_frob": None, "extra_bp_all_delta_frob": None}
        if self.arm.extra_gradient == "current":
            base.optimizer.zero_grad(set_to_none=True)
            output, _ = base._forward_sequence(chunk["observations"].to(base.device),
                                               chunk["start_h"].to(base.device),
                                               chunk["elapsed"], chunk["resets"])
            loss = base._loss(output, targets)
            extra["extra_grad_norm"], measured = self._optimize_audited(loss)
            extra.update({f"extra_{name}": value for name, value in measured.items()})
            extra["extra_loss"] = float(loss.detach())
            self.work["current_reprocess_forward_ticks"] += self.integration_config.live_ticks
            self.work["supervised_ticks"] += self.integration_config.live_ticks
            self.work["backward_horizon_ticks"] += self.integration_config.live_ticks
        if self.arm.extra_gradient == "replay" or self.arm.local_source is not None:
            base.optimizer.zero_grad(set_to_none=True)
            extra.update(self._old_pass(gradient=self.arm.extra_gradient == "replay",
                                        local=self.arm.local_source is not None))
        if self.local_live is not None:
            mixture = self.local_replay if self.arm.local_source == "mix" else None
            extra["local"] = self.local_live.apply_(base.model.recurrent, other=mixture)
            local_delta = extra["local"]["delta_frob"]
            self.update_norms["local_recurrent_frob_sum"] += local_delta
            self.update_norms["local_recurrent_frob_squared_sum"] += local_delta**2
            if mixture is None:
                self.local_replay.clear_accumulator()
            self.local_updates += 1
            self.work["local_contributing_ticks"] += self.integration_config.live_ticks
            if mixture is not None:
                self.work["local_contributing_ticks"] += self.base.config.replay_block_ticks - self.base.config.replay_burnin
        if continuity.state_digest(base.h) != live_hidden_hash:
            raise AssertionError("historical processing overwrote the live hidden value")
        base.optimizer.zero_grad(set_to_none=True)
        base.updates += 1
        action_chunk_hash = continuity.tensor_digest({"actions": chunk["actions"]})
        data_chunk_hash = continuity.state_digest({key: chunk[key] for key in
                    ("observations", "targets", "elapsed", "resets", "start_tick")})
        base.action_hash = hashlib.sha256(bytes.fromhex(base.action_hash)
                                         + bytes.fromhex(action_chunk_hash)).hexdigest()
        base.data_hash = hashlib.sha256(bytes.fromhex(base.data_hash)
                                       + bytes.fromhex(data_chunk_hash)).hexdigest()
        point = {"update": base.updates, "stage_update": base.stage_updates, "stage": base.stage,
                 "tick": base.ticks, "stage_ticks": self.stage_ticks,
                 "world_tick": base.world.tick, "now": float(base.world.now),
                 "dt": float(base.world.dt), "elapsed_min": min(chunk["elapsed"]),
                 "elapsed_max": max(chunk["elapsed"]), "loss": live_loss,
                 "grad_norm": grad_norm, "gradient_horizon_this_chunk": gradient_ticks,
                 "gradient_updates": base.gradient_updates,
                 "gradient_updates_this_chunk": base.gradient_updates - gradient_before,
                 "local_updates": self.local_updates,
                 "local_updates_this_chunk": self.local_updates - local_before,
                 "hidden_resets": base.hidden_resets, "live_hidden_hash": live_hidden_hash,
                 "action_chunk_hash": action_chunk_hash, "data_chunk_hash": data_chunk_hash,
                 "action_hash": base.action_hash, "data_hash": base.data_hash,
                 "event_hash": base.world.event_hash, "work": deepcopy(self.work),
                 "update_norms": deepcopy(self.update_norms),
                 "actual_action_counts": torch.bincount(chunk["actions"].reshape(-1),
                                             minlength=base.model.n_actions).tolist(),
                 "target_action_counts": torch.bincount(chunk["targets"].reshape(-1),
                                             minlength=base.model.n_actions).tolist(), **deltas, **extra}
        base.trace.append(deepcopy(point))
        base._at_boundary = True
        continuity.synchronize(base.device)
        base.seconds += time.perf_counter() - started
        if base.stage_updates == 1 or base.stage_updates % base.config.log_every == 0:
            point = {**point, "seconds": base.seconds, "world_metrics": base.world.metrics(),
                     "agreement": base.stats.result(), "replay": base.replay.summary()}
            base.curve.append(deepcopy(point))
        return point

    def run_updates(self, count: int) -> list[dict[str, Any]]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("update count must be a nonnegative integer")
        return [self.train_update() for _ in range(count)]

    def semantic_state(self) -> dict[str, Any]:
        return {"format": FORMAT_VERSION, "continuation": self.base.semantic_state(),
                "integration_config": asdict(self.integration_config), "origin": deepcopy(self.origin),
                "replay_rng": self.replay_rng.getstate(),
                "local_live": None if self.local_live is None else self.local_live.state_dict(),
                "local_replay": None if self.local_replay is None else self.local_replay.state_dict(),
                "local_updates": self.local_updates, "work": deepcopy(self.work),
                "update_norms": deepcopy(self.update_norms)}

    def state_dict(self) -> dict[str, Any]:
        state = self.semantic_state()
        state["continuation"] = self.base.state_dict()
        state.update(sources=deepcopy(self.sources), source_hashes=deepcopy(self.source_hashes))
        return state

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None,
                        strict_sources: bool = True):
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls._from_state(state, device=device, strict_sources=strict_sources)

    @classmethod
    def _from_state(cls, state: dict[str, Any], device: str | None = None,
                    strict_sources: bool = True):
        if state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported integration checkpoint format")
        if strict_sources:
            current = source_hashes()
            saved = source_hashes(state["sources"])
            for name in ("run.py", "local.py"):
                if saved.get(name) != state["source_hashes"].get(name):
                    raise ValueError(f"integration snapshot/hash mismatch for {name}")
                if state["source_hashes"].get(name) != current.get(name):
                    raise ValueError(f"integration checkpoint source mismatch for {name}")
        trainer = cls.__new__(cls)
        trainer.base = _restore_continuation(state["continuation"], device, strict_sources)
        trainer.integration_config = IntegrationConfig(**state["integration_config"])
        trainer.arm = ARMS[trainer.integration_config.mode]
        for name in ("origin", "local_updates", "work", "update_norms", "sources", "source_hashes"):
            setattr(trainer, name, deepcopy(state[name]))
        trainer.replay_rng = random.Random()
        trainer.replay_rng.setstate(state["replay_rng"])
        for name, namespace in (("local_live", "live"), ("local_replay", "replay")):
            rule = trainer._make_local(namespace) if state[name] is not None else None
            if rule is not None:
                rule.load_state_dict(state[name])
            setattr(trainer, name, rule)
        return trainer

    def report(self) -> dict[str, Any]:
        report = self.base.report()
        adam_steps = int(self.arm.current_gradient) + int(self.arm.extra_gradient is not None)
        if not self.arm.current_gradient:
            gradient_semantics = (f"Adam performs 0 updates per {self.integration_config.live_ticks} live ticks. "
                                  "This stage has no label-gradient credit; hidden state values persist.")
            if self.arm.local_source is None:
                learning = ("Weights remain frozen. Own-action causal-oracle labels are recorded only "
                            "for metrics; no learning updates occur in this stage.")
            else:
                learning = ("Forward temporal coactivity updates only recurrent weights. This adaptation "
                            "stage uses no target-label gradients; encoder and readout retain their "
                            "supervised pretrained values.")
        else:
            gradient_semantics = ("One weighted live loss covers all live ticks; hidden state values persist. "
                                  f"Live graphs detach every {self.integration_config.gradient_horizon} ticks. "
                                  f"Adam performs {adam_steps} updates per {self.integration_config.live_ticks} live ticks.")
            learning = "Adam weighted causal-oracle cross entropy trains on the model's own live trajectory."
            if self.arm.extra_gradient == "current":
                learning += " A second Adam pass reprocesses the same current observations and causal labels."
            elif self.arm.extra_gradient == "replay":
                learning += " A second Adam pass trains on recorded old experiences and their original causal labels."
            if self.arm.extra_gradient is not None:
                gradient_semantics += " The extra pass has a separate hidden state and does not extend the live gradient."
            if self.arm.local_source is not None:
                learning += (" A forward temporal coactivity update also changes recurrent weights without "
                             "reading target labels. This is supervised hybrid training.")
            else:
                learning += " This is supervised training."
        report.update(format=FORMAT_VERSION, integration_config=asdict(self.integration_config),
                      arm=asdict(self.arm), origin=deepcopy(self.origin), stage_ticks=self.stage_ticks,
                      work=deepcopy(self.work), update_norms=deepcopy(self.update_norms),
                      local_updates=self.local_updates,
                      stage_batch_observations=self.stage_ticks * self.base.config.batch_size,
                      stage_action_labels_observed=self.stage_ticks * self.base.config.batch_size * self.base.config.n_jobs,
                      parameter_updates={"adam_parameters": sum(p.numel() for p in self.base.model.parameters()
                                                                if p.requires_grad) if self.arm.current_gradient else 0,
                                         "local_parameters": self.base.model.recurrent.numel()
                                         if self.arm.local_source is not None else 0},
                      local_live=None if self.local_live is None else self.local_live.summary(),
                      local_replay=None if self.local_replay is None else self.local_replay.summary(),
                      integration_source_hashes=deepcopy(self.source_hashes),
                      learning=learning, gradient_semantics=gradient_semantics,
                      budget_note=("Report live events, forward ticks, loss ticks, gradient steps and local "
                                   "updates separately. Old passes include inherited burn-in. Historical "
                                   "hidden states and zero-start local traces are approximate reconstructions."))
        return report


def load_checkpoint(path: str | Path, device: str | None = None,
                    strict_sources: bool = True):
    """Load either generation without allowing old code to silently drop traces."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("format") == FORMAT_VERSION:
        return IntegrationTrainer._from_state(state, device=device, strict_sources=strict_sources)
    if state.get("format") == continuity.FORMAT_VERSION:
        return _restore_continuation(state, device, strict_sources)
    raise ValueError("unsupported checkpoint format")


def write_run_artifacts(trainer: IntegrationTrainer, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(folder / "checkpoint.pt")
    continuity.atomic_json(folder / "report.json", trainer.report())
    for prefix, sources in (("integration", trainer.sources), ("continuity", trainer.base.sources)):
        destination = folder / "sources" / prefix
        destination.mkdir(parents=True, exist_ok=True)
        for name, source in sources.items():
            (destination / name).write_text(source, encoding="utf-8")


def run_to_target(trainer: IntegrationTrainer, target_stage_ticks: int, folder: Path,
                  checkpoint_every: int = 0, stop_after_updates: int | None = None) -> None:
    live_ticks = trainer.integration_config.live_ticks
    if target_stage_ticks < trainer.stage_ticks or target_stage_ticks % live_ticks:
        raise ValueError("target ticks must be an attainable complete live-chunk boundary")
    if checkpoint_every < 0 or (stop_after_updates is not None and stop_after_updates < 0):
        raise ValueError("checkpoint/stop intervals must be nonnegative")
    folder.mkdir(parents=True, exist_ok=True)
    completed = 0
    with (folder / "training.jsonl").open("a", encoding="utf-8") as stream:
        while trainer.stage_ticks < target_stage_ticks:
            if stop_after_updates is not None and completed >= stop_after_updates:
                break
            point = trainer.train_update()
            completed += 1
            if (trainer.stage_updates == 1 or trainer.stage_updates % trainer.config.log_every == 0
                    or trainer.stage_ticks == target_stage_ticks):
                record = {"event": "train", "mode": trainer.integration_config.mode, **point}
                encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
                stream.write(encoded + "\n")
                stream.flush()
                print(encoded, flush=True)
            if checkpoint_every and trainer.stage_updates % checkpoint_every == 0:
                trainer.save_checkpoint(folder / f"checkpoint_{trainer.stage_updates:06d}.pt")
                trainer.save_checkpoint(folder / "checkpoint.pt")
    write_run_artifacts(trainer, folder)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(ARMS), default="bp64")
    parser.add_argument("--ticks", type=int, default=65536)
    parser.add_argument("--live-ticks", type=int)
    parser.add_argument("--gradient-horizon", type=int)
    parser.add_argument("--duration", nargs=2, type=float)
    parser.add_argument("--gap", nargs=2, type=float)
    parser.add_argument("--local-lr", type=float, default=0.001)
    parser.add_argument("--local-tau", type=float, default=16.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.resume:
        trainer = IntegrationTrainer.from_checkpoint(args.resume, device=args.device)
    else:
        if args.output.exists():
            raise FileExistsError(f"output exists; explicitly resume its checkpoint: {args.output}")
        config = IntegrationConfig(mode=args.mode, live_ticks=args.live_ticks,
                    gradient_horizon=args.gradient_horizon,
                    duration_range=None if args.duration is None else tuple(args.duration),
                    gap_range=None if args.gap is None else tuple(args.gap),
                    local_lr=args.local_lr, local_tau=args.local_tau)
        trainer = IntegrationTrainer.from_base_checkpoint(args.from_checkpoint, config, device=args.device)
    run_to_target(trainer, args.ticks, args.output, args.checkpoint_every, args.stop_after_updates)


if __name__ == "__main__":
    main()
