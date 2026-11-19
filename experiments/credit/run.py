"""Full-state continuation with BPTT or approximate forward eligibility credit.

Every learned mode makes one inherited-Adam update per live chunk. Replay
sources share the pre-update weights and contribute half of the independently
normalized current/old gradient. Old hidden states never replace live state.
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
from experiments.integration.run import _restore_continuation


FORMAT_VERSION = "credit-trainer-v1"


@dataclass(frozen=True)
class Mode:
    method: str
    replay: bool = False
    feedback: str = "symmetric"
    reset: str = "none"
    shuffle: bool = False


MODES = {
    "frozen": Mode("frozen"),
    "readout_only": Mode("readout"),
    "bp64": Mode("bp"),
    "bp_replay": Mode("bp", replay=True),
    "diag_sym": Mode("diag"),
    "diag_sym_replay": Mode("diag", replay=True),
    "diag_random": Mode("diag", feedback="random"),
    "diag_random_replay": Mode("diag", replay=True, feedback="random"),
    "diag_reset64": Mode("diag", reset="block"),
    "diag_instant": Mode("diag", reset="tick"),
    "diag_shuffle": Mode("diag", shuffle=True),
}


@dataclass(frozen=True)
class CreditConfig:
    mode: str = "bp64"
    chunk_ticks: int = 64
    duration_range: tuple[float, float] = (4., 8.)
    gap_range: tuple[float, float] = (4., 12.)
    regime_name: str = "credit"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown credit mode")
        if isinstance(self.chunk_ticks, bool) or not isinstance(self.chunk_ticks, int) or self.chunk_ticks < 1:
            raise ValueError("chunk_ticks must be a positive integer")
        for name in ("duration_range", "gap_range"):
            value = tuple(map(float, getattr(self, name)))
            if len(value) != 2 or any(not math.isfinite(x) for x in value) or value[0] <= 0 or value[1] < value[0]:
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not isinstance(self.regime_name, str) or not self.regime_name:
            raise ValueError("regime_name must be nonempty")


def source_snapshots():
    folder = Path(__file__).parent
    return {name: (folder / name).read_text(encoding="utf-8") for name in
            ("run.py", "local.py", "test_run.py", "test_local.py") if (folder / name).is_file()}


def source_hashes(sources=None):
    return {name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in (source_snapshots() if sources is None else sources).items()}


def dependency_snapshots():
    folder = Path(continuity.ROOT) / "experiments/integration"
    return {"integration/run.py": (folder / "run.py").read_text(encoding="utf-8")}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CreditTrainer:
    def __init__(self, base: continuity.Trainer, config: CreditConfig, *, origin=None):
        if not base._at_boundary or base.stage != "A":
            raise ValueError("credit stage must branch from a complete A boundary")
        if base.variant.kind != "learned" or base.variant.policy != "online" or base.variant.reset_every:
            raise ValueError("credit requires a continuous learned own-action source")
        self.base, self.credit_config, self.mode = base, deepcopy(config), MODES[config.mode]
        if self.mode.replay and (not base.replay.blocks or any(any(x["resets"]) for x in base.replay.blocks)):
            raise ValueError("replay needs complete continuous A histories")
        if self.mode.replay and base.replay.block_ticks - base.config.replay_burnin != config.chunk_ticks:
            raise ValueError("current and old supervised suffixes must have the same tick budget")
        self.origin = deepcopy(origin or {})
        self.origin.update(semantic_hash=continuity.state_digest(base.semantic_state()),
                           model_hash=continuity.state_digest(base.model.state_dict()),
                           optimizer_hash=continuity.state_digest(base.optimizer.state_dict()),
                           hidden_hash=continuity.state_digest(base.h),
                           world_hash=continuity.state_digest(base.world.state_dict()),
                           replay_hash=continuity.state_digest(base.replay.state_dict()),
                           rng_hash=continuity.state_digest(continuity.rng_state()),
                           tick=base.ticks, update=base.updates,
                           inherited_source_hashes=deepcopy(base.source_hashes))
        self.sources, self.dependencies = source_snapshots(), dependency_snapshots()
        self.source_hashes = source_hashes(self.sources)
        self.dependency_hashes = source_hashes(self.dependencies)
        self.sampler_rng = random.Random()
        self.sampler_rng.setstate(base.replay.rng.getstate())
        self.live_credit = self._make_credit() if self.mode.method == "diag" else None
        self.old_credit = self._make_credit() if self.mode.method == "diag" and self.mode.replay else None
        self.work = {name: 0 for name in (
            "live_forward_ticks", "old_forward_ticks", "old_warmup_ticks", "live_loss_ticks",
            "old_loss_ticks", "optimizer_steps", "backward_calls", "bp_graph_ticks",
            "manual_gradient_passes", "old_samples", "live_trace_resets", "old_trace_resets")}
        self.update_norms = {name: 0. for name in (
            "recurrent_delta_sum", "encoder_delta_sum", "head_delta_sum", "all_delta_sum",
            "gradient_norm_sum", "clip_scale_sum")}
        base.begin_adaptation("frozen" if self.mode.method == "frozen" else "current", config.duration_range)
        base.world.set_regime(gap_range=config.gap_range, name=config.regime_name)
        base.adaptation_mode = config.mode
        self.regime_changes = [self._regime_record("initial credit regime")]

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def stage_ticks(self):
        return self.base.ticks - self.base.phase_start_tick

    def _make_credit(self):
        from experiments.credit.local import ForwardEligibility, LocalConfig

        config = LocalConfig(feedback=self.mode.feedback, shuffle=self.mode.shuffle,
                             seed=continuity.seed_for(self.base.config.seed, "credit-feedback"),
                             n_jobs=self.base.model.n_jobs, n_actions=self.base.model.n_actions,
                             wait_weight=self.base.config.wait_weight)
        updater = ForwardEligibility(self.base.config.batch_size, self.base.model.width,
                                     self.base.model.input_size, config,
                                     device=self.base.device, dtype=self.base.h.dtype)
        updater.initialize_feedback_(self.base.model.head.weight)
        return updater

    def _regime_record(self, reason):
        return {"tick": self.base.ticks, "stage_tick": self.stage_ticks,
                "world_tick": self.base.world.tick, "now": float(self.base.world.now),
                "regime_id": self.base.world.regime_id, "regime": asdict(self.base.world.regime),
                "reason": reason}

    def change_regime(self, duration, gap, name):
        """Change future contracts only; preserve all clocks and learning state."""
        if not self.base._at_boundary:
            raise RuntimeError("regime changes require a complete optimizer boundary")
        if not isinstance(name, str) or not name:
            raise ValueError("regime name must be nonempty")
        self.base.world.set_regime(duration_range=duration, gap_range=gap, name=name)
        self.regime_changes.append(self._regime_record("explicit boundary regime change"))

    @classmethod
    def from_base_checkpoint(cls, path, config, device=None, strict_sources=True):
        path = Path(path).resolve()
        base = continuity.Trainer.from_checkpoint(path, device=device, strict_sources=strict_sources)
        return cls(base, config, origin={"checkpoint": str(path), "checkpoint_sha256": file_sha256(path)})

    def _bp_gradients(self, loss, graph_ticks):
        self.base.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in self.base.model.named_parameters()}
        self.base.optimizer.zero_grad(set_to_none=True)
        self.work["backward_calls"] += 1
        self.work["bp_graph_ticks"] += graph_ticks
        return gradients

    def _live_chunk(self):
        base, method, ticks = self.base, self.mode.method, self.credit_config.chunk_ticks
        observations, targets, actions, elapsed, outputs = [], [], [], [], []
        start_tick = base.ticks
        head_gradients = None
        head_mass = torch.zeros((), device=base.device, dtype=torch.float64)
        head_loss = torch.zeros_like(head_mass)
        if self.live_credit is not None:
            self.live_credit.clear_gradients()
            if self.mode.reset == "block":
                self.live_credit.reset_eligibility_()
                self.work["live_trace_resets"] += 1
        context = nullcontext() if method == "bp" else torch.no_grad()
        with context:
            for _ in range(ticks):
                _, reset = base._before_observe()
                if reset:
                    raise AssertionError("credit may not reset the live hidden state")
                observation = continuity.cpu_tree(base.world.observe())
                target = continuity.cpu_tree(base.world.oracle("midpoint"))
                dt, pre_h = float(base.world.elapsed), base.h
                x, y = observation.to(base.device), target.to(base.device)
                output, base.h = base.model.step(x, base.h, dt)
                action = output.detach().argmax(-1).cpu()
                base.world.act(action)
                if self.live_credit is not None:
                    if self.mode.reset == "tick":
                        self.live_credit.reset_eligibility_()
                        self.work["live_trace_resets"] += 1
                    self.live_credit.observe(base.model, x, pre_h.detach(), base.h.detach(), dt,
                                             output.detach(), y, accumulate=True)
                elif method == "readout":
                    from experiments.credit.local import analytic_head_gradients
                    raw, metrics = analytic_head_gradients(output, base.h, y, base.config.wait_weight)
                    if head_gradients is None:
                        head_gradients = {name: value.clone() for name, value in raw.items()}
                    else:
                        for name, value in raw.items():
                            head_gradients[name].add_(value)
                    head_mass.add_(metrics["weight_mass"])
                    head_loss.add_(metrics["weighted_loss_sum"])
                observations.append(observation)
                targets.append(target)
                actions.append(action)
                elapsed.append(dt)
                outputs.append(output)
                base.ticks += 1
        logits, target = torch.stack(outputs), torch.stack(targets).to(base.device)
        loss = base._loss(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite live loss")
        base.stats.add(logits, target)
        if method == "bp":
            gradients = self._bp_gradients(loss, ticks)
            metrics = {"weighted_loss": float(loss.detach())}
        elif method == "diag":
            gradients, metrics = self.live_credit.normalized_gradients()
            self.work["manual_gradient_passes"] += 1
        elif method == "readout":
            gradients = {name: value / head_mass.to(value.dtype) for name, value in head_gradients.items()}
            metrics = {"weight_mass": float(head_mass), "weighted_loss_sum": float(head_loss),
                       "weighted_loss": float(head_loss / head_mass)}
            self.work["manual_gradient_passes"] += 1
        else:
            gradients, metrics = {}, {"weighted_loss": float(loss.detach())}
        base.h = base.h.detach()
        self.work["live_forward_ticks"] += ticks
        self.work["live_loss_ticks"] += ticks if method != "frozen" else 0
        chunk = {"observations": torch.stack(observations), "targets": torch.stack(targets),
                 "actions": torch.stack(actions), "elapsed": tuple(elapsed),
                 "resets": (False,) * ticks, "start_tick": start_tick}
        return gradients, metrics, chunk

    def _old_gradients(self):
        base = self.base
        index = self.sampler_rng.randrange(len(base.replay.blocks))
        block = base.replay.blocks[index]
        warmup = base.config.replay_burnin
        if self.old_credit is not None:
            self.old_credit.clear_gradients()
            self.old_credit.reset_eligibility_()
            self.work["old_trace_resets"] += 1
        h, outputs = block["start_h"].to(base.device), []
        context = nullcontext() if self.mode.method == "bp" else torch.no_grad()
        with context:
            for tick, (observation, dt, reset) in enumerate(zip(block["observations"], block["elapsed"], block["resets"])):
                if reset:
                    raise ValueError("old experiences must be continuous")
                x, pre_h = observation.to(base.device), h
                output, h = base.model.step(x, h, dt)
                if self.old_credit is not None:
                    target = block["targets"][tick].to(base.device) if tick >= warmup else None
                    self.old_credit.observe(base.model, x, pre_h, h, dt, output, target,
                                            accumulate=tick >= warmup)
                if tick >= warmup:
                    outputs.append(output)
            target = block["targets"][warmup:].to(base.device)
            loss = base._loss(torch.stack(outputs), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite historical loss")
        if self.mode.method == "bp":
            gradients = self._bp_gradients(loss, len(block["elapsed"]))
            metrics = {"weighted_loss": float(loss.detach())}
        else:
            gradients, metrics = self.old_credit.normalized_gradients()
            self.work["manual_gradient_passes"] += 1
        self.work["old_samples"] += 1
        self.work["old_forward_ticks"] += len(block["elapsed"])
        self.work["old_warmup_ticks"] += warmup
        self.work["old_loss_ticks"] += len(block["elapsed"]) - warmup
        return gradients, {**metrics, "block_index": index, "start_tick": block["start_tick"]}

    def _apply_gradients(self, gradients):
        base = self.base
        base.optimizer.zero_grad(set_to_none=True)
        parameters = dict(base.model.named_parameters())
        before = {name: value.detach().clone() for name, value in parameters.items()}
        raw_norms = {}
        for name, value in parameters.items():
            # Missing core gradients must be None, so inherited Adam moments do
            # not move the fixed reservoir in the readout-only control.
            value.grad = None if name not in gradients else gradients[name].detach().clone()
            if value.grad is not None:
                if value.grad.shape != value.shape or not bool(torch.isfinite(value.grad).all()):
                    raise FloatingPointError(f"invalid gradient for {name}")
                raw_norms[name] = float(value.grad.double().norm())
        norm = float(torch.nn.utils.clip_grad_norm_(base.model.parameters(), base.config.clip, error_if_nonfinite=True))
        base.optimizer.step()
        base.gradient_updates += 1
        self.work["optimizer_steps"] += 1
        with torch.no_grad():
            squares = {name: (value.detach().double() - before[name].double()).square().sum()
                       for name, value in parameters.items()}
            deltas = {"recurrent_delta": float(squares["recurrent"].sqrt()),
                      "encoder_delta": float(squares["encoder.weight"].sqrt()),
                      "head_delta": float((squares["head.weight"] + squares["head.bias"]).sqrt()),
                      "all_delta": float(sum(squares.values()).sqrt())}
        clip_scale = min(1., base.config.clip / (norm + 1e-6))
        for name, value in deltas.items():
            self.update_norms[name + "_sum"] += value
        self.update_norms["gradient_norm_sum"] += norm
        self.update_norms["clip_scale_sum"] += clip_scale
        base.optimizer.zero_grad(set_to_none=True)
        return {**deltas, "gradient_norm": norm, "clip_scale": clip_scale, "raw_parameter_gradient_norms": raw_norms}

    def train_update(self):
        base = self.base
        if not base._at_boundary:
            raise RuntimeError("not at a complete credit boundary")
        base._at_boundary = False
        started = time.perf_counter()
        base.model.train()
        base.optimizer.zero_grad(set_to_none=True)
        before_updates = base.gradient_updates
        gradients, live_metrics, chunk = self._live_chunk()
        live_hash = continuity.state_digest(base.h)
        old_metrics = None
        if self.mode.replay:
            old_gradients, old_metrics = self._old_gradients()
            gradients = {name: .5 * value + .5 * old_gradients[name] for name, value in gradients.items()}
        update = None if self.mode.method == "frozen" else self._apply_gradients(gradients)
        if continuity.state_digest(base.h) != live_hash:
            raise AssertionError("historical learning replaced live hidden state")
        for updater in (self.live_credit, self.old_credit):
            if updater is not None:
                updater.clear_gradients()
        base.updates += 1
        action_chunk_hash = continuity.state_digest({"actions": chunk["actions"]})
        data_chunk_hash = continuity.state_digest({name: chunk[name] for name in
                              ("observations", "targets", "elapsed", "resets", "start_tick")})
        base.action_hash = hashlib.sha256(bytes.fromhex(base.action_hash) + bytes.fromhex(action_chunk_hash)).hexdigest()
        base.data_hash = hashlib.sha256(bytes.fromhex(base.data_hash) + bytes.fromhex(data_chunk_hash)).hexdigest()
        point = {"update": base.updates, "stage_update": base.stage_updates, "stage": "credit",
                 "tick": base.ticks, "stage_ticks": self.stage_ticks, "world_tick": base.world.tick,
                 "now": float(base.world.now), "dt": float(base.world.dt),
                 "loss": live_metrics["weighted_loss"], "live": live_metrics, "old": old_metrics,
                 "parameter_update": update, "gradient_updates": base.gradient_updates,
                 "optimizer_steps_this_chunk": base.gradient_updates - before_updates,
                 "work": deepcopy(self.work), "update_norms": deepcopy(self.update_norms),
                 "live_hidden_hash": live_hash, "action_hash": base.action_hash, "data_hash": base.data_hash,
                 "action_chunk_hash": action_chunk_hash, "data_chunk_hash": data_chunk_hash,
                 "event_hash": base.world.event_hash,
                 "actual_action_counts": torch.bincount(chunk["actions"].reshape(-1), minlength=base.model.n_actions).tolist(),
                 "target_action_counts": torch.bincount(chunk["targets"].reshape(-1), minlength=base.model.n_actions).tolist()}
        base.trace.append(deepcopy(point))
        base._at_boundary = True
        continuity.synchronize(base.device)
        base.seconds += time.perf_counter() - started
        if base.stage_updates == 1 or base.stage_updates % base.config.log_every == 0:
            base.curve.append({**deepcopy(point), "seconds": base.seconds, "world_metrics": base.world.metrics()})
        return point

    def run_updates(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        return [self.train_update() for _ in range(count)]

    def semantic_state(self):
        return {"format": FORMAT_VERSION, "continuation": self.base.semantic_state(),
                "credit_config": asdict(self.credit_config), "origin": deepcopy(self.origin),
                "sampler_rng": self.sampler_rng.getstate(), "regime_changes": deepcopy(self.regime_changes),
                "live_credit": None if self.live_credit is None else self.live_credit.state_dict(),
                "old_credit": None if self.old_credit is None else self.old_credit.state_dict(),
                "work": deepcopy(self.work), "update_norms": deepcopy(self.update_norms)}

    def state_dict(self):
        state = self.semantic_state()
        state["continuation"] = self.base.state_dict()
        state.update(sources=deepcopy(self.sources), source_hashes=deepcopy(self.source_hashes),
                     dependencies=deepcopy(self.dependencies), dependency_hashes=deepcopy(self.dependency_hashes))
        return state

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, device=None, strict_sources=True):
        return cls._from_state(torch.load(Path(path), map_location="cpu", weights_only=False), device, strict_sources)

    @classmethod
    def _from_state(cls, state, device=None, strict_sources=True):
        if state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported credit checkpoint format")
        if strict_sources:
            for name in ("run.py", "local.py"):
                if (state["source_hashes"].get(name) != source_hashes().get(name)
                        or source_hashes(state["sources"]).get(name) != state["source_hashes"].get(name)):
                    raise ValueError(f"credit checkpoint source mismatch: {name}")
            if (source_hashes(state["dependencies"]) != state["dependency_hashes"]
                    or source_hashes(dependency_snapshots()) != state["dependency_hashes"]):
                raise ValueError("credit dependency source mismatch")
        trainer = cls.__new__(cls)
        trainer.base = _restore_continuation(state["continuation"], device, strict_sources)
        trainer.credit_config = CreditConfig(**state["credit_config"])
        trainer.mode = MODES[trainer.credit_config.mode]
        for name in ("origin", "regime_changes", "work", "update_norms", "sources", "source_hashes",
                     "dependencies", "dependency_hashes"):
            setattr(trainer, name, deepcopy(state[name]))
        trainer.sampler_rng = random.Random()
        trainer.sampler_rng.setstate(state["sampler_rng"])
        for name in ("live_credit", "old_credit"):
            updater = trainer._make_credit() if state[name] is not None else None
            if updater is not None:
                updater.load_state_dict(state[name])
            setattr(trainer, name, updater)
        return trainer

    def report(self):
        report = self.base.report()
        report.update(format=FORMAT_VERSION, credit_config=asdict(self.credit_config), mode=asdict(self.mode),
                      stage_ticks=self.stage_ticks, origin=deepcopy(self.origin), work=deepcopy(self.work),
                      update_norms=deepcopy(self.update_norms), regime_changes=deepcopy(self.regime_changes),
                      credit_source_hashes=deepcopy(self.source_hashes), dependency_hashes=deepcopy(self.dependency_hashes),
                      live_credit=None if self.live_credit is None else self.live_credit.summary(self.base.model),
                      old_credit=None if self.old_credit is None else self.old_credit.summary(self.base.model),
                      effective_updated_parameters=(0 if self.mode.method == "frozen" else
                          sum(p.numel() for p in self.base.model.head.parameters()) if self.mode.method == "readout" else
                          sum(p.numel() for p in self.base.model.parameters())),
                      learning=("Weights frozen; labels used only for reporting." if self.mode.method == "frozen" else
                                "Analytic supervised readout gradient; pretrained encoder and recurrent reservoir fixed, not a spiking LSM." if self.mode.method == "readout" else
                                "Supervised BPTT with causal labels on own history." if self.mode.method == "bp" else
                                "Supervised approximate forward temporal derivatives with analytic readout gradients; no backward-time graph."),
                      optimizer_semantics=(f"{0 if self.mode.method == 'frozen' else 1} inherited Adam step per "
                                           f"{self.credit_config.chunk_ticks} live ticks; shared inherited learning rate and joint clipping."),
                      replay_semantics=("Current and old histories use identical pre-update weights. Independently normalized gradients "
                                        "are averaged equally before one joint clip and Adam step. Old BP graph spans all historical ticks; "
                                        "only its final supervised suffix has loss. Old eligibility starts zero and evolves through warm-up."
                                        if self.mode.replay else "No historical pass."),
                      budget_note=("All learned modes take one Adam step per live chunk. Replay adds one full historical forward "
                                   "and one independently normalized supervised suffix. Backward calls, graph ticks, reconstructed "
                                   "flows, trace storage and effective updated parameters are reported separately; FLOPs are unequal."),
                      checkpoint_boundary="after a complete live/history chunk and optimizer boundary; live eligibility is retained",
                      continuity="Live world, hidden values, Adam and live eligibility persist across updates and explicit regime changes.")
        return report


def load_checkpoint(path, device=None, strict_sources=True):
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("format") == FORMAT_VERSION:
        return CreditTrainer._from_state(state, device, strict_sources)
    if state.get("format") == continuity.FORMAT_VERSION:
        return _restore_continuation(state, device, strict_sources)
    raise ValueError("unsupported credit/source checkpoint format")


def write_run_artifacts(trainer, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(folder / "checkpoint.pt")
    continuity.atomic_json(folder / "report.json", trainer.report())
    for prefix, sources in (("credit", trainer.sources), ("continuity", trainer.base.sources),
                            ("dependencies", trainer.dependencies)):
        for name, source in sources.items():
            path = folder / "sources" / prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")


def run_to_target(trainer, target_ticks, folder, checkpoint_every=256, stop_after_updates=None):
    folder = Path(folder)
    if target_ticks < trainer.stage_ticks or target_ticks % trainer.credit_config.chunk_ticks:
        raise ValueError("target must be an attainable complete chunk boundary")
    if checkpoint_every < 0 or (stop_after_updates is not None and stop_after_updates < 0):
        raise ValueError("intervals must be nonnegative")
    folder.mkdir(parents=True, exist_ok=True)
    completed = 0
    with (folder / "training.jsonl").open("a", encoding="utf-8") as stream:
        while trainer.stage_ticks < target_ticks:
            if stop_after_updates is not None and completed >= stop_after_updates:
                break
            point = trainer.train_update()
            completed += 1
            if trainer.stage_updates == 1 or trainer.stage_updates % trainer.config.log_every == 0 or trainer.stage_ticks == target_ticks:
                encoded = json.dumps({"event": "train", "mode": trainer.credit_config.mode, **point}, allow_nan=False)
                stream.write(encoded + "\n")
                stream.flush()
                print(encoded, flush=True)
            if checkpoint_every and trainer.stage_updates % checkpoint_every == 0:
                trainer.save_checkpoint(folder / f"checkpoint_{trainer.stage_updates:06d}.pt")
                trainer.save_checkpoint(folder / "checkpoint.pt")
    write_run_artifacts(trainer, folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=tuple(MODES), default="bp64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=65536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.resume:
        trainer = CreditTrainer.from_checkpoint(args.resume, device=args.device)
    else:
        if args.output.exists():
            raise FileExistsError("output exists; explicitly resume its complete checkpoint")
        trainer = CreditTrainer.from_base_checkpoint(args.from_checkpoint, CreditConfig(mode=args.mode), device=args.device)
    run_to_target(trainer, args.ticks, args.output, args.checkpoint_every, args.stop_after_updates)


if __name__ == "__main__":
    main()
