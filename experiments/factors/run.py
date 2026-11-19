"""Sparse-B continuation crossing structural mode with an idle WAIT objective.

The inherited controller, live state, optimizer, clocks, history and RNG remain
continuous. Labels and idle masks enter the training loss and diagnostics only;
the unchanged model always chooses its full, unfiltered argmax action.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.continuity import run as continuity
from experiments.credit.run import CreditTrainer, file_sha256
from experiments.factors.structure import FactorConfig, FactorOperator, MODES
from experiments.rewiring.run import (
    RewireTrainer, _validate_continuation, optimizer_parameter_names,
)


FORMAT_VERSION = "factors-trainer-v1"
BUCKETS = ("idle", "active_wait", "legal_begin", "legal_finish", "disabled")
BUCKET_FIELDS = ("count", "weight_mass", "weighted_ce_sum", "target_nll_sum",
                 "actual_nonwait_count", "p_wrong_sum", "target_margin_sum", "wait_margin_sum")


def source_snapshots():
    folder = Path(__file__).parent
    return {name: (folder / name).read_text(encoding="utf-8") for name in
            ("run.py", "structure.py", "test_run.py", "test_structure.py")
            if (folder / name).is_file()}


def source_hashes(sources=None):
    return {name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in (source_snapshots() if sources is None else sources).items()}


def dependency_snapshots():
    root = Path(continuity.ROOT) / "experiments"
    names = ("rewiring/run.py", "rewiring/plasticity.py", "rewiring/test_run.py",
             "rewiring/test_plasticity.py", "credit/run.py", "credit/local.py",
             "integration/run.py", "continuity/run.py", "continuity/model.py", "continuity/world.py")
    return {name: (root / name).read_text(encoding="utf-8") for name in names}


def factor_loss(base, logits, targets, idle_mask, idle_lambda):
    """Return the original CE tensor unchanged at lambda zero (or empty idle).

    Idle is an externally captured *pre-action* mask. This function never
    infers idle from labels: active WAIT positions are not idle examples.
    """
    if idle_mask.shape != targets.shape or idle_mask.dtype != torch.bool:
        raise ValueError("idle mask must be boolean and match the target shape")
    if not math.isfinite(float(idle_lambda)) or idle_lambda < 0:
        raise ValueError("idle_lambda must be finite and nonnegative")
    base_loss = base._loss(logits, targets)
    if bool(idle_mask.any()):
        idle_nll = -F.log_softmax(logits, dim=-1)[..., 0][idle_mask].mean()
    else:
        idle_nll = logits.new_zeros(())
    loss = base_loss if idle_lambda == 0 or not bool(idle_mask.any()) else base_loss + idle_lambda * idle_nll
    return loss, base_loss, idle_nll


def empty_diagnostics():
    return {"base_weight_mass": 0., "base_weighted_ce_sum": 0., "idle_nll_sum": 0.,
            "idle_count": 0, "chunks": 0,
            "buckets": {name: {field: 0 if field in ("count", "actual_nonwait_count") else 0.
                                for field in BUCKET_FIELDS} for name in BUCKETS}}


@torch.no_grad()
def chunk_diagnostics(logits, targets, actions, idle_mask, active_mask, enabled_mask, loss_weights):
    """Raw sufficient statistics, with denominators kept separate per bucket."""
    z = logits.detach().double()
    y, a = targets.to(z.device), actions.to(z.device)
    idle, active, enabled = (value.to(z.device) for value in (idle_mask, active_mask, enabled_mask))
    logp = F.log_softmax(z, dim=-1)
    nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    weight = loss_weights.double()[y]
    weighted = weight * nll
    p_wrong = 1. - (-nll).exp()
    target_logit = z.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    alternatives = z.clone().scatter_(-1, y.unsqueeze(-1), -torch.inf).amax(-1)
    target_margin = target_logit - alternatives
    wait_margin = z[..., 0] - z[..., 1:].amax(-1)
    masks = {"idle": idle, "active_wait": enabled & active & y.eq(0),
             "legal_begin": enabled & y.gt(0) & y.lt(3),
             "legal_finish": enabled & y.ge(3), "disabled": ~enabled}
    if not torch.equal(sum(value.long() for value in masks.values()), torch.ones_like(y)):
        raise AssertionError("pre-action diagnostic buckets must partition all labels")
    result = empty_diagnostics()
    result.update(base_weight_mass=float(weight.sum()), base_weighted_ce_sum=float(weighted.sum()),
                  idle_nll_sum=float((-logp[..., 0])[idle].sum()), idle_count=int(idle.sum()), chunks=1)
    for name, mask in masks.items():
        result["buckets"][name] = {
            "count": int(mask.sum()), "weight_mass": float(weight[mask].sum()),
            "weighted_ce_sum": float(weighted[mask].sum()), "target_nll_sum": float(nll[mask].sum()),
            "actual_nonwait_count": int((a.ne(0) & mask).sum()), "p_wrong_sum": float(p_wrong[mask].sum()),
            "target_margin_sum": float(target_margin[mask].sum()), "wait_margin_sum": float(wait_margin[mask].sum())}
    return result


def add_diagnostics(total, chunk):
    for name in ("base_weight_mass", "base_weighted_ce_sum", "idle_nll_sum", "idle_count", "chunks"):
        total[name] += chunk[name]
    for name in BUCKETS:
        for field in BUCKET_FIELDS:
            total["buckets"][name][field] += chunk["buckets"][name][field]


def diagnostic_summary(raw):
    result = deepcopy(raw)
    result["base_weighted_ce"] = raw["base_weighted_ce_sum"] / max(raw["base_weight_mass"], 1e-300)
    result["idle_nll_mean"] = raw["idle_nll_sum"] / max(raw["idle_count"], 1)
    for name, values in result["buckets"].items():
        count, mass = max(values["count"], 1), max(values["weight_mass"], 1e-300)
        values.update(weighted_ce=values["weighted_ce_sum"] / mass,
                      target_nll_mean=values["target_nll_sum"] / count,
                      actual_nonwait_rate=values["actual_nonwait_count"] / count,
                      p_wrong_mean=values["p_wrong_sum"] / count,
                      target_margin_mean=values["target_margin_sum"] / count,
                      wait_margin_mean=values["wait_margin_sum"] / count)
    return result


class FactorTrainer(CreditTrainer):
    def __init__(self, inherited: RewireTrainer, config: FactorConfig, *, origin=None):
        if not isinstance(inherited, RewireTrainer) or not isinstance(config, FactorConfig):
            raise ValueError("factor continuation requires RewireTrainer and FactorConfig")
        if (not inherited.base._at_boundary or inherited.phase_returned
                or inherited.stage_ticks != inherited.phase_ticks
                or inherited.credit_config.mode != "bp64"
                or inherited.live_credit is not None or inherited.old_credit is not None
                or inherited.rewire_config.arm != "retire"):
            raise ValueError("factor continuation must branch at the complete sparse retire B boundary")
        if inherited.rewire_config.shrink_step != config.shrink_step:
            raise ValueError("factor shrink step must match the inherited retire rule")
        semantic = inherited.semantic_state()
        self.__dict__.update(inherited.__dict__)
        self.factor_config = deepcopy(config)
        self.factor_start_tick, self.factor_start_update = self.base.ticks, self.base.updates
        self.factor_start_stage_tick, self.factor_start_stage_update = self.stage_ticks, self.stage_updates
        core = semantic["credit"]["continuation"]
        handoff = {name + "_hash": continuity.state_digest(core[name]) for name in
                   ("model", "optimizer", "hidden", "world", "replay", "rng", "stats")}
        handoff.update(tick=self.base.ticks, update=self.base.updates, stage_ticks=self.stage_ticks,
                       stage_updates=self.stage_updates, action_hash=self.action_hash, data_hash=self.data_hash,
                       recurrent_nonzero=int(self.model.recurrent.detach().ne(0).sum()))
        self.factor_origin = {**deepcopy(origin or {}),
            "inherited_semantic_hash": continuity.state_digest(semantic),
            "inherited_operator_hash": continuity.state_digest(inherited.operator.state_dict()),
            "inherited_source_hashes": deepcopy(inherited.rewire_source_hashes),
            "inherited_dependency_hashes": deepcopy(inherited.rewire_dependency_hashes), "handoff": handoff}
        self.inherited_rewiring = {key: deepcopy(semantic[key]) for key in
            ("format", "rewire_config", "origin", "optimizer_parameter_names", "phase_ticks",
             "phase_returned", "return_regime", "structural_windows")}
        self.operator = FactorOperator(self.model.recurrent, config, inherited_operator=inherited.operator)
        self.structural_windows = []
        self.diagnostics = empty_diagnostics()
        self.factor_sources, self.factor_dependencies = source_snapshots(), dependency_snapshots()
        self.factor_source_hashes = source_hashes(self.factor_sources)
        self.factor_dependency_hashes = source_hashes(self.factor_dependencies)
        # Exactly the old _ensure_return_phase action: future NEW contracts only.
        CreditTrainer.change_regime(self, self.return_regime["duration_range"],
            self.return_regime["gap_range"], self.return_regime["name"])
        self.phase_returned = True

    @property
    def factor_ticks(self):
        return self.base.ticks - self.factor_start_tick

    @property
    def factor_updates(self):
        return self.base.updates - self.factor_start_update

    @classmethod
    def from_source_checkpoint(cls, path, config, device=None, strict_sources=True):
        path = Path(path).resolve()
        digest = file_sha256(path)
        inherited = RewireTrainer.from_checkpoint(path, device=device, strict_sources=strict_sources)
        if (inherited.model.width != 128 or inherited.stage_ticks != 65536
                or inherited.credit_config.chunk_ticks != 64
                or inherited.return_regime != {"duration_range": (1., 4.), "gap_range": (.5, 2.), "name": "rewiring-A2"}):
            raise ValueError("formal factors require the width-128, 65536-tick sparse B checkpoint")
        trainer = cls(inherited, config, origin={"checkpoint": str(path), "checkpoint_sha256": digest})
        if digest != file_sha256(path):
            raise AssertionError("source checkpoint bytes changed during loading")
        return trainer

    def _live_chunk(self):
        # The bp64 live loop is copied locally solely to capture pre-act masks.
        # Its observations, actions, base stats, work accounting and hash input
        # remain exactly those of CreditTrainer._live_chunk.
        base, ticks = self.base, self.credit_config.chunk_ticks
        observations, targets, actions, elapsed, outputs = [], [], [], [], []
        idle_masks, active_masks, enabled_masks = [], [], []
        start_tick = base.ticks
        for _ in range(ticks):
            _, reset = base._before_observe()
            if reset:
                raise AssertionError("factor continuation may not reset live hidden state")
            observation = continuity.cpu_tree(base.world.observe())
            target = continuity.cpu_tree(base.world.oracle("midpoint"))
            active = base.world.active.detach().clone().cpu()
            enabled = (torch.arange(base.world.n_jobs) < base.world.active_jobs).expand_as(active).clone()
            idle = enabled & ~active
            dt = float(base.world.elapsed)
            x, y = observation.to(base.device), target.to(base.device)
            output, base.h = base.model.step(x, base.h, dt)
            action = output.detach().argmax(-1).cpu()
            base.world.act(action)
            observations.append(observation)
            targets.append(target)
            actions.append(action)
            elapsed.append(dt)
            outputs.append(output)
            idle_masks.append(idle)
            active_masks.append(active)
            enabled_masks.append(enabled)
            base.ticks += 1
        logits, target = torch.stack(outputs), torch.stack(targets).to(base.device)
        idle = torch.stack(idle_masks).to(base.device)
        loss, base_loss, idle_nll = factor_loss(base, logits, target, idle, self.factor_config.idle_lambda)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite factor loss")
        base.stats.add(logits, target)
        gradients = self._bp_gradients(loss, ticks)
        diagnostic = chunk_diagnostics(logits, target, torch.stack(actions), idle,
            torch.stack(active_masks), torch.stack(enabled_masks), base.loss_weights)
        add_diagnostics(self.diagnostics, diagnostic)
        metrics = {"weighted_loss": float(loss.detach()), "base_weighted_ce": float(base_loss.detach()),
                   "idle_nll_mean": float(idle_nll.detach()),
                   "idle_penalty": self.factor_config.idle_lambda * float(idle_nll.detach()),
                   "diagnostics": diagnostic}
        base.h = base.h.detach()
        self.work["live_forward_ticks"] += ticks
        self.work["live_loss_ticks"] += ticks
        chunk = {"observations": torch.stack(observations), "targets": torch.stack(targets),
                 "actions": torch.stack(actions), "elapsed": tuple(elapsed),
                 "resets": (False,) * ticks, "start_tick": start_tick}
        return gradients, metrics, chunk

    def _apply_gradients(self, gradients):
        parameters = dict(self.model.named_parameters())
        if set(gradients) != set(parameters):
            raise ValueError("factor BPTT must cover every model parameter")
        for name, value in gradients.items():
            if (not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape
                    or value.device != parameters[name].device or not bool(torch.isfinite(value).all())):
                raise FloatingPointError(f"invalid gradient for {name}")
        before = {name: value.detach().double().clone() for name, value in parameters.items()}
        raw = gradients["recurrent"]
        update = CreditTrainer._apply_gradients(self, {**gradients, "recurrent": self.operator.before_gradient(raw)})
        group = self.optimizer.param_groups[0]
        event = self.operator.apply_after_optimizer(self.model.recurrent,
            self.optimizer.state[self.model.recurrent], raw, update["clip_scale"],
            group["lr"], group["betas"], group["eps"])
        squares = {name: (value.detach().double() - before[name]).square().sum()
                   for name, value in parameters.items()}
        for name, actual in (("recurrent_delta", float(squares["recurrent"].sqrt())),
                             ("all_delta", float(sum(squares.values()).sqrt()))):
            self.update_norms[name + "_sum"] += actual - update[name]
            update[name] = actual
        event = {**event, "stage_ticks": self.stage_ticks, "stage_update": self.stage_updates + 1,
                 "factor_ticks": self.factor_ticks, "factor_update": self.factor_updates + 1,
                 "recurrent_weight_l2": float(self.model.recurrent.detach().double().norm())}
        if self.operator.updates_in_window == 0:
            self.structural_windows.append(deepcopy(event))
        update.update(structure=event, potential_recurrent_gradient_norm=float(raw.double().norm()))
        return update

    def train_update(self):
        point = CreditTrainer.train_update(self)
        point.update(factor_ticks=self.factor_ticks, factor_update=self.factor_updates,
                     factor_mode=self.factor_config.mode, idle_lambda=self.factor_config.idle_lambda)
        self.base.trace[-1] = deepcopy(point)
        if self.base.curve and self.base.curve[-1]["update"] == point["update"]:
            self.base.curve[-1].update(deepcopy(point))
        return point

    def semantic_state(self):
        return {"format": FORMAT_VERSION, "factor_config": asdict(self.factor_config),
            "factor_origin": deepcopy(self.factor_origin),
            "factor_start_tick": self.factor_start_tick, "factor_start_update": self.factor_start_update,
            "factor_start_stage_tick": self.factor_start_stage_tick,
            "factor_start_stage_update": self.factor_start_stage_update,
            "factor_ticks": self.factor_ticks, "factor_updates": self.factor_updates,
            "operator": self.operator.state_dict(), "structural_windows": deepcopy(self.structural_windows),
            "diagnostics": deepcopy(self.diagnostics), "inherited_rewiring": deepcopy(self.inherited_rewiring),
            "credit": CreditTrainer.semantic_state(self)}

    def state_dict(self):
        state = self.semantic_state()
        state["credit"]["continuation"] = self.base.state_dict()
        state["credit"].update(sources=deepcopy(self.sources), source_hashes=deepcopy(self.source_hashes),
            dependencies=deepcopy(self.dependencies), dependency_hashes=deepcopy(self.dependency_hashes))
        state.update(sources=deepcopy(self.factor_sources), source_hashes=deepcopy(self.factor_source_hashes),
            dependencies=deepcopy(self.factor_dependencies), dependency_hashes=deepcopy(self.factor_dependency_hashes))
        return state

    @classmethod
    def from_checkpoint(cls, path, device=None, strict_sources=True):
        return cls._from_state(torch.load(Path(path), map_location="cpu", weights_only=False), device, strict_sources)

    @classmethod
    def _from_state(cls, state, device=None, strict_sources=True):
        if state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported factor checkpoint")
        if strict_sources:
            for key, current in (("sources", source_snapshots()), ("dependencies", dependency_snapshots())):
                hash_key = "source_hashes" if key == "sources" else "dependency_hashes"
                if source_hashes(state[key]) != state[hash_key] or source_hashes(current) != state[hash_key]:
                    raise ValueError(f"factor checkpoint {key} mismatch; use its saved snapshots")
        config = FactorConfig(**state["factor_config"])
        history = state["inherited_rewiring"]
        _validate_continuation(state["credit"]["continuation"], history["optimizer_parameter_names"])
        credit = CreditTrainer._from_state(state["credit"], device, strict_sources)
        if credit.credit_config.mode != "bp64" or credit.live_credit is not None or credit.old_credit is not None:
            raise ValueError("factor checkpoint requires exact bp64 credit")
        trainer = cls.__new__(cls)
        trainer.__dict__.update(credit.__dict__)
        trainer.factor_config = config
        for name in ("factor_origin", "factor_start_tick", "factor_start_update", "factor_start_stage_tick",
                     "factor_start_stage_update", "structural_windows", "diagnostics", "inherited_rewiring"):
            setattr(trainer, name, deepcopy(state[name]))
        trainer.operator = FactorOperator.from_state(trainer.model.recurrent, state["operator"])
        if trainer.operator.config != config:
            raise ValueError("factor operator/trainer configuration mismatch")
        for name in ("factor_start_tick", "factor_start_update", "factor_start_stage_tick", "factor_start_stage_update"):
            value = getattr(trainer, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("invalid factor offset")
        if (trainer.factor_ticks != state["factor_ticks"] or trainer.factor_updates != state["factor_updates"]
                or trainer.factor_ticks < 0 or trainer.factor_updates < 0
                or trainer.factor_ticks != trainer.factor_updates * trainer.credit_config.chunk_ticks
                or trainer.stage_ticks != trainer.factor_start_stage_tick + trainer.factor_ticks
                or trainer.stage_updates != trainer.factor_start_stage_update + trainer.factor_updates
                or trainer.operator.updates != trainer.stage_updates
                or trainer.operator.factor_updates != trainer.factor_updates
                or trainer.diagnostics["chunks"] != trainer.factor_updates):
            raise ValueError("inconsistent factor checkpoint progress")
        returns = [event for event in trainer.regime_changes if event["regime"]["name"] == history["return_regime"]["name"]]
        if len(returns) != 1 or returns[0]["stage_tick"] != trainer.factor_start_stage_tick:
            raise ValueError("factor return-regime history mismatch")
        trainer.phase_ticks, trainer.phase_returned = history["phase_ticks"], True
        trainer.return_regime = deepcopy(history["return_regime"])
        for attr, key in (("factor_sources", "sources"), ("factor_source_hashes", "source_hashes"),
                          ("factor_dependencies", "dependencies"), ("factor_dependency_hashes", "dependency_hashes")):
            setattr(trainer, attr, deepcopy(state[key]))
        return trainer

    def report(self):
        result = CreditTrainer.report(self)
        trace, curve = result.pop("trace"), result.pop("curve")
        result.update(format=FORMAT_VERSION, factor_config=asdict(self.factor_config),
            factor_origin=deepcopy(self.factor_origin), factor_start_tick=self.factor_start_tick,
            factor_start_update=self.factor_start_update, factor_start_stage_tick=self.factor_start_stage_tick,
            factor_start_stage_update=self.factor_start_stage_update, factor_ticks=self.factor_ticks,
            factor_updates=self.factor_updates, structure=self.operator.summary(),
            structural_windows=deepcopy(self.structural_windows), diagnostics=diagnostic_summary(self.diagnostics),
            trace_points=len(trace), curve_points=len(curve), last_update=deepcopy(trace[-1]) if self.factor_updates else None,
            inherited_rewiring=deepcopy(self.inherited_rewiring), factor_source_hashes=deepcopy(self.factor_source_hashes),
            factor_dependency_hashes=deepcopy(self.factor_dependency_hashes),
            objective="Original weighted CE (WAIT weight 0.1) + idle_lambda * mean idle pre-action -log P(WAIT). Empty idle contributes zero.",
            idle_definition="Enabled job with no active contract after observe and oracle, cloned before the current argmax action.",
            action_policy="Unchanged AgentModel and full unfiltered argmax; private labels and idle masks enter loss and diagnostics only.",
            continuity="Inherited sparse B model, hidden state, world, Adam, shadow state, RNG and global stage counters persist into A2.")
        return result


def write_artifacts(trainer, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(folder / "checkpoint.pt")
    continuity.atomic_json(folder / "report.json", trainer.report())
    for prefix, sources in (("factors", trainer.factor_sources), ("dependencies", trainer.factor_dependencies),
                            ("credit", trainer.sources), ("continuity", trainer.base.sources)):
        for name, source in sources.items():
            path = folder / "sources" / prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")


def run_to_target(trainer, target_ticks, folder, checkpoint_every=256, stop_after_updates=None):
    if (isinstance(target_ticks, bool) or not isinstance(target_ticks, int)
            or target_ticks < trainer.factor_ticks or target_ticks % trainer.credit_config.chunk_ticks):
        raise ValueError("target must be an attainable complete factor chunk boundary")
    if checkpoint_every < 0 or (stop_after_updates is not None and stop_after_updates < 0):
        raise ValueError("intervals must be nonnegative")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    completed = 0
    with (folder / "training.jsonl").open("a", encoding="utf-8") as stream:
        while trainer.factor_ticks < target_ticks:
            if stop_after_updates is not None and completed >= stop_after_updates:
                break
            point = trainer.train_update()
            completed += 1
            if trainer.factor_updates == 1 or trainer.factor_updates % trainer.config.log_every == 0 or trainer.factor_ticks == target_ticks:
                encoded = json.dumps({"event": "train", **point}, allow_nan=False)
                stream.write(encoded + "\n")
                stream.flush()
                print(encoded, flush=True)
            if checkpoint_every and trainer.factor_updates % checkpoint_every == 0:
                trainer.save_checkpoint(folder / f"checkpoint_{trainer.factor_updates:06d}.pt")
                trainer.save_checkpoint(folder / "checkpoint.pt")
    write_artifacts(trainer, folder)
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=tuple(MODES), default="dense")
    parser.add_argument("--idle-lambda", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-ticks", type=int, default=65536)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.resume:
        trainer = FactorTrainer.from_checkpoint(args.resume, device=args.device)
    else:
        if args.output.exists():
            raise FileExistsError("output exists; explicitly resume its complete checkpoint")
        trainer = FactorTrainer.from_source_checkpoint(args.source,
            FactorConfig(mode=args.mode, idle_lambda=args.idle_lambda), device=args.device)
    return run_to_target(trainer, args.target_ticks, args.output, args.checkpoint_every, args.stop_after_updates)


if __name__ == "__main__":
    main()
