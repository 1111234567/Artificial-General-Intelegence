"""Continue the unchanged recurrent controller with gradual connection birth/death.

The inherited live world, hidden state and Adam remain in place. Full BPTT
sees the ordinary dense recurrent parameter, whose dormant entries are exact
zeros. Only the live Adam gradient is masked; potential gradients feed the
separate structural proposal state.
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

from experiments.continuity import run as continuity
from experiments.continuity.model import AgentModel
from experiments.credit.run import CreditConfig, CreditTrainer, file_sha256
from experiments.rewiring.plasticity import RewireConfig, RewireOperator


FORMAT_VERSION = "rewiring-trainer-v1"
ARMS = ("dense", "retire", "learned", "random")
PARAMETER_NAMES = ["recurrent", "encoder.weight", "head.weight", "head.bias"]


def source_snapshots():
    folder = Path(__file__).parent
    return {name: (folder / name).read_text(encoding="utf-8") for name in
            ("run.py", "plasticity.py", "test_run.py", "test_plasticity.py")
            if (folder / name).is_file()}


def source_hashes(sources=None):
    return {name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in (source_snapshots() if sources is None else sources).items()}


def dependency_snapshots():
    root = Path(continuity.ROOT) / "experiments"
    names = ("credit/run.py", "credit/local.py", "integration/run.py",
             "continuity/run.py", "continuity/model.py", "continuity/world.py")
    return {name: (root / name).read_text(encoding="utf-8") for name in names}


def optimizer_parameter_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        groups = [[names[id(parameter)] for parameter in group["params"]]
                  for group in optimizer.param_groups]
    except KeyError as error:
        raise ValueError("optimizer contains a parameter absent from the model") from error
    if groups != [PARAMETER_NAMES]:
        raise ValueError("rewiring requires the inherited named Adam parameter order")
    return groups


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_adam_group(group):
    if (group.get("weight_decay", 0.) != 0. or group.get("amsgrad", False)
            or group.get("maximize", False)):
        raise ValueError("rewiring requires ordinary Adam without weight decay or AMSGrad")
    lr, eps = float(group["lr"]), float(group["eps"])
    betas = tuple(map(float, group["betas"]))
    if (not math.isfinite(lr) or lr <= 0 or not math.isfinite(eps) or eps <= 0
            or len(betas) != 2 or any(not math.isfinite(x) or not 0 <= x < 1 for x in betas)):
        raise ValueError("invalid inherited Adam hyperparameters")


def _validate_continuation(state, names):
    """Reject malformed model/Adam payloads before allocating a continuation."""
    if state.get("format") != continuity.FORMAT_VERSION:
        raise ValueError("unsupported inherited rewiring checkpoint")
    if names != [PARAMETER_NAMES] or set(state["model"]) != set(PARAMETER_NAMES):
        raise ValueError("checkpoint model/optimizer names differ from AgentModel")
    variant, config = state["variant"], state["config"]
    if variant["kind"] != "learned" or variant["policy"] != "online" or variant["reset_every"]:
        raise ValueError("checkpoint must contain a learned continuous own-action controller")
    width = _positive_integer("checkpoint width", variant["width"])
    jobs, batch = _positive_integer("checkpoint jobs", config["n_jobs"]), config["batch_size"]
    shapes = {"recurrent": (width, width), "encoder.weight": (width, jobs * 24),
              "head.weight": (jobs * 11, width), "head.bias": (jobs * 11,),
              "hidden": (batch, width)}
    tensors = {**state["model"], "hidden": state["hidden"]}
    for name, value in tensors.items():
        if (not isinstance(value, torch.Tensor) or value.shape != shapes[name]
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid checkpoint tensor: {name}")
    groups = state["optimizer"]["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(len(PARAMETER_NAMES))):
        raise ValueError("unexpected checkpoint Adam parameter IDs or order")
    _validate_adam_group(groups[0])
    moments = state["optimizer"]["state"]
    if set(moments) != set(range(len(PARAMETER_NAMES))):
        raise ValueError("checkpoint must retain every inherited Adam state")
    for index, values in moments.items():
        if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incomplete or unsupported Adam checkpoint state")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid Adam checkpoint tensor")
            if name == "step":
                if value.numel() != 1 or float(value) < 0 or not float(value).is_integer():
                    raise ValueError("invalid Adam checkpoint step")
            elif value.shape != shapes[PARAMETER_NAMES[index]]:
                raise ValueError("Adam checkpoint moment shape mismatch")
            elif name == "exp_avg_sq" and bool((value < 0).any()):
                raise ValueError("negative Adam checkpoint second moment")


class RewireTrainer(CreditTrainer):
    def __init__(self, base, config: RewireConfig, *, origin=None, chunk_ticks=64,
                 phase_ticks=65536):
        if not isinstance(config, RewireConfig):
            raise ValueError("config must be RewireConfig")
        if type(base.model) is not AgentModel or base.model.kind != "learned":
            raise ValueError("rewiring requires the unchanged learned AgentModel")
        _positive_integer("phase_ticks", phase_ticks)
        _positive_integer("chunk_ticks", chunk_ticks)
        if phase_ticks % chunk_ticks:
            raise ValueError("phase_ticks must end on a complete live chunk")
        if not isinstance(base.optimizer, torch.optim.Adam):
            raise ValueError("rewiring requires inherited Adam")
        optimizer_parameter_names(base.model, base.optimizer)
        _validate_adam_group(base.optimizer.param_groups[0])
        inherited = base.semantic_state()
        _validate_continuation(inherited, [PARAMETER_NAMES])
        inherited_hash = continuity.state_digest(inherited)
        return_regime = {"duration_range": tuple(base.world.regime.duration_range),
                         "gap_range": tuple(base.world.regime.gap_range), "name": "rewiring-A2"}
        super().__init__(base, CreditConfig(mode="bp64", chunk_ticks=chunk_ticks,
            duration_range=(4., 8.), gap_range=(4., 12.), regime_name="rewiring-B"), origin=origin)
        self.rewire_config = deepcopy(config)
        self.operator = RewireOperator(base.model.recurrent, config)
        self.rewire_origin = {**deepcopy(origin or {}), "inherited_semantic_hash": inherited_hash}
        self.phase_ticks, self.phase_returned, self.return_regime = phase_ticks, False, return_regime
        self.structural_windows = []
        self.rewire_sources, self.rewire_dependencies = source_snapshots(), dependency_snapshots()
        self.rewire_source_hashes = source_hashes(self.rewire_sources)
        self.rewire_dependency_hashes = source_hashes(self.rewire_dependencies)

    @property
    def arm(self):
        return self.rewire_config.arm

    @property
    def total_stage_ticks(self):
        return self.stage_ticks

    @classmethod
    def from_source_checkpoint(cls, path, config, device=None, strict_sources=True,
                               phase_ticks=65536):
        path = Path(path).resolve()
        source_hash = file_sha256(path)
        base = continuity.Trainer.from_checkpoint(path, device=device, strict_sources=strict_sources)
        if base.model.width != 128:
            raise ValueError("formal rewiring continuation requires width 128")
        trainer = cls(base, config, origin={"checkpoint": str(path), "checkpoint_sha256": source_hash},
                      phase_ticks=phase_ticks)
        if source_hash != file_sha256(path):
            raise AssertionError("source checkpoint bytes changed")
        return trainer

    def _ensure_return_phase(self):
        if not self.phase_returned and self.total_stage_ticks >= self.phase_ticks:
            if self.total_stage_ticks != self.phase_ticks:
                raise ValueError("missed the exact return-regime boundary")
            self.change_regime(self.return_regime["duration_range"],
                               self.return_regime["gap_range"], self.return_regime["name"])
            self.phase_returned = True

    def train_update(self):
        self._ensure_return_phase()
        return super().train_update()

    def _apply_gradients(self, gradients):
        parameters = dict(self.model.named_parameters())
        if set(gradients) != set(parameters):
            raise ValueError("BPTT gradients must cover every AgentModel parameter exactly once")
        for name, value in gradients.items():
            if (not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape
                    or value.device != parameters[name].device or not bool(torch.isfinite(value).all())):
                raise FloatingPointError(f"invalid gradient for {name}")
        before = {name: value.detach().double().clone() for name, value in parameters.items()}
        raw = gradients["recurrent"]
        masked = {**gradients, "recurrent": self.operator.before_gradient(raw)}
        update = super()._apply_gradients(masked)
        group = self.optimizer.param_groups[0]
        event = self.operator.apply_after_optimizer(self.model.recurrent,
            self.optimizer.state[self.model.recurrent], raw, update["clip_scale"],
            group["lr"], group["betas"], group["eps"])
        # The credit update measured Adam alone. Replace its recurrent and total
        # norms with the actual combined Adam + shrink + birth/death displacement.
        squares = {name: (value.detach().double() - before[name]).square().sum()
                   for name, value in parameters.items()}
        actual_recurrent = float(squares["recurrent"].sqrt())
        actual_all = float(sum(squares.values()).sqrt())
        for name, actual in (("recurrent_delta", actual_recurrent), ("all_delta", actual_all)):
            self.update_norms[name + "_sum"] += actual - update[name]
            update[name] = actual
        event = {**event, "stage_ticks": self.total_stage_ticks,
                 "stage_update": self.stage_updates + 1,
                 "recurrent_weight_l2": float(self.model.recurrent.detach().double().norm())}
        if self.operator.updates_in_window == 0:
            self.structural_windows.append(deepcopy(event))
        update.update(structure=event, potential_recurrent_gradient_norm=float(raw.double().norm()))
        return update

    def semantic_state(self):
        return {"format": FORMAT_VERSION, "rewire_config": asdict(self.rewire_config),
                "origin": deepcopy(self.rewire_origin),
                "optimizer_parameter_names": optimizer_parameter_names(self.model, self.optimizer),
                "phase_ticks": self.phase_ticks, "phase_returned": self.phase_returned,
                "return_regime": deepcopy(self.return_regime),
                "structural_windows": deepcopy(self.structural_windows),
                "rewiring": self.operator.state_dict(), "credit": CreditTrainer.semantic_state(self)}

    def state_dict(self):
        state = self.semantic_state()
        credit = state["credit"]
        credit["continuation"] = self.base.state_dict()
        credit.update(sources=deepcopy(self.sources), source_hashes=deepcopy(self.source_hashes),
                      dependencies=deepcopy(self.dependencies), dependency_hashes=deepcopy(self.dependency_hashes))
        state.update(sources=deepcopy(self.rewire_sources), source_hashes=deepcopy(self.rewire_source_hashes),
                     dependencies=deepcopy(self.rewire_dependencies),
                     dependency_hashes=deepcopy(self.rewire_dependency_hashes))
        return state

    @classmethod
    def from_checkpoint(cls, path, device=None, strict_sources=True):
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls._from_state(state, device=device, strict_sources=strict_sources)

    @classmethod
    def _from_state(cls, state, device=None, strict_sources=True):
        if state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported rewiring checkpoint")
        if strict_sources:
            saved = source_hashes(state["sources"])
            if saved != state["source_hashes"] or saved != source_hashes():
                raise ValueError("rewiring checkpoint source mismatch; use its saved snapshots")
            saved = source_hashes(state["dependencies"])
            if saved != state["dependency_hashes"] or saved != source_hashes(dependency_snapshots()):
                raise ValueError("rewiring dependency source mismatch")
        config = RewireConfig(**state["rewire_config"])
        credit_state = state["credit"]
        credit_config = CreditConfig(**credit_state["credit_config"])
        if credit_config.mode != "bp64" or credit_state["live_credit"] is not None or credit_state["old_credit"] is not None:
            raise ValueError("rewiring requires exact BPTT without replay")
        _validate_continuation(credit_state["continuation"], state["optimizer_parameter_names"])
        phase_ticks = _positive_integer("checkpoint phase_ticks", state["phase_ticks"])
        if phase_ticks % credit_config.chunk_ticks or type(state["phase_returned"]) is not bool:
            raise ValueError("invalid rewiring checkpoint phase schedule")
        continuation = credit_state["continuation"]
        ticks = continuation["ticks"] - continuation["phase_start_tick"]
        if ticks < 0 or ticks % credit_config.chunk_ticks or (
                state["phase_returned"] and ticks < phase_ticks) or (
                not state["phase_returned"] and ticks > phase_ticks):
            raise ValueError("inconsistent rewiring checkpoint phase progress")
        return_regime = deepcopy(state["return_regime"])
        CreditConfig(duration_range=return_regime["duration_range"], gap_range=return_regime["gap_range"],
                     regime_name=return_regime["name"])
        returns = [change for change in credit_state["regime_changes"]
                   if change["regime"]["name"] == return_regime["name"]]
        if len(returns) != int(state["phase_returned"]) or any(
                change["stage_tick"] != phase_ticks for change in returns):
            raise ValueError("rewiring checkpoint return-regime history mismatch")
        credit = CreditTrainer._from_state(credit_state, device, strict_sources)
        trainer = cls.__new__(cls)
        trainer.__dict__.update(credit.__dict__)
        trainer.rewire_config = config
        trainer.operator = RewireOperator.from_state(trainer.model.recurrent, state["rewiring"])
        if trainer.operator.config != config:
            raise ValueError("rewiring operator/trainer configuration mismatch")
        if trainer.operator.updates != trainer.stage_updates:
            raise ValueError("rewiring operator/update progress mismatch")
        adam = trainer.optimizer.state[trainer.model.recurrent]
        # Dense Adam can land on an exact zero without discarding its moments.
        # At attachment, every arm must also retain the inherited Adam intact.
        # Structural arms clear dormant moments at their first update boundary.
        if config.arm != "dense" and trainer.operator.updates:
            for name in ("exp_avg", "exp_avg_sq"):
                if bool((adam[name][~trainer.operator.mask] != 0).any()):
                    raise ValueError("dormant recurrent connections retain Adam checkpoint moments")
        trainer.rewire_origin = deepcopy(state["origin"])
        trainer.phase_ticks, trainer.phase_returned = phase_ticks, state["phase_returned"]
        trainer.return_regime, trainer.structural_windows = return_regime, deepcopy(state["structural_windows"])
        if len(trainer.structural_windows) != trainer.operator.updates // config.window_updates:
            raise ValueError("structural window history/progress mismatch")
        for name, key in (("rewire_sources", "sources"), ("rewire_source_hashes", "source_hashes"),
                          ("rewire_dependencies", "dependencies"), ("rewire_dependency_hashes", "dependency_hashes")):
            setattr(trainer, name, deepcopy(state[key]))
        return trainer

    def report(self):
        report = super().report()
        trace, curve = report.pop("trace"), report.pop("curve")
        report.update(format=FORMAT_VERSION, arm=self.arm, rewire_config=asdict(self.rewire_config),
            total_stage_ticks=self.total_stage_ticks, phase_ticks=self.phase_ticks,
            phase_returned=self.phase_returned, return_regime=deepcopy(self.return_regime),
            rewire_origin=deepcopy(self.rewire_origin), structure=self.operator.summary(),
            structural_windows=deepcopy(self.structural_windows), trace_points=len(trace), curve_points=len(curve),
            last_update=deepcopy(trace[-1]) if self.stage_updates else None,
            rewiring_source_hashes=deepcopy(self.rewire_source_hashes),
            rewiring_dependency_hashes=deepcopy(self.rewire_dependency_hashes),
            topology_semantics="Dormant recurrent weights are exact zeros in the unchanged AgentModel; full BPTT potential gradients precede the live Adam mask.",
            continuity="One inherited live hidden state, world, parameter set, Adam and all RNG states persist through B and the A2 return.")
        return report


def _atomic_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def write_artifacts(trainer, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(folder / "checkpoint.pt")
    continuity.atomic_json(folder / "report.json", trainer.report())
    points = (point for point in trainer.trace if "structure" in (point.get("parameter_update") or {}))
    _atomic_jsonl(folder / "training.jsonl", ({"event": "train", "arm": trainer.arm, **point} for point in points))
    _atomic_jsonl(folder / "structural_windows.jsonl", trainer.structural_windows)
    for prefix, sources in (("rewiring", trainer.rewire_sources), ("credit", trainer.sources),
                            ("continuity", trainer.base.sources), ("dependencies", trainer.rewire_dependencies)):
        for name, source in sources.items():
            path = folder / "sources" / prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")


def run_to_target(trainer, target_ticks, folder, checkpoint_every=256, stop_after_updates=None):
    if (isinstance(target_ticks, bool) or not isinstance(target_ticks, int)
            or target_ticks < trainer.total_stage_ticks or target_ticks % trainer.credit_config.chunk_ticks):
        raise ValueError("target must be a future complete chunk boundary")
    if (isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 0
            or (stop_after_updates is not None and (isinstance(stop_after_updates, bool)
                or not isinstance(stop_after_updates, int) or stop_after_updates < 0))):
        raise ValueError("checkpoint and stop intervals must be nonnegative integers")
    folder = Path(folder)
    write_artifacts(trainer, folder)
    completed = 0
    try:
        while trainer.total_stage_ticks < target_ticks:
            if stop_after_updates is not None and completed >= stop_after_updates:
                break
            point = trainer.train_update()
            completed += 1
            event = {"event": "train", "arm": trainer.arm, **point}
            with (folder / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if trainer.operator.updates_in_window == 0:
                with (folder / "structural_windows.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(trainer.structural_windows[-1], allow_nan=False) + "\n")
            if trainer.stage_updates == 1 or trainer.stage_updates % trainer.config.log_every == 0 or trainer.total_stage_ticks == target_ticks:
                structure = point["parameter_update"]["structure"]
                print(f"arm={trainer.arm} ticks={trainer.total_stage_ticks} loss={point['loss']:.6f} "
                      f"active={structure['active']} births={structure['births']} deaths={structure['deaths']}", flush=True)
            boundary = trainer.total_stage_ticks in (trainer.phase_ticks, 2 * trainer.phase_ticks)
            if boundary:
                name = "checkpoint_B.pt" if trainer.total_stage_ticks == trainer.phase_ticks else "checkpoint_A2.pt"
                trainer.save_checkpoint(folder / name)
            if boundary or (checkpoint_every and trainer.stage_updates % checkpoint_every == 0):
                if checkpoint_every and trainer.stage_updates % checkpoint_every == 0:
                    trainer.save_checkpoint(folder / f"checkpoint_{trainer.stage_updates:06d}.pt")
                write_artifacts(trainer, folder)
    finally:
        if trainer.base._at_boundary:
            write_artifacts(trainer, folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--shrink-step", type=float)
    parser.add_argument("--window-updates", type=int)
    parser.add_argument("--structural-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-ticks", type=int, default=131072)
    parser.add_argument("--phase-ticks", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="resume output/checkpoint.pt")
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.checkpoint_every < 0 or args.target_ticks < 0:
        parser.error("invalid threads, checkpoint interval, or target")
    torch.set_num_threads(args.threads)
    if args.resume:
        trainer = RewireTrainer.from_checkpoint(args.output / "checkpoint.pt", device=args.device)
        expected = {"arm": trainer.arm, "shrink_step": trainer.rewire_config.shrink_step,
                    "window_updates": trainer.rewire_config.window_updates,
                    "structural_seed": trainer.rewire_config.structural_seed, "phase_ticks": trainer.phase_ticks}
        if any(getattr(args, key) is not None and getattr(args, key) != value for key, value in expected.items()):
            parser.error("requested rewiring configuration differs from resume checkpoint")
        if args.source is not None and str(args.source.resolve()) != trainer.rewire_origin.get("checkpoint"):
            parser.error("requested source differs from resume checkpoint")
    else:
        if args.source is None or args.arm is None:
            parser.error("--source and --arm are required for a new continuation")
        if args.output.exists():
            raise FileExistsError("output exists; explicitly resume its own checkpoint")
        config = RewireConfig(arm=args.arm, shrink_step=5e-5 if args.shrink_step is None else args.shrink_step,
            window_updates=8 if args.window_updates is None else args.window_updates,
            structural_seed=0 if args.structural_seed is None else args.structural_seed)
        trainer = RewireTrainer.from_source_checkpoint(args.source, config, device=args.device,
            phase_ticks=65536 if args.phase_ticks is None else args.phase_ticks)
    run_to_target(trainer, args.target_ticks, args.output, args.checkpoint_every)


if __name__ == "__main__":
    main()
