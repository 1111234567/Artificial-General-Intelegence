"""Versioned BPTT continuation with zero-initialized nonlinear field extensions."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch

from experiments.continuity import run as continuity
from experiments.credit.run import CreditConfig, CreditTrainer, file_sha256
from experiments.loops.model import ARMS, LoopModel


FORMAT_VERSION = "loops-trainer-v1"


def source_snapshots():
    folder = Path(__file__).parent
    return {name: (folder / name).read_text(encoding="utf-8") for name in
            ("model.py", "train.py", "test_model.py", "test_train.py") if (folder / name).is_file()}


def source_hashes(sources=None):
    return {name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in (source_snapshots() if sources is None else sources).items()}


def optimizer_parameter_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        groups = [[names[id(parameter)] for parameter in group["params"]]
                  for group in optimizer.param_groups]
    except KeyError as error:
        raise ValueError("optimizer contains a parameter absent from the model") from error
    flat = [name for group in groups for name in group]
    if len(flat) != len(set(flat)) or set(flat) != set(names.values()):
        raise ValueError("optimizer must cover every model parameter exactly once")
    return groups


def attach_model(base, arm):
    inherited_optimizer = base.optimizer
    base.model = LoopModel.attach(base.model, arm)
    if arm != "baseline":
        if len(inherited_optimizer.param_groups) != 1:
            raise ValueError("expected the inherited single Adam parameter group")
        group = {key: deepcopy(value) for key, value in inherited_optimizer.param_groups[0].items()
                 if key != "params"}
        inherited_optimizer.add_param_group({**group, "params": [base.model.branch]})
        if base.model.branch in inherited_optimizer.state:
            raise AssertionError("new branch Adam moments must start absent")
    optimizer_parameter_names(base.model, inherited_optimizer)


class LoopTrainer(CreditTrainer):
    def __init__(self, base, arm, *, origin=None, chunk_ticks=64):
        if arm not in ARMS:
            raise ValueError("unknown loop arm")
        inherited_semantic_hash = continuity.state_digest(base.semantic_state())
        super().__init__(base, CreditConfig(mode="bp64", chunk_ticks=chunk_ticks,
            duration_range=(4., 8.), gap_range=(4., 12.), regime_name="loops-B"), origin=origin)
        self.arm = arm
        self.loop_origin = {**deepcopy(origin or {}), "inherited_semantic_hash": inherited_semantic_hash}
        attach_model(base, arm)
        self.loop_sources = source_snapshots()
        self.loop_source_hashes = source_hashes(self.loop_sources)
        self.branch_update_l2_sum = 0.

    @classmethod
    def from_source_checkpoint(cls, path, arm, device=None, strict_sources=True):
        path = Path(path).resolve()
        source_hash = file_sha256(path)
        base = continuity.Trainer.from_checkpoint(path, device=device, strict_sources=strict_sources)
        if base.model.width != 128:
            raise ValueError("formal loop continuation requires width 128")
        trainer = cls(base, arm, origin={"checkpoint": str(path), "checkpoint_sha256": source_hash})
        if source_hash != file_sha256(path):
            raise AssertionError("source checkpoint bytes changed")
        return trainer

    def _apply_gradients(self, gradients):
        parameters = dict(self.model.named_parameters())
        if set(gradients) != set(parameters):
            raise ValueError("BPTT gradients must cover every loop model parameter exactly once")
        for name, gradient in gradients.items():
            if gradient.shape != parameters[name].shape or not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"invalid gradient for {name}")
        before = None if self.arm == "baseline" else self.model.branch.detach().double().clone()
        update = super()._apply_gradients(gradients)
        delta = 0. if before is None else float((self.model.branch.detach().double() - before).norm())
        self.branch_update_l2_sum += delta
        update.update(branch_delta=delta,
            branch_raw_gradient_norm=0. if self.arm == "baseline" else float(gradients["branch"].double().norm()))
        return update

    def semantic_state(self):
        return {"format": FORMAT_VERSION, "arm": self.arm,
            "origin": deepcopy(self.loop_origin),
            "optimizer_parameter_names": optimizer_parameter_names(self.model, self.optimizer),
            "branch_update_l2_sum": self.branch_update_l2_sum,
            "credit": CreditTrainer.semantic_state(self)}

    def state_dict(self):
        state = self.semantic_state()
        credit = state["credit"]
        credit["continuation"] = self.base.state_dict()
        credit.update(sources=deepcopy(self.sources), source_hashes=deepcopy(self.source_hashes),
            dependencies=deepcopy(self.dependencies), dependency_hashes=deepcopy(self.dependency_hashes))
        state.update(sources=deepcopy(self.loop_sources), source_hashes=deepcopy(self.loop_source_hashes))
        return state

    @classmethod
    def from_checkpoint(cls, path, device=None, strict_sources=True):
        return cls._from_state(torch.load(Path(path), map_location="cpu", weights_only=False),
                               device=device, strict_sources=strict_sources)

    @classmethod
    def _from_state(cls, state, device=None, strict_sources=True):
        if state.get("format") != FORMAT_VERSION or state.get("arm") not in ARMS:
            raise ValueError("unsupported loop checkpoint")
        if strict_sources:
            saved = source_hashes(state["sources"])
            current = source_hashes()
            if saved != state["source_hashes"] or saved != current:
                raise ValueError("loop checkpoint source mismatch; use its saved source snapshots")
        original = state["credit"]
        if original["credit_config"]["mode"] != "bp64":
            raise ValueError("loop continuation only supports exact BPTT without replay")
        credit_state = deepcopy(original)
        names = deepcopy(state["optimizer_parameter_names"])
        groups = credit_state["continuation"]["optimizer"]["param_groups"]
        flat = [name for group in names for name in group]
        if len(names) != len(groups) or any(len(a) != len(b["params"]) for a, b in zip(names, groups)):
            raise ValueError("optimizer group/name mismatch")
        if len(flat) != len(set(flat)) or set(flat) != set(credit_state["continuation"]["model"]):
            raise ValueError("optimizer parameter names do not match model parameters")
        ids = [index for group in groups for index in group["params"]]
        if ids != list(range(len(ids))):
            raise ValueError("unexpected optimizer parameter IDs or order")
        if set(credit_state["continuation"]["optimizer"]["state"]) - set(ids):
            raise ValueError("optimizer state contains an unknown parameter ID")
        extended = state["arm"] != "baseline"
        expected_old = ["recurrent", "encoder.weight", "head.weight", "head.bias"]
        if names != ([expected_old, ["branch"]] if extended else [expected_old]):
            raise ValueError("unexpected named Adam parameter order")
        continuation = credit_state["continuation"]
        for name, value in {**continuation["model"], "hidden": continuation["hidden"]}.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite checkpoint tensor: {name}")
        for index, moments in continuation["optimizer"]["state"].items():
            expected_shape = continuation["model"][flat[index]].shape
            if not {"step", "exp_avg", "exp_avg_sq"}.issubset(moments):
                raise ValueError("incomplete Adam checkpoint state")
            for name, value in moments.items():
                value = torch.as_tensor(value)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("nonfinite Adam checkpoint state")
                if name == "step":
                    if value.numel() != 1 or float(value) < 0:
                        raise ValueError("invalid Adam checkpoint step")
                elif value.shape != expected_shape:
                    raise ValueError("Adam checkpoint moment shape mismatch")
                elif name in {"exp_avg_sq", "max_exp_avg_sq"} and bool((value < 0).any()):
                    raise ValueError("negative Adam checkpoint second moment")
        if not math.isfinite(state["branch_update_l2_sum"]) or state["branch_update_l2_sum"] < 0:
            raise ValueError("invalid accumulated branch update norm")
        if extended:
            credit_state["continuation"]["model"].pop("branch")
            branch_group = groups.pop()
            credit_state["continuation"]["optimizer"]["state"].pop(branch_group["params"][0], None)
        credit = CreditTrainer._from_state(credit_state, device, strict_sources)
        trainer = cls.__new__(cls)
        trainer.__dict__.update(credit.__dict__)
        trainer.arm = state["arm"]
        trainer.loop_origin = deepcopy(state["origin"])
        attach_model(trainer.base, trainer.arm)
        if optimizer_parameter_names(trainer.model, trainer.optimizer) != names:
            raise ValueError("restored optimizer parameter names differ")
        trainer.model.load_state_dict(original["continuation"]["model"])
        trainer.optimizer.load_state_dict(original["continuation"]["optimizer"])
        trainer.loop_sources = deepcopy(state["sources"])
        trainer.loop_source_hashes = deepcopy(state["source_hashes"])
        trainer.branch_update_l2_sum = float(state["branch_update_l2_sum"])
        return trainer

    def report(self):
        report = super().report()
        report.update(format=FORMAT_VERSION, arm=self.arm, loop_origin=deepcopy(self.loop_origin),
            loop_source_hashes=deepcopy(self.loop_source_hashes),
            branch_update_l2_sum=self.branch_update_l2_sum,
            branch_metrics=self.model.branch_metrics(self.h),
            interpretation="Matched extra parameter counts compare source squares with recipient/source products; topology is complete in every arm.")
        return report


def write_artifacts(trainer, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(folder / "checkpoint.pt")
    continuity.atomic_json(folder / "report.json", trainer.report())
    sources = folder / "sources"
    sources.mkdir(exist_ok=True)
    for name, source in trainer.loop_sources.items():
        (sources / name).write_text(source, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-ticks", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="resume output/checkpoint.pt")
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.checkpoint_every < 0 or args.target_ticks < 0:
        parser.error("invalid threads, checkpoint interval, or target")
    torch.set_num_threads(args.threads)
    checkpoint = args.output / "checkpoint.pt"
    if args.resume:
        trainer = LoopTrainer.from_checkpoint(checkpoint, device=args.device)
        if args.arm is not None and args.arm != trainer.arm:
            parser.error("requested arm differs from resume checkpoint")
        if args.source is not None and str(args.source.resolve()) != trainer.loop_origin.get("checkpoint"):
            parser.error("requested source differs from resume checkpoint")
    else:
        if args.source is None or args.arm is None:
            parser.error("--source and --arm are required for a new continuation")
        if args.output.exists():
            raise FileExistsError("output exists; explicitly resume its own checkpoint")
        trainer = LoopTrainer.from_source_checkpoint(args.source, args.arm, device=args.device)
    if args.target_ticks < trainer.stage_ticks or args.target_ticks % trainer.credit_config.chunk_ticks:
        parser.error("target must be a future complete chunk boundary")
    args.output.mkdir(parents=True, exist_ok=True)
    # A complete boundary exists before the first update, even if the process
    # is interrupted before its first periodic checkpoint.
    write_artifacts(trainer, args.output)
    try:
        while trainer.stage_ticks < args.target_ticks:
            point = trainer.train_update()
            if trainer.stage_updates == 1 or trainer.stage_updates % trainer.config.log_every == 0:
                print(f"arm={trainer.arm} ticks={trainer.stage_ticks} loss={point['loss']:.6f}", flush=True)
            if args.checkpoint_every and trainer.stage_updates % args.checkpoint_every == 0:
                trainer.save_checkpoint(args.output / f"checkpoint_{trainer.stage_updates:06d}.pt")
                write_artifacts(trainer, args.output)
    finally:
        if trainer.base._at_boundary:
            write_artifacts(trainer, args.output)


if __name__ == "__main__":
    main()
