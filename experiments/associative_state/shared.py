"""Post-v1 follow-up with one shared query-encoding matrix.

This module reuses the frozen v1 training and evaluation operations through
process-local activation.  It never edits the v1 source or output directory.
The independent embedding rows in v1 isolated unqueried A addresses from
B-only online gradients; a shared projection removes that structural barrier.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.model import AssociationMemory
from experiments.associative_state import evaluate as base_evaluate
from experiments.associative_state import launch as base_launch
from experiments.associative_state import train as base_train

OUT = ROOT / "data/associative-state/shared-v2"
SOURCE = Path(__file__).resolve()
PLAN = SOURCE.with_name("shared-plan.md")
ENCODER = "fixed_codes_shared_linear"
EXPERIMENT = "associative-state-shared-v2"


class FixedCodesSharedProjection(nn.Module):
    """Fixed random address codes encoded by one learned linear projection."""

    def __init__(self, codes):
        super().__init__()
        if codes.ndim != 2 or not torch.is_floating_point(codes):
            raise ValueError("codes must be a floating [n_keys, key_dim] tensor")
        self.register_buffer("codes", codes.detach().clone())
        self.projection = nn.Linear(codes.shape[1], codes.shape[1], bias=False,
                                    device=codes.device, dtype=codes.dtype)
        nn.init.eye_(self.projection.weight)

    @property
    def weight(self):
        """Expose module dtype/device for the unchanged base-state checks."""
        return self.projection.weight

    def forward(self, keys):
        return self.projection(self.codes[keys])


class SharedAssociationMemory(AssociationMemory):
    """The same memory updates and readout with a shared key encoder.

    The superclass initializes every original parameter first.  Replacing its
    embedding afterward preserves the original initial keys and, for the GRU
    control, the exact original core under the same externally supplied seed.
    The fixed code table is persistent checkpoint data, not learned parameters
    and not per-actor fast state.
    """

    def __init__(self, kind="add", key_dim=16, n_keys=8, n_values=4):
        super().__init__(kind=kind, key_dim=key_dim, n_keys=n_keys, n_values=n_values)
        self.key_embedding = FixedCodesSharedProjection(self.key_embedding.weight)

    def parameter_report(self):
        report = super().parameter_report()
        codes = self.key_embedding.codes
        report.update(encoder=ENCODER,
                      shared_encoder_parameters=self.key_embedding.weight.numel(),
                      fixed_code_scalars=codes.numel(),
                      fixed_code_bytes=codes.numel() * codes.element_size(),
                      fixed_codes_trainable=False,
                      fixed_codes_in_parameter_count=False,
                      fixed_codes_in_fast_state_count=False,
                      fixed_code_storage="persistent buffer; additional fixed constants",
                      buffers_by_name={name: value.numel()
                                       for name, value in self.named_buffers()})
        return report


def activate():
    """Select shared-v2 inside this process, including spawned workers."""
    base_train.OUT = OUT
    base_evaluate.OUT = OUT
    base_launch.OUT = OUT
    base_train.AssociationMemory = SharedAssociationMemory
    base_evaluate.AssociationMemory = SharedAssociationMemory
    return OUT


def declare():
    """Declare a separate prospective follow-up, disclosing the v1 inspection."""
    activate()
    # Fail before creating a protocol if its new design document is absent.
    for source in (SOURCE, PLAN):
        if not source.is_file():
            raise FileNotFoundError(source)
    base_train.declare()
    path = OUT / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for source in (SOURCE, PLAN):
        protocol["sources"][str(source.relative_to(ROOT))] = base_train.sha(source)
    protocol.update(experiment=EXPERIMENT, encoder=ENCODER,
                    followup=dict(
                        parent="data/associative-state/v1",
                        declared_after_inspection_of_v1_results=True,
                        disclosure=("Added after partial or complete v1 results had been inspected; "
                                    "this is not the original v1 preregistration."),
                        analytical_reason=("Independent v1 embedding rows receive no gradient from "
                                           "other keys, and positive read scale cannot change argmax. "
                                           "A shared projection permits B-only gradients to alter A queries."),
                        unchanged_sample_update_and_evaluation_budgets=True,
                        parameter_budget_matched_to_v1=False),
                    learned_parameters={"add": 258, "delta": 258, "recurrent": 16769},
                    fixed_code_scalars=128,
                    state_scalars_per_actor=64)
    base_train.write_json(path, protocol)
    return protocol


def check_protocol():
    protocol = base_train.check_protocol()
    assert protocol.get("experiment") == EXPERIMENT
    assert protocol.get("encoder") == ENCODER
    for source in (SOURCE, PLAN):
        assert str(source.relative_to(ROOT)) in protocol["sources"], source
    return protocol


def check_freeze():
    """Require all nine frozen training endpoints before any shared-v2 score."""
    check_protocol()
    frozen = json.loads((OUT / "training-freeze.json").read_text(encoding="utf-8"))
    assert frozen["all_training_complete_before_scoring"] is True
    assert frozen["paired_train_samples"] is True
    assert frozen["paired_add_delta_initial_parameters"] is True
    expected = {str(Path(f"seed_{seed:03d}") / arm / "final.pt")
                for seed in range(3) for arm in base_train.ARMS}
    assert set(frozen["checkpoints"]) == expected
    for name, digest in frozen["checkpoints"].items():
        assert base_train.sha(OUT / name) == digest, name
    return frozen


def train_one(job):
    activate()
    check_protocol()
    return base_launch.train_one(job)


def eval_one(job):
    activate()
    check_freeze()
    return base_launch.eval_one(job)


def launch(stage, workers=9):
    """Run paired arms with a completed-training barrier before evaluation."""
    if stage not in ("train", "eval"):
        raise ValueError("stage must be 'train' or 'eval'")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    activate()
    if stage == "eval":
        check_freeze()
    else:
        check_protocol()
    with ProcessPoolExecutor(max_workers=workers, initializer=activate) as pool:
        worker = train_one if stage == "train" else eval_one
        futures = [pool.submit(worker, (seed, arm))
                   for seed in range(3) for arm in base_train.ARMS]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    if stage == "train":
        base_launch.freeze()
        check_freeze()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "train", "eval"))
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.command == "declare":
        protocol = declare()
        print(json.dumps(dict(event="declared", experiment=protocol["experiment"],
                              path=str(OUT / "protocol.json"))), flush=True)
    else:
        launch(args.command, args.workers)


if __name__ == "__main__":
    main()
