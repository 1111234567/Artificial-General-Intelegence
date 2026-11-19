"""Freeze the original random encoder in the read-before-write policy.

This post-hoc control changes only encoder trainability relative to read-first
v2.  The public inputs, initialization, forward computation and h/z/M states
are inherited unchanged.  Only query/key/value/gate/read_scale learn, from the
existing actual-action CE and public-outcome PG objectives without binding
labels.  All outputs belong to a separate frozen-trace-v3 directory.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil

import torch

from experiments.learned_binding import audit, common, pretrain, run
from experiments.learned_binding.model import FAMILY_MAPPINGS
from experiments.learned_binding.read_first import ReadFirstPolicy


ROOT = Path(__file__).resolve().parents[2]
V1_OUT = ROOT / "data/learned-binding/v1"
READ_FIRST_OUT = ROOT / "data/learned-binding/read-first-v2"
OUT = ROOT / "data/learned-binding/frozen-trace-v3"
SOURCE = Path(__file__).resolve()
READ_FIRST_SOURCE = SOURCE.with_name("read_first.py")
PLAN = SOURCE.with_name("frozen-trace-plan.md")
TRAINABLE_NAMES = frozenset(("read_scale_raw",) + tuple(
    f"{head}.{part}" for head in ("query", "key", "value", "gate")
    for part in ("weight", "bias")))


class FrozenTracePolicy(ReadFirstPolicy):
    """Keep the seeded random GRU trace fixed while learning its memory heads."""

    def __init__(self, executor=None, executor_config=None,
                 original_mappings=FAMILY_MAPPINGS):
        super().__init__(executor=executor, executor_config=executor_config,
                         original_mappings=original_mappings)
        self.encoder.requires_grad_(False)

    def parameter_report(self):
        report = super().parameter_report()
        report["encoder_frozen"] = True
        return report


def activate():
    """Select the variant/root in both the parent and Windows-spawned workers."""
    common.OUT = OUT
    common.NativeMemoryPolicy = FrozenTracePolicy
    pretrain.OUT = OUT
    run.OUT = OUT
    audit.NativeMemoryPolicy = FrozenTracePolicy
    return OUT


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reference(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=common.sha(path))


def _completed_audit(root):
    path = root / "audit.json"
    report = _read_json(path)
    expected = dict(probe_count=39, B_block_count=384, native_replay_count=9,
                    independently_recomputed_pg_updates=6)
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"completed 39-probe/384-block audit required: {path}: {name}")
    for name in ("all_world_actions_and_public_rewards_replayed",
                 "all_controller_live_states_preserved_across_optimizer_updates",
                 "all_source_executors_unchanged"):
        if report.get(name) is not True:
            raise ValueError(f"parent audit check failed or absent: {path}: {name}")
    return _reference(path)


def _parent_audit():
    """The copied manual controls were executed and audited in v1."""
    return _completed_audit(V1_OUT)


def declare():
    """Declare before fitting; preserve the byte-identical v1 data manifest."""
    activate()
    for path in (SOURCE, PLAN, READ_FIRST_SOURCE, V1_OUT / "protocol.json",
                 V1_OUT / "data-freeze.json", V1_OUT / "training-freeze.json",
                 READ_FIRST_OUT / "protocol.json", READ_FIRST_OUT / "training-freeze.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (OUT / "protocol.json").exists() or (OUT / "data-freeze.json").exists():
        raise FileExistsError("frozen-trace protocol/data manifest already exists")
    parent_audit = _parent_audit()
    read_first_audit = _completed_audit(READ_FIRST_OUT)
    for root in (V1_OUT, READ_FIRST_OUT):
        protocol = _read_json(root / "protocol.json")
        if protocol["settings"] != common.SETTINGS:
            raise ValueError("frozen-trace requires the unchanged v1/v2 settings")
        for name, digest in protocol["sources"].items():
            if common.sha(ROOT / name) != digest:
                raise ValueError("frozen parent source changed: " + name)
    data_source = _reference(V1_OUT / "data-freeze.json")
    common.declare()
    path = OUT / "protocol.json"
    protocol = _read_json(path)
    for source in (SOURCE, PLAN, READ_FIRST_SOURCE, Path(audit.__file__)):
        protocol["sources"][str(source.relative_to(ROOT))] = common.sha(source)
    parent_artifacts = {"v1/" + name: _reference(V1_OUT / name) for name in
        ("protocol.json", "data-freeze.json", "training-freeze.json", "audit.json")}
    for name in ("protocol.json", "training-freeze.json", "audit.json"):
        parent_artifacts["read-first-v2/" + name] = _reference(READ_FIRST_OUT / name)
    for name in ("diagnostics/action-loss.json", "diagnostics/binding-probe.json"):
        if (V1_OUT / name).is_file():
            parent_artifacts["v1/" + name] = _reference(V1_OUT / name)
    for reference in parent_artifacts.values():
        protocol["sources"][reference["path"]] = reference["sha256"]
    protocol.update(experiment="learned-binding-frozen-trace-v3",
        read_write_order="read_then_write_job0_then_job1", encoder_frozen=True,
        trainable_parameters=2258, encoder_parameters=5952, state_scalars=320,
        trainable_parameter_names=sorted(TRAINABLE_NAMES),
        followup=dict(parent="data/learned-binding/read-first-v2",
            declared_after_v1_and_v2_results_inspected=True,
            disclosure="Post-hoc encoder-trainability control after inspecting v1/v2 failures and representation diagnostics; the evaluation benchmark is reused.",
            only_change_from_read_first="Freeze the original seeded random encoder parameters; train only query/key/value/gate/read_scale.",
            identical_initial_state_dict_and_forward_to_read_first=True,
            unchanged_total_parameters_state_and_config=True,
            unchanged_training_data_objective_and_512_update_budget=True,
            unchanged_32_block_outcome_and_heldout_protocol=True,
            binding_labels_used_for_policy_training=False,
            parent_artifacts=parent_artifacts, parent_audit=parent_audit,
            read_first_audit=read_first_audit),
        training_data_reuse=dict(source=data_source, manifest_byte_identical=True,
            paths_remain_original_v1=True, large_training_streams_copied=False),
        manual_behavior_reuse=dict(method="byte-for-byte copytree of completed v1 behavior controls",
            kinds=["manual_hard", "manual_soft"], controls_reexecuted=False,
            manual_native_training_budget_equal=False,
            limitation="Manual historical pretraining differs from native; the new same-budget native control changes only encoder trainability relative to read-first-v2."))
    shutil.copyfile(V1_OUT / "data-freeze.json", OUT / "data-freeze.json")
    if common.sha(OUT / "data-freeze.json") != data_source["sha256"]:
        raise AssertionError("copied v1 training manifest differs")
    common.write_json(path, protocol)
    return protocol


def _encoder_state(model_state):
    return {name.removeprefix("encoder."): value for name, value in model_state.items()
            if name.startswith("encoder.")}


def _initial_encoder(seed):
    return {name: value.detach().cpu().clone()
            for name, value in common.new_native(seed).encoder.state_dict().items()}


def _check_trainability(model):
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != TRAINABLE_NAMES:
        raise AssertionError("only query/key/value/gate/read_scale may train")
    report = model.parameter_report()
    if (report["trainable_parameters"], report["encoder_parameters"], report["state_scalars"]) != (2258, 5952, 320):
        raise AssertionError("frozen-trace parameter/state budget changed")


def _train_worker(job):
    activate()
    return pretrain.train_one(job)


def train_all(device="mixed", workers=3):
    activate()
    common.check_protocol()
    if device != "mixed":
        raise ValueError("frozen-trace retains seed0 CUDA and seed1/2 CPU")
    if (common.SETTINGS["ce_updates"], common.SETTINGS["pg_blocks"], common.SETTINGS["seeds"]) != (512, 32, [0, 1, 2]):
        raise ValueError("frozen-trace requires the fixed three-seed, 512-CE/32-PG budget")
    if (OUT / "training-freeze.json").exists():
        raise FileExistsError(OUT / "training-freeze.json")
    jobs = [(seed, "cuda" if seed == 0 else "cpu") for seed in range(3)]
    with ProcessPoolExecutor(max_workers=workers, initializer=activate) as pool:
        for completed in as_completed([pool.submit(_train_worker, job) for job in jobs]):
            print(json.dumps(dict(event="frozen_trace_trained_seed", seed=completed.result())), flush=True)
    checkpoints, model_hashes, metadata, encoder_hashes = {}, {}, {}, {}
    for seed, declared_device in jobs:
        path = common.native_checkpoint(seed)
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if cp["updates"] != 512 or cp["settings"] != common.SETTINGS:
            raise ValueError("incomplete/different-budget frozen-trace endpoint")
        model = FrozenTracePolicy(**cp["model_config"])
        model.load_state_dict(cp["model"], strict=True)
        _check_trainability(model)
        common.assert_tree_equal(_initial_encoder(seed), _encoder_state(cp["model"]),
                                 f"seed{seed}: CE changed random encoder")
        name = str(path.relative_to(ROOT))
        checkpoints[name] = common.sha(path)
        model_hashes[name] = common.parameter_hash(model)
        encoder_hashes[str(seed)] = common.parameter_hash(model.encoder)
        training = path.with_name("training.json")
        info = _read_json(training)
        if (info["checkpoint_sha256"] != checkpoints[name]
                or info["executor_unchanged"] is not True or info["device"] != declared_device):
            raise ValueError("frozen-trace training metadata/checkpoint/device mismatch")
        metadata[str(seed)] = _reference(training)
    freeze = dict(all_complete_before_scoring=True, checkpoints=checkpoints,
        model_parameter_hashes=model_hashes, training_metadata=metadata,
        source_protocol_sha256=common.sha(OUT / "protocol.json"),
        architecture="FrozenTracePolicy", encoder_frozen=True,
        encoder_hashes=encoder_hashes, encoder_matches_seeded_initialization=True,
        read_write_order="read_then_write_job0_then_job1", trainable_parameters=2258,
        seeds=[0, 1, 2], updates_per_seed=512, devices={str(s): d for s, d in jobs})
    common.write_json(OUT / "training-freeze.json", freeze)
    return freeze


def _check_training_barrier():
    run.check_freeze()
    freeze = _read_json(OUT / "training-freeze.json")
    expected = {str(common.native_checkpoint(seed).relative_to(ROOT)) for seed in range(3)}
    if (set(freeze["checkpoints"]) != expected or freeze.get("seeds") != [0, 1, 2]
            or freeze.get("architecture") != "FrozenTracePolicy"
            or freeze.get("encoder_matches_seeded_initialization") is not True
            or freeze.get("source_protocol_sha256") != common.sha(OUT / "protocol.json")):
        raise ValueError("all three frozen-trace endpoints must be frozen before behavior")


def _tree_hashes(folder):
    return {str(path.relative_to(folder)): common.sha(path)
            for path in sorted(Path(folder).rglob("*")) if path.is_file()}


def reuse_manual_controls():
    """Copy v1 controls, preserving their source audit and every file hash."""
    activate()
    _check_training_barrier()
    parent_audit = _parent_audit()
    rows = []
    for seed in range(3):
        for kind in ("manual_hard", "manual_soft"):
            source = V1_OUT / f"seed_{seed:03d}/behavior" / kind
            target = OUT / f"seed_{seed:03d}/behavior" / kind
            completion = _read_json(source / "completion.json")
            if completion["seed"] != seed or completion["kind"] != kind:
                raise ValueError("source manual behavior identity")
            expected = _tree_hashes(source)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
            if not expected or _tree_hashes(target) != expected:
                raise AssertionError("manual reuse differs from v1: " + str(target))
            rows.append(dict(seed=seed, kind=kind, source=str(source.relative_to(ROOT)),
                destination=str(target.relative_to(ROOT)), file_count=len(expected),
                files_sha256=expected, copied_byte_for_byte=True, reexecuted=False))
    manifest = dict(parent_audit=parent_audit, controls_reexecuted=False,
                    manual_native_training_budget_equal=False, copies=rows)
    common.write_json(OUT / "manual-reuse.json", manifest)
    return manifest


def _behavior_worker(seed):
    activate()
    return run.worker((seed, "learned"))


def behavior(workers=3):
    activate()
    _check_training_barrier()
    for seed in range(3):
        path = OUT / f"seed_{seed:03d}/behavior/learned"
        if path.exists():
            raise FileExistsError(path)
    reuse_manual_controls()
    with ProcessPoolExecutor(max_workers=workers, initializer=activate) as pool:
        for completed in as_completed([pool.submit(_behavior_worker, seed) for seed in range(3)]):
            print(json.dumps(dict(event="frozen_trace_behavior_complete", **completed.result())), flush=True)
    record = dict(learned_seeds=[0, 1, 2], manual_controls_reexecuted=False,
        manual_reuse_manifest=_reference(OUT / "manual-reuse.json"),
        completions={str(seed): _reference(OUT / f"seed_{seed:03d}/behavior/learned/completion.json")
                     for seed in range(3)})
    common.write_json(OUT / "behavior-freeze.json", record)
    return record


def audit_all():
    activate()
    _check_training_barrier()
    initial_encoders = {seed: _initial_encoder(seed) for seed in range(3)}
    for seed in range(3):
        cp = audit.load_pt(common.native_checkpoint(seed))
        audit.same_tree(initial_encoders[seed], _encoder_state(cp["model"]),
                        f"seed{seed}: CE changed random encoder")
    # Extend the existing independent audit's snapshot checks without changing
    # its archived source or reconstructing a different optimizer parameter set.
    original_check = audit.check_snapshot
    checked = 0

    def check_snapshot(snapshot, label="snapshot"):
        nonlocal checked
        original_check(snapshot, label)
        if snapshot["kind"] == "learned":
            audit.same_tree(initial_encoders[snapshot["seed"]], _encoder_state(snapshot["model"]),
                            label + ": random encoder changed")
            checked += 1

    audit.check_snapshot = check_snapshot
    try:
        report = audit.audit_all(OUT)
    finally:
        audit.check_snapshot = original_check
    report.update(variant="frozen-trace-v3", audit_source_sha256=common.sha(Path(audit.__file__)),
        frozen_trace_source_sha256=common.sha(SOURCE), read_first_source_sha256=common.sha(READ_FIRST_SOURCE),
        encoder_frozen=True, encoder_matches_seeded_initialization=True,
        learned_snapshot_encoder_checks=checked,
        initial_encoder_hashes={str(seed): audit.parameter_hash(state) for seed, state in initial_encoders.items()})
    common.write_json(OUT / "audit.json", report)
    return dict(probe_count=report["probe_count"], B_block_count=report["B_block_count"],
        native_replay_count=report["native_replay_count"],
        independently_recomputed_pg_updates=report["independently_recomputed_pg_updates"],
        learned_snapshot_encoder_checks=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "train", "behavior", "audit"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("mixed",), default="mixed")
    args = parser.parse_args()
    if args.command == "declare":
        declare()
        print(json.dumps(dict(event="frozen_trace_declared", output=str(OUT))))
    elif args.command == "train":
        train_all(args.device, args.workers)
    elif args.command == "behavior":
        behavior(args.workers)
    else:
        print(json.dumps(audit_all()))


if __name__ == "__main__":
    main()
