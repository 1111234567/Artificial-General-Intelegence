"""Post-v1 read-before-write follow-up using the frozen training/behavior code.

The sole policy-computation change is that both queries read the matrix that
entered the tick, before job0 and job1 delta writes.  Initialization, public
inputs, recurrent states, parameters, objectives and budgets stay as in v1.
This is declared after inspecting v1 results and reuses its existing benchmark.
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
from torch.nn import functional as F

from experiments.learned_binding.model import (
    NativeMemoryPolicy, FEATURES_PER_JOB, N_JOBS, ENCODER_WIDTH,
    KEY_DIM, N_VALUES, MEMORY_SCALARS, soft_translate,
)
from experiments.learned_binding import common, pretrain, run


ROOT = Path(__file__).resolve().parents[2]
V1_OUT = ROOT / "data/learned-binding/v1"
OUT = ROOT / "data/learned-binding/read-first-v2"
SOURCE = Path(__file__).resolve()
PLAN = SOURCE.with_name("read-first-plan.md")


class ReadFirstPolicy(NativeMemoryPolicy):
    """The original parameterization with reads preceding both tick writes."""

    def forward(self, obs, state, *, write_enabled=True, read_enabled=True,
                return_diagnostics=False):
        self._check_inputs(obs, state, write_enabled, read_enabled)
        ticks, batch = obs.shape[:2]
        encoded, next_z = self.encoder(
            obs.reshape(ticks, batch * N_JOBS, FEATURES_PER_JOB),
            state["z"].reshape(1, batch * N_JOBS, ENCODER_WIDTH))
        encoded = encoded.reshape(ticks, batch, N_JOBS, ENCODER_WIDTH)
        head_input = torch.cat((encoded, obs.reshape(
            ticks, batch, N_JOBS, FEATURES_PER_JOB)), dim=-1)
        q = F.normalize(self.query(head_input), dim=-1)
        k = F.normalize(self.key(head_input), dim=-1)
        value = self.value(head_input).softmax(-1)
        gate = self.gate(head_input).sigmoid()
        matrix = state["memory"].reshape(batch, KEY_DIM, N_VALUES)
        read_scale = self.read_scale
        reads = []
        if write_enabled:
            # Keep the frozen optimized scan, including one-time unbind and
            # weighted keys.  Only this read's position in the tick changes.
            keys_0, keys_1 = k.unbind(2)
            weighted_0, weighted_1 = (gate * k).unbind(2)
            values_0, values_1 = value.unbind(2)
            frames = zip(q.unbind(0), keys_0.unbind(0), keys_1.unbind(0),
                         weighted_0.unbind(0), weighted_1.unbind(0),
                         values_0.unbind(0), values_1.unbind(0))
            for query, key_0, key_1, weighted_0, weighted_1, value_0, value_1 in frames:
                if read_enabled:
                    reads.append((read_scale * torch.bmm(query, matrix)).softmax(-1))
                prediction_0 = torch.bmm(key_0[:, None], matrix).squeeze(1)
                matrix = torch.baddbmm(matrix, weighted_0[:, :, None],
                                      (value_0 - prediction_0)[:, None])
                prediction_1 = torch.bmm(key_1[:, None], matrix).squeeze(1)
                matrix = torch.baddbmm(matrix, weighted_1[:, :, None],
                                      (value_1 - prediction_1)[:, None])
        elif read_enabled:
            read_logits = read_scale * torch.matmul(q, matrix[None])
            reads = [read_logits.softmax(-1)]
        if read_enabled:
            read_probs = torch.stack(reads) if write_enabled else reads[0]
        else:
            read_probs = obs.new_full((ticks, batch, N_JOBS, N_VALUES), .25)
        mapped_obs = soft_translate(obs, read_probs, self.original_mappings)
        logits, next_h = self.executor(mapped_obs, state["h"])
        memory = (matrix.reshape(batch, MEMORY_SCALARS) if write_enabled
                  else state["memory"])
        next_state = dict(h=next_h,
                          z=next_z.reshape(batch, N_JOBS, ENCODER_WIDTH),
                          memory=memory)
        if not return_diagnostics:
            return logits, next_state
        diagnostics = dict(q=q, k=k, value=value, gate=gate,
                           read_probs=read_probs, mapped_obs=mapped_obs, z=encoded)
        return logits, next_state, diagnostics

    def parameter_report(self):
        report = super().parameter_report()
        report["read_write_order"] = "read_then_write_job0_then_job1"
        return report


def activate():
    """Select this class/root in the parent and every Windows-spawned worker."""
    common.OUT = OUT
    common.NativeMemoryPolicy = ReadFirstPolicy
    pretrain.OUT = OUT
    run.OUT = OUT
    return OUT


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reference(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=common.sha(path))


def _parent_audit():
    path = V1_OUT / "audit.json"
    audit = _read_json(path)
    if audit.get("probe_count") != 39 or audit.get("B_block_count") != 384:
        raise ValueError("read-first follow-up requires the completed v1 39-probe/384-block audit")
    if audit.get("native_replay_count") != 9 or audit.get("independently_recomputed_pg_updates") != 6:
        raise ValueError("read-first follow-up requires all nine v1 replays and six PG recomputations")
    for name in ("all_world_actions_and_public_rewards_replayed",
                 "all_controller_live_states_preserved_across_optimizer_updates",
                 "all_source_executors_unchanged"):
        if audit.get(name) is not True:
            raise ValueError("v1 audit check failed or absent: " + name)
    return _reference(path)


def declare():
    """Declare before v2 fitting; reuse the byte-identical v1 training manifest."""
    activate()
    for path in (SOURCE, PLAN, V1_OUT / "protocol.json", V1_OUT / "data-freeze.json",
                 V1_OUT / "training-freeze.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (OUT / "protocol.json").exists() or (OUT / "data-freeze.json").exists():
        raise FileExistsError("v2 protocol/data manifest already exists")
    parent_audit = _parent_audit()
    parent_protocol = _read_json(V1_OUT / "protocol.json")
    if parent_protocol["settings"] != common.SETTINGS:
        raise ValueError("v1 settings differ; this follow-up changes read/write order only")
    for name, digest in parent_protocol["sources"].items():
        if common.sha(ROOT / name) != digest:
            raise ValueError("frozen v1 source changed: " + name)
    data_source = _reference(V1_OUT / "data-freeze.json")
    common.declare()
    path = OUT / "protocol.json"
    protocol = _read_json(path)
    for source in (SOURCE, PLAN):
        protocol["sources"][str(source.relative_to(ROOT))] = common.sha(source)
    parent_artifacts = {name: _reference(V1_OUT / name) for name in
                        ("protocol.json", "data-freeze.json", "training-freeze.json", "audit.json")}
    for name in ("diagnostics/action-loss.json", "diagnostics/binding-probe.json"):
        if (V1_OUT / name).is_file():
            parent_artifacts[name] = _reference(V1_OUT / name)
    for source in parent_artifacts.values():
        protocol["sources"][source["path"]] = source["sha256"]
    protocol.update(experiment="learned-binding-read-first-v2",
        read_write_order="read_then_write_job0_then_job1",
        followup=dict(parent="data/learned-binding/v1", declared_after_v1_results_inspected=True,
            disclosure="Post-hoc architecture follow-up after inspecting v1 behavior and representation diagnostics; this is not an unseen new benchmark.",
            only_policy_computation_change="Read the entering matrix before both learned delta writes in each tick.",
            unchanged_initialization_parameters_state_config=True,
            unchanged_training_data_objective_and_512_update_budget=True,
            unchanged_32_block_outcome_and_heldout_protocol=True,
            parent_artifacts=parent_artifacts, parent_audit=parent_audit),
        training_data_reuse=dict(source=data_source, manifest_byte_identical=True,
            paths_remain_original_v1=True, large_training_streams_copied=False),
        manual_behavior_reuse=dict(method="byte-for-byte copytree of completed v1 behavior controls",
            kinds=["manual_hard", "manual_soft"], controls_reexecuted=False,
            manual_native_training_budget_equal=False,
            limitation="Manual historical pretraining differs from native; native v1/v2 is the same-budget read/write-order follow-up."))
    shutil.copyfile(V1_OUT / "data-freeze.json", OUT / "data-freeze.json")
    if common.sha(OUT / "data-freeze.json") != data_source["sha256"]:
        raise AssertionError("copied v1 training manifest differs")
    common.write_json(path, protocol)
    return protocol


def _train_worker(job):
    activate()
    return pretrain.train_one(job)


def train_all(device="mixed", workers=3):
    activate()
    common.check_protocol()
    if device != "mixed":
        raise ValueError("the declared follow-up retains v1 mixed devices: seed0 CUDA, seed1/2 CPU")
    if common.SETTINGS["ce_updates"] != 512 or common.SETTINGS["seeds"] != [0, 1, 2]:
        raise ValueError("v2 requires the fixed three-seed, 512-update budget")
    if (OUT / "training-freeze.json").exists():
        raise FileExistsError(OUT / "training-freeze.json")
    jobs = [(seed, ("cuda" if seed == 0 else "cpu") if device == "mixed" else device)
            for seed in range(3)]
    with ProcessPoolExecutor(max_workers=workers, initializer=activate) as pool:
        for completed in as_completed([pool.submit(_train_worker, job) for job in jobs]):
            print(json.dumps(dict(event="read_first_trained_seed", seed=completed.result())), flush=True)
    checkpoints, model_hashes, metadata = {}, {}, {}
    for seed in range(3):
        path = common.native_checkpoint(seed)
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if cp["updates"] != 512 or cp["settings"] != common.SETTINGS:
            raise ValueError("incomplete/different-budget v2 training endpoint")
        model = ReadFirstPolicy(**cp["model_config"])
        model.load_state_dict(cp["model"], strict=True)
        name = str(path.relative_to(ROOT))
        checkpoints[name] = common.sha(path)
        model_hashes[name] = common.parameter_hash(model)
        training = path.with_name("training.json")
        info = _read_json(training)
        if info["checkpoint_sha256"] != checkpoints[name] or info["executor_unchanged"] is not True:
            raise ValueError("v2 training metadata/checkpoint mismatch")
        metadata[str(seed)] = _reference(training)
    freeze = dict(all_complete_before_scoring=True, checkpoints=checkpoints,
        model_parameter_hashes=model_hashes, training_metadata=metadata,
        source_protocol_sha256=common.sha(OUT / "protocol.json"),
        architecture="ReadFirstPolicy", read_write_order="read_then_write_job0_then_job1",
        seeds=[0, 1, 2], updates_per_seed=512, devices={str(s): d for s, d in jobs})
    common.write_json(OUT / "training-freeze.json", freeze)
    return freeze


def _tree_hashes(folder):
    return {str(path.relative_to(folder)): common.sha(path)
            for path in sorted(Path(folder).rglob("*")) if path.is_file()}


def reuse_manual_controls():
    """Copy existing controls, preserving every source file byte for byte."""
    activate()
    run.check_freeze()
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
    run.check_freeze()
    for seed in range(3):
        path = OUT / f"seed_{seed:03d}/behavior/learned"
        if path.exists():
            raise FileExistsError(path)
    reuse_manual_controls()
    with ProcessPoolExecutor(max_workers=workers, initializer=activate) as pool:
        for completed in as_completed([pool.submit(_behavior_worker, seed) for seed in range(3)]):
            print(json.dumps(dict(event="read_first_behavior_complete", **completed.result())), flush=True)
    record = dict(learned_seeds=[0, 1, 2], manual_controls_reexecuted=False,
        manual_reuse_manifest=_reference(OUT / "manual-reuse.json"),
        completions={str(seed): _reference(OUT / f"seed_{seed:03d}/behavior/learned/completion.json")
                     for seed in range(3)})
    common.write_json(OUT / "behavior-freeze.json", record)
    return record


def audit_all():
    activate()
    from experiments.learned_binding import audit
    audit.NativeMemoryPolicy = ReadFirstPolicy
    report = audit.audit_all(OUT)
    report.update(variant="read-first-v2", audit_source_sha256=common.sha(Path(audit.__file__)),
                  read_first_source_sha256=common.sha(SOURCE))
    common.write_json(OUT / "audit.json", report)
    return dict(probe_count=report["probe_count"], B_block_count=report["B_block_count"],
        native_replay_count=report["native_replay_count"],
        independently_recomputed_pg_updates=report["independently_recomputed_pg_updates"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "train", "behavior", "audit"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("mixed",), default="mixed")
    args = parser.parse_args()
    if args.command == "declare":
        declare()
        print(json.dumps(dict(event="read_first_declared", output=str(OUT))))
    elif args.command == "train":
        train_all(args.device, args.workers)
    elif args.command == "behavior":
        behavior(args.workers)
    else:
        print(json.dumps(audit_all()))


if __name__ == "__main__":
    main()
