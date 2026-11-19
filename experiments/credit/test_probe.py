"""Frozen-trajectory, grouping, preprocessing and probe-domain controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experiments.integration.diagnose import diagnose_fork as evaluate_fork
from experiments.continuity.model import AgentModel
from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.continuity.world import NEW, OnlineWorld
from experiments.credit.probe import (
    CONTENT_STATUS, FEATURES, PHASES, ProbeConfig, ProbeData, _controlled, actor_partition,
    classification_metrics, collect_checkpoint, collect_fork, fit_collection,
    fit_probe_suite, fit_ridge, fit_select, validate_partition,
)


def _config(**kwargs):
    values = dict(steps=96, actor_split=(2, 2, 2), requests_per_actor_job=16)
    values.update(kwargs)
    return ProbeConfig(**values)


def _world(**kwargs):
    values = dict(seed=31, batch_size=6, n_jobs=1, dt=.25)
    values.update(kwargs)
    return OnlineWorld(**values)


class PublicController(nn.Module):
    """A causal test policy using public pulses/receipts and a numerical h."""
    def __init__(self, world):
        super().__init__()
        self.input_size, self.n_jobs, self.n_actions = world.input_size, world.n_jobs, 11
        self.width = world.n_jobs * 5
        self.anchor = nn.Parameter(torch.zeros(()))

    def initial_state(self, batch):
        return torch.zeros(batch, self.width)

    def config(self):
        return dict(input_size=self.input_size, n_jobs=self.n_jobs, n_actions=11, width=self.width)

    def step(self, obs, state, dt):
        features = obs.reshape(-1, self.n_jobs, 24)
        memory = state.clone().reshape(-1, self.n_jobs, 5)
        memory[..., 3] += float(dt) * memory[..., 4]
        command = (features[..., 0] == 1) | (features[..., 1] == 1)
        cancel = features[..., 2] == 1
        completed = (features[..., 14:22].sum(-1) > 0) & (features[..., 22] == 1)
        memory[..., 4][cancel | completed] = 0
        memory[..., 0][command] = features[..., 3:5].argmax(-1)[command].float()
        memory[..., 1][command] = features[..., 5:9].argmax(-1)[command].float()
        memory[..., 2][command] = features[..., 9][command]
        memory[..., 3][command] = 0
        memory[..., 4][command] = 1
        version, payload = memory[..., 0].long(), memory[..., 1].long()
        finish = 3 + 4 * version + payload
        due = (memory[..., 4] == 1) & (memory[..., 3] >= memory[..., 2] - 1e-6)
        action = torch.where(command, 1 + version, torch.where(due, finish, 0))
        logits = obs.new_full((*features.shape[:2], 11), -8)
        logits[..., 0] = 4
        logits.scatter_(-1, finish[..., None], 1)
        logits.scatter_(-1, action[..., None], 8)
        return logits, memory.flatten(1)


def _encoded_dataset(condition, *, reverse=False, jobs=1):
    """Known information in h, with a controlled domain-specific sign map."""
    rows = []
    for actor in range(6):
        for job in range(jobs):
            for joint in range(8):
                for phase in (0, 2, 3, 4):
                    h = np.zeros(12, np.float32)
                    h[joint] = -1 if reverse else 1
                    h[8 + (0, 2, 3, 4).index(phase)] = 1
                    active, started = int(phase != 4), int(phase in (2, 3))
                    rows.append(dict(raw_h=h, tanh_h=np.tanh(h).astype(np.float32), obs96=h.copy(),
                                     actor=actor, job=job, request=joint, phase=phase, tick=joint * 8 + phase,
                                     condition=condition, content_status=1,
                                     payload=joint % 4, version=joint // 4, joint=joint,
                                     active=active, raw_started=started, active_started=active * started,
                                     native_joint=(joint + 1) % 8, public_payload_token=int(phase == 0),
                                     public_version_token=int(phase == 0), receipt_present=0))
    return ProbeData({name: np.stack([row[name] for row in rows]) for name in FEATURES},
                     {name: np.asarray([row[name] for row in rows], dtype="U32" if name == "condition" else np.int64)
                      for name in rows[0] if name not in FEATURES})


def _result(suite, *, scenario, feature="raw_h", target="joint", control="none"):
    return next(row for row in suite["records"] if row["scenario"] == scenario and row["feature"] == feature
                and row["target"] == target and row["control"] == control)


@pytest.mark.parametrize("condition", ("iid", "longer", "long_sparse"))
def test_collection_is_identical_to_original_own_action_fork_and_preserves_sources(condition):
    world = _world(n_jobs=2)
    model = PublicController(world)
    h = model.initial_state(6)
    model.train()
    source = state_digest({"model": model.state_dict(), "world": world.state_dict(), "h": h})
    random_before = state_digest(rng_state())
    native = evaluate_fork(model, world, h, condition=condition, steps=96)
    data, report = collect_fork(model, world, h, condition=condition, config=_config())
    assert report["end_world_hash"] == native["end_world_state_hash"]
    assert report["end_hidden_hash"] == native["end_hidden_hash"]
    assert report["action_hash"] == native["action_hash"]
    assert report["observation_hash"] == native["observation_hash"]
    assert report["end_event_hash"] == native["end_event_hash"]
    assert source == state_digest({"model": model.state_dict(), "world": world.state_dict(), "h": h})
    assert random_before == state_digest(rng_state())
    assert model.training
    assert all(report[key] == 0 for key in ("world_resets", "hidden_resets", "parameter_updates", "teacher_actions"))
    assert data.views["raw_h"].shape[1] == model.width
    assert data.views["obs96"].shape[1] == world.input_size
    assert np.array_equal(data.views["tanh_h"], torch.from_numpy(data.views["raw_h"]).tanh().numpy())


def test_collection_sampling_is_deterministic_bounded_and_keeps_requests_together():
    world = _world(n_jobs=2)
    model = PublicController(world)
    config = _config(steps=160, requests_per_actor_job=3)
    left, first = collect_fork(model, world, model.initial_state(6), config=config)
    right, second = collect_fork(model, world, model.initial_state(6), config=config)
    assert first == second
    for name in FEATURES:
        assert np.array_equal(left.views[name], right.views[name])
    partitions = actor_partition(world.actor_ids, config)
    split = validate_partition(left, partitions)
    assert [sum(value == index for value in partitions.values()) for index in range(3)] == [2, 2, 2]
    for actor in world.actor_ids:
        assert len(set(split[left.columns["actor"] == actor])) == 1
        for job in range(2):
            mask = (left.columns["actor"] == actor) & (left.columns["job"] == job)
            assert len(set(left.columns["request"][mask])) <= 3
            for request in set(left.columns["request"][mask]):
                for phase in range(len(PHASES)):
                    rows = mask & (left.columns["request"] == request) & (left.columns["phase"] == phase)
                    assert rows.sum() <= (1 if phase == 0 else 2)
    different, report = collect_fork(model, world, model.initial_state(6), config=_config(steps=160, requests_per_actor_job=30))
    for name in ("action_hash", "observation_hash", "end_hidden_hash", "end_world_hash", "end_scheduler_hash"):
        assert first[name] == report[name]
    assert len(different) > len(left)


def test_unknown_inherited_and_prior_contract_replacements_are_not_content_labels():
    world = _world(duration_range=(4, 4), gap_range=(.5, .5), window_range=(2, 2), cancel=0, replace=1)
    for _ in range(16):
        world.observe()
        first = bool((world.event_kind == NEW).any())
        world.act(world.oracle())
        if first:
            break
    model = PublicController(world)
    data, report = collect_fork(model, world, model.initial_state(6), condition="longer", config=_config(steps=160))
    inherited = data.columns["content_status"] == CONTENT_STATUS.index("inherited_unobserved")
    carryover = data.columns["content_status"] == CONTENT_STATUS.index("carryover_contract")
    assert inherited.any() and carryover.any()
    assert np.all(data.columns["joint"][inherited | carryover] == -1)
    primary = data.columns["content_status"] == CONTENT_STATUS.index("primary")
    assert primary.any() and np.all(data.columns["joint"][primary] >= 0)
    assert report["sample_exposure"]["unknown_or_out_of_scope_content"] > 0


def test_known_retired_content_and_raw_started_expiry_flag_have_explicit_meaning():
    world = _world()
    controller = PublicController(world)
    data, _ = collect_fork(controller, world, controller.initial_state(6), config=_config(steps=160))
    completion = data.columns["phase"] == PHASES.index("post_completion")
    cancellation = data.columns["phase"] == PHASES.index("post_cancel")
    assert completion.any() and cancellation.any()
    assert np.all(data.columns["active"][completion | cancellation] == 0)
    assert np.any(data.columns["joint"][completion] >= 0)
    assert np.all(data.columns["active_started"][completion | cancellation] == 0)

    # BEGIN forever yields accepted BEGIN then duplicate errors and eventual
    # expiry. The frozen evaluator intentionally retains raw started=True.
    model = AgentModel(input_size=world.input_size, n_jobs=1, width=4)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.head.bias[1] = 5
    expired, _ = collect_fork(model, world, model.initial_state(6), config=_config(steps=160))
    mask = (expired.columns["phase"] == PHASES.index("post_expiry")) & (expired.columns["joint"] >= 0)
    assert mask.any()
    assert np.all(expired.columns["active"][mask] == 0)
    assert np.any(expired.columns["raw_started"][mask] == 1)
    assert np.all(expired.columns["active_started"][mask] == 0)


def test_ridge_preprocessing_uses_training_only_and_missing_classes_are_undefined():
    x = np.asarray([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
    y = np.asarray([0, 0, 1, 1])
    fitted = fit_ridge(x, y, 2, 1e-4)
    assert np.array_equal(fitted.mean, [0, 1])
    assert fitted.scale[1] == 1
    assert np.array_equal(fitted.predict(x), y)
    # A million-unit validation/test shift cannot alter fitted preprocessing.
    shifted = np.asarray([[1e6, -1e6]])
    fitted.predict(shifted)
    assert np.array_equal(fitted.mean, [0, 1])
    missing = classification_metrics([0, 0], [0, 0], 2)
    assert missing["accuracy"] == 1 and missing["balanced_accuracy"] is None
    empty = classification_metrics([], [], 2)
    assert empty["n"] == 0 and empty["accuracy"] is None


def test_validation_selection_is_independent_of_test_values_and_ties_prefer_regularization():
    x = np.asarray([[-2.], [-1.], [1.], [2.], [-3.], [3.], [100.]])
    y = np.asarray([0, 0, 1, 1, 0, 1, 0])
    train, validation = np.arange(7) < 4, np.isin(np.arange(7), [4, 5])
    first, _ = fit_select(x, y, train, validation, 2, (1e-4, 1e-2, 1))
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[-1], changed_y[-1] = -1e9, 1
    second, _ = fit_select(changed_x, changed_y, train, validation, 2, (1e-4, 1e-2, 1))
    assert first.alpha == second.alpha == 1
    assert np.array_equal(first.weight, second.weight)
    assert np.array_equal(first.mean, second.mean)


def test_condition_trained_probe_can_decode_when_train_range_transfer_fails():
    datasets = {name: _encoded_dataset(name, reverse=name == "longer")
                for name in ("iid", "long_wait", "longer", "sparse", "long_sparse")}
    config = _config()
    partition = actor_partition(range(6), config)
    suite = fit_probe_suite(datasets, partition, config=config, controls=("none",))
    within = _result(suite, scenario="within/longer")
    transfer = _result(suite, scenario="transfer_1to8_to_8to16")
    assert within["test"]["overall"]["accuracy"] == 1
    assert transfer["test"]["overall"]["accuracy"] == 0
    assert transfer["training_conditions"] == ["iid", "long_wait", "sparse", "long_sparse"]
    assert transfer["testing_conditions"] == ["longer"]
    assert within["classification"] == "mixed_phase_within_condition"
    assert within["native_finish_head"]["overall"]["accuracy"] == 0
    for record in suite["records"]:
        assert record["input_dimension"] == 12


def test_probe_does_not_use_phase_tick_actor_or_target_metadata_as_features():
    data = _encoded_dataset("iid")
    config, partition = _config(), actor_partition(range(6), _config())
    first = fit_probe_suite({"iid": data}, partition, config=config, controls=("none",))
    changed = ProbeData(data.views, {key: value.copy() for key, value in data.columns.items()})
    changed.columns["phase"] = (changed.columns["phase"] + 3) % len(PHASES)
    changed.columns["tick"] += 99999
    second = fit_probe_suite({"iid": changed}, partition, config=config, controls=("none",))
    assert [m["parameter_hash"] for r in first["records"] for m in r["models"]] == [
        m["parameter_hash"] for r in second["records"] for m in r["models"]]
    assert [m["preprocessing_hash"] for r in first["records"] for m in r["models"]] == [
        m["preprocessing_hash"] for r in second["records"] for m in r["models"]]


def test_negative_controls_keep_random_labels_request_consistent_and_shuffle_inside_splits():
    data = ProbeData.concatenate((_encoded_dataset("iid", jobs=2), _encoded_dataset("longer", jobs=2)))
    config = _config()
    partition = actor_partition(range(6), config)
    splits = validate_partition(data, partition)
    random = _controlled(data, splits, "random_joint_labels", config.seed)
    seen = {}
    for actor, job, request, condition, label in zip(data.columns["actor"], data.columns["job"],
            data.columns["request"], data.columns["condition"], random.columns["joint"]):
        key = (int(actor), int(job), int(request), str(condition))
        assert key not in seen or seen[key] == label
        seen[key] = label
    # Encode source row identity solely for inspection of this null control.
    numbered = ProbeData({name: np.arange(len(data), dtype=np.float32)[:, None] for name in FEATURES}, data.columns)
    shuffled = _controlled(numbered, splits, "shuffled_features", config.seed)
    indices = shuffled.views["raw_h"][:, 0].astype(int)
    assert np.any(indices != np.arange(len(data)))
    assert np.array_equal(splits, splits[indices])
    assert np.array_equal(data.columns["condition"], data.columns["condition"][indices])
    assert np.array_equal(data.columns["job"], data.columns["job"][indices])
    assert np.any(data.columns["phase"] != data.columns["phase"][indices])
    assert all(np.array_equal(shuffled.views["raw_h"], shuffled.views[name]) for name in FEATURES)


def test_cached_datasets_roundtrip_and_checksum_tampering_is_rejected(tmp_path):
    data = _encoded_dataset("iid")
    path = tmp_path / "iid.npz"
    data.save(path)
    restored = ProbeData.load(path)
    assert all(np.array_equal(data.views[name], restored.views[name]) for name in FEATURES)
    with pytest.raises(FileExistsError):data.save(path)
    manifest = dict(format="credit-frozen-probe-v1", kind="frozen_state_collection", config=vars(_config()),
                    collector_source_sha256=hashlib.sha256(Path("experiments/credit/probe.py").read_bytes()).hexdigest(),
                    actor_partition={str(k): v for k, v in actor_partition(range(6), _config()).items()},
                    records=[dict(condition="iid", dataset=str(path), dataset_sha256="wrong")])
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_collection(manifest)


def test_checkpoint_collection_preserves_complete_semantics_bytes_and_rng(tmp_path):
    config = TrainConfig(batch_size=6, n_jobs=1, tbptt=4, updates=2, train_dts=(.25,),
                         replay_block_ticks=8, replay_burnin=4, replay_capacity=2, log_every=100, device="cpu")
    trainer = Trainer(Variant("tiny", width=4, policy="online"), config)
    trainer.run_updates(2)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    file_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    semantic = state_digest(trainer.semantic_state())
    rng_before = state_digest(rng_state())
    manifest = collect_checkpoint(checkpoint, tmp_path / "samples", conditions=("iid", "longer"),
                                  config=_config(steps=24), device="cpu")
    assert manifest["checkpoint_sha256"] == file_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest["checkpoint_semantic_hash"] == semantic == state_digest(trainer.semantic_state())
    assert rng_before == state_digest(rng_state())
    assert manifest["records"][0]["actor_partition"] == manifest["records"][1]["actor_partition"]
    assert all(Path(record["dataset"]).is_file() for record in manifest["records"])


def test_invalid_config_and_unfinished_forks_are_rejected():
    for kwargs in (dict(steps=0), dict(dt=0), dict(actor_split=(4, 2)), dict(ridge_alphas=(0,)),
                   dict(samples_per_request_phase=True)):
        with pytest.raises(ValueError):ProbeConfig(**kwargs)
    with pytest.raises(ValueError):actor_partition(range(6), ProbeConfig())
    world = _world()
    model = PublicController(world)
    world.observe()
    with pytest.raises(ValueError, match="completed action"):
        collect_fork(model, world, model.initial_state(6), config=_config())
