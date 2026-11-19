"""Phase-agnostic observational readouts of complete frozen continuations.

Only raw h, the model's exact tanh(h), or the actual public observation are
inputs to a probe. Job-specific heads do not receive phase, age, actor/request
identity, world flags or targets as features. Probe predictions never act.
Actor partitions are shared across all conditions; a request is never split by
tick. A condition-trained 8--16 diagnostic is explicitly distinct from transfer
of a readout trained only on 1--8 observations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, state_digest
from experiments.continuity.world import CANCEL, NEW, REPLACE, OnlineWorld
from experiments.integration.diagnose import CONDITIONS
from experiments.integration.run import load_checkpoint

FORMAT_VERSION = "credit-frozen-probe-v1"
CONDITION_NAMES = ("iid", "long_wait", "longer", "sparse", "long_sparse")
TRANSFER_TRAIN = ("iid", "long_wait", "sparse", "long_sparse")
PHASES = ("cue", "unstarted", "waiting", "legal", "post_completion", "post_cancel", "post_expiry", "inactive_unknown")
FEATURES = ("raw_h", "tanh_h", "obs96")
TARGETS = {"payload": 4, "version": 2, "joint": 8, "active": 2, "raw_started": 2, "active_started": 2}
CONTENT_TARGETS = ("payload", "version", "joint")
SPLITS = ("train", "validation", "test")
CONTENT_STATUS = ("no_request_yet", "primary", "inherited_unobserved", "carryover_contract")
_EPS = 1e-6


@dataclass(frozen=True)
class ProbeConfig:
    steps: int = 1024
    dt: float = .25
    samples_per_request_phase: int = 2
    requests_per_actor_job: int = 32
    actor_split: tuple[int, int, int] = (40, 12, 12)
    seed: int = 91731
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0)

    def __post_init__(self):
        for name in ("steps", "samples_per_request_phase", "requests_per_actor_job"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        object.__setattr__(self, "actor_split", tuple(self.actor_split))
        if len(self.actor_split) != 3 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in self.actor_split):
            raise ValueError("actor_split requires three positive counts")
        object.__setattr__(self, "ridge_alphas", tuple(map(float, self.ridge_alphas)))
        if not self.ridge_alphas or any(not math.isfinite(x) or x <= 0 for x in self.ridge_alphas):
            raise ValueError("ridge_alphas must be finite and positive")


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _priority(*values) -> int:
    data = json.dumps(values, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def actor_partition(actor_ids: Iterable[int], config: ProbeConfig) -> dict[int, int]:
    identities = tuple(map(int, actor_ids))
    if len(identities) != len(set(identities)) or len(identities) != sum(config.actor_split):
        raise ValueError("distinct actor IDs must match actor_split total")
    # A private, identity-only assignment; weights, labels and phase never enter.
    ordered = sorted(identities, key=lambda actor: (_priority(config.seed, "actor", actor), actor))
    result, offset = {}, 0
    for split, count in enumerate(config.actor_split):
        result.update((actor, split) for actor in ordered[offset:offset + count])
        offset += count
    return result


@dataclass
class ProbeData:
    """Compact CPU samples. ``columns`` are metadata/labels, never inputs."""
    views: dict[str, np.ndarray]
    columns: dict[str, np.ndarray]

    def __len__(self):
        return len(self.columns["actor"])

    def subset(self, mask) -> "ProbeData":
        return ProbeData({k: v[mask] for k, v in self.views.items()},
                         {k: v[mask] for k, v in self.columns.items()})

    @classmethod
    def concatenate(cls, values: Iterable["ProbeData"]) -> "ProbeData":
        values = tuple(values)
        if not values:
            raise ValueError("no probe datasets")
        return cls({k: np.concatenate([v.views[k] for v in values]) for k in FEATURES},
                   {k: np.concatenate([v.columns[k] for v in values]) for k in values[0].columns})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{f"view__{k}": v for k, v in self.views.items()},
                                **{f"column__{k}": v for k, v in self.columns.items()})
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ProbeData":
        with np.load(path, allow_pickle=False) as data:
            return cls({k: data[f"view__{k}"].copy() for k in FEATURES},
                       {k.removeprefix("column__"): data[k].copy() for k in data.files if k.startswith("column__")})


def _exposure(data: ProbeData, mask=None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(data), bool)
    indices = np.flatnonzero(mask)
    keys = {(int(data.columns["actor"][i]), int(data.columns["job"][i]),
             int(data.columns["request"][i]), str(data.columns["condition"][i])) for i in indices}
    return {
        "samples": len(indices), "actors": len(set(map(int, data.columns["actor"][mask]))),
        "request_groups": len(keys), "known_primary_content": int((data.columns["joint"][mask] >= 0).sum()),
        "unknown_or_out_of_scope_content": int((data.columns["joint"][mask] < 0).sum()),
        "unknown_inactive_content": int((mask & (data.columns["joint"] < 0) & (data.columns["active"] == 0)).sum()),
        "content_status": {name: int((mask & (data.columns["content_status"] == index)).sum())
                           for index, name in enumerate(CONTENT_STATUS)},
        "phases": {phase: int((mask & (data.columns["phase"] == index)).sum()) for index, phase in enumerate(PHASES)},
        "per_actor": {str(actor): int((mask & (data.columns["actor"] == actor)).sum())
                      for actor in sorted(set(map(int, data.columns["actor"][mask])))},
    }


def validate_partition(data: ProbeData, partition: dict[int, int]) -> np.ndarray:
    splits = np.asarray([partition[int(actor)] for actor in data.columns["actor"]], dtype=np.int8)
    seen = {}
    for actor, job, request, condition, split in zip(data.columns["actor"], data.columns["job"],
            data.columns["request"], data.columns["condition"], splits):
        key = (int(actor), int(job), int(request), str(condition))
        if key in seen and seen[key] != split:
            raise AssertionError("request crossed probe partitions")
        seen[key] = split
    return splits


@torch.inference_mode()
def collect_fork(model, world: OnlineWorld, hidden: torch.Tensor, *, condition="iid",
                 config: ProbeConfig | None = None, device=None) -> tuple[ProbeData, dict[str, Any]]:
    """Sample an unchanged own-action continuation; private labels only observe."""
    config = config or ProbeConfig()
    if condition not in CONDITION_NAMES:
        raise ValueError("unknown probe condition")
    partition = actor_partition(world.actor_ids, config)
    saved = world.state_dict()
    if not saved["acted"]:
        raise ValueError("collection must fork after a completed action")
    if hidden.shape != (world.batch_size, model.width) or model.input_size != world.input_size or model.n_jobs != world.n_jobs or model.n_actions != 11:
        raise ValueError("model/world/hidden dimensions differ")
    before = state_digest({"model": model.state_dict(), "world": saved, "h": hidden})
    target_device = torch.device(hidden.device if device is None else device)
    live_model = deepcopy(model).to(target_device).eval()
    live_world = OnlineWorld.from_state_dict(saved, device=world.device)
    state = hidden.detach().to(target_device).clone()
    fork_tick = world.tick
    live_world.set_regime(**CONDITIONS[condition], name=f"evaluation/{condition}", dt=config.dt)
    regime_id = live_world.regime_id
    shape = (world.batch_size, world.n_jobs)
    primary_content = np.zeros(shape, bool)
    content_status = np.where(_array(live_world.request_id) >= 0, 2, 0).astype(np.int8)
    ended = np.zeros(shape, np.int8)  # 1 completion, 2 cancel, 3 expiry; 0 inherited/unknown
    ack_finish = (_array(live_world._ack_result) == 1) & (_array(live_world._ack_action) >= 3)
    ended[ack_finish] = 1
    groups: dict[tuple[int, int, int, int], list[tuple[int, dict]]] = {}
    phase_opportunities = dict.fromkeys(PHASES, 0)
    # Retain the frozen diagnostic hash domain for direct trajectory identity.
    action_digest = hashlib.sha256(b"integration-own-trace-v1/actions")
    obs_digest = hashlib.sha256(b"integration-own-trace-v1/observations")
    parameter_hash = state_digest(live_model.state_dict())
    for _ in range(config.steps):
        previous_active, previous_deadline = _array(live_world.active), _array(live_world.deadline)
        observation = live_world.observe()
        kind = _array(live_world.event_kind)
        command = (kind == NEW) | (kind == REPLACE)
        expired = previous_active & (live_world.now > previous_deadline + _EPS)
        ended[expired] = 3
        ended[kind == CANCEL] = 2
        ended[command] = 0
        primary_content[command] = ((_array(live_world.event_regime_id) == regime_id) & (
            _array(live_world.event_request_id) // world.n_jobs > fork_tick))[command]
        content_status[command] = np.where(primary_content[command], 1, 3)
        logits, state = live_model.step(observation.to(target_device), state, live_world.elapsed)
        if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite frozen probe continuation")
        # Exactly match the original head's floating-point nonlinearity.
        raw_h, tanh_h, observations = _array(state), _array(state.tanh()), _array(observation)
        values = _array(logits)
        actions = values.argmax(-1)
        native = values[..., 3:].argmax(-1)
        active, started = _array(live_world.active), _array(live_world.started)
        ready = live_world.now >= _array(live_world.ready_time) - _EPS
        phases = np.where(command, 0, np.where(active, np.where(~started, 1, np.where(ready, 3, 2)),
                    np.where(ended == 1, 4, np.where(ended == 2, 5, np.where(ended == 3, 6, 7)))))
        versions, payloads, requests = _array(live_world.version), _array(live_world.payload), _array(live_world.request_id)
        public = observations.reshape(world.batch_size, world.n_jobs, 24)
        for actor in range(world.batch_size):
            actor_id = int(world.actor_ids[actor])
            for job in range(world.active_jobs):
                phase, request = int(phases[actor, job]), int(requests[actor, job])
                phase_opportunities[PHASES[phase]] += 1
                key = (actor_id, job, request, phase)
                priority = _priority(config.seed, "tick", condition, *key, live_world.tick)
                slots = groups.setdefault(key, [])
                cap = 1 if phase == 0 else config.samples_per_request_phase
                replace = None if len(slots) < cap else max(range(len(slots)), key=lambda index: slots[index][0])
                if replace is not None and priority >= slots[replace][0]:
                    continue
                known = bool(primary_content[actor, job])
                row = dict(actor=actor_id, job=job, request=request, phase=phase,
                           tick=live_world.tick, condition=condition,
                           content_status=int(content_status[actor, job]),
                           payload=int(payloads[actor, job]) if known else -1,
                           version=int(versions[actor, job]) if known else -1,
                           joint=int(4 * versions[actor, job] + payloads[actor, job]) if known else -1,
                           active=int(active[actor, job]), raw_started=int(started[actor, job]),
                           active_started=int(active[actor, job] and started[actor, job]),
                           native_joint=int(native[actor, job]),
                           public_payload_token=int(public[actor, job, 5:9].sum() > 0 or public[actor, job, 14:22].sum() > 0),
                           public_version_token=int(public[actor, job, 3:5].sum() > 0 or public[actor, job, 12:22].sum() > 0),
                           receipt_present=int(public[actor, job, 22:24].sum() > 0),
                           raw_h=raw_h[actor].copy(), tanh_h=tanh_h[actor].copy(), obs96=observations[actor].copy())
                if replace is None:
                    slots.append((priority, row))
                else:
                    slots[replace] = priority, row
        action_digest.update(np.ascontiguousarray(actions, dtype=np.int64).tobytes())
        obs_digest.update(np.ascontiguousarray(observations).tobytes())
        live_world.act(torch.as_tensor(actions, dtype=torch.long, device=live_world.device))
        accepted_finish = (_array(live_world.last_result) == 1) & (actions >= 3)
        ended[accepted_finish] = 1
    requests_by_actor_job: dict[tuple[int, int], set[int]] = {}
    for actor, job, request, _ in groups:
        requests_by_actor_job.setdefault((actor, job), set()).add(request)
    selected = {(actor, job, request) for (actor, job), identities in requests_by_actor_job.items()
                for request in sorted(identities, key=lambda value: (
                    _priority(config.seed, "request", condition, actor, job, value), value))[:config.requests_per_actor_job]}
    rows = [row for key, slots in sorted(groups.items()) if key[:3] in selected
            for _, row in sorted(slots, key=lambda item: item[1]["tick"])]
    views = {name: np.stack([row[name] for row in rows]) for name in FEATURES}
    names = tuple(name for name in rows[0] if name not in FEATURES)
    columns = {name: np.asarray([row[name] for row in rows], dtype="U32" if name == "condition" else (
        np.uint64 if name == "actor" else np.int64)) for name in names}
    data = ProbeData(views, columns)
    splits = validate_partition(data, partition)
    if before != state_digest({"model": model.state_dict(), "world": world.state_dict(), "h": hidden}):
        raise AssertionError("probe collection changed source model/world/h")
    if parameter_hash != state_digest(live_model.state_dict()):
        raise AssertionError("probe collection changed frozen parameters")
    return data, {
        "condition": condition, "config": asdict(config), "fork_tick": fork_tick,
        "fork_time": world.now, "end_tick": live_world.tick, "end_time": live_world.now,
        "actor_partition": {str(actor): SPLITS[split] for actor, split in partition.items()},
        "source_state_hash": before, "start_world_hash": state_digest(saved),
        "end_world_hash": state_digest(live_world.state_dict()), "end_hidden_hash": state_digest({"hidden": state}),
        "start_event_hash": world.event_hash, "end_event_hash": live_world.event_hash,
        "start_scheduler_hash": state_digest(saved["scheduler"]),
        "end_scheduler_hash": state_digest(live_world.state_dict()["scheduler"]),
        "parameter_hash": parameter_hash, "action_hash": action_digest.hexdigest(),
        "observation_hash": obs_digest.hexdigest(), "available_opportunities_by_phase": phase_opportunities,
        "sample_exposure": _exposure(data),
        "split_exposure": {name: _exposure(data, splits == index) for index, name in enumerate(SPLITS)},
        "world_resets": 0, "hidden_resets": 0, "parameter_updates": 0, "teacher_actions": 0,
        "sampling": "lowest private hash priorities; request cap independent of content, phase availability and outcome",
        "content_scope": "observed post-fork evaluation-regime NEW/REPLACE, including retired content; inherited/carryover labels masked",
    }


@dataclass
class RidgeReadout:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    alpha: float

    def predict(self, x):
        return (((np.asarray(x, np.float64) - self.mean) / self.scale) @ self.weight + self.bias).argmax(-1)


def fit_ridge(x: np.ndarray, y: np.ndarray, n_classes: int, alpha: float) -> RidgeReadout:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.int64)
    if x.ndim != 2 or not len(x) or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("finite nonempty training matrix and matching labels required")
    if np.any((y < 0) | (y >= n_classes)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid labels or ridge alpha")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    target = np.eye(n_classes, dtype=np.float64)[y]
    bias = target.mean(0)
    covariance = z.T @ z / len(z)
    covariance.flat[::covariance.shape[0] + 1] += alpha
    weight = np.linalg.solve(covariance, z.T @ (target - bias) / len(z))
    return RidgeReadout(mean, scale, weight, bias, float(alpha))


def classification_metrics(y, prediction, n_classes: int, majority_class: int | None = None) -> dict[str, Any]:
    y, prediction = np.asarray(y, np.int64), np.asarray(prediction, np.int64)
    if len(y) != len(prediction) or np.any((y < 0) | (y >= n_classes)) or np.any((prediction < 0) | (prediction >= n_classes)):
        raise ValueError("classification labels outside declared classes")
    confusion = np.bincount(y * n_classes + prediction, minlength=n_classes**2).reshape(n_classes, n_classes)
    support = confusion.sum(1)
    recalls = np.divide(np.diag(confusion), support, out=np.zeros(n_classes), where=support > 0)
    return {
        "n": len(y), "correct": int(np.trace(confusion)),
        "accuracy": float(np.trace(confusion) / len(y)) if len(y) else None,
        "balanced_accuracy": float(recalls.mean()) if np.all(support > 0) else None,
        "class_support": support.tolist(), "confusion": confusion.tolist(),
        "uniform_chance": 1 / n_classes,
        "train_majority_accuracy": float((y == majority_class).mean()) if len(y) and majority_class is not None else None,
    }


def _selection_score(metrics):
    return metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else metrics["accuracy"]


def fit_select(x, y, train, validation, n_classes, alphas):
    candidates, chosen, best = [], None, -float("inf")
    for alpha in alphas:
        model = fit_ridge(x[train], y[train], n_classes, alpha)
        metric = classification_metrics(y[validation], model.predict(x[validation]), n_classes)
        score = _selection_score(metric)
        candidates.append(dict(alpha=alpha, validation=metric))
        if score is not None and (score > best or (score == best and (chosen is None or alpha > chosen.alpha))):
            chosen, best = model, score
    return chosen, candidates


def _controlled(data: ProbeData, splits: np.ndarray, control: str, seed: int) -> ProbeData:
    if control == "none":
        return data
    if control == "random_joint_labels":
        columns = {key: value.copy() for key, value in data.columns.items()}
        known = columns["joint"] >= 0
        values = np.asarray([_priority(seed, "random-joint", int(actor), int(job), int(request), str(condition)) % 8
                             for actor, job, request, condition in zip(columns["actor"], columns["job"],
                                 columns["request"], columns["condition"])], dtype=np.int64)
        columns["joint"][known], columns["payload"][known], columns["version"][known] = values[known], values[known] % 4, values[known] // 4
        return ProbeData(data.views, columns)
    if control != "shuffled_features":
        raise ValueError("unknown probe control")
    permutation = np.arange(len(data))
    for condition in np.unique(data.columns["condition"]):
        for job in np.unique(data.columns["job"]):
            for split in range(3):
                indices = np.flatnonzero((data.columns["condition"] == condition) & (data.columns["job"] == job) & (splits == split))
                rng = np.random.default_rng(_priority(seed, "shuffle", str(condition), int(job), split))
                permutation[indices] = rng.permutation(indices)
    return ProbeData({key: value[permutation] for key, value in data.views.items()}, data.columns)


def _evaluate_groups(data, selected, predictions, n_classes, majority_by_job, target):
    y = data.columns[target]
    # The majority baseline is also job-specific and fitted on training only.
    majority = np.asarray([majority_by_job.get(int(job), -1) for job in data.columns["job"]])
    def metrics(mask):
        value = classification_metrics(y[mask], predictions[mask], n_classes)
        value["train_majority_accuracy"] = float((y[mask] == majority[mask]).mean()) if mask.any() else None
        value["exposure"] = _exposure(data, mask)
        return value
    return {
        "overall": metrics(selected),
        "by_phase": {phase: metrics(selected & (data.columns["phase"] == index)) for index, phase in enumerate(PHASES)},
        "by_condition": {str(condition): metrics(selected & (data.columns["condition"] == condition))
                         for condition in np.unique(data.columns["condition"][selected])},
        "by_public_payload_token": {str(flag): metrics(selected & (data.columns["public_payload_token"] == flag)) for flag in (0, 1)},
        "per_actor": {str(actor): metrics(selected & (data.columns["actor"] == actor))
                      for actor in sorted(set(map(int, data.columns["actor"][selected])))},
    }


def fit_probe_suite(datasets: dict[str, ProbeData], partition: dict[int, int], *,
                    config: ProbeConfig | None = None, controls=("none", "random_joint_labels", "shuffled_features")) -> dict[str, Any]:
    """Fit CPU ridge readouts on cached data; never load or act in a world."""
    config = config or ProbeConfig()
    data = ProbeData.concatenate(datasets.values())
    splits = validate_partition(data, partition)
    scenarios = [(f"within/{name}", (name,), (name,), "mixed_phase_within_condition") for name in datasets]
    if "longer" in datasets and all(name in datasets for name in TRANSFER_TRAIN):
        scenarios.append(("transfer_1to8_to_8to16", TRANSFER_TRAIN, ("longer",), "probe_domain_transfer"))
    results = []
    for control in controls:
        controlled = _controlled(data, splits, control, config.seed)
        for scenario, training_conditions, testing_conditions, classification in scenarios:
            in_train_domain = np.isin(data.columns["condition"], training_conditions)
            in_test_domain = np.isin(data.columns["condition"], testing_conditions)
            for view in FEATURES:
                x = controlled.views[view]
                for target, n_classes in TARGETS.items():
                    if control != "none" and target not in CONTENT_TARGETS:
                        continue
                    y = controlled.columns[target]
                    models, majority_by_job = [], {}
                    predictions = np.full(len(data), -1, np.int64)
                    test = (splits == 2) & in_test_domain & (y >= 0)
                    for job in sorted(set(map(int, data.columns["job"]))):
                        job_mask = data.columns["job"] == job
                        train = job_mask & in_train_domain & (splits == 0) & (y >= 0)
                        validation = job_mask & in_train_domain & (splits == 1) & (y >= 0)
                        if not train.any() or not validation.any():
                            models.append(dict(job=job, status="missing_train_or_validation", train=_exposure(data, train), validation=_exposure(data, validation)))
                            continue
                        fitted, candidates = fit_select(x, y, train, validation, n_classes, config.ridge_alphas)
                        if fitted is None:
                            continue
                        majority_by_job[job] = int(np.bincount(y[train], minlength=n_classes).argmax())
                        predict_here = test & job_mask
                        predictions[predict_here] = fitted.predict(x[predict_here])
                        models.append(dict(job=job, status="fitted", alpha=fitted.alpha,
                                           train=_exposure(data, train), validation=_exposure(data, validation),
                                           train_class_support=np.bincount(y[train], minlength=n_classes).tolist(),
                                           train_metrics=classification_metrics(y[train], fitted.predict(x[train]), n_classes),
                                           majority_class=majority_by_job[job], candidates=candidates,
                                           constant_feature_dimensions=int((x[train].std(0) <= 1e-8).sum()),
                                           preprocessing_hash=state_digest({"mean": torch.from_numpy(fitted.mean), "scale": torch.from_numpy(fitted.scale)}),
                                           parameter_hash=state_digest({"weight": torch.from_numpy(fitted.weight), "bias": torch.from_numpy(fitted.bias)})))
                    scored = test & (predictions >= 0)
                    result = dict(scenario=scenario, classification=classification, control=control, feature=view, target=target,
                                  training_conditions=list(training_conditions), testing_conditions=list(testing_conditions),
                                  input_dimension=x.shape[1], models=models,
                                  test=_evaluate_groups(controlled, scored, predictions, n_classes, majority_by_job, target),
                                  excluded_test_samples=int((test & ~scored).sum()))
                    if control == "none" and target in CONTENT_TARGETS:
                        native = data.columns["native_joint"]
                        if target == "payload":
                            native = native % 4
                        elif target == "version":
                            native = native // 4
                        result["native_finish_head"] = _evaluate_groups(data, scored, native, n_classes, majority_by_job, target)
                    results.append(result)
    return {
        "format": FORMAT_VERSION, "kind": "observational_ridge_probe", "config": asdict(config),
        "actor_partition": {str(actor): SPLITS[split] for actor, split in partition.items()},
        "exposure": {name: _exposure(data, splits == index) for index, name in enumerate(SPLITS)},
        "records": results,
        "definitions": {
            "inputs": "Only the named view; one head per fixed job, no phase/time/request/actor/world metadata input.",
            "tanh_h": "Exact model-dtype torch tanh(h), saved before float64 ridge computations.",
            "obs96": "Actual public observation received on that tick, including public commands and receipts; no added environment metadata.",
            "ridge": "Train-only mean/std; constant std<=1e-8 uses scale1. Fit one-hot least squares with covariance/n + alpha I; unpenalized intercept.",
            "selection": "Validation balanced accuracy when all declared classes occur, else validation accuracy; ties choose larger alpha. No test selection.",
            "balanced_accuracy": "Mean recall over all declared classes; undefined if any declared class has zero support.",
            "controls": "Random joint labels are request-consistent. Features shuffle globally within each condition/job/split, NOT phase-preserving; all views use the same permutation.",
            "status_targets": "active and raw started are frozen-world flags before action; active_started is their conjunction. Raw started may persist after expiry.",
            "within_longer": "A readout trained with 8--16 examples is an in-condition diagnostic, not zero-shot agent generalization.",
        },
        "limitations": [
            "Successful decoding shows accessibility to this supervised probe, not that the original controller uses it or that it controls correctly.",
            "Failed linear decoding does not prove information absence; finite actor/sample coverage and readout expressivity remain limits.",
            "Train-range transfer failure with successful within-condition decoding is probe-domain failure, not by itself erased state information.",
            "Phase/age populations depend on the actual policy and censoring; no phase/window average is an encoding or information-retention proof.",
            "Content in public command/receipt features can support immediate decoding; obs-only and no-public-payload-token slices expose this limitation.",
            "Independent replication is by training seed. Actors, requests and repeated frozen forks are not independent trained models.",
        ],
    }


def collect_checkpoint(checkpoint: str | Path, dataset_dir: str | Path, *, conditions=CONDITION_NAMES,
                        config: ProbeConfig | None = None, device="cpu") -> dict[str, Any]:
    config = config or ProbeConfig()
    checkpoint, folder = Path(checkpoint).resolve(), Path(dataset_dir).resolve()
    conditions = tuple(conditions)
    if not conditions or len(set(conditions)) != len(conditions) or any(name not in CONDITION_NAMES for name in conditions):
        raise ValueError("distinct known conditions required")
    checkpoint_hash = _hash_file(checkpoint)
    source_hash = _hash_file(__file__)
    with _preserve_process_state():
        trainer = load_checkpoint(checkpoint, device=device, strict_sources=True)
        semantic = state_digest(trainer.semantic_state())
        partition = actor_partition(trainer.world.actor_ids, config)
        records = []
        for condition in conditions:
            data, record = collect_fork(trainer.model, trainer.world, trainer.h, condition=condition, config=config, device=device)
            path = folder / f"{condition}.npz"
            data.save(path)
            record.update(dataset=str(path), dataset_sha256=_hash_file(path))
            records.append(record)
        if semantic != state_digest(trainer.semantic_state()):
            raise AssertionError("collector changed source trainer including optimizer/RNG/replay/local state")
        result = dict(format=FORMAT_VERSION, kind="frozen_state_collection", config=asdict(config),
                      checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, checkpoint_semantic_hash=semantic,
                      training_seed=trainer.config.seed, training_arm=trainer.adaptation_mode,
                      checkpoint_source_hashes=trainer.source_hashes, collector_source_sha256=source_hash,
                      actor_partition={str(actor): int(split) for actor, split in partition.items()}, records=records)
    if checkpoint_hash != _hash_file(checkpoint) or source_hash != _hash_file(__file__):
        raise AssertionError("checkpoint or collector source changed during collection")
    return result


def fit_collection(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("format") != FORMAT_VERSION or manifest.get("kind") != "frozen_state_collection":
        raise ValueError("unsupported collection format")
    if manifest["collector_source_sha256"] != _hash_file(__file__):
        raise ValueError("collection source differs; use its frozen source snapshot")
    datasets = {}
    for record in manifest["records"]:
        if _hash_file(record["dataset"]) != record["dataset_sha256"]:
            raise ValueError("cached probe dataset hash mismatch")
        datasets[record["condition"]] = ProbeData.load(record["dataset"])
    result = fit_probe_suite(datasets, {int(actor): int(split) for actor, split in manifest["actor_partition"].items()},
                             config=ProbeConfig(**manifest["config"]))
    result["collection"] = manifest
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--collection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--conditions", nargs="+", choices=CONDITION_NAMES, default=CONDITION_NAMES)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.output.exists():
        parser.error("positive threads and a new output path required")
    torch.set_num_threads(args.threads)
    if args.collection:
        if args.collect_only:
            parser.error("--collect-only requires --checkpoint")
        result = fit_collection(json.loads(args.collection.read_text(encoding="utf-8")))
    else:
        config = ProbeConfig(steps=args.steps)
        manifest = collect_checkpoint(args.checkpoint, args.dataset_dir or args.output.with_suffix(".datasets"),
                                      conditions=args.conditions, config=config, device=args.device)
        result = manifest if args.collect_only else fit_collection(manifest)
    atomic_json(args.output, result)
    print(json.dumps(dict(output=str(args.output.resolve()), kind=result.get("kind"), records=len(result["records"]))))


if __name__ == "__main__":
    main()
