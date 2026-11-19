"""Independent ledger counting and complete native replay of saved probes.

This script never starts a formal evaluation.  ``--self-test`` creates a small
temporary fixture; the ordinary command requires every declared evaluation
artifact to exist before writing the final audit.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state import evaluate as evaluation
from experiments.associative_state import train as training
from experiments.associative_state.bridge import PublicReceiptBinder, RandomMappingWorld
from experiments.learn_remember.gru_train import GRUAgent

REPLAY_CASES = ((0, "add", "immediate"), (0, "delta", "delay"),
                (0, "recurrent", "adapt"), (1, "add", "theta_only"),
                (1, "delta", "clear"), (1, "recurrent", "no_read"),
                (2, "add", "no_write"), (2, "delta", "revise"),
                (2, "recurrent", "immediate"))
ARRAY_NAMES = {"obs", "actions", "prediction", "event_kind", "event_rid",
               "event_eligible", "event_family", "raw", "target", "mature_mask",
               "mature_rid", "mature_done", "mature_strict", "active"}
GROUPS = ("all", "A", "B", "A_conflict", "B_conflict", "A_shared", "B_shared")


class AuditError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def tensor_equal(left, right):
    return array_equal(left.detach().cpu().numpy(), right.detach().cpu().numpy())


def same_tree(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and tensor_equal(left, right), path)
    elif isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), path + ": keys")
        for key in left:
            same_tree(left[key], right[key], path + "." + str(key))
    elif isinstance(left, (tuple, list)):
        require(type(left) is type(right) and len(left) == len(right), path + ": sequence")
        for index, (x, y) in enumerate(zip(left, right)):
            same_tree(x, y, path + f"[{index}]")
    else:
        require(left == right, path + ": value")


def parameter_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_bundle(prefix):
    prefix = Path(prefix)
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    require(file_hash(prefix.with_suffix(".npz")) == report["arrays_sha256"], str(prefix) + ": npz hash")
    require(file_hash(prefix.with_suffix(".pt")) == report["states_sha256"], str(prefix) + ": pt hash")
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(set(arrays) == ARRAY_NAMES, str(prefix) + ": array keys")
    states = load_pt(prefix.with_suffix(".pt"))
    require(set(states) == {"start", "final"}, str(prefix) + ": snapshot keys")
    return report, arrays, states


def check_snapshot(snapshot, name):
    require(parameter_hash(snapshot["model"]) == snapshot["parameter_hash"], name + ": parameter hash")
    require(tensor_hash(snapshot["memory"]) == snapshot["memory_hash"], name + ": memory hash")
    for label in ("memory", "hidden"):
        require(torch.isfinite(snapshot[label]).all().item(), name + ": finite " + label)
    for key, value in snapshot["model"].items():
        require(torch.isfinite(value).all().item(), name + ": finite parameter " + key)


def model_from_parameters(parameters, state):
    config = {key: parameters[key] for key in ("kind", "key_dim", "n_keys", "n_values")}
    model = evaluation.AssociationMemory(**config)
    model.load_state_dict(state, strict=True)
    return model.eval()


def recount(report, arrays, states):
    """Join independently enumerated cues and maturities; derive censoring by time."""
    start, final = states["start"], states["final"]
    check_snapshot(start, "start")
    check_snapshot(final, "final")
    world = start["world"]
    batch, jobs = world["spec"]["batch_size"], world["spec"]["n_jobs"]
    obs = arrays["obs"]
    require(obs.ndim == 3 and obs.shape[1:] == (batch, jobs * 28), "observation shape")
    ticks = len(obs)
    require(ticks > 0, "empty evaluation")
    for name, value in arrays.items():
        expected = (ticks, batch, jobs * 28) if name == "obs" else (ticks, batch, jobs)
        require(value.shape == expected, name + ": shape")
    require(np.isfinite(obs).all(), "finite observations")
    public = obs.reshape(ticks, batch, jobs, 28)
    kinds = arrays["event_kind"]
    parsed_kind = np.where(public[..., :3].sum(-1) > 0, public[..., :3].argmax(-1) + 1, 0)
    require(np.array_equal(kinds, parsed_kind), "public event kinds")
    commands = (kinds == 1) | (kinds == 2)
    require(np.all((arrays["actions"] >= 0) & (arrays["actions"] < 11)), "native action codes")
    require(np.all((arrays["prediction"] >= 0) & (arrays["prediction"] < 4)), "prediction codes")
    require(np.all(~arrays["event_eligible"] | commands), "eligibility only on commands")
    require(np.all(~arrays["mature_strict"] | arrays["mature_done"]), "strict implies ordinary")
    require(np.all(~arrays["mature_done"] | arrays["mature_mask"]), "ordinary implies maturity")
    require(np.array_equal(public[..., 5:9].argmax(-1)[commands], arrays["raw"][commands]), "public raw symbols")
    require(np.array_equal(public[..., 24:28].argmax(-1)[commands], arrays["event_family"][commands]), "public contexts")

    contexts = tuple(world["random_mapping"]["contexts"])
    table = world["random_mapping"]["mappings"].numpy()
    start_tick, dt = world["tick"], world["spec"]["dt"]
    absolute_ticks = start_tick + 1 + np.arange(ticks)
    times = world["now"] + (np.arange(ticks) + (1 if start_tick >= 0 else 0)) * dt
    require(report["start_tick"] == start_tick and report["end_tick"] == int(absolute_ticks[-1]), "reported time boundaries")
    require(final["world"]["tick"] == int(absolute_ticks[-1]), "final world tick")
    require(final["world"]["now"] == float(times[-1]), "final world time")
    require(report["start_work"] == start["work"] and report["final_work"] == final["work"], "work snapshots")
    require(final["work"]["ticks"] - start["work"]["ticks"] == ticks, "probe work ticks")
    require(tensor_equal(start["memory"], final["memory"]), "probe altered memory")
    same_tree(start["model"], final["model"], "probe altered parameters")
    require(report["memory_readonly"] and report["parameters_frozen_during_probe"], "reported frozen flags")
    require(report["coverage_min"] == int(start["coverage"].min()), "coverage minimum")
    require(report["memory_norm"] == float(start["memory"].norm(dim=-1).mean()), "memory norm")
    model = model_from_parameters(report["parameters"], start["model"])
    parameter_sizes = {name: value.numel() for name, value in model.named_parameters()}
    require(report["parameters"]["parameters_by_name"] == parameter_sizes, "named parameter counts")
    require(report["parameters"]["total_parameters"] == sum(parameter_sizes.values()), "parameter count")
    for name in ("contexts", "mappings"):
        same_tree(world["random_mapping"][name], final["world"]["random_mapping"][name], "probe mapping configuration." + name)
    # Per-job applied rows legitimately advance with NEW/REPLACE, even though
    # the future mapping table itself remains fixed throughout the probe.
    RandomMappingWorld.from_state_dict(world)
    RandomMappingWorld.from_state_dict(final["world"])

    maturities = {}
    for index, lane, job in np.argwhere(arrays["mature_mask"]):
        key = (int(lane), int(job), int(arrays["mature_rid"][index, lane, job]))
        require(key not in maturities, "duplicate maturity")
        maturities[key] = int(index)
    finished, censored, cue_keys = [], [], set()
    translation_correct = 0
    eligible = 0
    for index, lane, job in np.argwhere(commands):
        family, raw = int(arrays["event_family"][index, lane, job]), int(arrays["raw"][index, lane, job])
        require(family in contexts, "unknown command family")
        fam = contexts.index(family)
        target = int(arrays["target"][index, lane, job])
        require(target == int(table[lane, fam, raw]), "command target differs from private mapping")
        correct = int(arrays["prediction"][index, lane, job]) == target
        translation_correct += correct
        rid = int(arrays["event_rid"][index, lane, job])
        require(rid == int(absolute_ticks[index]) * jobs + int(job), "command request ID")
        if not arrays["event_eligible"][index, lane, job]:
            continue
        eligible += 1
        key = (int(lane), int(job), rid)
        require(key not in cue_keys, "duplicate eligible cue")
        cue_keys.add(key)
        row = dict(lane=int(lane), job=int(job), rid=rid, issued_tick=int(absolute_ticks[index]),
                   family="AB"[fam], raw=raw, target=target,
                   conflict=bool(table[lane, 0, raw] != table[lane, 1, raw]), translation_correct=bool(correct))
        deadline = float(times[index]) + float(public[index, lane, job, 9]) + float(public[index, lane, job, 10])
        expected_index = int(np.searchsorted(times, deadline + 1e-6, side="right"))
        if expected_index < ticks:
            require(key in maturities and maturities[key] == expected_index, "missing or mistimed maturity for " + str(key))
            row.update(mature_tick=int(absolute_ticks[expected_index]),
                       ordinary=bool(arrays["mature_done"][expected_index, lane, job]),
                       strict=bool(arrays["mature_strict"][expected_index, lane, job]))
            finished.append(row)
        else:
            require(key not in maturities, "right-censored request matured")
            censored.append(row)
    finished.sort(key=lambda row: (row["mature_tick"], row["lane"], row["job"]))
    # np.argwhere(commands) already orders surviving cues as the runner's book.
    require(finished == report["requests"], "finished request ledger")
    require(censored == report["right_censored"], "right-censored request ledger")
    left_mature = [key for key in maturities if key not in cue_keys]
    for lane, job, rid in left_mature:
        require(bool(world["tensors"]["_score_pending"][lane, job])
                and rid == int(world["tensors"]["_score_request_id"][lane, job]), "unknown left-boundary maturity")

    groups = {}
    for group in GROUPS:
        selected = [row for row in finished if group == "all" or (
            row["family"] == group[0] and ("_" not in group or
            row["conflict"] == group.endswith("conflict")))]
        groups[group] = dict(n=len(selected), strict=sum(row["strict"] for row in selected),
                             ordinary=sum(row["ordinary"] for row in selected),
                             translation_correct=sum(row["translation_correct"] for row in selected))
    require(groups == report["groups"], "group counts")
    idle = ~arrays["active"]
    idle_opportunities = int(idle.sum())
    idle_actions = int((idle & (arrays["actions"] != 0)).sum())
    require(report["idle_opportunities"] == idle_opportunities and report["idle_actions"] == idle_actions, "idle counts")
    require(report["command_translations"] == int(commands.sum())
            and report["correct_translations"] == translation_correct, "command translations")
    require(eligible == len(finished) + len(censored), "eligible partition")
    return dict(seed=report["seed"], arm=report["arm"], condition=report["condition"], ticks=ticks,
                eligible_cues=eligible, matured=len(finished), right_censored=len(censored),
                inherited_maturities=len(left_mature), groups=groups,
                translations=[translation_correct, int(commands.sum())], idle=[idle_actions, idle_opportunities])


def restore_subject(subject, snapshot):
    subject.model.load_state_dict(snapshot["model"], strict=True)
    subject.model.eval()
    subject.world = RandomMappingWorld.from_state_dict(snapshot["world"])
    subject.binder = PublicReceiptBinder.from_state_dict(snapshot["binder"])
    subject.memory = snapshot["memory"].clone()
    subject.hidden = snapshot["hidden"].clone()
    subject.coverage = snapshot["coverage"].clone()
    subject.work = deepcopy(snapshot["work"])
    same_tree(subject.snapshot(), snapshot, "restored start")


def replay_loaded(subject, report, arrays, states):
    restore_subject(subject, states["start"])
    executor_start = deepcopy(subject.executor.state_dict())
    require(not any(p.requires_grad for p in subject.executor.parameters()), "executor requires gradients")
    subject.world.set_family_policy(subject.contexts)
    for index in range(len(arrays["obs"])):
        obs, prediction, actions, _ = subject.advance(write=False, no_read=report["condition"] == "no_read")
        world = subject.world
        values = dict(obs=obs, actions=actions, prediction=prediction, event_kind=world.event_kind,
                      event_rid=world.event_request_id, event_eligible=world.event_eligible,
                      event_family=world.event_family, raw=world.raw_event_payload, target=world.event_payload,
                      mature_mask=world.matured_mask, mature_rid=world.matured_request_id,
                      mature_done=world.matured_completed, mature_strict=world.matured_clean_completed, active=world.active)
        for name, value in values.items():
            require(array_equal(value.detach().cpu().numpy(), arrays[name][index]),
                    f"native replay {name} at index {index}")
        world.act(actions)
        subject.work["ticks"] += 1
    same_tree(subject.snapshot(), states["final"], "native replay final")
    same_tree(subject.executor.state_dict(), executor_start, "executor mutated")
    return dict(seed=report["seed"], arm=report["arm"], condition=report["condition"],
                ticks=len(arrays["obs"]), compared_arrays=len(ARRAY_NAMES), bitwise=True,
                final_full_snapshot_equal=True, executor_parameters_frozen=True,
                executor_parameter_sha256=parameter_hash(executor_start))


def configure_root(root):
    root = Path(root).resolve()
    if root.name == "shared-v2":
        from experiments.associative_state import shared
        shared.activate()
    training.OUT = root
    evaluation.OUT = root
    return root


def replay_case(task):
    root, seed, arm, condition = task
    torch.set_num_threads(1)
    root = configure_root(root)
    prefix = root / f"seed_{seed:03d}" / arm / "evaluation" / condition
    report, arrays, states = load_bundle(prefix)
    subject = evaluation.Subject(seed, arm)
    return replay_loaded(subject, report, arrays, states)


def key_changes(before, after, parameters):
    model = model_from_parameters(parameters, before)
    indices = torch.arange(model.n_keys)
    with torch.no_grad():
        x = model._key(indices).clone()
        model.load_state_dict(after, strict=True)
        y = model._key(indices).clone()
    delta = y - x
    changed = [name for name, _ in model.named_parameters() if not tensor_equal(before[name], after[name])]
    return dict(raw_embedding_changed=(not tensor_equal(before["key_embedding.weight"], after["key_embedding.weight"]))
                if "key_embedding.weight" in before else None,
                changed_parameter_names=changed,
                normalized_key_l2_by_key=[float(value) for value in delta.norm(dim=-1)],
                normalized_key_l2_max=float(delta.norm(dim=-1).max()),
                normalized_key_l2_mean=float(delta.norm(dim=-1).mean()))


def check_segment(segment, label, archived=False):
    start, final = segment["start"], segment["final"]
    check_snapshot(start, label + ".start")
    check_snapshot(final, label + ".final")
    require(final["world"]["tick"] - start["world"]["tick"] ==
            final["work"]["ticks"] - start["work"]["ticks"], label + ": world/work duration")
    if archived:
        observations = segment["observations"]
        require(observations is not None, label + ": absent observations")
        count, batch, width = observations.shape
        jobs = width // 28
        require(count == final["world"]["tick"] - start["world"]["tick"], label + ": archive duration")
        digest = hashlib.sha256()
        for index, obs in enumerate(observations):
            if index + 1 < count:
                action_features = observations[index + 1].reshape(batch, jobs, 28)[..., 11:22]
                actions = action_features.argmax(-1) * (action_features.sum(-1) > 0)
            else:
                actions = final["world"]["tensors"]["_ack_action"]
            digest.update(obs.numpy().tobytes())
            digest.update(actions.long().numpy().tobytes())
        require(digest.hexdigest() == segment["teacher_stream_sha256"], label + ": independently rebuilt teacher hash")


def audit_support(folder, seed, arm, model_config=None):
    support = load_pt(folder / "support.pt")
    require(set(support) == {"A", "B"}, "support sides")
    for side in ("A", "B"):
        check_segment(support[side], side, archived=True)
        same_tree(support[side]["start"]["model"], support[side]["final"]["model"], side + ": frozen support parameters")
    common = support["B"]["final"]
    segments = {condition: load_pt(folder / f"{condition}_support.pt")
                for condition in ("delay", "adapt", "theta_only", "revise")}
    for condition, segment in segments.items():
        check_segment(segment, condition)
        same_tree(common["model"], segment["start"]["model"], condition + ": common parameters")
        same_tree(common["memory"], segment["start"]["memory"], condition + ": common memory")
        same_tree(common["hidden"], segment["start"]["hidden"], condition + ": common executor h")
        same_tree(common["binder"], segment["start"]["binder"], condition + ": common binder")
    hashes = {condition: segments[condition]["teacher_stream_sha256"] for condition in ("delay", "adapt", "theta_only")}
    require(len(set(hashes.values())) == 1, "delay/adapt/theta_only teacher streams differ")
    for condition in ("adapt", "theta_only"):
        same_tree(segments["delay"]["start"]["world"], segments[condition]["start"]["world"], condition + ": common teacher start world")
        same_tree(segments["delay"]["final"]["world"], segments[condition]["final"]["world"], condition + ": identical teacher final world")
        same_tree(segments["delay"]["final"]["binder"], segments[condition]["final"]["binder"], condition + ": identical teacher final binder")
        same_tree(segments["delay"]["final"]["coverage"], segments[condition]["final"]["coverage"], condition + ": identical successful examples")
    theta = segments["theta_only"]
    require(tensor_equal(theta["start"]["memory"], theta["final"]["memory"]), "theta_only changed M")
    same_tree(segments["delay"]["start"]["model"], segments["delay"]["final"]["model"], "delay changed parameters")
    # Detached M makes only the shared reader/key parameters receive gradients.
    writer_unchanged = {}
    for condition in ("adapt", "theta_only"):
        segment = segments[condition]
        names = [name for name in segment["start"]["model"] if name == "eta_logit" or name.startswith("core.")]
        for name in names:
            same_tree(segment["start"]["model"][name], segment["final"]["model"][name], condition + ": unexpected writer update " + name)
        writer_unchanged[condition] = names
    if model_config is None:
        model_config = load_pt(folder.parent / "final.pt")["metadata"]["model_config"]
    return dict(seed=seed, arm=arm, teacher_stream_sha256=hashes["delay"],
                common_teacher_streams_equal=True, archived_support_hashes_rebuilt=True,
                theta_only_memory_bitwise_unchanged=True, unchanged_online_writer_parameters=writer_unchanged,
                reader_changes={condition: key_changes(segments[condition]["start"]["model"],
                                                       segments[condition]["final"]["model"],
                                                       model_config)
                                for condition in ("adapt", "theta_only")})


def run(root_override=None, replay_workers=9):
    timer = time.perf_counter()
    torch.set_num_threads(1)
    root = configure_root(training.OUT if root_override is None else root_override)
    protocol = training.check_protocol()
    require(protocol["seeds"] == [0, 1, 2] and tuple(protocol["arms"]) == tuple(training.ARMS), "declared models")
    require(replay_workers >= 1, "replay workers must be positive")
    metric_results, support_results, artifact_hashes = [], [], {}
    sample_hashes = {}
    expected_names = {"completion.json", "support.pt"}
    expected_names |= {f"{condition}.{suffix}" for condition in evaluation.CONDITIONS for suffix in ("npz", "json", "pt")}
    expected_names |= {f"{condition}_support.pt" for condition in ("delay", "adapt", "theta_only", "revise")}
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            model_folder = root / f"seed_{seed:03d}" / arm
            folder = model_folder / "evaluation"
            require(folder.is_dir(), "evaluation artifacts not ready: " + str(folder))
            actual_names = {path.name for path in folder.iterdir() if path.suffix in (".json", ".npz", ".pt")}
            require(actual_names == expected_names, str(folder) + ": missing or unexpected artifacts " + str(actual_names ^ expected_names))
            completion = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
            require(completion["seed"] == seed and completion["arm"] == arm, "completion identity")
            require(tuple(completion["conditions"]) == evaluation.CONDITIONS, "completion conditions")
            require(completion["support_sha256"] == file_hash(folder / "support.pt"), "support file hash")
            require(completion["pretraining_sha256"] == file_hash(model_folder / "final.pt"), "pretraining file hash")
            checkpoint = load_pt(model_folder / "final.pt")
            metadata = checkpoint["metadata"]
            require(metadata["updates"] == training.SETTINGS["updates"], "pretraining update budget")
            sample_hashes.setdefault(seed, set()).add(metadata["train_sample_sha256"])
            for condition in evaluation.CONDITIONS:
                prefix = folder / condition
                require(completion["reports"][condition] == file_hash(prefix.with_suffix(".json")), "completion report hash")
                report, arrays, states = load_bundle(prefix)
                require((report["seed"], report["arm"], report["condition"]) == (seed, arm, condition), "report identity")
                require(len(arrays["obs"]) == training.SETTINGS["eval_ticks"], "formal evaluation duration")
                metric_results.append(recount(report, arrays, states))
            support_results.append(audit_support(folder, seed, arm))
            for name in sorted(expected_names):
                path = folder / name
                artifact_hashes[str(path.relative_to(root)).replace("\\", "/")] = file_hash(path)
    require(all(len(values) == 1 for values in sample_hashes.values()), "arms used different pretraining samples")
    for origin in protocol["origins"].values():
        require(file_hash(ROOT / origin["path"]) == origin["sha256"], "frozen executor origin hash")
    tasks = [(str(root), *case) for case in REPLAY_CASES]
    if replay_workers == 1:
        replays = [replay_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(replay_workers, len(tasks))) as pool:
            replays = list(pool.map(replay_case, tasks))
    result = dict(format="associative-state-independent-audit-v1", status="passed",
                  created_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - timer,
                  audit_source_sha256=file_hash(__file__), protocol_sha256=file_hash(root / "protocol.json"),
                  artifact_count=len(artifact_hashes), artifacts_sha256=artifact_hashes,
                  metrics=metric_results, support=support_results, native_replays=replays,
                  all_reports_recounted=True, pretraining_samples_equal_across_arms=True,
                  pinned_executor_origins_verified=True,
                  limitation="Executor freezing is checked against pinned origins, frozen source, and complete native replay; evaluation snapshots do not archive a second executor weight copy.")
    training.write_json(root / "audit.json", result)
    print(json.dumps(dict(status="passed", metric_cases=len(metric_results), native_replays=len(replays),
                          artifact_count=len(artifact_hashes), output=str(root / "audit.json"))), flush=True)
    return result


def self_test():
    """Exercise counters and replay without loading or writing formal artifacts."""
    torch.set_num_threads(1)
    torch.manual_seed(9821)
    saved_settings = deepcopy(evaluation.SETTINGS)
    try:
        evaluation.SETTINGS["actors"] = 2
        evaluation.SETTINGS["eval_ticks"] = 96
        subject = evaluation.Subject.__new__(evaluation.Subject)
        subject.seed, subject.arm, subject.contexts = 0, "delta", (0, 3)
        subject.model = evaluation.AssociationMemory(kind="delta").eval()
        subject.executor = GRUAgent(width=8).eval().requires_grad_(False)
        subject.original = torch.tensor(((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 2, 3, 0)))
        subject.table = torch.tensor([[[2, 3, 0, 1], [3, 2, 0, 1]], [[1, 0, 3, 2], [0, 1, 3, 2]]])
        subject.revised = subject.table.clone()
        subject.revised[:, 0, 2:] = subject.revised[:, 0, 2:].flip(-1)
        subject.world = RandomMappingWorld(seed=712, batch_size=2, n_jobs=2,
            contexts=subject.contexts, mappings=subject.table, family_allowed=(0,),
            duration_range=(.5, 1.), gap_range=(.5, 1.), window_range=(.5, 1.))
        subject.binder = PublicReceiptBinder(2, contexts=subject.contexts)
        subject.memory = subject.model.initial_state(2)
        subject.hidden = subject.executor.initial_state(2)
        subject.coverage = torch.zeros(2, 8, dtype=torch.long)
        subject.work = dict(ticks=0, public_examples=0, parameter_updates=0, gradient_examples=0)
        with tempfile.TemporaryDirectory(prefix="agi-assoc-audit-fixture-") as directory:
            folder = Path(directory)
            with redirect_stdout(io.StringIO()):
                first = subject.support(48, 0, archive=True)
                second = subject.support(48, 1, archive=True)
                common = deepcopy(subject)
                torch.save(dict(A=first, B=second), folder / "support.pt")
                for condition in ("delay", "adapt", "theta_only", "revise"):
                    branch = deepcopy(common)
                    if condition == "revise":
                        branch.world.set_mappings(branch.revised)
                    segment = branch.support(32, 0 if condition == "revise" else 1,
                        write=condition != "theta_only", learn=condition in ("adapt", "theta_only"))
                    torch.save(segment, folder / f"{condition}_support.pt")
                evaluation.probe(subject, folder, "immediate")
            report, arrays, states = load_bundle(folder / "immediate")
            counts = recount(report, arrays, states)
            replay = replay_loaded(subject, report, arrays, states)
            support = audit_support(folder, 0, "delta", model_config=subject.model.config())
            require(counts["matured"] > 0 and counts["right_censored"] > 0, "fixture must exercise maturity and right censoring")
            failures = []
            bad_report = deepcopy(report)
            bad_report["groups"]["all"]["n"] += 1
            try:
                recount(bad_report, arrays, states)
            except AuditError:
                failures.append("wrong_denominator")
            corrupted = {key: value.copy() for key, value in arrays.items()}
            corrupted["obs"][0, 0, 9] += .125
            try:
                replay_loaded(subject, report, corrupted, states)
            except AuditError:
                failures.append("corrupted_public_observation")
            missing = {key: value.copy() for key, value in arrays.items()}
            request = report["requests"][0]
            index = request["mature_tick"] - report["start_tick"] - 1
            missing["mature_mask"][index, request["lane"], request["job"]] = False
            try:
                recount(report, missing, states)
            except AuditError:
                failures.append("missing_expected_maturity")
            require(len(failures) == 3, "fixture did not reject all injected faults")
            result = dict(status="self_test_passed", temporary_fixture=True,
                          matured=counts["matured"], right_censored=counts["right_censored"],
                          replay_ticks=replay["ticks"], support_hashes_equal=support["common_teacher_streams_equal"],
                          rejected_faults=failures)
            print(json.dumps(result), flush=True)
            return result
    finally:
        evaluation.SETTINGS.clear()
        evaluation.SETTINGS.update(saved_settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=training.OUT)
    parser.add_argument("--replay-workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        run(arguments.root, arguments.replay_workers)
