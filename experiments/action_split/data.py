"""Native earliest-teacher streams for the predeclared split-action experiment.

Only ``obs`` and constant DT are learner inputs.  All other arrays are targets
or private audit facts.  Eligibility never selects supervised examples: an
active request is supervised even when a later external event will replace or
cancel it.  Every saved row is after observe() and before its actual action.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.experience_transfer.train_transfer import SEEDS, mapping_scope, mapping_table
from experiments.lifelong.world import ContextWorld, FEATURE_LAYOUT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/action-split/streams-v1"
PLAN = OUT / "protocol.json"
WARM, TRAIN_STEPS, BLOCK, BATCH, DT = 512, 32768, 128, 32, .25
TRAIN = TRAIN_STEPS
TOTAL = WARM + TRAIN_STEPS
EPS = 1e-6
ACTORS = tuple(range(95000, 95032))
TRAIN_KEYS = ("obs", "actions", "mode_target", "begin_target", "finish_target", "active")

# Names deliberately match prior raw-fact archives for cross-checks.  This is
# factual current state, not a feature extension for the learner.
FACT_FIELDS = {
    "rid": "request_id", "active": "active", "started": "started",
    "ready": "ready_time", "deadline": "deadline", "issued_time": "issued_time",
    "version": "version", "payload": "payload", "raw": "raw_payload",
    "family": "current_family", "eligible": "current_eligible",
    "duration": "duration", "had_error": "had_error",
    "interrupted": "current_interrupted", "regime_id": "current_regime_id",
    "event_kind": "event_kind", "event_id": "event_request_id",
    "event_eligible": "event_eligible", "event_family": "event_family",
    "event_raw": "raw_event_payload", "event_payload": "event_payload",
    "event_version": "event_version", "event_duration": "event_duration",
    "event_window": "event_window", "event_regime_id": "event_regime_id",
    "mature": "matured_mask", "mature_id": "matured_request_id",
    "mature_issued": "matured_issued_at", "mature_duration": "matured_duration",
    "mature_family": "matured_family", "mature_done": "matured_completed",
    "mature_strict": "matured_clean_completed",
    "last_external_tick": "last_external_tick", "last_external_time": "last_external_time",
    "score_pending": "_score_pending", "score_deadline": "_score_deadline",
    "score_id": "_score_request_id", "score_completed": "_score_completed",
    "score_clean": "_score_clean", "score_interrupted": "_score_interrupted",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def stream_path(seed):
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    return OUT / f"seed_{int(seed):03d}" / "stream.npz"


def config(seed):
    spec = SEEDS[int(seed)]
    return dict(seed=850000 + int(seed), batch_size=BATCH, n_jobs=2, active_jobs=2,
                actor_ids=ACTORS, family_allowed=(spec["history_cue"], spec["B_cue"]),
                dt=DT, duration_range=(8., 16.), gap_range=(16., 32.),
                window_range=(1.5, 2.5), cancel=.20, replace=.30,
                regime_name="action-split/training/long_both", device="cpu")


def load_stream(seed, keys=None):
    """Load all facts, or only named learner arrays to reduce memory use."""
    with np.load(stream_path(seed), allow_pickle=False) as archive:
        names = archive.files if keys is None else keys
        return {name: archive[name] for name in names}


def source_files():
    return [Path(__file__), Path(__file__).with_name("data_test.py"),
            ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py",
            ROOT / "experiments/experience_transfer/train_transfer.py"]


def declare():
    """Freeze the collection protocol before opening the first training world."""
    plan = dict(format="action-split-native-earliest-streams-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
        warm=WARM, train_steps=TRAIN_STEPS, total_ticks=TOTAL, block=BLOCK,
        updates=TRAIN_STEPS // BLOCK, batch_size=BATCH, dt=DT,
        world_configs={str(seed): config(seed) for seed in SEEDS},
        mappings={str(seed): mapping_table(seed, "related") for seed in SEEDS},
        source_files={str(path.relative_to(ROOT)): sha(path) for path in source_files()},
        teacher="Every tick: native world.observe(); labels=world.oracle('earliest'); world.act(labels). Native actions and receipts are actual. No original-head model is instantiated.",
        continuity="One world per raw model seed, 33280 consecutive observations, no resets. First512 warm ticks then256 sequential128-tick training blocks. Learner numeric h is continuous and detached at each update; no repeated data passes.",
        inputs="Exactly original public56 observation plus constant dt=.25. World clock, phase, private state, target labels, eligibility, maturity, and counts are not model inputs.",
        mode_target="WAIT=0 for native0; BEGIN=1 for native1..2; FINISH=2 for native3..10. Every tick and both slots are supervised.",
        begin_target="Current version0..1 at active pre-action slots, else-1; supervised only where mode_target==BEGIN.",
        finish_target="Current4*version+mapped_payload0..7 at every active pre-action slot, else-1. Supervise all active slots, including later canceled/replaced requests and processing WAIT rows.",
        no_future_filter="Private current/event eligibility and eventual maturity are audit/scorer facts only. They never select or weight a training label.",
        data_dtypes="obs float32[T,32,56], actions/mode/begin/finish int64[T,32,2], active bool[T,32,2]. Raw audit facts retain native NumPy dtypes.",
        receipts="Post-action receipts are archived separately. Their next-observation public channels must exactly reflect those same actual actions/results.",
        public_feature_layout=FEATURE_LAYOUT,
        raw_fact_fields=FACT_FIELDS)
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", path=str(PLAN), sha256=sha(PLAN))), flush=True)
    return plan


def checked():
    plan = read(PLAN)
    for name, digest in plan["source_files"].items():
        if sha(ROOT / name) != digest:
            raise AssertionError(f"Frozen data source changed: {name}")
    assert (plan["warm"], plan["train_steps"], plan["total_ticks"], plan["block"]) == (
        WARM, TRAIN_STEPS, TOTAL, BLOCK)
    return plan


def targets(actions, active, version, payload):
    """Current-state targets; no eligibility or future-event argument exists."""
    actions = np.asarray(actions)
    assert np.all((actions >= 0) & (actions <= 10))
    mode = np.where(actions == 0, 0, np.where(actions <= 2, 1, 2)).astype(np.int64)
    begin = np.where(active, version, -1).astype(np.int64)
    finish = np.where(active, 4 * version + payload, -1).astype(np.int64)
    return mode, begin, finish


def array_hash(facts, names):
    digest = hashlib.sha256()
    for name in names:
        value = np.asarray(facts[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(np.ascontiguousarray(value)))
    return digest.hexdigest()


def counts(facts, start=0, stop=None):
    stop = len(facts["actions"]) if stop is None else stop
    mode = facts["mode_target"][start:stop]
    active = facts["active"][start:stop]
    begin = mode == 1
    eligible = facts["eligible"][start:stop]
    versions = facts["begin_target"][start:stop][begin]
    command = np.isin(facts["event_kind"][start:stop], (1, 2))
    return dict(ticks=int(stop-start), opportunities=int(mode.size),
        mode_WAIT=int((mode == 0).sum()), mode_BEGIN=int(begin.sum()),
        mode_FINISH=int((mode == 2).sum()), active_content=int(active.sum()),
        active_eligible_content=int((active & eligible).sum()),
        active_ineligible_content=int((active & ~eligible).sum()),
        begin_version_0=int((versions == 0).sum()), begin_version_1=int((versions == 1).sum()),
        begin_ineligible=int((begin & ~eligible).sum()),
        commands=int(command.sum()),
        ineligible_commands=int((command & ~facts["event_eligible"][start:stop]).sum()),
        mature=int(facts["mature"][start:stop].sum()),
        mature_done=int(facts["mature_done"][start:stop].sum()),
        mature_strict=int(facts["mature_strict"][start:stop].sum()))


def verify_facts(facts):
    """Rebuild causal oracle, public fields, receipts, targets and maturations."""
    action = facts["actions"]
    active, started = facts["active"], facts["started"]
    now = facts["now"][:, None, None]
    begin = active & ~started & (now + facts["duration"] <= facts["deadline"] + EPS)
    finish = active & started & (now >= facts["ready"] - EPS) & (now <= facts["deadline"] + EPS)
    expected = np.where(begin, 1 + facts["version"], np.where(
        finish, 3 + 4 * facts["version"] + facts["payload"], 0))
    np.testing.assert_array_equal(expected, action)
    for name, value in zip(("mode_target", "begin_target", "finish_target"),
                          targets(action, active, facts["version"], facts["payload"])):
        np.testing.assert_array_equal(facts[name], value)
    receipts = (action != 0).astype(np.int64)
    np.testing.assert_array_equal(facts["receipts"], receipts)
    np.testing.assert_array_equal(facts["tick"], np.arange(len(action)))
    np.testing.assert_allclose(facts["now"], np.arange(len(action)) * DT, atol=0, rtol=0)
    obs = facts["obs"].reshape(len(action), action.shape[1], 2, 28)
    np.testing.assert_array_equal(obs[1:, ..., 11:22],
        np.eye(11, dtype=np.float32)[action[:-1]] * (action[:-1] != 0)[..., None])
    np.testing.assert_array_equal(obs[1:, ..., 22], receipts[:-1] == 1)
    np.testing.assert_array_equal(obs[1:, ..., 23], receipts[:-1] == -1)
    assert not obs[0, ..., 11:24].any()
    kind = facts["event_kind"]
    command = np.isin(kind, (1, 2))
    event = kind != 0
    np.testing.assert_array_equal(obs[..., :3], np.eye(4, dtype=np.float32)[kind][..., 1:])
    np.testing.assert_array_equal(obs[..., 3:5],
        np.eye(2, dtype=np.float32)[facts["event_version"]] * event[..., None])
    np.testing.assert_array_equal(obs[..., 5:9],
        np.eye(4, dtype=np.float32)[facts["event_raw"]] * command[..., None])
    np.testing.assert_array_equal(obs[..., 9], facts["event_duration"] * command)
    np.testing.assert_array_equal(obs[..., 10], facts["event_window"] * command)
    np.testing.assert_array_equal(obs[..., 24:28],
        np.eye(4, dtype=np.float32)[facts["event_family"].clip(min=0)] * command[..., None])
    assert np.all(~facts["mature"] | (now > facts["score_deadline"] + EPS))
    assert np.all(~facts["mature_done"] | facts["mature"])
    assert np.all(~facts["mature_strict"] | facts["mature_done"])
    assert not facts["had_error"].any()
    # In this teacher stream every accepted BEGIN/FINISH is causal, including
    # commands marked ineligible by the private future scheduler.
    return dict(native_earliest_actions="exact", current_state_targets="exact",
                public_observation_and_receipts="exact", native_receipts="all accepted",
                clock_continuity="exact", no_rejected_active_action=True,
                checked_ticks=int(len(action)), checked_opportunities=int(action.size))


@torch.inference_mode()
def collect_native(seed, total=TOTAL, world_config=None, progress=False):
    """Collect one uninterrupted native stream, also usable by short tests."""
    torch.set_num_threads(1)
    kwargs = config(seed) if world_config is None else world_config
    with mapping_scope(mapping_table(seed, "related")):
        world = ContextWorld(**kwargs)
        shape = (total, world.batch_size, 2)
        facts = {name: np.empty((total, *getattr(world, attr).shape),
                               dtype=getattr(world, attr).numpy().dtype)
                 for name, attr in FACT_FIELDS.items()}
        facts.update(obs=np.empty((total, world.batch_size, 56), dtype=np.float32),
            actions=np.empty(shape, dtype=np.int64), receipts=np.empty(shape, dtype=np.int64),
            mode_target=np.empty(shape, dtype=np.int64), begin_target=np.empty(shape, dtype=np.int64),
            finish_target=np.empty(shape, dtype=np.int64), now=np.empty(total, dtype=np.float64),
            tick=np.empty(total, dtype=np.int64))
        initial = world.state_dict()
        for tick in range(total):
            obs = world.observe()
            action = world.oracle("earliest")
            for name, attr in FACT_FIELDS.items():
                facts[name][tick] = getattr(world, attr).numpy()
            facts["obs"][tick] = obs.numpy()
            facts["actions"][tick] = action.numpy()
            facts["now"][tick], facts["tick"][tick] = world.now, world.tick
            labels = targets(facts["actions"][tick], facts["active"][tick],
                             facts["version"][tick], facts["payload"][tick])
            for name, label in zip(("mode_target", "begin_target", "finish_target"), labels):
                facts[name][tick] = label
            world.act(action)
            facts["receipts"][tick] = world.last_result.numpy()
            if progress and (tick+1) % 4096 == 0:
                print(json.dumps(dict(event="progress", seed=seed, ticks=tick+1)), flush=True)
        facts["actor_ids"] = np.asarray(world.actor_ids, dtype=np.int64)
        context = dict(initial_world=initial, final_world=world.state_dict(),
                       metrics=world.metrics(), events_sha256=world.event_hash,
                       case_ids=list(world.case_ids))
    return facts, context


def collect(seed):
    checked()
    path = stream_path(seed)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    facts, context = collect_native(seed, progress=True)
    audit = verify_facts(facts)
    block_counts = [dict(update=i+1, start_tick=WARM+i*BLOCK, stop_tick=WARM+(i+1)*BLOCK,
                         **counts(facts, WARM+i*BLOCK, WARM+(i+1)*BLOCK))
                    for i in range(TRAIN_STEPS // BLOCK)]
    path.parent.mkdir(parents=True, exist_ok=True)
    states_path = path.with_name("world_states.pt")
    with states_path.open("xb") as handle:
        torch.save(dict(initial=context["initial_world"], final=context["final_world"]), handle)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **facts)
    metadata = dict(format="action-split-native-earliest-stream-v1", seed=int(seed),
        world_config=config(seed), mapping_table=mapping_table(seed, "related"),
        warm=WARM, train_steps=TRAIN_STEPS, total_ticks=TOTAL, block=BLOCK,
        arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in facts.items()},
        counts_all=counts(facts), counts_warm=counts(facts, 0, WARM),
        counts_training=counts(facts, WARM, TOTAL), block_counts=block_counts,
        public_observation_sha256=array_hash(facts, ("obs",)),
        public_stream_sha256=array_hash(facts, ("obs", "actions", "receipts")),
        actions_sha256=array_hash(facts, ("actions",)),
        events_sha256=context["events_sha256"], native_metrics=context["metrics"],
        case_ids=context["case_ids"], audit=audit,
        stream_sha256=sha(path), world_states_sha256=sha(states_path), protocol_sha256=sha(PLAN),
        source_files=checked()["source_files"])
    for name, value in metadata["counts_training"].items():
        assert sum(row[name] for row in block_counts) == value
    write_new(path.with_suffix(".json"), metadata)
    print(json.dumps(dict(event="collected", seed=seed, counts=metadata["counts_training"],
                          stream_sha256=metadata["stream_sha256"])), flush=True)
    return metadata


def launch():
    """Three independent single-thread CPU workers with hidden Windows windows."""
    checked()
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers = []
    for seed in SEEDS:
        logfile = logs / f"seed_{seed:03d}.log"
        handle = logfile.open("x", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen([sys.executable, "-m", "experiments.action_split.data", "collect",
                                    "--seed", str(seed)], cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, creationflags=flags)
        workers.append((seed, process, handle, logfile))
    finished = []
    while workers:
        remaining = []
        for seed, process, handle, logfile in workers:
            code = process.poll()
            if code is None:
                remaining.append((seed, process, handle, logfile))
            else:
                handle.close()
                finished.append(dict(seed=seed, pid=process.pid, exit_code=code, log=str(logfile)))
                print(json.dumps(finished[-1]), flush=True)
        workers = remaining
        if workers:
            time.sleep(.25)
    write_new(OUT / "workers.json", dict(workers=finished))
    if any(row["exit_code"] for row in finished):
        raise RuntimeError("Data worker failed; see logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "collect", "launch"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command == "collect":
        collect(args.seed)
    else:
        launch()
