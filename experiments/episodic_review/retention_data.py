"""Prospective fixed B-only teacher suffix, after the primary replay endpoint.

The entire immutable withdraw final world is restored unchanged. One actual
8192-tick midpoint history per seed is shared by all model arms. This module
never loads a model or an evaluation result.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

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

from experiments.continuity.run import state_digest
from experiments.coverage_retention import data as prior
from experiments.experience_transfer.train_transfer import SEEDS as SEED_SPECS
from experiments.lifelong.world import ContextWorld, FEATURE_LAYOUT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/episodic-review/retention-streams-v1"
PLAN = OUT / "protocol.json"
FORMAT = "episodic-review-prospective-B-only-retention-stream-v1"
SEEDS = (0, 1, 2)
WARM, TOTAL, TRAIN_STEPS, BLOCK, BATCH, UPDATES = 0, 8192, 8192, 128, 32, 64
SOURCE_CURSOR, DT, TIME_SCALE = 66048, .25, 16.
STREAM_FIELDS = ("obs", "actions", "time_target", "time_mask")
FACT_FIELDS = prior.FACT_FIELDS
sha, write_new = prior.sha, prior.write_new


def seed_dir(seed):
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    return OUT / f"seed_{int(seed):03d}"


def raw_path(seed):
    return seed_dir(seed) / "stream.npz"


def states_path(seed):
    return seed_dir(seed) / "world_states.pt"


def metadata_path(seed):
    return seed_dir(seed) / "stream.json"


def audit_path(seed):
    return seed_dir(seed) / "raw_audit.json"


def source_world_path(seed):
    return prior.states_path(seed, "withdraw")


def load_stream(seed):
    with np.load(raw_path(seed), allow_pickle=False) as archive:
        return {key: archive[key] for key in STREAM_FIELDS}


def load_facts(seed):
    with np.load(raw_path(seed), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def source_inputs(seed):
    return [source_world_path(seed), prior.raw_path(seed, "withdraw"), prior.metadata_path(seed, "withdraw"),
        prior.audit_path(seed, "withdraw"), prior.PLAN, prior.OUT / "completion.json",
        prior.OUT / "audit_completion.json"]


def input_paths(seed):
    return [PLAN, raw_path(seed), states_path(seed), metadata_path(seed), audit_path(seed), *source_inputs(seed)]


def sources():
    return list(dict.fromkeys([Path(__file__).resolve(), Path(__file__).with_name("retention_data_test.py"),
        Path(prior.__file__).resolve(), ROOT / "experiments/teacher_timing/data.py",
        ROOT / "experiments/action_split/data.py", ROOT / "experiments/experience_transfer/train_transfer.py",
        ROOT / "experiments/continuity/run.py", ROOT / "experiments/continuity/world.py",
        ROOT / "experiments/lifelong/world.py"]))


def family_exposure(facts, seed, start=0, stop=None):
    stop = len(facts["actions"]) if stop is None else int(stop)
    take = slice(start, stop)
    out = prior.source.counts(facts, start, stop)
    for label, cue in (("A", SEED_SPECS[int(seed)]["history_cue"]), ("B", SEED_SPECS[int(seed)]["B_cue"])):
        family = facts["family"][take] == cue
        event_family = facts["event_family"][take] == cue
        inherited = facts["rid"][take] < SOURCE_CURSOR*2
        out[label+"_active_labels"] = int((family & facts["active"][take]).sum())
        out[label+"_time_labels"] = int((family & facts["time_mask"][take]).sum())
        out[label+"_nonWAIT_labels"] = int((family & (facts["actions"][take] != 0)).sum())
        out[label+"_inherited_active_labels"] = int((family & inherited & facts["active"][take]).sum())
        out[label+"_inherited_time_labels"] = int((family & inherited & facts["time_mask"][take]).sum())
        for kind, code in (("NEW", 1), ("REPLACE", 2), ("CANCEL", 3)):
            out[label+"_"+kind] = int((event_family & (facts["event_kind"][take] == code)).sum())
        out[label+"_mature"] = int(((facts["mature_family"][take] == cue) & facts["mature"][take]).sum())
    return out


@torch.inference_mode()
def collect_native(seed, *, total=TOTAL, initial_state=None, progress=False):
    torch.set_num_threads(1)
    if initial_state is None:
        initial_state = torch.load(source_world_path(seed), map_location="cpu", weights_only=False)["final"]
    with prior.mapping_scope(prior.mapping_table(seed, "related")):
        world = ContextWorld.from_state_dict(initial_state, device="cpu")
        initial = world.state_dict()
        assert state_digest(initial) == state_digest(initial_state)
        # No boundary setter, event, reset, warmup, or family change occurs.
        facts = {name: np.empty((total, *getattr(world, attr).shape), dtype=getattr(world, attr).numpy().dtype)
                 for name, attr in FACT_FIELDS.items()}
        shape = total, world.batch_size, world.n_jobs
        facts.update(obs=np.empty((total, world.batch_size, 56), np.float32),
            actions=np.empty(shape, np.int64), receipts=np.empty(shape, np.int64),
            now=np.empty(total, np.float64), tick=np.arange(total, dtype=np.int64),
            world_tick=np.empty(total, np.int64))
        for local in range(total):
            observation = world.observe()
            action = world.oracle("midpoint")
            for key, attr in FACT_FIELDS.items():
                facts[key][local] = getattr(world, attr).numpy()
            facts["obs"][local], facts["actions"][local] = observation.numpy(), action.numpy()
            facts["now"][local], facts["world_tick"][local] = world.now, world.tick
            world.act(action)
            facts["receipts"][local] = world.last_result.numpy()
            if progress and (local+1) % 2048 == 0:
                print(json.dumps(dict(event="retention_collect_progress", seed=seed, ticks=local+1)), flush=True)
        facts["actor_ids"] = np.asarray(world.actor_ids, np.int64)
        facts["time_target"], facts["time_mask"] = prior.signed_targets(
            facts["ready"], facts["now"], facts["active"], facts["started"])
        context = dict(initial=initial, final=world.state_dict(), native_metrics=world.metrics(),
            events_sha256=world.event_hash, case_ids=list(world.case_ids))
    return facts, context


def verify(facts, seed, initial):
    prior.verify_facts(facts, seed, "withdraw", initial)
    b_cue, a_cue = SEED_SPECS[int(seed)]["B_cue"], SEED_SPECS[int(seed)]["history_cue"]
    assert initial["context"]["family_allowed"] == [b_cue]
    assert bool((initial["context"]["tensors"]["current_family"] == b_cue).all())
    assert not np.any((facts["event_kind"] != 0) & (facts["event_family"] != b_cue))
    assert not np.any(facts["family"] != b_cue)
    exposure = family_exposure(facts, seed)
    assert not any(value for key, value in exposure.items() if key.startswith("A_"))
    assert not np.any(facts["family"] == a_cue)
    return exposure


def source_prefix(seed, count=512):
    """Past-only accepted-BEGIN evidence for all retained finite ready clocks."""
    with np.load(prior.raw_path(seed, "withdraw"), allow_pickle=False) as archive:
        return {key: archive[key][-count:] for key in ("rid", "actions", "receipts", "now", "duration")}


def accepted_begin_ledger(facts, initial, prefix):
    """Independent labels from commands and accepted BEGIN, in global time.

    Existing finite clocks are proved against accepted BEGIN in the immutable
    source suffix. No ready value in the new archive constructs a label.
    """
    tensors = initial["tensors"]
    names = dict(active="active", started="started", version="version", payload="payload",
        duration="duration", deadline="deadline", rid="request_id")
    current = {key: tensors[attr].numpy().copy() for key, attr in names.items()}
    ready = np.full(current["rid"].shape, np.inf, np.float64)
    begin_time = np.full(ready.shape, np.nan, np.float64)
    finite = np.isfinite(tensors["ready_time"].numpy())
    for lane, slot in zip(*np.nonzero(finite)):
        matching = ((prefix["rid"][:, lane, slot] == current["rid"][lane, slot]) &
            ((prefix["actions"][:, lane, slot] == 1) | (prefix["actions"][:, lane, slot] == 2)) &
            (prefix["receipts"][:, lane, slot] == 1))
        positions = np.flatnonzero(matching)
        assert len(positions) == 1, ("source accepted BEGIN not unique", lane, slot)
        at = int(positions[0])
        assert prefix["duration"][at, lane, slot] == current["duration"][lane, slot]
        begin_time[lane, slot] = prefix["now"][at]
        ready[lane, slot] = begin_time[lane, slot]+current["duration"][lane, slot]
    np.testing.assert_array_equal(ready, tensors["ready_time"].numpy())
    active, started, version, payload = (current[key] for key in ("active", "started", "version", "payload"))
    duration, deadline, rid = (current[key] for key in ("duration", "deadline", "rid"))
    valid = begins = finishes = commands = 0
    for local in range(len(facts["actions"])):
        global_tick = initial["tick"]+1+local
        now = global_tick*.25
        assert facts["tick"][local] == local and facts["world_tick"][local] == global_tick
        assert facts["now"][local] == now
        active &= now <= deadline+1e-6
        kind = facts["event_kind"][local]
        command = (kind == 1) | (kind == 2)
        cancel = (kind == 3) & (facts["event_version"][local] == version)
        active = np.where(command, True, active & ~cancel)
        started[command | cancel] = False
        ready[command | cancel], begin_time[command | cancel] = np.inf, np.nan
        for value, key in ((version, "event_version"), (payload, "event_payload"),
                           (rid, "event_id"), (duration, "event_duration")):
            value[command] = facts[key][local][command]
        deadline[command] = now+facts["event_duration"][local][command].astype(np.float64)+facts["event_window"][local][command].astype(np.float64)
        for key, value in (("active", active), ("started", started), ("version", version), ("payload", payload),
                           ("rid", rid), ("duration", duration), ("deadline", deadline), ("ready", ready)):
            np.testing.assert_array_equal(value, facts[key][local], err_msg=f"independent ledger {global_tick}/{key}")
        mask = active & started & np.isfinite(ready)
        expected = np.zeros(mask.shape, np.float32)
        expected[mask] = ((begin_time[mask]+duration[mask]-now)/16.).astype(np.float32)
        np.testing.assert_array_equal(mask, facts["time_mask"][local])
        np.testing.assert_array_equal(expected, facts["time_target"][local])
        action = facts["actions"][local]
        begin = ((action == 1) | (action == 2)) & active & ~started & (action-1 == version)
        finish = ((action >= 3) & (action <= 10) & active & started & ((action-3)//4 == version)
            & ((action-3)%4 == payload) & (now >= ready-1e-6) & (now <= deadline+1e-6))
        receipts = (begin | finish).astype(np.int64)
        receipts[(action != 0) & (receipts == 0)] = -1
        np.testing.assert_array_equal(receipts, facts["receipts"][local])
        begin_time[begin] = now
        ready[begin] = begin_time[begin]+duration[begin]
        started = (started | begin) & ~finish
        active &= ~finish
        valid += int(mask.sum()); begins += int(begin.sum()); finishes += int(finish.sum()); commands += int(command.sum())
    return dict(passed=True, ticks=len(facts["actions"]), time_rows=valid,
        accepted_BEGIN=begins, accepted_FINISH=finishes, commands=commands,
        inherited_finite_ready_clocks_verified=int(finite.sum()), prefix_ticks=len(prefix["now"]),
        labels="float32 bitwise exact accepted-BEGIN+public duration minus current global time, /16",
        ready_from_new_archive_used_to_construct_labels=False, eligibility_used=False)


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    prior.checked()
    inputs, initial = {}, {}
    for seed in SEEDS:
        state = torch.load(source_world_path(seed), map_location="cpu", weights_only=False)["final"]
        b_cue = SEED_SPECS[seed]["B_cue"]
        assert state["tick"]+1 == SOURCE_CURSOR and state["acted"]
        assert state["spec"]["actor_ids"] == tuple(range(95000, 95032))
        assert state["context"]["family_allowed"] == [b_cue]
        assert bool((state["context"]["tensors"]["current_family"] == b_cue).all())
        metadata = json.loads(prior.metadata_path(seed, "withdraw").read_text(encoding="utf-8"))
        audit = json.loads(prior.audit_path(seed, "withdraw").read_text(encoding="utf-8"))
        assert audit["complete"] and audit["native_replay_ticks"] == 32768
        assert audit["stream_sha256"] == metadata["stream_sha256"] == sha(prior.raw_path(seed, "withdraw"))
        inputs[str(seed)] = {str(path): sha(path) for path in source_inputs(seed)}
        initial[str(seed)] = dict(world_digest=state_digest(state), world_tick=state["tick"], native_time=state["now"],
            family_allowed=state["context"]["family_allowed"], regime=state["regime"],
            inherited_all_retained_families_are_B=True, active_requests=int(state["tensors"]["active"].sum()))
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
        warm=WARM, total_ticks=TOTAL, train_steps=TRAIN_STEPS, block=BLOCK, updates=UPDATES,
        batch=BATCH, dt=DT, time_scale=TIME_SCALE, source_cursor=SOURCE_CURSOR,
        first_world_tick=SOURCE_CURSOR, final_world_tick=SOURCE_CURSOR+TOTAL-1,
        source_sha256={str(path): sha(path) for path in sources()}, input_sha256=inputs,
        initial_worlds=initial, mappings={str(seed): prior.mapping_table(seed, "related") for seed in SEEDS},
        purpose="Prospectively fixed delayed retention after8192 additional B-only learning ticks with zero review. Separate suffix protocol; never edits or changes the primary256-update replay experiment. No heldout outcomes inspected to declare length or targets.",
        world="Restore complete immutable coverage-retention withdraw final world at66047; no setter/change/reset/warm. Actual midpoint teacher observe/oracle/act for global66048..74239; same one stream per seed for all3 model arms.",
        labels="Native public56 only as inputs. All current native11 actions; float32 signed(ready-now)/16 on active&started&finite-ready, zero offmask. No eligibility/future outcome filtering.",
        family="Initial retained requests and current/scheduled family already B. Audit zero A NEW/REPLACE/CANCEL, active/nonWAIT/time exposure and mature pulses; no special exclusion of inherited tasks.",
        model_use="Data collection does not load models. Future learner must wait for all primary endpoints frozen, restore each complete model+Adam512+live h, run exactly64 native B-only updates without replay, then freeze before evaluating.",
        audit="Full independent native replay compares every public obs, native teacher action, receipt, private fact hash and exact final state. Separate accepted-BEGIN ledger proves current signed labels, seeding retained finite clocks only from accepted BEGIN in the last512 immutable past withdraw rows.",
        raw_fact_fields=FACT_FIELDS, public_feature_layout=FEATURE_LAYOUT)
    # Native regime tuples become JSON arrays. Hash the exact canonical JSON
    # structure that checked() will read, preserving all parameter values.
    plan = json.loads(json.dumps(plan))
    plan["protocol_hash"] = state_digest(plan)
    write_new(PLAN, plan)
    return plan


def checked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert state_digest({key: value for key, value in plan.items() if key != "protocol_hash"}) == plan["protocol_hash"]
    for paths in (plan["source_sha256"], *plan["input_sha256"].values()):
        for path, digest in paths.items():
            assert sha(path) == digest, "frozen retention data binding changed: "+path
    return plan


def collect(seed):
    plan = checked()
    for path in (raw_path(seed), states_path(seed), metadata_path(seed)):
        if path.exists():
            raise FileExistsError(path)
    facts, context = collect_native(seed, progress=True)
    assert state_digest(context["initial"]) == plan["initial_worlds"][str(seed)]["world_digest"]
    counts = verify(facts, seed, context["initial"])
    assert facts["world_tick"][0] == SOURCE_CURSOR and facts["world_tick"][-1] == SOURCE_CURSOR+TOTAL-1
    raw_path(seed).parent.mkdir(parents=True, exist_ok=True)
    with raw_path(seed).open("xb") as handle:
        np.savez_compressed(handle, **facts)
    with states_path(seed).open("xb") as handle:
        torch.save({stage: context[stage] for stage in ("initial", "final")}, handle)
    blocks = [dict(update=update+1, start_tick=update*BLOCK, stop_tick=(update+1)*BLOCK,
        **family_exposure(facts, seed, update*BLOCK, (update+1)*BLOCK)) for update in range(UPDATES)]
    assert all(sum(row[key] for row in blocks) == value for key, value in counts.items())
    result = dict(format=FORMAT, seed=int(seed), protocol_hash=plan["protocol_hash"], total_ticks=TOTAL,
        warm=0, block=BLOCK, updates=UPDATES, source_cursor=SOURCE_CURSOR,
        raw_path=str(raw_path(seed)), stream_sha256=sha(raw_path(seed)),
        states_path=str(states_path(seed)), states_sha256=sha(states_path(seed)),
        initial_world_digest=state_digest(context["initial"]), final_world_digest=state_digest(context["final"]),
        source_world_sha256=sha(source_world_path(seed)), source_sha256=plan["source_sha256"],
        input_sha256=plan["input_sha256"][str(seed)], counts_all=counts, block_counts=blocks,
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in facts.items()},
        events_sha256=context["events_sha256"], case_ids=context["case_ids"], native_metrics=context["native_metrics"])
    write_new(metadata_path(seed), result)
    print(json.dumps(dict(event="retention_collected", seed=int(seed), stream_sha256=result["stream_sha256"],
        time_rows=counts["time_rows"])), flush=True)


@torch.inference_mode()
def audit(seed):
    plan = checked()
    torch.set_num_threads(1)
    facts, metadata = load_facts(seed), json.loads(metadata_path(seed).read_text(encoding="utf-8"))
    states = torch.load(states_path(seed), map_location="cpu", weights_only=False)
    source = torch.load(source_world_path(seed), map_location="cpu", weights_only=False)["final"]
    assert sha(raw_path(seed)) == metadata["stream_sha256"] and sha(states_path(seed)) == metadata["states_sha256"]
    assert state_digest(states["initial"]) == state_digest(source) == plan["initial_worlds"][str(seed)]["world_digest"]
    replay_hashes = {key: hashlib.sha256() for key in FACT_FIELDS}
    with prior.mapping_scope(plan["mappings"][str(seed)]):
        world = ContextWorld.from_state_dict(states["initial"], device="cpu")
        for local in range(TOTAL):
            np.testing.assert_array_equal(world.observe().numpy(), facts["obs"][local])
            np.testing.assert_array_equal(world.oracle("midpoint").numpy(), facts["actions"][local])
            assert world.tick == facts["world_tick"][local] == SOURCE_CURSOR+local
            assert world.now == facts["now"][local] == (SOURCE_CURSOR+local)*DT
            for key, attr in FACT_FIELDS.items():
                replay_hashes[key].update(getattr(world, attr).numpy())
            world.act(torch.from_numpy(facts["actions"][local]))
            np.testing.assert_array_equal(world.last_result.numpy(), facts["receipts"][local])
            if (local+1) % 2048 == 0:
                print(json.dumps(dict(event="retention_audit_progress", seed=seed, ticks=local+1)), flush=True)
        assert state_digest(world.state_dict()) == state_digest(states["final"]) == metadata["final_world_digest"]
        assert world.event_hash == metadata["events_sha256"] and list(world.case_ids) == metadata["case_ids"]
    for key, digest in replay_hashes.items():
        assert digest.hexdigest() == hashlib.sha256(np.ascontiguousarray(facts[key])).hexdigest(), key
    counts = verify(facts, seed, states["initial"])
    assert counts == metadata["counts_all"]
    ledger = accepted_begin_ledger(facts, states["initial"], source_prefix(seed))
    assert ledger["time_rows"] == counts["time_rows"]
    result = dict(format=FORMAT, seed=int(seed), complete=True, protocol_hash=plan["protocol_hash"],
        stream_sha256=sha(raw_path(seed)), states_sha256=sha(states_path(seed)),
        initial_source_world_all_leaves_exact=True, native_replay_ticks=TOTAL,
        public_obs_actions_receipts="bitwise exact every tick", native_private_fact_fields=list(FACT_FIELDS),
        final_native_world_all_leaves_exact=True, accepted_begin_ledger=ledger,
        A_exposures={key: value for key, value in counts.items() if key.startswith("A_")},
        counts_all=counts, auditor_sha256=sha(__file__))
    write_new(audit_path(seed), result)
    print(json.dumps(dict(event="retention_audited", seed=int(seed), ticks=TOTAL, ledger=ledger)), flush=True)


def launch(command):
    checked()
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers = []
    for seed in SEEDS:
        path = logs / f"{command}_seed_{seed:03d}.log"
        handle = path.open("x", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.episodic_review.retention_data",
            command, "--seed", str(seed)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        workers.append((seed, process, handle))
    done = []
    while workers:
        running = []
        for seed, process, handle in workers:
            status = process.poll()
            if status is None:
                running.append((seed, process, handle))
            else:
                handle.close()
                done.append(dict(seed=seed, exit_code=status))
                print(json.dumps(dict(event="retention_worker_exit", command=command, **done[-1])), flush=True)
        workers = running
        if workers:
            time.sleep(.25)
    write_new(OUT / (command+"_workers.json"), done)
    assert not any(row["exit_code"] for row in done), "retention data worker failed; see logs"
    return done


def complete():
    plan = checked()
    reports = {}
    for seed in SEEDS:
        report = json.loads(audit_path(seed).read_text(encoding="utf-8"))
        assert report["complete"] and report["protocol_hash"] == plan["protocol_hash"]
        assert report["stream_sha256"] == sha(raw_path(seed)) and report["states_sha256"] == sha(states_path(seed))
        reports[str(seed)] = {str(path): sha(path) for path in (raw_path(seed), states_path(seed), metadata_path(seed), audit_path(seed))}
    write_new(OUT / "completion.json", dict(format=FORMAT, complete=True, seeds=list(SEEDS),
        protocol_hash=plan["protocol_hash"], reports=reports, models_loaded=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "collect", "audit", "all", "complete"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "declare":
        print(json.dumps(dict(event="retention_data_declared", protocol_hash=declare()["protocol_hash"])), flush=True)
    elif args.command in ("collect", "audit"):
        if args.seed is None:
            launch(args.command)
        else:
            (collect if args.command == "collect" else audit)(args.seed)
    elif args.command == "complete":
        complete()
    else:
        launch("collect")
        launch("audit")
        complete()


if __name__ == "__main__":
    main()
