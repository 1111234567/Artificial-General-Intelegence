"""Actual native continuations of the immutable midpoint teacher endpoints.

All three arms restore the entire same final native world. The only boundary
changes are the declared future NEW configuration and its native history.
Local stream rows do not change the global native clock or request identities.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.teacher_timing import data as source
from experiments.continuity.run import state_digest
from experiments.experience_transfer.train_transfer import SEEDS as SEED_SPECS
from experiments.lifelong.world import ContextWorld, FEATURE_LAYOUT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/coverage-retention/streams-v1"
PLAN = OUT / "protocol.json"
SEEDS, ARMS = (0, 1, 2), ("control", "wide", "withdraw")
WARM, TRAIN_STEPS, TOTAL, BLOCK, BATCH, UPDATES, DT = 0, 32768, 32768, 128, 32, 256, .25
SOURCE_CURSOR, TIME_SCALE = 33280, 16.
FACT_FIELDS, STREAM_FIELDS = source.FACT_FIELDS, source.STREAM_FIELDS
sha, read, write_new, array_hash = source.sha, source.read, source.write_new, source.array_hash
mapping_scope, mapping_table = source.mapping_scope, source.mapping_table
signed_targets = source.signed_targets


def source_checkpoint(seed):
    assert int(seed) in SEEDS
    return ROOT / f"data/teacher-timing/train-v1/midpoint/seed_{int(seed):03d}/action/step256.pt"


def source_world_path(seed):
    return source.states_path(seed, "midpoint")


def arm_dir(seed, arm):
    if int(seed) not in SEEDS or arm not in ARMS:
        raise ValueError((seed, arm))
    return OUT / f"seed_{int(seed):03d}" / arm


def raw_path(seed, arm):
    return arm_dir(seed, arm) / "stream.npz"


def metadata_path(seed, arm):
    return arm_dir(seed, arm) / "stream.json"


def states_path(seed, arm):
    return arm_dir(seed, arm) / "world_states.pt"


def audit_path(seed, arm):
    return arm_dir(seed, arm) / "raw_audit.json"


def load_stream(seed, arm):
    with np.load(raw_path(seed, arm), allow_pickle=False) as archive:
        return {key: archive[key] for key in STREAM_FIELDS}


def load_facts(seed, arm):
    with np.load(raw_path(seed, arm), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def input_paths(seed, arm):
    return [raw_path(seed, arm), metadata_path(seed, arm), states_path(seed, arm),
            audit_path(seed, arm), OUT / f"seed_{int(seed):03d}" / "control_withdraw_pair.json"]


def regime_change(seed, arm):
    arm_dir(seed, arm)
    return dict(duration_range=(8., 32.) if arm == "wide" else (8., 16.),
                gap_range=(16., 32.), window_range=(1.5, 2.5), cancel=.2, replace=.3,
                name="coverage-retention/continuation")


def family_policy(seed, arm):
    spec = SEED_SPECS[int(seed)]
    return (spec["B_cue"],) if arm == "withdraw" else (spec["history_cue"], spec["B_cue"])


def boundary_audit(before, after, seed, arm):
    """Prove every undeclared native state leaf is exactly preserved."""
    expected = deepcopy(before)
    change = regime_change(seed, arm)
    expected["regime"] = {**before["regime"], **change}
    expected["regime"]["name"] = expected["regime"].pop("name")
    expected["regime_id"] += 1
    expected["regime_history"].append(dict(tick=before["tick"], regime_id=expected["regime_id"],
                                           **expected["regime"]))
    policy = family_policy(seed, arm)
    expected["context"]["family_allowed"] = list(policy)
    expected["context"]["future_family"] = policy[0] if len(policy) == 1 else None
    expected["context"]["history"].append(dict(tick=before["tick"], allowed=list(policy)))
    assert state_digest(expected) == state_digest(after), "undeclared native boundary change"
    invariant_before, invariant_after = deepcopy(before), deepcopy(after)
    for value in (invariant_before, invariant_after):
        for key in ("regime", "regime_id", "regime_history"):
            value.pop(key)
        for key in ("family_allowed", "future_family", "history"):
            value["context"].pop(key)
    assert state_digest(invariant_before) == state_digest(invariant_after)
    return dict(exact=True, source_world_digest=state_digest(before),
        after_change_world_digest=state_digest(after),
        unchanged_numerical_state_digest=state_digest(invariant_before),
        permitted_changes=["regime", "regime_id", "regime_history", "context/family_allowed",
                           "context/future_family", "context/history"],
        preserved="All other leaves bitwise exact, including scheduler/pending durations and families, world time, event prefix, RNG, assigned contexts, actions/receipts and counters.")


@torch.inference_mode()
def collect_native(seed, arm, total=TOTAL, initial_state=None, progress=False):
    torch.set_num_threads(1)
    if initial_state is None:
        initial_state = torch.load(source_world_path(seed), map_location="cpu", weights_only=False)["final"]
        assert initial_state["tick"] + 1 == SOURCE_CURSOR
    with mapping_scope(mapping_table(seed, "related")):
        world = ContextWorld.from_state_dict(initial_state, device="cpu")
        before = world.state_dict()
        assert state_digest(before) == state_digest(initial_state)
        world.set_regime(**regime_change(seed, arm))
        world.set_family_policy(family_policy(seed, arm))
        after = world.state_dict()
        audit = boundary_audit(before, after, seed, arm)
        facts = {name: np.empty((total, *getattr(world, attr).shape), dtype=getattr(world, attr).numpy().dtype)
                 for name, attr in FACT_FIELDS.items()}
        shape = total, world.batch_size, world.n_jobs
        facts.update(obs=np.empty((total, world.batch_size, 56), np.float32),
            actions=np.empty(shape, np.int64), receipts=np.empty(shape, np.int64),
            cycle_id=np.empty(shape, np.int64),event_cycle_id=np.empty(shape,np.int64),mature_cycle_id=np.empty(shape,np.int64),
            now=np.empty(total, np.float64), tick=np.arange(total, dtype=np.int64),
            world_tick=np.empty(total, np.int64))
        for local in range(total):
            previous_cycle=world._modify_request_id.copy()
            obs = world.observe()
            action = world.oracle("midpoint")
            for name, attr in FACT_FIELDS.items():
                facts[name][local] = getattr(world, attr).numpy()
            facts["obs"][local], facts["actions"][local] = obs.numpy(), action.numpy()
            facts["now"][local], facts["world_tick"][local] = world.now, world.tick
            facts["cycle_id"][local]=world._modify_request_id
            facts["event_cycle_id"][local]=np.where(world.event_kind.numpy()!=0,world._modify_request_id,-1)
            facts["mature_cycle_id"][local]=np.where(world.matured_mask.numpy(),previous_cycle,-1)
            world.act(action)
            facts["receipts"][local] = world.last_result.numpy()
            if progress and (local+1) % 4096 == 0:
                print(json.dumps(dict(event="collect_progress", seed=seed, arm=arm, ticks=local+1)), flush=True)
        facts["actor_ids"] = np.asarray(world.actor_ids, np.int64)
        facts["time_target"], facts["time_mask"] = signed_targets(
            facts["ready"], facts["now"], facts["active"], facts["started"])
        context = dict(before_change=before, after_change=after, final=world.state_dict(),
            boundary_audit=audit, native_metrics=world.metrics(), events_sha256=world.event_hash,
            case_ids=list(world.case_ids))
    return facts, context


def counts(facts, seed, start=0, stop=None, source_cursor=SOURCE_CURSOR):
    stop = len(facts["actions"]) if stop is None else stop
    result = source.counts(facts, start, stop)
    f = {key: value[start:stop] for key, value in facts.items() if key != "actor_ids"}
    families = (("A", SEED_SPECS[seed]["history_cue"]), ("B", SEED_SPECS[seed]["B_cue"]))
    old = (f["rid"] >= 0) & (f["rid"] < source_cursor*2)
    for name, cue in families:
        current = f["family"] == cue
        old_cycle=(f["cycle_id"]>=0)&(f["cycle_id"]<source_cursor*2)
        for origin, origin_mask in (("all", np.ones(current.shape, bool)), ("inherited", old), ("new", ~old),
                                    ("inherited_cycle",old_cycle),("new_cycle",~old_cycle)):
            valid = current & origin_mask
            prefix = f"{name}_{origin}_"
            result[prefix+"action_labels"] = int(valid.sum())
            result[prefix+"active_labels"] = int((valid & f["active"]).sum())
            result[prefix+"nonWAIT_labels"] = int((valid & (f["actions"] != 0)).sum())
            result[prefix+"time_labels"] = int((valid & f["time_mask"]).sum())
            result[prefix+"BEGIN"] = int((valid & (f["actions"] >= 1) & (f["actions"] <= 2)).sum())
            result[prefix+"FINISH"] = int((valid & (f["actions"] >= 3)).sum())
            event_origin = ((f["event_id"] >= 0) & (f["event_id"] < source_cursor*2)) if origin == "inherited" else (
                f["event_id"] >= source_cursor*2 if origin == "new" else np.ones(valid.shape, bool))
            if origin.endswith("_cycle"):
                event_origin=(f["event_cycle_id"]>=0)&(f["event_cycle_id"]<source_cursor*2) if origin=="inherited_cycle" else f["event_cycle_id"]>=source_cursor*2
            for kind, code in (("NEW", 1), ("REPLACE", 2), ("CANCEL", 3)):
                result[prefix+kind] = int(((f["event_family"] == cue) & (f["event_kind"] == code) & event_origin).sum())
            mature_origin = ((f["mature_id"] >= 0) & (f["mature_id"] < source_cursor*2)) if origin == "inherited" else (
                f["mature_id"] >= source_cursor*2 if origin == "new" else np.ones(valid.shape, bool))
            if origin.endswith("_cycle"):
                mature_origin=(f["mature_cycle_id"]>=0)&(f["mature_cycle_id"]<source_cursor*2) if origin=="inherited_cycle" else f["mature_cycle_id"]>=source_cursor*2
            result[prefix+"mature"] = int(((f["mature_family"] == cue) & f["mature"] & mature_origin).sum())
        for bin_name, lo, hi in (("8_16", 8., 16.), ("16_32", 16., 32.)):
            duration = (f["duration"] >= lo) & (f["duration"] <= hi)
            if lo == 16.:
                duration &= f["duration"] > 16.
            prefix = f"{name}_duration_{bin_name}_"
            result[prefix+"active_labels"] = int((current & f["active"] & duration).sum())
            result[prefix+"time_labels"] = int((current & f["time_mask"] & duration).sum())
            result[prefix+"nonWAIT_labels"] = int((current & (f["actions"] != 0) & duration).sum())
            event_duration = (f["event_duration"] >= lo) & (f["event_duration"] <= hi)
            mature_duration = (f["mature_duration"] >= lo) & (f["mature_duration"] <= hi)
            if lo == 16.:
                event_duration &= f["event_duration"] > 16.
                mature_duration &= f["mature_duration"] > 16.
            for kind, code in (("NEW", 1), ("REPLACE", 2)):
                result[prefix+kind] = int(((f["event_family"] == cue) & (f["event_kind"] == code) & event_duration).sum())
            result[prefix+"mature"] = int(((f["mature_family"] == cue) & f["mature"] & mature_duration).sum())
    return result


def withdrawal_tail(facts, seed, source_cursor=SOURCE_CURSOR):
    cue = SEED_SPECS[seed]["history_cue"]
    inherited = (facts["family"] == cue) & (facts["cycle_id"] >= 0) & (facts["cycle_id"] < source_cursor*2)
    selections = dict(active_labels=inherited & facts["active"], time_labels=inherited & facts["time_mask"],
        nonWAIT_labels=inherited & (facts["actions"] != 0),
        REPLACE=(facts["event_family"] == cue) & (facts["event_kind"] == 2) & (facts["event_cycle_id"] < source_cursor*2),
        CANCEL=(facts["event_family"] == cue) & (facts["event_kind"] == 3) & (facts["event_cycle_id"] < source_cursor*2),
        mature=(facts["mature_family"] == cue) & facts["mature"] & (facts["mature_cycle_id"] < source_cursor*2))
    result = {}
    for name, mask in selections.items():
        rows = np.nonzero(np.any(mask, axis=(1,2)))[0]
        result[name] = dict(count=int(mask.sum()), last_local_tick=int(rows[-1]) if len(rows) else None,
            last_world_tick=int(facts["world_tick"][rows[-1]]) if len(rows) else None,
            last_native_time=float(facts["now"][rows[-1]]) if len(rows) else None)
    return result


def verify_facts(facts, seed, arm, before):
    a, now = facts["actions"], facts["now"][:, None, None]
    begin = facts["active"] & ~facts["started"] & (now+facts["duration"] <= facts["deadline"]+1e-6)
    finish = (facts["active"] & facts["started"] & (now >= facts["ready"]-1e-6)
              & (now <= facts["deadline"]+1e-6) & ((now >= .5*(facts["ready"]+facts["deadline"])-1e-6)
              | (now+DT > facts["deadline"]+1e-6)))
    expected = np.where(begin, 1+facts["version"], np.where(finish, 3+4*facts["version"]+facts["payload"], 0))
    np.testing.assert_array_equal(expected, a)
    np.testing.assert_array_equal(facts["receipts"], (a != 0).astype(np.int64))
    target, mask = signed_targets(facts["ready"], facts["now"], facts["active"], facts["started"])
    np.testing.assert_array_equal(target, facts["time_target"])
    np.testing.assert_array_equal(mask, facts["time_mask"])
    n = len(a)
    np.testing.assert_array_equal(facts["tick"], np.arange(n))
    np.testing.assert_array_equal(facts["world_tick"], before["tick"]+1+np.arange(n))
    np.testing.assert_array_equal(facts["now"], before["now"]+DT*(1+np.arange(n)))
    obs = facts["obs"].reshape(n, a.shape[1], 2, 28)
    previous = np.concatenate((before["tensors"]["_ack_action"].numpy()[None], a[:-1]))
    previous_result = np.concatenate((before["tensors"]["_ack_result"].numpy()[None], facts["receipts"][:-1]))
    np.testing.assert_array_equal(obs[...,11:22], np.eye(11,dtype=np.float32)[previous]*(previous != 0)[...,None])
    np.testing.assert_array_equal(obs[...,22], previous_result == 1)
    np.testing.assert_array_equal(obs[...,23], previous_result == -1)
    assert not facts["had_error"].any()
    np.testing.assert_array_equal(facts["mature"], facts["mature_done"])
    np.testing.assert_array_equal(facts["mature"], facts["mature_strict"])
    if arm == "withdraw":
        assert not ((facts["event_kind"] == 1) & (facts["event_family"] == SEED_SPECS[seed]["history_cue"])).any()
    return dict(teacher="actual native midpoint every tick", checked_ticks=n,
        source_receipt_row0="exact source final actions/results", native_clock="global and continuous",
        labels="all active started finite-ready rows, signed /16; eligibility never masks",
        matured_ordinary_completion=1., matured_strict_completion=1.)


def sources():
    return list(dict.fromkeys([Path(__file__), Path(__file__).with_name("data_test.py"), OUT/"audit_raw.py",
        *source.sources()]))


source_files = sources


def declare():
    source.checked()
    inputs, initial = {}, {}
    for seed in SEEDS:
        paths = [source_checkpoint(seed), source_checkpoint(seed).with_suffix(".json"), source_world_path(seed),
            source.raw_path(seed,"midpoint"), source.metadata_path(seed,"midpoint"), source.audit_path(seed,"midpoint")]
        inputs[str(seed)] = {str(p.relative_to(ROOT)):sha(p) for p in paths}
        state = torch.load(source_world_path(seed),map_location="cpu",weights_only=False)["final"]
        checkpoint = torch.load(source_checkpoint(seed),map_location="cpu",weights_only=False)
        assert (state["tick"]+1, checkpoint["stream_cursor"], checkpoint["step"], checkpoint["arm"]) == (SOURCE_CURSOR,SOURCE_CURSOR,256,"action")
        initial[str(seed)] = dict(world_digest=state_digest(state), source_world_file_sha256=sha(source_world_path(seed)),
            source_checkpoint_sha256=sha(source_checkpoint(seed)), source_model_digest=state_digest(checkpoint["model"]),
            source_Adam_digest=state_digest(checkpoint["optimizer"]), source_hidden_digest=state_digest(checkpoint["hidden"]),
            world_tick=state["tick"], native_time=state["now"], source_cursor=SOURCE_CURSOR,
            regime=state["regime"], family_allowed=state["context"]["family_allowed"])
    plan = dict(format="coverage-retention-native-continuations-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), arms=list(ARMS), runs=9, warm=WARM, train_steps=TRAIN_STEPS,total_ticks=TOTAL,
        block=BLOCK,updates=UPDATES,batch_size=BATCH,dt=DT,time_scale=TIME_SCALE,source_cursor=SOURCE_CURSOR,
        mappings={str(s):mapping_table(s,"related") for s in SEEDS},
        source_files={str(p.relative_to(ROOT)):sha(p) for p in sources()}, immutable_input_sha256=inputs,
        source_protocol_sha256=sha(source.PLAN), initial_worlds=initial,
        changes={str(s):{a:dict(set_regime=regime_change(s,a), set_family_policy=list(family_policy(s,a))) for a in ARMS} for s in SEEDS},
        changes_order=["restore full source final world", "set_regime", "set_family_policy", "save after-change full state", "first observe"],
        continuity="No world clock, RNG, receipts, scheduler, pending request/duration/family or counters reset. Pending inherited replacements retain their original A/B context and sampled duration. Only future NEW draws see changed policies.",
        teacher="Actual same native midpoint policy observe/oracle/act on all32768 new ticks. No new warmup.",
        request_id="Native global world_tick*2+job for both NEW and REPLACE. Local tick is separate; never index local facts using request_id//2. cycle_id retains original NEW id, so a pending inherited REPLACE can have new global rid while retaining inherited cycle/family/duration. Both origins counted separately.",
        target="Native float64(ready-now)/16 saved float32 on active & started & finite ready, zero elsewhere; every valid row including future-ineligible requests retained.",
        pair="control and withdraw have exact physical RAW event histories, version/duration/scheduler and RNG; mapped target/context differs. Wide changes duration and resulting schedules, without an equality assertion.",
        audit="After collection, independent full native replay from saved after-change state plus independent accepted-BEGIN ledger reconstructed through immutable source public history, across all3seeds9worlds.",
        counts="Totals and each128tick block: native11 actions; A/B retained-family action labels, active/nonWAIT/time labels, BEGIN/FINISH, NEW/REPLACE/CANCEL/mature by inherited/new global request IDs AND inherited/new original-NEW cycle IDs; active/time/nonWAIT/NEW/REPLACE/mature duration bins[8,16],(16,32]. Withdrawal inherited-cycle A counts and exact last local/global tick retained, including newly identified REPLACE sampled before change.",
        raw_fact_fields=FACT_FIELDS,public_feature_layout=FEATURE_LAYOUT)
    write_new(PLAN,plan)
    print(json.dumps(dict(event="declared",protocol_sha256=sha(PLAN),runs=9)),flush=True)
    return plan


def checked():
    plan=read(PLAN)
    assert (plan["warm"],plan["train_steps"],plan["total_ticks"],plan["source_cursor"]) == (0,32768,32768,33280)
    for name,digest in plan["source_files"].items():
        assert sha(ROOT/name)==digest,"Frozen continuation source changed: "+name
    for paths in plan["immutable_input_sha256"].values():
        for name,digest in paths.items():
            assert sha(ROOT/name)==digest,"Immutable source changed: "+name
    assert sha(source.PLAN)==plan["source_protocol_sha256"]
    return plan


def collect(seed,arm):
    plan=checked()
    if raw_path(seed,arm).exists() or metadata_path(seed,arm).exists():
        raise FileExistsError(raw_path(seed,arm))
    facts,context=collect_native(seed,arm,progress=True)
    report=verify_facts(facts,seed,arm,context["before_change"])
    assert state_digest(context["before_change"])==plan["initial_worlds"][str(seed)]["world_digest"]
    path=raw_path(seed,arm)
    path.parent.mkdir(parents=True,exist_ok=True)
    with states_path(seed,arm).open("xb") as handle:
        torch.save({k:context[k] for k in ("before_change","after_change","final")},handle)
    with path.open("xb") as handle:
        np.savez_compressed(handle,**facts)
    total=counts(facts,seed)
    blocks=[dict(update=i+1,start_tick=i*BLOCK,stop_tick=(i+1)*BLOCK,
                 **counts(facts,seed,i*BLOCK,(i+1)*BLOCK)) for i in range(UPDATES)]
    for key,value in total.items():
        assert sum(b[key] for b in blocks)==value,key
    result=dict(format="coverage-retention-native-stream-v1",seed=seed,arm=arm,teacher="midpoint",
        protocol_sha256=sha(PLAN),source_checkpoint_sha256=sha(source_checkpoint(seed)),
        source_world_sha256=sha(source_world_path(seed)),source_cursor=SOURCE_CURSOR,
        warm=0,train_steps=TOTAL,total_ticks=TOTAL,block=BLOCK,updates=UPDATES,
        stream_sha256=sha(path),world_states_sha256=sha(states_path(seed,arm)),
        raw_path=str(path.relative_to(ROOT)),world_states_path=str(states_path(seed,arm).relative_to(ROOT)),
        mapping_table=mapping_table(seed,"related"),changes=plan["changes"][str(seed)][arm],
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in facts.items()},
        counts_all=total,counts_training=total,block_counts=blocks,
        boundary_audit=context["boundary_audit"],audit=report,
        training_stream_sha256=array_hash(facts,STREAM_FIELDS),public_observation_sha256=array_hash(facts,("obs",)),
        events_sha256=context["events_sha256"],case_ids=context["case_ids"],native_metrics=context["native_metrics"],
        A_inherited_tail=withdrawal_tail(facts,seed),source_files=plan["source_files"])
    write_new(metadata_path(seed,arm),result)
    checked()
    print(json.dumps(dict(event="collected",seed=seed,arm=arm,stream_sha256=result["stream_sha256"],
                         time_rows=total["time_rows"],A_inherited_tail=result["A_inherited_tail"])),flush=True)


PAIR_FIELDS = ("event_kind","event_id","event_eligible","event_raw","event_version","event_duration",
    "event_window","event_regime_id","rid","version","raw","duration","deadline","issued_time",
    "eligible","last_external_tick","last_external_time","mature","mature_id","mature_issued",
    "mature_duration","tick","world_tick","now","actor_ids","ready","cycle_id","event_cycle_id","mature_cycle_id")


def compare_pair(seed):
    left,right=load_facts(seed,"control"),load_facts(seed,"withdraw")
    for key in PAIR_FIELDS:
        np.testing.assert_array_equal(left[key],right[key],err_msg="control/withdraw physical "+key)
    for arm in ("control","withdraw"):
        assert np.all(left["event_duration"][left["event_kind"]==1] <= 16.)
    states=[torch.load(states_path(seed,a),map_location="cpu",weights_only=False) for a in ("control","withdraw")]
    for boundary in ("before_change","after_change","final"):
        a,b=(s[boundary] for s in states)
        assert state_digest(a["scheduler"])==state_digest(b["scheduler"]),boundary
        assert torch.equal(a["context"]["rng_state"],b["context"]["rng_state"]),boundary
    np.testing.assert_array_equal((left["actions"]>=1)&(left["actions"]<=2),(right["actions"]>=1)&(right["actions"]<=2))
    result=dict(seed=seed,protocol_sha256=sha(PLAN),physical_fields_bitwise_exact=list(PAIR_FIELDS),
        scheduler_and_external_RNG="all leaves exact at before/after/final",context_RNG="exact at before/after/final",
        actual_context_differences=int(np.count_nonzero(left["family"]!=right["family"])),
        actual_mapped_payload_differences=int(np.count_nonzero(left["payload"]!=right["payload"])),
        begin_opportunities="exact",wide="duration and scheduler may differ by design; no match claim",
        streams={a:sha(raw_path(seed,a)) for a in ("control","withdraw")})
    write_new(OUT/f"seed_{seed:03d}"/"control_withdraw_pair.json",result)
    return result


def launch(max_workers=6):
    checked()
    assert 1 <= max_workers <= 6
    logs=OUT/"logs"; logs.mkdir(parents=True,exist_ok=True)
    pending=[(s,a) for s in SEEDS for a in ARMS]
    active=[]; done=[]
    while pending or active:
        while pending and len(active)<max_workers:
            seed,arm=pending.pop(0)
            log=logs/f"seed_{seed:03d}_{arm}.log"
            handle=log.open("x",encoding="utf-8")
            process=subprocess.Popen([sys.executable,"-m","experiments.coverage_retention.data","collect","--seed",str(seed),"--arm",arm],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            active.append((seed,arm,process,handle,log))
        running=[]
        for seed,arm,process,handle,log in active:
            code=process.poll()
            if code is None:
                running.append((seed,arm,process,handle,log))
            else:
                handle.close(); done.append(dict(seed=seed,arm=arm,pid=process.pid,exit_code=code,log=str(log)))
                print(json.dumps(done[-1]),flush=True)
        active=running
        if pending or active:
            time.sleep(.25)
    write_new(OUT/"workers.json",dict(workers=done))
    assert not any(row["exit_code"] for row in done),"Collection worker failed; see logs"
    for seed in SEEDS:
        compare_pair(seed)
    write_new(OUT/"completion.json",dict(complete=True,seeds=list(SEEDS),arms=list(ARMS),protocol_sha256=sha(PLAN),
        metadata={str(s):{a:sha(metadata_path(s,a)) for a in ARMS} for s in SEEDS}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","collect","launch"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--max-workers",type=int,default=6)
    args=parser.parse_args()
    if args.command=="declare": declare()
    elif args.command=="collect": collect(args.seed,args.arm)
    else: launch(args.max_workers)
