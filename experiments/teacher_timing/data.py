"""Matched actual earliest/midpoint teacher streams for a fixed 2x2 experiment.

The earliest learner arrays delegate to immutable original artifacts. Midpoint
observations are newly collected from native actions, including their receipts.
Only public56 observations enter either learner. All clocks below are labels
or audit facts and eligibility never selects a supervised row.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import json

import numpy as np
import torch

from experiments.action_split import data as original
from experiments.remaining_time import targets as original_targets
from experiments.lifelong.world import ContextWorld, FEATURE_LAYOUT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/teacher-timing/streams-v1"
PLAN = OUT / "protocol.json"
SEEDS, TEACHERS = (0, 1, 2), ("earliest", "midpoint")
WARM, TRAIN_STEPS, TOTAL, BLOCK, BATCH, DT = 512, 32768, 33280, 128, 32, .25
UPDATES, TIME_SCALE = 256, 16.
FACT_FIELDS = original.FACT_FIELDS
STREAM_FIELDS = ("obs", "actions", "time_target", "time_mask")
sha, read, write_new, array_hash = original.sha, original.read, original.write_new, original.array_hash
config = original.config
mapping_scope, mapping_table = original.mapping_scope, original.mapping_table


def teacher_dir(seed, teacher):
    if int(seed) not in SEEDS or teacher not in TEACHERS:
        raise ValueError((seed, teacher))
    return OUT / f"seed_{int(seed):03d}" / teacher


def raw_path(seed, teacher):
    return (original.stream_path(seed) if teacher == "earliest"
            else teacher_dir(seed, teacher) / "stream.npz")


def metadata_path(seed, teacher):
    return teacher_dir(seed, teacher) / "stream.json"


def states_path(seed, teacher):
    return (original.stream_path(seed).with_name("world_states.pt") if teacher == "earliest"
            else teacher_dir(seed, teacher) / "world_states.pt")


def audit_path(seed, teacher):
    return teacher_dir(seed, teacher) / "raw_audit.json"


def load_stream(seed, teacher):
    """Exact training-facing array keys, including unchanged earliest anchors."""
    teacher_dir(seed, teacher)
    if teacher == "earliest":
        result = original.load_stream(seed, ("obs", "actions"))
        result.update(original_targets.load_targets(seed, original_targets.TARGET_KEYS))
        return result
    with np.load(raw_path(seed, teacher), allow_pickle=False) as archive:
        return {key: archive[key] for key in STREAM_FIELDS}


def load_facts(seed, teacher):
    with np.load(raw_path(seed, teacher), allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if teacher == "earliest":
        result.update(original_targets.load_targets(seed))
    return result


def signed_targets(ready, now, active, started):
    """Native float64 physical subtraction; finite float32 zero off-mask."""
    ready = np.asarray(ready, np.float64)
    current = np.asarray(now, np.float64)
    if current.ndim == 1:
        current = current.reshape((len(current),) + (1,) * (ready.ndim-1))
    mask = active & started & np.isfinite(ready)
    value = np.zeros(ready.shape, np.float64)
    np.subtract(ready, current, out=value, where=mask)
    return (value / TIME_SCALE).astype(np.float32), mask


def counts(facts, start=0, stop=None):
    stop = len(facts["actions"]) if stop is None else stop
    take = slice(start, stop)
    action, mask = facts["actions"][take], facts["time_mask"][take]
    value = facts["time_target"][take]
    physical = np.zeros(mask.shape, np.float64)
    np.subtract(facts["ready"][take], facts["now"][take, None, None], out=physical, where=mask)
    wait = int((action == 0).sum())
    result = dict(ticks=stop-start, action_rows=int(action.size), action_WAIT_rows=wait,
        action_BEGIN_rows=int(((action == 1) | (action == 2)).sum()),
        action_FINISH_rows=int((action >= 3).sum()),
        action_weight_sum=float(action.size-.75*wait), time_rows=int(mask.sum()),
        time_masked=int((~mask).sum()), time_positive=int((mask & (value > 0)).sum()),
        time_zero=int((mask & (value == 0)).sum()), time_negative=int((mask & (value < 0)).sum()),
        negative_seconds_0_to_025=int((mask & (physical < 0) & (physical >= -.25)).sum()),
        negative_seconds_025_to_1=int((mask & (physical < -.25) & (physical >= -1.)).sum()),
        negative_seconds_lt_1=int((mask & (physical < -1.)).sum()),
        time_eligible_diagnostic=int((mask & facts["eligible"][take]).sum()),
        time_ineligible_diagnostic=int((mask & ~facts["eligible"][take]).sum()),
        matured=int(facts["mature"][take].sum()), matured_done=int(facts["mature_done"][take].sum()),
        matured_strict=int(facts["mature_strict"][take].sum()))
    result.update({f"action_code_{code:02d}": int((action == code).sum()) for code in range(11)})
    assert result["time_rows"] == result["time_positive"]+result["time_zero"]+result["time_negative"]
    assert result["time_negative"] == sum(result[k] for k in (
        "negative_seconds_0_to_025", "negative_seconds_025_to_1", "negative_seconds_lt_1"))
    return result


def finish_records(facts):
    tick, lane, job = np.nonzero((facts["actions"] >= 3) & (facts["receipts"] == 1))
    index = tick, lane, job
    return dict(tick=tick, lane=lane, actor=facts["actor_ids"][lane], job=job,
        rid=facts["rid"][index], eligible=facts["eligible"][index], now=facts["now"][tick],
        ready=facts["ready"][index], deadline=facts["deadline"][index],
        latency_seconds=facts["now"][tick]-facts["ready"][index])


def distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(n=0, min=None, mean=None, p05=None, p25=None, median=None, p75=None, p95=None, max=None)
    q = np.quantile(values, [.05, .25, .5, .75, .95])
    return dict(n=len(values), min=float(values.min()), mean=float(values.mean()),
        p05=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]),
        p95=float(q[4]), max=float(values.max()))


def verify_facts(facts, teacher):
    a, now = facts["actions"], facts["now"][:, None, None]
    begin = facts["active"] & ~facts["started"] & (now+facts["duration"] <= facts["deadline"]+1e-6)
    finish = (facts["active"] & facts["started"] & (now >= facts["ready"]-1e-6)
              & (now <= facts["deadline"]+1e-6))
    if teacher == "midpoint":
        finish &= ((now >= .5*(facts["ready"]+facts["deadline"])-1e-6)
                   | (now+DT > facts["deadline"]+1e-6))
    expected = np.where(begin, 1+facts["version"],
        np.where(finish, 3+4*facts["version"]+facts["payload"], 0))
    np.testing.assert_array_equal(expected, a)
    np.testing.assert_array_equal(facts["receipts"], (a != 0).astype(np.int64))
    target, mask = signed_targets(facts["ready"], facts["now"], facts["active"], facts["started"])
    np.testing.assert_array_equal(target, facts["time_target"])
    np.testing.assert_array_equal(mask, facts["time_mask"])
    obs = facts["obs"].reshape(len(a), a.shape[1], 2, 28)
    np.testing.assert_array_equal(obs[1:, ..., 11:22],
        np.eye(11, dtype=np.float32)[a[:-1]]*(a[:-1] != 0)[..., None])
    np.testing.assert_array_equal(obs[1:, ..., 22], facts["receipts"][:-1] == 1)
    np.testing.assert_array_equal(obs[1:, ..., 23], facts["receipts"][:-1] == -1)
    np.testing.assert_array_equal(facts["tick"], np.arange(len(a)))
    np.testing.assert_array_equal(facts["now"], np.arange(len(a))*DT)
    assert not facts["had_error"].any()
    assert np.array_equal(facts["mature"], facts["mature_done"])
    assert np.array_equal(facts["mature"], facts["mature_strict"])
    return dict(teacher=teacher, actions="exact causal native policy", receipts="all attempted actions accepted",
        labels="float32 exact current-state signed time; no eligibility filter",
        checked_ticks=len(a), matured_ordinary_completion=1., matured_strict_completion=1.)


@torch.inference_mode()
def collect_native(seed, teacher="midpoint", total=TOTAL, world_config=None, progress=False):
    teacher_dir(seed, teacher)
    torch.set_num_threads(1)
    with mapping_scope(mapping_table(seed, "related")):
        world = ContextWorld(**(config(seed) if world_config is None else world_config))
        facts = {name: np.empty((total, *getattr(world, attr).shape),
                    dtype=getattr(world, attr).numpy().dtype) for name, attr in FACT_FIELDS.items()}
        shape = total, world.batch_size, 2
        facts.update(obs=np.empty((total, world.batch_size, 56), np.float32),
            actions=np.empty(shape, np.int64), receipts=np.empty(shape, np.int64),
            now=np.empty(total, np.float64), tick=np.empty(total, np.int64))
        initial = world.state_dict()
        for tick in range(total):
            obs = world.observe()
            action = world.oracle(teacher)
            for name, attr in FACT_FIELDS.items():
                facts[name][tick] = getattr(world, attr).numpy()
            facts["obs"][tick], facts["actions"][tick] = obs.numpy(), action.numpy()
            facts["now"][tick], facts["tick"][tick] = world.now, world.tick
            world.act(action)
            facts["receipts"][tick] = world.last_result.numpy()
            if progress and (tick+1) % 4096 == 0:
                print(json.dumps(dict(event="progress", seed=seed, teacher=teacher, ticks=tick+1)), flush=True)
        facts["actor_ids"] = np.asarray(world.actor_ids, np.int64)
        facts["time_target"], facts["time_mask"] = signed_targets(
            facts["ready"], facts["now"], facts["active"], facts["started"])
        context = dict(initial_world=initial, final_world=world.state_dict(), metrics=world.metrics(),
            events_sha256=world.event_hash, case_ids=list(world.case_ids))
    return facts, context


EXOGENOUS_FIELDS = tuple(k for k in FACT_FIELDS if k.startswith("event_")) + (
    "rid", "version", "payload", "raw", "duration", "deadline", "issued_time", "family", "eligible",
    "last_external_tick", "last_external_time", "mature", "mature_id", "mature_issued", "mature_duration", "mature_family")


def compare_pair(early, middle):
    for key in EXOGENOUS_FIELDS + ("actor_ids", "tick", "now", "ready"):
        np.testing.assert_array_equal(early[key], middle[key], err_msg="paired "+key)
    begin_e = (early["actions"] == 1) | (early["actions"] == 2)
    begin_m = (middle["actions"] == 1) | (middle["actions"] == 2)
    np.testing.assert_array_equal(begin_e, begin_m)
    np.testing.assert_array_equal(early["actions"][begin_e], middle["actions"][begin_m])
    eobs = early["obs"].reshape(len(begin_e), begin_e.shape[1], 2, 28)
    mobs = middle["obs"].reshape(eobs.shape)
    exo = list(range(11))+list(range(24, 28))
    np.testing.assert_array_equal(eobs[..., exo], mobs[..., exo])
    records = {teacher: finish_records(f) for teacher, f in (("earliest", early), ("midpoint", middle))}
    identities = {teacher: {(int(l), int(j), int(r)): i for i, (l, j, r) in enumerate(zip(
        rec["lane"], rec["job"], rec["rid"]))} for teacher, rec in records.items()}
    ei, mi = set(identities["earliest"]), set(identities["midpoint"])
    unmatched = {}
    for teacher, other in (("earliest", mi), ("midpoint", ei)):
        rec = records[teacher]
        unmatched[teacher] = [dict(lane=l, job=j, rid=r,
            finish_tick=int(rec["tick"][identities[teacher][(l,j,r)]]),
            deadline=float(rec["deadline"][identities[teacher][(l,j,r)]]))
            for l,j,r in sorted(set(identities[teacher])-other)]
        for row in unmatched[teacher]:
            assert row["deadline"] >= early["now"][-1]-1e-6, "nonboundary request identity mismatch"
    return dict(exogenous_fields="every row bitwise exact", exogenous_field_names=list(EXOGENOUS_FIELDS),
        ready_clocks="every row bitwise exact", BEGIN_ticks_codes="every row exact",
        matured_request_identities="every row exact", public_exogenous_channels="bitwise exact",
        changed_observation_scalars=int(np.count_nonzero(eobs != mobs)),
        changed_public_action_receipt_scalars=int(np.count_nonzero(eobs[..., 11:24] != mobs[..., 11:24])),
        changed_public_action_receipt_slot_rows=int(np.any(eobs[..., 11:24] != mobs[..., 11:24], axis=-1).sum()),
        action_different_slot_rows=int((early["actions"] != middle["actions"]).sum()),
        finished_request_intersection=len(ei & mi), identical_FINISH_request_sets=ei == mi,
        FINISH_request_horizon_only_differences=unmatched,
        label_budgets={teacher: counts(f, WARM, len(f["actions"])) for teacher, f in (("earliest", early), ("midpoint", middle))},
        interpretation="Actual teacher changes the action/receipt observation background and time-valid row coverage. Equal input ticks, updates and action-row budget do not imply equal auxiliary valid-row budgets or boundary label counts.")


def sources():
    return [Path(__file__), Path(__file__).with_name("data_test.py"),
        OUT / "audit_raw.py", *original.source_files(), *original_targets.source_files()]


source_files = sources


def declare():
    original.checked(); original_targets.checked()
    inputs = {}
    for seed in SEEDS:
        raw, aux = original.stream_path(seed), original_targets.target_path(seed)
        paths = (raw, raw.with_suffix(".json"), raw.with_name("raw_audit.json"), raw.with_name("world_states.pt"),
            aux, aux.with_suffix(".json"))
        inputs[str(seed)] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    plan = dict(format="actual-teacher-timing-streams-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), teachers=list(TEACHERS), warm=WARM, train_steps=TRAIN_STEPS, total_ticks=TOTAL,
        block=BLOCK, updates=UPDATES, batch_size=BATCH, dt=DT, time_scale=TIME_SCALE,
        world_configs={str(seed): config(seed) for seed in SEEDS},
        mappings={str(seed): mapping_table(seed, "related") for seed in SEEDS},
        source_files={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(sources())},
        original_raw_protocol_sha256=sha(original.PLAN), original_target_protocol_sha256=sha(original_targets.PLAN),
        immutable_input_sha256=inputs,
        teacher="earliest: delegate immutable original raw obs/actions and remaining_time targets. midpoint: one new continuous native world, observe; oracle('midpoint'); act every tick. Both teachers really execute their own actions.",
        native_semantics="Unchanged ContextWorld source/state/clock/action/cancel/replace/maturation and action-independent external scheduler; identical seed actors configurations mappings and random streams.",
        target="float64 (native ready_time-native now)/16 saved float32; offmask zero; mask active & started & finite ready; eligibility never selects or weights labels",
        learner_inputs="Exactly native public56 observations and constant dt=.25. Receipt channels come from each teacher's actual previous actions; no clock/private-state/target/eligibility input.",
        budget="Each teacher has one33280tick stream:512warm then256sequential128tick updates and32768*32*2 action rows. Each of action/joint reuses that teacher's same stream once. Midpoint valid auxiliary rows may be larger; no claim of equal auxiliary label budgets. Positive rows at terminal horizon need not be identical.",
        counts="All/warm/postwarm plus each128tick block: native action0..10, WAIT/BEGIN/FINISH, weighted CE denominator WAIT.25 others1, auxiliary positive/zero/negative, physical negative seconds[-.25,0),[-1,-.25),(-inf,-1), eligibility diagnostic only.",
        pair_check="Bitwise external fields, event hash, case IDs, final scheduler/context RNG, BEGIN tick/code and retained ready clock; report actual FINISH identity boundary differences and latency distributions. Matured ordinary/strict completion100% both.",
        audit="Independent native full replay and separate accepted-BEGIN ledger; no collector target helper/oracle/private ready value used by ledger to construct expected labels.",
        public_feature_layout=FEATURE_LAYOUT, raw_fact_fields=FACT_FIELDS)
    write_new(PLAN, plan)
    print(json.dumps(dict(event="declared", protocol_sha256=sha(PLAN))), flush=True)
    return plan


def checked():
    plan = read(PLAN)
    assert (plan["warm"], plan["train_steps"], plan["total_ticks"], plan["updates"]) == (WARM, TRAIN_STEPS, TOTAL, UPDATES)
    assert sha(original.PLAN) == plan["original_raw_protocol_sha256"]
    assert sha(original_targets.PLAN) == plan["original_target_protocol_sha256"]
    for name, digest in plan["source_files"].items():
        assert sha(ROOT/name) == digest, "Frozen data source changed: "+name
    for bindings in plan["immutable_input_sha256"].values():
        for name, digest in bindings.items():
            assert sha(ROOT/name) == digest, "Immutable original input changed: "+name
    original.checked(); original_targets.checked()
    return plan


def save_metadata(seed, teacher, facts, context):
    path = metadata_path(seed, teacher)
    audit = verify_facts(facts, teacher)
    blocks = [dict(update=i+1, start_tick=WARM+i*BLOCK, stop_tick=WARM+(i+1)*BLOCK,
        **counts(facts, WARM+i*BLOCK, WARM+(i+1)*BLOCK)) for i in range(UPDATES)]
    training = counts(facts, WARM, TOTAL)
    for key, value in training.items():
        assert sum(row[key] for row in blocks) == value, key
    records = finish_records(facts)
    record_path = path.with_name("finish_latency.npz")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("xb") as handle:
        np.savez_compressed(handle, **records)
    result = dict(format="actual-native-teacher-stream-v1", seed=int(seed), teacher=teacher,
        world_config=config(seed), mapping_table=mapping_table(seed,"related"), warm=WARM,
        train_steps=TRAIN_STEPS, total_ticks=TOTAL, block=BLOCK, updates=UPDATES,
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in facts.items()},
        raw_path=str(raw_path(seed, teacher).relative_to(ROOT)), stream_sha256=sha(raw_path(seed, teacher)),
        world_states_path=str(states_path(seed, teacher).relative_to(ROOT)), world_states_sha256=sha(states_path(seed, teacher)),
        protocol_sha256=sha(PLAN), source_files=checked()["source_files"],
        counts_all=counts(facts), counts_warm=counts(facts,0,WARM), counts_training=training,
        block_counts=blocks, public_observation_sha256=array_hash(facts,("obs",)),
        training_stream_sha256=array_hash(facts,STREAM_FIELDS),
        public_stream_sha256=array_hash(facts,("obs","actions","receipts")),
        events_sha256=context["events_sha256"], case_ids=context["case_ids"], native_metrics=context["metrics"],
        finish_latency_path=str(record_path.relative_to(ROOT)), finish_latency_sha256=sha(record_path),
        finish_latency_seconds_all=distribution(records["latency_seconds"]),
        finish_latency_seconds_training=distribution(records["latency_seconds"][records["tick"] >= WARM]),
        audit=audit)
    if teacher == "earliest":
        result["original_target_path"] = str(original_targets.target_path(seed).relative_to(ROOT))
        result["original_target_sha256"] = sha(original_targets.target_path(seed))
    write_new(path, result)
    return result


def collect(seed):
    checked()
    for teacher in TEACHERS:
        if metadata_path(seed, teacher).exists():
            raise FileExistsError(metadata_path(seed,teacher))
    early = load_facts(seed,"earliest")
    oldmeta = read(original.stream_path(seed).with_suffix(".json"))
    save_metadata(seed,"earliest",early,dict(metrics=oldmeta["native_metrics"],
        events_sha256=oldmeta["events_sha256"],case_ids=oldmeta["case_ids"]))
    middle, context = collect_native(seed,"midpoint",progress=True)
    pair = compare_pair(early,middle)
    assert context["events_sha256"] == oldmeta["events_sha256"]
    assert context["case_ids"] == oldmeta["case_ids"]
    oldstates = torch.load(states_path(seed,"earliest"),map_location="cpu",weights_only=False)
    from experiments.continuity.run import state_digest
    for stage, key in (("initial","initial_world"),("final","final_world")):
        assert state_digest(oldstates[stage]["scheduler"]) == state_digest(context[key]["scheduler"])
        for field in ("rng_state","assigned"):
            assert torch.equal(oldstates[stage]["context"][field],context[key]["context"][field])
    pair["initial_final_scheduler_rng"] = "bitwise exact"
    path = raw_path(seed,"midpoint")
    path.parent.mkdir(parents=True,exist_ok=True)
    with states_path(seed,"midpoint").open("xb") as handle:
        torch.save(dict(initial=context["initial_world"],final=context["final_world"]),handle)
    with path.open("xb") as handle:
        np.savez_compressed(handle,**middle)
    meta = save_metadata(seed,"midpoint",middle,context)
    pair.update(seed=int(seed),protocol_sha256=sha(PLAN),earliest_stream_sha256=sha(raw_path(seed,"earliest")),
        midpoint_stream_sha256=sha(path), events_sha256=context["events_sha256"],case_ids=context["case_ids"])
    write_new(OUT/f"seed_{seed:03d}"/"pair_comparison.json",pair)
    checked()
    print(json.dumps(dict(event="collected",seed=seed,counts=meta["counts_training"],pair=pair)),flush=True)


def launch():
    checked()
    logs=OUT/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    workers=[]
    for seed in SEEDS:
        log=logs/f"seed_{seed:03d}.log"
        handle=log.open("x",encoding="utf-8")
        process=subprocess.Popen([sys.executable,"-m","experiments.teacher_timing.data","collect","--seed",str(seed)],
            cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        workers.append((seed,process,handle,log))
    finished=[]
    while workers:
        running=[]
        for seed,process,handle,log in workers:
            code=process.poll()
            if code is None:
                running.append((seed,process,handle,log))
            else:
                handle.close()
                finished.append(dict(seed=seed,pid=process.pid,exit_code=code,log=str(log)))
                print(json.dumps(finished[-1]),flush=True)
        workers=running
        if workers:
            time.sleep(.25)
    write_new(OUT/"workers.json",dict(workers=finished))
    assert not any(row["exit_code"] for row in finished),"Data worker failed; see logs"
    write_new(OUT/"completion.json",dict(complete=True,seeds=list(SEEDS),protocol_sha256=sha(PLAN),
        metadata={str(seed):{teacher:sha(metadata_path(seed,teacher)) for teacher in TEACHERS} for seed in SEEDS}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","collect","launch"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.command=="declare":
        declare()
    elif args.command=="collect":
        collect(args.seed)
    else:
        launch()
