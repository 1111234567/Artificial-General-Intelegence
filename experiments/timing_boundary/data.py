"""Frozen timing-boundary histories; no learner is updated by this experiment.

Teacher histories are a common off-policy diagnostic.  Own-action histories
are separately collected on the held-out test cases and retain native receipts.
``h`` is the state AFTER consuming the row's current public observation, exactly
the state used by the unchanged policy head.  It is never reset at requests.
"""
from __future__ import annotations

import os
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import argparse
from collections import Counter
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
from experiments.learn_remember.gru_train import GRUAgent
from experiments.lifelong.evaluate import CONDITIONS
from experiments.lifelong.world import ContextWorld

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/timing-boundary/v1"
SPLITS = ("train", "validation", "test")
TRAINED_ARMS = ("midpoint", "first", "interior")
ARMS = (*TRAINED_ARMS, "random")
WARM, STEPS, DT, BATCH = 512, 4096, .25, 32
OFFSETS = (-8, -4, -2, -1, 0, 1, 2)
EPS = 1e-6

FACT_FIELDS = {
    "rid": "request_id", "active": "active", "started": "started",
    "ready": "ready_time", "deadline": "deadline", "issued_time": "issued_time",
    "version": "version", "payload": "payload", "raw": "raw_payload",
    "family": "current_family", "eligible": "current_eligible",
    "duration": "duration", "had_error": "had_error",
    "event_kind": "event_kind", "event_id": "event_request_id",
    "event_eligible": "event_eligible", "event_family": "event_family",
    "event_raw": "raw_event_payload", "event_version": "event_version",
    "event_duration": "event_duration", "event_window": "event_window",
    "mature": "matured_mask", "mature_id": "matured_request_id",
    "mature_issued": "matured_issued_at", "mature_duration": "matured_duration",
    "mature_family": "matured_family", "mature_done": "matured_completed",
    "mature_strict": "matured_clean_completed",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        x = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(x.dtype).encode())
        digest.update(np.asarray(x.shape, dtype=np.int64).tobytes()); digest.update(x.tobytes())
    return digest.hexdigest()


def checkpoint(seed, arm):
    if arm not in TRAINED_ARMS:
        raise ValueError("random is reproducibly initialized, not a trained checkpoint")
    base = ROOT / "data/finish-position"
    sub = "interior-v1" if arm == "interior" else "train-v1"
    return base / sub / f"seed_{seed:03d}" / arm / "B008.pt"


def actors(split):
    start = 91000 + 1000 * SPLITS.index(split)
    return tuple(range(start, start + BATCH))


def world_seed(seed, split):
    return 740000 + int(seed) + 100 * SPLITS.index(split)


def folder(seed, split, mode="teacher"):
    if split not in SPLITS or mode not in ("teacher", "own"):
        raise ValueError((split, mode))
    if mode == "own" and split != "test":
        raise ValueError("Own-action streams are held-out test only")
    return OUT / mode / split / f"seed_{seed:03d}"


def sample_path(seed, split, arm, mode="teacher"):
    return folder(seed, split, mode) / f"{arm}.npz"


def facts_path(seed, split, mode="teacher", arm=None):
    if mode == "own" and arm not in TRAINED_ARMS:
        raise ValueError("Own-action streams require a trained arm")
    return folder(seed, split, mode) / ("facts.npz" if mode == "teacher" else f"{arm}_facts.npz")


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_new(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_facts(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def load_model(seed, arm):
    if arm == "random":
        with torch.random.fork_rng():
            torch.manual_seed(870000 + int(seed))
            model = GRUAgent(width=192)
    else:
        state = torch.load(checkpoint(seed, arm), weights_only=False, map_location="cpu")
        model = GRUAgent(**state["model_config"])
        model.load_state_dict(state["model"])
    assert model.config() == dict(width=192, input_size=56, n_jobs=2, n_actions=11)
    return model.eval().requires_grad_(False)


def source_files():
    return [Path(__file__), Path(__file__).with_name("data_test.py"), ROOT / "experiments/learn_remember/gru_train.py",
            ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py",
            ROOT / "experiments/lifelong/evaluate.py",
            ROOT / "experiments/experience_transfer/train_transfer.py",
            ROOT / "experiments/finish_position/boundary_readout.py",
            ROOT / "experiments/finish_position/switch_test.py",
            ROOT / "experiments/finish_position/interior_checks.py"]


def declare():
    torch.set_num_threads(1)
    models = {}
    for seed in range(3):
        for arm in ARMS:
            models[f"{seed}/{arm}"] = dict(state_sha256=state_sha(load_model(seed, arm)))
            if arm == "random":
                models[f"{seed}/{arm}"].update(initialization_seed=870000 + seed)
            else:
                cp = checkpoint(seed, arm)
                models[f"{seed}/{arm}"].update(path=str(cp.relative_to(ROOT)), checkpoint_sha256=sha(cp))
    plan = dict(format="timing-boundary-data-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(range(3)), splits=SPLITS, arms=ARMS, actors={s:actors(s) for s in SPLITS},
        world_seed="740000+seed+100*split_index", warm=WARM, collected_ticks=STEPS, dt=DT,
        n_jobs=2, active_jobs=2, regime=CONDITIONS["long_both"], native_cancel=.2, native_replace=.3,
        mapping={str(seed):mapping_table(seed, "related") for seed in range(3)}, models=models,
        source_files={str(p.relative_to(ROOT)):sha(p) for p in source_files()},
        teacher="One native midpoint-executed stream per seed/split. All frozen arms including random consume exactly those public observations from initial h=0 at stream origin, including warmup. No resets, gradients, or private inputs.",
        own="Only test, each trained arm executes native eleven-way argmax every tick including warmup, continuously. Same actors/world seeds/regime/exogenous prefix as teacher. Own histories do not train probes.",
        cohort="All NEW and REPLACE commands archived. Include eligible requests issued at tick>=512 whose native maturity pulse appears by final observed tick and which have active+started last pre-ready and first legal ticks consecutive for the same lane/job/rid. Shared and conflicting raw symbols included; full denominators and exclusions retained.",
        offsets=OFFSETS, incomplete_offsets="Keep each available offset if the same request is active and started. The two boundary rows -1/0 are mandatory; missing other offsets never discard the pair.",
        hidden="post-observation h, before action, used directly by unchanged linear policy head",
        pair_id="world_seed*10**10 + actor_id*10**5 + job*10**4 + rid; int64; unique for this fixed protocol",
        labels="family A=0/B=1; target=3+4*version+mapped_payload; legal includes active, started, now>=ready-EPS and now<=deadline+EPS; remaining_ticks=(ready-now)/0.25. Labels never enter GRU.",
        success="Teacher samples are off-policy diagnostics and never autonomous successes. Own ordinary/strict counts use matured eligible postwarm requests; strict means successful native completion with no rejected active action before that completion.")
    write_new(OUT / "protocol.json", plan)
    print(json.dumps(dict(event="declared", path=str(OUT / "protocol.json"))), flush=True)


def checked():
    plan = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for path, digest in plan["source_files"].items():
        assert sha(ROOT / path) == digest, f"Frozen data source changed: {path}"
    for row in plan["models"].values():
        if "path" in row:
            assert sha(ROOT / row["path"]) == row["checkpoint_sha256"]
    return plan


def request_key(seed, split, actor, job, rid):
    assert 0 <= rid < 10000 and 0 <= job < 2 and 0 <= actor < 100000
    return world_seed(seed, split) * 10**10 + int(actor) * 10**5 + int(job) * 10**4 + int(rid)


def request_table(facts, seed, split, warm=WARM):
    """Full request denominator plus independently selected boundary/offset rows."""
    spec = SEEDS[int(seed)]
    maturation = {}
    for t, lane, job in zip(*np.where(facts["mature"])):
        key = (int(lane), int(job), int(facts["mature_id"][t, lane, job]))
        assert key not in maturation
        maturation[key] = int(t)
    rows, requests = [], []
    exclusions = Counter()
    by_family = {"A":Counter(), "B":Counter()}
    for issued, lane, job in zip(*np.where(np.isin(facts["event_kind"], (1, 2)))):
        issued, lane, job = map(int, (issued, lane, job))
        rid = int(facts["event_id"][issued, lane, job])
        assert rid == facts["rid"][issued, lane, job]
        cue = int(facts["family"][issued, lane, job])
        assert cue in (spec["history_cue"], spec["B_cue"])
        family = 0 if cue == spec["history_cue"] else 1
        label = "AB"[family]
        raw = int(facts["raw"][issued, lane, job])
        target = 3 + 4 * int(facts["version"][issued, lane, job]) + spec[label][raw]
        assert target == 3 + 4 * facts["version"][issued,lane,job] + facts["payload"][issued,lane,job]
        mature_tick = maturation.get((lane, job, rid))
        item = dict(pair_id=request_key(seed, split, int(facts["actor_ids"][lane]), job, rid),
            lane=lane, actor=int(facts["actor_ids"][lane]), job=job, rid=rid,
            issued=issued, issued_time=float(facts["issued_time"][issued,lane,job]),
            kind=int(facts["event_kind"][issued,lane,job]), family=family, cue=cue,
            raw=raw, conflict=raw in spec["conflict_raw"], target=int(target),
            version=int(facts["version"][issued,lane,job]), duration=float(facts["duration"][issued,lane,job]),
            deadline=float(facts["deadline"][issued,lane,job]),
            eligible=bool(facts["event_eligible"][issued,lane,job]),
            mature=mature_tick is not None, mature_tick=mature_tick,
            ordinary=bool(facts["mature_done"][mature_tick,lane,job]) if mature_tick is not None else False,
            strict=bool(facts["mature_strict"][mature_tick,lane,job]) if mature_tick is not None else False,
            included=False, offsets=[], missing_offsets=list(OFFSETS))
        c = by_family[label]; c["commands"] += 1
        if item["eligible"] and issued >= warm:
            c["eligible_issued_afterwarm"] += 1
        if not item["eligible"]:
            reason = "ineligible"
        elif issued < warm:
            reason = "issued_beforewarm"
        elif mature_tick is None:
            reason = "not_matured_at_horizon"
        else:
            c["matured_eligible_afterwarm"] += 1
            c["ordinary"] += item["ordinary"]; c["strict"] += item["strict"]
            same = facts["rid"][:,lane,job] == rid
            active_started = same & facts["active"][:,lane,job] & facts["started"][:,lane,job]
            remaining = (facts["ready"][:,lane,job] - facts["now"]) / DT
            processing = np.flatnonzero(active_started & (remaining > EPS / DT))
            legal = np.flatnonzero(active_started & (remaining <= EPS / DT) &
                                   (facts["now"] <= facts["deadline"][:,lane,job] + EPS))
            if not len(processing):
                reason = "no_active_started_preready"
            elif not len(legal):
                reason = "no_active_started_legal"
            elif processing[-1] + 1 != legal[0]:
                reason = "nonconsecutive_boundary"
            else:
                reason = None
                first = int(legal[0]); item.update(included=True, last_preready=int(processing[-1]), first_legal=first)
                for offset in OFFSETS:
                    tick = first + offset
                    if 0 <= tick < len(facts["now"]) and active_started[tick]:
                        rows.append(dict(pair_id=item["pair_id"], offset=offset, job=job,
                            actor=item["actor"], lane=lane, rid=rid, tick=tick,
                            family=family, raw=raw, conflict=item["conflict"], target=int(target),
                            duration=item["duration"], remaining_ticks=float(remaining[tick]),
                            legal=bool(remaining[tick] <= EPS/DT and facts["now"][tick] <= facts["deadline"][tick,lane,job]+EPS)))
                        item["offsets"].append(offset)
                assert -1 in item["offsets"] and 0 in item["offsets"]
                item["missing_offsets"] = [x for x in OFFSETS if x not in item["offsets"]]
                c["included_pairs"] += 1; c["sample_rows"] += len(item["offsets"])
                c["incomplete_offset_pairs"] += bool(item["missing_offsets"])
        item["exclusion"] = reason
        if reason:
            exclusions[reason] += 1; c["excluded_" + reason] += 1
        requests.append(item)
    assert sum(item["mature"] for item in requests) == len(maturation)
    assert len({item["pair_id"] for item in requests}) == len(requests)
    totals = Counter()
    for c in by_family.values(): totals.update(c)
    return rows, dict(counts=dict(totals), by_family={k:dict(v) for k,v in by_family.items()},
                      exclusions=dict(exclusions), requests=requests)


def verify_native_facts(facts, requests):
    """Check native receipts, eligibility/maturity and clean completions from facts."""
    actions = facts["actions"]
    begin = (actions == 1) | (actions == 2)
    finish = (actions >= 3) & (actions < 11)
    version = np.where(begin, actions-1, (actions-3)//4)
    accepted_begin = begin & facts["active"] & ~facts["started"] & (version == facts["version"])
    accepted_finish = finish & facts["active"] & facts["started"] & (version == facts["version"]) & (
        (actions-3)%4 == facts["payload"]) & (facts["now"][:,None,None] >= facts["ready"]-EPS) & (
        facts["now"][:,None,None] <= facts["deadline"]+EPS)
    expected = np.where(actions == 0, 0, np.where(accepted_begin | accepted_finish, 1, -1))
    np.testing.assert_array_equal(expected, facts["receipts"])
    obs = facts["obs"].reshape(len(actions), actions.shape[1], 2, 28)
    previous = np.eye(11, dtype=np.float32)[actions[:-1]] * (actions[:-1] != 0)[...,None]
    np.testing.assert_array_equal(obs[1:,...,11:22], previous)
    np.testing.assert_array_equal(obs[1:,...,22], (expected[:-1] == 1))
    np.testing.assert_array_equal(obs[1:,...,23], (expected[:-1] == -1))
    assert not obs[0,...,11:24].any()
    for row in requests:
        lane, job, rid, issued = (row[k] for k in ("lane", "job", "rid", "issued"))
        same = facts["rid"][:,lane,job] == rid
        times = np.flatnonzero(same)
        # A NEW is eligible iff its external cycle has no later CANCEL/REPLACE;
        # a REPLACE is always eligible.  For matured NEWs the full modification
        # opportunity necessarily precedes their deadline and is observed.
        if row["mature"]:
            assert row["eligible"]
            if row["kind"] == 1:
                modify = np.isin(facts["event_kind"][issued+1:row["mature_tick"]+1,lane,job], (2,3))
                assert not modify.any()
            mt = row["mature_tick"]
            assert facts["now"][mt] > row["deadline"] + EPS
            assert mt == 0 or facts["now"][mt-1] <= row["deadline"] + EPS
            done_ticks = times[accepted_finish[times,lane,job]]
            done = bool(len(done_ticks))
            clean = done and not np.any((expected[issued:done_ticks[0]+1,lane,job] == -1) &
                                         facts["active"][issued:done_ticks[0]+1,lane,job])
            assert row["ordinary"] == done and row["strict"] == clean
        if row["eligible"]:
            mature_expected = facts["now"][-1] > row["deadline"] + EPS
            assert row["mature"] == mature_expected
    return dict(receipts="exact", previous_public_receipts="exact", matured_eligibility="exact",
                ordinary_and_strict="exact", checked_ticks=int(len(actions)))


@torch.inference_mode()
def collect(seed, split, mode="teacher", arm=None):
    checked(); torch.set_num_threads(1)
    path = facts_path(seed, split, mode, arm)
    if path.exists(): raise FileExistsError(path)
    spec = SEEDS[int(seed)]
    model = load_model(seed, arm) if mode == "own" else None
    hidden = model.initial_state(BATCH) if model is not None else None
    saved = {k:[] for k in (*FACT_FIELDS, "obs", "actions", "receipts", "teacher", "now", "tick")}
    with mapping_scope(mapping_table(seed, "related")):
        world = ContextWorld(seed=world_seed(seed,split), batch_size=BATCH, n_jobs=2, active_jobs=2,
            actor_ids=actors(split), family_allowed=(spec["history_cue"],spec["B_cue"]),
            dt=DT, **CONDITIONS["long_both"])
        case_ids = world.case_ids
        for _ in range(WARM + STEPS):
            obs = world.observe(); teacher = world.oracle("midpoint")
            if model is None:
                action = teacher
            else:
                logits, hidden = model.step(obs, hidden, DT)
                action = logits.argmax(-1)
            for key, attribute in FACT_FIELDS.items():
                saved[key].append(getattr(world, attribute).numpy().copy())
            saved["obs"].append(obs.numpy().copy()); saved["actions"].append(action.numpy().copy())
            saved["teacher"].append(teacher.numpy().copy())
            saved["now"].append(world.now); saved["tick"].append(world.tick)
            world.act(action)
            saved["receipts"].append(world.last_result.numpy().copy())
        metrics = world.metrics()
    facts = {k:np.stack(v) for k,v in saved.items()}
    facts["actor_ids"] = np.asarray(actors(split), dtype=np.int32)
    rows, report = request_table(facts, seed, split)
    audit = verify_native_facts(facts, report["requests"])
    after = slice(WARM, None)
    idle = ~facts["active"][after]
    report.update(seed=seed, split=split, mode=mode, arm=arm, world_seed=world_seed(seed,split),
        actor_ids=list(actors(split)), case_ids=case_ids, events_sha256=world.event_hash,
        world_config=dict(seed=world_seed(seed,split),batch_size=BATCH,n_jobs=2,active_jobs=2,
            actor_ids=list(actors(split)),family_allowed=[spec["history_cue"],spec["B_cue"]],
            dt=DT,**CONDITIONS["long_both"]),
        protocol_sha256=sha(OUT / "protocol.json"), native_metrics_all_ticks=metrics,
        idle_opportunities=int(idle.sum()), idle_actions=int((idle & (facts["actions"][after] != 0)).sum()),
        facts_audit=audit, off_policy=mode == "teacher")
    if mode == "own":
        reference = load_facts(facts_path(seed, split))
        for name in ("event_kind", "event_id", "event_raw", "event_family", "event_version", "event_duration", "event_window", "event_eligible"):
            np.testing.assert_array_equal(facts[name], reference[name])
        teacher_report = json.loads(facts_path(seed,split).with_suffix(".json").read_text(encoding="utf-8"))
        assert report["events_sha256"] == teacher_report["events_sha256"]
        report["exogenous_teacher_match"] = True
    save_new(path, **facts)
    report["facts_sha256"] = sha(path)
    write_new(path.with_suffix(".json"), report)
    print(json.dumps(dict(event="collected", seed=seed, split=split, mode=mode, arm=arm,
                          counts=report["counts"], idle_actions=report["idle_actions"])), flush=True)
    return rows, report


ROW_DTYPES = dict(pair_id=np.int64, offset=np.int16, job=np.int8, actor=np.int32,
    rid=np.int64, tick=np.int32, family=np.int8, raw=np.int8, conflict=np.bool_,
    target=np.int8, duration=np.float64, remaining_ticks=np.float64, legal=np.bool_)


@torch.inference_mode()
def score(seed, split, arm, mode="teacher"):
    plan = checked(); torch.set_num_threads(1)
    path = sample_path(seed,split,arm,mode)
    if path.exists(): raise FileExistsError(path)
    fp = facts_path(seed,split,mode,arm)
    facts = load_facts(fp)
    rows, report = request_table(facts,seed,split)
    model = load_model(seed,arm)
    model_hash = state_sha(model)
    assert model_hash == plan["models"][f"{seed}/{arm}"]["state_sha256"]
    at_tick = {}
    for index,row in enumerate(rows): at_tick.setdefault(row["tick"], []).append(index)
    values = {key:np.asarray([row[key] for row in rows],dtype=dtype) for key,dtype in ROW_DTYPES.items()}
    values["h"] = np.empty((len(rows),192),dtype=np.float32)
    values["obs"] = np.empty((len(rows),56),dtype=np.float32)
    values["logits"] = np.empty((len(rows),2,11),dtype=np.float32)
    h = model.initial_state(len(facts["actor_ids"]))
    for tick,x in enumerate(torch.from_numpy(facts["obs"])):
        logits,h = model.step(x,h,DT)
        if mode == "own":
            np.testing.assert_array_equal(logits.argmax(-1).numpy(), facts["actions"][tick])
        indices = at_tick.get(tick, [])
        if indices:
            lanes = [rows[i]["lane"] for i in indices]
            values["h"][indices] = h[lanes].numpy()
            values["obs"][indices] = x[lanes].numpy()
            values["logits"][indices] = logits[lanes].numpy()
    assert state_sha(model) == model_hash
    save_new(path, **values)
    metadata = dict(seed=seed,split=split,arm=arm,mode=mode,rows=len(rows),
        pairs=sum(r["included"] for r in report["requests"]),counts=report["counts"],
        sample_sha256=sha(path),facts_path=str(fp.relative_to(ROOT)),facts_sha256=sha(fp),
        facts_metadata_sha256=sha(fp.with_suffix(".json")),protocol_sha256=sha(OUT/"protocol.json"),
        model=plan["models"][f"{seed}/{arm}"], hidden_semantics="post-current-observation; continuous from stream origin",
        array_shapes={k:list(v.shape) for k,v in values.items()},off_policy=mode == "teacher")
    write_new(path.with_suffix(".json"),metadata)
    print(json.dumps(dict(event="scored",seed=seed,split=split,arm=arm,mode=mode,rows=len(rows))),flush=True)


def run_teacher(seed, split):
    collect(seed, split)
    for arm in ARMS: score(seed, split, arm)


def run_own(seed, arm):
    collect(seed,"test","own",arm)
    score(seed,"test",arm,"own")


def launch(max_workers=6):
    """Hidden independent single-thread CPU workers; logs stay with outputs."""
    checked()
    if not 1 <= max_workers <= 6: raise ValueError("max_workers must be 1..6")
    logs = OUT / "logs"; logs.mkdir(parents=True,exist_ok=True)
    pending = [("teacher",seed,split) for seed in range(3) for split in SPLITS]
    pending += [("own",seed,arm) for seed in range(3) for arm in TRAINED_ARMS]
    running = []; finished = []
    while pending or running:
        while pending and len(running) < max_workers:
            # Own workers depend only on their matching test teacher being done.
            chosen = next((i for i,x in enumerate(pending) if x[0] == "teacher" or
                facts_path(x[1],"test").with_suffix(".json").exists()), None)
            if chosen is None: break
            kind,seed,label = pending.pop(chosen)
            logfile = logs / f"{kind}_{seed}_{label}.log"
            handle = logfile.open("x",encoding="utf-8")
            cmd = [sys.executable,"-m","experiments.timing_boundary.data",kind,"--seed",str(seed)]
            cmd += ["--split",label] if kind == "teacher" else ["--arm",label]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=flags)
            running.append((process,handle,logfile))
        keep = []
        for process,handle,path in running:
            code = process.poll()
            if code is None:
                keep.append((process,handle,path))
            else:
                handle.close(); finished.append(dict(log=str(path),exit_code=code))
                print(json.dumps(finished[-1]),flush=True)
                if code:
                    raise RuntimeError(f"Worker failed; inspect {path}")
        running = keep
        if pending or running: time.sleep(.25)
    write_new(OUT/"workers.json",dict(workers=finished))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","teacher","own","score","launch"))
    parser.add_argument("--seed",type=int,choices=range(3)); parser.add_argument("--split",choices=SPLITS)
    parser.add_argument("--arm",choices=ARMS); parser.add_argument("--mode",choices=("teacher","own"),default="teacher")
    parser.add_argument("--workers",type=int,default=6)
    args = parser.parse_args()
    if args.command == "declare": declare()
    elif args.command == "teacher": run_teacher(args.seed,args.split)
    elif args.command == "own": run_own(args.seed,args.arm)
    elif args.command == "score": score(args.seed,args.split,args.arm,args.mode)
    else: launch(args.workers)
