"""Native free-action evaluation of frozen split-policy checkpoints.

All post-warmup eligible matured requests enter the denominator, including
those that never start or never reach a legal completion window.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from experiments.action_split.model import SplitGRUAgent
from experiments.action_split.data import sha, read, write_new
from experiments.learn_remember.gru_train import GRUAgent
from experiments.experience_transfer.train_transfer import SEEDS, mapping_scope, mapping_table
from experiments.lifelong.world import ContextWorld
from experiments.timing_boundary.data import FACT_FIELDS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/action-split/eval-v1"
WARM, STEPS, DT = 512, 4096, .25
POLICIES = ("original", "frozen064", "full064", "frozen256", "full256")
CONDITIONS = {"long": (8., 16.), "longer": (16., 32.)}
EPS = 1e-6


def checkpoint(seed, policy):
    if policy == "original":
        return ROOT / f"data/finish-position/train-v1/seed_{seed:03d}/midpoint/B008.pt"
    from experiments.action_split.train import checkpoint_path
    return checkpoint_path(seed, policy[:-3], int(policy[-3:]))


def config(seed, condition):
    spec = SEEDS[seed]
    return dict(seed=851000+seed, batch_size=32, n_jobs=2, active_jobs=2,
        actor_ids=tuple(range(96000, 96032)), family_allowed=(spec["history_cue"], spec["B_cue"]),
        dt=DT, duration_range=CONDITIONS[condition], gap_range=(16.,32.))


def stem(seed, policy, condition):
    return OUT / f"seed_{seed:03d}" / f"{policy}_{condition}"


def declare():
    """Fix evaluation without consuming any new-model outcome."""
    sources = [Path(__file__), Path(__file__).with_name("eval_test.py"),
        ROOT/"experiments/action_split/model.py", ROOT/"experiments/lifelong/world.py",
        ROOT/"experiments/continuity/world.py", ROOT/"experiments/learn_remember/gru_train.py"]
    write_new(OUT/"protocol.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources}, seeds=list(range(3)),
        policies=list(POLICIES), conditions=CONDITIONS, warm=WARM, test=STEPS, dt=DT,
        actor_ids=list(range(96000,96032)), world_seed="851000+seed",
        primary="step256, mixed A/B long(8..16) strict and ordinary completion, A/B conflict subsets, idle errors per actual idle opportunity",
        secondary="step064 progress and longer(16..32) duration only; same gap16..32. No selection from outcomes.",
        execution="Every policy executes native actions throughout warmup+test. Independent origin h=0, then fully continuous. No oracle action or private state enters decoder.",
        denominator="All eligible actual NEW/REPLACE issued at tick>=512 and with a matching native maturity pulse before horizon. Include no-BEGIN/no-window failures. No boundary-pair filter.",
        diagnostics="Actual accepted BEGIN; actual early FINISH independent of version/content; first rejected active action; reached legal but no FINISH; legal first attempt correct/wrong/absent. Whole legal window, not mandatory immediate FINISH.",
        reference="Untouched original midpoint B008 had its old head and no new budget; new heads start from zero. Frozen/full samearchitecture/data/updatebudget is the primary comparison."))


def freeze():
    from experiments.action_split import train
    training_freeze = train.OUT / "freeze.json"
    assert training_freeze.exists(), "all training checkpoints must be frozen first"
    protocol = read(OUT/"protocol.json")
    for p, digest in protocol["sources"].items():
        assert sha(ROOT/p) == digest, p
    files = {str(checkpoint(s,p).relative_to(ROOT)):sha(checkpoint(s,p)) for s in range(3) for p in POLICIES}
    write_new(OUT/"freeze.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/"protocol.json"), training_freeze_sha256=sha(training_freeze), checkpoints=files))


def checked(seed, policy):
    protocol, frozen = read(OUT/"protocol.json"), read(OUT/"freeze.json")
    assert sha(OUT/"protocol.json") == frozen["protocol_sha256"]
    for p,digest in protocol["sources"].items():
        assert sha(ROOT/p) == digest, p
    path = checkpoint(seed,policy)
    assert sha(path) == frozen["checkpoints"][str(path.relative_to(ROOT))]
    return path


def accepted_actions(facts):
    a = facts["actions"]
    begin = (a == 1) | (a == 2)
    finish = (a >= 3) & (a <= 10)
    version = np.where(begin, a-1, (a-3)//4)
    start = begin & facts["active"] & ~facts["started"] & (version == facts["version"])
    end = finish & facts["active"] & facts["started"] & (version == facts["version"]) & (
        (a-3)%4 == facts["payload"]) & (facts["now"][:,None,None] >= facts["ready"]-EPS) & (
        facts["now"][:,None,None] <= facts["deadline"]+EPS)
    receipts = np.where(a == 0, 0, np.where(start | end, 1, -1))
    np.testing.assert_array_equal(receipts, facts["receipts"])
    return start, end


def summarize_facts(facts, seed, warm=WARM):
    start, end = accepted_actions(facts)
    mature = {}
    for t,lane,job in zip(*np.where(facts["mature"])):
        key = (int(lane), int(job), int(facts["mature_id"][t,lane,job]))
        assert key not in mature
        mature[key] = int(t)
    requests = []
    coverage = Counter()
    for issued,lane,job in zip(*np.where(np.isin(facts["event_kind"], (1,2)))):
        issued,lane,job = int(issued),int(lane),int(job)
        rid = int(facts["event_id"][issued,lane,job])
        eligible = bool(facts["event_eligible"][issued,lane,job])
        when = mature.get((lane,job,rid))
        if issued < warm:
            coverage["prewarm_commands"] += 1
            continue
        coverage["postwarm_commands"] += 1
        if not eligible:
            coverage["ineligible"] += 1
            continue
        coverage["eligible_issued"] += 1
        if when is None:
            coverage["eligible_censored"] += 1
            continue
        coverage["matured"] += 1
        ticks = np.flatnonzero(facts["rid"][:,lane,job] == rid)
        active_ticks = ticks[facts["active"][ticks,lane,job]]
        processing = active_ticks[facts["started"][active_ticks,lane,job] & (
            facts["now"][active_ticks] < facts["ready"][active_ticks,lane,job]-EPS)]
        legal = active_ticks[facts["started"][active_ticks,lane,job] & (
            facts["now"][active_ticks] >= facts["ready"][active_ticks,lane,job]-EPS) & (
            facts["now"][active_ticks] <= facts["deadline"][active_ticks,lane,job]+EPS)]
        rejected = active_ticks[facts["receipts"][active_ticks,lane,job] == -1]
        accepted = ticks[end[ticks,lane,job]]
        ordinary = len(accepted) > 0
        strict = ordinary and not np.any(rejected < accepted[0])
        assert ordinary == bool(facts["mature_done"][when,lane,job])
        assert strict == bool(facts["mature_strict"][when,lane,job])
        starts = ticks[start[ticks,lane,job]]
        attempts = legal[facts["actions"][legal,lane,job] >= 3]
        premature = processing[facts["actions"][processing,lane,job] >= 3]
        first_error = int(rejected[0]) if len(rejected) else None
        first_early = first_error in set(premature.tolist())
        family = "A" if facts["family"][issued,lane,job] == SEEDS[seed]["history_cue"] else "B"
        raw = int(facts["raw"][issued,lane,job])
        requests.append(dict(actor=int(facts["actor_ids"][lane]),lane=lane,job=job,rid=rid,
            issued=issued,mature_tick=when,family=family,raw=raw,conflict=raw in SEEDS[seed]["conflict_raw"],
            ordinary=ordinary,strict=strict,started=bool(len(starts)),legal_window=bool(len(legal)),
            no_begin=not len(starts),started_no_window=bool(len(starts) and not len(legal)),
            legal_no_finish=bool(len(legal) and not len(attempts)),
            any_early=bool(len(premature)),first_error_early=first_early,
            first_legal_attempt=("correct" if facts["receipts"][attempts[0],lane,job] == 1 else "wrong") if len(attempts) else "none",
            first_legal_delay=float(facts["now"][attempts[0]]-facts["ready"][attempts[0],lane,job]) if len(attempts) else None))
    groups = {}
    masks = {"all":lambda r:True}
    for fam in ("A","B"):
        masks["family_"+fam] = lambda r,fam=fam:r["family"] == fam
        masks["family_"+fam+"_conflict"] = lambda r,fam=fam:r["family"] == fam and r["conflict"]
    masks["shared"] = lambda r:not r["conflict"]
    flags = ("ordinary","strict","started","legal_window","no_begin","started_no_window","legal_no_finish","any_early","first_error_early")
    for name, fn in masks.items():
        selected = [r for r in requests if fn(r)]
        group = dict(matured=len(selected), **{k:sum(bool(r[k]) for r in selected) for k in flags})
        for answer in ("correct","wrong","none"):
            group["first_legal_"+answer] = sum(r["first_legal_attempt"] == answer for r in selected)
        group["first_legal_delay_sum"] = sum(r["first_legal_delay"] for r in selected if r["first_legal_delay"] is not None)
        group["first_legal_delay_n"] = sum(r["first_legal_delay"] is not None for r in selected)
        assert group["no_begin"] + group["started_no_window"] + group["legal_window"] == len(selected)
        assert group["legal_no_finish"]+group["first_legal_correct"]+group["first_legal_wrong"] == group["legal_window"]
        groups[name] = group
    idle = ~facts["active"][warm:]
    return dict(coverage=dict(coverage), groups=groups, requests=requests,
        idle_opportunities=int(idle.sum()),idle_actions=int((idle & (facts["actions"][warm:] != 0)).sum()),
        native_receipts_reconstructed=True, mature_completions_reconstructed=True)


@torch.inference_mode()
def run(seed, policy, condition):
    torch.set_num_threads(1)
    cp_path = checked(seed,policy)
    saved = torch.load(cp_path,map_location="cpu",weights_only=False)
    cls = GRUAgent if policy == "original" else SplitGRUAgent
    model = cls(**saved["model_config"])
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    h = model.initial_state(32)
    streams = {k:[] for k in (*FACT_FIELDS,"obs","actions","receipts","now")}
    with mapping_scope(mapping_table(seed,"related")):
        world = ContextWorld(**config(seed,condition))
        for _ in range(WARM+STEPS):
            obs = world.observe()
            scores,h = model.step(obs,h,DT)
            actions = scores.argmax(-1) if policy == "original" else model.choose(scores)
            streams["obs"].append(obs.numpy().copy())
            streams["actions"].append(actions.numpy().copy())
            streams["now"].append(world.now)
            for key, attr in FACT_FIELDS.items():
                streams[key].append(getattr(world,attr).numpy().copy())
            world.act(actions)
            streams["receipts"].append(world.last_result.numpy().copy())
        external_hash = world.event_hash
    facts = {k:np.stack(v) for k,v in streams.items()}
    facts["actor_ids"] = np.asarray(config(seed,condition)["actor_ids"])
    facts["final_h"] = h.numpy().copy()
    result = summarize_facts(facts,seed)
    path = stem(seed,policy,condition)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(".npz").open("xb") as f:
        np.savez_compressed(f,**facts)
    result.update(seed=seed,policy=policy,condition=condition,world_config=config(seed,condition),
        checkpoint_path=str(cp_path.relative_to(ROOT)),checkpoint_sha256=sha(cp_path),
        protocol_sha256=sha(OUT/"protocol.json"),freeze_sha256=sha(OUT/"freeze.json"),
        facts_sha256=sha(path.with_suffix(".npz")),external_hash=external_hash)
    write_new(path.with_suffix(".json"),result)
    print(json.dumps(dict(seed=seed,policy=policy,condition=condition,all=result["groups"]["all"])),flush=True)


def launch(workers=6):
    logs = OUT/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    def job(key):
        s,p,c = key
        logfile = logs/f"{s}_{p}_{c}.log"
        with logfile.open("x",encoding="utf8") as stream:
            cmd = [sys.executable,"-m","experiments.action_split.evaluate","run","--seed",str(s),"--policy",p,"--condition",c]
            done = subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return dict(seed=s,policy=p,condition=c,exit_code=done.returncode,log=str(logfile.relative_to(ROOT)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(job,[(s,p,c) for s in range(3) for p in POLICIES for c in CONDITIONS]))
    write_new(OUT/"workers.json",dict(results=results))
    assert all(r["exit_code"] == 0 for r in results), results
    hashes = {}
    for s in range(3):
        for c in CONDITIONS:
            values = [read(stem(s,p,c).with_suffix(".json")) for p in POLICIES]
            assert len({v["external_hash"] for v in values}) == 1
            identities = [{(r["actor"],r["job"],r["rid"]) for r in v["requests"]} for v in values]
            assert all(x == identities[0] for x in identities)
            for p in POLICIES:
                path = stem(s,p,c).with_suffix(".json")
                hashes[str(path.relative_to(ROOT))] = sha(path)
    write_new(OUT/"completion.json",dict(complete=True,runs=len(results),identical_external_requests=True,files=hashes))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("declare","freeze","run","all"))
    parser.add_argument("--seed",type=int,choices=range(3))
    parser.add_argument("--policy",choices=POLICIES)
    parser.add_argument("--condition",choices=CONDITIONS)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    if args.command == "declare":declare()
    elif args.command == "freeze":freeze()
    elif args.command == "run":run(args.seed,args.policy,args.condition)
    else:launch(args.workers)
