"""Posthoc read-only first-legal-FINISH signatures in existing saved rollouts.

This analysis was requested after the primary outcomes were known. It does not
instantiate or execute a policy, native world, optimizer, teacher or time head.
All matured test-issued requests remain denominators, including timing misses.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments.action_split.data import read, sha, write_new
from experiments.experience_transfer.train_transfer import SEEDS, mapping_table

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/coverage-retention/error-signature-v1"
PLAN=OUT/"protocol.json"
EPS=1e-6
CATEGORIES=("correct_current_rule","other_rule_substitution","correct_version_other_payload","wrong_version","no_attempt")
GROUPS=("A_conflict","B_conflict","shared")
POLICIES={"fresh":("baseline","control256","withdraw256"),"live":("control","withdraw")}
RAW_KEYS=("actions","receipts","rid","active","started","ready","deadline","now","version","payload",
    "raw","family","eligible","issued_time","duration","event_kind","event_id","event_eligible","event_family",
    "event_raw","event_version","event_duration","event_window","mature","mature_id",
    "mature_issued","mature_duration","mature_family","mature_done","mature_strict","actor_ids")


def run_specs():
    return [dict(mode=mode,seed=seed,policy=policy,
        stem=f"data/coverage-retention/{'eval-v1' if mode=='fresh' else 'live-v1'}/seed_{seed:03d}/{policy}_long",
        output=f"data/coverage-retention/error-signature-v1/{mode}/seed_{seed:03d}/{policy}_long.json")
        for mode,policies in POLICIES.items() for seed in range(3) for policy in policies]


def category(action,version,current,other,conflict):
    if action is None:return "no_attempt"
    assert 3<=action<=10
    if (action-3)//4!=version:return "wrong_version"
    payload=(action-3)%4
    if payload==current:return "correct_current_rule"
    if conflict and payload==other:return "other_rule_substitution"
    return "correct_version_other_payload"


def self_test():
    assert category(None,0,0,1,True)=="no_attempt"
    assert category(3,0,0,1,True)=="correct_current_rule"
    assert category(4,0,0,1,True)=="other_rule_substitution"
    assert category(5,0,0,1,True)=="correct_version_other_payload"
    assert category(8,0,0,1,True)=="wrong_version"
    assert category(3,0,0,0,False)=="correct_current_rule"
    assert category(4,0,0,0,False)=="correct_version_other_payload"


def declare():
    self_test()
    inputs={};runs=run_specs();known={}
    for mode in POLICIES:
        folder=ROOT/f"data/coverage-retention/{'eval-v1' if mode=='fresh' else 'live-v1'}"
        audit=ROOT/f"data/coverage-retention/{'eval-audit-v1' if mode=='fresh' else 'live-audit-v1'}"
        for path in [*(folder/name for name in ("protocol.json","freeze.json","completion.json")),
                     *(audit/name for name in ("audit_plan.json","metrics.json","replay.json"))]:
            inputs[str(path.relative_to(ROOT))]=sha(path)
        assert read(folder/"completion.json")["complete"]
        assert read(audit/"metrics.json")["passed"] and read(audit/"replay.json")["passed"]
        known[mode]={}
        for policy in POLICIES[mode]:
            known[mode][policy]={}
            for family in ("A","B"):
                values=[]
                for run in runs:
                    if run["mode"]==mode and run["policy"]==policy:
                        g=read((ROOT/run["stem"]).with_suffix(".json"))["groups"][f"family_{family}_conflict"]
                        values.append(g["strict"]/g["matured"])
                known[mode][policy][family+"_conflict_strict_equal_seed_mean"]=float(np.mean(values))
    for run in runs:
        stem=ROOT/run["stem"]
        report=read(stem.with_suffix(".json"))
        assert report["facts_sha256"]==sha(stem.with_suffix(".npz"))
        cp=ROOT/report["checkpoint_path"]
        assert report["checkpoint_sha256"]==sha(cp)
        for path in (stem.with_suffix(".json"),stem.with_suffix(".npz"),cp):
            inputs[str(path.relative_to(ROOT))]=sha(path)
        for key in ("source_native_world_path","initial_state_path"):
            if key in report:
                path=ROOT/report[key]
                assert sha(path)==report[key.removesuffix("_path")+"_sha256"]
                inputs[str(path.relative_to(ROOT))]=sha(path)
        run["checkpoint_path"]=report["checkpoint_path"]
        run["checkpoint_sha256"]=report["checkpoint_sha256"]
        run["facts_sha256"]=report["facts_sha256"]
    sources=(Path(__file__),ROOT/"experiments/experience_transfer/train_transfer.py")
    plan=dict(format="coverage-retention-posthoc-error-signature-plan-v1",created_utc=datetime.now(timezone.utc).isoformat(),
        posthoc=True,primary_outcomes_known_before_declaration=known,runs=runs,expected_runs=15,
        inputs=inputs,sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        policies=POLICIES,conditions=["long"],warm=512,dt=.25,native_epsilon=EPS,
        groups=list(GROUPS),categories=list(CATEGORIES),
        denominator="Every native eligible actual NEW/REPLACE issued at local tick>=512 with matching native maturity before horizon. Exactly the existing request-report ledger. No selection on success, failure, BEGIN, legal-window existence or action choice.",
        legal_window="Current request active and started, native now>=ready-1e-6 and now<=deadline+1e-6. Pick the chronologically first actual native FINISH action3..10 in that window; never replace it by later accepted FINISH or conditional head argmax.",
        classification="Priority: no actual legal FINISH ->no_attempt; wrong decodedversion ->wrong_version; currentversion/currentmappedpayload ->correct_current_rule; currentversion/other A-or-B mapping(raw) on conflict ->other_rule_substitution; else correct_version_other_payload. Shared raw symbols have other_rule_notapplicable, never count their correct answer as substitution.",
        crosschecks="Full native receipts reconstructed for every archived action. All selected issue/maturity/current family/raw/version/payload facts crosschecked against exact seed A/B mappings and report identities/outcomes. Global native tick derived from native now, checked against saved world_tick when available. Never use globalrid//2 as local index.",
        aggregation="Raw seed counts, pooled counts/rates, and equal-seed means of full-denominator category rates. Retain ordinary/strict and strict-failure cross-tabs: firstlegalcorrect can follow an earlier rejection and is not equivalent to strict success.",
        read_only="Existing saved action/fact/report/checkpoint hashes only. No new model execution, training, native world execution or intervention. Existing inputs are never overwritten.",
        limits="Posthoc descriptive error signature in these finite saved trajectories. Other-rule answers indicate behavioral substitution, not complete erasure or proof that context is the sole cause. Fresh and live cases are distinct cohorts.")
    write_new(PLAN,plan)
    print(json.dumps(dict(event="declared_posthoc",runs=15,protocol_sha256=sha(PLAN))),flush=True)
    return plan


def checked():
    plan=read(PLAN)
    assert plan["posthoc"] and len(plan["runs"])==15
    amendment_path=OUT/"parser_amendment_001.json"
    amendment=read(amendment_path) if amendment_path.exists() else None
    if amendment is not None:
        assert amendment["original_protocol_sha256"]==sha(PLAN)
        assert amendment["categories_cohorts_and_estimands_changed"] is False
        assert set(amendment["sources"])=={str(Path(__file__).relative_to(ROOT))}
        assert amendment["previous_sources"]=={name:plan["sources"][name] for name in amendment["sources"]}
    for table in ("sources","inputs"):
        for name,digest in plan[table].items():
            effective=amendment["sources"].get(name,digest) if amendment is not None and table=="sources" else digest
            assert sha(ROOT/name)==effective,name
    return plan


def summarize(records):
    result={}
    for group in GROUPS:
        rows=[r for r in records if r["group"]==group]
        n=len(rows)
        counts={c:sum(r["category"]==c for r in rows) for c in CATEGORIES}
        assert sum(counts.values())==n
        fields=("ordinary","strict","no_begin","started_no_window","legal_no_finish","any_early","first_error_early")
        flags={k:sum(r[k] for r in rows) for k in fields}
        assert flags["no_begin"]+flags["started_no_window"]+flags["legal_no_finish"]==counts["no_attempt"]
        strict_fail={c:sum(r["category"]==c and not r["strict"] for r in rows) for c in CATEGORIES}
        ordinary_fail={c:sum(r["category"]==c and not r["ordinary"] for r in rows) for c in CATEGORIES}
        result[group]=dict(matured=n,categories=counts,category_rates={k:v/n if n else None for k,v in counts.items()},
            other_rule_applicability="conflict" if group!="shared" else "other_rule_notapplicable",
            **flags,strict_failure_categories=strict_fail,ordinary_failure_categories=ordinary_fail)
    return result


def classify_run(spec):
    seed=spec["seed"];stem=ROOT/spec["stem"];report=read(stem.with_suffix(".json"))
    with np.load(stem.with_suffix(".npz"),allow_pickle=False) as archive:
        facts={k:archive[k] for k in RAW_KEYS}
        world_ticks=np.rint(facts["now"]/.25).astype(np.int64)
        np.testing.assert_array_equal(facts["now"],world_ticks*.25)
        if "world_tick" in archive:np.testing.assert_array_equal(world_ticks,archive["world_tick"])
    f=facts;a=f["actions"];now=f["now"][:,None,None]
    assert np.all((a>=0)&(a<=10))
    begin=(a==1)|(a==2);finish=a>=3
    legal=f["active"]&f["started"]&(now>=f["ready"]-EPS)&(now<=f["deadline"]+EPS)
    accepted_begin=begin&f["active"]&~f["started"]&(a-1==f["version"])
    accepted_finish=finish&legal&((a-3)//4==f["version"])&((a-3)%4==f["payload"])
    receipts=np.where(a==0,0,np.where(accepted_begin|accepted_finish,1,-1))
    np.testing.assert_array_equal(receipts,f["receipts"])
    maturity={}
    for tick,lane,job in zip(*np.nonzero(f["mature"])):
        key=(int(lane),int(job),int(f["mature_id"][tick,lane,job]))
        assert key not in maturity
        maturity[key]=int(tick)
    records=[];identities=[];mapping=np.asarray(mapping_table(seed,"related"),np.int64)
    for issued,lane,job in zip(*np.nonzero((f["event_kind"]==1)|(f["event_kind"]==2))):
        issued,lane,job=int(issued),int(lane),int(job)
        rid=int(f["event_id"][issued,lane,job]);key=lane,job,rid
        if issued<512 or not f["event_eligible"][issued,lane,job] or key not in maturity:continue
        mature_tick=maturity[key];index=issued,lane,job
        actor=int(f["actor_ids"][lane]);cue=int(f["family"][index]);raw=int(f["raw"][index]);version=int(f["version"][index])
        assert rid==2*int(world_ticks[issued])+job
        assert cue in (SEEDS[seed]["history_cue"],SEEDS[seed]["B_cue"])
        family="A" if cue==SEEDS[seed]["history_cue"] else "B"
        other_family="B" if family=="A" else "A"
        target=int(SEEDS[seed][family][raw]);other=int(SEEDS[seed][other_family][raw])
        conflict=raw in SEEDS[seed]["conflict_raw"]
        assert conflict==(target!=other)
        assert mapping[cue,raw]==target==int(f["payload"][index])
        assert version==int(f["event_version"][index]) and raw==int(f["event_raw"][index])
        assert cue==int(f["event_family"][index]) and f["eligible"][index]
        assert int(f["mature_family"][mature_tick,lane,job])==cue
        assert f["mature_duration"][mature_tick,lane,job]==f["duration"][index]
        assert f["mature_issued"][mature_tick,lane,job]==f["now"][issued]==f["issued_time"][index]
        ticks=np.flatnonzero(f["rid"][:,lane,job]==rid)
        for name,value in (("family",cue),("raw",raw),("version",version),("payload",target)):
            assert np.all(f[name][ticks,lane,job]==value),(seed,spec["policy"],rid,name)
        starts=ticks[accepted_begin[ticks,lane,job]]
        windows=ticks[legal[ticks,lane,job]]
        attempts=windows[finish[windows,lane,job]]
        successes=ticks[accepted_finish[ticks,lane,job]]
        rejected=ticks[f["active"][ticks,lane,job]&(f["receipts"][ticks,lane,job]==-1)]
        ordinary=bool(len(successes));strict=ordinary and not np.any(rejected<successes[0])
        assert ordinary==bool(f["mature_done"][mature_tick,lane,job])
        assert strict==bool(f["mature_strict"][mature_tick,lane,job])
        processing=ticks[f["active"][ticks,lane,job]&f["started"][ticks,lane,job]&(f["now"][ticks]<f["ready"][ticks,lane,job]-EPS)]
        early=processing[finish[processing,lane,job]]
        first=int(attempts[0]) if len(attempts) else None
        action=int(a[first,lane,job]) if first is not None else None
        signature=category(action,version,target,other,conflict)
        receipt=int(f["receipts"][first,lane,job]) if first is not None else None
        if first is not None:assert receipt==(1 if signature=="correct_current_rule" else -1)
        reason=None if first is not None else "no_begin" if not len(starts) else "started_no_window" if not len(windows) else "legal_no_finish"
        record=dict(actor=actor,lane=lane,job=job,rid=rid,issued=issued,mature_tick=mature_tick,
            paired_request_id=f"{seed}:{actor}:{job}:{rid}",family=family,cue=cue,raw=raw,version=version,conflict=conflict,
            group=family+"_conflict" if conflict else "shared",target_payload=target,other_rule_payload=other,
            A_target_payload=int(SEEDS[seed]["A"][raw]),B_target_payload=int(SEEDS[seed]["B"][raw]),
            target_native_action=3+4*version+target,other_rule_native_action=3+4*version+other,
            other_rule_applicability="conflict" if conflict else "other_rule_notapplicable",
            issued_world_tick=int(world_ticks[issued]),issued_native_time=float(f["now"][issued]),
            mature_world_tick=int(world_ticks[mature_tick]),mature_native_time=float(f["now"][mature_tick]),
            legal_ticks_count=len(windows),first_legal_opportunity_tick=int(windows[0]) if len(windows) else None,
            first_legal_finish_tick=first,first_legal_finish_world_tick=int(world_ticks[first]) if first is not None else None,
            first_legal_finish_native_time=float(f["now"][first]) if first is not None else None,
            first_legal_finish_action=action,first_legal_finish_version=(action-3)//4 if action is not None else None,
            first_legal_finish_payload=(action-3)%4 if action is not None else None,first_legal_finish_receipt=receipt,
            category=signature,no_attempt_reason=reason,no_begin=not len(starts),
            started_no_window=bool(len(starts) and not len(windows)),legal_no_finish=bool(len(windows) and not len(attempts)),
            ordinary=ordinary,strict=strict,any_early=bool(len(early)),
            first_error_early=bool(len(rejected) and rejected[0] in set(early.tolist())))
        records.append(record);identities.append((actor,lane,job,rid,issued,mature_tick))
    assert identities==[(r["actor"],r["lane"],r["job"],r["rid"],r["issued"],r["mature_tick"]) for r in report["requests"]]
    for actual,prior in zip(records,report["requests"]):
        for key in ("family","raw","conflict","ordinary","strict","no_begin","started_no_window","legal_no_finish","any_early","first_error_early"):
            assert actual[key]==prior[key],(key,actual["rid"])
        assert prior["first_legal_attempt"]==("none" if actual["category"]=="no_attempt" else "correct" if actual["category"]=="correct_current_rule" else "wrong")
    groups=summarize(records)
    for group in ("A_conflict","B_conflict"):
        original=report["groups"]["family_"+group]
        assert groups[group]["matured"]==original["matured"]
    assert groups["shared"]["matured"]==report["groups"]["shared"]["matured"]
    assert groups["shared"]["categories"]["other_rule_substitution"]==0
    return dict(format="coverage-retention-posthoc-error-signature-run-v1",mode=spec["mode"],seed=seed,policy=spec["policy"],
        posthoc=True,condition="long",protocol_sha256=sha(PLAN),parser_amendment_sha256=sha(OUT/"parser_amendment_001.json"),checkpoint_path=spec["checkpoint_path"],
        checkpoint_sha256=spec["checkpoint_sha256"],facts_sha256=spec["facts_sha256"],report_path=str(stem.with_suffix(".json").relative_to(ROOT)),
        report_sha256=sha(stem.with_suffix(".json")),source_bindings={k:v for k,v in report.items() if k.endswith("sha256") or k.endswith("_path")},
        entire_native_receipts_exact=True,request_denominator_exact=True,groups=groups,requests=records)


def aggregate(runs):
    summaries={}
    for mode,policies in POLICIES.items():
        summaries[mode]={}
        for policy in policies:
            selected=[r for r in runs if r["mode"]==mode and r["policy"]==policy]
            assert sorted(r["seed"] for r in selected)==[0,1,2]
            pooled=summarize([request for r in selected for request in r["requests"]])
            for group,values in pooled.items():
                values["equal_seed_category_rates"]={c:float(np.mean([r["groups"][group]["category_rates"][c] for r in selected])) for c in CATEGORIES}
                values["equal_seed_strict_rate"]=float(np.mean([r["groups"][group]["strict"]/r["groups"][group]["matured"] for r in selected]))
                values["equal_seed_ordinary_rate"]=float(np.mean([r["groups"][group]["ordinary"]/r["groups"][group]["matured"] for r in selected]))
            summaries[mode][policy]=dict(seeds={str(r["seed"]):r["groups"] for r in selected},pooled=pooled)
    return summaries


def run():
    plan=checked();self_test();runs=[];outputs={}
    for spec in plan["runs"]:
        report=classify_run(spec);path=ROOT/spec["output"]
        write_new(path,report);outputs[spec["output"]]=sha(path);runs.append(report)
        print(json.dumps(dict(event="classified_saved_actions",mode=spec["mode"],seed=spec["seed"],policy=spec["policy"],groups=report["groups"])),flush=True)
    paired=[]
    for mode,policies in POLICIES.items():
        for seed in range(3):
            candidates=[r for r in runs if r["mode"]==mode and r["seed"]==seed]
            tables={r["policy"]:{x["paired_request_id"]:x for x in r["requests"]} for r in candidates}
            ids=list(tables[policies[0]])
            assert all(list(table)==ids for table in tables.values())
            for request_id in ids:
                anchor=tables[policies[0]][request_id]
                for table in tables.values():
                    for key in ("actor","lane","job","rid","issued","mature_tick","family","raw","version","group","target_payload","other_rule_payload"):
                        assert table[request_id][key]==anchor[key],(mode,seed,request_id,key)
                paired.append(dict(mode=mode,seed=seed,paired_request_id=request_id,group=anchor["group"],family=anchor["family"],raw=anchor["raw"],
                    policies={policy:{k:table[request_id][k] for k in ("category","ordinary","strict","first_legal_finish_tick","first_legal_finish_action","no_attempt_reason")} for policy,table in tables.items()}))
    pairs=OUT/"paired_requests.json";write_new(pairs,dict(protocol_sha256=sha(PLAN),requests=paired))
    summary=OUT/"summary.json"
    write_new(summary,dict(format="coverage-retention-posthoc-error-signature-summary-v1",posthoc=True,complete=True,
        protocol_sha256=sha(PLAN),parser_amendment_sha256=sha(OUT/"parser_amendment_001.json"),created_utc=datetime.now(timezone.utc).isoformat(),runs=15,groups=aggregate(runs),
        run_outputs=outputs,paired_requests_path=str(pairs.relative_to(ROOT)),paired_requests_sha256=sha(pairs),
        limits=plan["limits"],interpretation="All categories use complete matured request denominators. First legal FINISH signatures and strict-success outcomes are separately cross-tabulated."))
    checked()
    write_new(OUT/"completion.json",dict(complete=True,posthoc=True,runs=15,protocol_sha256=sha(PLAN),summary_sha256=sha(summary),
        parser_amendment_sha256=sha(OUT/"parser_amendment_001.json"),paired_requests_sha256=sha(pairs),run_outputs=outputs))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("declare","run","self-test"));a=p.parse_args()
    if a.command=="declare":declare()
    elif a.command=="run":run()
    else:self_test();print("Seven synthetic category checks passed; no model executed.")
