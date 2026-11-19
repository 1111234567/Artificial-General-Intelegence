"""Post-B/C online ranking-only continuation; three reused B full controls."""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong.evaluate import evaluate_fork
from experiments.undirected.run import _atomic_json, file_sha256
from experiments.coverage_exit.retirement_study import THREAD_ENV, read_plan as read_base_plan, checked_baseline
from experiments.coverage_exit.retirement_report import checked_original_result, condition_rates
from experiments.coverage_exit.fast_window import install_window
from experiments.coverage_exit.retirement_magnitude import from_original, NativeTrainer, RANKING

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/coverage-exit/magnitude-v1'
B=ROOT/'data/coverage-exit/retirement-v1/protocol.json'
OWN=('retirement_magnitude.py','retirement_magnitude_study.py','retirement_magnitude_tests.py')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():raise ValueError('do not overwrite prospective E protocol')
    plan=deepcopy(read_base_plan(B))
    controls=[]
    for seed in range(3):
        p=ROOT/f'data/coverage-exit/retirement-v1/seed_{seed:03d}/full_eligible/result.json'
        checked_original_result(json.loads(p.read_text()))
        controls.append(dict(seed=seed,result=str(p),sha256=file_sha256(p)))
    plan.pop('protocol_hash')
    plan.update(format='retirement-magnitude-online-prospective-v1',created_utc=datetime.now(timezone.utc).isoformat(),output=str(OUT),
        arms=['magnitude'],controls=controls,b_protocol=str(B),b_protocol_sha256=file_sha256(B),
        posthoc='Follow-up chosen after observing B/C. B did not accept its signed candidates; C weak-pair forced ablations had smaller short behavior changes. No E arm training/effect results seen before protocol freeze.',
        mechanical_validation=dict(path=str(OUT/'validation.json'),sha256=file_sha256(OUT/'validation.json')))
    plan['execution_files'].update({str(Path(__file__).with_name(n)):file_sha256(Path(__file__).with_name(n)) for n in OWN})
    plan['execution_files'][str(Path(__file__).with_name('retirement_report.py'))]=file_sha256(Path(__file__).with_name('retirement_report.py'))
    plan['runtime'].update(jobs=3,max_concurrent_workers=3)
    plan['training'].update(ranking=RANKING,single_switch='Compared with B full_eligible, only fade candidate ranking changes. Full removal still requires strictly lower actual mixed loss; otherwise use unchanged gradual fallback. Growth remains disabled.')
    plan['evaluation']['aggregation']='Magnitude-minus-reused-B-full endpoint differences paired within each seed; strict/ordinary differences reported as percentage points; idle difference per1000 actual idle opportunities.'
    plan['limits']+=['This is a declared post-B/C follow-up, not a pre-B/C independent confirmation.',
        'An accepted mixed-loss batch can worsen live or history loss. True online deletion does not by itself establish retained behavior or specialized memory.',
        '15 proposals execute inside128 updates; pending proposal16 is not executed. No further structural variants are planned in this round.']
    plan['protocol_hash']=state_digest(plan);_atomic_json(path,plan);return path


def read_plan():
    plan=read_base_plan(OUT/'protocol.json')
    if file_sha256(plan['b_protocol'])!=plan['b_protocol_sha256']:raise ValueError('B protocol changed')
    for row in plan['controls']:
        if file_sha256(row['result'])!=row['sha256']:raise ValueError('B control changed')
        checked_original_result(json.loads(Path(row['result']).read_text()))
    validation=plan['mechanical_validation']
    if file_sha256(validation['path'])!=validation['sha256']:raise ValueError('mechanical validation changed')
    return plan


def worker(seed):
    for k,v in THREAD_ENV.items():
        if os.environ.get(k)!=v:raise ValueError('worker environment mismatch')
    torch.set_num_threads(1);plan=read_plan();src=plan['sources'][seed]
    out=OUT/f'seed_{seed:03d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise ValueError('completed result exists')
    trainer=from_original(src['checkpoint']);install_window(trainer)
    baseline=checked_baseline(src,trainer);work0=deepcopy(trainer.work);started=time.perf_counter()
    initial=trainer.semantic_state()
    _atomic_json(out/'adoption.json',dict(source=src,fade_ranking=RANKING,
        preserved={k:state_digest(initial[k]) for k in initial.keys()-{'format','config','operator','retirement_origin'}},operator=trainer.operator.summary()))
    for i in range(128):
        point=trainer.train_update()
        with (out/'trace_suffix.jsonl').open('a') as stream:stream.write(json.dumps(point,allow_nan=False)+'\n')
        _atomic_json(out/'progress.json',dict(stage='training',updates=i+1,deaths=trainer.operator.cumulative_deaths,tick=trainer.ticks))
        if (i+1)%8==0:print(seed,'update',i+1,'deaths',trainer.operator.cumulative_deaths,flush=True)
    trainer._validate_continuation();cp=out/'checkpoint_final.pt';trainer.save_checkpoint(cp)
    restored=NativeTrainer.from_checkpoint(cp,device='cpu');install_window(restored)
    if state_digest(restored.semantic_state())!=state_digest(trainer.semantic_state()):raise AssertionError('actual endpoint restore differs')
    del restored
    source_hash=state_digest(trainer.semantic_state());records=[]
    with _preserve_process_state():
        for family in range(4):
            for condition in ('short','long_both'):
                r=evaluate_fork(trainer.model,trainer.world,trainer.h,family=family,condition=condition,steps=2048,device='cpu')
                if any(r[k] for k in ('world_resets','hidden_resets','parameter_updates')):raise ValueError('biased fork')
                r['probe']=dict(tick=147456,family=family,condition=condition,steps=2048);records.append(r)
                _atomic_json(out/'evaluation_partial.json',dict(records=records))
                _atomic_json(out/'progress.json',dict(stage='evaluation',forks=len(records)))
                print(seed,'evaluated',family,condition,flush=True)
    if state_digest(trainer.semantic_state())!=source_hash:raise AssertionError('evaluation changed source')
    events=[r['event'] for r in trainer.operator.event_history]
    work={k:trainer.work[k]-work0[k] for k in work0}
    if len(events)!=15 or trainer.operator.probe_calls!=60 or work['optimizer_steps']!=128:raise ValueError('budget')
    read_plan()
    value=dict(format='retirement-magnitude-result-v1',seed=seed,status='success',protocol_hash=plan['protocol_hash'],
        source=src,endpoint_tick=trainer.ticks,checkpoint=str(cp),checkpoint_sha256=file_sha256(cp),endpoint_semantic_hash=source_hash,
        source_unchanged=True,checkpoint_roundtrip_verified=True,operator=trainer.operator.summary(),events=events,work_delta=work,
        baseline_records=baseline,endpoint_records=records,seconds=time.perf_counter()-started)
    value=json.loads(json.dumps(value,allow_nan=False));value['result_hash']=state_digest(value);_atomic_json(out/'result.json',value)
    _atomic_json(out/'progress.json',dict(stage='complete',seconds=value['seconds']))


def launch():
    read_plan()
    def one(seed):
        with (OUT/f'seed_{seed:03d}.log').open('w') as stream:
            p=subprocess.Popen([sys.executable,'-m','experiments.coverage_exit.retirement_magnitude_study','worker','--seed',str(seed)],cwd=ROOT,env={**os.environ,**THREAD_ENV},stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,pid=p.pid,returncode=p.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(one,range(3)))
    _atomic_json(OUT/'exit_codes.json',codes)
    if any(r['returncode'] for r in codes):raise RuntimeError(codes)


def report():
    plan=read_plan();rows=[];paired=[];counts=[];bindings=[]
    for seed in range(3):
        path=OUT/f'seed_{seed:03d}/result.json';value=json.loads(path.read_text())
        if value['result_hash']!=state_digest({k:v for k,v in value.items() if k!='result_hash'}) or value['protocol_hash']!=plan['protocol_hash']:raise ValueError('new result digest')
        if not value['checkpoint_roundtrip_verified'] or not value['source_unchanged'] or file_sha256(value['checkpoint'])!=value['checkpoint_sha256']:raise ValueError('endpoint source')
        control=json.loads(Path(plan['controls'][seed]['result']).read_text());checked_original_result(control)
        if value['work_delta']!=control['work_delta']:raise ValueError('paired scheduled work differs')
        if value['events'][0]['baseline']!=control['events'][0]['baseline'] or value['events'][0]['birth_indices']!=control['events'][0]['birth_indices']:raise ValueError('pre-switch boundary differs')
        if len(value['events'])!=15 or value['operator']['probe_calls']!=60 or value['operator']['cumulative_births']!=0:raise ValueError('structural budget')
        for r in value['endpoint_records']:
            c=next(c for c in control['endpoint_records'] if c['family']==r['family'] and c['condition']==r['condition'])
            for k in ('case_ids','actor_ids','fork_tick','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash'):
                if r[k]!=c[k]:raise ValueError(f'exogenous cases differ {k}')
            for cohort in ('primary','carryover_contract'):
                if r[cohort]['request_fingerprint']!=c[cohort]['request_fingerprint']:raise ValueError('matured cases differ')
        for arm,result in (('magnitude',value),('B_full_eligible',control)):
            op,events=result['operator'],result['events'];accepted=[e for e in events if e['fade_accepted']]
            damage=[e['full_ablation_damage'] if e['full_accepted'] else e['half_fade_damage'] for e in accepted]
            rows.append(dict(seed=seed,arm=arm,deaths=op['cumulative_deaths'],eligible_pairs=op['eligible_pairs'],nonzero_pairs=op['nonzero_pairs'],
                full_batches=sum(e['full_accepted'] for e in events),half_batches=sum(e['fade_accepted'] and not e['full_accepted'] for e in events),
                accepted_live_damage_batches=sum(d['live']>0 for d in damage),accepted_history_damage_batches=sum(d['history']>0 for d in damage),
                accepted_live_damage_sum=sum(d['live'] for d in damage),accepted_history_damage_sum=sum(d['history'] for d in damage),accepted_mixed_damage_sum=sum(d['mixed'] for d in damage),
                rates={c:condition_rates(result['endpoint_records'],c) for c in ('short','long_both')}))
            for r in result['endpoint_records']:
                p,o=r['primary'],r['opportunities'];counts.append(dict(seed=seed,arm=arm,family=r['family'],condition=r['condition'],matured=p['matured'],clean_completed=p['clean_completed'],completed=p['completed'],inactive_actions=o['inactive_actions'],inactive_opportunities=o['inactive_opportunities']))
        for c in ('short','long_both'):
            a,b=[next(r for r in rows if r['seed']==seed and r['arm']==arm)['rates'][c] for arm in ('magnitude','B_full_eligible')]
            paired.append(dict(seed=seed,condition=c,delta_strict_pp=100*(a['strict']-b['strict']),delta_ordinary_pp=100*(a['ordinary']-b['ordinary']),delta_idle_per1000=a['idle_per1000']-b['idle_per1000']))
        bindings.append(dict(path=str(path),sha256=file_sha256(path)))
    aggregate={c:{k:dict(mean=statistics.mean(r[k] for r in paired if r['condition']==c),sample_sd=statistics.stdev(r[k] for r in paired if r['condition']==c)) for k in ('delta_strict_pp','delta_ordinary_pp','delta_idle_per1000')} for c in ('short','long_both')}
    value=dict(format='retirement-magnitude-audit-v1',audit='passed',protocol_hash=plan['protocol_hash'],results=rows,paired=paired,aggregate=aggregate,
        result_bindings=bindings,checks=['exact source/protocol/checkpoint/JSON result digests','all48 new/reused forks share paired exogenous scheduler/event/request streams','all paired work_delta equal','complete checkpoint restore and no source reset/mutation','all15 proposal/event and lifetime ledgers validated on restore'],
        limits=plan['limits'],units=dict(delta_strict_pp='percentage points',delta_ordinary_pp='percentage points',delta_idle_per1000='actions per1000 actual idle opportunities'))
    value['report_hash']=state_digest(value);_atomic_json(OUT/'report.json',value)
    for name,data in (('paired.csv',paired),('family_counts.csv',counts)):
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','worker','launch','report'));p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='prepare':print(prepare())
    elif a.command=='worker':worker(a.seed)
    elif a.command=='launch':launch()
    else:print(json.dumps(report()['aggregate'],indent=2))
