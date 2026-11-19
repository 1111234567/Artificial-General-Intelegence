"""Prospective, future-only family0 withdrawal/return in six inherited systems."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import csv
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch

from experiments.action_return.absence import AbsenceContinuation, START, WITHDRAW_END, RETURN_END
from experiments.continuity.run import state_digest, restore_rng_state
from experiments.continuity.evaluate import _preserve_process_state
from experiments.lifelong.evaluate import evaluate_fork
from experiments.undirected.run import _atomic_json, file_sha256
from experiments.coverage_exit.retirement_study import THREAD_ENV
from experiments.coverage_exit.retirement_report import checked_original_result

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/action-return/absence-v1'
ARMS=('magnitude','signed_full')
PHASES=('initial','withdrawal','return')


def sources():
    rows=[]
    for seed in range(3):
        for arm in ARMS:
            folder=ROOT/(f'data/coverage-exit/magnitude-v1/seed_{seed:03d}' if arm=='magnitude' else f'data/coverage-exit/retirement-v1/seed_{seed:03d}/full_eligible')
            path=folder/'result.json';value=json.loads(path.read_text())
            if arm=='signed_full':checked_original_result(value)
            elif value['result_hash']!=state_digest({k:v for k,v in value.items() if k!='result_hash'}):raise ValueError('magnitude result digest')
            cp=Path(value['checkpoint'])
            if file_sha256(cp)!=value['checkpoint_sha256'] or value['endpoint_tick']!=START:raise ValueError('source endpoint binding')
            rows.append(dict(seed=seed,arm=arm,checkpoint=str(cp),checkpoint_sha256=file_sha256(cp),
                             result=str(path),result_sha256=file_sha256(path),endpoint_semantic_hash=value['endpoint_semantic_hash']))
    return rows


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise ValueError('prospective protocol already exists')
    value=dict(format='action-return-absence-prospective-v1',created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),
               execution_files={str(Path(__file__).with_name(n)):file_sha256(Path(__file__).with_name(n)) for n in ('absence.py','absence_study.py','absence_tests.py')},
               validation=dict(path=str(OUT/'validation.json'),sha256=file_sha256(OUT/'validation.json')),
               schedule=dict(start_tick=START,withdrawal_ticks=WITHDRAW_END-START,withdrawal_end_tick=WITHDRAW_END,
                             return_ticks=RETURN_END-WITHDRAW_END,return_end_tick=RETURN_END,
                             native_original_main_boundary=196608,withdrawal_future_NEW=[1,2,3],return_future_NEW=[0],
                             rule='future-only overlay beginning at actual current boundary; historical original schedule remains unchanged'),
               training=dict(updates=192,withdrawal_updates=128,return_updates=64,chunk_ticks=128,
                             inherited_support='both eligibility and numeric masks checked exactly against inherited endpoint each update; finite nonzero active weights continue learning',
                             optimizer='native Adam and clipping over all potential coordinates; inactive weights projected0; dormant moments retained',
                             pending='archive entire endpoint operator including pending proposal16; permanently cancel structural proposal/commit throughout follow-up; no resume',
                             retained='complete original h/world including pending events, RNG, Adam, recorded512 credit tape/age, replay reservoir and sampling RNG, traces and causal hashes',
                             replay='old family0 history remains eligible for ordinary history sampling; no zero-information claim'),
               evaluation=dict(initial='reuse SHA-bound initial8 native endpoint records after exact parameter/world/hidden binding',
                               families=[0,1,2,3],conditions=['short','long_both'],steps=2048,
                               endpoints=[WITHDRAW_END,RETURN_END],policy='unmodified full11 argmax, detached full-state fork, no hidden/world reset, no parameter updates',
                               metrics=['primary clean_completed/matured','primary completed/matured','inactive_actions/inactive_opportunities'],
                               primary='family0 short: previously acquired strict behavior, report per-seed counts and rates',
                               secondary='family0 long_both floor and nonwithdrawn families1..3 controls'),
               estimand=dict(primary='paired magnitude-minus-signed_full change from own initial endpoint through withdrawal and return',
                             raw='paired magnitude-minus-signed_full endpoint difference at each boundary',
                             units='strict/ordinary percentage points; idle actions per1000 actual inactive opportunities',
                             interpretation='two inherited systems already differ in weights, hidden state, optimizer and prior policy; subsequent policies identical and support frozen; total inherited strategy comparison, not pure deletion isolation'),
               runtime=dict(workers=3,cpu_threads=1,dtype='float64',backend='unchanged validated fast_window',environment=THREAD_ENV,
                            memory_gate='launch only with >=5GiB available RAM; max3 workers'),
               limits=['Chosen after prior20/8/36 retirement results; protocol frozen before these follow-up behavioral results.',
                       'Only future NEW family draws are withdrawn; old pending/replacement events and historical family0 replay remain.',
                       'Short family0 had measurable acquired behavior; long_both near floor cannot establish preservation from a null change.',
                       'Three paired seeds, descriptive mean/sample SD; no architecture-wide forgetting or memory localization claim.'])
    value['protocol_hash']=state_digest(value);_atomic_json(OUT/'protocol.json',value);return value


def read_plan():
    p=json.loads((OUT/'protocol.json').read_text())
    if p['protocol_hash']!=state_digest({k:v for k,v in p.items() if k!='protocol_hash'}):raise ValueError('protocol digest')
    for path,sha in p['execution_files'].items():
        if file_sha256(path)!=sha:raise ValueError('execution source changed '+path)
    v=p['validation']
    if file_sha256(v['path'])!=v['sha256']:raise ValueError('validation changed')
    for s in p['sources']:
        for name in ('checkpoint','result'):
            if file_sha256(s[name])!=s[name+'_sha256']:raise ValueError('input changed')
    return p


def snapshot(t):
    return dict(tick=t.ticks,updates=t.updates,live_positions=list(t.live_supervised_family_positions),
                auxiliary_suffix_positions=list(t.auxiliary_stats['suffix_family_positions']),
                last_live_family_tick=list(t.last_live_family_tick),last_aux_family_tick=list(t.last_aux_family_tick),
                last_live_family_age=[None if x<0 else t.ticks-x for x in t.last_live_family_tick],
                last_aux_family_age=[None if x<0 else t.ticks-x for x in t.last_aux_family_tick],
                family_exposures=t.world.family_exposures(),family_recency=t.world.family_recency(),
                event_hash=t.world.event_hash,scheduler_hash=state_digest(t.world.state_dict()['scheduler']),
                world_hash=state_digest(t.world.state_dict()),hidden_hash=state_digest(t.h),work=deepcopy(t.work),
                operator=t.operator.summary())


def baseline(src,run):
    value=json.loads(Path(src['result']).read_text());t=run.trainer
    if state_digest(t.operator.frozen_origin)!=run.adoption['source_field_hashes']['operator']:raise ValueError('initial operator binding')
    records=value['endpoint_records']
    if len(records)!=8 or run.adoption['source_semantic_hash']!=src['endpoint_semantic_hash']:raise ValueError('initial semantics')
    for r in records:
        if (r['parameter_hash']!=state_digest(t.model.state_dict()) or r['start_world_state_hash']!=state_digest(t.world.state_dict())
                or r['start_hidden_hash']!=state_digest({'hidden':t.h})):raise ValueError('initial fork binding')
    return records


def adoption_record(run):
    value={k:v for k,v in run.adoption.items() if k!='original_operator'}
    op=run.adoption['original_operator']
    value['original_operator_hash']=state_digest(op)
    value['archived_pending_generated_update']=None if op['pending'] is None else op['pending']['generated_update']
    value['archived_pending_execute_update']=None if op['pending'] is None else op['pending']['execute_update']
    return value


def worker(seed,arm):
    for key,value in THREAD_ENV.items():
        if os.environ.get(key)!=value:raise ValueError('thread environment')
    torch.set_num_threads(1);plan=read_plan();src=next(s for s in plan['sources'] if s['seed']==seed and s['arm']==arm)
    out=OUT/f'seed_{seed:03d}'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise ValueError('completed worker already exists')
    started=time.perf_counter();run=AbsenceContinuation(src['checkpoint'],arm);t=run.trainer
    records={'initial':baseline(src,run)};snapshots={'initial':snapshot(t)};checkpoints=[]
    _atomic_json(out/'adoption.json',adoption_record(run))
    phase_start=snapshot(t)
    for phase,end in (('withdrawal',WITHDRAW_END),('return',RETURN_END)):
        while t.ticks<end:
            point=run.train_update()
            with (out/'trace_suffix.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(point,allow_nan=False)+'\n')
            if t.updates%8==0:
                _atomic_json(out/'progress.json',dict(phase=phase,tick=t.ticks,seconds=time.perf_counter()-started))
                print(seed,arm,phase,t.ticks,flush=True)
        run.validate();cp=out/f'checkpoint_{phase}.pt';run.save_checkpoint(cp)
        checkpoint_hash=state_digest(run.semantic_state())
        restored=AbsenceContinuation.from_checkpoint(cp)
        if state_digest(restored.semantic_state())!=checkpoint_hash:raise ValueError('checkpoint roundtrip')
        del restored
        snapshots[phase]=snapshot(t)
        checkpoints.append(dict(phase=phase,path=str(cp),sha256=file_sha256(cp),semantic_hash=checkpoint_hash,roundtrip=True))
        source_hash=state_digest(run.semantic_state());records[phase]=[]
        with _preserve_process_state():
            for family in range(4):
                for condition in ('short','long_both'):
                    r=evaluate_fork(t.model,t.world,t.h,family=family,condition=condition,steps=2048,device='cpu')
                    if any(r[k] for k in ('world_resets','hidden_resets','parameter_updates')):raise ValueError('evaluation reset')
                    r['probe']=dict(tick=end,family=family,condition=condition,steps=2048);records[phase].append(r)
                    _atomic_json(out/'evaluation_partial.json',records)
        if state_digest(run.semantic_state())!=source_hash:raise ValueError('evaluation mutated training source')
        print(seed,arm,'evaluated',phase,flush=True)
    read_plan()
    value=dict(format='action-return-absence-result-v1',status='success',seed=seed,arm=arm,protocol_hash=plan['protocol_hash'],
               source=src,adoption=adoption_record(run),records=records,snapshots=snapshots,checkpoints=checkpoints,
               source_unchanged=True,seconds=time.perf_counter()-started)
    value=json.loads(json.dumps(value,allow_nan=False));value['result_hash']=state_digest(value);_atomic_json(out/'result.json',value)
    _atomic_json(out/'progress.json',dict(phase='complete',tick=t.ticks,seconds=value['seconds']))


def available_memory():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    s=Status();s.length=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):raise OSError('RAM query failed')
    return s.available


def launch():
    read_plan()
    def one(item):
        seed,arm=item
        # Three processes are the hard upper bound. A low-RAM condition is
        # explicit and recorded rather than risking paging/OOM corruption.
        memory=available_memory()
        if memory<5*1024**3:raise MemoryError(f'RAM gate: {memory}')
        with (OUT/f'seed_{seed:03d}_{arm}.log').open('w') as stream:
            p=subprocess.Popen([sys.executable,'-m','experiments.action_return.absence_study','worker','--seed',str(seed),'--arm',arm],
                               cwd=ROOT,env={**os.environ,**THREAD_ENV},stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,arm=arm,pid=p.pid,available_ram_before_launch=memory,returncode=p.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(one,[(s,a) for s in range(3) for a in ARMS]))
    _atomic_json(OUT/'exit_codes.json',codes)
    if any(v['returncode'] for v in codes):raise RuntimeError(codes)


def rates(record):
    p,o=record['primary'],record['opportunities']
    if min(p['matured'],o['inactive_opportunities'])<=0:raise ValueError('undefined denominator')
    return dict(strict_pp=100*p['clean_completed']/p['matured'],ordinary_pp=100*p['completed']/p['matured'],
                idle_per1000=1000*o['inactive_actions']/o['inactive_opportunities'])


def report():
    plan=read_plan();results={};counts=[];deltas=[];exposure=[];bindings=[]
    for src in plan['sources']:
        seed,arm=src['seed'],src['arm'];path=OUT/f'seed_{seed:03d}'/arm/'result.json';v=json.loads(path.read_text())
        if v['result_hash']!=state_digest({k:x for k,x in v.items() if k!='result_hash'}) or v['protocol_hash']!=plan['protocol_hash']:raise ValueError('result digest')
        if v['source']!=src or not v['source_unchanged']:raise ValueError('result source')
        for cp in v['checkpoints']:
            if not cp['roundtrip'] or file_sha256(cp['path'])!=cp['sha256']:raise ValueError('checkpoint binding')
        results[seed,arm]=v;bindings.append(dict(path=str(path),sha256=file_sha256(path)))
        for phase in PHASES:
            for r in v['records'][phase]:
                p,o=r['primary'],r['opportunities']
                initial=next(x for x in v['records']['initial'] if x['family']==r['family'] and x['condition']==r['condition'])
                previous=next(x for x in v['records']['withdrawal' if phase=='return' else 'initial'] if x['family']==r['family'] and x['condition']==r['condition'])
                rr,ir,pr=rates(r),rates(initial),rates(previous)
                counts.append(dict(seed=seed,arm=arm,phase=phase,family=r['family'],condition=r['condition'],
                    matured=p['matured'],strict_completed=p['clean_completed'],ordinary_completed=p['completed'],
                    inactive_actions=o['inactive_actions'],inactive_opportunities=o['inactive_opportunities'],**rr,
                    **{'change_from_initial_'+k:rr[k]-ir[k] for k in rr},
                    **{'change_within_phase_'+k:rr[k]-pr[k] for k in rr}))
        for phase,prev in (('withdrawal','initial'),('return','withdrawal')):
            a,b=v['snapshots'][prev],v['snapshots'][phase]
            if b['operator']['eligible_pairs']!=a['operator']['eligible_pairs'] or b['operator']['cumulative_deaths']!=a['operator']['cumulative_deaths']:raise ValueError('support changed')
            work={k:b['work'][k]-a['work'][k] for k in a['work']}
            if work['structural_probe_calls']!=0 or work['optimizer_steps']!=({'withdrawal':128,'return':64}[phase]):raise ValueError('continuation budget')
            exposure.append(dict(seed=seed,arm=arm,phase=phase,
                live_positions=[y-x for x,y in zip(a['live_positions'],b['live_positions'])],
                auxiliary_suffix_positions=[y-x for x,y in zip(a['auxiliary_suffix_positions'],b['auxiliary_suffix_positions'])],
                actual_family_exposures={k:[y-x for x,y in zip(a['family_exposures'][k],b['family_exposures'][k])] for k in a['family_exposures']},
                last_live_family_age=b['last_live_family_age'],last_aux_family_age=b['last_aux_family_age'],work_delta=work))
    for seed in range(3):
        for phase in PHASES:
            m,c=results[seed,'magnitude'],results[seed,'signed_full']
            for key in ('event_hash','scheduler_hash','family_exposures'):
                if m['snapshots'][phase][key]!=c['snapshots'][phase][key]:raise ValueError('paired external stream differs '+key)
            if phase!='initial':
                prev='withdrawal' if phase=='return' else 'initial'
                work_deltas=[{k:v['snapshots'][phase]['work'][k]-v['snapshots'][prev]['work'][k] for k in v['snapshots'][prev]['work']} for v in (m,c)]
                if work_deltas[0]!=work_deltas[1]:raise ValueError('paired native scheduled work differs')
            for family in range(4):
                for condition in ('short','long_both'):
                    def find(v,p):return next(r for r in v['records'][p] if r['family']==family and r['condition']==condition)
                    a,b=find(m,phase),find(c,phase)
                    for key in ('case_ids','actor_ids','fork_tick','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash'):
                        if a[key]!=b[key]:raise ValueError('paired evaluation exogenous mismatch '+key)
                    for cohort in ('primary','carryover_contract'):
                        if a[cohort]['request_fingerprint']!=b[cohort]['request_fingerprint']:raise ValueError('paired matured cases differ')
                    ar,br=rates(a),rates(b);ai,bi=rates(find(m,'initial')),rates(find(c,'initial'))
                    previous='withdrawal' if phase=='return' else 'initial'
                    ap,bp=rates(find(m,previous)),rates(find(c,previous))
                    deltas.append(dict(seed=seed,phase=phase,family=family,condition=condition,
                        **{f'raw_{k}':ar[k]-br[k] for k in ar},
                        **{f'difference_in_changes_{k}':(ar[k]-ai[k])-(br[k]-bi[k]) for k in ar},
                        **{f'difference_in_phase_changes_{k}':(ar[k]-ap[k])-(br[k]-bp[k]) for k in ar}))
    aggregate=[]
    for phase in PHASES:
        for family in range(4):
            for condition in ('short','long_both'):
                rows=[r for r in deltas if r['phase']==phase and r['family']==family and r['condition']==condition]
                aggregate.append(dict(phase=phase,family=family,condition=condition,
                    **{k:dict(mean=statistics.mean(r[k] for r in rows),sample_sd=statistics.stdev(r[k] for r in rows)) for k in rows[0] if k.startswith(('raw_','difference_'))}))
    value=dict(format='action-return-absence-audit-v1',audit='passed',protocol_hash=plan['protocol_hash'],
               counts=counts,paired=deltas,aggregate=aggregate,training_exposures=exposure,result_bindings=bindings,
               checks=['all6 source/results SHA and complete semantic bindings','all12 continuation checkpoints verified restore',
                       '96 new frozen free forks plus48 SHA-bound reused initial forks','paired exogenous scheduler/event/request streams',
                       'historical causal traces unchanged; inherited eligibility and numeric support fixed','zero further structural probes/commits; native128+64 Adam updates',
                       'evaluation full-state source remains unchanged'],limits=plan['limits'])
    value['report_hash']=state_digest(value);_atomic_json(OUT/'report.json',value)
    for name,rows in (('counts.csv',counts),('paired.csv',deltas)):
        with (OUT/name).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','worker','launch','report'));p.add_argument('--seed',type=int);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.command=='prepare':print(prepare()['protocol_hash'])
    elif a.command=='worker':worker(a.seed,a.arm)
    elif a.command=='launch':launch()
    else:print(json.dumps([r for r in report()['aggregate'] if r['family']==0],indent=2))
