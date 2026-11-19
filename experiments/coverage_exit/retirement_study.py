"""Prospectively bind and execute paired full-eligible versus old-half retire-only study."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from experiments.continuity.run import state_digest, cpu_tree
from experiments.continuity.evaluate import _preserve_process_state
from experiments.lifelong.evaluate import evaluate_fork
from experiments.undirected.run import file_sha256, _atomic_json
from experiments.undirected.study import execution_files as original_files
from experiments.coverage_exit.retirement_run import from_original, NativeTrainer
from experiments.coverage_exit.fast_window import install_window

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'data/coverage-exit/retirement-v1'
ARMS=('half_only','full_eligible')
THREAD_ENV={key:'1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
THREAD_ENV['CUDA_VISIBLE_DEVICES']='-1'
OWN=('retirement.py','retirement_run.py','retirement_study.py','test_retirement.py','fast_window.py','test_fast_window.py','benchmark_window.py')
VALIDATION=ROOT/'data/coverage-exit/perf/recorded-window-cpu-v1.json'


def prepare(output=DEFAULT):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists(): raise ValueError('protocol exists; do not overwrite prospective evidence')
    files={**original_files(), **{str(Path(__file__).with_name(n)):file_sha256(Path(__file__).with_name(n)) for n in OWN}}
    sources=[]
    for seed in range(3):
        identifier=f'seed_{seed:03d}__core__undirected__recorded512__fixed__warmup'
        cp=ROOT/f'data/undirected/main-v1/training/seed_{seed:03d}/core__undirected__recorded512__fixed__warmup/checkpoints/tick_131072.pt'
        baseline=ROOT/f'data/undirected/main-v1/evaluation/{identifier}.json'
        sources.append(dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=file_sha256(cp),baseline=str(baseline),baseline_sha256=file_sha256(baseline)))
    plan=dict(format='coverage-exit-retirement-prospective-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        output=str(output),execution_files=files,sources=sources,arms=list(ARMS),seeds=[0,1,2],
        mechanical_validation=dict(path=str(ROOT/'data/coverage_exit/retirement-validation-v1.json'),
            sha256=file_sha256(ROOT/'data/coverage_exit/retirement-validation-v1.json'),
            scope='8 retirement tests plus canonical source adoption/actual-update restore and bitwise next-update recovery; excluded from effect estimation'),
        source_tick=131072,source_updates=1024,continuation_updates=128,endpoint_tick=147456,
        expected_proposals_generated=16,expected_executed_windows=15,expected_probe_calls=60,
        runtime=dict(device='cpu',world_device='cpu',dtype='float64',cpu_threads=1,jobs=6,max_concurrent_workers=3,environment=THREAD_ENV,
            backend='install_window in both arms; local constant-parameter matrix hoisting; live world path remains native',
            backend_validation_path=str(VALIDATION),backend_validation_sha256=file_sha256(VALIDATION)),
        training=dict(policy='retire-only in both arms; no growth',candidates=4,window_updates=8,
            ranking='unchanged stable signed -weight*raw-gradient mean',
            single_switch='full_eligible commits full zeroing when measured full batch mixed loss is strictly below actual fade baseline; else use old half/threshold strict-loss acceptance',
            probe_budget='same baseline, disabled-birth trial, full-ablation trial, half-fade trial; exactly four forwards per execution',
            adoption='retain all original optimizer moments/age, h, world, replay, credit tape/age, lifetime masks/counters/update ledger and trace; reset only newly activated proposal gradient sum/window/pending at source boundary',
            endpoint='fixed finite horizon; do not stop when deletion count favorable'),
        evaluation=dict(steps=2048,conditions=['short','long_both'],families=[0,1,2,3],action_policy='unmodified full11 argmax',
            primary='each endpoint complete world/hidden/parameters, detached with no reset and free full11 argmax; paired arms must share external future commands/scheduler and cases while receipts/h follow their own actions',
            baseline='reuse frozen original source evaluation only after SHA and complete parameter/world/h/condition bindings; establishes nonzero source behavior; source-versus-endpoint uses different time boundaries and is descriptive',
            metrics=['strict completion clean_completed/matured','ordinary completion completed/matured','inactive_actions/inactive_opportunities'],
            aggregation='equal family rates within seed; paired full-minus-half per seed; three seeds sample mean/SD only after all finish; raw counts retained'),
        structural_endpoints=['true eligibility deaths','numeric nonzero pairs','eligible pairs','accepted full batches','accepted half batches','full mixed improvements','all per-window live/history/mixed baseline/full/half losses'],
        limits=['This tests a loss-gated operator in trained core/shared finite-window models, not the zero-behavior ports/RTRL architecture.',
            'Deletion of a jointly tested batch does not establish that each pair is harmless or identify a disposable memory.',
            'A lower candidate mixed loss may increase history loss. Equal probe counts and scheduled learning budgets do not force identical trajectories.',
            'Source-versus-endpoint change is descriptive across time; primary causal contrast is paired full-minus-half complete endpoint systems under matched external commands.',
            '15 proposals execute inside128 updates; proposal16 is pending and remains unexecuted.'])
    plan['protocol_hash']=state_digest(plan)
    _atomic_json(path,plan)
    return path


def read_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan['protocol_hash']!=state_digest({k:v for k,v in plan.items() if k!='protocol_hash'}): raise ValueError('protocol digest mismatch')
    for path,sha in plan['execution_files'].items():
        if file_sha256(path)!=sha: raise ValueError(f'source changed: {path}')
    runtime=plan['runtime']
    if file_sha256(runtime['backend_validation_path'])!=runtime['backend_validation_sha256']:
        raise ValueError('backend validation changed')
    for src in plan['sources']:
        for field in ('checkpoint','baseline'):
            if file_sha256(src[field])!=src[field+'_sha256']: raise ValueError(f'input changed: {src[field]}')
    return plan


def checked_baseline(src,trainer):
    value=json.loads(Path(src['baseline']).read_text())
    if value['result_hash']!=state_digest({k:v for k,v in value.items() if k!='result_hash'}): raise ValueError('old evaluation digest')
    saved=value['states']['131072']
    if saved['checkpoint_sha256']!=src['checkpoint_sha256'] or saved['state_hash']!=trainer.retirement_origin['semantic_hash']: raise ValueError('old source binding')
    records=[r for r in value['records'] if r['probe']['tick']==131072]
    if len(records)!=8: raise ValueError('baseline missing cases')
    for r in records:
        if r['parameter_hash']!=state_digest(trainer.model.state_dict()) or r['start_world_state_hash']!=state_digest(trainer.world.state_dict()) or r['start_hidden_hash']!=state_digest({'hidden':trainer.h}): raise ValueError('baseline actual state binding')
    return records


def worker(path,seed,arm):
    for key,value in THREAD_ENV.items():
        if os.environ.get(key)!=value: raise ValueError(f'worker env mismatch: {key}')
    torch.set_num_threads(1)
    plan=read_plan(path); src=plan['sources'][seed]
    if arm not in ARMS: raise ValueError('arm')
    out=Path(plan['output'])/f'seed_{seed:03d}'/arm
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise ValueError('completed worker output exists')
    started=time.perf_counter()
    trainer=from_original(src['checkpoint'],full_eligible=arm=='full_eligible')
    install_window(trainer)
    baseline=checked_baseline(src,trainer)
    before=trainer.semantic_state(); work0=deepcopy(trainer.work)
    if (trainer.ticks,trainer.config.interface,trainer.config.credit,trainer.config.dtype)!=(131072,'core','recorded512','float64'): raise ValueError('source spec')
    if not any(r['primary']['clean_completed'] for r in baseline): raise ValueError('source behavior absent')
    _atomic_json(out/'adoption.json',dict(source=src,arm=arm,activation_update=trainer.updates,
        preserved={key:state_digest(before[key]) for key in before.keys()-{'format','config','operator','retirement_origin'}},
        operator=trainer.operator.summary(),baseline_primary=[dict(probe=r['probe'],primary=r['primary']) for r in baseline]))
    for i in range(plan['continuation_updates']):
        point=trainer.train_update()
        with (out/'trace_suffix.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(point,allow_nan=False)+'\n')
        _atomic_json(out/'progress.json',dict(stage='training',completed_updates=i+1,target_updates=128,tick=trainer.ticks,
            eligible_pairs=int(trainer.operator.mask.sum()),deaths=trainer.operator.cumulative_deaths,seconds=time.perf_counter()-started))
        if (i+1)%8==0: print(seed,arm,'update',i+1,'deaths',trainer.operator.cumulative_deaths,flush=True)
    trainer._validate_continuation()
    final=out/'checkpoint_final.pt'; trainer.save_checkpoint(final)
    # A real complete reload validates both the new operator semantics and retained trace/history.
    restored=NativeTrainer.from_checkpoint(final,device='cpu')
    install_window(restored)
    if state_digest(restored.semantic_state())!=state_digest(trainer.semantic_state()): raise AssertionError('endpoint restore differs')
    del restored
    records=[]; unchanged=state_digest(trainer.semantic_state())
    with _preserve_process_state():
        for family in plan['evaluation']['families']:
            for condition in plan['evaluation']['conditions']:
                record=evaluate_fork(trainer.model,trainer.world,trainer.h,family=family,condition=condition,steps=2048,device='cpu')
                base=next(r for r in baseline if r['probe']['family']==family and r['condition']==condition)
                if any(record[k] for k in ('world_resets','hidden_resets','parameter_updates')): raise AssertionError('biased eval')
                record['probe']=dict(tick=147456,family=family,condition=condition,steps=2048)
                records.append(record)
                _atomic_json(out/'evaluation_partial.json',dict(records=records))
                _atomic_json(out/'progress.json',dict(stage='evaluation',completed_forks=len(records),target_forks=8,seconds=time.perf_counter()-started))
                print(seed,arm,'evaluated',family,condition,flush=True)
    if state_digest(trainer.semantic_state())!=unchanged: raise AssertionError('evaluation mutated training')
    events=[r['event'] for r in trainer.operator.event_history]
    work={k:trainer.work[k]-work0[k] for k in work0}
    if len(events)!=15 or trainer.operator.probe_calls!=60 or work['optimizer_steps']!=128: raise AssertionError('budget')
    result=dict(format='coverage-exit-retirement-result-v1',protocol_hash=plan['protocol_hash'],seed=seed,arm=arm,status='success',
        source=src,endpoint_tick=trainer.ticks,checkpoint=str(final),checkpoint_sha256=file_sha256(final),
        endpoint_semantic_hash=unchanged,operator=trainer.operator.summary(),events=events,work_delta=work,
        baseline_records=baseline,endpoint_records=records,training_suffix_records=128,source_unchanged=True,
        checkpoint_roundtrip_verified=True,seconds=time.perf_counter()-started)
    read_plan(path)
    result['result_hash']=state_digest(result); _atomic_json(out/'result.json',result)
    _atomic_json(out/'progress.json',dict(stage='complete',seconds=result['seconds']))
    print(seed,arm,'COMPLETE',result['seconds'],flush=True)


def launch(path):
    plan=read_plan(path); processes=[]; lock=threading.Lock()
    def one(seed,arm):
        log=Path(plan['output'])/f'seed_{seed:03d}__{arm}.log'
        with log.open('w',encoding='utf-8') as stream:
            env={**os.environ,**THREAD_ENV}
            proc=subprocess.Popen([sys.executable,'-m','experiments.coverage_exit.retirement_study','worker','--protocol',str(Path(path).resolve()),'--seed',str(seed),'--arm',arm],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            with lock:
                processes.append(dict(seed=seed,arm=arm,pid=proc.pid))
                _atomic_json(Path(plan['output'])/'launch.json',dict(workers=list(processes),jobs=6,max_concurrent_workers=3,thread_env=THREAD_ENV))
            return dict(seed=seed,arm=arm,returncode=proc.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one,seed,arm) for seed in plan['seeds'] for arm in ARMS]
        codes=[future.result() for future in futures]
    _atomic_json(Path(plan['output'])/'exit_codes.json',codes)
    if any(r['returncode'] for r in codes): raise RuntimeError(codes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','worker','launch'))
    p.add_argument('--protocol',type=Path,default=DEFAULT/'protocol.json');p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--seed',type=int);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.command=='prepare': print(prepare(a.output))
    elif a.command=='worker': worker(a.protocol,a.seed,a.arm)
    else: launch(a.protocol)

if __name__=='__main__': main()
