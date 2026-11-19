"""Post hoc frozen forced ablations, separated from online acceptance study B."""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import state_digest
from experiments.lifelong.evaluate import evaluate_fork
from experiments.undirected.run import file_sha256, _atomic_json
from experiments.coverage_exit.retirement_run import NativeTrainer
from experiments.coverage_exit.retirement_study import THREAD_ENV, read_plan as read_b_plan
from experiments.coverage_exit.retirement_report import checked_original_result

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/coverage-exit/retirement-ablation-v1'
B=ROOT/'data/coverage-exit/retirement-v1/protocol.json'
ARMS=('pending_full','weakest4')


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'design.json'
    if path.exists(): raise ValueError('do not overwrite post hoc declaration')
    value=dict(format='retirement-frozen-ablation-posthoc-design-v1',declared_utc=datetime.now(timezone.utc).isoformat(),
        selection='Post hoc question after first three completed B branches had no accepted full or half batches. All three half_only endpoints fixed regardless of remaining outcomes.',
        seen='Seed0 both and seed1 half endpoint summaries seen. Seed2 training progress at128/0 deaths seen; no seed2 fork scores, pending candidate indices or weak-pair values inspected before this declaration. All modified-group outcomes unmeasured.',
        source_rule='B half_only complete final endpoints for exactly seeds0/1/2, tick147456; each retains its own full h/world/parameters; no training',
        arms=dict(original='reuse exactly the four SHA-bound B short forks per seed',
            pending_full='set the actual physical shared weights of the four pending proposal16 fade_indices to exactly zero on a cloned model',
            weakest4='set the four eligible smallest abs(shared weight) pairs to zero, stable canonical pair-index tie break, on a cloned model'),
        evaluation=dict(condition='short',families=[0,1,2,3],steps=2048,action_policy='full11 argmax, free own actions/receipts, no hidden/world reset',
            new_forks=24,reused_forks=12,matching='same complete endpoint source h/world; future scheduler/event streams and matured request identity/timing fingerprints must match source original forks'),
        metrics=['strict clean_completed/matured','ordinary completed/matured','inactive_actions/inactive_opportunities',
            'candidate canonical indices, node pairs, signed weights, abs weights and overlap'],
        aggregation='all per-seed and per-family counts/rates retained; each seed equal-weights four families; ablation-minus-original paired per-seed, then three-seed mean/sampleSD',
        fixed_budget='exactly four pairs per intervention and2048ticks per declared family; no tuning, pruning search or optional stopping',
        scope='Forced frozen physical-weight ablation, NOT online algorithmic deletion or evidence that the learning policy learned to retire connections. Eligibility and optimizer/credit history do not change.',
        failure='Preserve failed/missing group and do not fill zero, replace seed, alter candidate count or choose another checkpoint.')
    value['design_hash']=state_digest(value);_atomic_json(path,value)
    return path


def bind():
    design=json.loads((OUT/'design.json').read_text())
    if design['design_hash']!=state_digest({k:v for k,v in design.items() if k!='design_hash'}): raise ValueError('declaration changed')
    read_b_plan(B)
    path=OUT/'sources.json'
    if path.exists(): raise ValueError('source binding exists')
    sources=[]
    for seed in range(3):
        result=ROOT/f'data/coverage-exit/retirement-v1/seed_{seed:03d}/half_only/result.json'
        value=json.loads(result.read_text())
        if value['status']!='success' or value['endpoint_tick']!=147456: raise ValueError('source incomplete')
        checked_original_result(value)
        sources.append(dict(seed=seed,result=str(result),result_sha256=file_sha256(result),
            checkpoint=value['checkpoint'],checkpoint_sha256=file_sha256(value['checkpoint'])))
    value=dict(format='retirement-ablation-source-binding-v1',bound_utc=datetime.now(timezone.utc).isoformat(),
        design=str(OUT/'design.json'),design_sha256=file_sha256(OUT/'design.json'),b_protocol=str(B),b_protocol_sha256=file_sha256(B),
        execution_files={str(p.resolve()):file_sha256(p) for p in (Path(__file__),Path(__file__).with_name('retirement_report.py'))},sources=sources,runtime=dict(environment=THREAD_ENV,cpu_threads=1,max_concurrent_workers=3,dtype='float64',device='cpu'))
    value['binding_hash']=state_digest(value);_atomic_json(path,value);return path


def load_binding():
    value=json.loads((OUT/'sources.json').read_text())
    if value['binding_hash']!=state_digest({k:v for k,v in value.items() if k!='binding_hash'}): raise ValueError('source manifest digest')
    for field in ('design','b_protocol'):
        if file_sha256(value[field])!=value[field+'_sha256']: raise ValueError('protocol changed')
    for path,sha in value['execution_files'].items():
        if file_sha256(path)!=sha: raise ValueError('ablation code changed')
    for src in value['sources']:
        for field in ('result','checkpoint'):
            if file_sha256(src[field])!=src[field+'_sha256']: raise ValueError('B endpoint changed')
    read_b_plan(B)
    return value


def candidates(trainer):
    pending=trainer.operator.pending
    if pending is None or pending['generated_update']!=trainer.updates or len(pending['fade_indices'])!=4:
        raise ValueError('source lacks pending proposal16 with four fade candidates')
    eligible=trainer.operator.mask.nonzero().flatten()
    weights=trainer.model.edge_shared.detach().cpu()
    weak=eligible[torch.argsort(weights[eligible].abs(),stable=True)[:4]]
    result={name:indices.clone() for name,indices in (('pending_full',pending['fade_indices']),('weakest4',weak))}
    if any(not bool(trainer.operator.mask[i].all()) or len(i.unique())!=4 for i in result.values()):
        raise ValueError('ineligible/duplicate candidate')
    return result


def worker(seed):
    for k,v in THREAD_ENV.items():
        if os.environ.get(k)!=v: raise ValueError('environment mismatch')
    torch.set_num_threads(1)
    binding=load_binding();src=binding['sources'][seed]
    b=json.loads(Path(src['result']).read_text())
    trainer=NativeTrainer.from_checkpoint(src['checkpoint'],device='cpu')
    source_hash=state_digest(trainer.semantic_state())
    if source_hash!=b['endpoint_semantic_hash']: raise ValueError('complete B source mismatch')
    original=[r for r in b['endpoint_records'] if r['condition']=='short']
    if len(original)!=4: raise ValueError('original short forks missing')
    for r in original:
        if r['parameter_hash']!=state_digest(trainer.model.state_dict()) or r['start_world_state_hash']!=state_digest(trainer.world.state_dict()) or r['start_hidden_hash']!=state_digest({'hidden':trainer.h}):
            raise ValueError('reused baseline source differs')
    groups=candidates(trainer); records=[];details={}
    for arm,indices in groups.items():
        model=deepcopy(trainer.model)
        weights=model.edge_shared.detach().cpu()[indices].clone()
        with torch.no_grad(): model.edge_shared[indices.to(model.edge_shared.device)]=0.
        if not bool(weights.ne(0).all()): raise ValueError('intervention did not remove four actual nonzero pairs')
        details[arm]=dict(indices=indices.tolist(),node_pairs=trainer.model.pair_index[:,indices].T.tolist(),
            signed_weights=weights.tolist(),abs_weights=weights.abs().tolist(),
            source_nonzero_pairs=int(trainer.model.edge_shared.ne(0).sum()),ablated_nonzero_pairs=int(model.edge_shared.ne(0).sum()),
            eligible_pairs_unchanged=int(trainer.operator.mask.sum()),parameter_hash=state_digest(model.state_dict()))
        with _preserve_process_state():
            for family in range(4):
                record=evaluate_fork(model,trainer.world,trainer.h,family=family,condition='short',steps=2048,device='cpu')
                base=next(r for r in original if r['probe']['family']==family)
                for field in ('start_world_state_hash','start_hidden_hash','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash','case_ids'):
                    if record[field]!=base[field]: raise ValueError(f'forced fork cases differ: {field}')
                if record['primary']['request_fingerprint']!=base['primary']['request_fingerprint']: raise ValueError('matured cases differ')
                if any(record[k] for k in ('world_resets','hidden_resets','parameter_updates')): raise ValueError('biased fork')
                record['probe']=dict(tick=147456,family=family,condition='short',steps=2048)
                record['ablation_arm']=arm;records.append(record)
                _atomic_json(OUT/f'seed_{seed:03d}_partial.json',dict(records=records))
                print(seed,arm,family,'complete',flush=True)
    if state_digest(trainer.semantic_state())!=source_hash: raise AssertionError('forced evaluation mutated B source')
    load_binding()
    value=dict(format='retirement-forced-ablation-result-v1',seed=seed,status='success',binding_hash=binding['binding_hash'],
        source=src,source_semantic_hash=source_hash,source_unchanged=True,candidates=details,
        overlap=sorted(set(groups['pending_full'].tolist())&set(groups['weakest4'].tolist())),
        original_records=original,ablation_records=records,new_forks=8,reused_forks=4)
    value=json.loads(json.dumps(value,allow_nan=False))
    value['result_hash']=state_digest(value);_atomic_json(OUT/f'seed_{seed:03d}.json',value)


def launch():
    load_binding()
    def one(seed):
        with (OUT/f'seed_{seed:03d}.log').open('w') as log:
            p=subprocess.Popen([sys.executable,'-m','experiments.coverage_exit.retirement_ablation','worker','--seed',str(seed)],cwd=ROOT,env={**os.environ,**THREAD_ENV},stdout=log,stderr=subprocess.STDOUT)
            return dict(seed=seed,pid=p.pid,returncode=p.wait())
    with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(one,range(3)))
    _atomic_json(OUT/'exit_codes.json',results)
    if any(r['returncode'] for r in results): raise RuntimeError(results)


def report():
    import csv
    binding=load_binding(); rows=[];paired=[];raw=[]
    for seed in range(3):
        path=OUT/f'seed_{seed:03d}.json';x=json.loads(path.read_text())
        if x['result_hash']!=state_digest({k:v for k,v in x.items() if k!='result_hash'}) or x['binding_hash']!=binding['binding_hash'] or not x['source_unchanged']: raise ValueError('unbound result')
        means={}
        for arm in ('original',*ARMS):
            records=x['original_records'] if arm=='original' else [r for r in x['ablation_records'] if r['ablation_arm']==arm]
            if len(records)!=4: raise ValueError('incomplete family data')
            rates=[]
            for r in records:
                p,o=r['primary'],r['opportunities']
                if not p['matured'] or not o['inactive_opportunities']: raise ValueError('undefined rate')
                row=dict(seed=seed,arm=arm,family=r['probe']['family'],matured=p['matured'],clean_completed=p['clean_completed'],completed=p['completed'],
                    inactive_actions=o['inactive_actions'],inactive_opportunities=o['inactive_opportunities'],strict=p['clean_completed']/p['matured'],ordinary=p['completed']/p['matured'],idle_per1000=1000*o['inactive_actions']/o['inactive_opportunities'])
                rows.append(row);rates.append(row)
            means[arm]={key:statistics.mean(r[key] for r in rates) for key in ('strict','ordinary','idle_per1000')}
        for arm in ARMS:
            paired.append(dict(seed=seed,arm=arm,**{k:means[arm][k]-means['original'][k] for k in means[arm]}))
        raw.append(dict(seed=seed,means=means,candidates=x['candidates'],overlap=x['overlap'],path=str(path),sha256=file_sha256(path)))
    aggregate={arm:{key:dict(mean=statistics.mean(r[key] for r in paired if r['arm']==arm),sample_sd=statistics.stdev(r[key] for r in paired if r['arm']==arm)) for key in ('strict','ordinary','idle_per1000')} for arm in ARMS}
    value=dict(format='retirement-forced-ablation-report-v1',audit='passed',binding_hash=binding['binding_hash'],seeds=raw,paired_ablation_minus_original=paired,aggregate=aggregate,
        counts=dict(independent_seeds=3,new_forks=24,reused_forks=12),scope=json.loads((OUT/'design.json').read_text())['scope'])
    value['report_hash']=state_digest(value);_atomic_json(OUT/'report.json',value)
    for name,data in (('family_counts.csv',rows),('paired.csv',paired)):
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('declare','bind','worker','launch','report'));p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='declare': print(declare())
    elif a.command=='bind': print(bind())
    elif a.command=='worker': worker(a.seed)
    elif a.command=='launch': launch()
    else: print(json.dumps(report()['aggregate'],indent=2))
