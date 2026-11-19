"""Post-result instantaneous silencing of exact historically retired coordinates.

No online retirement or continuation training occurs here. Copies of the
unpruned control model are changed immediately before frozen free evaluation;
the actual source world/h and full training state are never modified.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from experiments.action_return import absence_study as study
from experiments.action_return.absence import AbsenceContinuation,source_trainer
from experiments.continuity.run import state_digest
from experiments.continuity.evaluate import _preserve_process_state
from experiments.lifelong.evaluate import evaluate
from experiments.coverage_exit.coverage import read,write,sha

torch.set_num_threads(1)
FORMAT='action-return-exact-edge-identity-posthoc-v1'
OUT=ROOT/'data/action-return/edge-identity-v1';PLAN=OUT/'protocol.json';SOURCE=Path(__file__).resolve()
PHASES=('initial','withdrawal','return');TICKS=(147456,163840,172032)

def retired_identity(trainer,expected_count):
    op=trainer.operator
    indices=(op.initial_mask&~op.mask).nonzero().flatten().tolist()
    ledger=sorted({int(i) for row in op.event_history for i in row['event']['actual_retire_indices']})
    if indices!=ledger or len(indices)!=expected_count or op.cumulative_deaths!=expected_count or op.cumulative_births!=0:raise AssertionError('historical actual-retirement identities differ')
    if bool(trainer.model.edge_shared.detach()[indices].ne(0).any()):raise AssertionError('retired magnitude coordinates are not physically zero')
    return indices

def prepare():
    previous=study.read_plan()
    if PLAN.exists():raise FileExistsError('exact-edge protocol immutable')
    sources=[]
    for seed,expected_count in enumerate((20,8,36)):
        magnitude=next(x for x in previous['sources'] if x['seed']==seed and x['arm']=='magnitude')
        mag=source_trainer(magnitude['checkpoint'],'magnitude')
        if state_digest(mag.semantic_state())!=magnitude['endpoint_semantic_hash']:raise AssertionError('magnitude source semantic mismatch')
        ids=retired_identity(mag,expected_count);pairs=mag.model.pair_index[:,ids].T.tolist();del mag
        signed=next(x for x in previous['sources'] if x['seed']==seed and x['arm']=='signed_full')
        result_path=study.OUT/f'seed_{seed:03d}'/'signed_full'/'result.json';result=read(result_path)
        if result['result_hash']!=state_digest({k:v for k,v in result.items() if k!='result_hash'}):raise AssertionError('absence result digest')
        stages=[]
        for phase,tick in zip(PHASES,TICKS):
            cp=dict(path=signed['checkpoint'],sha256=signed['checkpoint_sha256'],semantic_hash=signed['endpoint_semantic_hash']) if phase=='initial' else next(c for c in result['checkpoints'] if c['phase']==phase)
            if sha(cp['path'])!=cp['sha256']:raise AssertionError('signed source checkpoint bytes differ')
            stages.append(dict(phase=phase,tick=tick,checkpoint=cp))
        sources.append(dict(seed=seed,retired_count=expected_count,pair_indices=ids,node_pairs=pairs,
            magnitude_source=magnitude,signed_source=signed,baseline_result=dict(path=str(result_path),sha256=sha(result_path)),stages=stages))
    plan=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),posthoc=True,
        timing='Selected after seeing small whole-system magnitude-minus-signed inherited absence/return differences. Frozen before these instantaneous exact-coordinate ablation results.',
        question='What is the immediate effect of silencing exactly the20/8/36 historically retired coordinates in the unpruned control, at each of the three actual inherited states?',
        source=dict(path=str(SOURCE),sha256=sha(SOURCE)),absence_protocol=dict(path=str(study.OUT/'protocol.json'),sha256=sha(study.OUT/'protocol.json')),
        sources=sources,seeds=[0,1,2],phases=list(PHASES),ticks=list(TICKS),families=[0,1,2,3],condition='short',steps=2048,
        intervention='Deepcopy each signed_full model, set precisely the fixed historical pair_indices edge_shared values to0 once before the fork. No new ranking, mask/eligibility editing, state editing, adaptation, training or threshold. Both symmetric physical directions disappear because Wuv=Wvu=edge_shared/sqrt(2).',
        identity='Actual magnitude147456 retirement ledger exact set=initial_mask minus endpoint mask;20/8/36, no births. Canonical node pairs also bound; same fixed identities at all time points.',
        baseline='Reuse36 SHA-bound signed_full short evaluations from absence-v1 result.records[phase]; verify actual source model/world/h hashes, full11 argmax,2048 steps, frozen/no-reset, original evaluator source pins.',
        evaluation='Detached autonomous eleven-action argmax using unchanged original evaluator and exact source world/h at each point. Historical h is retained, including prior influence of these edges. No source continuation is retrained or declared a valid newly pruned checkpoint.',
        metrics=dict(primary='family0 per-seed strict/ordinary completed/matured and paired differences in percentage points; idle actions/inactive opportunities and paired difference per1000',
            secondary='all four families separately; equal-family mean of paired rate differences within each seed, then mean/sample SD across three seeds',
            weights='Per selected coordinate actual signed shared weight and physical Wuv; abs mean/min/median/max, RMS, L1/L2, relative selected L2 to complete shared weight L2 at each time. Shared-coordinate L2 equals removed bidirectional matrix Frobenius norm.',
            denominators='Primary matured request counts/fingerprints must match each pair; actual idle opportunity denominators can change with own completion policy and both are saved.'),
        matching='source world/h and external scheduler/event/actor/case streams; primary and carryover matured request fingerprints; model copy differs only at fixed shared coordinates; actual source full semantic state unchanged',
        limits=['Instantaneous frozen forward ablation, not online retirement.','Estimates present-edge reliance given the unpruned inherited world/h; compares neither newly trained pruned systems nor a complete longitudinal mediator.','The two prior inherited systems had different training histories; this ablation fixes a common unpruned source per time point rather than attributing their total outcome difference.','Three seeds only; fixed edge identities selected from already observed magnitude exits, not a random subset.'],
        new_forks=36,reused_baseline_forks=36,runtime=dict(python=sys.executable,torch=torch.__version__,dtype='float64',cpu_threads=1,max_workers=3,environment={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    OUT.mkdir(parents=True,exist_ok=True);write(PLAN,plan);return plan

def check():
    plan=read(PLAN);study.read_plan()
    if plan['format']!=FORMAT or sha(SOURCE)!=plan['source']['sha256'] or sha(plan['absence_protocol']['path'])!=plan['absence_protocol']['sha256']:raise ValueError('edge/source protocol changed')
    for src in plan['sources']:
        if sha(src['baseline_result']['path'])!=src['baseline_result']['sha256']:raise ValueError('baseline result changed')
        for bound in (src['magnitude_source'],src['signed_source']):
            for key in ('checkpoint','result'):
                if sha(bound[key])!=bound[key+'_sha256']:raise ValueError('original inherited source changed')
        for stage in src['stages']:
            if sha(stage['checkpoint']['path'])!=stage['checkpoint']['sha256']:raise ValueError('signed point checkpoint changed')
    return plan

def silence(model,indices):
    if model.mode!='undirected' or model.edge_shared.dtype!=torch.float64:raise ValueError('FP64 undirected shared-coordinate model required')
    if indices!=sorted(set(indices)) or not indices or min(indices)<0 or max(indices)>=model.pair_count:raise ValueError('invalid fixed coordinate identity')
    before=state_digest(model.state_dict());clone=deepcopy(model);weights=model.edge_shared.detach()[indices].clone();all_weights=model.edge_shared.detach()
    with torch.no_grad():clone.edge_shared[indices]=0
    for key,value in model.state_dict().items():
        expected=value.clone()
        if key=='edge_shared':expected[indices]=0
        if not torch.equal(expected,clone.state_dict()[key]):raise AssertionError('silencing changed another model coordinate')
    if state_digest(model.state_dict())!=before:raise AssertionError('silencing changed source model')
    absolute=weights.abs();norm=float(weights.norm());total=float(all_weights.norm());u,v=model.pair_index[:,indices];physical=model.recurrent_matrix().detach()[u,v]
    stats=dict(count=len(indices),shared_values=weights.tolist(),physical_Wuv_values=physical.tolist(),
        abs_min=float(absolute.min()),abs_median=float(torch.quantile(absolute,.5)),abs_mean=float(absolute.mean()),abs_max=float(absolute.max()),
        rms=float(weights.square().mean().sqrt()),l1=float(absolute.sum()),l2=norm,
        removed_bidirectional_matrix_frobenius=norm,source_complete_shared_l2=total,selected_l2_fraction=norm/total if total else None,
        source_nonzero_selected=int(weights.ne(0).sum()),new_model_sha256=state_digest(clone.state_dict()),source_model_sha256=before)
    return clone,stats

def canonical(value):return json.loads(json.dumps(value,allow_nan=False))

def verify_baseline(base,t,tick):
    if base['condition']!='short' or base['steps']!=2048 or base['probe']['tick']!=tick:raise AssertionError('baseline budget/spec differs')
    hashes=dict(parameter_hash=state_digest(t.model.state_dict()),start_world_state_hash=state_digest(t.world.state_dict()),start_hidden_hash=state_digest({'hidden':t.h}),
        source_continuation_hash=state_digest(dict(model=t.model.state_dict(),world=t.world.state_dict(),hidden=t.h)))
    if any(base[k]!=v for k,v in hashes.items()):raise AssertionError('baseline source model/world/h mismatch')
    if base['action_hash']!=base['model_action_hash'] or base['intervention']!='none' or any(base[k] for k in ('world_resets','hidden_resets','parameter_updates')):raise AssertionError('baseline not unforced autonomous frozen')
    for path,expected in base['source_hashes'].items():
        if sha(ROOT/path)!=expected:raise AssertionError('baseline frozen execution source changed')

def rates(record):
    p,o=record['primary'],record['opportunities']
    return dict(strict_pp=100*p['clean_completed']/p['matured'],ordinary_pp=100*p['completed']/p['matured'],idle_per1000=1000*o['inactive_actions']/o['inactive_opportunities'])

def run(seed):
    plan=check();src=next(s for s in plan['sources'] if s['seed']==seed);out=OUT/f'seed_{seed:03d}';out.mkdir(exist_ok=False);started=time.perf_counter();records=[];weight_rows=[]
    mag=source_trainer(src['magnitude_source']['checkpoint'],'magnitude')
    if retired_identity(mag,src['retired_count'])!=src['pair_indices'] or mag.model.pair_index[:,src['pair_indices']].T.tolist()!=src['node_pairs']:raise AssertionError('retirement identity binding failed')
    del mag
    saved=read(src['baseline_result']['path'])
    for stage in src['stages']:
        phase=stage['phase'];cp=stage['checkpoint'];container=source_trainer(cp['path'],'signed_full') if phase=='initial' else AbsenceContinuation.from_checkpoint(cp['path'])
        t=container if phase=='initial' else container.trainer;before=state_digest(container.semantic_state())
        if before!=cp['semantic_hash'] or t.ticks!=stage['tick'] or t.config.dtype!='float64':raise AssertionError('signed complete source binding failed')
        if int(t.operator.mask.sum())!=640 or not bool(t.operator.mask[src['pair_indices']].all()) or int(t.model.edge_shared.detach().ne(0).sum())!=640:raise AssertionError('unpruned control is not actual640 support')
        if t.model.pair_index[:,src['pair_indices']].T.tolist()!=src['node_pairs']:raise AssertionError('canonical pair mapping changed')
        copy_model,stats=silence(t.model,src['pair_indices']);stats.update(seed=seed,phase=phase,tick=t.ticks,pair_indices=src['pair_indices'],node_pairs=src['node_pairs']);weight_rows.append(stats)
        for family in plan['families']:
            base=next(x for x in saved['records'][phase] if x['family']==family and x['condition']=='short');verify_baseline(base,t,t.ticks)
            with _preserve_process_state():actual=evaluate(copy_model,t.world,t.h,family=family,condition='short',steps=2048,device='cpu')
            actual['probe']=dict(tick=t.ticks,family=family,condition='short',steps=2048)
            for key in ('case_ids','actor_ids','fork_tick','start_world_state_hash','start_hidden_hash','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash','initial_eligible_pending'):
                if canonical(actual[key])!=base[key]:raise AssertionError('unmatched ablation source/cases/external stream '+key)
            for cohort in ('primary','carryover_contract'):
                if actual[cohort]['request_fingerprint']!=base[cohort]['request_fingerprint'] or actual[cohort]['matured']!=base[cohort]['matured']:raise AssertionError('unmatched matured ablation cases')
            if any(actual[k] for k in ('world_resets','hidden_resets','parameter_updates')) or actual['action_hash']!=actual['model_action_hash']:raise AssertionError('biased frozen ablation')
            ar,br=rates(actual),rates(base);records.append(dict(seed=seed,phase=phase,tick=t.ticks,family=family,baseline=base,ablated=actual,delta={k:ar[k]-br[k] for k in ar},baseline_rates=br,ablated_rates=ar))
            write(out/'partial.json',dict(seed=seed,records=records,weights=weight_rows));print(json.dumps(dict(seed=seed,phase=phase,family=family,done=len(records),seconds=time.perf_counter()-started)),flush=True)
        if state_digest(container.semantic_state())!=before:raise AssertionError('ablation changed actual full source continuation')
        del container,t,copy_model
    check();value=dict(format=FORMAT,seed=seed,protocol_sha256=sha(PLAN),records=records,weights=weight_rows,source_unchanged=True,new_forks=12,reused_forks=12,seconds=time.perf_counter()-started)
    write(out/'result.json',value);return value

def summary():
    plan=check();rows=[];weights=[];bindings=[]
    for seed in plan['seeds']:
        path=OUT/f'seed_{seed:03d}'/'result.json';value=read(path)
        if value['protocol_sha256']!=sha(PLAN) or len(value['records'])!=12 or not value['source_unchanged']:raise AssertionError('incomplete edge identity result')
        bindings.append(dict(path=str(path),sha256=sha(path)));weights.extend(value['weights'])
        for r in value['records']:
            counts={}
            for arm in ('baseline','ablated'):
                p,o=r[arm]['primary'],r[arm]['opportunities'];counts[arm]=dict(matured=p['matured'],strict_completed=p['clean_completed'],ordinary_completed=p['completed'],idle_actions=o['inactive_actions'],idle_opportunities=o['inactive_opportunities'])
            rows.append({k:r[k] for k in ('seed','phase','tick','family','delta','baseline_rates','ablated_rates')}|dict(counts=counts))
    aggregates=[]
    for phase in PHASES:
        for group in ('family0','all4_equal_family'):
            per_seed=[]
            for seed in plan['seeds']:
                selected=[x for x in rows if x['seed']==seed and x['phase']==phase and (group!='family0' or x['family']==0)]
                per_seed.append(dict(seed=seed,delta={k:statistics.mean(x['delta'][k] for x in selected) for k in selected[0]['delta']}))
            aggregates.append(dict(phase=phase,scope=group,per_seed=per_seed,delta={k:dict(mean=statistics.mean(x['delta'][k] for x in per_seed),sample_sd=statistics.stdev(x['delta'][k] for x in per_seed)) for k in per_seed[0]['delta']}))
    value=dict(format=FORMAT,protocol_sha256=sha(PLAN),rows=rows,weights=weights,aggregates=aggregates,result_bindings=bindings,posthoc=True,new_forks=36,reused_forks=36)
    write(OUT/'summary.json',value)
    import csv
    flat=[]
    for row in rows:
        flat.append({k:row[k] for k in ('seed','phase','tick','family')}|{a+'_'+k:v for a in ('baseline','ablated') for k,v in row['counts'][a].items()}|{'delta_'+k:v for k,v in row['delta'].items()})
    with (OUT/'paired.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','summary'));p.add_argument('--seed',type=int,choices=(0,1,2));args=p.parse_args()
    value=prepare() if args.command=='prepare' else run(args.seed) if args.command=='run' else summary();print(json.dumps(dict(format=value['format'],output=str(OUT))),flush=True)
