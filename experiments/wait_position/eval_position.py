"""Frozen wrapper over verified autonomous worlds and eleven-action scoring."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.review_selection import eval_review as prior
from experiments.experience_transfer import eval_transfer as base
OUT=ROOT/'data/wait-position/eval-v1'
TRAIN=ROOT/'data/wait-position/train-v1'
PLAN=OUT/'protocol.json'
FORMAT='wait-position-autonomous-eval-v1'
ARMS=('midpoint','hard','random','uniform','no_review')
NEW_ARMS=ARMS[:3]
SNAPSHOTS=base.SNAPSHOTS
MIXED=prior.MIXED
sha,read,state_digest=base.sha,base.read,base.state_digest
scenarios=prior.scenarios

def json_metadata(value):
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {str(k):json_metadata(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_metadata(v) for v in value]
    return value

def write(path,value):
    """Write once, or require exact content if resuming completed evidence."""
    path=Path(path)
    if path.exists():
        assert read(path)==json_metadata(value),('existing evidence differs',str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(json_metadata(value),stream,indent=2,allow_nan=False);stream.write('\n')

def result_path(seed,arm,snapshot,scenario):
    return OUT/f'seed_{seed:03d}'/f'{arm}_{snapshot}_{scenario}.json'

def checkpoint_path(seed,arm,snapshot):
    return (TRAIN/f'seed_{seed:03d}'/arm/f'{snapshot}.pt' if arm in NEW_ARMS else prior.checkpoint_path(seed,arm,snapshot))

def checked():
    plan=read(PLAN)
    assert plan['protocol_hash']==state_digest({k:v for k,v in plan.items() if k!='protocol_hash'})
    prior.checked()
    for path,expected in plan['frozen_files'].items():assert sha(path)==expected,('frozen dependency changed',path)
    return plan

def prepare():
    assert not PLAN.exists()
    old=prior.checked()
    files=[Path(__file__).resolve(),Path(__file__).with_name('eval_timing.py'),Path(__file__).with_name('eval_report.py'),Path(__file__).with_name('eval_finish.py'),
           Path(__file__).with_name('eval_position_test.py'),prior.PLAN,prior.OUT/'completion.json',prior.OUT/'audit.json']
    for seed in range(3):
        files.append(base.OUT/f'common_seed_{seed:03d}_related.pt')
        for arm in ('uniform','no_review'):
            for snapshot in SNAPSHOTS:
                files.append(checkpoint_path(seed,arm,snapshot))
                for scenario in scenarios(snapshot):
                    path=prior.result_path(seed,arm,snapshot,scenario);row=read(path)
                    files.append(path);files.extend(Path(p) for p in row['artifacts'])
        for scenario in ('B','mixed','history'):
            ref=base.reference(seed,'related',scenario)
            files.extend([base.OUT/f'reference_seed_{seed:03d}_related_{scenario}.json',Path(ref['external_artifact'])])
    plan=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen before reading any new WAIT-placement autonomous outcomes. All arms, seeds and snapshots retained.',
        arms=list(ARMS),new_arms=list(NEW_ARMS),seeds=[0,1,2],snapshots=list(SNAPSHOTS),mixed_snapshots=list(MIXED),
        expected_records=210,steps=old['steps'],batch=old['batch'],spec=old['spec'],
        source_protocol_hash=old['protocol_hash'],frozen_files={str(p):sha(p) for p in dict.fromkeys(files)},
        autonomous='Unmodified experience_transfer rollout, native11-action logits.argmax(-1), no gate, no learning. Same1024 public WAIT warmup and2048 long heldout future events/request identities at every checkpoint.',
        tests='B-only at B0/first2/first8/B1/B2/B4/B8; random-per-NEW/job mixed A/B at B0/B1/B2/B4/B8, A and B reported separately; auxiliary A-only at B0/B8. Random mixing is not strict alternation.',
        baselines='Old uniform and no_review results bound to unchanged raw evidence. Any new checkpoint, including midpoint, may reuse an old result only when its actual model state hash matches exactly. Otherwise run its actual frozen weights.',
        metrics=old['metrics'],warmup=old['warmup'],scorer=old['scorer'],
        timing_denominator='All eligible postfork requests in evaluation regime that mature within rollout; inherited and censored excluded. Diagnose both A and B and each conflicting raw separately at every mixed checkpoint.',
        timing_legal='Original pre-action world active & started & ready_time-eps <= now <= deadline+eps. First FINISH inside this actual window is classified correct version/content, other-context content, or other error. Retain no-BEGIN, no-window, and window-but-no-FINISH separately.',
        timing_early='Actual early FINISH requires original world active & started & now < ready_time-eps, irrespective of action version/content. Earliest active rejection is stored with all overlapping flags; primary exclusive category puts this actual-early flag first. Thus wrong-version early FINISH cannot disappear into wrong-version bins. Report both all matured and completed-but-nonstrict cohorts, early lead=ready_time-now in seconds.',
        analysis='Equal weight mean and sample SD of3 seed rates; per-seed integer numerators/denominators retained. Mixed A cannot be hidden by B success. Idle actions per1000 inactive enabled slot opportunities, retained raw/family. No pooled rates replace equal-weight means.',
        budget='Bind every actual checkpoint work record and serialized family_new_counts(B)-originalB0, including no-loss boundary. Require equal real B exposures at matched seed/snapshot. Training audits separately establish candidate/request/gradient budgets.',
        audit='Reconstruct base completion/strict/idle and timing request rows from saved raw arrays; verify receipts, logits argmax, actual-world timings, common future fields, model/checkpoint/artifact hashes, and matched matured identities.',
        limits=['Supervised prototype; three fixed seeds and one heldout external stream per seed.', 'Common WAIT warmup is an explicit matched deployment choice; continuous hidden/world state within each fork.', 'Timing analysis describes actual behavior; it does not prove a semantic association is erased.'])
    plan['protocol_hash']=state_digest(plan);write(PLAN,plan)
    for seed in range(3):
        for scenario in ('B','mixed','history'):
            ref=deepcopy(base.reference(seed,'related',scenario))
            ref.update(wait_position_protocol_hash=plan['protocol_hash'])
            write(OUT/f'reference_seed_{seed:03d}_{scenario}.json',ref)
    return plan

def bind_checkpoint(row,path,cp,seed,arm,snapshot,scenario,plan):
    ss=plan['spec'][str(seed)]
    origin=torch.load(prior.checkpoint_path(seed,'no_review','B000'),map_location='cpu',weights_only=False)
    assert cp['mapping_table']==ss['mapping_table'] and cp['transfer']['seed']==seed
    assert cp['world']['spec']['seed']==origin['world']['spec']['seed']
    assert cp['stage']=='B' and cp['round']==(1 if snapshot.startswith('Bfirst') else int(snapshot[1:]))
    if snapshot=='B000':
        for key in ('model','optimizer','world','hidden'):assert state_digest(cp[key])==state_digest(origin[key]),(seed,arm,key)
    assert row['model_hash']==state_digest(cp['model'])
    actual=prior.actual_B_new(cp,origin,ss['B_family'])
    if 'B_actual_NEW_including_boundary' in cp['work'] and snapshot!='B000':assert cp['work']['B_actual_NEW_including_boundary']==actual
    row.update(format=FORMAT,protocol_hash=plan['protocol_hash'],seed=seed,arm=arm,snapshot=snapshot,scenario=scenario,
        B_round=0 if snapshot.startswith('Bfirst') else int(snapshot[1:]),checkpoint_path=str(path.resolve()),checkpoint_sha256=sha(path),
        checkpoint_work=json_metadata(cp['work']),checkpoint_transfer=json_metadata(cp.get('transfer')),
        checkpoint_wait_position=json_metadata(cp.get('wait_position')),checkpoint_review_selection=json_metadata(cp.get('review_selection')),
        actual_unique_B_NEW=actual,work_B_unique_new_commands=cp['work'].get('B_unique_new_commands'),
        mapping_table=ss['mapping_table'],history_family=ss['history_family'],B_family=ss['B_family'])
    return row

@torch.inference_mode()
def run_checkpoint(seed,arm,snapshot):
    plan=checked();path=checkpoint_path(seed,arm,snapshot)
    model,_,_,cp=base.training_api().load_checkpoint(path,device='cpu');model.eval()
    ss=plan['spec'][str(seed)];table,hf,bf=ss['mapping_table'],ss['history_family'],ss['B_family']
    shared=base.common(seed,table,hf,bf,arm='related');hidden=None
    for scenario in scenarios(snapshot):
        target=result_path(seed,arm,snapshot,scenario)
        if target.exists():continue
        candidates=[prior.result_path(seed,arm,snapshot,scenario)] if arm not in NEW_ARMS else [prior.result_path(seed,'uniform',snapshot,scenario),prior.result_path(seed,'no_review',snapshot,scenario)]
        source=next((p for p in candidates if p.exists() and read(p)['model_hash']==state_digest(cp['model'])),None)
        if source is not None:
            row=deepcopy(read(source));assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
            row.update(evidence_reused=True,original_result_path=str(source),original_result_sha256=sha(source))
        else:
            if hidden is None:hidden=base.original.warmup(model,shared)
            families={'B':(bf,),'mixed':(hf,bf),'history':(hf,)}[scenario];started=time.perf_counter()
            with base.training_api().mapping_scope(table):
                world=base.ContextWorld.from_state_dict(shared['world'],device='cpu')
                row,frames,states=base.rollout(model,world,hidden,families=families,steps=plan['steps'],table=table,
                    history_family=hf,b_family=bf,raw_groups=ss['raw_groups'])
            stem=target.with_suffix('');stem.parent.mkdir(parents=True,exist_ok=True)
            for ext in ('.npz','.pt','.requests.json'):assert not stem.with_suffix(ext).exists(),('partial evidence exists',stem,ext)
            requests=dict(matured=row.pop('primary_requests'),actions=row.pop('action_requests'))
            np.savez_compressed(stem.with_suffix('.npz'),**frames);torch.save(states,stem.with_suffix('.pt'));write(stem.with_suffix('.requests.json'),requests)
            row.update(evidence_stem=str(stem),evidence_reused=False,warmup_observation_hash=shared['observation_hash'],
                elapsed_seconds=time.perf_counter()-started,artifacts={str(stem.with_suffix(ext)):sha(stem.with_suffix(ext)) for ext in ('.npz','.pt','.requests.json')})
        expected=base.reference(seed,'related',scenario)
        for key in ('external_digest','primary_identity_hash','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash'):assert row[key]==expected[key],(seed,arm,snapshot,scenario,key)
        bind_checkpoint(row,path,cp,seed,arm,snapshot,scenario,plan);write(target,row)
        print(json.dumps(dict(event='wait_position_eval',seed=seed,arm=arm,snapshot=snapshot,scenario=scenario,
                             reused=row['evidence_reused'],actual_unique_B_NEW=row['actual_unique_B_NEW'])),flush=True)

def available(seed=None,watch=False):
    torch.set_num_threads(1)
    pending=[(s,a,t) for s in ([seed] if seed is not None else range(3)) for a in ARMS for t in SNAPSHOTS]
    while pending:
        progressed=False
        for s,a,t in list(pending):
            if all(result_path(s,a,t,q).exists() for q in scenarios(t)):pending.remove((s,a,t));continue
            path=checkpoint_path(s,a,t)
            if path.exists() and path.with_suffix('.json').exists():
                run_checkpoint(s,a,t);pending.remove((s,a,t));progressed=True
        if not watch or not pending:break
        if not progressed:time.sleep(5)
    return dict(pending=pending,complete=not pending)

def records(seed=None):
    return [read(p) for p in sorted(OUT.glob('seed_*/*.json')) if not p.name.endswith('.requests.json') and (seed is None or p.parent.name==f'seed_{seed:03d}')]

def audit(seed=None,require_complete=False):
    plan=checked();rows=records(seed);expected=210 if seed is None else 70
    if require_complete:assert len(rows)==expected
    matched={};exposures={};audited={};cp_cache={}
    from experiments.wait_position import eval_timing
    timing=[]
    for row in rows:
        assert row['format']==FORMAT and row['protocol_hash']==plan['protocol_hash']
        path=Path(row['checkpoint_path']);assert sha(path)==row['checkpoint_sha256']
        if str(path) not in cp_cache:cp_cache[str(path)]=torch.load(path,map_location='cpu',weights_only=False)
        rebound=bind_checkpoint(deepcopy(row),path,cp_cache[str(path)],row['seed'],row['arm'],row['snapshot'],row['scenario'],plan)
        for key in ('checkpoint_work','actual_unique_B_NEW','checkpoint_wait_position','checkpoint_review_selection'):assert rebound[key]==row[key]
        if row['evidence_reused']:
            assert sha(row['original_result_path'])==row['original_result_sha256'];source=read(row['original_result_path'])
            for key in ('model_hash','counts','metrics','strata','by_family_raw','artifacts'):assert source[key]==row[key]
        for artifact,digest in row['artifacts'].items():assert sha(artifact)==digest
        stem=Path(row['evidence_stem'])
        if str(stem) not in audited:
            states=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=False)
            with np.load(stem.with_suffix('.npz')) as saved:frames={k:saved[k] for k in saved.files}
            assert np.array_equal(frames['actions'],frames['logits'].argmax(-1))
            for which,prefix in (('initial','start'),('end','end')):
                assert state_digest(states[which+'_world'])==row[prefix+'_world_hash']
                assert state_digest(states[which+'_h'])==row[prefix+'_h_hash']
            rebuilt=base.score(frames,states['initial_world'],table=row['mapping_table'],history_family=row['history_family'],
                b_family=row['B_family'],fork_tick=row['fork_tick'],regime_id=row['evaluation_regime_id'],raw_groups=row['raw_groups'])
            for key in ('metrics','strata','by_family_raw','primary_identity_hash'):assert rebuilt[key]==row[key],(stem,key)
            for key,value in rebuilt['counts'].items():assert row['counts'][key]==value
            requests=read(stem.with_suffix('.requests.json'))
            assert rebuilt['primary_requests']==requests['matured'] and rebuilt['action_requests']==requests['actions']
            ref=base.reference(row['seed'],'related',row['scenario'])
            with np.load(ref['external_artifact']) as external:
                for key in base.EXTERNAL_FIELDS:assert np.array_equal(frames[key],external[key]),(stem,key)
            audited[str(stem)]=(frames,states,requests) if row['scenario']=='mixed' else None
        if row['scenario']=='mixed':
            frames,states,requests=audited[str(stem)]
            diagnostic=eval_timing.analyze(row,frames,states,requests)
            diagnostic_path=OUT/'timing'/f"seed_{row['seed']:03d}"/f"{row['arm']}_{row['snapshot']}.json"
            write(diagnostic_path,diagnostic);timing.append(dict(path=str(diagnostic_path),sha256=sha(diagnostic_path)))
        match=row['seed'],row['scenario'];signature=tuple(row[k] for k in ('external_digest','primary_identity_hash','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash','warmup_observation_hash'))
        if match in matched:assert matched[match]==signature
        matched[match]=signature
        key=row['seed'],row['snapshot']
        if key in exposures:assert exposures[key]==row['actual_unique_B_NEW'],('unequal B exposure',key)
        exposures[key]=row['actual_unique_B_NEW']
    value=dict(format=FORMAT,verified=True,complete=len(rows)==expected,records=len(rows),seed=seed,
        protocol_hash=plan['protocol_hash'],unique_evidence_forks=len(audited),matched_external_groups=len(matched),
        timing_artifacts=timing,checks=['frozen sources and unchanged baseline evidence','actual model/checkpoint metadata and serialized B NEW exposure',
            'eleven-action argmax and semantic receipts','base raw completion/strict/idle reconstruction',
            'same external future and matured identities','actual active/started/before-ready early FINISH and legal-window first actual actions'])
    write(OUT/('audit.json' if seed is None else f'audit_seed_{seed:03d}.json'),value);return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','available','watch','audit','report'))
    p.add_argument('--seed',type=int,choices=range(3));p.add_argument('--complete',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    if args.command=='prepare':value=prepare();print(json.dumps(dict(protocol_hash=value['protocol_hash'],expected_records=value['expected_records'])))
    elif args.command in ('available','watch'):print(json.dumps(available(args.seed,args.command=='watch')))
    elif args.command=='audit':print(json.dumps(audit(args.seed,args.complete)))
    else:
        from experiments.wait_position import eval_report
        eval_report.run()
if __name__=='__main__':main()
