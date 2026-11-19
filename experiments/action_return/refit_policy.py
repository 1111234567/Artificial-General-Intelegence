"""One matched post-error policy-data refit of the frozen 24-state controller.

Reuse the original feature flow, two fitting algorithms, group-first combined
controller, and read-only autonomous scorer. No teacher action is executed in
the on-policy collection arm. This single supplement adds no threshold search.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiments.action_return import readout as b, finish_readout as f
from experiments.action_return.absence_study import available_memory
import torch

OUT=b.ROOT/'data/action-return/refit-policy-v1'
PLAN=OUT/'protocol.json'
ARMS=('assisted_archive','own_policy')
MID=139264


def load_gzip(path):
    return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)


def prepare():
    if PLAN.exists():raise FileExistsError('immutable supplement protocol exists')
    previous,parent=f.checked();verified=b.old.read(b.old.RAW/'verification.json');inputs={};seeds={}
    for seed in parent['seeds']:
        src=parent['sources'][str(seed)];folder=b.OUT/f'seed_{seed:03d}'
        audit=sorted(Path(src['audits'][0]['path']).parent.glob('*.pt.gz'))
        if len(audit)!=128:raise ValueError('need the exact original128 blocks')
        for path in audit:
            sha=b.old.sha(path)
            if verified['files'][str(path)]!=sha:raise ValueError('unverified assisted trajectory')
            inputs[str(path)]=sha
        for path in [folder/'train_dataset.pt.gz',folder/'fit.json',folder/'fits.pt',
                     f.ROOT/f'seed_{seed:03d}'/'model.pt',f.ROOT/f'seed_{seed:03d}'/'result.json']:
            inputs[str(path)]=b.old.sha(path)
        seeds[str(seed)]=dict(audits=[str(p) for p in audit],source=src,
                             original_fit=str(folder/'fit.json'),dataset=str(folder/'train_dataset.pt.gz'),
                             fits=str(folder/'fits.pt'),collector_model=str(f.ROOT/f'seed_{seed:03d}'/'model.pt'),
                             anchor=str(f.ROOT/f'seed_{seed:03d}'/'result.json'))
    files=[Path(__file__).resolve(),Path(b.__file__).resolve(),Path(f.__file__).resolve()]
    plan=dict(format='matched-policy-data-refit-prospective-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Single supplement chosen after observing ordinary completion with high idle actions and poor strict completion in the first combined controller. No result from this appended comparison was read before this protocol freeze. No further arms or threshold choices.',
        source_files={str(p):b.old.sha(p) for p in files},input_files=inputs,seeds=seeds,
        previous_protocols={str(b.PLAN):b.old.sha(b.PLAN),str(f.PLAN):b.old.sha(f.PLAN)},
        collection=dict(shared_prefix='Exactly original assisted offsets0..63 from CP131072; actual changing historical model actions, teacher labels, observations and world endpoints verified. Endpoint147456-frozen backbone continuously replays the same observations from actual CP131072 h.',
            midpoint_tick=MID,midpoint='Actual assisted world after first64 blocks; continuous frozen-backbone h must match the saved original readout fit reconstruction hash. Original changing-model h separately verified against block64 start_h.',
            assisted_archive='Replay exact original assisted offsets64..127 with archived executed actions and midpoint labels; features continue the same frozen backbone h, never reset.',
            own_policy='From identical midpoint world and frozen-backbone h, execute the already fitted non_input24 BEGIN+conditionalFINISH eleven-way argmax for64x128 ticks. Midpoint teacher labels are read only and never intervene. Original combined model state must match saved model.pt exactly.',
            actors='Run full actual8-actor world; fit only actors0..3 as before.',
            equality='Both added halves32768 positions per job, same8192 ticks and actors0..3; external scheduler/event hashes, command exposures and request identities must match.',
            persistence='Save collected observations, own executed actions, teacher labels, continuous frozen features, request sets and per-block causal endpoint hashes.'),
        fit=dict(dataset='Same old first32768+new/archive32768=65536 positions per seed/job in each matched arm.',
            algorithm='Direct readout.fit plus finish_readout.train_finish; same non_input24 features, original WAIT=.1/nonWAIT=1 weights, train-only population scaling, zero initialization, slopeL2=.001 and original LBFGS settings.',
            optimizer=parent['train']['optimizer'],conditional='Keep teacher non-BEGIN rows for9-way WAIT+FINISH exactly as original. Report resulting filtered row counts and actual iterations; equal data opportunity budget and identical optimizer caps, not forced equal convergence iterations.'),
        evaluation=dict(source='Common actual147456 complete endpoint h/world and endpoint-frozen backbone, not either collection endpoint h/world.',
            primary_actors=[4,5,6,7],overlap_actors=[0,1,2,3],families=[0,1,2,3],conditions=['short','long_both'],steps=2048,
            new_forks=48,anchor_forks=24,anchor='SHA-bound original combined non_input24 unrefit controller forks; anchor has half the training positions and is descriptive, not the budget-matched contrast.',
            policy='Unmodified existing combined group-first BEGIN/conditionalFINISH controller, full11 argmax. No teacher, private state gates, target forcing or parameter updates at deployment.',
            matching='Exact primary matured request identities and external scheduler/event streams across both refits and unrefit anchor; future test requests disjoint from all fit requests.',
            metrics=['strict/ordinary completion per primary matured','feasible accepted BEGIN per primary matured','all idle actions per actual inactive opportunity','false idle BEGIN, duplicate BEGIN and total BEGIN/FINISH attempts','legally timed FINISH and accepted FINISH'],
            multiple_attempts='Report attempts per matured and accepted FINISH, duplicate BEGIN, idle action counts and strict completion so ordinary completion is not presented as clean execution.'),
        runtime=dict(workers=3,cpu_threads=1,dtype='float64',ram_gate_GiB=5),
        limits=['One externally supervised policy-data correction, not native recurrent learning.',
                'Only readout fitting holds actors4..7 out; original backbone training saw all8 actors.',
                'On-policy labels and receipts change together; comparison tests the collected training distribution as a whole, not one isolated feature.',
                'Three paired seeds, fixed optimizer settings and one supplement; no selection by results.'])
    plan=json.loads(json.dumps(plan));plan['protocol_hash']=b.state_digest(plan)
    OUT.mkdir(parents=True,exist_ok=True);b.old.write(PLAN,plan);return plan


def checked():
    p=b.old.read(PLAN);f.checked()
    if p['protocol_hash']!=b.state_digest({k:v for k,v in p.items() if k!='protocol_hash'}):raise ValueError('refit protocol digest')
    for group in ('source_files','input_files','previous_protocols'):
        for path,sha in p[group].items():
            if b.old.sha(path)!=sha:raise ValueError('bound input/source changed '+path)
    return p


def empty_rows():
    return {j:dict(x=[],label=[],action=[],request=[]) for j in range(2)}


def append_rows(rows,h,labels,actions,world,indices):
    for j,row in rows.items():
        row['x'].append(h[:4,indices].clone());row['label'].append(labels[:4,j].clone());row['action'].append(actions[:4,j].clone())
        row['request'].extend((world.case_ids[i],j,int(world.request_id[i,j])) for i in range(4) if int(world.request_id[i,j])>=0)


def finish_rows(rows):
    for row in rows.values():
        for key in ('x','label','action'):row[key]=torch.cat(row[key])
        row['request']=sorted(set(row['request']))
    return rows


def world_record(world):
    saved=world.state_dict()
    return dict(world_hash=b.state_digest(saved),scheduler_hash=b.state_digest(saved['scheduler']),
                event_hash=world.event_hash,family_exposures=world.family_exposures(),tick=world.tick,
                case_ids=list(world.case_ids),actor_ids=list(world.actor_ids))


def replay_archive(original,frozen,actual_h,frozen_h,paths,indices):
    rows=empty_rows();trace=[];world=original.world
    with torch.no_grad():
        for path in paths:
            audit=load_gzip(path);record=audit['record']
            if not torch.equal(actual_h,record['start_h']):raise ValueError('historical actual h start mismatch')
            original.model.load_state_dict({**record['parameters'],**record['buffers']},strict=True)
            for t,(obs,dt) in enumerate(zip(record['observations'],record['elapsed'])):
                actual_obs=world.observe().double();labels=world.oracle('midpoint')
                if not torch.equal(actual_obs,obs) or not torch.equal(labels,record['targets'][t]):raise ValueError('archive observations/labels mismatch')
                logits,actual_h=original.model.step(obs,actual_h,dt)
                if not torch.equal(logits.argmax(-1),audit['actions']['model_actions'][t]):raise ValueError('archive native actions mismatch')
                _,frozen_h=frozen.step(obs,frozen_h,dt);actions=audit['actions']['actual_actions'][t]
                append_rows(rows,frozen_h,labels,actions,world,indices);world.act(actions)
            if not torch.equal(actual_h,record['end_h']) or b.state_digest(world.state_dict())!=audit['trace']['live_world_hash']:raise ValueError('archive actual endpoint mismatch')
            trace.append(dict(path=path,**world_record(world),frozen_h_hash=b.state_digest(frozen_h)))
    return finish_rows(rows),actual_h,frozen_h,trace


def collect_own(world,h,model,indices):
    rows=empty_rows();trace=[];observations=[];labels_log=[];actions_log=[];start_h=h.clone();initial=world_record(world)
    with torch.no_grad():
        for block in range(64):
            for tick in range(128):
                obs=world.observe().double();labels=world.oracle('midpoint');logits,h=model.step(obs,h,float(world.elapsed));actions=logits.argmax(-1)
                append_rows(rows,h,labels,actions,world,indices)
                observations.append(obs);labels_log.append(labels.clone());actions_log.append(actions.clone());world.act(actions)
            trace.append(dict(block=block,**world_record(world),frozen_h_hash=b.state_digest(h)))
            if (block+1)%16==0:print('collected own blocks',block+1,flush=True)
    return finish_rows(rows),dict(initial=initial,trace=trace,start_h=start_h,end_h=h.clone(),
        observations=torch.stack(observations),labels=torch.stack(labels_log),actual_actions=torch.stack(actions_log))


def fit_model(endpoint,groups,indices,options):
    begins=[];finishes=[];infos=[]
    for j in range(2):
        row=groups[j];bf,bi=b.fit(row['x'],row['label'],dict(train=dict(optimizer=options)))
        ff,fi=f.train_finish(row['x'],row['label'],options);begins.append(bf);finishes.append(ff)
        infos.append(dict(job=j,begin=bi,finish=fi,executed_action_counts=torch.bincount(row['action'],minlength=11).tolist(),
                          teacher_label_counts=torch.bincount(row['label'],minlength=11).tolist(),positions=len(row['label'])))
    return f.FinishController(b.BeginController(endpoint.model,'non_input24',begins,indices),finishes),infos


def run(seed):
    torch.set_num_threads(1);plan=checked();src=plan['seeds'][str(seed)];out=OUT/f'seed_{seed:03d}';out.mkdir(exist_ok=False)
    started=time.perf_counter();endpoint=b.native.NativeTrainer.from_checkpoint(src['source']['endpoint']['path'],device='cpu')
    endpoint_hash=b.state_digest(endpoint.semantic_state())
    if endpoint_hash!=src['source']['endpoint']['semantic_hash']:raise ValueError('wrong actual evaluation endpoint')
    tables=load_gzip(src['dataset']);indices=tables['non_input'];old=deepcopy(tables['groups'])
    original=b.native.NativeTrainer.from_checkpoint(src['source']['start']['path'],device='cpu')
    if b.state_digest(original.semantic_state())!=src['source']['start']['semantic_hash']:raise ValueError('wrong original start')
    prefix,actual_h,frozen_h,prefix_trace=replay_archive(original,endpoint.model,original.h.clone(),original.h.clone(),src['audits'][:64],indices)
    for j in range(2):
        if not torch.equal(prefix[j]['x'],old[j]['h'][:,indices]) or not torch.equal(prefix[j]['label'],old[j]['label']):raise ValueError('saved feature table reconstruction differs')
    reconstruction=b.old.read(src['original_fit'])['reconstruction']
    if b.state_digest(frozen_h)!=reconstruction['frozen_feature_end_h_hash']:raise ValueError('common midpoint frozen h hash differs')
    next_audit=load_gzip(src['audits'][64])
    if next_audit['record']['start_tick']!=MID or not torch.equal(actual_h,next_audit['record']['start_h']):raise ValueError('wrong actual block64 midpoint')
    midpoint=world_record(original.world);midpoint.update(frozen_h_hash=b.state_digest(frozen_h),actual_h_hash=b.state_digest(actual_h))
    if midpoint['tick']!=MID-1 or midpoint['world_hash']!=prefix_trace[-1]['world_hash']:raise ValueError('wrong midpoint world')
    own_world=b.ContextWorld.from_state_dict(original.world.state_dict(),device='cpu');own_h=frozen_h.clone()
    old_begin=torch.load(src['fits'],weights_only=False,map_location='cpu')['fits']['non_input24']
    saved_model=torch.load(src['collector_model'],weights_only=False,map_location='cpu')
    old_finish=[{k:saved_model[f'j{j}_{k}'].clone() for k in ('mean','std','coef','bias')} for j in range(2)]
    collector=f.FinishController(b.BeginController(endpoint.model,'non_input24',old_begin,indices),old_finish)
    if b.state_digest(collector.state_dict())!=b.state_digest(saved_model):raise ValueError('collector differs from original combined controller')
    archive,actual_end,frozen_end,archive_trace=replay_archive(original,endpoint.model,actual_h,frozen_h,src['audits'][64:],indices)
    if not torch.equal(actual_end,endpoint.h) or b.state_digest(original.world.state_dict())!=b.state_digest(endpoint.world.state_dict()):raise ValueError('original assisted actual147456 endpoint differs')
    own,trajectory=collect_own(own_world,own_h,collector,indices)
    for a,c in zip(archive_trace,trajectory['trace']):
        for key in ('scheduler_hash','event_hash','family_exposures','tick','case_ids','actor_ids'):
            if a[key]!=c[key]:raise ValueError('collection external stream differs '+key)
    if any(archive[j]['request']!=own[j]['request'] for j in range(2)):raise ValueError('collection request identities differ')
    collection=dict(midpoint=midpoint,prefix_trace=prefix_trace,archive_trace=archive_trace,own_trace=trajectory['trace'],
        original_assisted_actual_endpoint_verified=True,original_dataset_reconstruction_exact=True,original_collector_state_exact=True,
        matched_external_streams=True,archive_frozen_h_end_hash=b.state_digest(frozen_end),own_frozen_h_end_hash=b.state_digest(trajectory['end_h']),
        distributions={name:[dict(job=j,positions=len(row['label']),
            executed_action_counts=torch.bincount(row['action'],minlength=11).tolist(),
            teacher_label_counts=torch.bincount(row['label'],minlength=11).tolist()) for j,row in rows.items()]
            for name,rows in (('shared_first64',prefix),('assisted_last64',archive),('own_policy_last64',own))})
    torch.save(dict(rows=own,trajectory=trajectory),out/'own_collection.pt')
    torch.save(dict(rows=archive,trace=archive_trace),out/'archive_collection.pt')
    b.old.write(out/'collection.json',collection)
    records=[];fit_info={};model_bindings={}
    for arm,new in (('assisted_archive',archive),('own_policy',own)):
        groups={j:{key:torch.cat((prefix[j][key],new[j][key])) for key in ('x','label','action')} for j in range(2)}
        for j in range(2):
            groups[j]['request']=sorted(set(prefix[j]['request'])|set(new[j]['request']))
            if len(groups[j]['label'])!=65536:raise ValueError('unequal fitting budget')
        requests=sorted(set(k for g in groups.values() for k in g['request']))
        model,info=fit_model(endpoint,groups,indices,plan['fit']['optimizer']);fit_info[arm]=info
        torch.save(model.state_dict(),out/f'model_{arm}.pt');model_bindings[arm]=b.old.sha(out/f'model_{arm}.pt')
        for condition in ('short','long_both'):
            for family in range(4):
                value=b.evaluate_shadow(model,endpoint,family=family,condition=condition,requests=requests)
                if any(value['record'][k] for k in ('world_resets','hidden_resets','parameter_updates')):raise ValueError('biased free evaluation')
                value.update(arm=arm,condition=condition,family=family);records.append(value)
                b.old.write(out/'evaluation_partial.json',dict(records=records))
                print(seed,arm,condition,family,flush=True)
        if b.state_digest(endpoint.semantic_state())!=endpoint_hash:raise ValueError('evaluation source mutated')
    checked()
    result=dict(format='matched-policy-data-refit-result-v1',seed=seed,protocol_hash=plan['protocol_hash'],
                source_semantic_hash=endpoint_hash,source_unchanged=True,collection=collection,
                fit=fit_info,model_sha256=model_bindings,records=records,
                collection_files={str(out/n):b.old.sha(out/n) for n in ('own_collection.pt','archive_collection.pt')},
                seconds=time.perf_counter()-started)
    result=json.loads(json.dumps(result));result['result_hash']=b.state_digest(result);b.old.write(out/'result.json',result)


def launch():
    checked()
    def one(seed):
        memory=available_memory()
        if memory<5*1024**3:raise MemoryError('refit RAM gate')
        with (OUT/f'seed_{seed:03d}.log').open('w') as stream:
            p=subprocess.Popen([sys.executable,'-m','experiments.action_return.refit_policy','run','--seed',str(seed)],cwd=b.ROOT,
                env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},'CUDA_VISIBLE_DEVICES':'-1'},stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,pid=p.pid,available_ram_before_launch=memory,returncode=p.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(one,range(3)))
    b.old.write(OUT/'exit_codes.json',codes)
    if any(v['returncode'] for v in codes):raise RuntimeError(codes)


def rates(c):
    d=c['primary_matured'];idle=c['idle_opportunities']
    return dict(strict_pp=100*c.get('primary_clean_completed',0)/d,ordinary_pp=100*c.get('primary_completed',0)/d,
        feasible_begin_pp=100*c.get('primary_feasible_began',0)/d,idle_actions_per1000=1000*c.get('idle_actions',0)/idle,
        idle_begin_per1000=1000*c.get('idle_false_begin',0)/idle,begin_attempts_per_matured=c.get('begin_attempts',0)/d,
        finish_attempts_per_matured=c.get('finish_attempts',0)/d)


def summarize():
    plan=checked();rows=[];paired=[];bindings={};fit=[];training_distributions=[]
    for seed in range(3):
        path=OUT/f'seed_{seed:03d}'/'result.json';v=b.old.read(path);anchor=b.old.read(plan['seeds'][str(seed)]['anchor'])
        if v['protocol_hash']!=plan['protocol_hash'] or v['result_hash']!=b.state_digest({k:x for k,x in v.items() if k!='result_hash'}):raise ValueError('refit result digest')
        if not v['source_unchanged']:raise ValueError('source changed')
        for filename,sha in v['collection_files'].items():
            if b.old.sha(filename)!=sha:raise ValueError('collection changed')
        for arm,sha in v['model_sha256'].items():
            if b.old.sha(path.parent/f'model_{arm}.pt')!=sha:raise ValueError('refit model changed')
        bindings[str(path)]=b.old.sha(path)
        fit.extend(dict(seed=seed,arm=arm,**j) for arm,info in v['fit'].items() for j in info)
        training_distributions.extend(dict(seed=seed,part=part,**row) for part,items in v['collection']['distributions'].items() for row in items)
        for condition in ('short','long_both'):
            total={a:Counter() for a in (*ARMS,'unrefit_anchor')}
            for family in range(4):
                cases=[next(r for r in v['records'] if r['arm']==arm and r['condition']==condition and r['family']==family) for arm in ARMS]
                cases.append(next(r for r in anchor['records'] if r['condition']==condition and r['family']==family))
                for c in cases[1:]:
                    for key in ('start_world_state_hash','start_hidden_hash','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash'):
                        if c['record'][key]!=cases[0]['record'][key]:raise ValueError('unmatched deployment fork '+key)
                    for group in ('heldout','overlap','all'):
                        if c['shadow'][group]['primary_request_set_sha256']!=cases[0]['shadow'][group]['primary_request_set_sha256']:raise ValueError('unmatched matured identities')
                for arm,c in zip(total,cases):total[arm].update(c['shadow']['heldout']['counts'])
            for arm,c in total.items():rows.append(dict(seed=seed,condition=condition,arm=arm,counts=dict(c),rates=rates(c)))
            a,c=rates(total['own_policy']),rates(total['assisted_archive'])
            paired.append(dict(seed=seed,condition=condition,**{key:a[key]-c[key] for key in a}))
    aggregate={condition:{key:dict(mean=statistics.mean(r[key] for r in paired if r['condition']==condition),
        sample_sd=statistics.stdev(r[key] for r in paired if r['condition']==condition)) for key in paired[0] if key not in ('seed','condition')} for condition in ('short','long_both')}
    result=dict(format='matched-policy-data-refit-summary-v1',audit='passed',protocol_hash=plan['protocol_hash'],new_forks=48,reused_anchor_forks=24,
        heldout=rows,paired=paired,aggregate=aggregate,fit=fit,training_distributions=training_distributions,results=bindings,limits=plan['limits'],
        checks=['each65536-row/job fit budget identical','old fitted feature table and midpoint h exactly reconstructed','original actual assisted world/hidden endpoints and observations/actions/labels replayed exactly',
                'own-policy collection executed original combined controller without teacher intervention','collection external event/scheduler/command/request streams matched',
                'all free evaluation matured primary requests matched with both arms and unrefit anchor','source unchanged and model/data SHA verified'])
    result=json.loads(json.dumps(result));result['summary_hash']=b.state_digest(result);b.old.write(OUT/'summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','launch','summarize'));p.add_argument('--seed',type=int,choices=range(3));a=p.parse_args()
    if a.command=='prepare':print(prepare()['protocol_hash'])
    elif a.command=='run':run(a.seed)
    elif a.command=='launch':launch()
    else:print(json.dumps(summarize()['aggregate'],indent=2))
