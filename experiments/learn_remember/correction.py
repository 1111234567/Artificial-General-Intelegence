"""Matched continued correction on own receipts versus frozen collector receipts.

The learner sees only the unchanged public56-vector. The original own_policy
collector executes its own actions in the replay control; oracle labels never
enter either controller or the world action. All new state remains continuous.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from experiments.wait_content import train as tr, eval as ev, long_followup as lf
from experiments.lifelong.world import ContextWorld
from experiments.lifelong.evaluate import CONDITIONS, _phases

ROOT=tr.ROOT
OUT=ROOT/'data/learn-remember/correction-v1'
PLAN=OUT/'protocol.json'
FORMAT='own-receipt-versus-frozen-collector-correction-v1'
MODES=('own_receipts','original_replay','no_learning')
BLOCK=128
UPDATES=32
TEST=1024
SCHEDULE=tuple((family,condition) for condition in ('short','long_both') for family in range(4))
PRIVATE=lf.PRIVATE
EXTERNAL=lf.EXTERNAL


def write(path,value): ev.write(path,value)
def read(path): return ev.read(path)
def sha(path): return ev.sha(path)
def digest(value): return ev.state_digest(value)


def load_source(seed):
    model,saved=tr.load_trained(seed,'end_only')
    warmed=torch.load(ev.OUT/f'seed_{seed:03d}/end_only_warmup.pt',map_location='cpu',weights_only=False)
    endpoint=torch.load(tr.DATA/f'seed_{seed:03d}/endpoint.pt',map_location='cpu',weights_only=False)
    if warmed['model_hash']!=digest(model.state_dict()): raise ValueError('warmup/model mismatch')
    world=ContextWorld.from_state_dict(endpoint['world'],device='cpu')
    h=warmed['h'].clone()
    if h.shape!=(8,80) or not endpoint['world']['acted']: raise ValueError('incomplete real source')
    params=[p for p in model.parameters() if p.requires_grad]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if names!=saved['trainable_names']: raise ValueError('optimizer parameter ordering')
    optimizer=torch.optim.Adam(params,lr=.001,betas=(.9,.999),eps=1e-8)
    optimizer.load_state_dict(saved['optimizer'])
    if digest(optimizer.state_dict())!=digest(saved['optimizer']): raise ValueError('Adam resume')
    return model,optimizer,world,h,saved


def load_collector(seed):
    model,_=tr.load_initial(seed)
    endpoint=torch.load(tr.DATA/f'seed_{seed:03d}/endpoint.pt',map_location='cpu',weights_only=False)
    warmed=torch.load(ev.OUT/f'seed_{seed:03d}/untrained_anchor_warmup.pt',map_location='cpu',weights_only=False)
    if digest(model.state_dict())!=warmed['model_hash'] or not torch.equal(endpoint['h'],warmed['h']):
        raise ValueError('original collector has wrong corresponding hidden history')
    return model.eval(),warmed['h'].clone()


def snapshot(model,optimizer,world,h,*,seed,mode,ticks,collector=None,collector_h=None):
    value=dict(format=FORMAT,seed=seed,mode=mode,ticks=ticks,model=deepcopy(model.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()),world=world.state_dict(),hidden=h.clone(),
        collector_model=deepcopy(collector.state_dict()) if collector is not None else None,
        collector_hidden=collector_h.clone() if collector_h is not None else None)
    value['state_hash']=digest(value)
    return value


def check_snapshot(path):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['state_hash']!=digest({k:v for k,v in value.items() if k!='state_hash'}): raise ValueError('checkpoint hash')
    return value


def prepare():
    if PLAN.exists(): raise FileExistsError('immutable correction protocol exists')
    paths=[Path(__file__).resolve(),Path(tr.__file__).resolve(),Path(ev.__file__).resolve(),Path(lf.__file__).resolve(),
        ROOT/'experiments/lifelong/world.py',ROOT/'experiments/continuity/world.py',
        ROOT/'experiments/action_return/finish_readout.py',ROOT/'experiments/undirected/model.py']
    bindings={}
    sources={}
    for seed in range(3):
        model,optimizer,world,h,saved=load_source(seed)
        collector,ch=load_collector(seed)
        sourcefiles=[tr.OUT/f'seed_{seed:03d}/end_only/checkpoint.pt',
            ev.OUT/f'seed_{seed:03d}/end_only_warmup.pt',ev.OUT/f'seed_{seed:03d}/untrained_anchor_warmup.pt',
            tr.DATA/f'seed_{seed:03d}/endpoint.pt',tr.DATA/f'seed_{seed:03d}/requests.json']
        sourcefiles += [Path(v['path']) for v in tr.source_info(seed).values()]
        bindings.update({str(p):sha(p) for p in sourcefiles})
        sources[str(seed)]=dict(model_hash=digest(model.state_dict()),hidden_hash=digest(h),world_hash=digest(world.state_dict()),
            optimizer_hash=digest(optimizer.state_dict()),collector_model_hash=digest(collector.state_dict()),
            collector_hidden_hash=digest(ch),source_tick=world.tick,source_time=world.now,
            trainable_names=saved['trainable_names'],previous_Adam_updates=128)
    plan=dict(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),seeds=[0,1,2],source_arm='end_only',
        declaration='Prospective fixed4096-tick correction and1024-tick tests; no outcome-driven extension, tuning, new probe or winner selection.',
        source_files={str(p):sha(p) for p in paths},input_files=bindings,sources=sources,
        modes=list(MODES),train=dict(ticks=UPDATES*BLOCK,updates=UPDATES,block_ticks=BLOCK,fit_lanes=[0,1,2,3],
            optimizer='Resume the end_only checkpoint Adam exactly, including all128-step moments and learning rate. Both correction groups start with identical Adam and receive32 further updates.',
            parameters='Same640 supported undirected shared edges plus existing400 conditionalFINISH weights/biases; fixed topology, ports, normalization, BEGIN affine decoder.',
            objective='Exactly inherited all-position action loss: OTHER/BEGIN1/BEGIN2 group CE and WAIT/eightFINISH conditional CE on nonBEGIN targets. WAIT weight0.1, other1, shared all-position weighted denominator. No auxiliary content term. Same target FUNCTION and hyperparameters; realized labels/class counts can differ with actual world state.',
            schedule=[dict(block_mod8=i,family=f,condition=c,**CONDITIONS[c]) for i,(f,c) in enumerate(SCHEDULE)],
            carry='Full8-lane h and real acted world persist through all32 blocks. Weights fixed within each128 ticks. Replay same block differentiably from its actual starting h, detach at update only, carry pre-update final h. No request/reset/initial observation injection.',
            own_receipts='Learner executes its exact existing11-action argmax in its world; learn on those public observations, its own receipts, and midpoint labels from that actual world.',
            original_replay='Frozen ORIGINAL own_policy collector executes its own argmax from corresponding own_policy warmup h in a clone of exactly the same source world. Student starts end_only h and only consumes collector public observations. Student decisions are never executed. Collector h and learner h remain distinct.',
            no_learning='End_only model executes its own actions through the same4096 external ticks, with no optimizer calls; preserves the equal-exposure uncorrected comparator.',
            matching='All modes start identical end_only weights+h and identical source world. Collector h is explicitly different. Check full exogenous request stream hashes per seed; never claim realized oracle labels equal.'),
        evaluation=dict(steps=TEST,conditions=['long_both','short'],families=[0,1,2,3],heldout_lanes=[4,5,6,7],
            stages=['initial_end_only_before_any_new_learning','each_mode_after4096_continuous_ticks'],
            source='After4096 use each mode own full continuous world+h, including actual action receipts; freeze weights and preserve pending/carryover contracts. Fork only family/condition. Also test unchanged source endpoint before correction.',
            primary='Only eligible evaluation-regime NEW/REPLACE requests issued after test fork and actually matured in1024ticks; exclude all crossing training requests and right-censored IDs, not conditional on actions/start/success.',
            metrics=['ordinary_completion','strict_completion','idle_actions/opportunities','feasible_accepted_BEGIN/matured','legal_FINISH_attempts/opportunities','legal_FINISH_correct/attempts'],
            isolation='Learning only lanes0..3. Test primary lanes4..7. Reject future primary(case,job,requestID) found anywhere in training observations; train/test full clocks and ID sets are saved.',
            matching='After4096 match exogenous stream and mature request identities across3modes for each seed/family/condition. Initial stage uses earlier future and is separately descriptive.'),
        runtime=dict(workers=3,threads_per_worker=1,dtype='float64',device='cpu'),
        limitations=['External midpoint supervision, not autonomous discovery of labels.','An original8-lane shared-public-history warmup is the inherited start, not an assertion end_only generated that historical trajectory.','Three paired seeds; family forks and requests are not independent model repeats.'])
    plan=ev.normalized(plan);plan['protocol_hash']=digest(plan);write(PLAN,plan)
    return plan


def checked():
    plan=read(PLAN)
    if plan['protocol_hash']!=digest({k:v for k,v in plan.items() if k!='protocol_hash'}): raise ValueError('protocol hash')
    replacements={}
    amendment=OUT/'implementation-amendment.json'
    if amendment.exists():
        note=read(amendment)
        if note['protocol_hash']!=plan['protocol_hash']:raise ValueError('foreign amendment')
        for item in note['changed_sources']:
            if plan['source_files'][item['path']]!=item['original_sha256'] or sha(item['original_snapshot'])!=item['original_sha256']:raise ValueError('original frozen implementation lost')
            replacements[item['path']]=item['amended_sha256']
    for group in ('source_files','input_files'):
        for path,expected in plan[group].items():
            if group=='source_files':expected=replacements.get(path,expected)
            if sha(path)!=expected: raise ValueError('bound input changed '+path)
    return plan


def fields():
    return {name:[] for name in (*PRIVATE,'obs','tick','time','elapsed','phase','enabled','actions','result','teacher_actions')}


def save_pre(logs,world,obs,actions,labels):
    for name in PRIVATE: logs[name].append(getattr(world,name).numpy().copy())
    logs['obs'].append(obs.numpy().copy())
    for key,value in dict(tick=world.tick,time=world.now,elapsed=float(world.elapsed)).items(): logs[key].append(value)
    logs['phase'].append(_phases(world))
    enabled=np.zeros_like(world.active.numpy());enabled[:,:world.active_jobs]=True
    logs['enabled'].append(enabled)
    logs['actions'].append(actions.numpy().copy());logs['teacher_actions'].append(labels.numpy().copy())


def arrays(logs): return {k:np.asarray(v) for k,v in logs.items()}


def all_ids(data,case_ids):
    return sorted({(case_ids[i],j,int(r)) for i in range(8) for j in range(2)
        for r in np.unique(data['request_id'][:,i,j]) if r>=0})


def evaluate(model,world,h,*,seed,mode,stage,folder,train_ids=()):
    records=[];initial_hash=digest(model.state_dict());source_hash=digest(world.state_dict())
    for condition in ('long_both','short'):
        for family in range(4):
            live=ContextWorld.from_state_dict(world.state_dict(),device='cpu');state=h.clone()
            live.set_family(family);live.set_regime(**CONDITIONS[condition],name=f'correction-evaluation/{family}/{condition}',dt=.25)
            regime=live.regime_id;logs=fields()
            with torch.no_grad():
                for _ in range(TEST):
                    obs=live.observe().double();labels=live.oracle('midpoint').clone()
                    logits,state=model.step(obs,state,float(live.elapsed));actions=logits.argmax(-1)
                    save_pre(logs,live,obs,actions,labels);live.act(actions)
                    logs['result'].append(live.last_result.numpy().copy())
            data=arrays(logs);rows=lf.request_rows(data,live.case_ids,live.actor_ids,regime)
            forbidden=set(map(tuple,train_ids));primary_ids={(r['case_id'],r['job'],r['request_id']) for r in rows}
            if primary_ids&forbidden: raise ValueError('future evaluation identity seen during learning/collection')
            score=lf.compute(data,rows,0,TEST)
            for group,lanes in (('heldout',range(4,8)),('overlap',range(4)),('all',range(8))):
                enabled=data['enabled'].copy();enabled[:,[i for i in range(8) if i not in lanes]]=False
                target=data['teacher_actions'];action=data['actions'];c=score[group]['counts']
                mask=enabled&(target>=3)
                c['teacher_finish_positions']=int(mask.sum());c['teacher_finish_chosen_finish']=int((mask&(action>=3)).sum())
                c['teacher_finish_chosen_wait']=int((mask&(action==0)).sum())
                c['teacher_finish_chosen_begin']=int((mask&np.isin(action,[1,2])).sum())
                c['teacher_finish_exact_action']=int((mask&(action==target)).sum())
            stem=f'{stage}_{mode}_{condition}_family_{family}'
            facts=folder/(stem+'.facts.npz');np.savez_compressed(facts,**{k:v for k,v in data.items() if k!='obs'})
            requests=folder/(stem+'.requests.json.gz');ev.compressed_json(requests,rows)
            record=dict(seed=seed,mode=mode,stage=stage,condition=condition,family=family,steps=TEST,
                fork_tick=world.tick,fork_time=world.now,end_tick=live.tick,regime=regime,
                start_world_hash=source_hash,start_h_hash=digest(h),end_world_hash=digest(live.state_dict()),
                end_h_hash=digest(state),end_scheduler_hash=digest(live.state_dict()['scheduler']),end_event_hash=live.event_hash,
                external_hash=lf.digest_arrays(data,EXTERNAL),action_hash=lf.action_hash(data['actions']),model_hash=initial_hash,
                score=score,train_test_ID_overlap=0,primary_issued=len(rows),
                facts=dict(path=str(facts),sha256=sha(facts)),requests=dict(path=str(requests),sha256=sha(requests)))
            records.append(record)
            print(seed,mode,stage,condition,family,'eval done',flush=True)
    if digest(model.state_dict())!=initial_hash or digest(world.state_dict())!=source_hash: raise ValueError('evaluation mutated source')
    return records


def train_mode(seed,mode,folder):
    model,optimizer,world,h,source=load_source(seed);mask=source['mask'];params=[p for p in model.parameters() if p.requires_grad]
    collector,ch=load_collector(seed) if mode=='original_replay' else (None,None)
    initial=snapshot(model,optimizer,world,h,seed=seed,mode=mode,ticks=0,collector=collector,collector_h=ch)
    torch.save(initial,folder/f'{mode}.initial.pt');fixed=tr.frozen_buffers(model)
    logs=fields();updates=[];start=time.perf_counter();first_audit=None
    with (folder/f'{mode}.updates.jsonl').open('w',encoding='utf-8') as stream:
        for block in range(UPDATES):
            family,condition=SCHEDULE[block%len(SCHEDULE)]
            world.set_family(family);world.set_regime(**CONDITIONS[condition],name=f'correction-train/{block}/{family}/{condition}',dt=.25)
            h_start=h.detach().clone();begin_tick=len(logs['obs']);before=digest(model.state_dict())
            with torch.no_grad():
                for _ in range(BLOCK):
                    obs=world.observe().double();labels=world.oracle('midpoint').clone()
                    if collector is None:
                        logits,h=model.step(obs,h,float(world.elapsed));actions=logits.argmax(-1)
                    else:
                        logits,ch=collector.step(obs,ch,float(world.elapsed));actions=logits.argmax(-1)
                    save_pre(logs,world,obs,actions,labels);world.act(actions)
                    logs['result'].append(world.last_result.numpy().copy())
            if mode!='no_learning':
                obs=torch.as_tensor(np.asarray(logs['obs'][begin_tick:]),dtype=torch.float64)
                elapsed=np.asarray(logs['elapsed'][begin_tick:]);targets=torch.as_tensor(np.asarray(logs['teacher_actions'][begin_tick:]),dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                begin,finish,h_end,_=tr.rollout(model,obs,h_start,elapsed)
                if mode=='own_receipts' and not torch.equal(h,h_end.detach()): raise ValueError('own action rollout/replay state not bit exact')
                chosen=np.empty((0,8),dtype=np.int64);positions=torch.empty((0,3),dtype=torch.long)
                loss,parts,budget=tr.objective(begin[:,:4],finish[:,:4],targets[:,:4],chosen,positions)
                if block==0:
                    first_audit=dict(before=deepcopy(model.state_dict()),optimizer_before=deepcopy(optimizer.state_dict()),h_start=h_start,
                        obs=obs,elapsed=elapsed,targets=targets,mask=mask,loss=float(loss.detach()))
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                with torch.no_grad():model.begin_head.base.edge_shared.masked_fill_(~mask,0.)
                h=h_end.detach()
                if block==0:
                    first_audit.update(after=deepcopy(model.state_dict()),optimizer_after=deepcopy(optimizer.state_dict()),h_end=h.clone())
                    torch.save(first_audit,folder/f'{mode}.first_update.pt')
                entry=dict(block=block,ticks=(block+1)*BLOCK,loss=float(loss.detach()),gradient_norm=float(norm),**budget)
            else:
                entry=dict(block=block,ticks=(block+1)*BLOCK,parameter_updates=0)
            entry.update(h_start_hash=digest(h_start),h_end_hash=digest(h),model_before_hash=before,model_after_hash=digest(model.state_dict()),
                end_world_hash=digest(world.state_dict()),end_scheduler_hash=digest(world.state_dict()['scheduler']),end_event_hash=world.event_hash)
            updates.append(entry);stream.write(json.dumps(entry,separators=(',',':'))+'\n');stream.flush()
            if (block+1)%8==0: print(seed,mode,'trained',block+1,'seconds',round(time.perf_counter()-start,1),flush=True)
    if any(not torch.equal(value,model.state_dict()[key]) for key,value in fixed.items()): raise ValueError('frozen parameter/buffer changed')
    model.begin_head.base.validate_eligibility(mask)
    final=snapshot(model,optimizer,world,h,seed=seed,mode=mode,ticks=UPDATES*BLOCK,collector=collector,collector_h=ch)
    torch.save(final,folder/f'{mode}.checkpoint.pt')
    data=arrays(logs);facts=folder/f'{mode}.train.npz';np.savez_compressed(facts,**data)
    ids=all_ids(data,world.case_ids);write(folder/f'{mode}.train_ids.json',ids)
    result=dict(mode=mode,seed=seed,updates=UPDATES if mode!='no_learning' else 0,ticks=UPDATES*BLOCK,
        first_world_hash=digest(initial['world']),first_h_hash=digest(initial['hidden']),first_model_hash=digest(initial['model']),
        first_optimizer_hash=digest(initial['optimizer']),last_world_hash=digest(world.state_dict()),last_h_hash=digest(h),
        last_model_hash=digest(model.state_dict()),last_optimizer_hash=digest(optimizer.state_dict()),
        external_hash=lf.digest_arrays(data,EXTERNAL),end_scheduler_hash=digest(world.state_dict()['scheduler']),end_event_hash=world.event_hash,
        class_counts=np.bincount(data['teacher_actions'][:,:4].reshape(-1),minlength=11).tolist(),action_positions=UPDATES*BLOCK*4*2,
        no_hidden_reset=True,all_block_h_continuous=all(a['h_end_hash']==z['h_start_hash'] for a,z in zip(updates,updates[1:])),
        collector_model_hash=digest(collector.state_dict()) if collector is not None else None,
        facts=dict(path=str(facts),sha256=sha(facts)),checkpoint=dict(path=str(folder/f'{mode}.checkpoint.pt'),sha256=sha(folder/f'{mode}.checkpoint.pt')),
        train_ids=dict(path=str(folder/f'{mode}.train_ids.json'),sha256=sha(folder/f'{mode}.train_ids.json')),seconds=time.perf_counter()-start)
    if mode=='original_replay' and result['collector_model_hash']!=read(PLAN)['sources'][str(seed)]['collector_model_hash']: raise ValueError('collector weights changed')
    write(folder/f'{mode}.train_result.json',result)
    return model,world,h,ids,result


def run(seed):
    torch.set_num_threads(1);plan=checked();folder=OUT/f'seed_{seed:03d}';folder.mkdir(exist_ok=False)
    start=time.perf_counter();records=[];train=[]
    for mode in MODES:
        model,world,h,ids,result=train_mode(seed,mode,folder);train.append(result)
        records += evaluate(model,world,h,seed=seed,mode=mode,stage='after4096',folder=folder,train_ids=ids)
        write(folder/'partial.json',dict(seed=seed,train=train,records=records,seconds=time.perf_counter()-start))
    for row in train[1:]:
        for key in ('first_world_hash','first_h_hash','first_model_hash','first_optimizer_hash','external_hash','end_scheduler_hash','end_event_hash'):
            if row[key]!=train[0][key]: raise ValueError('unmatched training stream '+key)
    model,_,world,h,_=load_source(seed)
    records += evaluate(model,world,h,seed=seed,mode='end_only',stage='initial',folder=folder)
    checked();result=dict(format=FORMAT,seed=seed,protocol_hash=plan['protocol_hash'],train=train,records=records,seconds=time.perf_counter()-start)
    result['result_hash']=digest(result);write(folder/'result.json',result)
    return result


def summarize():
    plan=checked();all_records=[];train=[];bindings={};rows=[];paired=[]
    for seed in range(3):
        path=OUT/f'seed_{seed:03d}/result.json';value=read(path);bindings[str(path)]=sha(path)
        if value['result_hash']!=digest({k:v for k,v in value.items() if k!='result_hash'}): raise ValueError('result hash')
        all_records+=value['records'];train+=value['train']
        for condition in ('long_both','short'):
            rates={}
            for mode in MODES:
                selected=[r for r in value['records'] if r['stage']=='after4096' and r['condition']==condition and r['mode']==mode]
                if len(selected)!=4: raise ValueError('missing family')
                c=Counter()
                for record in selected:c.update(record['score']['heldout']['counts'])
                rates[mode]=lf.behavior_rates(c);rows.append(dict(seed=seed,stage='after4096',condition=condition,mode=mode,counts=dict(c),rates=rates[mode]))
            for family in range(4):
                group=[r for r in value['records'] if r['stage']=='after4096' and r['condition']==condition and r['family']==family]
                for record in group[1:]:
                    for key in ('external_hash','end_scheduler_hash','end_event_hash'):
                        if record[key]!=group[0][key]: raise ValueError('unmatched evaluation external stream '+key)
                    if record['score']['heldout']['primary_request_set_sha256']!=group[0]['score']['heldout']['primary_request_set_sha256']: raise ValueError('matured test IDs mismatch')
            paired.append(dict(seed=seed,condition=condition,delta_own_minus_replay={key:rates['own_receipts'][key]-rates['original_replay'][key]
                if rates['own_receipts'][key] is not None and rates['original_replay'][key] is not None else None for key in rates['own_receipts']}))
            c=Counter()
            for r in value['records']:
                if r['stage']=='initial' and r['condition']==condition:c.update(r['score']['heldout']['counts'])
            rows.append(dict(seed=seed,stage='initial',condition=condition,mode='end_only',counts=dict(c),rates=lf.behavior_rates(c)))
    aggregate={}
    for condition in ('long_both','short'):
        aggregate[condition]={}
        for mode in (*MODES,'initial_end_only'):
            chosen=[r for r in rows if r['condition']==condition and (r['mode']==mode if mode!='initial_end_only' else r['stage']=='initial')]
            c=Counter()
            for r in chosen:c.update(r['counts'])
            aggregate[condition][mode]=dict(counts=dict(c),pooled_rates=lf.behavior_rates(c),
                equal_seed_mean={key:statistics.mean([r['rates'][key] for r in chosen if r['rates'][key] is not None]) if any(r['rates'][key] is not None for r in chosen) else None for key in chosen[0]['rates']})
    value=dict(format=FORMAT,protocol_hash=plan['protocol_hash'],per_seed=rows,paired=paired,aggregate=aggregate,training=train,input_files=bindings,
        seeds=3,train_test_ID_overlap=0,external_streams_matched=True,completed_forks=len(all_records))
    value['summary_hash']=digest(value);write(OUT/'summary.json',value)
    return value


def launch():
    checked()
    def one(seed):
        path=OUT/f'seed_{seed:03d}.log'
        with path.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen([sys.executable,'-m','experiments.learn_remember.correction','run','--seed',str(seed)],cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
            return dict(seed=seed,pid=process.pid,returncode=process.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,range(3)))
    write(OUT/'exit_codes.json',results)
    if any(r['returncode'] for r in results): raise RuntimeError(results)
    summarize()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','launch','summarize'));parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='run':run(args.seed)
    elif args.command=='launch':launch()
    else:summarize()


if __name__=='__main__': main()
