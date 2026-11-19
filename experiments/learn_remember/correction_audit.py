"""Independent raw-fact counters and first-step optimizer reconstruction."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from experiments.learn_remember import correction as c
from experiments.lifelong.world import ContextWorld,NEW,REPLACE
from experiments.lifelong.evaluate import CONDITIONS


def native_first_update(model,audit):
    model=deepcopy(model);model.load_state_dict(audit['before'],strict=True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=.001);optimizer.load_state_dict(deepcopy(audit['optimizer_before']))
    optimizer.zero_grad(set_to_none=True);h=audit['h_start'].clone();states=[]
    for obs,dt in zip(audit['obs'],audit['elapsed']):
        _,h=model.step(obs,h,float(dt));states.append(h)
    states=torch.stack(states);features=states[:,:4].index_select(-1,model.begin_head.indices)
    group=[];conditional=[]
    for job in range(2):
        z=(features-getattr(model.begin_head,f'j{job}_mean'))/getattr(model.begin_head,f'j{job}_std')
        group.append(z@getattr(model.begin_head,f'j{job}_coeff').T+getattr(model.begin_head,f'j{job}_bias'))
        z=(features-getattr(model,f'j{job}_mean'))/getattr(model,f'j{job}_std')
        conditional.append(z@getattr(model,f'j{job}_coef').T+getattr(model,f'j{job}_bias'))
    group=torch.stack(group,-2);conditional=torch.stack(conditional,-2)
    target=audit['targets'][:,:4];weight=torch.ones_like(target,dtype=torch.float64);weight[target==0]=.1
    # Derive weighted log probabilities directly rather than calling train.objective.
    ga=torch.cat((torch.zeros_like(group[...,:1]),group),-1)
    gt=torch.where(target<=2,target,0)
    group_nll=-F.log_softmax(ga,dim=-1).gather(-1,gt[...,None]).squeeze(-1)
    ca=torch.cat((torch.zeros_like(conditional[...,:1]),conditional),-1)
    keep=(target==0)|(target>=3);ct=torch.where(target==0,0,target-2)
    cond_nll=-F.log_softmax(ca[keep],dim=-1).gather(-1,ct[keep].unsqueeze(-1)).squeeze(-1)
    loss=((weight*group_nll).sum()+(weight[keep]*cond_nll).sum())/weight.sum()
    if abs(float(loss.detach())-audit['loss'])>1e-12:raise AssertionError('independent loss mismatch')
    loss.backward()
    # Tensor backward hooks are not retained by deepcopy; enforce the declared
    # eligibility mask explicitly before clipping rather than inherit a hook.
    model.begin_head.base.edge_shared.grad.masked_fill_(~audit['mask'],0.)
    torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
    with torch.no_grad():model.begin_head.base.edge_shared.masked_fill_(~audit['mask'],0.)
    error=max(float((value-audit['after'][key]).abs().max()) for key,value in model.state_dict().items() if value.numel())
    if error>2e-10:raise AssertionError('independent first parameter update mismatch '+str(error))
    if not torch.equal(h.detach(),audit['h_end']):raise AssertionError('native continuous h differs')
    optimizer_error=max(float((value-audit['optimizer_after']['state'][index][key]).abs().max())
        for index,state in optimizer.state_dict()['state'].items() for key,value in state.items())
    if optimizer_error>2e-10:raise AssertionError('independent Adam moments differ')
    return dict(independent_loss=float(loss.detach()),saved_loss=audit['loss'],maximum_parameter_error=error,
        maximum_optimizer_error=optimizer_error,
        hidden_bit_exact=True,Adam_updates_before=sorted({int(v['step']) for v in audit['optimizer_before']['state'].values()}),
        Adam_updates_after=sorted({int(v['step']) for v in optimizer.state_dict()['state'].values()}))


def counters(data,case_ids,regime):
    rows={};steps=len(data['actions'])
    command=np.isin(data['event_kind'],[NEW,REPLACE])&data['event_eligible']&(data['event_regime_id']==regime)
    for tick,lane,job in zip(*np.where(command)):
        key=(int(lane),int(job),int(data['request_id'][tick,lane,job]))
        if key in rows:raise AssertionError('duplicate new request')
        rows[key]=dict(matured=False,completed=False,clean=False,began=False,feasible=False)
    for tick in range(steps):
        for lane,job in zip(*np.where(data['matured_mask'][tick])):
            row=rows.get((int(lane),int(job),int(data['matured_request_id'][tick,lane,job])))
            if row is not None:
                if row['matured']:raise AssertionError('duplicate maturity')
                row.update(matured=True,completed=bool(data['matured_completed'][tick,lane,job]),clean=bool(data['matured_clean_completed'][tick,lane,job]))
        accepted=np.isin(data['actions'][tick],[1,2])&(data['result'][tick]==1)
        for lane,job in zip(*np.where(accepted)):
            row=rows.get((int(lane),int(job),int(data['request_id'][tick,lane,job])))
            if row is not None:
                row['began']=True;row['feasible']|=data['actions'][tick,lane,job]==data['teacher_actions'][tick,lane,job]
    selected=[(key,row) for key,row in rows.items() if key[0]>=4 and row['matured']]
    ids=[(case_ids[key[0]],key[1],key[2]) for key,row in selected]
    result=Counter(primary_matured=len(selected),primary_completed=sum(row['completed'] for key,row in selected),
        primary_clean_completed=sum(row['clean'] for key,row in selected),primary_began=sum(row['began'] for key,row in selected),
        primary_feasible_began=sum(row['feasible'] for key,row in selected))
    enabled=data['enabled'].copy();enabled[:,:4]=False;active=enabled&data['active'];idle=enabled&~data['active']
    action=data['actions'];begin=np.isin(action,[1,2]);finish=action>=3;accepted=data['result']==1
    legal=active&(data['phase']==3);processing=active&(data['phase']==2)
    exact=action-3==4*data['version']+data['payload']
    masks=dict(opportunities=enabled,active_opportunities=active,idle_opportunities=idle,idle_actions=idle&(action!=0),
        idle_begin=idle&begin,idle_finish=idle&finish,begin_attempts=enabled&begin,finish_attempts=enabled&finish,
        accepted_begin=enabled&begin&accepted,accepted_finish=enabled&finish&accepted,legal_finish_opportunities=legal,
        legal_finish_attempts=legal&finish,legal_finish_correct=legal&finish&exact,legal_finish_wrong=legal&finish&~exact,
        processing_opportunities=processing,processing_early_finish=processing&finish,processing_wait=processing&(action==0))
    teacher_finish=enabled&(data['teacher_actions']>=3)
    masks.update(teacher_finish_positions=teacher_finish,teacher_finish_chosen_finish=teacher_finish&finish,
        teacher_finish_chosen_wait=teacher_finish&(action==0),teacher_finish_chosen_begin=teacher_finish&begin,
        teacher_finish_exact_action=teacher_finish&(action==data['teacher_actions']))
    result.update({key:int(value.sum()) for key,value in masks.items()})
    return dict(result),ids


def replay_training(seed,mode):
    folder=c.OUT/f'seed_{seed:03d}'
    initial=c.check_snapshot(folder/f'{mode}.initial.pt');final=c.check_snapshot(folder/f'{mode}.checkpoint.pt')
    with np.load(folder/f'{mode}.train.npz',allow_pickle=False) as bundle:data={k:bundle[k] for k in bundle.files}
    world=ContextWorld.from_state_dict(initial['world'],device='cpu')
    collector,ch=c.load_collector(seed) if mode=='original_replay' else (None,None)
    for tick in range(len(data['actions'])):
        if tick%c.BLOCK==0:
            block=tick//c.BLOCK;family,condition=c.SCHEDULE[block%len(c.SCHEDULE)]
            world.set_family(family);world.set_regime(**CONDITIONS[condition],name=f'correction-train/{block}/{family}/{condition}',dt=.25)
        obs=world.observe().double();label=world.oracle('midpoint')
        if not np.array_equal(obs.numpy(),data['obs'][tick]) or not np.array_equal(label.numpy(),data['teacher_actions'][tick]):
            raise AssertionError('saved public observation or teacher differs from actual world')
        if collector is not None:
            with torch.no_grad():logits,ch=collector.step(obs,ch,float(world.elapsed))
            if not np.array_equal(logits.argmax(-1).numpy(),data['actions'][tick]):raise AssertionError('control did not execute original collector')
        world.act(torch.from_numpy(data['actions'][tick]))
        if not np.array_equal(world.last_result.numpy(),data['result'][tick]):raise AssertionError('real receipt mismatch')
    if c.digest(world.state_dict())!=c.digest(final['world']):raise AssertionError('reconstructed final world differs')
    if collector is not None and not torch.equal(ch,final['collector_hidden']):raise AssertionError('collector h reconstruction differs')
    logs=[json.loads(line) for line in (folder/f'{mode}.updates.jsonl').read_text().splitlines()]
    if len(logs)!=c.UPDATES or any(a['h_end_hash']!=z['h_start_hash'] for a,z in zip(logs,logs[1:])):raise AssertionError('h discontinuity')
    if mode=='no_learning' and c.digest(initial['model'])!=c.digest(final['model']):raise AssertionError('no-learning model changed')
    return dict(seed=seed,mode=mode,ticks=len(data['actions']),world_receipts_public_labels_exact=True,
        collector_actions_hidden_exact=True if collector is not None else None,continuous_hidden_hashes=True)


def run(seed):
    torch.set_num_threads(1);c.checked();folder=c.OUT/f'seed_{seed:03d}';value=c.read(folder/'result.json')
    model,_,_,_,_=c.load_source(seed);first=[]
    for mode in c.MODES[:2]:
        saved=torch.load(folder/f'{mode}.first_update.pt',map_location='cpu',weights_only=False)
        first.append(dict(seed=seed,mode=mode,**native_first_update(model,saved)))
    training=[replay_training(seed,mode) for mode in c.MODES]
    tests=[]
    for record in value['records']:
        with np.load(record['facts']['path'],allow_pickle=False) as bundle:data={k:bundle[k] for k in bundle.files}
        source=c.check_snapshot(folder/f"{record['mode']}.checkpoint.pt") if record['stage']=='after4096' else c.check_snapshot(folder/'own_receipts.initial.pt')
        world=ContextWorld.from_state_dict(source['world'],device='cpu');counts,ids=counters(data,world.case_ids,record['regime'])
        expected=record['score']['heldout']['counts']
        for key,number in counts.items():
            if expected.get(key,0)!=number:raise AssertionError(f'raw count differs {seed} {record["mode"]} {key}')
        if c.ev.identity_hash(ids)!=record['score']['heldout']['primary_request_set_sha256']:raise AssertionError('primary identity digest')
        tests.append(dict(stage=record['stage'],mode=record['mode'],condition=record['condition'],family=record['family'],
            independent_count_keys=len(counts),matured=len(ids),all_counts_and_ID_exact=True))
    result=dict(seed=seed,passed=True,first_updates=first,training=training,evaluations=tests,
        source_sha256=c.sha(__file__),result_sha256=c.sha(folder/'result.json'))
    c.write(folder/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();run(args.seed)
