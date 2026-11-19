"""Supplementary static last eligible WAIT control, selected after interim diagnostics."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.experience_transfer import train_transfer as t
from experiments.continuity.run import state_digest
from experiments.review_selection import train_review as previous
from experiments.wait_position import train_review as matched

OUT = ROOT / 'data/wait-position/last-v1'
BANK = ROOT / 'data/wait-position/bank-v1'
PLAN = OUT / 'protocol.json'
ARMS = ('last',)
REVIEW_EPOCHS = (8, 16, 24, 32)
CANDIDATES, SELECTED = 128, 64
SNAPSHOTS = ('B000', 'Bfirst002', 'Bfirst008', 'B001', 'B002', 'B004', 'B008')


def bank_path(seed):
    return BANK / f'seed_{seed:03d}' / 'bank.pt'


def origin(seed):
    return t.OUT / f'seed_{seed:03d}' / 'related' / 'B000.pt'


def declare():
    if PLAN.exists():raise FileExistsError(PLAN)
    main=matched.checked()
    sources=[Path(__file__).resolve(),ROOT/'experiments/wait_position/last_audit.py',
             Path(matched.__file__).resolve(),Path(t.__file__).resolve(),
             ROOT/'experiments/learn_remember/gru_train.py',ROOT/'experiments/lifelong/world.py',ROOT/'experiments/continuity/world.py']
    inputs={}
    for seed in range(3):
        paths=[origin(seed),bank_path(seed),BANK/'protocol.json',matched.PLAN,previous.PLAN,t.PLAN]+[
            t.OUT/f'seed_{seed:03d}'/'related'/f'trajectory_B_{r:03d}.pt' for r in range(8)]
        inputs[str(seed)]={str(path):t.sha(path) for path in paths}
    plan=dict(format='wait-position-static-last-supplement-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2],arms=list(ARMS),snapshots=list(SNAPSHOTS),device='cuda',
        source_sha256={str(path):t.sha(path) for path in sources},input_sha256=inputs,
        main_protocol_hash=main['protocol_hash'],
        declaration='Mechanistic supplement chosen after observing that over92percent of first-round hard choices fell at the last eligible WAIT tick and after seeing the intermediate B002 premature-FINISH diagnostic. The last control was not selected by final B008 performance. This is not a prospective fourth arm of the original three-arm protocol.',
        start=main['start'],B=main['B'],updates=main['updates'],old_bank=main['old_bank'],gradient=main['gradient'],state=main['state'],
        candidates='Exactly the main and previous uniform candidate128 and selected64 identities at each index. Execute the same full no_grad128x128 current-model scoring, max margins, argmax intents and fixed random draws. Those scores never select the last tick.',
        selection=dict(last='For each identical selected request, use the final chronological tick in its actual same-request eligible processing-WAIT set. No model score or ranking enters this position.'),
        evidence='Same snapshot/checkpoint schema and all-step ID hashes,integer budgets,actual B exposure. Independent first moved mask/loss/Adam/live-h/world replay,all256 request selections checked against prior uniform; B000 and first B snapshots bit exact.',
        limits=main['limits']+['Post-interim static last control distinguishes learned risk targeting from simply supervising close to readiness.'])
    plan['protocol_hash']=state_digest(plan);t.write(PLAN,plan)
    print(json.dumps(dict(event='declared',path=str(PLAN),protocol_hash=plan['protocol_hash'])),flush=True)


def checked():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    assert state_digest({k:v for k,v in plan.items() if k!='protocol_hash'})==plan['protocol_hash']
    for path,value in plan['source_sha256'].items():
        assert t.sha(path)==value,'frozen source changed: '+path
    for inputs in plan['input_sha256'].values():
        for path,value in inputs.items():
            assert t.sha(path)==value,'frozen input changed: '+path
    return plan


candidate_ids=matched.candidate_ids
scores=matched.scores
ReviewLearner=matched.ReviewLearner


def selected_ids(arm,ids,values,bank,seed,index):
    assert arm=='last'
    return matched.selected_ids('midpoint',ids,values,bank,seed,index)


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def review_update(v,bank,gpu_bank,epoch,block):
    index=v.review_index;ids=candidate_ids(bank,v.seed,index)
    candidate={k:value[:,ids].contiguous() if k in ('obs','labels','mask','eligible_mask') else value[ids].contiguous()
               for k,value in gpu_bank.items()}
    score=scores(v.model,candidate['obs'],candidate['anchor'],candidate['eligible_mask'])
    selected,local=selected_ids(v.arm,ids,None,bank,v.seed,index)
    slots=bank['target_slot'][ids]
    midpoint=bank['positions'][ids,1]
    eligible=bank['eligible_mask'][:,ids]
    random_rng=np.random.default_rng(3140000+v.seed*10000+index)
    random_ticks=torch.tensor([random_rng.choice(torch.where(eligible[:,i,slots[i]])[0].numpy()) for i in range(len(ids))])
    all_ticks=torch.tensor([torch.where(eligible[:,i,slots[i]])[0][-1] for i in range(len(ids))])
    ticks=all_ticks[local];chosen_slots=slots[local]
    batch={k:value[:,local].contiguous() if k in ('obs','labels','mask','eligible_mask') else value[local].contiguous()
           for k,value in candidate.items()}
    # Remove only the old WAIT row and insert the selected eligible WAIT row.
    mask=batch['mask'].clone();mask[bank['positions'][selected,1].cuda(),torch.arange(64,device='cuda'),chosen_slots.cuda()]=False
    mask[ticks.cuda(),torch.arange(64,device='cuda'),chosen_slots.cuda()]=True
    assert int(mask.sum())==192 and int((mask&(batch['labels']!=0)).sum())==128
    assert batch['labels'][ticks.cuda(),torch.arange(64,device='cuda'),chosen_slots.cuda()].eq(0).all()
    moved=ticks.ne(bank['positions'][selected,1]);first=index==0
    first_moved=bool(moved.any()) and not v.first_moved_saved
    record=dict(candidate_ids=ids,selected_ids=selected,selected_local=local,seed=v.seed,arm=v.arm,
        review_index=index,round=v.round,epoch=epoch,block=block,protocol_hash=v.review_protocol_hash,
        candidate_ticks=all_ticks,selected_ticks=ticks,midpoint_ticks=bank['positions'][selected,1],
        candidate_hard_ticks=score['hard_ticks'],candidate_maximum=score['maximum'],
        candidate_any_finish=score['any_finish'],supervision_mask=mask.cpu())
    if first:
        v.save('first_review_before');torch.save(record,v.folder/'first_selection.pt')
    if first_moved:
        v.save('first_moved_before');torch.save(record,v.folder/'first_moved_selection.pt')
    live_before=state_digest(dict(h=v.hidden,world=v.world_state()))
    v.step_identity=dict(review_index=index,candidate_id_hash=state_digest(ids),selected_id_hash=state_digest(selected),
        selected_time_hash=state_digest(ticks),supervision_mask_hash=state_digest(mask.cpu()),
        bank_sha256=v.bank_sha256,epoch=epoch,block=block)
    old_counts=v.selection_counts[selected].clone()
    _,loss,gradnorm=v.update(batch['obs'],batch['labels'],mask,batch['anchor'],kind='A')
    assert live_before==state_digest(dict(h=v.hidden,world=v.world_state()))
    v.selection_counts[selected]+=1;v.review_index+=1
    v.work['A_score_forward_lane_ticks']+=128*128
    v.work['A_score_label_comparisons']+=0
    v.work['A_score_eligible_tick_comparisons']+=int(eligible.sum())
    v.work['A_candidate_presentations']+=128;v.work['A_selected_requests']+=64
    v.work['A_repeated_requests']+=int((old_counts>0).sum());v.work['A_unique_requests']=int((v.selection_counts>0).sum())
    v.work['A_selected_conflict_requests']+=int(bank['conflict'][selected].sum());v.work['old_history_replay_positions']+=192
    v.work['A_moved_WAIT_rows']+=int(moved.sum())
    local_indices=torch.arange(64)
    chosen_margin=score['margin'][ticks,local,chosen_slots]
    chosen_action=score['action'][ticks,local,chosen_slots]
    mid_margin=score['margin'][midpoint[local],local,chosen_slots]
    mid_action=score['action'][midpoint[local],local,chosen_slots]
    row=dict(event='review_update',seed=v.seed,arm=v.arm,round=v.round,epoch=epoch,block=block,index=index,
        candidates=ids.tolist(),selected=selected.tolist(),candidate_id_hash=state_digest(ids),selected_id_hash=state_digest(selected),
        selected_ticks=ticks.tolist(),midpoint_ticks=midpoint[local].tolist(),hard_ticks=score['hard_ticks'][local].tolist(),
        random_ticks=random_ticks[local].tolist(),selected_margins=chosen_margin.tolist(),midpoint_margins=mid_margin.tolist(),
        maximum_margins=score['maximum'][local].tolist(),selected_actions=chosen_action.tolist(),midpoint_actions=mid_action.tolist(),
        selected_finish_intent=chosen_action.ge(3).tolist(),midpoint_finish_intent=mid_action.ge(3).tolist(),
        window_any_finish=score['any_finish'][local].tolist(),window_no_finish_count=int((~score['any_finish'][local]).sum()),
        selected_nonwait_intent_count=int(chosen_action.ne(0).sum()),selected_finish_intent_count=int(chosen_action.ge(3).sum()),
        midpoint_equal_count=int((~moved).sum()),moved_count=int(moved.sum()),eligible_counts=eligible.sum((0,2))[local].tolist(),
        selected_time_to_ready=bank['eligible_time_to_ready'][ticks,selected,chosen_slots].tolist(),
        selected_processing_fraction=bank['eligible_processing_fraction'][ticks,selected,chosen_slots].tolist(),
        supervision_mask_hash=state_digest(mask.cpu()),raw_counts=torch.bincount(bank['raw'][selected],minlength=4).tolist(),
        conflict_count=int(bank['conflict'][selected].sum()),unique_so_far=int((v.selection_counts>0).sum()),
        reused_requests=int((old_counts>0).sum()),max_repetitions=int(v.selection_counts.max()),candidate_count=128,
        candidate_scan_tensor=[128,128,2,11],eligible_tick_comparisons=int(eligible.sum()),scoring_rows=0,
        gradient_rows=192,gradient_nonwait_rows=128,gradient_WAIT_rows=64,denominator=144,
        gradient_tensor=[128,64,2,11],loss=loss,preclip_grad_norm=gradnorm,live_hidden_world_unchanged=True)
    append(v.folder/'selections.jsonl',row)
    if first:v.save('first_review_after')
    if first_moved:v.save('first_moved_after');v.first_moved_saved=True
    return loss


def run(seed):
    torch.set_num_threads(1)
    plan=checked();began=time.perf_counter()
    bank=torch.load(bank_path(seed),map_location='cpu',weights_only=False)
    assert bank['obs'].shape==(128,256,56) and bank['anchor'].shape==(256,192)
    assert torch.all(bank['mask'].sum((0,2))==3)
    assert all(int((bank['raw']==r).sum())==64 for r in range(4))
    assert bank['eligible_mask'].shape==bank['mask'].shape and bank['eligible_mask'].dtype==torch.bool
    assert bank['eligible_mask'].any((0,2)).all()
    gpu_bank={k:bank[k].cuda() for k in ('obs','labels','mask','anchor','eligible_mask')}
    learners=[];bindings=[]
    original=torch.load(origin(seed),map_location='cpu',weights_only=False)
    for arm in ARMS:
        v=ReviewLearner.from_checkpoint(origin(seed),OUT/f'seed_{seed:03d}'/arm)
        v.arm=arm;v.review_protocol_hash=plan['protocol_hash'];v.bank_sha256=t.sha(bank_path(seed))
        v.review_index=0;v.selection_counts=torch.zeros(256,dtype=torch.long);v.first_moved_saved=False
        for key in ('A_gradient_updates','A_gradient_label_positions','A_gradient_nonwait_positions',
                    'B_gradient_updates','B_gradient_nonwait_positions','A_score_forward_lane_ticks',
                    'A_score_label_comparisons','A_candidate_presentations','A_selected_requests',
                    'A_repeated_requests','A_unique_requests','A_selected_conflict_requests',
                    'A_score_eligible_tick_comparisons','A_moved_WAIT_rows'):
            v.work[key]=0
        assert state_digest(v.model.state_dict())==state_digest(original['model'])
        assert state_digest(v.optimizer.state_dict())==state_digest(original['optimizer'])
        assert state_digest(v.world_state())==state_digest(original['world'])
        assert torch.equal(v.hidden,original['hidden'])
        bindings.append(dict(arm=arm,model=state_digest(original['model']),optimizer=state_digest(original['optimizer']),
            hidden=state_digest(original['hidden']),world=state_digest(original['world']),origin_sha256=t.sha(origin(seed))))
        v.stage='B';v.round=0;v.save('B000');learners.append(v)
    folder=OUT/f'seed_{seed:03d}'
    t.write(folder/'bindings.json',bindings)
    shared=learners[0].world
    table=learners[0].table;cue=learners[0].spec['B_cue']
    before_new=int(shared.family_new_counts[...,cue].sum())
    with t.mapping_scope(table):
        shared.set_family(cue);shared.set_regime(duration_range=(1.,16.),gap_range=(.5,32.),name='experience-transfer/B')
        boundary_obs=shared.observe()
    for v in learners:
        policy=deepcopy(v.model).cpu().eval()
        with torch.no_grad():_,v.hidden=policy.step(boundary_obs,v.hidden)
        v.work['environment_ticks']+=1;v.work['lane_ticks']+=64
    boundary_new=int(shared.family_new_counts[...,cue].sum())-before_new
    with t.mapping_scope(table):shared.act(torch.zeros((64,2),dtype=torch.long))
    audit=dict(seed=seed,boundary_B_new_commands=boundary_new,archived_obs_exact=True,archived_labels_exact=True,
        archived_masks_exact=True,archived_actions_exact=True,first_epoch_comparison=[],rounds=[])
    for ri in range(8):
        starttime=time.perf_counter()
        path=t.OUT/f'seed_{seed:03d}'/'related'/f'trajectory_B_{ri:03d}.pt'
        data=torch.load(path,map_location='cpu',weights_only=False)
        with t.mapping_scope(table):
            assert state_digest(shared.state_dict())==state_digest(data['initial_world'])
        anchors={v.arm:v.hidden.clone() for v in learners}
        for v in learners:v.round=ri+1
        first_losses={v.arm:[] for v in learners}
        gpu={k:data[k].cuda() for k in ('obs','labels','mask')}
        data_sha256=t.sha(path)
        block_ids=[dict(trajectory_sha256=data_sha256,tick_start=k*128,tick_stop=(k+1)*128,
            input_identity_hash=state_digest({key:data[key][k*128:(k+1)*128] for key in ('obs','labels','mask','actions')})) for k in range(8)]
        for block in range(8):
            sl=slice(block*128,(block+1)*128)
            policies={v.arm:deepcopy(v.model).cpu().eval() for v in learners}
            live_anchors={v.arm:v.hidden.clone() for v in learners}
            for tick in range(block*128,(block+1)*128):
                with t.mapping_scope(table):obs=shared.observe();target=shared.oracle('midpoint')
                mask=shared.current_family==cue
                labels=torch.where(mask,target,torch.zeros_like(target))
                assert torch.equal(obs,data['obs'][tick]) and torch.equal(mask,data['mask'][tick])
                assert torch.equal(labels,data['labels'][tick]) and torch.equal(labels,data['actions'][tick])
                new=((shared.event_kind==t.worlds.NEW)&(shared.event_family==cue))
                replace=((shared.event_kind==t.worlds.REPLACE)&(shared.event_family==cue))
                for v in learners:
                    with torch.no_grad():_,v.hidden=policies[v.arm].step(obs,v.hidden)
                    v.work['B_unique_new_commands']+=int(new.sum());v.work['B_unique_replace_commands']+=int(replace.sum())
                    v.work['B_unique_label_positions']+=int(mask.sum());v.work['B_unique_nonwait_labels']+=int((labels!=0).sum())
                with t.mapping_scope(table):shared.act(data['actions'][tick])
            for v in learners:
                v.world=shared;v.work['environment_ticks']+=128;v.work['lane_ticks']+=128*64
                v.step_identity=dict(epoch=1,block=block,**block_ids[block])
                _,loss,_=v.update(gpu['obs'][sl],gpu['labels'][sl],gpu['mask'][sl],live_anchors[v.arm].cuda(),kind='B')
                first_losses[v.arm].append(loss)
                if ri==0 and block+1 in (2,8):
                    name=f'Bfirst{block+1:03d}'
                    v.save(name)
                    reference=torch.load(t.OUT/f'seed_{seed:03d}'/'related'/(name+'.pt'),map_location='cpu',weights_only=False)
                    delta=max(float((value.cpu()-reference['model'][key]).abs().max()) for key,value in v.model.state_dict().items())
                    hdelta=float((v.hidden-reference['hidden']).abs().max())
                    world_exact=state_digest(v.world_state())==state_digest(reference['world'])
                    audit['first_epoch_comparison'].append(dict(arm=v.arm,snapshot=name,model_max_abs_difference=delta,
                        hidden_max_abs_difference=hdelta,world_exact=world_exact,
                        Adam_exact=state_digest(v.optimizer.state_dict())==state_digest(reference['optimizer'])))
                    assert delta<1e-4 and hdelta<1e-3 and world_exact
        for epoch in range(2,33):
            h={v.arm:anchors[v.arm].cuda() for v in learners}
            for block in range(8):
                sl=slice(block*128,(block+1)*128)
                for v in learners:
                    if epoch in REVIEW_EPOCHS:
                        # Continue the disposable B pass; A gradients must not leak A h into it.
                        with torch.no_grad():_,h[v.arm]=v.model(gpu['obs'][sl],h[v.arm])
                        review_update(v,bank,gpu_bank,epoch,block)
                        v.work['B_replay_no_loss_lane_ticks']=v.work.get('B_replay_no_loss_lane_ticks',0)+128*64
                    else:
                        v.step_identity=dict(epoch=epoch,block=block,**block_ids[block])
                        h[v.arm],_,_=v.update(gpu['obs'][sl],gpu['labels'][sl],gpu['mask'][sl],h[v.arm],kind='B')
        for v in learners:
            v.work['B_actual_NEW_including_boundary']=int(shared.family_new_counts[...,cue].sum())-before_new
            assert v.work['B_gradient_updates']==(ri+1)*224
            assert v.work['A_gradient_updates']==(ri+1)*32
            assert v.work['A_gradient_label_positions']==(ri+1)*32*192
            row=dict(event='B_round',seed=seed,arm=v.arm,round=ri+1,seconds=time.perf_counter()-starttime,
                first_epoch_block_loss=first_losses[v.arm],work=deepcopy(v.work))
            v.log(row)
            if ri+1 in (1,2,4,8):v.save(f'B{ri+1:03d}')
        with t.mapping_scope(table):world_hash=state_digest(shared.state_dict())
        audit['rounds'].append(dict(round=ri+1,public_stream_sha256=t.sha(path),world_hash=world_hash,
            actual_B_NEW=int(shared.family_new_counts[...,cue].sum())-before_new))
        t.write(folder/'stream_audit.json',audit)
    from experiments.wait_position import last_audit as train_audit
    replay=train_audit.audit(seed)
    checked()
    t.write(folder/'complete.json',dict(passed=True,seed=seed,protocol_hash=plan['protocol_hash'],seconds=time.perf_counter()-began,
        work={v.arm:v.work for v in learners},first_step_audit=replay,public_stream_audit=audit))
    print(json.dumps(dict(event='complete',seed=seed,seconds=time.perf_counter()-began)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('declare','run','all'))
    parser.add_argument('--seed',type=int,choices=range(3))
    args=parser.parse_args()
    if args.command=='declare':declare()
    elif args.command=='run':run(args.seed)
    else:
        processes=[]
        for seed in range(3):
            stream=(OUT/f'seed_{seed:03d}.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u','-m','experiments.wait_position.last_train','run','--seed',str(seed)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            processes.append((seed,proc,stream))
        results=[]
        for seed,proc,stream in processes:
            results.append(dict(seed=seed,exit_code=proc.wait()));stream.close()
            t.write(OUT/'exit_codes.json',results)
        if any(r['exit_code'] for r in results):raise RuntimeError(results)


if __name__=='__main__':main()
