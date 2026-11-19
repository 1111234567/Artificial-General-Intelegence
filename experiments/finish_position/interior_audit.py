"""Independent native legality and optimizer audit for the fixed interior point."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from experiments.continuity.run import state_digest
from experiments.experience_transfer import train_transfer as t
from experiments.learn_remember.gru_train import GRUAgent
from experiments.finish_position import train_audit as base

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/finish-position/interior-v1'
BANK=ROOT/'data/finish-position/bank-v1'


def supervision(bank,selected):
    labels=bank['labels'][:,selected].clone();mask=bank['mask'][:,selected].clone()
    row=torch.arange(len(selected));slot=bank['target_slot'][selected]
    teacher=bank['positions'][selected,2]
    first=bank['first_legal_positions'][selected]
    interior=(first+teacher)//2
    wait=torch.tensor([torch.where(bank['eligible_mask'][:,int(i),int(s)])[0][-1] for i,s in zip(selected,slot)])
    correct=3+4*torch.tensor([bank['metadata'][int(i)]['version'] for i in selected])+torch.tensor([bank['metadata'][int(i)]['A_target'] for i in selected])
    assert torch.equal(bank['labels'][teacher,selected,slot],correct)
    assert (first<interior).all() and (interior<teacher).all()
    mask[bank['positions'][selected,1],row,slot]=False;mask[wait,row,slot]=True
    mask[teacher,row,slot]=False;mask[interior,row,slot]=True
    labels[interior,row,slot]=correct
    assert mask.sum((0,2)).eq(3).all()
    assert labels[wait,row,slot].eq(0).all()
    for j,i in enumerate(selected):
        m=bank['metadata'][int(i)]
        assert int(labels[m['positions'][0],j,int(slot[j])])==1+m['version']
        assert labels[:,j][mask[:,j]].tolist()==[1+m['version'],0,int(correct[j])]
    assert int(mask.sum())==192 and int((mask&(labels!=0)).sum())==128
    return labels,mask,wait,interior,correct


def native_legality(seed):
    torch.set_num_threads(1);began=time.perf_counter()
    bank=base.load(BANK/f'seed_{seed:03d}/bank.pt')
    manifest=json.loads((BANK/f'seed_{seed:03d}/manifest.json').read_text())
    groups=defaultdict(list)
    for i,row in enumerate(bank['metadata']):groups[row['source']].append(i)
    records=[];ticks_checked=0
    for path,indices in sorted(groups.items()):
        assert t.sha(ROOT/path)==manifest['sources'][path]
        trace=base.load(ROOT/path)
        targets=defaultdict(list)
        for i in indices:
            row=bank['metadata'][i]
            local=(int(bank['first_legal_positions'][i])+int(bank['teacher_finish_positions'][i]))//2
            tick=row['start_tick']+local
            assert int(bank['first_legal_positions'][i])<local<int(bank['teacher_finish_positions'][i])
            targets[tick].append((i,local))
            assert torch.equal(bank['obs'][:,i],trace['obs'][row['start_tick']:row['stop_tick'],row['lane']])
            assert torch.equal(bank['actions'][:,i],trace['actions'][row['start_tick']:row['stop_tick'],row['lane']])
        with t.mapping_scope(t.mapping_table(seed,'related')):
            world=t.worlds.ContextWorld.from_state_dict(trace['initial_world'],device='cpu')
            for tick in range(len(trace['obs'])):
                obs=world.observe();teacher=world.oracle('midpoint')
                assert torch.equal(obs,trace['obs'][tick]) and torch.equal(teacher,trace['labels'][tick])
                if targets[tick]:
                    earliest=world.oracle('earliest');trial=deepcopy(world);actions=torch.zeros_like(teacher)
                    for i,position in targets[tick]:
                        row=bank['metadata'][i];lane,slot=row['lane'],row['slot']
                        assert int(world.actor_ids[lane])==row['actor_id']
                        assert int(world.request_id[lane,slot])==row['request_id']
                        assert int(world.version[lane,slot])==row['version']
                        assert int(world.raw_payload[lane,slot])==row['raw']
                        assert int(world.current_family[lane,slot])==row['context']
                        assert int(world.payload[lane,slot])==row['A_target']
                        assert bool(world.active[lane,slot]) and bool(world.started[lane,slot])
                        assert world.now>=float(world.ready_time[lane,slot])-1e-6
                        assert world.now<=float(world.deadline[lane,slot])+1e-6
                        correct=3+4*row['version']+row['A_target']
                        assert int(earliest[lane,slot])==correct==int(bank['finish_labels'][i])
                        assert int(teacher[lane,slot])==int(bank['labels'][position,i,slot])==0
                        actions[lane,slot]=correct
                        records.append(dict(sample_index=i,id=bank['ids'][i],source=path,source_tick=tick,
                            interior_position=position,first_position=int(bank['first_legal_positions'][i]),
                            teacher_position=int(bank['teacher_finish_positions'][i]),correct_FINISH=correct,
                            actor_id=row['actor_id'],lane=lane,slot=slot,request_id=row['request_id'],
                            raw=row['raw'],version=row['version'],context=row['context'],payload=row['A_target'],
                            now=world.now,ready_time=float(world.ready_time[lane,slot]),deadline=float(world.deadline[lane,slot])))
                    trial.act(actions)
                    for i,position in targets[tick]:
                        row=bank['metadata'][i];lane,slot=row['lane'],row['slot']
                        assert int(trial.last_result[lane,slot])==1
                        assert not bool(trial.active[lane,slot]) and not bool(trial.started[lane,slot])
                world.act(trace['actions'][tick]);ticks_checked+=1
    assert len(records)==len({r['sample_index'] for r in records})==256
    records.sort(key=lambda row:row['sample_index'])
    folder=OUT/f'seed_{seed:03d}';folder.mkdir(parents=True,exist_ok=False)
    t.write(folder/'native_interior_positions.json',records)
    result=dict(seed=seed,passed=True,requests=256,native_counterfactual_FINISH_accepted=256,
        original_public_obs_and_actions_exact=True,original_teacher_labels_exact=True,
        actor_slot_request_raw_version_context_payload_identity_exact=True,
        active_started_readiness_deadline_native_earliest_verified=True,
        positions_reconstructed_as_integer_floor=True,original_interior_teacher_labels_all_WAIT=True,
        replay_world_ticks=ticks_checked,metadata_sha256=t.sha(folder/'native_interior_positions.json'),
        bank_sha256=t.sha(BANK/f'seed_{seed:03d}/bank.pt'),seconds=time.perf_counter()-began)
    t.write(folder/'native_legality_audit.json',result)
    print(json.dumps(dict(event='native_interior_legality',**result)),flush=True)
    return result


def audit(seed):
    torch.set_num_threads(1)
    bank=base.load(BANK/f'seed_{seed:03d}/bank.pt');folder=OUT/f'seed_{seed:03d}';armfolder=folder/'interior'
    native=json.loads((folder/'native_legality_audit.json').read_text());assert native['passed']
    assert native['metadata_sha256']==t.sha(folder/'native_interior_positions.json')
    before=base.load(armfolder/'first_moved_before.pt');after=base.load(armfolder/'first_moved_after.pt')
    record=base.load(armfolder/'first_moved_selection.pt')
    ids,local,selected=base.independent_ids(bank,seed,record['review_index'])
    assert torch.equal(ids,record['candidate_ids']) and torch.equal(local,record['selected_local'])
    assert torch.equal(selected,record['selected_ids'])
    labels,mask,wait,finish,correct=supervision(bank,selected)
    assert torch.equal(labels,record['supervision_labels']) and torch.equal(mask,record['supervision_mask'])
    assert torch.equal(wait,record['selected_wait_ticks']) and torch.equal(finish,record['selected_finish_ticks'])
    assert torch.equal(correct,record['finish_labels'])
    model=GRUAgent(**before['model_config']).cuda();model.load_state_dict(before['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=before['config']['lr']);optimizer.load_state_dict(deepcopy(before['optimizer']))
    obs=bank['obs'][:,ids].contiguous().cuda();anchor=bank['anchor'][ids].contiguous().cuda()
    with torch.no_grad():
        logits,_=model(obs,anchor)
        margin=logits[...,3:11].amax(-1)-logits[...,0:3].amax(-1);eligible=bank['eligible_mask'][:,ids].cuda()
        maximum,flat=margin.masked_fill(~eligible,-torch.inf).permute(1,0,2).reshape(128,-1).max(-1)
        assert torch.equal((flat//2).cpu(),record['candidate_hard_ticks'])
        assert torch.equal(maximum.cpu(),record['candidate_maximum'])
        assert torch.equal((eligible&logits.argmax(-1).ge(3)).any((0,2)).cpu(),record['candidate_any_finish'])
    logits,_=model(obs[:,local].contiguous(),anchor[local].contiguous());labels,mask=labels.cuda(),mask.cuda()
    weights=torch.ones(11,device='cuda');weights[0]=before['config']['wait_weight']
    ce=F.cross_entropy(logits.reshape(-1,11),labels.reshape(-1),weight=weights,reduction='none').reshape_as(labels)
    denominator=(weights[labels]*mask).sum();loss=(ce*mask).sum()/denominator
    assert float(denominator)==144
    optimizer.zero_grad(set_to_none=True);loss.backward();gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));optimizer.step()
    exact=dict(model_exact=all(torch.equal(v.cpu(),after['model'][k]) for k,v in model.state_dict().items()),
        Adam_exact=state_digest(optimizer.state_dict())==state_digest(after['optimizer']),
        live_hidden_unchanged=torch.equal(before['hidden'],after['hidden']),
        world_unchanged=state_digest(before['world'])==state_digest(after['world']))
    assert all(exact.values()),exact
    parent=ROOT/f'data/finish-position/train-v1/seed_{seed:03d}/midpoint'
    calibration={}
    for name in ('B000','Bfirst002','Bfirst008','first_review_before'):
        calibration[name]=base.compare_checkpoint(base.load(armfolder/(name+'.pt')),base.load(parent/(name+'.pt')))
        assert all(calibration[name].values())
    changed_delta=max(float((v-base.load(parent/'first_review_after.pt')['model'][k]).abs().max()) for k,v in after['model'].items())
    assert changed_delta>0
    oldfolder=ROOT/f'data/wait-position/last-v1/seed_{seed:03d}/last'
    selections=base.rows(armfolder/'selections.jsonl');oldrows=base.rows(oldfolder/'selections.jsonl')
    assert len(selections)==len(oldrows)==256
    for index,(row,old) in enumerate(zip(selections,oldrows)):
        ids,local,selected=base.independent_ids(bank,seed,index)
        labels,mask,wait,finish,correct=supervision(bank,selected)
        assert row['index']==index and row['candidates']==ids.tolist()==old['candidates']
        assert row['selected']==selected.tolist()==old['selected']
        assert row['selected_wait_ticks']==wait.tolist()==old['selected_ticks']
        assert row['selected_finish_ticks']==finish.tolist() and row['finish_labels']==correct.tolist()
        assert row['supervision_mask_hash']==state_digest(mask) and row['supervision_labels_hash']==state_digest(labels)
        assert row['supervised_identity_hash']==state_digest(dict(mask=mask,labels=labels[mask]))
        assert (row['gradient_rows'],row['gradient_nonwait_rows'],row['gradient_WAIT_rows'],row['denominator'])==(192,128,64,144)
    steps=base.rows(armfolder/'step_updates.jsonl');oldsteps=base.rows(oldfolder/'step_updates.jsonl')
    assert len(steps)==len(oldsteps)==2048
    assert sum(r['kind']=='A' for r in steps)==256 and sum(r['kind']=='B' for r in steps)==1792
    for row,old in zip(steps,oldsteps):
        assert row['kind']==old['kind'] and row['step']==old['step']
        if row['kind']=='B':
            for key in ('round','epoch','block','trajectory_sha256','tick_start','tick_stop','input_identity_hash','supervised_rows','nonwait_rows','denominator'):
                assert row[key]==old[key]
        else:
            actual=selections[row['review_index']]
            for key in ('candidate_id_hash','selected_id_hash','supervision_mask_hash','supervision_labels_hash'):assert row[key]==actual[key]
            assert (row['supervised_rows'],row['nonwait_rows'],row['denominator'])==(192,128,144)
    initial=base.load(armfolder/'B000.pt');final=base.load(armfolder/'B008.pt');historical=base.load(oldfolder/'B008.pt');w=final['work']
    assert all(w[k]==v for k,v in historical['work'].items())
    assert w['A_gradient_updates']==256 and w['B_gradient_updates']==1792 and w['A_moved_FINISH_rows']==16384
    assert w['gradient_updates']-initial['work']['gradient_updates']==2048
    budget=dict(A_updates=256,B_updates=1792,total_new_updates=2048,A_rows=49152,A_BEGIN_rows=16384,A_WAIT_rows=16384,A_FINISH_rows=16384,
        A_denominator_sum=36864,A_candidate_presentations=w['A_candidate_presentations'],A_selected_requests=w['A_selected_requests'],
        A_score_forward_lane_ticks=w['A_score_forward_lane_ticks'],B_gradient_label_positions=w['B_gradient_label_positions'],
        B_gradient_nonwait_positions=w['B_gradient_nonwait_positions'],B_unique_label_positions=w['B_unique_label_positions'],
        B_unique_NEW=w['B_unique_new_commands'],B_actual_NEW_including_boundary=w['B_actual_NEW_including_boundary'],
        gradient_lane_ticks=w['gradient_lane_ticks']-initial['work']['gradient_lane_ticks'],B_replay_no_loss_lane_ticks=w['B_replay_no_loss_lane_ticks'])
    result=dict(seed=seed,passed=True,first_changed_review_index=record['review_index'],first_changed_step_exact=exact,
        first_changed_loss=float(loss.detach()),first_changed_preclip_grad_norm=gradnorm,first_changed_model_delta_from_midpoint=changed_delta,
        all256_request_labels_masks_reconstructed=True,all_B_update_identities_and_original_work_counters_match=True,
        before_intervention_main_snapshots_bit_exact=calibration,native_legality=native,budgets={'interior':budget})
    t.write(folder/'first_step_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('native','train'));parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();print(json.dumps((native_legality if args.command=='native' else audit)(args.seed)),flush=True)
