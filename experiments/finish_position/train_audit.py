"""Independent reconstruction of FINISH labels, masks, IDs, and Adam updates."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.continuity.run import state_digest
from experiments.learn_remember.gru_train import GRUAgent

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/finish-position/train-v1'


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def independent_ids(bank,seed,index):
    rng=np.random.default_rng(1140000+seed*10000+index)
    raw=bank['raw'].numpy()
    ids=np.concatenate([rng.choice(np.flatnonzero(raw==r),32,replace=False) for r in range(4)])
    rng.shuffle(ids)
    ids=torch.from_numpy(ids)
    local=torch.from_numpy(np.random.default_rng(2140000+seed*10000+index).choice(128,64,replace=False))
    return ids,local,ids[local]


def independent_supervision(bank,selected,arm):
    # Rebuild from original sparse bank rather than trusting prebuilt arm masks.
    labels=bank['labels'][:,selected].clone()
    mask=bank['mask'][:,selected].clone()
    slot=bank['target_slot'][selected]
    batch=torch.arange(len(selected))
    oldwait=bank['positions'][selected,1]
    oldfinish=bank['positions'][selected,2]
    wait=torch.tensor([torch.where(bank['eligible_mask'][:,int(i),int(s)])[0][-1] for i,s in zip(selected,slot)])
    finish=oldfinish if arm=='midpoint' else bank['first_legal_positions'][selected]
    correct=bank['labels'][oldfinish,selected,slot]
    assert correct.ge(3).all() and correct.le(10).all()
    mask[oldwait,batch,slot]=False
    mask[wait,batch,slot]=True
    if arm=='first':
        mask[oldfinish,batch,slot]=False
        mask[finish,batch,slot]=True
        labels[finish,batch,slot]=correct
    assert torch.equal(wait,bank['last_wait_positions'][selected])
    assert torch.equal(oldfinish,bank['teacher_finish_positions'][selected])
    assert torch.equal(correct,bank['finish_labels'][selected])
    assert torch.equal(labels,bank['labels' if arm=='midpoint' else 'first_labels'][:,selected])
    assert torch.equal(mask,bank[arm+'_mask'][:,selected])
    assert mask[bank['positions'][selected,0],batch,slot].all()
    assert labels[bank['positions'][selected,0],batch,slot].gt(0).all()
    assert mask[wait,batch,slot].all() and labels[wait,batch,slot].eq(0).all()
    assert mask[finish,batch,slot].all() and torch.equal(labels[finish,batch,slot],correct)
    assert mask.sum((0,2)).eq(3).all()
    assert int(mask.sum())==192 and int((mask&(labels!=0)).sum())==128
    assert int((mask&(labels==0)).sum())==64
    return labels,mask,wait,finish,correct


def compare_checkpoint(left,right):
    return dict(model_exact=all(torch.equal(value,right['model'][key]) for key,value in left['model'].items()),
        Adam_exact=state_digest(left['optimizer'])==state_digest(right['optimizer']),
        hidden_exact=torch.equal(left['hidden'],right['hidden']),
        world_exact=state_digest(left['world'])==state_digest(right['world']))


def audit(seed):
    torch.set_num_threads(1)
    bank=load(ROOT/f'data/finish-position/bank-v1/seed_{seed:03d}/bank.pt')
    folder=OUT/f'seed_{seed:03d}'
    oldfolder=ROOT/f'data/wait-position/last-v1/seed_{seed:03d}/last'
    oldrows=rows(oldfolder/'selections.jsonl')
    oldsteps=rows(oldfolder/'step_updates.jsonl')
    reports=[];budgets={};calibration={};label_counts={};all_selections={}
    for arm in ('midpoint','first'):
        armfolder=folder/arm
        tag='first_review' if arm=='midpoint' else 'first_moved'
        before=load(armfolder/(tag+'_before.pt'))
        after=load(armfolder/(tag+'_after.pt'))
        record=load(armfolder/('first_selection.pt' if arm=='midpoint' else 'first_moved_selection.pt'))
        ids,local,selected=independent_ids(bank,seed,record['review_index'])
        assert torch.equal(ids,record['candidate_ids'])
        assert torch.equal(local,record['selected_local']) and torch.equal(selected,record['selected_ids'])
        labels,mask,wait,finish,correct=independent_supervision(bank,selected,arm)
        assert torch.equal(labels,record['supervision_labels']) and torch.equal(mask,record['supervision_mask'])
        assert torch.equal(wait,record['selected_wait_ticks']) and torch.equal(finish,record['selected_finish_ticks'])
        assert torch.equal(correct,record['finish_labels'])
        if arm=='first':assert finish.ne(bank['positions'][selected,2]).any()
        model=GRUAgent(**before['model_config']).cuda();model.load_state_dict(before['model'])
        optimizer=torch.optim.Adam(model.parameters(),lr=before['config']['lr'])
        optimizer.load_state_dict(deepcopy(before['optimizer']))
        obs=bank['obs'][:,ids].contiguous().cuda();anchor=bank['anchor'][ids].contiguous().cuda()
        with torch.no_grad():
            logits,_=model(obs,anchor)
            margin=logits[...,3:11].amax(-1)-logits[...,0:3].amax(-1)
            eligible=bank['eligible_mask'][:,ids].cuda()
            maximum,flat=margin.masked_fill(~eligible,-torch.inf).permute(1,0,2).reshape(128,-1).max(-1)
            hard=flat//2
            any_finish=(eligible&logits.argmax(-1).ge(3)).any((0,2))
        assert torch.equal(hard.cpu(),record['candidate_hard_ticks'])
        assert torch.equal(maximum.cpu(),record['candidate_maximum'])
        assert torch.equal(any_finish.cpu(),record['candidate_any_finish'])
        logits,branch_h=model(obs[:,local].contiguous(),anchor[local].contiguous())
        labels,mask=labels.cuda(),mask.cuda()
        weights=torch.ones(11,device='cuda');weights[0]=before['config']['wait_weight']
        ce=F.cross_entropy(logits.reshape(-1,11),labels.reshape(-1),weight=weights,reduction='none').reshape_as(labels)
        denominator=(weights[labels]*mask).sum();loss=(ce*mask).sum()/denominator
        assert float(denominator)==144
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        optimizer.step()
        exact=dict(model_exact=all(torch.equal(value.cpu(),after['model'][key]) for key,value in model.state_dict().items()),
            Adam_exact=state_digest(optimizer.state_dict())==state_digest(after['optimizer']),
            live_hidden_unchanged=torch.equal(before['hidden'],after['hidden']),
            world_unchanged=state_digest(before['world'])==state_digest(after['world']))
        assert all(exact.values()),exact
        reports.append(dict(arm=arm,tag=tag,review_index=record['review_index'],independent_candidate_scores_labels_mask_exact=True,
            true_label_rows=192,nonwait_rows=128,WAIT_rows=64,denominator=144,loss=float(loss.detach()),preclip_grad_norm=gradnorm,**exact))

        selections=rows(armfolder/'selections.jsonl');all_selections[arm]=selections
        assert len(selections)==len(oldrows)==256
        counts=dict(updates=0,BEGIN=0,WAIT=0,FINISH=0,moved_FINISH=0,inserted_FINISH_over_teacher_WAIT=0)
        for index,(row,old) in enumerate(zip(selections,oldrows)):
            ids,local,selected=independent_ids(bank,seed,index)
            assert row['index']==index and row['candidates']==ids.tolist() and row['selected']==selected.tolist()
            assert row['candidates']==old['candidates'] and row['selected']==old['selected']
            labels,mask,wait,finish,correct=independent_supervision(bank,selected,arm)
            assert row['selected_wait_ticks']==wait.tolist() and row['selected_finish_ticks']==finish.tolist()
            assert row['selected_wait_ticks']==old['selected_ticks']
            assert row['finish_labels']==correct.tolist()
            assert row['supervision_mask_hash']==state_digest(mask)
            assert row['supervision_labels_hash']==state_digest(labels)
            assert row['supervised_identity_hash']==state_digest(dict(mask=mask,labels=labels[mask]))
            assert (row['gradient_rows'],row['gradient_nonwait_rows'],row['gradient_WAIT_rows'],row['denominator'])==(192,128,64,144)
            assert row['live_hidden_world_unchanged']
            supervised=labels[mask]
            assert int(((supervised>=1)&(supervised<=2)).sum())==64
            assert int((supervised>=3).sum())==64
            counts['updates']+=1;counts['BEGIN']+=64;counts['WAIT']+=64;counts['FINISH']+=64
            moved=finish.ne(bank['positions'][selected,2])
            overwritten=(moved&bank['labels'][finish,selected,bank['target_slot'][selected]].eq(0))
            assert int(moved.sum())==row['moved_finish_count']
            assert int(overwritten.sum())==row['inserted_FINISH_over_teacher_WAIT_count']
            counts['moved_FINISH']+=int(moved.sum())
            counts['inserted_FINISH_over_teacher_WAIT']+=int(overwritten.sum())
        label_counts[arm]=counts
        steps=rows(armfolder/'step_updates.jsonl')
        assert len(steps)==len(oldsteps)==2048
        assert sum(row['kind']=='A' for row in steps)==256
        assert sum(row['kind']=='B' for row in steps)==1792
        for current,old in zip(steps,oldsteps):
            assert current['kind']==old['kind'] and current['step']==old['step']
            if current['kind']=='B':
                for key in ('round','epoch','block','trajectory_sha256','tick_start','tick_stop','input_identity_hash','supervised_rows','nonwait_rows','denominator'):
                    assert current[key]==old[key],(arm,key)
            else:
                review=selections[current['review_index']]
                assert current['candidate_id_hash']==review['candidate_id_hash']
                assert current['selected_id_hash']==review['selected_id_hash']
                assert current['supervision_mask_hash']==review['supervision_mask_hash']
                assert current['supervision_labels_hash']==review['supervision_labels_hash']
                assert (current['supervised_rows'],current['nonwait_rows'],current['denominator'])==(192,128,144)
        initial=load(armfolder/'B000.pt');final=load(armfolder/'B008.pt');w=final['work']
        assert w['A_gradient_updates']==256 and w['B_gradient_updates']==1792
        assert w['gradient_updates']-initial['work']['gradient_updates']==2048
        budgets[arm]=dict(A_updates=256,B_updates=1792,total_new_updates=2048,A_rows=49152,
            A_BEGIN_rows=16384,A_WAIT_rows=16384,A_FINISH_rows=16384,A_nonWAIT_rows=32768,A_denominator_sum=36864,
            A_candidate_requests=w['A_candidate_presentations'],A_selected_requests=w['A_selected_requests'],
            A_score_forward_lane_ticks=w['A_score_forward_lane_ticks'],A_score_eligible_tick_comparisons=w['A_score_eligible_tick_comparisons'],
            B_gradient_label_positions=w['B_gradient_label_positions'],B_gradient_nonwait_positions=w['B_gradient_nonwait_positions'],
            B_unique_label_positions=w['B_unique_label_positions'],B_unique_NEW=w['B_unique_new_commands'],
            B_actual_NEW_including_boundary=w['B_actual_NEW_including_boundary'],
            gradient_lane_ticks=w['gradient_lane_ticks']-initial['work']['gradient_lane_ticks'])
        assert compare_checkpoint(initial,load(oldfolder/'B000.pt'))==dict(model_exact=True,Adam_exact=True,hidden_exact=True,world_exact=True)
    assert budgets['midpoint']==budgets['first']
    for name in ('B000','Bfirst002','Bfirst008','first_review_before','B001','B002','B004','B008'):
        comparison=compare_checkpoint(load(folder/f'midpoint/{name}.pt'),load(oldfolder/(name+'.pt')))
        assert all(comparison.values()),(name,comparison)
        calibration[name]=comparison
    initial_moved=load(folder/'first/first_moved_before.pt')
    initial_mid=load(folder/'midpoint/first_review_before.pt')
    assert all(compare_checkpoint(initial_moved,initial_mid).values())
    assert label_counts['midpoint']['moved_FINISH']==0
    assert label_counts['first']['moved_FINISH']>0
    result=dict(seed=seed,passed=True,reports=reports,historical_last_all_snapshots_bit_exact=calibration,
        first_changed_update_before_states_match=True,all256_request_ids_labels_and_masks_reconstructed=True,
        all_B_update_input_identities_match_historical=True,all_arm_integer_budgets_equal=True,
        budgets=budgets,label_counts=label_counts)
    (folder/'first_step_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    print(json.dumps(audit(parser.parse_args().seed)),flush=True)
