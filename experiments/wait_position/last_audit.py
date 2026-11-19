"""Independent first static-last WAIT mask and Adam-step audit."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.learn_remember.gru_train import GRUAgent
from experiments.continuity.run import state_digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/wait-position/last-v1'


def compare_checkpoint(left,right):
    return dict(model_exact=all(torch.equal(v.cpu(),right['model'][k].cpu()) for k,v in left['model'].items()),
        Adam_exact=state_digest(left['optimizer'])==state_digest(right['optimizer']),
        hidden_exact=torch.equal(left['hidden'],right['hidden']),
        world_exact=state_digest(left['world'])==state_digest(right['world']))


def audit(seed):
    torch.set_num_threads(1)
    bank=torch.load(ROOT/f'data/wait-position/bank-v1/seed_{seed:03d}/bank.pt',map_location='cpu',weights_only=False)
    reports=[]
    for arm in ('last',):
        folder=OUT/f'seed_{seed:03d}'/arm
        tag='first_review' if arm=='midpoint' else 'first_moved'
        before=torch.load(folder/(tag+'_before.pt'),map_location='cpu',weights_only=False)
        after=torch.load(folder/(tag+'_after.pt'),map_location='cpu',weights_only=False)
        record=torch.load(folder/('first_selection.pt' if arm=='midpoint' else 'first_moved_selection.pt'),map_location='cpu',weights_only=False)
        index=record['review_index']
        rng=np.random.default_rng(1140000+seed*10000+index)
        raw=bank['raw'].numpy()
        ids=np.concatenate([rng.choice(np.flatnonzero(raw==r),32,replace=False) for r in range(4)])
        rng.shuffle(ids);ids=torch.from_numpy(ids)
        local=torch.from_numpy(np.random.default_rng(2140000+seed*10000+index).choice(128,64,replace=False))
        assert torch.equal(ids,record['candidate_ids']) and torch.equal(local,record['selected_local'])
        selected=ids[local];assert torch.equal(selected,record['selected_ids'])
        model=GRUAgent(**before['model_config']).cuda();model.load_state_dict(before['model'])
        optimizer=torch.optim.Adam(model.parameters(),lr=before['config']['lr']);optimizer.load_state_dict(deepcopy(before['optimizer']))
        obs=bank['obs'][:,ids].contiguous().cuda();anchor=bank['anchor'][ids].contiguous().cuda()
        with torch.no_grad():
            logits,_=model(obs,anchor)
            margin=logits[...,3:11].amax(-1)-logits[...,0:3].amax(-1)
            eligible=bank['eligible_mask'][:,ids].cuda()
            best,flattened=margin.masked_fill(~eligible,-torch.inf).permute(1,0,2).reshape(128,-1).max(-1)
            hard_ticks=flattened//2
            any_finish=(eligible&logits.argmax(-1).ge(3)).any((0,2))
        assert torch.equal(hard_ticks.cpu(),record['candidate_hard_ticks'])
        assert torch.equal(best.cpu(),record['candidate_maximum'])
        assert torch.equal(any_finish.cpu(),record['candidate_any_finish'])
        slots=bank['target_slot'][ids]
        if arm=='last':ticks=torch.tensor([torch.where(bank['eligible_mask'][:,ids[i],slots[i]])[0][-1] for i in range(128)])
        elif arm=='midpoint':ticks=bank['positions'][ids,1]
        elif arm=='hard':ticks=hard_ticks.cpu()
        else:
            rng=np.random.default_rng(3140000+seed*10000+index)
            ticks=torch.tensor([rng.choice(torch.where(bank['eligible_mask'][:,ids[i],slots[i]])[0].numpy()) for i in range(128)])
        assert torch.equal(ticks,record['candidate_ticks'])
        mask=bank['mask'][:,selected].clone()
        mask[bank['positions'][selected,1],torch.arange(64),bank['target_slot'][selected]]=False
        mask[ticks[local],torch.arange(64),bank['target_slot'][selected]]=True
        assert torch.equal(mask,record['supervision_mask'])
        if arm!='midpoint':assert ticks[local].ne(bank['positions'][selected,1]).any()
        # Every original BEGIN and FINISH row remains; target slot only, exactly 3.
        for j in (0,2):assert mask[bank['positions'][selected,j],torch.arange(64),bank['target_slot'][selected]].all()
        assert torch.all(mask.sum((0,2))==3)
        labels=bank['labels'][:,selected].contiguous().cuda();mask=mask.cuda()
        logits,branch_h=model(obs[:,local].contiguous(),anchor[local].contiguous())
        weights=torch.ones(11,device='cuda');weights[0]=before['config']['wait_weight']
        ce=F.cross_entropy(logits.reshape(-1,11),labels.reshape(-1),weight=weights,reduction='none').reshape_as(labels)
        denominator=(weights[labels]*mask).sum();loss=(ce*mask).sum()/denominator
        assert int(mask.sum())==192 and float(denominator)==144
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        exact=dict(model_exact=all(torch.equal(v.cpu(),after['model'][k]) for k,v in model.state_dict().items()),
            Adam_exact=state_digest(optimizer.state_dict())==state_digest(after['optimizer']),
            live_hidden_unchanged=torch.equal(before['hidden'],after['hidden']),
            world_unchanged=state_digest(before['world'])==state_digest(after['world']))
        assert all(exact.values()),exact
        reports.append(dict(arm=arm,tag=tag,review_index=index,independent_candidate_selection_scores_mask_exact=True,
            true_label_rows=192,nonwait_rows=128,WAIT_rows=64,denominator=144,loss=float(loss.detach()),**exact))
    prior=ROOT/f'data/review-selection/train-v1/seed_{seed:03d}/uniform'
    calibration={}
    for name in ('B000','Bfirst002','Bfirst008','first_review_before'):
        new=torch.load(OUT/f'seed_{seed:03d}/last/{name}.pt',map_location='cpu',weights_only=False)
        old=torch.load(prior/(name+'.pt'),map_location='cpu',weights_only=False)
        calibration[name]=compare_checkpoint(new,old)
        assert all(calibration[name].values()),(name,calibration[name])
    selections={arm:[json.loads(line) for line in (OUT/f'seed_{seed:03d}/{arm}/selections.jsonl').read_text().splitlines()]
                for arm in ('last',)}
    old_selections=[json.loads(line) for line in (prior/'selections.jsonl').read_text().splitlines()]
    budgets={};timing={}
    for arm,rows in selections.items():
        assert len(rows)==256
        for new,old in zip(rows,old_selections):
            assert new['candidates']==old['candidates'] and new['selected']==old['selected']
            assert new['gradient_rows']==192 and new['gradient_nonwait_rows']==128 and new['gradient_WAIT_rows']==64
            assert new['denominator']==144
        steps=[json.loads(line) for line in (OUT/f'seed_{seed:03d}/{arm}/step_updates.jsonl').read_text().splitlines()]
        assert len(steps)==2048 and sum(row['kind']=='A' for row in steps)==256 and sum(row['kind']=='B' for row in steps)==1792
        final=torch.load(OUT/f'seed_{seed:03d}/{arm}/B008.pt',map_location='cpu',weights_only=False)
        initial=torch.load(OUT/f'seed_{seed:03d}/{arm}/B000.pt',map_location='cpu',weights_only=False)
        w=final['work'];assert w['A_gradient_updates']==256 and w['B_gradient_updates']==1792
        assert w['gradient_updates']-initial['work']['gradient_updates']==2048
        budgets[arm]=dict(A_updates=256,B_updates=1792,total_new_updates=2048,
            A_rows=sum(row['gradient_rows'] for row in rows),A_nonWAIT_rows=sum(row['gradient_nonwait_rows'] for row in rows),
            A_WAIT_rows=16384,A_denominator_sum=36864,A_selected_requests=16384,
            candidate_forward_lane_ticks=w['A_score_forward_lane_ticks'],eligible_tick_comparisons=w['A_score_eligible_tick_comparisons'],
            B_gradient_label_positions=w['B_gradient_label_positions'],B_unique_label_positions=w['B_unique_label_positions'],
            B_unique_NEW=w['B_unique_new_commands'],B_actual_NEW_including_boundary=w['B_actual_NEW_including_boundary'])
        timing[arm]=dict(selected_requests=16384,moved_count=sum(row['moved_count'] for row in rows),
            midpoint_equal_count=sum(row['midpoint_equal_count'] for row in rows),
            FINISH_intent_count=sum(row['selected_finish_intent_count'] for row in rows),
            nonWAIT_intent_count=sum(row['selected_nonwait_intent_count'] for row in rows),
            window_no_FINISH_count=sum(row['window_no_finish_count'] for row in rows))
    assert len(budgets)==1 and 'last' in budgets
    result=dict(seed=seed,passed=True,reports=reports,prior_uniform_pre_review_bit_exact_calibration=calibration,
        every_request_id_matches_previous_uniform=True,all_arm_integer_budgets_equal=True,budgets=budgets,timing=timing)
    (OUT/f'seed_{seed:03d}/first_step_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    print(json.dumps(audit(parser.parse_args().seed)),flush=True)
