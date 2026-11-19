"""Matched WAIT placement with identical uniform old-A requests and B streams."""
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

OUT = ROOT / 'data/wait-position/train-v1'
BANK = ROOT / 'data/wait-position/bank-v1'
PLAN = OUT / 'protocol.json'
ARMS = ('midpoint', 'hard', 'random')
REVIEW_EPOCHS = (8, 16, 24, 32)
CANDIDATES, SELECTED = 128, 64
SNAPSHOTS = ('B000', 'Bfirst002', 'Bfirst008', 'B001', 'B002', 'B004', 'B008')


def bank_path(seed):
    return BANK / f'seed_{seed:03d}' / 'bank.pt'


def origin(seed):
    return t.OUT / f'seed_{seed:03d}' / 'related' / 'B000.pt'


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    sources = [Path(__file__).resolve(), ROOT/'experiments/wait_position/train_audit.py', Path(previous.__file__).resolve(),
               Path(t.__file__).resolve(), ROOT/'experiments/learn_remember/gru_train.py',
               ROOT/'experiments/lifelong/world.py', ROOT/'experiments/continuity/world.py']
    inputs = {}
    for seed in range(3):
        paths = [origin(seed), bank_path(seed), BANK/'protocol.json', previous.PLAN, t.PLAN,
                 previous.OUT/f'seed_{seed:03d}'/'uniform'/'B008.pt'] + [t.OUT/f'seed_{seed:03d}'/'related'/f'trajectory_B_{r:03d}.pt' for r in range(8)]
        inputs[str(seed)] = {str(p): t.sha(p) for p in paths}
    plan = dict(format='wait-position-training-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2], arms=list(ARMS), snapshots=list(SNAPSHOTS), device='cuda',
        source_sha256={str(p):t.sha(p) for p in sources}, input_sha256=inputs,
        original_protocol_sha256=t.sha(t.PLAN),previous_protocol_sha256=t.sha(previous.PLAN),bank_protocol_sha256=t.sha(BANK/'protocol.json'),
        start='Each arm restores the exact original related B000 model, Adam, full world, numeric h and RNG. No new A learning or bank-specific warmup is admitted to the live subject.',
        B='Reuse the exact eight archived original related B observation/action/label/mask streams,1024ticks64actors each. One original shared WAIT boundary. Each actual block runs current frozen-block policy on public observations to continue numeric live h, then receives its first B gradient update. Live h is never replaced by a training/review branch. The common world executes archived actions and its observations/labels/masks are checked exactly.',
        updates='Exactly256 optimizer updates per round: first temporal pass8 online B updates; epochs2..32 replay only this current B round. Epochs8,16,24,32 replace all8 B updates by A review:224 B plus32 A updates per round,1792 B plus256 A for8rounds. During each review epoch B branch h still advances in temporal order with no loss or Adam update, so later B chronology is well defined.',
        old_bank='The same256 genuine old-A requests,128tick windows, actual actions, public observations, labels and A anchors as prior uniform arm. Additional eligible mask is actual same request, teacher WAIT, strictly after accepted BEGIN and before ready_time. No hidden-state or receipt editing.',
        candidates='Exactly previous uniform candidate128 and selected64 IDs at every review index. All three arms run the same no_grad128x128 forward and per-tick margin/argmax computation. Scorer input is obs and anchor only; researcher eligibility masks restrict allowable times without becoming model input. No request identity, private time, target content, raw/conflict or B label enters margin.',
        selection=dict(midpoint='Keep the original fixed midpoint WAIT row.',
            hard='For each request maximize max(logit FINISH3..10)-max(logit WAIT0,BEGIN1,BEGIN2) over eligible processing WAIT ticks. Highest margin, earliest exact tie. Even when no eligible tick argmax is FINISH, retain nearest-to-FINISH maximum and separately count that case.',
            random='Uniform one eligible processing WAIT tick per candidate using fixed independent numpy RNG seed3140000+seed*10000+review_index, candidate order, one choice each. No model score enters random choice.'),
        gradient='All review arms update on128ticks x64samples x2jobs x11actions tensors and exactly192 supervised target-job rows/update. Original weighted CE(wait=.25),Adam,lr andclip1 unchanged. Two non-WAIT plus one WAIT per request gives equal loss denominator144 for all review updates. B update tensors have the same shape but many more supervised rows. Baseline no-review retains256B updates/round; it is equal total Adam update budget, not equal B exposures to gradients or equal supervised row count.',
        state='Only common archived B public inputs/actions advance the actual world/live hidden state. A scoring/gradient anchors and their returned h are disposable branch state. Review does not advance any world or live h.',
        evidence='Keep candidate/selected IDs and hashes, eligible counts, selected and midpoint ticks/margins/argmax intents, window any-FINISH flag, private timing only in audit output, integer budgets, all optimizer-step input identity hashes and actual B exposure. Save first review plus first actual moved update before/after complete checkpoints; independent reconstruction verifies masked loss/Adam/live h/world. Compare rerun midpoint first step and final model/Adam/h per seed against prior uniform.',
        limits=['Supervised prototype with stored teacher labels, not autonomous task choice or reward-only learning.',
                'Equal candidate scans, tensor shapes, supervised rows and optimizer budgets among three WAIT arms; eligible intervals are researcher-reconstructed.',
                'Original no-review results can be reused, but their B feedback positions differ from review arms.',
                'Three seeds and fixed permutations; selection result is conditional on this bank and feedback design.'])
    plan['protocol_hash'] = state_digest(plan)
    t.write(PLAN, plan)
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


def candidate_ids(bank, seed, index):
    rng=np.random.default_rng(1140000+seed*10000+index)
    raw=bank['raw'].cpu().numpy()
    ids=np.concatenate([rng.choice(np.flatnonzero(raw==r),32,replace=False) for r in range(4)])
    rng.shuffle(ids)
    return torch.from_numpy(ids)


def scores(model, obs, anchor, eligible_mask):
    """Public-input scoring only; eligibility is a restriction, never a feature."""
    with torch.no_grad():
        logits,_=model(obs,anchor)
        margin=logits[...,3:].max(-1).values-logits[...,:3].max(-1).values
        action=logits.argmax(-1)
        assert eligible_mask.any((0,2)).all()
        # At most one target slot eligible per request. Flatten in time-major order.
        values,ticks=margin.masked_fill(~eligible_mask,-torch.inf).permute(1,0,2).flatten(1).max(1)
        ticks=ticks//2
    return dict(margin=margin.cpu(),action=action.cpu(),maximum=values.cpu(),hard_ticks=ticks.cpu(),
        any_finish=((action>=3)&eligible_mask).any((0,2)).cpu())


def selected_ids(arm, ids, values, bank, seed, index):
    if arm not in ARMS:raise ValueError(arm)
    rng=np.random.default_rng(2140000+seed*10000+index)
    local=torch.from_numpy(rng.choice(len(ids),SELECTED,replace=False))
    assert len(local)==SELECTED and len(torch.unique(local))==SELECTED
    return ids[local],local


class ReviewLearner(t.Learner):
    def save(self,name):
        super().save(name)
        path=self.folder/(name+'.pt')
        cp=torch.load(path,map_location='cpu',weights_only=False)
        cp['wait_position']=dict(protocol_hash=self.review_protocol_hash,bank_sha256=self.bank_sha256,
            review_updates=self.review_index,selection_counts=self.selection_counts.clone(),
            original_B000_sha256=t.sha(origin(self.seed)),live_state='B public stream only')
        temporary=path.with_suffix('.pt.tmp')
        torch.save(cp,temporary);temporary.replace(path)
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        meta.update(sha256=t.sha(path),wait_position=cp['wait_position']|{'selection_counts':self.selection_counts.tolist()})
        t.write(path.with_suffix('.json'),meta)

    def update(self,obs,labels,mask,h,kind):
        logits,out=self.model(obs,h)
        weights=torch.ones(11,device='cuda');weights[0]=self.cfg.wait_weight
        ce=F.cross_entropy(logits.flatten(0,2),labels.flatten(),weight=weights,reduction='none').reshape_as(labels)
        denominator=(weights[labels]*mask).sum()
        assert denominator>0
        loss=(ce*mask).sum()/denominator
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        gradnorm=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.))
        self.optimizer.step()
        self.work['gradient_updates']+=1
        self.work['gradient_lane_ticks']+=len(obs)*len(h)
        self.work[kind+'_gradient_updates']+=1
        self.work[kind+'_gradient_label_positions']+=int(mask.sum())
        self.work[kind+'_gradient_nonwait_positions']+=int((mask&(labels!=0)).sum())
        append(self.folder/'step_updates.jsonl',dict(step=self.work['gradient_updates'],kind=kind,
            round=self.round,**self.step_identity,supervised_rows=int(mask.sum()),nonwait_rows=int((mask&(labels!=0)).sum()),
            denominator=float(denominator),loss=float(loss.detach()),preclip_grad_norm=gradnorm))
        return out.detach(),float(loss.detach()),gradnorm


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
    all_ticks={'midpoint':midpoint,'hard':score['hard_ticks'],'random':random_ticks}[v.arm]
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
    from experiments.wait_position import train_audit
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
            proc=subprocess.Popen([sys.executable,'-u','-m','experiments.wait_position.train_review','run','--seed',str(seed)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            processes.append((seed,proc,stream))
        results=[]
        for seed,proc,stream in processes:
            results.append(dict(seed=seed,exit_code=proc.wait()));stream.close()
            t.write(OUT/'exit_codes.json',results)
        if any(r['exit_code'] for r in results):raise RuntimeError(results)


if __name__=='__main__':main()
