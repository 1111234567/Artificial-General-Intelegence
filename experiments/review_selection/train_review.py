"""Frozen, compute-matched old-A request selection from exact related B000 branches."""
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

OUT = ROOT / 'data/review-selection/train-v1'
BANK = ROOT / 'data/review-selection/bank-v1'
PLAN = OUT / 'protocol.json'
ARMS = ('uniform', 'conflict', 'error')
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
    sources = [Path(__file__).resolve(), ROOT/'experiments/review_selection/train_audit.py',
               Path(t.__file__).resolve(), ROOT/'experiments/learn_remember/gru_train.py',
               ROOT/'experiments/lifelong/world.py', ROOT/'experiments/continuity/world.py']
    inputs = {}
    for seed in range(3):
        paths = [origin(seed), bank_path(seed)] + [t.OUT/f'seed_{seed:03d}'/'related'/f'trajectory_B_{r:03d}.pt' for r in range(8)]
        inputs[str(seed)] = {str(p): t.sha(p) for p in paths}
    plan = dict(format='review-selection-training-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2], arms=list(ARMS), snapshots=list(SNAPSHOTS), device='cuda',
        source_sha256={str(p):t.sha(p) for p in sources}, input_sha256=inputs,
        original_protocol_sha256=t.sha(t.PLAN),
        start='Each arm restores the exact original related B000 model, Adam, full world, numeric h and RNG. No new A learning or bank-specific warmup is admitted to the live subject.',
        B='Reuse the exact eight archived original related B observation/action/label/mask streams,1024ticks64actors each. One original shared WAIT boundary. Each actual block runs current frozen-block policy on public observations to continue numeric live h, then receives its first B gradient update. Live h is never replaced by a training/review branch. The common world executes archived actions and its observations/labels/masks are checked exactly.',
        updates='Exactly256 optimizer updates per round: first temporal pass8 online B updates; epochs2..32 replay only this current B round. Epochs8,16,24,32 replace all8 B updates by A review:224 B plus32 A updates per round,1792 B plus256 A for8rounds. During each review epoch B branch h still advances in temporal order with no loss or Adam update, so later B chronology is well defined.',
        old_bank='256 real old own-action A request snippets per seed,128ticks,one actual actor and both native public job slots. Shared frozen A-model reconstructed start states. Exactly three archived oracle label rows per request: BEGIN, an in-process WAIT, and first legal FINISH, only its target job. Other job labels are always masked. Actual actions are not replaced by these labels.',
        candidates='For each review update, one shared deterministic candidate pool without replacement:32 per each of4 raw symbols,128 total. Pool sequence does not depend on model or arm. Re-score every pool under every arm current model with no gradients:128x128tick forward and3 unweighted true-label CE rows per request. All arms incur the same scoring tensor shapes and384 label comparisons per update.',
        selection=dict(uniform='Uniform64 without replacement from shared128 candidates.',
            conflict='All64 candidates from the two prespecified A/B conflict raw symbols. Because32 of each raw is sampled uniformly from its64 bank requests, this is uniform stratified conflict-only old-A sampling. It uses researcher knowledge.',
            error='Top64 current model per-request mean unweighted true-label CE over its3 points, stable descending tie order within shared randomized candidate order. Scorer receives observations, old stored A labels, masks and A anchors only; selection never reads conflict/raw metadata or B labels.'),
        gradient='All review arms update on128ticks x64samples x2jobs x11actions tensors and exactly192 supervised target-job rows/update. Original weighted CE(wait=.25),Adam,lr andclip1 unchanged. Two non-WAIT plus one WAIT per request gives equal loss denominator144 for all review updates. B update tensors have the same shape but many more supervised rows. Baseline no-review retains256B updates/round; it is equal total Adam update budget, not equal B exposures to gradients or equal supervised row count.',
        state='Only common archived B public inputs/actions advance the actual world/live hidden state. A scoring/gradient anchors and their returned h are disposable branch state. Review does not advance any world or live h.',
        evidence='Keep complete checkpoints, cumulative work and exact input hashes; record actual candidates/selection/scores and CE components, raw/conflict distributions,bank coverage and reuse. First actual selection plus Adam step gets before/after full checkpoints and independent CUDA replay audit. Retain actual world B NEW count including no-loss boundary separately from original work counter.',
        limits=['Supervised prototype with stored teacher labels, not autonomous task choice or reward-only learning.',
                'Equal compute and feedback among three review arms; old A bank is researcher-constructed.',
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


def scores(model, obs, labels, mask, anchor):
    with torch.no_grad():
        logits,_=model(obs,anchor)
        ce=F.cross_entropy(logits.flatten(0,2),labels.flatten(),reduction='none').reshape_as(labels)
        assert torch.all(mask.sum((0,2))==3)
        total=(ce*mask).sum((0,2))/3
        # Temporal-order components are BEGIN, middle WAIT, legal FINISH.
        components=ce.permute(1,0,2)[mask.permute(1,0,2)].reshape(len(anchor),3)
    return total.cpu(),components.cpu()


def selected_ids(arm, ids, values, bank, seed, index):
    if arm=='uniform':
        rng=np.random.default_rng(2140000+seed*10000+index)
        local=torch.from_numpy(rng.choice(len(ids),SELECTED,replace=False))
    elif arm=='conflict':
        local=torch.where(bank['conflict'][ids].bool())[0]
    elif arm=='error':
        local=torch.argsort(values,descending=True,stable=True)[:SELECTED]
    else:
        raise ValueError(arm)
    assert len(local)==SELECTED and len(torch.unique(local))==SELECTED
    return ids[local],local


class ReviewLearner(t.Learner):
    def save(self,name):
        super().save(name)
        path=self.folder/(name+'.pt')
        cp=torch.load(path,map_location='cpu',weights_only=False)
        cp['review_selection']=dict(protocol_hash=self.review_protocol_hash,bank_sha256=self.bank_sha256,
            review_updates=self.review_index,selection_counts=self.selection_counts.clone(),
            original_B000_sha256=t.sha(origin(self.seed)),live_state='B public stream only')
        temporary=path.with_suffix('.pt.tmp')
        torch.save(cp,temporary);temporary.replace(path)
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        meta.update(sha256=t.sha(path),review_selection=cp['review_selection']|{'selection_counts':self.selection_counts.tolist()})
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
        return out.detach(),float(loss.detach()),gradnorm


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def review_update(v,bank,gpu_bank,epoch,block):
    index=v.review_index;ids=candidate_ids(bank,v.seed,index)
    candidate={k:value[:,ids].contiguous() if k in ('obs','labels','mask') else value[ids].contiguous()
               for k,value in gpu_bank.items()}
    values,components=scores(v.model,**candidate)
    selected,local=selected_ids(v.arm,ids,values,bank,v.seed,index)
    batch={k:value[:,local].contiguous() if k in ('obs','labels','mask') else value[local].contiguous()
           for k,value in candidate.items()}
    assert int(batch['mask'].sum())==192
    first=index==0
    if first:
        v.save('first_review_before')
        torch.save(dict(candidate_ids=ids,selected_ids=selected,selected_local=local,scores=values,
            components=components,seed=v.seed,arm=v.arm,review_index=index,round=v.round,epoch=epoch,block=block,
            protocol_hash=v.review_protocol_hash),v.folder/'first_selection.pt')
    old_counts=v.selection_counts[selected].clone()
    _,loss,gradnorm=v.update(batch['obs'],batch['labels'],batch['mask'],batch['anchor'],kind='A')
    v.selection_counts[selected]+=1
    v.review_index+=1
    v.work['A_score_forward_lane_ticks']+=128*128
    v.work['A_score_label_comparisons']+=384
    v.work['A_candidate_presentations']+=128
    v.work['A_selected_requests']+=64
    v.work['A_repeated_requests']+=int((old_counts>0).sum())
    v.work['A_unique_requests']=int((v.selection_counts>0).sum())
    v.work['A_selected_conflict_requests']+=int(bank['conflict'][selected].sum())
    v.work['old_history_replay_positions']+=192
    row=dict(event='review_update',seed=v.seed,arm=v.arm,round=v.round,epoch=epoch,block=block,index=index,
        candidates=ids.tolist(),selected=selected.tolist(),scores=values.tolist(),score_components=components.tolist(),
        selected_mean_score=float(values[local].mean()),candidate_mean_score=float(values.mean()),
        selected_component_mean=components[local].mean(0).tolist(),candidate_component_mean=components.mean(0).tolist(),
        raw_counts=torch.bincount(bank['raw'][selected],minlength=4).tolist(),
        conflict_count=int(bank['conflict'][selected].sum()),unique_so_far=int((v.selection_counts>0).sum()),
        reused_requests=int((old_counts>0).sum()),max_repetitions=int(v.selection_counts.max()),
        candidate_count=128,scoring_rows=384,gradient_rows=192,gradient_nonwait_rows=128,
        gradient_tensor=[128,64,2,11],loss=loss,preclip_grad_norm=gradnorm)
    append(v.folder/'selections.jsonl',row)
    if first:
        v.save('first_review_after')
    return loss


def run(seed):
    torch.set_num_threads(1)
    plan=checked();began=time.perf_counter()
    bank=torch.load(bank_path(seed),map_location='cpu',weights_only=False)
    assert bank['obs'].shape==(128,256,56) and bank['anchor'].shape==(256,192)
    assert torch.all(bank['mask'].sum((0,2))==3)
    assert all(int((bank['raw']==r).sum())==64 for r in range(4))
    gpu_bank={k:bank[k].cuda() for k in ('obs','labels','mask','anchor')}
    learners=[];bindings=[]
    original=torch.load(origin(seed),map_location='cpu',weights_only=False)
    for arm in ARMS:
        v=ReviewLearner.from_checkpoint(origin(seed),OUT/f'seed_{seed:03d}'/arm)
        v.arm=arm;v.review_protocol_hash=plan['protocol_hash'];v.bank_sha256=t.sha(bank_path(seed))
        v.review_index=0;v.selection_counts=torch.zeros(256,dtype=torch.long)
        for key in ('A_gradient_updates','A_gradient_label_positions','A_gradient_nonwait_positions',
                    'B_gradient_updates','B_gradient_nonwait_positions','A_score_forward_lane_ticks',
                    'A_score_label_comparisons','A_candidate_presentations','A_selected_requests',
                    'A_repeated_requests','A_unique_requests','A_selected_conflict_requests'):
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
    from experiments.review_selection import train_audit
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
            proc=subprocess.Popen([sys.executable,'-u','-m','experiments.review_selection.train_review','run','--seed',str(seed)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            processes.append((seed,proc,stream))
        results=[]
        for seed,proc,stream in processes:
            results.append(dict(seed=seed,exit_code=proc.wait()));stream.close()
            t.write(OUT/'exit_codes.json',results)
        if any(r['exit_code'] for r in results):raise RuntimeError(results)


if __name__=='__main__':main()
