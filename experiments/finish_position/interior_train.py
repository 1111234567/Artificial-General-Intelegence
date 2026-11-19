"""The single post-endpoint, fixed integer-interior FINISH supplement."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from experiments.continuity.run import state_digest
from experiments.experience_transfer import train_transfer as t
from experiments.finish_position import train as main_train
from experiments.wait_position import last_train as legacy
from experiments.wait_position import last_audit as legacy_audit
from experiments.wait_position import train_review as matched

OUT=ROOT/'data/finish-position/interior-v1'
PLAN=OUT/'protocol.json'
ARMS=('interior',)
SNAPSHOTS=main_train.SNAPSHOTS
_DERIVED={}


def derive(bank):
    first=bank['first_legal_positions']
    teacher=bank['teacher_finish_positions']
    interior=torch.div(first+teacher,2,rounding_mode='floor')
    assert (first<interior).all() and (interior<teacher).all()
    assert (interior-first).ge(1).all() and (interior-first).le(2).all()
    batch=torch.arange(256);slots=bank['target_slot']
    labels=bank['labels'].clone();mask=bank['mask'].clone()
    mask[bank['positions'][:,1],batch,slots]=False
    mask[bank['last_wait_positions'],batch,slots]=True
    mask[teacher,batch,slots]=False
    mask[interior,batch,slots]=True
    labels[interior,batch,slots]=bank['finish_labels']
    assert mask.sum((0,2)).eq(3).all()
    assert int((mask&(labels!=0)).sum())==512
    return dict(positions=interior,labels=labels,mask=mask)


def declare():
    if PLAN.exists():raise FileExistsError(PLAN)
    parent=main_train.checked()
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    sources=[Path(__file__).resolve(),ROOT/'experiments/finish_position/interior_audit.py',
        Path(main_train.__file__).resolve(),ROOT/'experiments/finish_position/train_audit.py',
        ROOT/'experiments/finish_position/bank_audit.py']
    inputs={str(main_train.PLAN):t.sha(main_train.PLAN),
        str(ROOT/'data/finish-position/eval-v1/summary.json'):t.sha(ROOT/'data/finish-position/eval-v1/summary.json')}
    identities={}
    for seed in range(3):
        bank=torch.load(main_train.bank_path(seed),map_location='cpu',weights_only=False)
        d=derive(bank)
        identities[str(seed)]=dict(interior_positions=d['positions'].tolist(),positions_hash=state_digest(d['positions']),
            labels_hash=state_digest(d['labels']),mask_hash=state_digest(d['mask']),
            interior_after_first_ticks=torch.bincount(d['positions']-bank['first_legal_positions']).tolist())
        for arm in ('midpoint','first'):
            path=ROOT/f'data/finish-position/pairs-v1/seed_{seed:03d}_{arm}.json'
            inputs[str(path)]=t.sha(path)
        for path in (main_train.BANK/f'seed_{seed:03d}/manifest.json',main_train.BANK/f'seed_{seed:03d}/audit.json'):
            inputs[str(path)]=t.sha(path)
    plan=dict(format='finish-position-single-interior-training-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2],arms=list(ARMS),snapshots=list(SNAPSHOTS),device='cuda',
        runtime=parent['runtime'],source_sha256={str(path):t.sha(path) for path in sources},input_sha256=inputs,
        original_training_protocol_hash=parent['protocol_hash'],fixed_intervention_identities=identities,
        prior_results_already_known=dict(new_B_window_no_FINISH_midpoint=33,new_B_window_no_FINISH_first=0,
            paired_old_A_new_B_same_slot_symbol_long_requests=317,paired_both_without_error_midpoint=95,
            paired_both_without_error_first=42,old_A_actual_early_FINISH='rose again under first',
            source='Endpoint eval-v1 and pairs-v1 were inspected before choosing this sole supplement.'),
        prospective_scope='Post-endpoint single mechanistic supplement. No interior training or evaluation result has been inspected at declaration. Exactly one fixed interior point is admitted; no further position scan or outcome-dependent position choice.',
        position='floor((actual_first_legal_position + original_teacher_FINISH_position)/2), fixed independently for each same old-A request before training. This is first+1 or first+2 ticks because endpoint advances are3..5 ticks. Keep original BEGIN and identical last eligible processing WAIT; move only the sparse correct FINISH positive and write its correct label at the chosen position.',
        start=parent['start'],B=parent['B'],updates=parent['updates'],gradient=parent['gradient'],state=parent['state'],
        candidates=parent['candidates'],
        native_check='Independently replay original native world before training; at every one of768 selected interior positions verify actor/slot/request/raw/version/context/payload identity, active/started readiness/deadline, correct earliest oracle, and native accepted proposed FINISH on a disposable cloned world. Do not advance live training state with this audit.',
        evidence='Same original related B000 model/Adam/world/live h/RNG and original B streams;256 A plus1792 B updates; independent first changed CE/clipping/Adam recomputation and all256 request/label/mask reconstructions. All historical main files remain unchanged.',
        limits=parent['limits']+['The interior point was selected after seeing endpoint tradeoffs; it is a single post-endpoint test and does not establish an optimum or linear interpolation.'])
    plan['protocol_hash']=state_digest(plan);t.write(PLAN,plan)
    print(json.dumps(dict(event='declared',path=str(PLAN),protocol_hash=plan['protocol_hash'])),flush=True)


def checked():
    main_train.checked()
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    assert state_digest({k:v for k,v in plan.items() if k!='protocol_hash'})==plan['protocol_hash']
    for path,digest in plan['source_sha256'].items():assert t.sha(path)==digest,path
    for path,digest in plan['input_sha256'].items():assert t.sha(path)==digest,path
    return plan


class ReviewLearner(main_train.ReviewLearner):
    def save(self,name):
        super().save(name)
        path=self.folder/(name+'.pt');cp=torch.load(path,map_location='cpu',weights_only=False)
        cp['interior_position']=dict(protocol_hash=self.review_protocol_hash,
            rule='floor((actual_first_legal+original_teacher_FINISH)/2)',
            positions_hash=state_digest(_DERIVED[self.seed]['positions']),single_post_endpoint_supplement=True)
        temporary=path.with_suffix('.pt.tmp');torch.save(cp,temporary);temporary.replace(path)
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        meta.update(sha256=t.sha(path),interior_position=cp['interior_position']);t.write(path.with_suffix('.json'),meta)


def review_update(v,bank,gpu_bank,epoch,block):
    index=v.review_index;d=_DERIVED[v.seed]
    ids=matched.candidate_ids(bank,v.seed,index)
    candidate={k:value[:,ids].contiguous() if k in ('obs','labels','mask','eligible_mask') else value[ids].contiguous()
        for k,value in gpu_bank.items()}
    score=matched.scores(v.model,candidate['obs'],candidate['anchor'],candidate['eligible_mask'])
    selected,local=matched.selected_ids('midpoint',ids,None,bank,v.seed,index)
    wait=bank['last_wait_positions'][selected];finish=d['positions'][selected]
    old=bank['teacher_finish_positions'][selected];first=bank['first_legal_positions'][selected]
    slots=bank['target_slot'][selected];r=torch.arange(64,device='cuda')
    labels=d['labels'][:,selected].contiguous().cuda();mask=d['mask'][:,selected].contiguous().cuda()
    assert int(mask.sum())==192 and int((mask&(labels!=0)).sum())==128
    assert labels[wait.cuda(),r,slots.cuda()].eq(0).all()
    assert torch.equal(labels[finish.cuda(),r,slots.cuda()].cpu(),bank['finish_labels'][selected])
    record=dict(seed=v.seed,arm=v.arm,review_index=index,round=v.round,epoch=epoch,block=block,
        protocol_hash=v.review_protocol_hash,candidate_ids=ids,selected_ids=selected,selected_local=local,
        selected_wait_ticks=wait,selected_finish_ticks=finish,teacher_finish_ticks=old,
        first_legal_finish_ticks=first,finish_labels=bank['finish_labels'][selected],
        supervision_mask=mask.cpu(),supervision_labels=labels.cpu(),candidate_hard_ticks=score['hard_ticks'],
        candidate_maximum=score['maximum'],candidate_any_finish=score['any_finish'])
    initial=index==0;first_moved=bool(finish.ne(old).any()) and not v.first_moved_saved
    if initial:v.save('first_review_before');torch.save(record,v.folder/'first_selection.pt')
    if first_moved:v.save('first_moved_before');torch.save(record,v.folder/'first_moved_selection.pt')
    live=state_digest(dict(h=v.hidden,world=v.world_state()))
    v.step_identity=dict(review_index=index,candidate_id_hash=state_digest(ids),selected_id_hash=state_digest(selected),
        selected_time_hash=state_digest(finish),supervision_mask_hash=state_digest(mask.cpu()),
        supervision_labels_hash=state_digest(labels.cpu()),bank_sha256=v.bank_sha256,epoch=epoch,block=block)
    old_counts=v.selection_counts[selected].clone()
    _,loss,gradnorm=v.update(candidate['obs'][:,local].contiguous(),labels,mask,candidate['anchor'][local].contiguous(),kind='A')
    assert live==state_digest(dict(h=v.hidden,world=v.world_state()))
    v.selection_counts[selected]+=1;v.review_index+=1
    v.work['A_score_forward_lane_ticks']+=128*128;v.work['A_score_label_comparisons']+=0
    v.work['A_score_eligible_tick_comparisons']+=int(bank['eligible_mask'][:,ids].sum())
    v.work['A_candidate_presentations']+=128;v.work['A_selected_requests']+=64
    v.work['A_repeated_requests']+=int((old_counts>0).sum());v.work['A_unique_requests']=int((v.selection_counts>0).sum())
    v.work['A_selected_conflict_requests']+=int(bank['conflict'][selected].sum());v.work['old_history_replay_positions']+=192
    v.work['A_moved_WAIT_rows']+=int(wait.ne(bank['positions'][selected,1]).sum())
    v.work['A_moved_FINISH_rows']=v.work.get('A_moved_FINISH_rows',0)+int(finish.ne(old).sum())
    matched.append(v.folder/'selections.jsonl',dict(event='review_update',seed=v.seed,arm=v.arm,index=index,
        round=v.round,epoch=epoch,block=block,candidates=ids.tolist(),selected=selected.tolist(),
        candidate_id_hash=state_digest(ids),selected_id_hash=state_digest(selected),selected_wait_ticks=wait.tolist(),
        selected_finish_ticks=finish.tolist(),teacher_finish_ticks=old.tolist(),first_legal_finish_ticks=first.tolist(),
        finish_labels=bank['finish_labels'][selected].tolist(),moved_finish_count=int(finish.ne(old).sum()),
        original_new_tick_labels=bank['labels'][finish,selected,slots].tolist(),
        supervision_mask_hash=state_digest(mask.cpu()),supervision_labels_hash=state_digest(labels.cpu()),
        supervised_identity_hash=state_digest(dict(mask=mask.cpu(),labels=labels.cpu()[mask.cpu()])),
        gradient_rows=192,gradient_nonwait_rows=128,gradient_WAIT_rows=64,denominator=144,
        candidate_count=128,candidate_scan_tensor=[128,128,2,11],gradient_tensor=[128,64,2,11],
        loss=loss,preclip_grad_norm=gradnorm,live_hidden_world_unchanged=True))
    if initial:v.save('first_review_after')
    if first_moved:v.save('first_moved_after');v.first_moved_saved=True
    return loss


def run(seed):
    from experiments.finish_position import interior_audit
    plan=checked();bank=torch.load(main_train.bank_path(seed),map_location='cpu',weights_only=False)
    _DERIVED[seed]=derive(bank)
    assert state_digest(_DERIVED[seed]['positions'])==plan['fixed_intervention_identities'][str(seed)]['positions_hash']
    native=interior_audit.native_legality(seed)
    assert native['passed']
    torch.save(_DERIVED[seed],OUT/f'seed_{seed:03d}'/'fixed_supervision.pt')
    print(json.dumps(dict(event='runtime',seed=seed,executable=sys.executable,GPU_name=torch.cuda.get_device_name(0))),flush=True)
    legacy.OUT,legacy.BANK,legacy.PLAN=OUT,main_train.BANK,PLAN
    legacy.ARMS=ARMS;legacy.checked=checked;legacy.ReviewLearner=ReviewLearner
    legacy.review_update=review_update;legacy_audit.audit=interior_audit.audit
    legacy.run(seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('declare','run','all'))
    parser.add_argument('--seed',type=int,choices=range(3));args=parser.parse_args()
    if args.command=='declare':declare()
    elif args.command=='run':run(args.seed)
    else:
        checked();processes=[]
        for seed in range(3):
            stream=(OUT/f'seed_{seed:03d}.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u','-m','experiments.finish_position.interior_train','run','--seed',str(seed)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            processes.append((seed,proc,stream))
        results=[]
        for seed,proc,stream in processes:
            results.append(dict(seed=seed,exit_code=proc.wait()));stream.close();t.write(OUT/'exit_codes.json',results)
        if any(r['exit_code'] for r in results):raise RuntimeError(results)


if __name__=='__main__':main()
