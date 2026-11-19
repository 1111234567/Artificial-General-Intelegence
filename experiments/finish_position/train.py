"""Matched sparse FINISH-position training, reusing the frozen last-WAIT run."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.continuity.run import state_digest
from experiments.experience_transfer import train_transfer as t
from experiments.wait_position import last_train as legacy
from experiments.wait_position import last_audit as legacy_audit
from experiments.wait_position import train_review as matched

OUT = ROOT / 'data/finish-position/train-v1'
BANK = ROOT / 'data/finish-position/bank-v1'
PLAN = OUT / 'protocol.json'
ARMS = ('midpoint', 'first')
SNAPSHOTS = legacy.SNAPSHOTS
HISTORICAL_CHECKED = legacy.checked


def bank_path(seed):
    return BANK / f'seed_{seed:03d}' / 'bank.pt'


def declare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    assert torch.cuda.is_available()
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    old = HISTORICAL_CHECKED()
    sources = [Path(__file__).resolve(), ROOT/'experiments/finish_position/train_audit.py',
               ROOT/'experiments/finish_position/bank.py', Path(legacy.__file__).resolve(),
               Path(legacy_audit.__file__).resolve(), Path(matched.__file__).resolve(),
               Path(t.__file__).resolve(), ROOT/'experiments/learn_remember/gru_train.py',
               ROOT/'experiments/lifelong/world.py', ROOT/'experiments/continuity/world.py',
               ROOT/'experiments/continuity/run.py']
    inputs = {}
    for seed in range(3):
        prior = ROOT/f'data/wait-position/last-v1/seed_{seed:03d}/last'
        paths = [legacy.origin(seed), bank_path(seed), BANK/'protocol.json', legacy.PLAN,
                 matched.PLAN, t.PLAN, prior/'selections.jsonl', prior/'step_updates.jsonl']
        paths += [prior/(name+'.pt') for name in (*SNAPSHOTS, 'first_review_before')]
        paths += [t.OUT/f'seed_{seed:03d}/related/trajectory_B_{r:03d}.pt' for r in range(8)]
        inputs[str(seed)] = {str(path): t.sha(path) for path in paths}
    plan = dict(format='finish-position-training-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2], arms=list(ARMS), snapshots=list(SNAPSHOTS), device='cuda',
        runtime=dict(executable=sys.executable,torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),GPU_name=torch.cuda.get_device_name(0),
            GPU_total_memory=torch.cuda.get_device_properties(0).total_memory),
        source_sha256={str(path):t.sha(path) for path in sources}, input_sha256=inputs,
        historical_last_protocol_hash=old['protocol_hash'],
        implementation='Reuse frozen experiments.wait_position.last_train.run through process-local bindings only. Historical sources are never edited.',
        start=old['start'], B=old['B'], updates=old['updates'], gradient=old['gradient'], state=old['state'],
        candidates=old['candidates'],
        review=dict(midpoint='Exact former last-WAIT control: original BEGIN, final eligible processing WAIT, original teacher FINISH.',
                    first='Same BEGIN and final eligible processing WAIT; remove only original sparse FINISH positive and insert the same correct FINISH label at the actual first legal tick of the identical old request. The teacher may have said WAIT at this new tick; explicitly overwrite the training label there. Public observations, archived actions, anchors and B labels are unchanged.'),
        evidence='All256 candidate and selected identities and labels/masks reconstructed independently; first actual changed FINISH gradient/Adam recomputed. All named midpoint snapshots including B008 compare model/Adam/live h/world bitwise to historical last. Preserve full checkpoints, per-update logs, actual compute budgets and shared-B stream checks.',
        limits=old['limits']+['FINISH placement intervention uses researcher-reconstructed actual legality. Original midpoint name denotes the retained original teacher FINISH, while WAIT remains the previous last-eligible control.'])
    plan['protocol_hash'] = state_digest(plan)
    t.write(PLAN, plan)
    print(json.dumps(dict(event='declared', path=str(PLAN), protocol_hash=plan['protocol_hash'])), flush=True)


def checked():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    assert state_digest({k:v for k,v in plan.items() if k!='protocol_hash'}) == plan['protocol_hash']
    for path, expected in plan['source_sha256'].items():
        assert t.sha(path)==expected, 'frozen source changed: '+path
    for paths in plan['input_sha256'].values():
        for path, expected in paths.items():
            assert t.sha(path)==expected, 'frozen input changed: '+path
    return plan


class ReviewLearner(matched.ReviewLearner):
    def save(self, name):
        super().save(name)
        path = self.folder/(name+'.pt')
        cp = torch.load(path,map_location='cpu',weights_only=False)
        cp['finish_position'] = dict(protocol_hash=self.review_protocol_hash, bank_sha256=self.bank_sha256,
            arm=self.arm, review_updates=self.review_index, selection_counts=self.selection_counts.clone(),
            original_B000_sha256=t.sha(legacy.origin(self.seed)), live_state='B public stream only')
        temporary = path.with_suffix('.pt.tmp')
        torch.save(cp,temporary); temporary.replace(path)
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        meta.update(sha256=t.sha(path), finish_position=cp['finish_position']|{'selection_counts':self.selection_counts.tolist()})
        t.write(path.with_suffix('.json'),meta)


def review_update(v, bank, gpu_bank, epoch, block):
    index=v.review_index
    ids=matched.candidate_ids(bank,v.seed,index)
    candidate={k:value[:,ids].contiguous() if k in ('obs','labels','mask','eligible_mask') else value[ids].contiguous()
               for k,value in gpu_bank.items()}
    score=matched.scores(v.model,candidate['obs'],candidate['anchor'],candidate['eligible_mask'])
    selected,local=matched.selected_ids('midpoint',ids,None,bank,v.seed,index)
    slots=bank['target_slot'][selected]
    wait_ticks=bank['last_wait_positions'][selected]
    old_finish=bank['teacher_finish_positions'][selected]
    first_finish=bank['first_legal_positions'][selected]
    finish_ticks=old_finish if v.arm=='midpoint' else first_finish
    labels=bank['labels' if v.arm=='midpoint' else 'first_labels'][:,selected].contiguous().cuda()
    mask=bank[v.arm+'_mask'][:,selected].contiguous().cuda()
    batch_obs=candidate['obs'][:,local].contiguous()
    batch_anchor=candidate['anchor'][local].contiguous()
    rows=torch.arange(64,device='cuda')
    assert int(mask.sum())==192 and int((mask&(labels!=0)).sum())==128
    assert mask[wait_ticks.cuda(),rows,slots.cuda()].all()
    assert labels[wait_ticks.cuda(),rows,slots.cuda()].eq(0).all()
    assert mask[finish_ticks.cuda(),rows,slots.cuda()].all()
    assert torch.equal(labels[finish_ticks.cuda(),rows,slots.cuda()].cpu(),bank['finish_labels'][selected])
    moved=finish_ticks.ne(old_finish)
    record=dict(seed=v.seed,arm=v.arm,review_index=index,round=v.round,epoch=epoch,block=block,
        protocol_hash=v.review_protocol_hash,candidate_ids=ids,selected_ids=selected,selected_local=local,
        selected_wait_ticks=wait_ticks,selected_finish_ticks=finish_ticks,teacher_finish_ticks=old_finish,
        first_legal_finish_ticks=first_finish,finish_labels=bank['finish_labels'][selected],
        supervision_mask=mask.cpu(),supervision_labels=labels.cpu(),
        candidate_hard_ticks=score['hard_ticks'],candidate_maximum=score['maximum'],candidate_any_finish=score['any_finish'])
    first=index==0
    first_moved=bool(moved.any()) and not v.first_moved_saved
    if first:
        v.save('first_review_before');torch.save(record,v.folder/'first_selection.pt')
    if first_moved:
        v.save('first_moved_before');torch.save(record,v.folder/'first_moved_selection.pt')
    live_before=state_digest(dict(h=v.hidden,world=v.world_state()))
    v.step_identity=dict(review_index=index,candidate_id_hash=state_digest(ids),selected_id_hash=state_digest(selected),
        selected_time_hash=state_digest(finish_ticks),supervision_mask_hash=state_digest(mask.cpu()),
        supervision_labels_hash=state_digest(labels.cpu()),bank_sha256=v.bank_sha256,epoch=epoch,block=block)
    old_counts=v.selection_counts[selected].clone()
    _,loss,gradnorm=v.update(batch_obs,labels,mask,batch_anchor,kind='A')
    assert live_before==state_digest(dict(h=v.hidden,world=v.world_state()))
    v.selection_counts[selected]+=1;v.review_index+=1
    eligible=bank['eligible_mask'][:,ids]
    v.work['A_score_forward_lane_ticks']+=128*128
    v.work['A_score_label_comparisons']+=0
    v.work['A_score_eligible_tick_comparisons']+=int(eligible.sum())
    v.work['A_candidate_presentations']+=128;v.work['A_selected_requests']+=64
    v.work['A_repeated_requests']+=int((old_counts>0).sum())
    v.work['A_unique_requests']=int((v.selection_counts>0).sum())
    v.work['A_selected_conflict_requests']+=int(bank['conflict'][selected].sum())
    v.work['old_history_replay_positions']+=192
    v.work['A_moved_WAIT_rows']+=int(wait_ticks.ne(bank['positions'][selected,1]).sum())
    v.work['A_moved_FINISH_rows']=v.work.get('A_moved_FINISH_rows',0)+int(moved.sum())
    row=dict(event='review_update',seed=v.seed,arm=v.arm,round=v.round,epoch=epoch,block=block,index=index,
        candidates=ids.tolist(),selected=selected.tolist(),candidate_id_hash=state_digest(ids),selected_id_hash=state_digest(selected),
        selected_wait_ticks=wait_ticks.tolist(),selected_finish_ticks=finish_ticks.tolist(),
        teacher_finish_ticks=old_finish.tolist(),first_legal_finish_ticks=first_finish.tolist(),
        finish_labels=bank['finish_labels'][selected].tolist(),moved_finish_count=int(moved.sum()),
        original_new_tick_labels=bank['labels'][finish_ticks,selected,slots].tolist(),
        inserted_FINISH_over_teacher_WAIT_count=int((moved&bank['labels'][finish_ticks,selected,slots].eq(0)).sum()),
        supervision_mask_hash=state_digest(mask.cpu()),supervision_labels_hash=state_digest(labels.cpu()),
        supervised_identity_hash=state_digest(dict(mask=mask.cpu(),labels=labels.cpu()[mask.cpu()])),
        eligible_tick_comparisons=int(eligible.sum()),candidate_count=128,gradient_rows=192,
        gradient_nonwait_rows=128,gradient_WAIT_rows=64,denominator=144,gradient_tensor=[128,64,2,11],
        candidate_scan_tensor=[128,128,2,11],loss=loss,preclip_grad_norm=gradnorm,
        unique_so_far=int((v.selection_counts>0).sum()),reused_requests=int((old_counts>0).sum()),
        live_hidden_world_unchanged=True)
    matched.append(v.folder/'selections.jsonl',row)
    if first:v.save('first_review_after')
    if first_moved:
        v.save('first_moved_after');v.first_moved_saved=True
    return loss


def run(seed):
    from experiments.finish_position import train_audit
    print(json.dumps(dict(event='runtime',seed=seed,executable=sys.executable,
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,GPU_name=torch.cuda.get_device_name(0))),flush=True)
    # Bind only this process's imported runner; no frozen source or input is changed.
    legacy.OUT,legacy.BANK,legacy.PLAN=OUT,BANK,PLAN
    legacy.ARMS=ARMS
    legacy.checked=checked
    legacy.ReviewLearner=ReviewLearner
    legacy.review_update=review_update
    legacy_audit.audit=train_audit.audit
    legacy.run(seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('declare','run','all'))
    parser.add_argument('--seed',type=int,choices=range(3))
    args=parser.parse_args()
    if args.command=='declare':declare()
    elif args.command=='run':run(args.seed)
    else:
        checked()
        processes=[]
        for seed in range(3):
            stream=(OUT/f'seed_{seed:03d}.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u','-m','experiments.finish_position.train','run','--seed',str(seed)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            processes.append((seed,proc,stream))
        results=[]
        for seed,proc,stream in processes:
            results.append(dict(seed=seed,exit_code=proc.wait()));stream.close()
            t.write(OUT/'exit_codes.json',results)
        if any(row['exit_code'] for row in results):raise RuntimeError(results)


if __name__=='__main__':main()
