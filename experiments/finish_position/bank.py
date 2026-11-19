"""Fixed old-A requests: move the sole FINISH target to first native legality.

Every original bank field is retained verbatim. Extra labels and masks are loss
metadata; model inputs remain the unmodified public obs and numeric anchor.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from experiments.wait_position import bank as waits
from experiments.review_selection import bank as old
from experiments.experience_transfer import train_transfer as source

OUT = ROOT / 'data/finish-position/bank-v1'
FORMAT = 'finish-position-fixed-A-first-legal-bank-v1'
EPS = 1e-6


def load(seed, root=OUT):
    return torch.load(Path(root)/f'seed_{seed:03d}'/'bank.pt', map_location='cpu', weights_only=False)


def summary(values):
    x = torch.as_tensor(values, dtype=torch.float64)
    return dict(n=x.numel(), min=float(x.min()), q25=float(x.quantile(.25)),
        median=float(x.median()), mean=float(x.mean()), q75=float(x.quantile(.75)), max=float(x.max()))


def action_name(value):
    return 'WAIT' if value == 0 else 'BEGIN' if value < 3 else 'FINISH'


def histogram(values):
    return dict(sorted(Counter(str(int(v)) for v in values).items(), key=lambda item:int(item[0])))


def distribution(metadata):
    counters = {key:Counter() for key in ('target_actions','target_receipts','target_action_receipt',
        'other_slot_actions','other_slot_receipts','target_teacher_labels')}
    for row in metadata:
        for key in counters: counters[key].update(row['interval_distributions'][key])
    n = len(metadata)
    return dict(requests=n, earlier=sum(r['advance_ticks']>0 for r in metadata),
        same_position=sum(r['advance_ticks']==0 for r in metadata),
        advance_ticks=summary([r['advance_ticks'] for r in metadata]),
        advance_time=summary([r['advance_time'] for r in metadata]),
        advance_tick_histogram=histogram([r['advance_ticks'] for r in metadata]),
        last_WAIT_to_first_FINISH_ticks=summary([r['first_legal_position']-r['last_wait_position'] for r in metadata]),
        last_WAIT_to_teacher_FINISH_ticks=summary([r['teacher_finish_position']-r['last_wait_position'] for r in metadata]),
        first_FINISH_adjacent_to_last_WAIT_fraction=sum(r['first_legal_position']-r['last_wait_position']==1 for r in metadata)/n,
        teacher_FINISH_adjacent_to_last_WAIT_fraction=sum(r['teacher_finish_position']-r['last_wait_position']==1 for r in metadata)/n,
        first_and_teacher_adjacent_fraction=sum(r['advance_ticks']==1 for r in metadata)/n,
        first_oracle_WAIT_fraction=sum(r['first_legal_teacher_label']==0 for r in metadata)/n,
        interval_definition='Original executed source ticks [first legal FINISH, original teacher FINISH), target slot and its other actor slot; receipts are native results of those actual actions. Empty when positions coincide.',
        interval_distributions={k:dict(v) for k,v in counters.items()})


def build(seed):
    torch.set_num_threads(1)
    began=time.perf_counter()
    folder=OUT/f'seed_{seed:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    data=waits.load(seed)
    old_folder=waits.OUT/f'seed_{seed:03d}'
    old_manifest=json.loads((old_folder/'manifest.json').read_text(encoding='utf-8'))
    assert old.sha(old_folder/'bank.pt')==old_manifest['bank_sha256']
    assert data['obs'].shape==(128,256,56)
    by_source=defaultdict(list)
    for i,row in enumerate(data['metadata']): by_source[row['source']].append(i)
    records=[None]*256
    counts=Counter()
    sources={}
    for path,indices in sorted(by_source.items()):
        sources[path]=old.sha(ROOT/path)
        assert sources[path]==old_manifest['sources'][path]
        trace=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        assert trace['teacher_probability']==0 and not trace['teacher_lanes'].any()
        accepted={}
        with source.mapping_scope(source.mapping_table(seed,'related')):
            world=source.worlds.ContextWorld.from_state_dict(trace['initial_world'],device='cpu')
            for tick in range(len(trace['obs'])):
                obs=world.observe()
                teacher=world.oracle('midpoint')
                earliest=world.oracle('earliest')
                assert torch.equal(obs,trace['obs'][tick])
                assert torch.equal(teacher,trace['labels'][tick])
                pending=[]
                for i in indices:
                    row=data['metadata'][i]
                    if not row['cue_tick']<=tick<=row['finish_tick']: continue
                    lane,slot,rid=row['lane'],row['slot'],row['request_id']
                    same=int(world.request_id[lane,slot])==rid
                    if not same: continue
                    assert int(world.actor_ids[lane])==row['actor_id']
                    assert int(world.version[lane,slot])==row['version']
                    assert int(world.raw_payload[lane,slot])==row['raw']
                    assert int(world.current_family[lane,slot])==row['context']
                    assert int(world.payload[lane,slot])==row['A_target']
                    begin=accepted.get((lane,slot,rid))
                    legal=(begin is not None and tick>begin['tick'] and bool(world.active[lane,slot])
                        and bool(world.started[lane,slot]) and world.now>=float(world.ready_time[lane,slot])-EPS
                        and world.now<=float(world.deadline[lane,slot])+EPS)
                    correct=3+4*row['version']+row['A_target']
                    assert (int(earliest[lane,slot])==correct)==legal
                    if legal and records[i] is None:
                        assert row['start_tick']<=tick<row['stop_tick']
                        last=int(data['eligible_mask'][:,i,slot].nonzero()[-1])
                        rec=dict(sample_index=i,id=row['id'],source=path,source_sha256=sources[path],
                            actor_id=row['actor_id'],lane=lane,slot=slot,request_id=rid,
                            version=row['version'],raw=row['raw'],context=row['context'],correct_payload=row['A_target'],
                            correct_finish_label=correct,accepted_begin_tick=begin['tick'],accepted_begin_time=begin['time'],
                            ready_time=float(world.ready_time[lane,slot]),deadline=float(world.deadline[lane,slot]),
                            first_legal_tick=tick,first_legal_position=tick-row['start_tick'],first_legal_time=world.now,
                            first_legal_world_tick=world.tick,first_legal_teacher_label=int(teacher[lane,slot]),
                            first_legal_active=True,first_legal_started=True,
                            last_wait_position=last,last_wait_source_tick=last+row['start_tick'],
                            teacher_finish_tick=row['finish_tick'],teacher_finish_position=row['positions'][2],
                            window_start_tick=row['start_tick'],window_stop_tick=row['stop_tick'],
                            interval_ticks=[],interval_distributions={k:Counter() for k in
                                ('target_actions','target_receipts','target_action_receipt','other_slot_actions',
                                 'other_slot_receipts','target_teacher_labels')})
                        records[i]=rec
                        assert last<rec['first_legal_position']<=rec['teacher_finish_position']
                    rec=records[i]
                    if tick==row['finish_tick']:
                        assert legal and rec is not None and int(teacher[lane,slot])==correct
                        assert float(world.ready_time[lane,slot])==rec['ready_time']
                        assert float(world.deadline[lane,slot])==rec['deadline']
                        rec.update(teacher_finish_time=world.now,teacher_finish_world_tick=world.tick,
                            teacher_finish_active=True,teacher_finish_started=True,
                            advance_ticks=tick-rec['first_legal_tick'],advance_time=world.now-rec['first_legal_time'])
                    if rec is not None and rec['first_legal_tick']<=tick<row['finish_tick']:
                        pending.append((i,lane,slot,int(teacher[lane,slot])))
                actions=trace['actions'][tick]
                begin_ok=(actions==1+world.version)&world.active&~world.started
                begin_records=[(int(l),int(s),int(world.request_id[l,s])) for l,s in begin_ok.nonzero().tolist()]
                now=world.now
                world.act(actions)
                assert bool((world.last_result[begin_ok]==1).all())
                for lane,slot,rid in begin_records: accepted[lane,slot,rid]=dict(tick=tick,time=now)
                for i,lane,slot,label in pending:
                    rec=records[i]
                    a,ack,oa,oack=(int(actions[lane,slot]),int(world.last_result[lane,slot]),
                                  int(actions[lane,1-slot]),int(world.last_result[lane,1-slot]))
                    rec['interval_ticks'].append(dict(source_tick=tick,target_action=a,target_result=ack,
                        other_slot_action=oa,other_slot_result=oack,target_teacher_label=label))
                    d=rec['interval_distributions']
                    for key,value in (('target_actions',str(a)),('target_receipts',str(ack)),
                        ('target_action_receipt',f'{action_name(a)}:{ack}'),('other_slot_actions',str(oa)),
                        ('other_slot_receipts',str(oack)),('target_teacher_labels',str(label))): d[key][value]+=1
                counts['native_world_ticks']+=1
                counts['native_obs_label_rows']+=teacher.numel()
    assert all(r is not None and 'advance_ticks' in r for r in records)
    first=torch.tensor([r['first_legal_position'] for r in records],dtype=torch.long)
    teacher=torch.tensor([r['teacher_finish_position'] for r in records],dtype=torch.long)
    last=torch.tensor([r['last_wait_position'] for r in records],dtype=torch.long)
    correct=torch.tensor([r['correct_finish_label'] for r in records],dtype=torch.long)
    midpoint_mask=torch.zeros_like(data['mask'])
    first_mask=torch.zeros_like(data['mask'])
    first_labels=data['labels'].clone()
    midpoint_positions=torch.stack((data['positions'][:,0],last,teacher),dim=1)
    first_positions=torch.stack((data['positions'][:,0],last,first),dim=1)
    for i,row in enumerate(data['metadata']):
        slot=row['slot']
        midpoint_mask[midpoint_positions[i],i,slot]=True
        first_mask[first_positions[i],i,slot]=True
        first_labels[first[i],i,slot]=correct[i]
        assert bool((data['labels'][midpoint_positions[i][1:2],i,slot]==0).all())
        for mask,labels in ((midpoint_mask,data['labels']),(first_mask,first_labels)):
            chosen=labels[:,i][mask[:,i]]
            assert len(chosen)==3 and int((chosen==0).sum())==1 and int((chosen>=3).sum())==1
            assert int(((chosen==1)|(chosen==2)).sum())==1
            assert not mask[:,i,1-slot].any()
    result=dict(data)
    result.update(finish_format=FORMAT,first_legal_positions=first,teacher_finish_positions=teacher,
        last_wait_positions=last,finish_labels=correct,first_labels=first_labels,
        midpoint_mask=midpoint_mask,first_mask=first_mask,midpoint_positions=midpoint_positions,
        first_positions=first_positions,finish_metadata=records)
    for key,value in data.items():
        if torch.is_tensor(value):
            assert torch.equal(result[key],value) or torch.equal(torch.isnan(result[key]),torch.isnan(value)) and torch.allclose(result[key],value,equal_nan=True)
        else: assert result[key]==value,key
    torch.save(result,folder/'bank.pt')
    old.write(folder/'finish_metadata.json',records)
    dist=distribution(records)
    manifest=dict(format=FORMAT,seed=seed,created_utc=datetime.now(timezone.utc).isoformat(),
        original_bank=str((old_folder/'bank.pt').relative_to(ROOT)),original_bank_sha256=old.sha(old_folder/'bank.pt'),
        original_manifest_sha256=old.sha(old_folder/'manifest.json'),original_audit_sha256=old.sha(old_folder/'audit.json'),
        original_fields_exact=list(data),samples=256,sequence_ticks=128,sources=sources,
        checkpoint_source_manifest=str((old.OUT/f'seed_{seed:03d}'/'manifest.json').relative_to(ROOT)),
        intervention='Keep BEGIN and last eligible processing WAIT fixed; move exactly one same-request correct-version correct-payload FINISH from original midpoint-teacher position to the first tick with native active and started, native now >= ready_time-EPS and now <= deadline+EPS after native accepted BEGIN. EPS=1e-6 matches native world. No request reselection.',
        inputs='Only original obs[128,256,56] and numeric anchor[256,192]. All added fields, all labels, IDs, private state, masks and timing are loss/audit metadata, never policy input.',
        losses='midpoint: original labels and midpoint_mask; first: first_labels and first_mask. Each mask selects exactly BEGIN, last eligible WAIT, and one FINISH on target slot per request. Every other row has zero A-loss weight; old teacher label may remain stored but is unmasked in first.',
        unchanged=['obs','anchor','actions','metadata','ids','original labels/masks/all wait-bank fields'],
        request_reselection=False,native_replay_checks=dict(counts),distribution=dist,
        bank_sha256=old.sha(folder/'bank.pt'),metadata_sha256=old.sha(folder/'finish_metadata.json'),
        implementation_sha256=old.sha(__file__),world_sha256=old.sha(source.worlds.__file__),
        base_world_sha256=old.sha(ROOT/'experiments/continuity/world.py'),elapsed_seconds=time.perf_counter()-began)
    old.write(folder/'manifest.json',manifest)
    print(json.dumps(dict(seed=seed,**dist,seconds=time.perf_counter()-began)),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=range(3))
    args=parser.parse_args()
    for seed in range(3) if args.seed is None else (args.seed,): build(seed)
