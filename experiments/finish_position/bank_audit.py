"""Independent frame reconstruction and native FINISH acceptance audit."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from experiments.finish_position import bank as builder
from experiments.wait_position import bank as waits
from experiments.review_selection import bank as old
from experiments.experience_transfer import train_transfer as source


def exact(a,b):
    return torch.equal(a,b) or (a.dtype.is_floating_point and torch.allclose(a,b,rtol=0,atol=0,equal_nan=True))


def audit(seed):
    torch.set_num_threads(1)
    began=time.perf_counter()
    folder=builder.OUT/f'seed_{seed:03d}'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    bank=builder.load(seed)
    original=waits.load(seed)
    for key,value in original.items():
        assert exact(bank[key],value) if torch.is_tensor(value) else bank[key]==value,key
    original_folder=waits.OUT/f'seed_{seed:03d}'
    for file,key in (('bank.pt','original_bank_sha256'),('manifest.json','original_manifest_sha256'),('audit.json','original_audit_sha256')):
        assert old.sha(original_folder/file)==manifest[key]
    original_audit=json.loads((original_folder/'audit.json').read_text(encoding='utf-8'))
    assert original_audit['passed'] and original_audit['bank_sha256']==manifest['original_bank_sha256']
    assert old.sha(folder/'bank.pt')==manifest['bank_sha256']
    assert old.sha(folder/'finish_metadata.json')==manifest['metadata_sha256']
    assert old.sha(builder.__file__)==manifest['implementation_sha256']
    assert old.sha(source.worlds.__file__)==manifest['world_sha256']
    assert old.sha(ROOT/'experiments/continuity/world.py')==manifest['base_world_sha256']
    base_manifest=json.loads((ROOT/manifest['checkpoint_source_manifest']).read_text(encoding='utf-8'))
    cp=ROOT/base_manifest['checkpoint']
    assert old.sha(cp)==base_manifest['checkpoint_sha256']
    model,_,_,_=source.load_checkpoint(cp,'cpu')
    model.eval()
    assert len(bank['ids'])==len(set(bank['ids']))==256
    assert torch.bincount(bank['raw']).tolist()==[64]*4
    assert bank['obs'].shape==(128,256,56) and bank['anchor'].shape==(256,192)
    groups=defaultdict(list)
    for i,row in enumerate(bank['metadata']):groups[row['source']].append(i)
    total=Counter()
    for path,indices in sorted(groups.items()):
        assert old.sha(ROOT/path)==manifest['sources'][path]
        trace=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        assert trace['teacher_probability']==0 and not trace['teacher_lanes'].any()
        starts=defaultdict(list)
        interventions=defaultdict(list)
        for i in indices:
            row=bank['metadata'][i]
            rec=bank['finish_metadata'][i]
            lane,start,stop=row['lane'],row['start_tick'],row['stop_tick']
            assert stop-start==128 and row['cue_tick']-start==32
            assert torch.equal(bank['obs'][:,i],trace['obs'][start:stop,lane])
            assert torch.equal(bank['actions'][:,i],trace['actions'][start:stop,lane])
            starts[start].append(i)
            for name in ('first_legal_tick','teacher_finish_tick'):
                interventions[rec[name]].append((i,name))
        frames=defaultdict(list)
        h=trace['anchor'].clone()
        with source.mapping_scope(source.mapping_table(seed,'related')),torch.no_grad():
            world=source.worlds.ContextWorld.from_state_dict(trace['initial_world'],device='cpu')
            for tick in range(len(trace['obs'])):
                for i in starts[tick]:
                    assert torch.equal(h[bank['metadata'][i]['lane']],bank['anchor'][i]),(seed,i,'anchor')
                    total['numeric_anchor_stepwise_exact']+=1
                obs=world.observe()
                teacher=world.oracle('midpoint')
                earliest=world.oracle('earliest')
                assert torch.equal(obs,trace['obs'][tick])
                assert torch.equal(teacher,trace['labels'][tick])
                for field in ('request_id','active','started','ready_time','deadline','duration','version',
                              'payload','raw_payload','current_family'):
                    frames[field].append(getattr(world,field).clone())
                frames['now'].append(world.now)
                frames['world_tick'].append(world.tick)
                frames['teacher'].append(teacher.clone())
                frames['earliest'].append(earliest.clone())
                if interventions[tick]:
                    # A separate cloned native world executes the proposed correct
                    # FINISH actions; the actual archived replay remains intact.
                    trial=deepcopy(world)
                    proposed=torch.zeros_like(teacher)
                    for i,name in interventions[tick]:
                        r=bank['metadata'][i]
                        assert int(world.request_id[r['lane'],r['slot']])==r['request_id']
                        proposed[r['lane'],r['slot']]=3+4*r['version']+r['A_target']
                    trial.act(proposed)
                    for i,name in interventions[tick]:
                        r=bank['metadata'][i]
                        lane,slot=r['lane'],r['slot']
                        assert int(trial.last_result[lane,slot])==1
                        assert not bool(trial.active[lane,slot]) and not bool(trial.started[lane,slot])
                        total['native_counterfactual_'+name+'_accepted']+=1
                _,h=model.step(obs,h)
                world.act(trace['actions'][tick])
                frames['result'].append(world.last_result.clone())
                total['native_world_ticks']+=1
                total['native_obs_label_rows']+=teacher.numel()
        f={key:torch.tensor(value,dtype=torch.float64 if key=='now' else torch.long)
            if key in ('now','world_tick') else torch.stack(value) for key,value in frames.items()}
        ticks=torch.arange(len(trace['obs']))
        begin=((trace['actions']==1+f['version'])&f['active']&~f['started'])
        assert bool((f['result'][begin]==1).all())
        for i in indices:
            row,rec=bank['metadata'][i],bank['finish_metadata'][i]
            lane,slot,rid=row['lane'],row['slot'],row['request_id']
            start,stop=row['start_tick'],row['stop_tick']
            same=f['request_id'][:,lane,slot]==rid
            native_begin=(same&begin[:,lane,slot]).nonzero().flatten()
            assert len(native_begin)==1 and int(native_begin[0])>=row['cue_tick']
            b=int(native_begin[0])
            legal=(same&(ticks>b)&f['active'][:,lane,slot]&f['started'][:,lane,slot]
                &(f['now']>=f['ready_time'][:,lane,slot]-1e-6)
                &(f['now']<=f['deadline'][:,lane,slot]+1e-6))
            correct=3+4*row['version']+row['A_target']
            legal_ticks=legal.nonzero().flatten()
            assert len(legal_ticks)>0
            first=int(legal_ticks[0])
            teacher_ticks=(same&(f['teacher'][:,lane,slot]==correct)).nonzero().flatten()
            teacher=int(teacher_ticks[0])
            assert teacher==row['finish_tick']
            assert int(bank['first_legal_positions'][i])==first-start
            assert int(bank['teacher_finish_positions'][i])==teacher-start
            # Reconstruct every admissible processing WAIT independently.
            wait=(same&(ticks>b)&f['active'][:,lane,slot]&f['started'][:,lane,slot]
                &(f['now']<f['ready_time'][:,lane,slot])&(f['teacher'][:,lane,slot]==0))
            expected_wait=torch.zeros((128,2),dtype=torch.bool)
            expected_wait[:,slot]=wait[start:stop]
            assert torch.equal(expected_wait,bank['eligible_mask'][:,i])
            last=int(wait[start:stop].nonzero()[-1])
            assert int(bank['last_wait_positions'][i])==last
            assert row['start_tick']<=first<=teacher<row['stop_tick']
            assert b<first and last<first-start
            for t in (first,teacher):
                assert bool(legal[t]) and int(f['earliest'][t,lane,slot])==correct
                assert int(f['request_id'][t,lane,slot])==rid
                assert int(f['version'][t,lane,slot])==row['version']
                assert int(f['raw_payload'][t,lane,slot])==row['raw']
                assert int(f['current_family'][t,lane,slot])==row['context']
                assert int(f['payload'][t,lane,slot])==row['A_target']
            for field,key,t in (('ready_time','ready_time',first),('deadline','deadline',first),
                ('now','accepted_begin_time',b),('now','first_legal_time',first),('now','teacher_finish_time',teacher)):
                value=float(f[field][t]) if field=='now' else float(f[field][t,lane,slot])
                assert rec[key]==value,(seed,i,key)
            assert rec['first_legal_tick']==first and rec['teacher_finish_tick']==teacher
            assert rec['accepted_begin_tick']==b and rec['last_wait_position']==last
            assert rec['correct_finish_label']==correct==int(bank['finish_labels'][i])
            assert rec['advance_ticks']==teacher-first
            assert rec['advance_time']==float(f['now'][teacher]-f['now'][first])
            assert rec['first_legal_teacher_label']==int(f['teacher'][first,lane,slot])
            assert rec['first_legal_world_tick']==int(f['world_tick'][first])
            assert rec['teacher_finish_world_tick']==int(f['world_tick'][teacher])
            expected_labels=bank['labels'][:,i].clone()
            expected_labels[first-start,slot]=correct
            assert torch.equal(expected_labels,bank['first_labels'][:,i])
            assert torch.equal(bank['labels'][:,i,slot],f['teacher'][start:stop,lane,slot])
            for arm,finish,labels in (('midpoint',teacher,bank['labels']),('first',first,bank['first_labels'])):
                positions=[row['positions'][0],last,finish-start]
                m=torch.zeros((128,2),dtype=torch.bool)
                m[positions,slot]=True
                assert torch.equal(bank[arm+'_positions'][i],torch.tensor(positions))
                assert torch.equal(m,bank[arm+'_mask'][:,i])
                chosen=labels[:,i][m]
                assert chosen.tolist()==[1+row['version'],0,correct]
                total['three_rows_one_BEGIN_one_WAIT_one_FINISH_'+arm]+=1
            interval=[]
            d={k:Counter() for k in rec['interval_distributions']}
            for t in range(first,teacher):
                a,ack,oa,oack=int(trace['actions'][t,lane,slot]),int(f['result'][t,lane,slot]),int(trace['actions'][t,lane,1-slot]),int(f['result'][t,lane,1-slot])
                label=int(f['teacher'][t,lane,slot])
                interval.append(dict(source_tick=t,target_action=a,target_result=ack,other_slot_action=oa,other_slot_result=oack,target_teacher_label=label))
                name='WAIT' if a==0 else 'BEGIN' if a<3 else 'FINISH'
                for key,value in (('target_actions',str(a)),('target_receipts',str(ack)),('target_action_receipt',f'{name}:{ack}'),
                    ('other_slot_actions',str(oa)),('other_slot_receipts',str(oack)),('target_teacher_labels',str(label))):d[key][value]+=1
            assert interval==rec['interval_ticks'] and d==rec['interval_distributions']
            total['requests']+=1
            total['interval_ticks_audited']+=len(interval)
    assert total['requests']==total['numeric_anchor_stepwise_exact']==256
    assert total['native_counterfactual_first_legal_tick_accepted']==256
    assert total['native_counterfactual_teacher_finish_tick_accepted']==256
    result=dict(seed=seed,passed=True,checks=dict(total),all_original_wait_bank_fields_exact=True,
        request_identity_actor_slot_version_payload_context_exact=True,within_original_128tick_windows=True,
        independent_complete_first_legal_scan=True,accepted_BEGIN_receipts_verified=True,
        both_proposed_FINISH_positions_native_acceptance_verified=True,
        exactly_one_FINISH_positive_per_request_and_arm=True,unchanged_BEGIN_and_last_eligible_WAIT=True,
        only_original_obs_and_anchor_are_model_inputs=True,private_fields_are_loss_audit_only=True,
        numeric_anchor_independently_rebuilt_exact=True,anchor_tolerance=dict(atol=0,rtol=0),
        bank_sha256=old.sha(folder/'bank.pt'),manifest_sha256=old.sha(folder/'manifest.json'),
        implementation_sha256=old.sha(__file__),elapsed_seconds=time.perf_counter()-began)
    old.write(folder/'audit.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=range(3))
    args=parser.parse_args()
    for seed in range(3) if args.seed is None else (args.seed,):audit(seed)
