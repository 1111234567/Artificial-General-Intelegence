"""Read-only boundary intents on the already-trained 256-request A bank.

This is an off-policy fitting-bank diagnostic from original numeric anchors,
not a closed-loop test. No model action changes any public history or world.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.finish_position import bank as banks
from experiments.review_selection import bank as old
from experiments.learn_remember.gru_train import GRUAgent
from experiments.experience_transfer import train_transfer as source
from experiments.continuity.run import state_digest

OUT=ROOT/'data/finish-position/bank-check-v1'
PLAN=OUT/'protocol.json'
ARMS=('B000','midpoint','first')
STAGES=('last_wait','first_legal','midpoint_teacher')


def checkpoint(seed,arm):
    return (source.OUT/f'seed_{seed:03d}/related/B000.pt' if arm=='B000' else
            ROOT/f'data/finish-position/train-v1/seed_{seed:03d}/{arm}/B008.pt')


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not PLAN.exists(),PLAN
    files=[Path(__file__),Path(banks.__file__),ROOT/'experiments/learn_remember/gru_train.py',
        ROOT/'data/finish-position/bank-v1/protocol.json']
    for seed in range(3):
        files.extend(banks.OUT/f'seed_{seed:03d}'/name for name in ('bank.pt','manifest.json','audit.json'))
    plan=dict(format='finish-position-A-fitting-bank-readout-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[0,1,2],arms=list(ARMS),stages=list(STAGES),device='cpu',
        checkpoints={str(seed):{arm:str(checkpoint(seed,arm).relative_to(ROOT)) for arm in ARMS} for seed in range(3)},
        stage_selection='Fixed before reading tested endpoint results: original B000 and both final B008 endpoints only. Wait until every B008 exists; do not select a stage from scores.',
        sample='All original 256 A fitting-bank requests per seed, all four raws equally represented. Same full128-tick public windows and original numeric anchors for all arms; no reselection or exclusion.',
        inputs='Policy receives only original obs and original anchor. Labels/IDs/private times and all selection masks are consulted only after both forward computations. The frozen B000-derived numeric anchor is deliberately retained even for B008 models.',
        metrics=dict(boundaries='Eleven-way argmax WAIT, any FINISH>=3, exact correct FINISH (same version and payload), and conditional8 correct argmax restricted to3..10, at fixed last processing WAIT/first legal tick/original midpoint-teacher FINISH.',
            paired='last WAIT eleven-way action0 AND the immediately next first-legal tick eleven-way exact correct FINISH; both must be true for same request.',
            processing='Native eligible_mask copied from bank: same request, active and started after real accepted BEGIN, strictly before ready, original teacher WAIT. Count requests with any eleven-way FINISH argmax and tick-level FINISH/correct-FINISH counts.',
            grouping='All requests, conflicting raw group, other/shared raw group, and each raw0..3; all numerators and denominators retained.'),
        numerical_verification='Full-sequence GRU forward from original anchor independently compared with128 sequential model.step calls from that same anchor. Require finite parameters/inputs/logits/hidden, identical eleven-way argmax for every tick/sample/slot, and allclose logits/hidden atol5e-6 rtol1e-5; retain max abs error and all logits for independent NumPy count audit.',
        interpretation='Fitting-bank, anchored, off-policy public-history readout. Scores do not measure closed-loop completion, actual native acceptance, heldout generalization or free-action A/B switching. Processing FINISH is an intent at stored history, not an executed action.',
        source_sha256={str(p.relative_to(ROOT)):old.sha(p) for p in files})
    plan['protocol_hash']=state_digest(plan)
    old.write(PLAN,plan)
    print(json.dumps(dict(event='declared',path=str(PLAN),protocol_hash=plan['protocol_hash'])),flush=True)


def checked():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    assert plan['protocol_hash']==state_digest({k:v for k,v in plan.items() if k!='protocol_hash'})
    for path,digest in plan['source_sha256'].items():assert old.sha(ROOT/path)==digest,path
    return plan


def group_names(raw,conflict):
    return ('all','conflict' if conflict else 'other',f'raw_{raw}')


def counts_torch(logits,bank):
    actions=logits.argmax(-1)
    conditional=logits[...,3:].argmax(-1)+3
    counts={key:Counter() for key in ('all','conflict','other','raw_0','raw_1','raw_2','raw_3')}
    rows=[]
    for i,row in enumerate(bank['metadata']):
        slot=int(bank['target_slot'][i]);target=int(bank['finish_labels'][i])
        positions=[int(bank['last_wait_positions'][i]),int(bank['first_legal_positions'][i]),int(bank['teacher_finish_positions'][i])]
        assert positions[0]+1==positions[1]
        selected=logits[positions,i,slot]
        acts=actions[positions,i,slot]
        mask=bank['eligible_mask'][:,i,slot]
        process=actions[:,i,slot][mask]
        one=Counter(n=1,matured=1,processing_ticks=int(mask.sum()),
            processing_any_finish=int(bool((process>=3).any())),
            processing_finish_ticks=int((process>=3).sum()),
            processing_correct_finish_ticks=int((process==target).sum()),
            wait_then_first_correct=int(int(acts[0])==0 and int(acts[1])==target))
        for j,stage in enumerate(STAGES):
            a=int(acts[j])
            one[stage+'_wait']=int(a==0)
            one[stage+'_finish']=int(a>=3)
            one[stage+'_correct']=int(a==target)
            one[stage+'_conditional8']=int(conditional[positions[j],i,slot]==target)
        for name in group_names(row['raw'],row['conflict']):counts[name].update(one)
        rows.append(dict(index=i,id=row['id'],raw=row['raw'],conflict=row['conflict'],slot=slot,target=target,
            positions=positions,actions=acts.tolist(),conditional8=conditional[positions,i,slot].tolist(),
            boundary_logits=selected.tolist(),boundary_finish_margins=(selected[:,3:].max(-1).values-selected[:,:3].max(-1).values).tolist(),
            processing_positions=mask.nonzero().flatten().tolist(),processing_actions=process.tolist(),
            processing_any_finish=bool((process>=3).any()),wait_then_first_correct=bool(one['wait_then_first_correct'])))
    return {key:dict(value) for key,value in counts.items()},rows


@torch.inference_mode()
def score(seed,arm):
    plan=checked()
    # No endpoint is read until all six final endpoints are available.
    for s in range(3):
        for a in ('midpoint','first'):assert checkpoint(s,a).exists(),checkpoint(s,a)
    torch.set_num_threads(1)
    began=time.perf_counter()
    folder=OUT/f'seed_{seed:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/f'{arm}.json'
    assert not result_path.exists(),result_path
    cp_path=checkpoint(seed,arm)
    cp_sha=old.sha(cp_path)
    bank_path=banks.OUT/f'seed_{seed:03d}/bank.pt'
    bank_sha=old.sha(bank_path)
    bank=banks.load(seed)
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    model=GRUAgent(**cp['model_config'])
    model.load_state_dict(cp['model']);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    obs,anchor=bank['obs'],bank['anchor']
    assert bool(torch.isfinite(obs).all()) and bool(torch.isfinite(anchor).all())
    # Labels, IDs, masks and timing never appear in either model invocation.
    logits,final=model(obs,anchor)
    h=anchor.clone();sequential=[]
    for public_obs in obs:
        output,h=model.step(public_obs,h)
        sequential.append(output)
    step_logits=torch.stack(sequential)
    assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(step_logits).all())
    assert bool(torch.isfinite(final).all()) and bool(torch.isfinite(h).all())
    assert torch.equal(logits.argmax(-1),step_logits.argmax(-1)),(seed,arm,'argmax')
    assert torch.allclose(logits,step_logits,atol=5e-6,rtol=1e-5),(seed,arm,'logits')
    assert torch.allclose(final,h,atol=5e-6,rtol=1e-5),(seed,arm,'hidden')
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    counts,rows=counts_torch(logits,bank)
    assert counts['all']['n']==256 and counts['conflict']['n']==counts['other']['n']==128
    assert all(counts['raw_'+str(raw)]['n']==64 for raw in range(4))
    tensor_path=folder/f'{arm}_logits.pt'
    assert not tensor_path.exists(),tensor_path
    torch.save(dict(logits=logits,step_logits=step_logits,final_hidden=final,step_final_hidden=h,
        protocol_hash=plan['protocol_hash'],bank_sha256=bank_sha,checkpoint_sha256=cp_sha),tensor_path)
    assert old.sha(cp_path)==cp_sha and old.sha(bank_path)==bank_sha
    result=dict(seed=seed,arm=arm,stage='B000' if arm=='B000' else 'B008',
        protocol_hash=plan['protocol_hash'],bank_sha256=bank_sha,checkpoint=str(cp_path.relative_to(ROOT)),checkpoint_sha256=cp_sha,
        tensor_path=str(tensor_path.relative_to(ROOT)),tensor_sha256=old.sha(tensor_path),counts=counts,rows=rows,
        verification=dict(finite=True,full_vs_step_argmax_exact=True,
            compared_logits=int(logits.numel()),compared_argmax_rows=int(logits.shape[0]*logits.shape[1]*logits.shape[2]),
            logits_max_abs_difference=float((logits-step_logits).abs().max()),hidden_max_abs_difference=float((final-h).abs().max()),
            atol=5e-6,rtol=1e-5,model_parameters_unchanged=True,checkpoint_and_bank_hash_unchanged=True,
            model_inputs=['original public obs','original numeric anchor']),elapsed_seconds=time.perf_counter()-began)
    old.write(result_path,result)
    print(json.dumps(dict(seed=seed,arm=arm,counts=counts['all'],verification=result['verification'],seconds=result['elapsed_seconds'])),flush=True)


def numpy_counts(logits,bank):
    """Independent count implementation, no scorer helper or stored row decisions."""
    act=np.argmax(logits,axis=-1);conditional=np.argmax(logits[...,3:],axis=-1)+3
    records=[]
    for i in range(256):
        slot=int(bank['target_slot'][i]);target=int(bank['finish_labels'][i])
        ts=[int(bank[k][i]) for k in ('last_wait_positions','first_legal_positions','teacher_finish_positions')]
        proc=np.flatnonzero(bank['eligible_mask'][:,i,slot].numpy())
        assert ts[0]+1==ts[1]
        a=act[ts,i,slot];p=act[proc,i,slot]
        record=dict(n=1,matured=1,processing_ticks=len(proc),processing_any_finish=int(np.any(p>=3)),
            processing_finish_ticks=int(np.sum(p>=3)),processing_correct_finish_ticks=int(np.sum(p==target)),
            wait_then_first_correct=int(a[0]==0 and a[1]==target))
        for j,name in enumerate(STAGES):
            record.update({name+'_wait':int(a[j]==0),name+'_finish':int(a[j]>=3),name+'_correct':int(a[j]==target),
                name+'_conditional8':int(conditional[ts[j],i,slot]==target)})
        records.append(record)
    raw=bank['raw'].numpy();conflict=bank['conflict'].numpy()
    selectors={'all':np.ones(256,dtype=bool),'conflict':conflict,'other':~conflict}
    selectors.update({'raw_'+str(r):raw==r for r in range(4)})
    return {name:{key:sum(records[i][key] for i in np.flatnonzero(m)) for key in records[0]}
        for name,m in selectors.items()}


def audit():
    plan=checked()
    total=Counter()
    hashes={}
    for seed in range(3):
        bank=banks.load(seed)
        for arm in ARMS:
            path=OUT/f'seed_{seed:03d}/{arm}.json'
            value=json.loads(path.read_text(encoding='utf-8'))
            tensor_path=ROOT/value['tensor_path']
            assert old.sha(tensor_path)==value['tensor_sha256']
            assert old.sha(ROOT/value['checkpoint'])==value['checkpoint_sha256']
            assert old.sha(banks.OUT/f'seed_{seed:03d}/bank.pt')==value['bank_sha256']
            saved=torch.load(tensor_path,map_location='cpu',weights_only=False)
            logits=saved['logits'].numpy();step=saved['step_logits'].numpy()
            assert logits.shape==(128,256,2,11) and np.isfinite(logits).all() and np.isfinite(step).all()
            assert np.array_equal(logits.argmax(-1),step.argmax(-1))
            np.testing.assert_allclose(logits,step,rtol=1e-5,atol=5e-6)
            independently=numpy_counts(logits,bank)
            assert independently==value['counts'],(seed,arm,'counts')
            for i,row in enumerate(value['rows']):
                slot=int(bank['target_slot'][i])
                ts=[int(bank[k][i]) for k in ('last_wait_positions','first_legal_positions','teacher_finish_positions')]
                assert row['index']==i and row['id']==bank['ids'][i] and row['positions']==ts
                assert row['target']==int(bank['finish_labels'][i]) and row['slot']==slot
                assert row['actions']==logits[ts,i,slot].argmax(-1).tolist()
                np.testing.assert_array_equal(row['boundary_logits'],logits[ts,i,slot])
                proc=np.flatnonzero(bank['eligible_mask'][:,i,slot].numpy())
                assert row['processing_positions']==proc.tolist()
                assert row['processing_actions']==logits[proc,i,slot].argmax(-1).tolist()
                total['requests']+=1
                total['processing_tick_decisions']+=len(proc)
            total['checkpoints']+=1
            total['finite_logits']+=int(logits.size)
            total['full_vs_step_argmax_rows']+=int(np.prod(logits.shape[:-1]))
            hashes[str(path.relative_to(ROOT))]=old.sha(path)
            hashes[str(tensor_path.relative_to(ROOT))]=old.sha(tensor_path)
    result=dict(passed=True,protocol_hash=plan['protocol_hash'],checks=dict(total),independent_numpy_counts_exact=True,
        every_retained_row_and_processing_tick_exact=True,all_saved_logits_finite=True,
        full_sequence_and_tickwise_decisions_exact=True,hashes=hashes)
    old.write(OUT/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'}),flush=True)


def report():
    checked()
    a=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    assert a['passed']
    pooled={arm:{g:Counter() for g in ('all','conflict','other','raw_0','raw_1','raw_2','raw_3')} for arm in ARMS}
    seeds={}
    for seed in range(3):
        seeds[str(seed)]={}
        for arm in ARMS:
            p=OUT/f'seed_{seed:03d}/{arm}.json'
            assert old.sha(p)==a['hashes'][str(p.relative_to(ROOT))]
            value=json.loads(p.read_text(encoding='utf-8'))
            seeds[str(seed)][arm]=value['counts']
            for group,c in value['counts'].items():pooled[arm][group].update(c)
    pooled={arm:{g:dict(c) for g,c in groups.items()} for arm,groups in pooled.items()}
    old.write(OUT/'summary.json',dict(pooled=pooled,seeds=seeds,audit_sha256=old.sha(OUT/'audit.json'),
        definitions=json.loads(PLAN.read_text(encoding='utf-8'))['metrics']))
    lines=['# Old-A fitting-bank boundary readout','',
        'Read-only frozen B000/B008 policies see the same256 original A public windows per seed and the exact original B000-derived numeric anchors. All768 requests are retained. This is an anchored off-policy diagnostic of the fitting bank. It measures action intents on stored histories; it does not measure free behavior, heldout generalization, actual acceptance or completion.','',
        '| Group | Endpoint | n | Last WAIT chooses WAIT | First legal exact FINISH | Teacher tick exact FINISH | WAIT then next exact FINISH | Any processing FINISH |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for group in ('all','conflict','other'):
        for arm in ARMS:
            c=pooled[arm][group];n=c['n']
            values=[c[k] for k in ('last_wait_wait','first_legal_correct','midpoint_teacher_correct','wait_then_first_correct','processing_any_finish')]
            cells=' | '.join(f'{v}/{n} ({100*v/n:.2f}%)' for v in values)
            lines.append(f'| {group} | {arm} | {n} | {cells} |')
    lines+=['','Exact FINISH means the native eleven-way argmax selects the correct version and payload. WAIT then FINISH requires both decisions on the same request and consecutive ticks. Processing FINISH means any FINISH argmax within the independently native-audited eligible processing interval, not an executed early action.','',
        '| Seed | Endpoint | Conflict n | Last WAIT correct | First legal exact FINISH | WAIT then next exact FINISH | Any processing FINISH |',
        '|---|---|---:|---:|---:|---:|---:|']
    for seed in range(3):
        for arm in ARMS:
            c=seeds[str(seed)][arm]['conflict'];n=c['n']
            lines.append(f"| {seed} | {arm} | {n} | {c['last_wait_wait']} | {c['first_legal_correct']} | {c['wait_then_first_correct']} | {c['processing_any_finish']} |")
    lines+=['','Full numerator/denominator counts for all, conflict, other and every raw are in summary.json. Per-request three-boundary logits/actions and every processing tick decision are in seed_NNN/ARM.json; full two-slot logits and independent sequential logits are in ARM_logits.pt. Conditional8 payload/version correctness is retained separately from the eleven-way action choice.','',
        'All nine full-sequence forwards were checked against128 separate model.step calls from the same original anchor. The audit independently recomputes all counts using NumPy and checks every saved boundary and processing action. All inputs, parameters, logits and hidden states are finite; model weights and checkpoint/bank hashes remain unchanged.','',
        'Compare these fitting-bank metrics with the separately declared independent-teacher A/B readout under data/finish-position/boundary-v1. Histories and anchor construction differ, so bank-to-heldout differences are descriptive, not a controlled effect of one history feature. No diagnostic selection changed the training protocol or selected checkpoints.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(OUT/'report.md'),pooled=pooled)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('declare','score','audit','report'))
    p.add_argument('--seed',type=int,choices=range(3));p.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    if args.command=='declare':declare()
    elif args.command=='score':score(args.seed,args.arm)
    elif args.command=='audit':audit()
    else:report()
