"""Read-only mechanism diagnosis from retained actual training selections.

No policy is rescored and no selection is changed.  Physical timing is joined
from the frozen bank solely to interpret the positions the trainer did choose.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import torch
from experiments.wait_position import bank
from experiments.review_selection import bank as old

TRAIN=ROOT/'data/wait-position/train-v1'
ARMS=('midpoint','hard','random')


def metric(values):
    a=torch.tensor(values,dtype=torch.float64)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a),min=float(a.min()),q10=float(a.quantile(.1)),q25=float(a.quantile(.25)),
                median=float(a.quantile(.5)),mean=float(a.mean()),q75=float(a.quantile(.75)),
                q90=float(a.quantile(.9)),max=float(a.max()))


def timing_subset(rows):
    n=len(rows)
    return dict(n=n,time_to_ready=metric([r['gap'] for r in rows]),
        processing_fraction=metric([r['fraction'] for r in rows]),
        admissible_tick_rank_fraction=metric([r['rank_fraction'] for r in rows]),
        selected_is_last_eligible_fraction=sum(r['remaining']==0 for r in rows)/n if n else None,
        selected_within_last_two_eligible_fraction=sum(r['remaining']<=1 for r in rows)/n if n else None)


def summarize(rows):
    n=len(rows)
    if not n:
        return dict(n=0)
    count=Counter(r['sample'] for r in rows)
    phases={'early_processing_0to25pct':lambda x:x<.25,'middle_processing_25to75pct':lambda x:.25<=x<.75,
            'late_processing_75to100pct':lambda x:x>=.75,
            'first_processing_10pct':lambda x:x<.1,'last_processing_10pct':lambda x:x>=.9}
    phase_coverage={name:dict(rows=sum(test(r['fraction']) for r in rows),
        fraction=sum(test(r['fraction']) for r in rows)/n,
        unique_requests=len({r['sample'] for r in rows if test(r['fraction'])}),
        requests_with_eligible_candidate_in_phase=len({r['sample'] for r in rows if r['available_phases'][name]>0}),
        uniform_candidate_expected_fraction=sum(r['available_phases'][name] for r in rows)/n) for name,test in phases.items()}
    any_true=sum(r['any_finish'] for r in rows)
    actual_true=sum(r['finish'] for r in rows)
    return dict(n=n,unique_requests=len(count),request_presentations=metric(list(count.values())),
        time_to_ready=metric([r['gap'] for r in rows]),processing_fraction=metric([r['fraction'] for r in rows]),
        admissible_tick_rank_fraction=metric([r['rank_fraction'] for r in rows]),
        remaining_eligible_ticks=metric([r['remaining'] for r in rows]),
        selected_finish_intent_count=actual_true,selected_finish_intent_fraction=actual_true/n,
        midpoint_finish_intent_count=sum(r['mid_finish'] for r in rows),
        midpoint_finish_intent_fraction=sum(r['mid_finish'] for r in rows)/n,
        any_eligible_finish_intent_count=any_true,any_eligible_finish_intent_fraction=any_true/n,
        selected_FINISH_given_any_eligible_FINISH=actual_true/any_true if any_true else None,
        all_eligible_no_FINISH_count=n-any_true,all_eligible_no_FINISH_fraction=(n-any_true)/n,
        selected_margin=metric([r['margin'] for r in rows]),midpoint_margin=metric([r['mid_margin'] for r in rows]),
        selected_is_last_eligible_fraction=sum(r['remaining']==0 for r in rows)/n,
        selected_within_last_two_eligible_fraction=sum(r['remaining']<=1 for r in rows)/n,
        selected_time_to_ready_lt_one_tick_fraction=sum(r['gap']<.25 for r in rows)/n,
        selected_equals_original_midpoint_fraction=sum(r['tick']==r['mid_tick'] for r in rows)/n,
        timing_by_intent=dict(
            selected_FINISH_gt_nonFINISH=timing_subset([r for r in rows if r['margin']>0]),
            selected_FINISH_le_nonFINISH=timing_subset([r for r in rows if r['margin']<=0]),
            entire_eligible_window_no_FINISH=timing_subset([r for r in rows if not r['any_finish']])),
        phase_coverage=phase_coverage)


def run(partial=False):
    torch.set_num_threads(1)
    protocol=json.loads((bank.OUT/'protocol.json').read_text(encoding='utf-8'))
    train_protocol=json.loads((TRAIN/'protocol.json').read_text(encoding='utf-8'))
    assert old.sha(bank.OUT/'protocol.json')==train_protocol['bank_protocol_sha256']
    all_rows=defaultdict(list)
    seeds={}
    log_hashes={}
    paired_ids={}
    eligible_phase_coverage={}
    for seed in range(3):
        data=bank.load(seed)
        folder=bank.OUT/f'seed_{seed:03d}'
        assert old.sha(folder/'bank.pt')==protocol['seeds'][str(seed)]['files'][str((folder/'bank.pt').relative_to(ROOT))]
        phase_predicates={'early_processing_0to25pct':lambda x:x<.25,
            'middle_processing_25to75pct':lambda x:.25<=x<.75,'late_processing_75to100pct':lambda x:x>=.75,
            'first_processing_10pct':lambda x:x<.1,'last_processing_10pct':lambda x:x>=.9}
        available_phases={}
        for sample in range(256):
            slot=int(data['target_slot'][sample]);m=data['eligible_mask'][:,sample,slot]
            fractions=data['eligible_processing_fraction'][:,sample,slot][m].tolist()
            available_phases[sample]={name:sum(test(f) for f in fractions)/len(fractions)
                for name,test in phase_predicates.items()}
        eligible_phase_coverage[str(seed)]={name:dict(
            requests_with_candidate=sum(v[name]>0 for v in available_phases.values()),
            request_equal_uniform_expected_fraction=sum(v[name] for v in available_phases.values())/256)
            for name in phase_predicates}
        arms={}
        for arm in ARMS:
            path=TRAIN/f'seed_{seed:03d}'/arm/'selections.jsonl'
            if not path.exists():
                if partial:continue
                raise FileNotFoundError(path)
            raw=path.read_text(encoding='utf-8')
            lines=raw.splitlines()
            # An in-progress write can be ignored only for an explicitly partial preview.
            if partial and raw and not raw.endswith('\n'):
                lines=lines[:-1]
            records=[json.loads(line) for line in lines if line]
            if not partial:
                assert len(records)==256,(seed,arm,len(records))
            assert [r['index'] for r in records]==list(range(len(records)))
            log_hashes[str(path.relative_to(ROOT))]=old.sha(path)
            rows=[]
            for record in records:
                assert len(record['selected'])==64 and len(set(record['selected']))==64
                key=(seed,record['index'])
                identity=(record['candidates'],record['selected'])
                if key in paired_ids:assert paired_ids[key]==identity,(key,arm,'unpaired requests')
                paired_ids[key]=identity
                assert record['gradient_rows']==192 and record['gradient_WAIT_rows']==64 and record['denominator']==144
                ids=torch.tensor(record['selected'],dtype=torch.long)
                ticks=torch.tensor(record['selected_ticks'],dtype=torch.long)
                slots=data['target_slot'][ids]
                assert bool(data['eligible_mask'][ticks,ids,slots].all())
                assert record['selected_time_to_ready']==data['eligible_time_to_ready'][ticks,ids,slots].tolist()
                assert record['selected_processing_fraction']==data['eligible_processing_fraction'][ticks,ids,slots].tolist()
                assert record['midpoint_ticks']==data['positions'][ids,1].tolist()
                assert record['eligible_counts']==data['eligible_mask'][:,ids].sum((0,2)).tolist()
                assert record['selected_finish_intent']==[a>=3 for a in record['selected_actions']]
                assert record['selected_finish_intent']==[v>0 for v in record['selected_margins']]
                assert record['selected_finish_intent_count']==sum(record['selected_finish_intent'])
                assert record['window_no_finish_count']==sum(not v for v in record['window_any_finish'])
                for j,sample in enumerate(record['selected']):
                    tick=int(ticks[j]);slot=int(slots[j])
                    allowed=data['eligible_mask'][:,sample,slot].nonzero().flatten().tolist()
                    rank=allowed.index(tick)
                    if arm=='hard':
                        assert tick==record['hard_ticks'][j]
                        assert record['selected_margins'][j]==record['maximum_margins'][j]
                        assert record['selected_finish_intent'][j]==record['window_any_finish'][j]
                    if arm=='midpoint':assert tick==record['midpoint_ticks'][j]
                    if arm=='random':assert tick==record['random_ticks'][j]
                    rows.append(dict(seed=seed,arm=arm,index=record['index'],round=record['round'],
                        sample=sample,tick=tick,mid_tick=record['midpoint_ticks'][j],
                        fraction=record['selected_processing_fraction'][j],gap=record['selected_time_to_ready'][j],
                        rank_fraction=rank/(len(allowed)-1) if len(allowed)>1 else 0.,remaining=len(allowed)-1-rank,
                        finish=record['selected_finish_intent'][j],mid_finish=record['midpoint_finish_intent'][j],
                        any_finish=record['window_any_finish'][j],margin=record['selected_margins'][j],
                        mid_margin=record['midpoint_margins'][j],available_phases=available_phases[sample]))
            all_rows[arm].extend(rows)
            early=[r for r in rows if r['index']<32]
            late=[r for r in rows if r['index']>=224]
            arms[arm]=dict(actual_logged_review_updates=len(records),all=summarize(rows),
                first_round_updates_0to31=summarize(early),last_round_updates_224to255=summarize(late),
                by_round={str(ri):summarize([r for r in rows if r['index']//32==ri]) for ri in range(8)},
                by_request=[dict(sample=i,id=data['ids'][i],
                    all=summarize([r for r in rows if r['sample']==i]),
                    early=summarize([r for r in early if r['sample']==i]),
                    late=summarize([r for r in late if r['sample']==i])) for i in range(256)])
        seeds[str(seed)]=arms
    pooled={}
    for arm,rows in all_rows.items():
        # Seed-scoped identities keep pooled unique-request coverage meaningful.
        pooled_rows=[dict(r,sample=r['seed']*256+r['sample']) for r in rows]
        pooled[arm]=dict(all=summarize(pooled_rows),
            first_round_updates_0to31=summarize([r for r in pooled_rows if r['index']<32]),
            last_round_updates_224to255=summarize([r for r in pooled_rows if r['index']>=224]))
    result=dict(format='wait-position-actual-selection-mechanism-v1',partial=partial,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source='Retained trainer selections.jsonl only. No model rescore, counterfactual choice or new request selection.',
        definitions=dict(early_training='review indices0..31, original first B round',late_training='indices224..255, eighth B round',
            early_processing='elapsed accepted-BEGIN to ready fraction<0.25',late_processing='fraction>=0.75',
            true_finish_intent='Actual logged full11-action argmax>=3 at the chosen WAIT tick, before that gradient update',
            eligible_rank='zero-based selected index among admissible ticks / (number admissible -1)',
            physical_distance='ready_time - observation now, from frozen native-world replay',
            time_unit='Native world physical time; archived source dt=0.25'),
        hashes=dict(bank_protocol=old.sha(bank.OUT/'protocol.json'),training_protocol=old.sha(TRAIN/'protocol.json'),
                    logs=log_hashes,implementation=old.sha(__file__)),
        actual_selected_ids_paired_across_arms=True,all_logged_selected_ticks_eligible=True,
        eligible_bank_phase_coverage=eligible_phase_coverage,seeds=seeds,pooled=pooled)
    if not partial:
        old.write(bank.OUT/'mechanism_diagnostics.json',result)
    small={arm:{phase:{k:value for k,value in result['pooled'][arm][phase].items()
        if k in ('n','time_to_ready','processing_fraction','selected_finish_intent_fraction','any_eligible_finish_intent_fraction',
                 'selected_is_last_eligible_fraction','phase_coverage')}
        for phase in ('all','first_round_updates_0to31','last_round_updates_224to255')}
        for arm in result['pooled']}
    print(json.dumps(dict(partial=partial,pooled=small)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    run(args.partial)
