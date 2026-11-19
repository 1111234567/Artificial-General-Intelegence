"""Audit paired retirement outcomes without importing frozen report machinery."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics

from experiments.continuity.run import state_digest
from experiments.undirected.run import _atomic_json, file_sha256
from experiments.coverage_exit.retirement_study import read_plan, ARMS


def checked_original_result(value):
    """Verify B's retained pre-JSON digest through its explicit typed codec.

    B computed its hash before JSON serialization. The native evaluator returns
    integer phase-code keys and three Regime tuples, whereas JSON has string
    keys and lists. Restore only those known schema types, without changing any
    content, and require the EXACT original digest. Original files are retained.
    """
    typed=deepcopy(value)
    expected=typed.pop('result_hash')
    for record in typed['endpoint_records']:
        record['phase_codes']={int(key):item for key,item in record['phase_codes'].items()}
        for key in ('duration_range','gap_range','window_range'):
            record['regime'][key]=tuple(record['regime'][key])
    if state_digest(typed)!=expected:
        raise ValueError('original pre-JSON result digest mismatch')
    return value


def condition_rates(records, condition):
    rows = [r for r in records if r['condition'] == condition]
    if len(rows) != 4 or sorted(r['probe']['family'] for r in rows) != list(range(4)):
        raise ValueError('four-family evaluation missing')
    values = {'strict': [], 'ordinary': [], 'idle_per1000': []}
    for row in rows:
        p, o = row['primary'], row['opportunities']
        if p['matured'] <= 0 or o['inactive_opportunities'] <= 0:
            raise ValueError('undefined primary or idle denominator')
        values['strict'].append(p['clean_completed']/p['matured'])
        values['ordinary'].append(p['completed']/p['matured'])
        values['idle_per1000'].append(1000*o['inactive_actions']/o['inactive_opportunities'])
    return {key: statistics.mean(value) for key, value in values.items()}


def audit(path):
    plan = read_plan(path)
    root = Path(plan['output'])
    results, bindings = {}, []
    rows, differences = [], []
    for seed in plan['seeds']:
        pair = {}
        for arm in ARMS:
            result_path = root/f'seed_{seed:03d}'/arm/'result.json'
            value = json.loads(result_path.read_text())
            checked_original_result(value)
            if (value['protocol_hash'] != plan['protocol_hash'] or value['status'] != 'success'
                    or value['seed'] != seed or value['arm'] != arm
                    or value['endpoint_tick'] != plan['endpoint_tick']
                    or file_sha256(value['checkpoint']) != value['checkpoint_sha256']):
                raise ValueError('result boundary/protocol mismatch')
            if not value['checkpoint_roundtrip_verified'] or not value['source_unchanged']:
                raise ValueError('unchecked source/restore')
            events, op, work = value['events'], value['operator'], value['work_delta']
            if (len(events) != 15 or op['probe_calls'] != 60 or op['cumulative_births'] != 0
                    or work['optimizer_steps'] != 128 or work['structural_probe_calls'] != 60
                    or len(value['endpoint_records']) != 8):
                raise ValueError('unmatched scheduled budget')
            for event in events:
                full = event['full_ablation_loss']['mixed'] < event['fade_baseline']['mixed']
                if event['full_improves'] != full or event['full_accepted'] != (arm == 'full_eligible' and full):
                    raise ValueError('full acceptance semantics changed')
                if event['birth_accepted'] or event['births'] or event['probe_calls'] != 4:
                    raise ValueError('retire-only/four-probe flags changed')
            row = dict(seed=seed, arm=arm, structural=dict(
                eligible_pairs=op['eligible_pairs'], nonzero_pairs=op['nonzero_pairs'],
                cumulative_births=op['cumulative_births'], cumulative_deaths=op['cumulative_deaths'],
                full_improvement_batches=sum(e['full_improves'] for e in events),
                accepted_full_batches=sum(e['full_accepted'] for e in events),
                accepted_half_batches=sum(e['fade_accepted'] and not e['full_accepted'] for e in events),
                full_mixed_damage_min=min(e['full_ablation_damage']['mixed'] for e in events),
                full_mixed_damage_mean=statistics.mean(e['full_ablation_damage']['mixed'] for e in events),
                full_mixed_damage_max=max(e['full_ablation_damage']['mixed'] for e in events),
                half_improvement_batches=sum(e['fade_improves'] for e in events),
                accepted_batches_history_damage=sum((e['full_ablation_damage'] if e['full_accepted'] else e['half_fade_damage'])['history'] > 0 for e in events if e['fade_accepted']),
                committed_mixed_damage_sum=sum((e['full_ablation_damage'] if e['full_accepted'] else e['half_fade_damage'])['mixed'] for e in events if e['fade_accepted'])),
                baseline={c: condition_rates(value['baseline_records'], c) for c in ('short','long_both')},
                endpoint={c: condition_rates(value['endpoint_records'], c) for c in ('short','long_both')},
                per_family_counts=[dict(condition=r['condition'], family=r['probe']['family'],
                    matured=r['primary']['matured'], completed=r['primary']['completed'], clean_completed=r['primary']['clean_completed'],
                    inactive_actions=r['opportunities']['inactive_actions'], inactive_opportunities=r['opportunities']['inactive_opportunities'])
                    for r in value['endpoint_records']])
            rows.append(row)
            pair[arm], results[(seed, arm)] = value, row
            bindings.append(dict(path=str(result_path),sha256=file_sha256(result_path),result_hash=value['result_hash']))
        if pair['half_only']['work_delta'] != pair['full_eligible']['work_delta']:
            raise ValueError('paired computation work differs')
        first_a, first_b = (pair[a]['events'][0] for a in ARMS)
        for key in ('fade_indices','birth_indices','baseline','birth_loss','full_ablation_loss','half_fade_loss'):
            if first_a[key] != first_b[key]:
                raise ValueError('first switch did not start from identical proposal/trials')
        for condition in ('short','long_both'):
            for family in range(4):
                a, b = [next(r for r in pair[arm]['endpoint_records'] if r['condition']==condition and r['probe']['family']==family) for arm in ARMS]
                for key in ('case_ids','actor_ids','fork_tick','start_scheduler_hash','end_scheduler_hash','start_event_hash','end_event_hash'):
                    if a[key] != b[key]: raise ValueError(f'paired external stream differs: {seed}/{condition}/{family}/{key}')
                for cohort in ('primary','carryover_contract'):
                    if a[cohort]['request_fingerprint'] != b[cohort]['request_fingerprint']:
                        raise ValueError('paired request identities/timing differ')
            delta = {key: results[(seed,'full_eligible')]['endpoint'][condition][key]-results[(seed,'half_only')]['endpoint'][condition][key]
                     for key in ('strict','ordinary','idle_per1000')}
            differences.append(dict(seed=seed,condition=condition,**delta))
    aggregate = {}
    for condition in ('short','long_both'):
        chosen = [r for r in differences if r['condition']==condition]
        aggregate[condition] = {key: dict(mean=statistics.mean(r[key] for r in chosen), sample_sd=statistics.stdev(r[key] for r in chosen))
                                for key in ('strict','ordinary','idle_per1000')}
    report = dict(format='coverage-exit-retirement-audited-v1',audit=dict(status='passed',checks=[
        'source/protocol/result/checkpoint SHA bindings','complete saved checkpoint roundtrip in each worker',
        '128 updates,15 executed windows,60 probes per branch and equal paired work deltas',
        'first proposal/trials identical before acceptance switch',
        'all8 endpoint paired forks share actor/case identities, complete scheduler/event streams and matured request fingerprints',
        'strict mixed-loss full-commit flags checked from recorded loss components']),
        protocol=str(Path(path).resolve()),protocol_hash=plan['protocol_hash'],results=rows,
        paired_full_minus_half=differences,aggregate_paired=aggregate,result_bindings=bindings,
        limits=plan['limits'],primary_scope=plan['evaluation']['primary'],
        digest_codec='Initial direct JSON digest check failed. Original B files/hash retained; exact original hash verified after restoring only native endpoint phase_codes integer keys and three Regime tuples.',
        analysis_source=dict(path=str(Path(__file__).resolve()),sha256=file_sha256(__file__)))
    report['report_hash']=state_digest(report)
    _atomic_json(root/'report.json',report)
    lines=['# Retirement switch: measured paired results','',
           'CPU float64, original core/shared/recorded512/fixed at tick131072, 128 continuing updates. Each endpoint evaluation retains its own world and h. Three independent seeds; 8 free-action forks per branch.','',
           '| seed | arm | deaths | eligible | full batches | half batches | short strict % | long strict % | short idle/1000 | long idle/1000 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        s,e=row['structural'],row['endpoint']
        lines.append(f"| {row['seed']} | {row['arm']} | {s['cumulative_deaths']} | {s['eligible_pairs']} | {s['accepted_full_batches']} | {s['accepted_half_batches']} | {100*e['short']['strict']:.4f} | {100*e['long_both']['strict']:.4f} | {e['short']['idle_per1000']:.4f} | {e['long_both']['idle_per1000']:.4f} |")
    lines+=['','Full-minus-half paired differences (mean ± sample SD across three seeds):','']
    for c,a in aggregate.items():
        lines.append(f"- {c}: strict {100*a['strict']['mean']:+.4f} ± {100*a['strict']['sample_sd']:.4f} percentage points; ordinary {100*a['ordinary']['mean']:+.4f} ± {100*a['ordinary']['sample_sd']:.4f} points; idle {a['idle_per1000']['mean']:+.4f} ± {a['idle_per1000']['sample_sd']:.4f} per1000.")
    lines+=['','Every executed window retains its baseline/full/half live/history/mixed loss components in the seed/arm result.json. Source baseline and all endpoint per-family counts are retained in report.json.','',*plan['limits']]
    (root/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args()
    result=audit(args.protocol)
    print(json.dumps(dict(audit=result['audit'],aggregate_paired=result['aggregate_paired']),indent=2))
