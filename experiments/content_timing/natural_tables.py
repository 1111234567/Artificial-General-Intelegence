"""Compact deterministic tables of the predeclared natural trajectory measures.

Reads the original per-request table only; does not run a model or mutate any
frozen protocol/source. All subsets and pairings follow natural-v1/protocol.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from experiments.content_timing.natural import OUT, STAGES, checked, write_csv


def metric(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def main():
    plan = checked()
    rows = []
    for seed in range(3):
        rows.extend(json.loads(line) for line in (OUT/f'seed_{seed:03d}/requests.jsonl').read_text(encoding='utf-8').splitlines())
    common, transitions, geometry, statuses, marginals = [], [], [], [], []
    for seed in range(3):
        for condition in ('short', 'long_both'):
            selected = [r for r in rows if r['seed'] == seed and r['condition'] == condition and r['matured']]
            complete = [r for r in selected if all(r[p] is not None for p in STAGES)]
            identity = dict(seed=seed, condition=condition)
            for label, key, classes in (('exact_finish', 'target', 8), ('version', 'version', 2),
                                         ('mapped_payload', 'mapped_payload', 4), ('raw_payload', 'raw_payload', 4)):
                counts = Counter(r[key] for r in complete)
                marginals.append(identity | dict(population='complete_three_stage', label=label,
                    total=len(complete), class_counts=json.dumps([counts[k] for k in range(classes)]),
                    majority_count=max(counts.values()) if counts else 0,
                    majority_proportion=max(counts.values())/len(complete) if complete else None))
            statuses.append(identity | dict(matured=len(selected), complete_three_stage=len(complete),
                never_began=sum(r['accepted_begin_time'] is None for r in selected),
                began_without_legal=sum(r['accepted_begin_time'] is not None and r['first_legal'] is None for r in selected),
                began_too_late_for_deadline=sum(r.get('begin_too_late_for_deadline', False) for r in selected),
                no_processing=sum(r['processing_mid'] is None for r in selected),
                no_cue=sum(r['cue'] is None for r in selected),
                ordinary_completed=sum(r['completed'] for r in selected),
                strict_completed=sum(r['clean_completed'] for r in selected)))
            for phase in STAGES:
                samples = [r[phase] for r in complete]
                count = len(samples)
                common.append(identity | dict(phase=phase, same_requests=count,
                    exact_correct=sum(p['correct'] for p in samples),
                    exact_accuracy=sum(p['correct'] for p in samples)/count if count else None,
                    version_correct=sum(p['version_correct'] for p in samples),
                    version_accuracy=sum(p['version_correct'] for p in samples)/count if count else None,
                    payload_correct=sum(p['payload_correct'] for p in samples),
                    payload_accuracy=sum(p['payload_correct'] for p in samples)/count if count else None,
                    mean_correct_rank=statistics.mean(p['correct_rank'] for p in samples) if count else None,
                    mean_margin=statistics.mean(p['margin'] for p in samples) if count else None,
                    actual_wait=sum(p['action'] == 0 for p in samples),
                    actual_begin=sum(p['action'] in (1, 2) for p in samples),
                    actual_finish=sum(p['action'] >= 3 for p in samples),
                    actual_correct_finish=sum(p['action'] == r['target']+3 for p, r in zip(samples, complete)),
                    samples_with_prior_finish=sum(p['prior_request_actions'].get('finish', 0) > 0 for p in samples),
                    samples_with_prior_rejection=sum(p['prior_request_actions'].get('rejected', 0) > 0 for p in samples),
                    samples_with_prior_begin=sum(p['prior_request_actions'].get('begin', 0) > 0 for p in samples)))
            for early, late in (('cue', 'processing_mid'), ('cue', 'first_legal'), ('processing_mid', 'first_legal')):
                pair = [r for r in selected if r[early] is not None and r[late] is not None]
                for measure in ('correct', 'version_correct', 'payload_correct'):
                    both = sum(r[early][measure] and r[late][measure] for r in pair)
                    lost = sum(r[early][measure] and not r[late][measure] for r in pair)
                    gained = sum(not r[early][measure] and r[late][measure] for r in pair)
                    neither = sum(not r[early][measure] and not r[late][measure] for r in pair)
                    transitions.append(identity | dict(early=early, late=late, measure=measure,
                        same_requests=len(pair), both_correct=both, early_correct_late_wrong=lost,
                        early_wrong_late_correct=gained, both_wrong=neither,
                        early_correct=both+lost, late_correct=both+gained))
            started = [r for r in selected if r['accepted_begin_time'] is not None]
            for name, values in (
                ('begin_delay_from_issue', [r['accepted_begin_time']-r['issued_time'] for r in started]),
                ('processing_mid_fraction', [r['processing_mid']['processing_fraction'] for r in selected if r['processing_mid'] is not None]),
                ('processing_mid_distance', [r['processing_mid']['distance_from_processing_mid'] for r in selected if r['processing_mid'] is not None]),
                ('first_legal_processing_fraction', [r['first_legal']['processing_fraction'] for r in selected if r['first_legal'] is not None])):
                geometry.append(identity | dict(measure=name) | metric(values))
    for name, values in (('common_three_stage.csv', common), ('paired_transitions.csv', transitions),
                         ('processing_geometry.csv', geometry), ('request_status.csv', statuses),
                         ('label_marginals.csv', marginals)):
        write_csv(OUT/name, values)
    (OUT/'compact_tables.json').write_text(json.dumps(dict(protocol_hash=plan['protocol_hash'],
        common_three_stage=common, transitions=transitions, geometry=geometry, request_status=statuses), indent=2), encoding='utf-8')
    print(json.dumps(dict(common_three_stage=common, request_status=statuses), indent=2))


if __name__ == '__main__':
    main()
