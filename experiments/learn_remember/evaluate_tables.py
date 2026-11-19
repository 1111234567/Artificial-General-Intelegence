"""Export seed-preserving counts, averages and paired retention differences."""
from __future__ import annotations

from collections import Counter
import csv
import json
import statistics

from experiments.learn_remember import evaluate_gru as e


def mean_sd(values):
    return dict(mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def export():
    e.checked()
    records = [e.read(path) for path in sorted(e.OUT.glob('seed_*/*_family_?.json'))]
    records = [row for row in records if row['source'] == 'common']
    scopes = {'family0': (0,), 'family1': (1,), 'family2': (2,), 'withdrawn_family3': (3,),
              'remaining_families012': (0, 1, 2), 'all_families': (0, 1, 2, 3)}
    rows, flat = [], []
    metrics = ('ordinary_completion_rate', 'strict_completion_rate', 'idle_actions_per1000')
    for stage in e.STAGES:
        for condition in ('short', 'long_both'):
            for scope, families in scopes.items():
                for raw_group in ('all', 'unique_raw_1_3'):
                    per_seed = []
                    for seed in range(3):
                        selected = [row for row in records if row['seed'] == seed and row['stage'] == stage
                                    and row['condition'] == condition and row['family'] in families]
                        if len(selected) != len(families):
                            continue
                        counts = Counter()
                        for record in selected:
                            values = record['counts'] if raw_group == 'all' else record[raw_group]
                            counts.update({key: value for key, value in values.items() if isinstance(value, int)})
                        per_seed.append(dict(seed=seed, counts=dict(counts), **e.rates(counts)))
                    if not per_seed:
                        continue
                    row = dict(stage=stage, condition=condition, scope=scope, raw_group=raw_group,
                               n_seeds=len(per_seed), per_seed=per_seed)
                    for metric in metrics:
                        row[metric] = mean_sd([value[metric] for value in per_seed if value[metric] is not None])
                    rows.append(row)
                    for seed in per_seed:
                        flat.append(dict(stage=stage, condition=condition, scope=scope, raw_group=raw_group,
                                         seed=seed['seed'], **seed['counts'], **{k:seed[k] for k in metrics}))
    differences = []
    for before, after in [('random', 'wide'), ('wide', 'withdrawal_mid'), ('wide', 'withdrawal'),
                          ('withdrawal_mid', 'review_mid'), ('withdrawal', 'review')]:
        for row in rows:
            if row['stage'] != after:
                continue
            parent = next((r for r in rows if r['stage'] == before and all(
                r[key] == row[key] for key in ('condition', 'scope', 'raw_group'))), None)
            if parent is None:
                continue
            diff = dict(before=before, after=after, **{k: row[k] for k in ('condition', 'scope', 'raw_group')})
            for metric in metrics:
                a = {r['seed']: r[metric] for r in parent['per_seed']}
                b = {r['seed']: r[metric] for r in row['per_seed']}
                values = [dict(seed=seed, delta=b[seed] - a[seed]) for seed in sorted(a.keys() & b.keys())
                          if a[seed] is not None and b[seed] is not None]
                diff[metric] = dict(per_seed=values, **mean_sd([v['delta'] for v in values]))
            differences.append(diff)
    result = dict(format='learn-remember-gru-tables-v1', forks=len(records), rows=rows, paired_differences=differences)
    e.write(e.OUT / 'tables.json', result)
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (e.OUT / 'seed-counts.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(dict(forks=len(records), rows=len(rows), seed_rows=len(flat))))
    return result


if __name__ == '__main__':
    export()
