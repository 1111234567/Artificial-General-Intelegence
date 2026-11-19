"""Frozen independent behavior evidence for equal-budget old-review selection.

The long-B and randomly interleaved A/B worlds, scorer and public warmup are
reused byte-for-byte from the earlier transfer evaluation. Existing no-review
results are bound, not rerun or rewritten. Missing intermediate mixed branches
are added here. This random-per-NEW mixed stream is not strict alternation.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.experience_transfer import eval_transfer as base

OUT = ROOT / 'data/review-selection/eval-v1'
TRAIN = ROOT / 'data/review-selection/train-v1'
PLAN = OUT / 'protocol.json'
FORMAT = 'review-selection-autonomous-eval-v1'
ARMS = ('uniform', 'conflict', 'error', 'no_review')
SNAPSHOTS = base.SNAPSHOTS
MIXED = ('B000', 'B001', 'B002', 'B004', 'B008')
sha, read, write, state_digest = base.sha, base.read, base.write, base.state_digest


def scenarios(snapshot):
    return ('B',) + (('mixed',) if snapshot in MIXED else ()) + (('history',) if snapshot in ('B000', 'B008') else ())


def result_path(seed, arm, snapshot, scenario):
    return OUT / f'seed_{seed:03d}' / f'{arm}_{snapshot}_{scenario}.json'


def checkpoint_path(seed, arm, snapshot):
    folder = base.TRAIN / f'seed_{seed:03d}' / 'related' if arm == 'no_review' else TRAIN / f'seed_{seed:03d}' / arm
    return folder / f'{snapshot}.pt'


def actual_B_new(cp, origin, b_family):
    """External NEW identities, including the boundary observation with no loss."""
    def count(value):
        return int(value['world']['context']['tensors']['family_new_counts'][..., b_family].sum())
    return count(cp) - count(origin)


def family_counts(row, part, raw_group):
    families = [row['B_family']] if part == 'B' else [row['history_family']] if part == 'history' else [row['history_family'], row['B_family']]
    c = Counter()
    for family in families:
        c.update(row['strata'][str(family)][raw_group]['counts'])
    return dict(c)


def checked():
    plan = read(PLAN)
    assert plan['protocol_hash'] == state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'})
    base.checked()
    for path, expected in plan['frozen_files'].items():
        assert sha(path) == expected, ('frozen evaluation dependency changed', path)
    return plan


def prepare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    old = base.checked()
    files = [Path(__file__).resolve(), Path(__file__).with_name('eval_review_test.py'), base.PLAN]
    spec = {}
    for seed in range(3):
        ss = old['spec'][str(seed)]
        spec[str(seed)] = dict(mapping_table=ss['mapping_tables']['related'], history_family=ss['history_family'],
                               B_family=ss['B_family'], raw_groups=ss['raw_groups'])
        files += [base.OUT / f'common_seed_{seed:03d}_related.pt', checkpoint_path(seed, 'no_review', 'B000')]
        for scenario in ('B', 'mixed', 'history'):
            ref = base.reference(seed, 'related', scenario)
            files += [base.OUT / f'reference_seed_{seed:03d}_related_{scenario}.json', Path(ref['external_artifact'])]
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen before reading new review-selection autonomous outcomes. Every declared seed, condition and snapshot retained; no selected winner or threshold.',
        arms=list(ARMS), seeds=[0, 1, 2], snapshots=list(SNAPSHOTS), mixed_snapshots=list(MIXED),
        expected_records=168, steps=old['steps'], batch=old['batch'], spec=spec,
        base_protocol_hash=old['protocol_hash'], frozen_files={str(p): sha(p) for p in files},
        B='Independent long_both B-only free action at B0/first2/first8/B1/B2/B4/B8; native11-action argmax. Same public warmup, external stream and request identities for all conditions and snapshots.',
        mixed='One continuous world/hidden state; future NEW family sampled independently per job from A/B, possibly concurrent within actor. This is random interleaving, NOT strict A/B alternation. Evaluate B0/1/2/4/8. Report family A and B separately, including each of the two common conflicting raw symbols.',
        history='Auxiliary separate A-only long heldout at B0/B8; mixed A/B is the primary retention/context distinction test.',
        warmup=old['warmup'], metrics=old['metrics'], scorer=old['scorer'],
        baseline='Existing related no-review true evaluation results are hash-bound with original raw artifacts; missing B1/B2/B4 mixed results are evaluated without altering original files. All B0 condition models must exactly equal original related B000.',
        exposure_axis='Actual unique external B NEW commands = checkpoint world family_new_counts(B) minus original B000 counter. Also retain work.B_unique_new_commands; this excludes original no-loss boundary NEW and is not silently treated as actual exposure. Replay positions and gradients are not new cases.',
        analysis='Per-seed rates and equal-weight means/sample SD, no pooled-rate replacement. All/all-conflict/each-conflicting-raw per family. Compare B gains and retention to B0 and no_review at identical snapshots. Three seeds and one independent stream per seed limit generalization.',
        strict_control='Explicit same-slot/symbol paired alternation is a separate supplementary protocol, not a reinterpretation of this random mixed stream.')
    plan['protocol_hash'] = state_digest(plan)
    write(PLAN, plan)
    # Archive references locally before opening any new review checkpoint.
    for seed in range(3):
        for scenario in ('B', 'mixed', 'history'):
            ref = deepcopy(base.reference(seed, 'related', scenario))
            ref.update(review_protocol_hash=plan['protocol_hash'], source_protocol_hash=ref['protocol_hash'])
            write(OUT / f'reference_seed_{seed:03d}_{scenario}.json', ref)
    return plan


def bind_checkpoint(row, path, cp, seed, arm, snapshot, scenario, plan):
    ss = plan['spec'][str(seed)]
    origin_path = checkpoint_path(seed, 'no_review', 'B000')
    origin = torch.load(origin_path, map_location='cpu', weights_only=False)
    assert cp['mapping_table'] == ss['mapping_table']
    assert cp['transfer']['seed'] == seed
    assert cp['world']['spec']['seed'] == origin['world']['spec']['seed']
    if snapshot == 'B000':
        assert state_digest(cp['model']) == state_digest(origin['model'])
        assert state_digest(cp['world']) == state_digest(origin['world'])
        assert state_digest(cp['hidden']) == state_digest(origin['hidden'])
    assert row['model_hash'] == state_digest(cp['model'])
    row.update(format=FORMAT, protocol_hash=plan['protocol_hash'], seed=seed, arm=arm, snapshot=snapshot,
        B_round=0 if snapshot.startswith('Bfirst') else int(snapshot[1:]), scenario=scenario,
        checkpoint_path=str(path.resolve()), checkpoint_sha256=sha(path), checkpoint_work=cp['work'],
        checkpoint_transfer=cp.get('transfer'), checkpoint_review_selection=cp.get('review_selection'),
        actual_unique_B_NEW=actual_B_new(cp, origin, ss['B_family']),
        work_B_unique_new_commands=cp['work'].get('B_unique_new_commands'),
        mapping_table=ss['mapping_table'], history_family=ss['history_family'], B_family=ss['B_family'])
    return row


@torch.inference_mode()
def run_checkpoint(seed, arm, snapshot):
    plan = checked()
    path = checkpoint_path(seed, arm, snapshot)
    model, _, _, cp = base.training_api().load_checkpoint(path, device='cpu')
    model.eval()
    ss = plan['spec'][str(seed)]
    table, hf, bf = ss['mapping_table'], ss['history_family'], ss['B_family']
    shared = base.common(seed, table, hf, bf, arm='related')
    hidden = None
    for scenario in scenarios(snapshot):
        target = result_path(seed, arm, snapshot, scenario)
        if target.exists():
            continue
        source = base.OUT / f'seed_{seed:03d}' / f'related_{snapshot}_{scenario}.json'
        # Identical B0 model branches can reuse the exact original evaluation.
        if source.exists() and (arm == 'no_review' or snapshot == 'B000'):
            row = deepcopy(read(source))
            assert sha(row['checkpoint_path']) == row['checkpoint_sha256']
            assert row['model_hash'] == state_digest(cp['model'])
            row.update(evidence_stem=str(source.with_suffix('')), evidence_reused=True,
                       original_result_path=str(source), original_result_sha256=sha(source))
        else:
            if hidden is None:
                hidden = base.original.warmup(model, shared)
            families = {'B': (bf,), 'mixed': (hf, bf), 'history': (hf,)}[scenario]
            started = time.perf_counter()
            with base.training_api().mapping_scope(table):
                world = base.ContextWorld.from_state_dict(shared['world'], device='cpu')
                row, frames, states = base.rollout(model, world, hidden, families=families,
                    steps=plan['steps'], table=table, history_family=hf, b_family=bf, raw_groups=ss['raw_groups'])
            target.parent.mkdir(parents=True, exist_ok=True)
            stem = target.with_suffix('')
            requests = dict(matured=row.pop('primary_requests'), actions=row.pop('action_requests'))
            np.savez_compressed(stem.with_suffix('.npz'), **frames)
            torch.save(states, stem.with_suffix('.pt'))
            write(stem.with_suffix('.requests.json'), requests)
            row.update(evidence_stem=str(stem), evidence_reused=False,
                warmup_observation_hash=shared['observation_hash'], elapsed_seconds=time.perf_counter()-started,
                artifacts={str(stem.with_suffix(ext)): sha(stem.with_suffix(ext)) for ext in ('.npz', '.pt', '.requests.json')})
        expected = base.reference(seed, 'related', scenario)
        for key in ('external_digest', 'primary_identity_hash', 'start_scheduler_hash', 'end_scheduler_hash', 'start_event_hash', 'end_event_hash'):
            assert row[key] == expected[key], (seed, arm, snapshot, scenario, key)
        row = bind_checkpoint(row, path, cp, seed, arm, snapshot, scenario, plan)
        write(target, row)
        print(json.dumps(dict(event='review_eval', seed=seed, arm=arm, snapshot=snapshot, scenario=scenario,
                             actual_unique_B_NEW=row['actual_unique_B_NEW'], reused=row['evidence_reused'])), flush=True)


def available(seed=None, watch=False):
    torch.set_num_threads(1)
    seeds = [seed] if seed is not None else range(3)
    pending = [(s, a, t) for s in seeds for a in ARMS for t in SNAPSHOTS]
    while pending:
        progressed = False
        for s, a, t in list(pending):
            if all(result_path(s, a, t, q).exists() for q in scenarios(t)):
                pending.remove((s, a, t))
                continue
            path = checkpoint_path(s, a, t)
            if path.exists() and path.with_suffix('.json').exists():
                run_checkpoint(s, a, t)
                pending.remove((s, a, t))
                progressed = True
        if not watch or not pending:
            break
        if not progressed:
            time.sleep(5)
    return dict(pending=pending, complete=not pending)


def records():
    return [read(p) for p in sorted(OUT.glob('seed_*/*.json')) if not p.name.endswith('.requests.json')]


def audit(require_complete=False):
    plan = checked()
    rows = records()
    if require_complete:
        assert len(rows) == plan['expected_records']
    matched = {}
    exposures = {}
    audited_evidence = {}
    for row in rows:
        assert row['protocol_hash'] == plan['protocol_hash'] and row['format'] == FORMAT
        path = Path(row['checkpoint_path'])
        assert sha(path) == row['checkpoint_sha256']
        cp = torch.load(path, map_location='cpu', weights_only=False)
        rebound = bind_checkpoint(deepcopy(row), path, cp, row['seed'], row['arm'], row['snapshot'], row['scenario'], plan)
        for key in ('checkpoint_work', 'actual_unique_B_NEW', 'work_B_unique_new_commands', 'checkpoint_review_selection'):
            assert rebound[key] == row[key], key
        if row['evidence_reused']:
            assert sha(row['original_result_path']) == row['original_result_sha256']
            source = read(row['original_result_path'])
            for key in ('model_hash', 'counts', 'metrics', 'strata', 'by_family_raw', 'primary_identity_hash', 'artifacts'):
                assert row[key] == source[key], ('bound old result changed', key)
        for artifact, expected in row['artifacts'].items():
            assert sha(artifact) == expected
        stem = Path(row['evidence_stem'])
        if str(stem) not in audited_evidence:
            states = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=False)
            with np.load(stem.with_suffix('.npz')) as saved:
                frames = {k: saved[k] for k in saved.files}
            assert np.array_equal(frames['actions'], frames['logits'].argmax(-1))
            for which in ('initial', 'end'):
                prefix = 'start' if which == 'initial' else 'end'
                assert state_digest(states[which+'_world']) == row[prefix+'_world_hash']
                assert state_digest(states[which+'_h']) == row[prefix+'_h_hash']
            rebuilt = base.score(frames, states['initial_world'], table=row['mapping_table'], history_family=row['history_family'],
                b_family=row['B_family'], fork_tick=row['fork_tick'], regime_id=row['evaluation_regime_id'], raw_groups=row['raw_groups'])
            for key in ('metrics', 'strata', 'by_family_raw', 'primary_identity_hash'):
                assert rebuilt[key] == row[key], (stem, key)
            for key, value in rebuilt['counts'].items():
                assert row['counts'][key] == value
            requests = read(stem.with_suffix('.requests.json'))
            assert rebuilt['primary_requests'] == requests['matured']
            assert rebuilt['action_requests'] == requests['actions']
            ref = base.reference(row['seed'], 'related', row['scenario'])
            with np.load(ref['external_artifact']) as external:
                for key in base.EXTERNAL_FIELDS:
                    assert np.array_equal(frames[key], external[key]), (stem, key)
            audited_evidence[str(stem)] = True
        key = (row['seed'], row['scenario'])
        signature = tuple(row[k] for k in ('external_digest', 'primary_identity_hash', 'start_scheduler_hash', 'end_scheduler_hash', 'start_event_hash', 'end_event_hash', 'warmup_observation_hash'))
        if key in matched:
            assert matched[key] == signature
        matched[key] = signature
        key = row['seed'], row['snapshot']
        exposure = row['actual_unique_B_NEW']
        if key in exposures:
            assert exposures[key] == exposure, ('unequal actual B NEW', key)
        exposures[key] = exposure
    value = dict(format=FORMAT, verified=True, complete=len(rows) == plan['expected_records'], records=len(rows),
        unique_evidence_forks=len(audited_evidence), matched_external_groups=len(matched), protocol_hash=plan['protocol_hash'],
        checks=['frozen dependency/common/reference hashes', 'original and new checkpoints model and real-world B0 binding',
            'native11-action argmax, no action gates', 'saved requests independently rebuilt against actual receipts',
            'ordinary/strict/idle raw denominators preserved per family and raw', 'same heldout external stream and identities across conditions/snapshots',
            'actual external B NEW exposures equal across all arms', 'original no-review evidence immutable and bound'])
    write(OUT / 'audit.json', value)
    return value


def describe(values):
    values = [x for x in values if x is not None]
    return dict(mean=statistics.mean(values) if values else None, sample_sd=statistics.stdev(values) if len(values)>1 else None)


def summarize():
    plan = checked()
    rs = records()
    flat = []
    for row in rs:
        groups = dict(row['raw_groups'], **{f'raw_{r}': [r] for r in row['raw_groups']['conflict']})
        for part in ('history', 'B'):
            family = row['history_family'] if part == 'history' else row['B_family']
            for group, raws in groups.items():
                if group.startswith('raw_'):
                    c = Counter()
                    for raw in raws:
                        c.update(row['by_family_raw'].get(f'{family}:{raw}', {}).get('counts', {}))
                    c = dict(c)
                else:
                    c = family_counts(row, part, group)
                if not c.get('matured'):
                    continue
                flat.append(dict(seed=row['seed'], arm=row['arm'], snapshot=row['snapshot'], scenario=row['scenario'],
                    family_part=part, raw_group=group, raws=raws, family=family, counts=c, metrics=base.rates(c),
                    actual_unique_B_NEW=row['actual_unique_B_NEW'], work_B_unique_new_commands=row['work_B_unique_new_commands'], work=row['checkpoint_work']))
    summaries = []
    # Raw_0/1/etc are seed-dependent, so aggregate common conflict position as well.
    for scenario in ('B', 'mixed', 'history'):
        for arm in ARMS:
            for snapshot in SNAPSHOTS:
                for part in ('history', 'B'):
                    for group in ('all', 'shared', 'conflict', 'conflict_first', 'conflict_second'):
                        per_seed = []
                        for seed in range(3):
                            g = group
                            if group.startswith('conflict_'):
                                r = plan['spec'][str(seed)]['raw_groups']['conflict'][0 if group.endswith('first') else 1]
                                g = f'raw_{r}'
                            per_seed += [x for x in flat if (x['seed'], x['arm'], x['snapshot'], x['scenario'], x['family_part'], x['raw_group']) == (seed, arm, snapshot, scenario, part, g)]
                        if not per_seed:
                            continue
                        summaries.append(dict(scenario=scenario, arm=arm, snapshot=snapshot, family_part=part, raw_group=group,
                            per_seed=per_seed, metrics={m: describe(x['metrics'][m] for x in per_seed) for m in base.rates({})},
                            actual_unique_B_NEW=describe(x['actual_unique_B_NEW'] for x in per_seed)))
    paired = []
    for item in summaries:
        if item['arm'] == 'no_review':
            continue
        baseline = next((x for x in summaries if all(x[k] == item[k] for k in ('scenario', 'snapshot', 'family_part', 'raw_group')) and x['arm'] == 'no_review'), None)
        initial = next((x for x in summaries if all(x[k] == item[k] for k in ('scenario', 'arm', 'family_part', 'raw_group')) and x['snapshot'] == 'B000'), None)
        if baseline is None or initial is None:
            continue
        d = {k: item[k] for k in ('scenario', 'arm', 'snapshot', 'family_part', 'raw_group')}
        d['metrics'] = {}
        for metric in base.rates({}):
            data = []
            for entry in item['per_seed']:
                b = next(x for x in baseline['per_seed'] if x['seed'] == entry['seed'])
                z = next(x for x in initial['per_seed'] if x['seed'] == entry['seed'])
                a, bb, zz = (x['metrics'][metric] for x in (entry, b, z))
                if None not in (a, bb, zz):
                    data.append(dict(seed=entry['seed'], review_minus_no_review=a-bb, change_from_B0=a-zz))
            d['metrics'][metric] = dict(per_seed=data, review_minus_no_review=describe(x['review_minus_no_review'] for x in data), change_from_B0=describe(x['change_from_B0'] for x in data))
        paired.append(d)
    value = dict(format=FORMAT, protocol_hash=plan['protocol_hash'], records=len(rs), complete=len(rs)==plan['expected_records'],
        summaries=summaries, paired=paired,
        note='Equal-weight means of seed rates; never pooled across A and B to decide retention. conflict_first/second index the two frozen per-seed conflicting raws; per-seed entries retain actual raw identity. Mixed contexts are random per NEW/job, not strict alternation.')
    write(OUT / 'summary.json', value)
    csv_rows = []
    for x in flat:
        csv_rows.append({k: v for k, v in x.items() if k not in ('metrics', 'counts', 'work', 'raws')} | {'raws': json.dumps(x['raws'])} | x['metrics'] | x['counts'] | {'work_'+k: v for k, v in x['work'].items()})
    if csv_rows:
        with (OUT/'curves.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(x.keys() for x in csv_rows))))
            writer.writeheader(); writer.writerows(csv_rows)
    return value


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'available', 'watch', 'audit', 'summarize'))
    p.add_argument('--seed', type=int, choices=range(3))
    p.add_argument('--complete', action='store_true')
    args = p.parse_args()
    if args.command == 'prepare':
        value = prepare(); print(json.dumps(dict(protocol_hash=value['protocol_hash'], expected_records=value['expected_records'])))
    elif args.command in ('available', 'watch'):
        print(json.dumps(available(args.seed, args.command=='watch')))
    elif args.command == 'audit':
        print(json.dumps(audit(args.complete)))
    else:
        value = summarize(); print(json.dumps(dict(records=value['records'], summaries=len(value['summaries']), complete=value['complete'])))


if __name__ == '__main__':
    main()
