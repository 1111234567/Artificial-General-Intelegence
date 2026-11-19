"""Write-isolated frozen FINISH-position autonomous evaluation wrapper.

The historical rollout, scorer and timing analyzer are imported unchanged.
Every write is confined to this experiment; existing evidence is immutable.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.wait_position import eval_position as old, eval_timing as timing
base = old.base
OUT = ROOT / 'data/finish-position/eval-v1'
TRAIN = ROOT / 'data/finish-position/train-v1'
LAST_TRAIN = ROOT / 'data/wait-position/last-v1'
LAST_EVAL = ROOT / 'data/wait-position/eval-last-v1'
MAIN_EVAL = ROOT / 'data/wait-position/eval-v1'
PLAN = OUT / 'protocol.json'
FORMAT = 'finish-position-autonomous-eval-v1'
ARMS = ('midpoint', 'first')
STAGES = tuple(base.SNAPSHOTS)
MIXED = ('B000', 'B001', 'B002', 'B004', 'B008')
sha, read, state_digest = base.sha, base.read, base.state_digest


def scenarios(stage):
    return ('B',) + (('mixed',) if stage in MIXED else ()) + (('history',) if stage in ('B000', 'B008') else ())


def expected_keys():
    return {(s, a, t, q) for s in range(3) for a in ARMS for t in STAGES for q in scenarios(t)}


def owned(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('refusing write outside FINISH evaluation: ' + str(path))
    return path


def write(path, value):
    path = owned(path)
    old.write(path, value)


@contextmanager
def source_write_guard():
    """No original evaluator can write to its historical output by accident."""
    saved = base.write, base.original.write, base.original.OUT, base.original.PLAN
    def forbidden(*args, **kwargs):
        raise RuntimeError('historical source evaluator write is forbidden')
    base.write = base.original.write = forbidden
    base.original.OUT = OUT / '_unused_original_output'
    base.original.PLAN = base.original.OUT / 'protocol.json'
    try:
        yield
    finally:
        base.write, base.original.write, base.original.OUT, base.original.PLAN = saved


def result_path(seed, arm, stage, scenario):
    return OUT / f'seed_{seed:03d}' / f'{arm}_{stage}_{scenario}.json'


def checkpoint_path(seed, arm, stage):
    return TRAIN / f'seed_{seed:03d}' / arm / f'{stage}.pt'


def reference_rows():
    rows = []
    for seed in range(3):
        for stage in STAGES:
            for scenario in scenarios(stage):
                rows.append(MAIN_EVAL / f'seed_{seed:03d}' / f'no_review_{stage}_{scenario}.json')
        for stage in MIXED:
            for scenario in (('B', 'mixed', 'history') if stage == 'B008' else ('B',)):
                rows.append(LAST_EVAL / f'seed_{seed:03d}' / f'last_{stage}_{scenario}.json')
    return rows


def prepare():
    assert not PLAN.exists()
    main = old.checked()
    train = read(TRAIN / 'protocol.json')
    files = [Path(__file__).resolve(), Path(__file__).with_name('eval_test.py'),
             Path(old.__file__), Path(timing.__file__), Path(base.__file__), Path(base.original.__file__),
             MAIN_EVAL / 'protocol.json', MAIN_EVAL / 'completion.json',
             LAST_EVAL / 'protocol.json', LAST_EVAL / 'completion.json',
             TRAIN / 'protocol.json', LAST_TRAIN / 'protocol.json']
    references = {}
    for path in reference_rows():
        row = read(path)
        files.append(path)
        references[str(path)] = dict(sha256=sha(path), artifacts=row['artifacts'],
            checkpoint_path=row['checkpoint_path'], checkpoint_sha256=row['checkpoint_sha256'])
    for seed in range(3):
        files.append(base.OUT / f'common_seed_{seed:03d}_related.pt')
        for scenario in ('B', 'mixed', 'history'):
            ref = base.reference(seed, 'related', scenario)
            files.extend((base.OUT / f'reference_seed_{seed:03d}_related_{scenario}.json', Path(ref['external_artifact'])))
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen before reading any new FINISH-position autonomous outcomes. All seeds, arms and declared snapshots retained. Earlier WAIT experiments and last supplement are known historical references.',
        arms=list(ARMS), seeds=[0, 1, 2], snapshots=list(STAGES), mixed_snapshots=list(MIXED), expected_records=len(expected_keys()),
        steps=main['steps'], batch=main['batch'], spec=main['spec'],
        training_protocol_hash=train['protocol_hash'], historical_protocol_hash=main['protocol_hash'],
        frozen_files={str(p): sha(p) for p in dict.fromkeys(files)}, historical_bindings=references,
        tests='2048-tick actual long_both B-only at B0/first2/first8/B1/B2/B4/B8; random per-NEW per-job A/B interleaving at B0/1/2/4/8; auxiliary A-only at B0/B8. Native eleven-action argmax, no learning, gating, oracle or hidden reset within each fork. Common original1024 WAIT warmup.',
        primary='Mixed old-A conflict cohort and new-B conflict cohort separately, with all matured denominators, each conflict raw and each seed. Frozen heldout identities, external events and scheduler must match original references.',
        timing_early=main['timing_early'], timing_legal=main['timing_legal'], timing_denominator=main['timing_denominator'],
        distinction='Not-started, started-without-legal-window and legal-window-without-FINISH remain distinct. Time error is actual active+started before-ready FINISH, independent of version/content. Legal first-FINISH correctness is separately assessed. Idle errors use actual enabled inactive opportunities.',
        calibration='midpoint must match historical last model/Adam/hidden/world bit-for-bit at all seven stages. Reuse raw evidence only on exact model-hash equality, then retain source and artifact hashes. Existing no_review42 and last21 result rows are immutable bound references.',
        analysis='Per-seed integer counts and equal-seed mean/sample SD; B learning gains relative to B0 and early first2/first8 snapshots. No pooled A/B rate, no causal rate or general meta-learning inference.',
        write_policy='Only data/finish-position/eval-v1 may be written. Imported original evaluator write functions are disabled during rollout and its output path points to an unused experiment-local path. Source files and historical frozen directories remain unchanged.')
    plan['protocol_hash'] = state_digest(plan)
    write(PLAN, plan)
    return plan


def checked():
    plan = read(PLAN)
    assert plan['protocol_hash'] == state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'})
    base.checked()
    for path, digest in plan['frozen_files'].items():
        assert sha(path) == digest, ('frozen dependency changed', path)
    return plan


def bind(row, path, cp, seed, arm, stage, scenario, plan):
    old.bind_checkpoint(row, path, cp, seed, arm, stage, scenario, plan)
    row.update(format=FORMAT, protocol_hash=plan['protocol_hash'], checkpoint_finish_position=old.json_metadata(cp.get('finish_position')))
    assert cp.get('finish_position'), ('missing FINISH metadata', path)
    assert cp['finish_position']['protocol_hash'] == plan['training_protocol_hash']
    assert cp['finish_position']['arm'] == arm
    if arm == 'midpoint':
        other_path = LAST_TRAIN / f'seed_{seed:03d}' / 'last' / f'{stage}.pt'
        other = torch.load(other_path, map_location='cpu', weights_only=False)
        hashes = {key: state_digest(cp[key]) for key in ('model', 'optimizer', 'hidden', 'world')}
        assert hashes == {key: state_digest(other[key]) for key in hashes}, ('midpoint/last calibration failed', seed, stage)
        row['last_calibration'] = dict(path=str(other_path), checkpoint_sha256=sha(other_path), state_hashes=hashes, exact=True)
    return row


def reuse_candidates(seed, arm, stage, scenario):
    if arm == 'midpoint':
        yield LAST_EVAL / f'seed_{seed:03d}' / f'last_{stage}_{scenario}.json'
    yield MAIN_EVAL / f'seed_{seed:03d}' / f'no_review_{stage}_{scenario}.json'


@torch.inference_mode()
def run_checkpoint(seed, arm, stage):
    plan = checked()
    path = checkpoint_path(seed, arm, stage)
    model, _, _, cp = base.training_api().load_checkpoint(path, device='cpu')
    model.eval()
    ss = plan['spec'][str(seed)]
    table, hf, bf = ss['mapping_table'], ss['history_family'], ss['B_family']
    model_hash = state_digest(cp['model'])
    shared = base.common(seed, table, hf, bf, arm='related', create=False)
    hidden = None
    for scenario in scenarios(stage):
        target = result_path(seed, arm, stage, scenario)
        if target.exists():
            continue
        started = time.perf_counter()
        source = next((p for p in reuse_candidates(seed, arm, stage, scenario)
                       if str(p) in plan['historical_bindings'] and read(p)['model_hash'] == model_hash), None)
        if source:
            row = deepcopy(read(source))
            assert sha(row['checkpoint_path']) == row['checkpoint_sha256']
            row.update(evidence_reused=True, original_result_path=str(source), original_result_sha256=sha(source))
        else:
            with source_write_guard():
                if hidden is None:
                    hidden = base.original.warmup(model, shared)
                with base.training_api().mapping_scope(table):
                    world = base.ContextWorld.from_state_dict(shared['world'], device='cpu')
                    row, frames, states = base.rollout(model, world, hidden,
                        families={'B': (bf,), 'mixed': (hf, bf), 'history': (hf,)}[scenario], steps=plan['steps'],
                        table=table, history_family=hf, b_family=bf, raw_groups=ss['raw_groups'])
            stem = owned(target.with_suffix(''))
            stem.parent.mkdir(parents=True, exist_ok=True)
            for ext in ('.npz', '.pt', '.requests.json'):
                assert not stem.with_suffix(ext).exists(), ('partial evidence exists', stem, ext)
            requests = dict(matured=row.pop('primary_requests'), actions=row.pop('action_requests'))
            np.savez_compressed(stem.with_suffix('.npz'), **frames)
            torch.save(states, stem.with_suffix('.pt'))
            write(stem.with_suffix('.requests.json'), requests)
            row.update(evidence_stem=str(stem), evidence_reused=False, warmup_observation_hash=shared['observation_hash'],
                artifacts={str(stem.with_suffix(ext)): sha(stem.with_suffix(ext)) for ext in ('.npz', '.pt', '.requests.json')})
        ref = base.reference(seed, 'related', scenario)
        for key in ('external_digest', 'primary_identity_hash', 'start_scheduler_hash', 'end_scheduler_hash', 'start_event_hash', 'end_event_hash'):
            assert row[key] == ref[key], (seed, arm, stage, scenario, key)
        bind(row, path, cp, seed, arm, stage, scenario, plan)
        row['evaluation_elapsed_seconds'] = time.perf_counter() - started
        write(target, row)
        print(json.dumps(dict(event='finish_position_eval', seed=seed, arm=arm, snapshot=stage, scenario=scenario,
                              reused=row['evidence_reused'], actual_unique_B_NEW=row['actual_unique_B_NEW'])), flush=True)


def available(seed=None, watch=False):
    pending = [(s, a, t) for s in ([seed] if seed is not None else range(3)) for a in ARMS for t in STAGES]
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
    return dict(complete=not pending, pending=pending)


def records():
    return [read(result_path(*key)) for key in sorted(expected_keys()) if result_path(*key).exists()]


def load_evidence(row):
    stem = Path(row['evidence_stem'])
    states = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=False)
    with np.load(stem.with_suffix('.npz')) as saved:
        frames = {key: saved[key] for key in saved.files}
    return frames, states, read(stem.with_suffix('.requests.json'))


def audit_row(row):
    for artifact, digest in row['artifacts'].items():
        assert sha(artifact) == digest
    frames, states, requests = load_evidence(row)
    assert np.array_equal(frames['actions'], frames['logits'].argmax(-1))
    for which, prefix in (('initial', 'start'), ('end', 'end')):
        assert state_digest(states[which + '_world']) == row[prefix + '_world_hash']
        assert state_digest(states[which + '_h']) == row[prefix + '_h_hash']
    rebuilt = base.score(frames, states['initial_world'], table=row['mapping_table'], history_family=row['history_family'],
        b_family=row['B_family'], fork_tick=row['fork_tick'], regime_id=row['evaluation_regime_id'], raw_groups=row['raw_groups'])
    for key in ('metrics', 'strata', 'by_family_raw', 'primary_identity_hash'):
        assert rebuilt[key] == row[key], (row['arm'], row['snapshot'], key)
    for key, value in rebuilt['counts'].items():
        assert row['counts'][key] == value
    assert rebuilt['primary_requests'] == requests['matured'] and rebuilt['action_requests'] == requests['actions']
    assert base.external_digest(frames) == row['external_digest']
    ref = base.reference(row['seed'], 'related', row['scenario'])
    with np.load(ref['external_artifact']) as external:
        for key in base.EXTERNAL_FIELDS:
            assert np.array_equal(frames[key], external[key]), (row['arm'], row['snapshot'], key)
    diagnostic = timing.analyze(row, frames, states, requests)
    for group in diagnostic['by_family'] + diagnostic['by_family_raw']:
        c = group['counts']
        c['not_started'] = c['matured'] - c['started']
        c['started_never_legal'] = c['started'] - c['reached_legal_window']
        assert c['matured'] == c['not_started'] + c['started_never_legal'] + c['first_legal_FINISH_emitted'] + c['legal_window_no_FINISH']
    return diagnostic


def audit(require_complete=False, seed=None):
    plan = checked()
    rows = [r for r in records() if seed is None or r['seed'] == seed]
    expected = {k for k in expected_keys() if seed is None or k[0] == seed}
    keys = {(r['seed'], r['arm'], r['snapshot'], r['scenario']) for r in rows}
    assert keys.issubset(expected) and len(keys) == len(rows)
    if require_complete:
        assert keys == expected
    signatures, exposures, timing_artifacts = {}, {}, []
    for row in rows:
        assert row['format'] == FORMAT and row['protocol_hash'] == plan['protocol_hash']
        assert sha(row['checkpoint_path']) == row['checkpoint_sha256']
        cp = torch.load(row['checkpoint_path'], map_location='cpu', weights_only=False)
        rebound = bind(deepcopy(row), Path(row['checkpoint_path']), cp, row['seed'], row['arm'], row['snapshot'], row['scenario'], plan)
        for key in ('checkpoint_work', 'actual_unique_B_NEW', 'checkpoint_finish_position', 'last_calibration'):
            assert rebound.get(key) == row.get(key)
        if row['evidence_reused']:
            assert sha(row['original_result_path']) == row['original_result_sha256']
            original = read(row['original_result_path'])
            for key in ('model_hash', 'counts', 'metrics', 'strata', 'by_family_raw', 'artifacts'):
                assert original[key] == row[key]
        diagnostic = audit_row(row)
        path = OUT / 'timing' / f"seed_{row['seed']:03d}" / f"{row['arm']}_{row['snapshot']}_{row['scenario']}.json"
        write(path, diagnostic)
        timing_artifacts.append(dict(path=str(path), sha256=sha(path)))
        sig = tuple(row[k] for k in ('external_digest', 'primary_identity_hash', 'start_scheduler_hash', 'end_scheduler_hash', 'start_event_hash', 'end_event_hash', 'warmup_observation_hash'))
        match = row['seed'], row['scenario']
        if match in signatures:
            assert signatures[match] == sig
        signatures[match] = sig
        match = row['seed'], row['snapshot']
        if match in exposures:
            assert exposures[match] == row['actual_unique_B_NEW']
        exposures[match] = row['actual_unique_B_NEW']
    reference_diagnostics = []
    for source in reference_rows():
        row = read(source)
        if seed is not None and row['seed'] != seed:
            continue
        assert sha(source) == plan['historical_bindings'][str(source)]['sha256']
        assert sha(row['checkpoint_path']) == row['checkpoint_sha256']
        for artifact, digest in row['artifacts'].items():
            assert sha(artifact) == digest
        peers = [r for r in rows if (r['seed'], r['snapshot'], r['scenario']) == (row['seed'], row['snapshot'], row['scenario'])]
        for peer in peers:
            for key in ('actual_unique_B_NEW', 'external_digest', 'primary_identity_hash', 'start_scheduler_hash', 'end_scheduler_hash', 'start_event_hash', 'end_event_hash', 'warmup_observation_hash'):
                assert peer[key] == row[key], ('reference mismatch', source, key)
        if row['scenario'] != 'mixed':
            continue
        diagnostic = audit_row(row)
        path = OUT / 'reference-timing' / f"seed_{row['seed']:03d}" / f"{row['arm']}_{row['snapshot']}_mixed.json"
        write(path, diagnostic)
        reference_diagnostics.append(dict(path=str(path), sha256=sha(path)))
    value = dict(format=FORMAT, protocol_hash=plan['protocol_hash'], verified=True, complete=keys == expected, seed=seed,
        records=len(rows), expected_records=len(expected), timing_artifacts=timing_artifacts, reference_timing_artifacts=reference_diagnostics,
        checks=['unchanged frozen sources and historical bindings', 'checkpoint model/metadata and actual B NEW exposure',
                'midpoint/last model Adam hidden world bit-for-bit calibration', 'native11 argmax and all per-tick semantic receipts',
                'raw request completion strict idle reconstruction', 'same external future and matured identity',
                'actual timing/content and no-BEGIN/no-window/no-FINISH distinction'])
    name = ('audit.json' if seed is None else f'audit_seed_{seed:03d}.json') if require_complete else f'audit-partial-seed{seed}-{len(rows):03d}.json'
    write(OUT / name, value)
    return value


def audit_worker(seed):
    path = OUT / f'audit_seed_{seed:03d}.json'
    if not path.exists():
        log_path = owned(OUT / f'audit_seed_{seed:03d}.log')
        with log_path.open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u', '-m', 'experiments.finish_position.eval_finish', 'audit', '--complete', '--seed', str(seed)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, ('seed audit failed', seed, str(log_path))
    return read(path)


def finalize():
    plan = checked()
    while not all(result_path(*key).exists() for key in expected_keys()):
        time.sleep(5)
    with ThreadPoolExecutor(max_workers=3) as pool:
        shards = list(pool.map(audit_worker, range(3)))
    for seed, shard in enumerate(shards):
        assert shard['verified'] and shard['complete'] and shard['seed'] == seed and shard['records'] == 28
        assert shard['protocol_hash'] == plan['protocol_hash']
    write(OUT / 'audit.json', dict(format=FORMAT, protocol_hash=plan['protocol_hash'], verified=True, complete=True, records=84,
        expected_records=84, checks=shards[0]['checks'],
        timing_artifacts=[r for shard in shards for r in shard['timing_artifacts']],
        reference_timing_artifacts=[r for shard in shards for r in shard['reference_timing_artifacts']],
        shard_reports={str(OUT / f'audit_seed_{s:03d}.json'): sha(OUT / f'audit_seed_{s:03d}.json') for s in range(3)}))
    return report()


def describe(values):
    values = [v for v in values if v is not None]
    return dict(mean=statistics.mean(values) if values else None, sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def report():
    plan = checked()
    audit_result = read(OUT / 'audit.json')
    assert audit_result['verified'] and audit_result['complete'] and audit_result['records'] == 84
    assert audit_result['protocol_hash'] == plan['protocol_hash']
    rows = records()
    assert len(rows) == len(expected_keys())
    refs = [read(p) for p in reference_rows()]
    flat = []
    for row in rows + refs:
        for family, cue in (('A', row['history_family']), ('B', row['B_family'])):
            for group, raws in dict(row['raw_groups'], **{f'raw_{r}': [r] for r in row['raw_groups']['conflict']}).items():
                counts = Counter()
                for raw in raws:
                    counts.update(row['by_family_raw'].get(f'{cue}:{raw}', {}).get('counts', {}))
                if counts.get('matured'):
                    flat.append(dict(seed=row['seed'], arm=row['arm'], snapshot=row['snapshot'], scenario=row['scenario'], family=family,
                        raw_group=group, counts=dict(counts), metrics=base.rates(counts), actual_unique_B_NEW=row['actual_unique_B_NEW']))
    summaries = []
    groups = sorted({tuple(r[k] for k in ('arm', 'snapshot', 'scenario', 'family', 'raw_group')) for r in flat})
    for arm, stage, scenario, family, group in groups:
        per = [r for r in flat if (r['arm'], r['snapshot'], r['scenario'], r['family'], r['raw_group']) == (arm, stage, scenario, family, group)]
        assert len(per) == 3
        summaries.append(dict(arm=arm, snapshot=stage, scenario=scenario, family=family, raw_group=group, per_seed=per,
            metrics={m: describe(r['metrics'][m] for r in per) for m in base.rates({})}))
    timing_rows = []
    for path in sorted((OUT / 'timing').glob('seed_*/*_mixed.json')) + sorted((OUT / 'reference-timing').glob('seed_*/*.json')):
        diagnostic = read(path)
        for group in diagnostic['by_family'] + diagnostic['by_family_raw']:
            timing_rows.append(dict(seed=diagnostic['seed'], arm=diagnostic['arm'], snapshot=diagnostic['snapshot'], **group))
    timing_summaries = []
    for arm, stage, family, raw in sorted({(r['arm'], r['snapshot'], r['family'], r.get('raw', -1)) for r in timing_rows}):
        per = [r for r in timing_rows if (r['arm'], r['snapshot'], r['family'], r.get('raw', -1)) == (arm, stage, family, raw)]
        assert len(per) == 3
        timing_summaries.append(dict(arm=arm, snapshot=stage, family=family, raw=raw, per_seed=per,
            metrics={m: describe(r['metrics'][m] for r in per) for m in per[0]['metrics']},
            early_lead_seconds=describe(r['first_error_actual_early_lead_seconds']['mean'] for r in per)))
    gains = []
    for arm in (*ARMS, 'no_review', 'last'):
        for stage in STAGES:
            selected = [r for r in flat if (r['arm'], r['snapshot'], r['scenario'], r['family'], r['raw_group']) == (arm, stage, 'B', 'B', 'all')]
            zero = {r['seed']: r for r in flat if (r['arm'], r['snapshot'], r['scenario'], r['family'], r['raw_group']) == (arm, 'B000', 'B', 'B', 'all')}
            if not selected:
                continue
            per = [dict(seed=r['seed'], actual_unique_B_NEW=r['actual_unique_B_NEW'], **{m: r['metrics'][m] - zero[r['seed']]['metrics'][m] for m in ('ordinary_completion_rate', 'strict_completion_rate')}) for r in selected]
            gains.append(dict(arm=arm, snapshot=stage, per_seed=per, metrics={m: describe(r[m] for r in per) for m in ('ordinary_completion_rate', 'strict_completion_rate')}))
    value = dict(format=FORMAT, protocol_hash=plan['protocol_hash'], complete=True, records=len(rows),
        summaries=summaries, mixed_timing=timing_summaries, B_gains_from_B000=gains, historical_bindings=plan['historical_bindings'])
    write(OUT / 'summary.json', value)
    def pick(arm, stage, scenario, family, group='all'):
        return next(r for r in summaries if (r['arm'], r['snapshot'], r['scenario'], r['family'], r['raw_group']) == (arm, stage, scenario, family, group))['metrics']
    def pct(v):
        return 'NA' if v is None else f'{100*v:.2f}%'
    lines = ['# FINISH位置：真实自由行动复测', '', '84/84新绑定；3 seed等权。midpoint是last WAIT加原FINISH中点，first仅把FINISH移到首次合法处。历史原始证据不改写。', '',
        '|快照|组|B普通|B严格|真实idle动作/千|', '|---|---|---:|---:|---:|']
    for stage in STAGES:
        for arm in (*ARMS, 'no_review'):
            m = pick(arm, stage, 'B', 'B')
            lines.append(f"|{stage}|{arm}|{pct(m['ordinary_completion_rate']['mean'])}|{pct(m['strict_completion_rate']['mean'])}|{m['idle_actions_per1000']['mean']:.3f}|")
    lines += ['', '|B8随机交错组|A冲突普通|A冲突严格|B冲突普通|B冲突严格|', '|---|---:|---:|---:|---:|']
    for arm in (*ARMS, 'last', 'no_review'):
        ma, mb = (pick(arm, 'B008', 'mixed', f, 'conflict') for f in ('A', 'B'))
        lines.append(f"|{arm}|{pct(ma['ordinary_completion_rate']['mean'])}|{pct(ma['strict_completion_rate']['mean'])}|{pct(mb['ordinary_completion_rate']['mean'])}|{pct(mb['strict_completion_rate']['mean'])}|")
    lines += ['', '|B8组|seed|家族|成熟|普通|严格|实际首错早发|未开始|已开始无窗口|首次legal正确|首次legal错误|窗口未发|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in timing_rows:
        if r['snapshot'] == 'B008' and 'raw' not in r:
            c = r['counts']
            lines.append('|' + '|'.join(map(str, [r['arm'], r['seed'], r['family'], *[c[k] for k in ('matured', 'completed', 'strict', 'first_error_actual_early_FINISH', 'not_started', 'started_never_legal', 'first_legal_FINISH_correct', 'first_legal_FINISH_wrong', 'legal_window_no_FINISH')]])) + '|')
    lines += ['', plan['timing_early'], '', plan['timing_legal'], '', plan['analysis'], '']
    report_path = owned(OUT / 'report.md')
    text = '\n'.join(lines)
    if report_path.exists():
        assert report_path.read_text(encoding='utf-8') == text
    else:
        report_path.write_text(text, encoding='utf-8')
    write(OUT / 'completion.json', dict(format=FORMAT, protocol_hash=plan['protocol_hash'], complete=True, records=len(rows),
        training_protocol_hash=plan['training_protocol_hash'], artifacts={str(OUT / n): sha(OUT / n) for n in ('audit.json', 'summary.json', 'report.md')}))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'available', 'watch', 'audit', 'report', 'finalize'))
    parser.add_argument('--seed', type=int, choices=range(3))
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'prepare':
        plan = prepare()
        print(json.dumps(dict(protocol_hash=plan['protocol_hash'], expected_records=plan['expected_records'])))
    elif args.command in ('available', 'watch'):
        print(json.dumps(available(args.seed, args.command == 'watch')))
    elif args.command == 'audit':
        print(json.dumps(audit(args.complete, args.seed)))
    elif args.command == 'report':
        print(json.dumps(dict(records=report()['records'])))
    else:
        print(json.dumps(dict(event='finish_position_eval_complete', records=finalize()['records'])), flush=True)


if __name__ == '__main__':
    main()
