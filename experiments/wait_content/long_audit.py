"""Independent read-only audit of the fixed longer autonomous continuations.

No model, world, learner or followup scorer is executed or imported.  Request
cohorts and action counts are reconstructed from saved compact fact arrays;
the complete common2048 prefixes are compared to original per-tick eval facts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/wait-content/long-v1'
EVAL = ROOT/'data/wait-content/eval-v1'
ARMS = ('untrained_anchor', 'end_only', 'wait_mixed')
GROUPS = {'heldout': range(4, 8), 'overlap': range(4), 'all': range(8)}
PREFIX, STOP = 2048, 8192
EPS = 1e-6
NEW, REPLACE = 1, 2
TRACE_PREFIX_FIELDS = ('request_id', 'active', 'started', 'version', 'payload',
    'duration', 'deadline', 'ready_time', 'event_kind', 'event_regime_id', 'event_eligible',
    'matured_mask', 'matured_request_id', 'matured_completed', 'matured_clean_completed',
    'tick', 'time', 'elapsed', 'phase', 'enabled', 'actions', 'result', 'teacher_actions')
EXTERNAL = ('request_id', 'version', 'payload', 'raw_payload', 'current_family', 'duration',
    'deadline', 'issued_time', 'current_eligible', 'current_regime_id', 'event_kind',
    'event_regime_id', 'event_eligible', 'matured_mask', 'matured_request_id')
INPUTS = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path)] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def compressed(path, expected=None):
    path = Path(path)
    actual = sha(path)
    if expected is not None:
        assert actual == expected, 'bound compressed fact changed: '+str(path)
    INPUTS[str(path)] = actual
    return json.loads(gzip.decompress(path.read_bytes()))


def identity(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()


def actions_hash(actions):
    return hashlib.sha256(b'lifelong-own-trace-v1/actions'+np.asarray(actions, dtype=np.int64).tobytes()).hexdigest()


def array_digest(data, names):
    digest = hashlib.sha256()
    for name in names:
        values = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def reconstruct(data, record):
    """Derive every eligible issuance, actual maturity and accepted BEGIN once."""
    requests = {}
    command = np.isin(data['event_kind'], [NEW, REPLACE]) & data['event_eligible'] & \
        (data['event_regime_id'] == record['evaluation_regime_id'])
    for offset, lane, job in np.argwhere(command):
        offset, lane, job = int(offset), int(lane), int(job)
        rid = int(data['request_id'][offset, lane, job])
        key = (lane, job, rid)
        assert key not in requests and rid//2 > record['fork_tick']
        assert rid//2 == int(data['tick'][offset]), 'request identity is not this actual issuance tick'
        requests[key] = dict(lane=lane, job=job, request_id=rid,
            case_id=record['case_ids'][lane], issued_offset=offset,
            duration=float(data['duration'][offset, lane, job]),
            deadline=float(data['deadline'][offset, lane, job]),
            matured=False, matured_offset=None, completed=False, clean_completed=False,
            began=False, feasible=False, accepted_begin_offset=None,
            receipt_completed=False, receipt_clean_completed=False, observed_active_error=False)
    for offset, lane, job in np.argwhere(data['matured_mask']):
        key = (int(lane), int(job), int(data['matured_request_id'][offset, lane, job]))
        row = requests.get(key)
        if row is None:
            continue
        assert not row['matured'] and int(offset) >= row['issued_offset']
        assert float(data['time'][offset]) > row['deadline']+EPS, 'deadline has not actually matured'
        row.update(matured=True, matured_offset=int(offset),
            completed=bool(data['matured_completed'][offset, lane, job]),
            clean_completed=bool(data['matured_clean_completed'][offset, lane, job]))
        assert not row['clean_completed'] or row['completed']
    accepted_begin = np.isin(data['actions'], [1, 2]) & (data['result'] == 1)
    for offset, lane, job in np.argwhere(accepted_begin):
        key = (int(lane), int(job), int(data['request_id'][offset, lane, job]))
        row = requests.get(key)
        if row is None:
            continue
        assert not row['began'], 'one identity accepted BEGIN twice'
        row.update(began=True, accepted_begin_offset=int(offset),
            feasible=float(data['time'][offset])+row['duration'] <= row['deadline']+EPS)
    # Independently derive ordinary/strict outcome from actual receipts.  A
    # rejected active attempt contaminates strict completion; idle actions after
    # completion remain separate and cannot retroactively contaminate it.
    errors = (data['result'] == -1) & data['active']
    finished = (data['result'] == 1) & (data['actions'] >= 3)
    for offset, lane, job in np.argwhere(errors | finished):
        key = (int(lane), int(job), int(data['request_id'][offset, lane, job]))
        row = requests.get(key)
        if row is None:
            continue
        if bool(errors[offset, lane, job]):
            row['observed_active_error'] = True
        if bool(finished[offset, lane, job]):
            assert not row['receipt_completed']
            assert bool(data['active'][offset, lane, job]) and int(data['phase'][offset, lane, job]) == 3
            target = 3+4*int(data['version'][offset, lane, job])+int(data['payload'][offset, lane, job])
            assert int(data['actions'][offset, lane, job]) == target
            if row['matured']:
                assert int(offset) < row['matured_offset']
            row['receipt_completed'] = True
            row['receipt_clean_completed'] = not row['observed_active_error']
    for row in requests.values():
        if row['matured']:
            assert row['completed'] == row['receipt_completed'], 'matured ordinary outcome differs from actual accepted FINISH'
            assert row['clean_completed'] == row['receipt_clean_completed'], 'matured strict outcome differs from actual active error history'
    return requests


def tally(data, requests, start, stop, suffix):
    output = {}
    for group, lanes in GROUPS.items():
        selected = [r for r in requests.values() if r['lane'] in lanes and r['matured'] and
            start <= r['matured_offset'] < stop and (not suffix or r['issued_offset'] >= start)]
        ids = [(r['case_id'], r['job'], r['request_id']) for r in selected]
        enabled = data['enabled'][start:stop].copy()
        enabled[:, [i for i in range(8) if i not in lanes]] = False
        active = enabled & data['active'][start:stop]
        idle = enabled & ~active
        action = data['actions'][start:stop]
        receipt = data['result'][start:stop]
        phase = data['phase'][start:stop]
        begin, finish = np.isin(action, [1, 2]), action >= 3
        legal, processing = active & (phase == 3), active & (phase == 2)
        exact = action-3 == 4*data['version'][start:stop]+data['payload'][start:stop]
        cohort = np.zeros_like(enabled)
        for r in selected:
            cohort[:, r['lane'], r['job']] |= data['request_id'][start:stop, r['lane'], r['job']] == r['request_id']
        c = Counter(primary_matured=len(selected), primary_completed=sum(r['completed'] for r in selected),
            primary_clean_completed=sum(r['clean_completed'] for r in selected),
            primary_began=sum(r['began'] for r in selected), primary_feasible_began=sum(r['feasible'] for r in selected))
        masks = dict(opportunities=enabled, active_opportunities=active,
            idle_opportunities=idle, idle_actions=idle & (action != 0), idle_begin=idle & begin, idle_finish=idle & finish,
            begin_attempts=enabled & begin, finish_attempts=enabled & finish,
            accepted_begin=enabled & begin & (receipt == 1), accepted_finish=enabled & finish & (receipt == 1),
            legal_finish_opportunities=legal, legal_finish_attempts=legal & finish,
            legal_finish_correct=legal & finish & exact, legal_finish_wrong=legal & finish & ~exact,
            processing_opportunities=processing, processing_early_finish=processing & finish,
            processing_wait=processing & (action == 0),
            cohort_legal_finish_opportunities=legal & cohort, cohort_legal_finish_attempts=legal & cohort & finish,
            cohort_legal_finish_correct=legal & cohort & finish & exact,
            cohort_legal_finish_wrong=legal & cohort & finish & ~exact,
            cohort_processing_opportunities=processing & cohort,
            cohort_processing_early_finish=processing & cohort & finish,
            cohort_accepted_finish=active & cohort & finish & (receipt == 1))
        for name, mask in masks.items():
            c[name] = int(mask.sum())
        crossing = [r for r in requests.values() if r['lane'] in lanes and r['issued_offset'] < start and
                    r['matured'] and start <= r['matured_offset'] < stop]
        censored = [r for r in requests.values() if r['lane'] in lanes and start <= r['issued_offset'] < stop and not r['matured']]
        issued = [r for r in requests.values() if r['lane'] in lanes and start <= r['issued_offset'] < stop]
        if suffix:
            assert len(issued) == len(selected)+len(censored), 'suffix-issued request lost or duplicated'
            assert not set(ids) & {(r['case_id'], r['job'], r['request_id']) for r in crossing}
            assert not set(ids) & {(r['case_id'], r['job'], r['request_id']) for r in censored}
        assert c['idle_actions'] == c['idle_begin']+c['idle_finish']
        assert c['legal_finish_attempts'] == c['legal_finish_correct']+c['legal_finish_wrong']
        output[group] = dict(counts=dict(c), primary_request_set_sha256=identity(ids),
            suffix_eligible_issued=len(issued) if suffix else None,
            excluded_prefix_crossing_requests=len(crossing) if suffix else 0,
            excluded_suffix_censored_requests=len(censored))
    return output


def compare_prefix(data, record, original, reconstructed):
    trace = compressed(original['trace_artifact']['path'], original['trace_artifact']['sha256'])
    assert len(trace) == PREFIX
    for name in TRACE_PREFIX_FIELDS:
        values = [f[name] for f in trace]
        if name == 'ready_time':
            values = [[[np.inf if v is None else v for v in row] for row in tick] for tick in values]
        original_array = np.asarray(values, dtype=data[name].dtype)
        assert np.array_equal(data[name][:PREFIX], original_array, equal_nan=True), 'original prefix fact differs: '+name
    assert actions_hash(data['actions'][:PREFIX]) == original['record']['action_hash'] == record['prefix_action_hash']
    for name, actual in record['prefix_endpoint'].items():
        assert actual == original['record'][name], 'declared original prefix endpoint differs: '+name
    for group in GROUPS:
        actual = reconstructed[group]
        previous = original['shadow'][group]
        assert actual['primary_request_set_sha256'] == previous['primary_request_set_sha256']
        for name, value in actual['counts'].items():
            # cohort_* is an added supplement-only breakdown; all original
            # action, completion, real-idle and timing fields must match facts.
            if name.startswith('cohort_'):
                continue
            assert value == previous['counts'].get(name, 0), (group, name, value, previous['counts'].get(name))
    return dict(prefix_ticks=PREFIX, independently_matched_fact_fields=list(TRACE_PREFIX_FIELDS),
                all_group_primary_and_action_counts_exact=True, action_hash_exact=True,
                declared_original_endpoint_world_scheduler_event_hidden_hashes_exact=True)


def audit_fork(record, original):
    artifact = record['facts_artifact']
    assert sha(artifact['path']) == artifact['sha256']
    INPUTS[artifact['path']] = artifact['sha256']
    with np.load(artifact['path'], allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    assert len(data['tick']) == STOP and record['steps'] == STOP and record['prefix_steps'] == PREFIX
    assert np.array_equal(data['tick'], record['fork_tick']+1+np.arange(STOP))
    assert data['actions'].shape == (STOP, 8, 2) and np.isin(data['actions'], np.arange(11)).all()
    assert np.isin(data['result'], [-1, 0, 1]).all()
    expected_phase = np.where(~data['started'],
        np.where(data['time'][:, None, None]+data['duration'] <= data['deadline']+EPS, 0, 1),
        np.where(data['time'][:, None, None] >= data['ready_time']-EPS, 3, 2))
    assert np.array_equal(expected_phase, data['phase'])
    assert actions_hash(data['actions']) == record['action_hash']
    assert array_digest(data, EXTERNAL) == record['external_stream_hash']
    assert array_digest({k: v[:PREFIX] for k, v in data.items()}, EXTERNAL) == record['prefix_external_stream_hash']
    requests = reconstruct(data, record)
    saved_rows = compressed(record['requests_artifact']['path'], record['requests_artifact']['sha256'])
    by_key = {(r['lane'], r['job'], r['request_id']): r for r in saved_rows}
    assert set(by_key) == set(requests), 'compact request table identity mismatch'
    for key, derived in requests.items():
        saved = by_key[key]
        for field in ('issued_offset', 'matured', 'matured_offset', 'accepted_begin_offset'):
            assert derived[field] == saved[field], (key, field)
        assert derived['feasible'] == saved['feasible_begin'], 'feasible BEGIN differs from actual time/deadline'
        if derived['matured']:
            assert derived['completed'] == saved['completed'] and derived['clean_completed'] == saved['clean_completed']
    prefix = tally(data, requests, 0, PREFIX, False)
    suffix = tally(data, requests, PREFIX, STOP, True)
    prefix_proof = compare_prefix(data, record, original, prefix)
    for scope, derived in (('prefix', prefix), ('suffix', suffix)):
        for group in GROUPS:
            actual, saved = derived[group], record[scope][group]
            assert actual['primary_request_set_sha256'] == saved['primary_request_set_sha256']
            for name in set(actual['counts']) | set(saved['counts']):
                assert actual['counts'].get(name, 0) == saved['counts'].get(name, 0), (scope, group, name)
            for name in ('excluded_prefix_crossing_requests', 'excluded_suffix_censored_requests'):
                assert actual[name] == saved[name], (scope, group, name)
    return dict(seed=record['seed'], arm=record['arm'], family=record['family'], prefix=prefix_proof,
                suffix=suffix, external_stream_hash=record['external_stream_hash'])


def make_rates(counts):
    c = Counter(counts)
    def ratio(numerator, denominator, scale=1):
        return scale*c[numerator]/c[denominator] if c[denominator] else None
    return dict(ordinary_completion_rate=ratio('primary_completed', 'primary_matured'),
        strict_completion_rate=ratio('primary_clean_completed', 'primary_matured'),
        idle_actions_per1000=ratio('idle_actions', 'idle_opportunities', 1000),
        idle_begin_per1000=ratio('idle_begin', 'idle_opportunities', 1000),
        idle_finish_per1000=ratio('idle_finish', 'idle_opportunities', 1000),
        feasible_begin_rate=ratio('primary_feasible_began', 'primary_matured'),
        legal_finish_attempt_rate=ratio('legal_finish_attempts', 'legal_finish_opportunities'),
        processing_early_finish_rate=ratio('processing_early_finish', 'processing_opportunities'),
        legal_finish_content_accuracy=ratio('legal_finish_correct', 'legal_finish_attempts'))


def run():
    plan = read(OUT/'protocol.json')
    assert (plan['prefix'], plan['steps'], plan['forks']) == (PREFIX, STOP, 36)
    original = {s: read(EVAL/f'seed_{s:03d}'/'result.json') for s in range(3)}
    proofs, aggregates = [], []
    long_summary = read(OUT/'summary.json')
    for seed in range(3):
        for arm in ARMS:
            path = OUT/f'seed_{seed:03d}'/arm/'result.json'
            result = read(path)
            assert result['protocol_hash'] == plan['protocol_hash']
            assert result['seed'] == seed and result['arm'] == arm and len(result['records']) == 4
            assert {r['family'] for r in result['records']} == set(range(4))
            summed = {group: Counter() for group in GROUPS}
            cross, censored, issued = Counter(), Counter(), Counter()
            for record in result['records']:
                before = [r for r in original[seed]['records'] if (r['arm'], r['condition'], r['family']) ==
                          (arm, 'long_both', record['family'])]
                assert len(before) == 1
                proof = audit_fork(record, before[0])
                proofs.append(proof)
                for group in GROUPS:
                    summed[group].update(proof['suffix'][group]['counts'])
                    cross[group] += proof['suffix'][group]['excluded_prefix_crossing_requests']
                    censored[group] += proof['suffix'][group]['excluded_suffix_censored_requests']
                    issued[group] += proof['suffix'][group]['suffix_eligible_issued']
            for group in GROUPS:
                aggregates.append(dict(seed=seed, arm=arm, actor_group=group, counts=dict(summed[group]),
                    suffix_eligible_issued=issued[group], excluded_prefix_crossing_requests=cross[group],
                    excluded_suffix_censored_requests=censored[group], rates=make_rates(summed[group])))
            published = [r for r in long_summary['heldout'] if r['seed'] == seed and r['arm'] == arm]
            assert len(published) == 1
            assert published[0]['counts'] == dict(summed['heldout'])
            assert published[0]['excluded_prefix_crossing_requests'] == cross['heldout']
            assert published[0]['excluded_suffix_censored_requests'] == censored['heldout']
            print(seed, arm, 'independent compact-fact audit passed', len(proofs), flush=True)
    assert len(proofs) == 36
    for seed in range(3):
        for family in range(4):
            same = [p for p in proofs if p['seed'] == seed and p['family'] == family]
            assert len(same) == 3 and len({p['external_stream_hash'] for p in same}) == 1
            for group in GROUPS:
                assert len({p['suffix'][group]['primary_request_set_sha256'] for p in same}) == 1
    pairs = []
    for seed in range(3):
        a = next(r for r in aggregates if (r['seed'], r['arm'], r['actor_group']) == (seed, 'wait_mixed', 'heldout'))
        b = next(r for r in aggregates if (r['seed'], r['arm'], r['actor_group']) == (seed, 'end_only', 'heldout'))
        delta = {key: a['rates'][key]-b['rates'][key] if a['rates'][key] is not None and b['rates'][key] is not None else None for key in a['rates']}
        published = next(r for r in long_summary['paired'] if r['seed'] == seed)
        for key, value in delta.items():
            assert value == published['delta'][key]
        pairs.append(dict(seed=seed, delta=delta))
    average = {}
    for key in pairs[0]['delta']:
        values = [r['delta'][key] for r in pairs if r['delta'][key] is not None]
        average[key] = dict(mean=statistics.mean(values) if values else None,
            sample_sd=statistics.stdev(values) if len(values) > 1 else None, seeds=len(values))
        assert average[key] == long_summary['paired_aggregate'][key]
    for path, expected in INPUTS.items():
        assert sha(path) == expected, 'input changed during independent audit: '+path
    output = dict(format='wait-content-fixed-long-independent-fact-audit-v1', audit='passed',
        protocol_hash=plan['protocol_hash'], forks=36, actual_new_behavior_ticks_executed=0,
        prefix_ticks_per_fork=PREFIX, suffix_ticks_per_fork=STOP-PREFIX,
        checks=['36 complete2048 prefixes match original per-tick23 fact fields and eleven-action hashes',
                'Request identity/issuance, eligibility, actual deadline maturity, BEGIN time/deadline rebuilt from facts',
                'Ordinary/strict outcomes independently derived from accepted FINISH and rejected active receipt history',
                'Suffix cohort excludes prefix-issued crossing requests and terminal right-censored requests',
                'All3 actor groups actual idle and legal/processing FINISH counters independently reconstructed',
                'Same external fact streams and mature suffix identities across all3 arms',
                'Every published perseed heldout count and paired primary difference recomputed exactly'],
        per_fork=proofs, per_seed=aggregates, paired=pairs, paired_aggregate=average, inputs=dict(INPUTS),
        source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        limits=['This is a read-only evidence audit; no hidden-state trajectory was recomputed.',
                'Prefix end world/scheduler/event/hidden fingerprints are compared with original declared fingerprints; per-tick fact arrays and action hashes are independently compared.'])
    path = OUT/'independent-audit.json'
    if path.exists():
        raise FileExistsError('immutable independent audit already exists')
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print('36prefix and suffix independent audit passed', flush=True)
    return output


def self_test():
    # A matured crossing request must be excluded; a suffix-issued matured one
    # included; a last-tick unobserved maturity must stay right-censored.
    shape = (6, 8, 2)
    data = {name: np.zeros(shape, dtype=np.int64) for name in
            ('request_id', 'actions', 'result', 'phase', 'version', 'payload')}
    data['enabled'] = np.ones(shape, dtype=bool)
    data['active'] = np.zeros(shape, dtype=bool)
    rows = {(4, 0, 1): dict(lane=4, job=0, request_id=1, case_id='c', issued_offset=0,
                matured=True, matured_offset=4, completed=True, clean_completed=True, began=True, feasible=True),
            (4, 0, 2): dict(lane=4, job=0, request_id=2, case_id='c', issued_offset=3,
                matured=True, matured_offset=5, completed=True, clean_completed=False, began=True, feasible=True),
            (4, 1, 3): dict(lane=4, job=1, request_id=3, case_id='c', issued_offset=5,
                matured=False, matured_offset=None, completed=False, clean_completed=False, began=False, feasible=False)}
    result = tally(data, rows, 3, 6, True)['heldout']
    assert result['counts']['primary_matured'] == 1 and result['counts']['primary_completed'] == 1
    assert result['counts']['primary_clean_completed'] == 0
    assert result['excluded_prefix_crossing_requests'] == 1
    assert result['excluded_suffix_censored_requests'] == 1 and result['suffix_eligible_issued'] == 2
    assert result['counts']['idle_opportunities'] == 24 and result['counts']['idle_actions'] == 0
    # Separate raw-fact fixture exercises issuance and strict deadline maturity;
    # this does not use a constructed model, world or source behavior.
    facts = {name: np.zeros(shape, dtype=np.int64) for name in
        ('request_id', 'event_kind', 'event_regime_id', 'actions', 'result', 'matured_request_id', 'phase', 'version', 'payload')}
    for name in ('event_eligible', 'matured_mask', 'matured_completed', 'matured_clean_completed'):
        facts[name] = np.zeros(shape, dtype=bool)
    facts['active'] = np.zeros(shape, dtype=bool)
    for name in ('duration', 'deadline'):
        facts[name] = np.zeros(shape, dtype=np.float64)
    facts['tick'] = np.arange(100, 106)
    facts['time'] = np.arange(6)*.25+25
    facts['event_kind'][3, 4, 0] = NEW
    facts['event_regime_id'][3, 4, 0] = 7
    facts['event_eligible'][3, 4, 0] = True
    facts['request_id'][3:, 4, 0] = 206
    facts['duration'][3:, 4, 0] = .25
    facts['deadline'][3:, 4, 0] = 26
    facts['matured_mask'][5, 4, 0] = True
    facts['matured_request_id'][5, 4, 0] = 206
    facts['actions'][3, 4, 0] = 1
    facts['result'][3, 4, 0] = 1
    facts['active'][3:5, 4, 0] = True
    facts['actions'][4, 4, 0] = 3
    facts['result'][4, 4, 0] = 1
    facts['phase'][4, 4, 0] = 3
    facts['matured_completed'][5, 4, 0] = True
    facts['matured_clean_completed'][5, 4, 0] = True
    record = dict(fork_tick=99, evaluation_regime_id=7, case_ids=[str(i) for i in range(8)])
    row = reconstruct(facts, record)[(4, 0, 206)]
    assert row['issued_offset'] == 3 and row['matured_offset'] == 5 and row['feasible']
    assert row['receipt_completed'] and row['receipt_clean_completed']
    facts['time'][5] = 26
    try:
        reconstruct(facts, record)
    except AssertionError as error:
        assert 'deadline has not actually matured' in str(error)
    else:
        raise AssertionError('audit accepted maturity exactly at deadline')
    print('independent suffix boundary/right-censor/idle and strict deadline fact fixtures passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'test'))
    args = parser.parse_args()
    self_test() if args.command == 'test' else run()
