"""Independently reconstruct scorer counts from saved actual preaction facts.

This audit does not call Observer, evaluate_observed, or rates.  It checks all
three actor groups and every72 completed free trajectory before writing proof.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.wait_content import eval as e


def audit_fork(value):
    record = value['record']
    frames = json.loads(gzip.decompress(Path(value['trace_artifact']['path']).read_bytes()))
    requests = json.loads(gzip.decompress(Path(value['request_artifact']['path']).read_bytes()))
    assert len(frames) == record['steps'] == 2048
    counts = {group: Counter() for group in e.GROUPS}
    requests_seen = {}
    action_digest = hashlib.sha256(b'lifelong-own-trace-v1/actions')
    for frame in frames:
        array = lambda key: np.asarray(frame[key])
        action, receipt = array('actions'), array('result')
        enabled, active = array('enabled'), array('active') & array('enabled')
        idle, started = enabled & ~active, array('started')
        duration, deadline, ready = array('duration'), array('deadline'), array('ready_time')
        ready = np.array([[float(v) if v is not None else np.inf for v in row] for row in ready])
        computed_phase = np.where(~started, np.where(frame['time']+duration <= deadline+1e-6, 0, 1),
                                  np.where(frame['time'] >= ready-1e-6, 3, 2))
        assert np.array_equal(computed_phase, array('phase'))
        assert np.array_equal(idle, array('idle'))
        command = ((array('event_kind') == e.NEW) | (array('event_kind') == e.REPLACE))
        primary_command = command & array('event_eligible') & (array('event_regime_id') == record['evaluation_regime_id'])
        assert np.array_equal(primary_command, array('primary_command'))
        for i, j in np.argwhere(array('matured_mask')):
            key = (int(i), int(j), int(frame['matured_request_id'][i][j]))
            if key not in requests_seen:
                continue
            row = requests_seen[key]
            assert not row['matured']
            row['matured'] = True
            for group in e.groups(int(i)):
                c = counts[group]
                c['primary_matured'] += 1
                c['primary_completed'] += int(frame['matured_completed'][i][j])
                c['primary_clean_completed'] += int(frame['matured_clean_completed'][i][j])
                c['primary_began'] += row['began']
                c['primary_feasible_began'] += row['feasible']
        for i, j in np.argwhere(primary_command):
            key = (int(i), int(j), int(frame['request_id'][i][j]))
            assert key not in requests_seen and key[2]//record['n_jobs'] > record['fork_tick']
            requests_seen[key] = dict(matured=False, began=False, feasible=False)
        primary = np.array([[(i, j, int(frame['request_id'][i][j])) in requests_seen
                             for j in range(record['n_jobs'])] for i in range(record['batch_size'])])
        assert np.array_equal(primary, array('primary_current'))
        begin, finish = (action == 1) | (action == 2), action >= 3
        legal, processing = active & (computed_phase == 3), active & (computed_phase == 2)
        exact = action-3 == array('target')
        target = 4*array('version')+array('payload')
        assert np.array_equal(target, array('target'))
        teacher = array('teacher_actions')
        masks = dict(opportunities=enabled, active_opportunities=active,
            idle_opportunities=idle, idle_actions=idle & (action != 0), idle_begin=idle & begin, idle_finish=idle & finish,
            begin_attempts=enabled & begin, finish_attempts=enabled & finish,
            duplicate_begin=active & begin & started & (action-1 == array('version')),
            wrong_version_begin=active & begin & (action-1 != array('version')),
            legal_finish_opportunities=legal, legal_finish_attempts=legal & finish,
            legal_finish_correct=legal & finish & exact, legal_finish_wrong=legal & finish & ~exact,
            processing_opportunities=processing, processing_early_finish=processing & finish,
            processing_early_correct_content=processing & finish & exact,
            before_begin_finish=active & ~started & finish, processing_wait=processing & (action == 0),
            processing_conditional_correct=processing & (array('conditional_finish_choice') == target),
            teacher_begin_positions=enabled & (teacher > 0) & (teacher < 3), teacher_finish_positions=enabled & (teacher >= 3),
            primary_processing_opportunities=processing & primary, primary_processing_early_finish=processing & primary & finish,
            primary_legal_finish_opportunities=legal & primary,
            primary_legal_finish_attempts=legal & primary & finish,
            primary_legal_finish_correct=legal & primary & finish & exact,
            primary_legal_finish_wrong=legal & primary & finish & ~exact,
            accepted_begin=(receipt == 1) & begin, accepted_finish=(receipt == 1) & finish)
        for i, j in np.argwhere((receipt == 1) & begin):
            key = (int(i), int(j), int(frame['request_id'][i][j]))
            if key in requests_seen:
                row = requests_seen[key]
                assert not row['began']
                row['began'] = True
                row['feasible'] = frame['time']+duration[i, j] <= deadline[i, j]+1e-6
        for group, sl in (('heldout', slice(4, 8)), ('overlap', slice(0, 4)), ('all', slice(0, 8))):
            for name, mask in masks.items():
                counts[group][name] += int(mask[sl].sum())
        action_digest.update(np.asarray(frame['actions'], dtype=np.int64).tobytes())
    assert action_digest.hexdigest() == record['action_hash'] == record['model_action_hash']
    request_by_key = {(r['lane'], r['job'], r['request_id']): r for r in requests}
    assert set(request_by_key) == set(requests_seen)
    for key, row in requests_seen.items():
        saved = request_by_key[key]
        assert row['matured'] == saved['matured']
        assert row['began'] == (saved['accepted_begin_time'] is not None)
        assert bool(row['feasible']) == saved['feasible_begin']
    for group in e.GROUPS:
        rows = [r for r in requests if r['matured'] and
                (group == 'all' or (r['lane'] >= 4) == (group == 'heldout'))]
        ids = [(r['case_id'], r['job'], r['request_id']) for r in rows]
        c = counts[group]
        for phase in ('cue', 'processing_mid', 'first_legal'):
            c['primary_'+phase+'_observed'] = sum(r[phase] is not None for r in rows)
            c['primary_'+phase+'_correct'] = sum(bool(r[phase] and r[phase]['correct']) for r in rows)
            complete = [r for r in rows if all(r[p] is not None for p in ('cue', 'processing_mid', 'first_legal'))]
            c['primary_three_stage_'+phase+'_correct'] = sum(r[phase]['correct'] for r in complete)
        c['primary_complete_three_stage'] = len(complete)
        saved = value['shadow'][group]
        assert e.identity_hash(ids) == saved['primary_request_set_sha256']
        for key in set(c) | set(saved['counts']):
            assert c[key] == saved['counts'].get(key, 0), (group, key, c[key], saved['counts'].get(key))
    if value['arm'] == 'untrained_anchor':
        public = np.load(value['observations_artifact']['path'])
        assert public['obs'].shape == (2048, 8, 56)
        assert np.array_equal(public['elapsed'], np.array([f['elapsed'] for f in frames]))
        for obs, frame in zip(public['obs'], frames):
            assert e.state_digest({'observation': __import__('torch').as_tensor(obs)}) == frame['observation_hash']
    return dict(arm=value['arm'], condition=value['condition'], family=value['family'], steps=len(frames),
                all_groups_all_counts_exact=True, original_action_hash_exact=True)


def main():
    plan = e.checked()
    proofs = []
    inputs = {}
    for seed in range(3):
        path = e.OUT/f'seed_{seed:03d}'/'result.json'
        value = e.read(path)
        e.assert_matching(value['records'])
        inputs[str(path)] = e.sha(path)
        for record in value['records']:
            proofs.append(dict(seed=seed, **audit_fork(record)))
        print(seed, 'independent raw72-scorer audit progress', len(proofs), flush=True)
    assert len(proofs) == 72
    e.write(e.OUT/'raw-audit.json', dict(protocol_hash=plan['protocol_hash'], audit='passed', forks=len(proofs),
            checks=proofs, inputs=inputs, source_file=dict(path=str(Path(__file__).resolve()), sha256=e.sha(__file__)),
            scope='Independent per-tick real identity, timing, idle/BEGIN/FINISH/opportunity/action counters for heldout/overlap/all; own action hashes; saved primary request rows and stage count arithmetic; all anchor public observation hashes. Stage nearest-midpoint choice is not independently reselected here.'))


if __name__ == '__main__':
    main()
