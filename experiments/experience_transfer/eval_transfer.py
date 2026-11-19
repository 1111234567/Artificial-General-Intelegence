"""Frozen public-history, autonomous transfer evaluation with raw evidence.

Private world fields are recorded only for an independent scorer. Controllers
receive observe() and their own recurrent state; all actions are native argmax.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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
from experiments.continuity.run import state_digest
from experiments.learn_remember import evaluate_gru as original
from experiments.learn_remember.gru_train import GRUAgent
from experiments.lifelong.world import ContextWorld, NEW, REPLACE
from experiments.lifelong.evaluate import CONDITIONS

OUT = ROOT / 'data/experience-transfer/eval-v1'
TRAIN = ROOT / 'data/experience-transfer/training-v1'
PLAN = OUT / 'protocol.json'
FORMAT = 'experience-transfer-autonomous-eval-v1'
ROUNDS = (0, 1, 2, 4, 8)
SNAPSHOTS = ('B000', 'Bfirst002', 'Bfirst008', 'B001', 'B002', 'B004', 'B008')
ARMS = ('related', 'disjoint')
CURVE_ARMS = (*ARMS, 'direct')
EPS = 1e-6
EXTERNAL_FIELDS = ('tick', 'now', 'event_kind', 'event_version', 'raw_event_payload',
                   'event_family', 'event_duration', 'event_window',
                   'event_request_id', 'event_eligible', 'event_regime_id')
PRIVATE_FIELDS = ('active', 'started', 'version', 'payload', 'ready_time', 'deadline',
    'current_eligible', 'had_error', 'request_id', 'current_regime_id', 'raw_payload',
    'current_family', 'issued_time', 'duration', 'event_kind', 'event_version',
    'raw_event_payload', 'event_family', 'event_duration', 'event_window',
    'event_request_id', 'event_eligible', 'event_regime_id', 'matured_mask',
    'matured_request_id', 'matured_completed', 'matured_clean_completed')

sha = original.sha
read = original.read
write = original.write
array = original.array


def external_digest(frames):
    digest = hashlib.sha256()
    for name in EXTERNAL_FIELDS:
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(frames[name]).tobytes())
    return digest.hexdigest()


def rates(c):
    def ratio(n, d, scale=1.):
        return scale * c.get(n, 0) / c[d] if c.get(d, 0) else None
    return dict(ordinary_completion_rate=ratio('completed', 'matured'),
        strict_completion_rate=ratio('clean_completed', 'matured'),
        matured_with_valid_begin_rate=ratio('matured_with_valid_begin', 'matured'),
        matured_with_feasible_begin_rate=ratio('matured_with_feasible_begin', 'matured'),
        matured_with_legal_finish_rate=ratio('matured_with_legal_finish', 'matured'),
        valid_begin_precision=ratio('valid_begin', 'begin_attempts'),
        feasible_begin_precision=ratio('feasible_begin', 'begin_attempts'),
        legal_finish_precision=ratio('legal_finish', 'finish_attempts'),
        idle_actions_per1000=ratio('idle_actions', 'idle_opportunities', 1000.))


def groups_for(table, history_family, b_family):
    same = [r for r in range(4) if table[history_family][r] == table[b_family][r]]
    return dict(all=[0, 1, 2, 3], shared=same,
                conflict=[r for r in range(4) if r not in same])


def score(frames, initial, *, table, history_family, b_family, fork_tick, regime_id, raw_groups=None):
    """Reconstruct requests and action acceptance independently from saved arrays.

    Maturation is processed before a simultaneous new command, exactly as the
    frozen world exposes its pulses. Finish acceptance is reconstructed from
    action/version/content/start/readiness/deadline, never a conditional head.
    """
    baseline = original.score_arrays(frames, initial, fork_tick=fork_tick,
                                    evaluation_regime_id=regime_id)
    shape = initial['tensors']['request_id'].shape
    actors = initial['spec']['actor_ids']
    n_jobs = initial['spec']['n_jobs']
    enabled = np.broadcast_to(np.arange(n_jobs) < initial['spec']['active_jobs'], shape)
    ids = array(initial['tensors']['request_id'])
    regimes = array(initial['tensors']['current_regime_id'])
    began = array(initial['tensors']['started'])
    feasible_began = began.copy()
    finished = array(initial['tensors']['_score_completed'])
    clean_finished = array(initial['tensors']['_score_clean'])
    rejected_before = array(initial['tensors']['had_error'])
    family = array(initial['context']['tensors']['current_family'])
    raw = array(initial['context']['tensors']['raw_payload'])
    counters = {}
    rows = []

    def bucket(f, r):
        return counters.setdefault((int(f), int(r)), Counter())

    for tick in range(len(frames['actions'])):
        f = {key: value[tick] for key, value in frames.items()}
        mature = f['matured_mask']
        assert not np.any(mature & (f['matured_request_id'] != ids))
        primary_mature = mature & (ids // n_jobs > fork_tick) & (regimes == regime_id)
        for i, j in np.argwhere(primary_mature):
            assert bool(finished[i, j]) == bool(f['matured_completed'][i, j])
            assert bool(clean_finished[i, j]) == bool(f['matured_clean_completed'][i, j])
            c = bucket(family[i, j], raw[i, j])
            c.update(matured=1, completed=int(finished[i, j]), clean_completed=int(clean_finished[i, j]),
                     matured_with_valid_begin=int(began[i, j]),
                     matured_with_feasible_begin=int(feasible_began[i, j]),
                     matured_with_legal_finish=int(finished[i, j]))
            rows.append(dict(actor=int(actors[i]), job=int(j), request_id=int(ids[i, j]),
                family=int(family[i, j]), raw=int(raw[i, j]),
                valid_begin=bool(began[i, j]), feasible_begin=bool(feasible_began[i, j]),
                legal_finish=bool(finished[i, j]), clean_finish=bool(clean_finished[i, j])))

        command = (f['event_kind'] == NEW) | (f['event_kind'] == REPLACE)
        ids = np.where(command, f['request_id'], ids)
        regimes = np.where(command, f['current_regime_id'], regimes)
        family = np.where(command, f['current_family'], family)
        raw = np.where(command, f['raw_payload'], raw)
        for track in (began, feasible_began, finished, clean_finished, rejected_before):
            track[command] = False
        mapped = np.asarray(table)[family.clip(0), raw]
        assert np.array_equal(f['payload'][command], mapped[command])
        assert np.array_equal(f['event_family'][command], family[command])
        assert np.array_equal(f['raw_event_payload'][command], raw[command])
        public = f['obs'].reshape(*shape, 28)
        assert np.array_equal(public[..., 5:9][command].argmax(-1), raw[command])
        assert np.array_equal(public[..., 24:28][command].argmax(-1), family[command])
        a = f['actions']
        begin = (a == 1) | (a == 2)
        finish = (a >= 3) & (a < 11)
        version = np.where(begin, a - 1, (a - 3) // 4)
        same_version = version == f['version']
        accepted_begin = begin & f['active'] & same_version & ~f['started']
        feasible_begin = accepted_begin & (f['now'] + f['duration'] <= f['deadline'] + EPS)
        accepted_finish = finish & f['active'] & same_version & f['started'] & (
            (a - 3) % 4 == f['payload']) & (f['now'] >= f['ready_time'] - EPS) & (
            f['now'] <= f['deadline'] + EPS)
        accepted = accepted_begin | accepted_finish
        rejected = (a != 0) & ~accepted
        # Actual receipt and complete semantic acceptance must agree.
        assert np.array_equal(f['result'], accepted.astype(int) - rejected.astype(int))
        primary = (ids // n_jobs > fork_tick) & (regimes == regime_id) & f['current_eligible']
        clean = accepted_finish & ~rejected_before
        began |= accepted_begin
        feasible_began |= feasible_begin
        finished |= accepted_finish & primary
        clean_finished |= clean & primary
        rejected_before |= rejected & f['active']
        assert np.array_equal(rejected_before, f['had_error'] | (rejected & f['active']))
        idle = enabled & ~f['active']
        for fam in range(4):
            for r in range(4):
                content = (family == fam) & (raw == r)
                pmask = primary & content
                imask = idle & content
                c = bucket(fam, r)
                c.update(primary_issued=int((command & f['event_eligible'] & pmask).sum()),
                    begin_attempts=int((pmask & begin).sum()),
                    valid_begin=int((pmask & accepted_begin).sum()),
                    feasible_begin=int((pmask & feasible_begin).sum()),
                    finish_attempts=int((pmask & finish).sum()),
                    legal_finish=int((pmask & accepted_finish).sum()),
                    idle_opportunities=int(imask.sum()), idle_actions=int((imask & (a != 0)).sum()))
    total = Counter()
    for c in counters.values():
        total.update(c)
    for name in ('matured', 'completed', 'clean_completed', 'idle_opportunities', 'idle_actions', 'primary_issued'):
        assert total[name] == baseline['counts'].get(name, 0), (name, total[name], baseline['counts'])
    raw_groups = groups_for(table, history_family, b_family) if raw_groups is None else raw_groups
    strata = {}
    for fam in range(4):
        strata[str(fam)] = {}
        for group, members in raw_groups.items():
            c = Counter()
            for r in members:
                c.update(counters.get((fam, r), {}))
            strata[str(fam)][group] = dict(counts=dict(c), metrics=rates(c), raws=members)
    baseline.update(counts=dict(total), metrics=rates(total), strata=strata, raw_groups=raw_groups,
        action_requests=rows,
        by_family_raw={f'{f}:{r}': dict(counts=dict(c), metrics=rates(c)) for (f, r), c in counters.items()})
    return baseline


@torch.inference_mode()
def rollout(model, world, hidden, *, families, steps, table, history_family, b_family, raw_groups=None):
    before = state_digest(dict(model=model.state_dict(), world=world.state_dict(), h=hidden))
    live = ContextWorld.from_state_dict(world.state_dict(), device='cpu')
    initial = live.state_dict()
    h = hidden.detach().clone()
    live.set_family_policy(families)
    family_name = families[0] if len(families)==1 else 'mixed'
    live.set_regime(**CONDITIONS['long_both'], name=f'evaluation/{family_name}/long_both', dt=.25)
    regime_id = live.regime_id
    frames = {name: [] for name in (*PRIVATE_FIELDS, 'tick', 'now', 'obs', 'logits', 'actions', 'result')}
    for _ in range(steps):
        obs = live.observe()
        logits, h = model.step(obs, h, live.elapsed)
        if not torch.isfinite(logits).all() or not torch.isfinite(h).all():
            raise FloatingPointError('nonfinite autonomous state')
        actions = logits.argmax(-1)
        for name in PRIVATE_FIELDS:
            frames[name].append(array(getattr(live, name)))
        frames['tick'].append(live.tick)
        frames['now'].append(live.now)
        frames['obs'].append(array(obs))
        frames['logits'].append(array(logits))
        frames['actions'].append(array(actions))
        live.act(actions)
        frames['result'].append(array(live.last_result))
    frames = {key: np.asarray(value) for key, value in frames.items()}
    result = score(frames, initial, table=table, history_family=history_family,
                   b_family=b_family, fork_tick=world.tick, regime_id=regime_id, raw_groups=raw_groups)
    pending = live._score_pending & (live._score_request_id // live.n_jobs > world.tick) & (live.current_regime_id == regime_id)
    result['counts']['primary_censored'] = int(pending.sum())
    assert result['counts']['primary_issued'] == result['counts']['matured'] + int(pending.sum())
    assert np.array_equal(frames['actions'], frames['logits'].argmax(-1))
    end = live.state_dict()
    assert before == state_digest(dict(model=model.state_dict(), world=world.state_dict(), h=hidden))
    result.update(fork_tick=world.tick, end_tick=live.tick, families=list(families), steps=steps,
        evaluation_regime_id=regime_id, external_digest=external_digest(frames),
        start_world_hash=state_digest(initial), end_world_hash=state_digest(end),
        start_scheduler_hash=state_digest(initial['scheduler']), end_scheduler_hash=state_digest(end['scheduler']),
        start_event_hash=world.event_hash, end_event_hash=live.event_hash,
        start_h_hash=state_digest(hidden), end_h_hash=state_digest(h), model_hash=state_digest(model.state_dict()),
        continuity=dict(world_resets=0, hidden_resets=0, parameter_updates=0, argmax_action_changes=0))
    both = np.stack([(frames['active'] & (frames['current_family'] == family)).any((1, 2)) for family in (history_family, b_family)])
    result['steps_with_both_contexts_active'] = int(both.all(0).sum())
    result['actor_ticks_with_two_different_active_contexts'] = int((
        frames['active'].all(2) & (frames['current_family'][:, :, 0] != frames['current_family'][:, :, 1])).sum())
    return result, frames, dict(initial_world=initial, initial_h=hidden.detach().clone(),
                               end_world=end, end_h=h.detach().clone())


def training_api():
    from experiments.experience_transfer import train_transfer
    return train_transfer


def checked():
    plan = read(PLAN)
    assert plan['protocol_hash'] == state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'})
    for path, expected in plan['source_files'].items():
        if sha(path) != expected:
            raise ValueError('frozen evaluation dependency changed: ' + path)
    return plan


def common(seed, table, history_family, b_family, *, arm='related', create=False):
    path = OUT / f'common_seed_{seed:03d}_{arm}.pt'
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=False)
        assert value['protocol_hash'] == read(PLAN)['protocol_hash']
        assert value['table'] == table
        return value
    if not create:
        raise FileNotFoundError(path)
    api = training_api()
    with api.mapping_scope(table):
        world = ContextWorld(seed=730000 + seed, actor_ids=list(range(730000, 730008)),
            batch_size=8, n_jobs=2, active_jobs=2, dt=.25, future_family=history_family)
        birth = world.state_dict()
        observations, elapsed = [], []
        # Fixed shared public experiences; no oracle execution, labels or updates.
        for tick in range(1024):
            if tick % 128 == 0:
                world.set_family(history_family if (tick // 128) % 2 == 0 else b_family)
                condition = 'short' if tick < 512 else 'long_both'
                world.set_regime(**CONDITIONS[condition], name=f'common/{tick // 128}', dt=.25)
            observations.append(world.observe().clone())
            elapsed.append(world.elapsed)
            world.act(torch.zeros((8, 2), dtype=torch.long))
        value = dict(protocol_hash=read(PLAN)['protocol_hash'], seed=seed, table=table,
            birth_world=birth, world=world.state_dict(), observations=torch.stack(observations),
            elapsed=elapsed, observation_hash=state_digest(torch.stack(observations)),
            warmup_policy='all WAIT; no labels or learning')
        torch.save(value, path)
        return value


def reference(seed, arm, scenario, *, create=False):
    path = OUT / f'reference_seed_{seed:03d}_{arm}_{scenario}.json'
    if path.exists():
        row = read(path)
        assert row['protocol_hash'] == read(PLAN)['protocol_hash']
        return row
    if not create:
        raise FileNotFoundError(path)
    ss = read(PLAN)['spec'][str(seed)]
    table = ss['mapping_tables'][arm]
    hf, bf = ss['history_family'], ss['B_family']
    families = {'B': (bf,), 'history': (hf,), 'mixed': (hf, bf)}[scenario]
    shared = common(seed, table, hf, bf, arm=arm)
    # A separate WAIT-only world establishes the complete future external case
    # identities before any trained checkpoint is read.
    model = GRUAgent(width=1).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    with training_api().mapping_scope(table):
        world = ContextWorld.from_state_dict(shared['world'], device='cpu')
        result, frames, _ = rollout(model, world, model.initial_state(8), families=families,
            steps=read(PLAN)['steps'], table=table, history_family=hf, b_family=bf,
            raw_groups=ss['raw_groups'])
    np.savez_compressed(path.with_suffix('.npz'), **{k: frames[k] for k in EXTERNAL_FIELDS})
    row = {k: result[k] for k in ('external_digest', 'primary_identity_hash', 'start_scheduler_hash',
        'end_scheduler_hash', 'start_event_hash', 'end_event_hash', 'primary_requests')}
    row.update(protocol_hash=read(PLAN)['protocol_hash'], seed=seed, arm=arm, scenario=scenario,
               external_artifact=str(path.with_suffix('.npz')), external_artifact_sha256=sha(path.with_suffix('.npz')))
    write(path, row)
    return row


def prepare(spec):
    """spec fixes every seed's table, history cue and unseen B cue."""
    if PLAN.exists():
        raise FileExistsError(PLAN)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('eval_transfer_test.py'),
        Path(original.__file__), ROOT / 'experiments/experience_transfer/train_transfer.py', ROOT / 'experiments/lifelong/world.py',
        ROOT / 'experiments/continuity/world.py', ROOT / 'experiments/lifelong/evaluate.py']
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen before reading any transfer autonomous outcomes. All seeds and checkpoints retained; no selected winner or local threshold replaces curves.',
        source_files={str(p): sha(p) for p in sources}, spec=spec, seeds=[0, 1, 2],
        arms=list(ARMS), curve_arms=list(CURVE_ARMS), B_rounds=list(ROUNDS), snapshots=list(SNAPSHOTS), steps=2048, batch=8,
        primary='long_both B-only free action at B000/Bfirst002/Bfirst008/B001/B002/B004/B008, identical external stream and request identities for both histories and every snapshot in each seed. First snapshots are first-epoch2 and8 block updates, measured before the remaining first-round epochs.',
        retention='At B000 and B008 test the same old history cue on the same long heldout stream, with shared/conflict raw strata. Conflict raws are the old-only answers relative to B.',
        interleaving='At B000 and B008 future NEW contexts are sampled per job from (history,B), permitting simultaneous different contexts in a single model invocation. No batch-wide context flag or state reset.',
        direct='Common short-origin checkpoints, before any B learning, are evaluated on B only as an untrained-B reference. Time-aligned direct arm is evaluated on every B snapshot, with frozen short weights during old-history time. Direct curves distinguish related-history facilitation from disjoint-history interference; direct has fewer total updates.',
        warmup='Every snapshot starts newborn h once and consumes the identical independent 1024-observation WAIT history under its own frozen weights. Warmup alternates history/B cue every128 ticks, short for512 then long for512. Numerical states may differ; public histories never do. No teacher or labels in warmup or tests.',
        metrics='Matured eligible postfork regime requests are denominators, independent of BEGIN/success. Ordinary and strict completion, request-level valid/feasible BEGIN, true semantic legal FINISH, action precision, idle nonWAIT per1000; all numerators and denominators saved. Shared/conflict groups are frozen by A versus B and applied identically to C. C and A have identical answers on the A/B conflict group, whereas C also differs from B on the A/B shared group. Idle raw refers to retained slot content.',
        scorer='Independent reconstruction from saved private arrays and actual receipts, crosschecked against frozen score_arrays and native evaluator. No private field is input or an action gate. Raw/context public pulses verified against requests and mapped targets. Full external digest, scheduler hash and primary identities checked against WAIT reference and across policies.',
        analysis='Report raw B curves, per-seed related-minus-disjoint differences at each stage, and paired changes from B000 (difference in learning gains) to separate initial transfer from subsequent learning. Three seeds constrain inference; no significance or general meta-learning claim.',
        continuity='No hidden/world resets after deployment birth, including mixed contexts or requests. Frozen test branches never change training; parameters and original checkpoints hashed.',
        limitations=['Supervised prototype; old and B stages receive external oracle labels in training, never evaluation.',
            'One independent external stream per seed limits coverage; common WAIT warmup is an explicit deployment choice.',
            'Contextual rule distinction is public at NEW/REPLACE; test measures remembered per-request context, timing and control.'])
    plan['protocol_hash'] = state_digest(plan)
    write(PLAN, plan)
    for seed in plan['seeds']:
        seed_spec = spec[str(seed)]
        histories = []
        for arm in ARMS:
            histories.append(common(seed, seed_spec['mapping_tables'][arm], seed_spec['history_family'],
                seed_spec['B_family'], arm=arm, create=True))
        assert histories[0]['observation_hash'] == histories[1]['observation_hash']
        for arm in ARMS:
            for scenario in ('B', 'history', 'mixed'):
                reference(seed, arm, scenario, create=True)
    return plan


@torch.inference_mode()
def run_checkpoint(path, seed, arm, snapshot, *, direct=False):
    plan = checked()
    api = training_api()
    ss = plan['spec'][str(seed)]
    table_arm = 'related' if direct or arm=='direct' else arm
    table = ss['mapping_tables'][table_arm]
    hf, bf = ss['history_family'], ss['B_family']
    model, _, _, cp = api.load_checkpoint(path, device='cpu')
    model.eval()
    assert cp['mapping_table'] == table
    shared = common(seed, table, hf, bf, arm=table_arm)
    hidden = original.warmup(model, shared)
    scenarios = {'B': (bf,)}
    if snapshot in ('B000', 'B008') and arm in ARMS:
        scenarios.update(history=(hf,), mixed=(hf, bf))
    records = []
    for scenario, families in scenarios.items():
        stem = OUT / f'seed_{seed:03d}' / f'{arm}_{snapshot}_{scenario}'
        if stem.with_suffix('.json').exists():
            records.append(read(stem.with_suffix('.json')))
            continue
        start = time.perf_counter()
        with api.mapping_scope(table):
            world = ContextWorld.from_state_dict(shared['world'], device='cpu')
            result, frames, states = rollout(model, world, hidden, families=families,
                steps=plan['steps'], table=table, history_family=hf, b_family=bf, raw_groups=ss['raw_groups'])
        expected = reference(seed, table_arm, scenario)
        for key in ('external_digest', 'primary_identity_hash', 'start_scheduler_hash',
                    'end_scheduler_hash', 'start_event_hash', 'end_event_hash'):
            assert result[key] == expected[key], (seed, arm, snapshot, scenario, key)
        rows = result.pop('primary_requests')
        action_rows = result.pop('action_requests')
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stem.with_suffix('.npz'), **frames)
        torch.save(states, stem.with_suffix('.pt'))
        write(stem.with_suffix('.requests.json'), dict(matured=rows, actions=action_rows))
        result.update(format=FORMAT, protocol_hash=plan['protocol_hash'], seed=seed, arm=arm,
            snapshot=snapshot, B_round=int(snapshot[1:]) if not snapshot.startswith('Bfirst') else 0,
            scenario=scenario, checkpoint_path=str(Path(path).resolve()),
            checkpoint_sha256=sha(path), checkpoint_transfer=cp.get('transfer'),
            checkpoint_work=cp.get('work'), mapping_table=table,
            history_family=hf, B_family=bf, warmup_observation_hash=shared['observation_hash'],
            elapsed_seconds=time.perf_counter()-start,
            artifacts={str(stem.with_suffix(ext)): sha(stem.with_suffix(ext)) for ext in ('.npz', '.pt', '.requests.json')})
        write(stem.with_suffix('.json'), result)
        records.append(result)
        print(json.dumps(dict(event='transfer_eval', seed=seed, arm=arm, snapshot=snapshot,
            scenario=scenario, counts=result['counts'], metrics=result['metrics'])), flush=True)
    return records


def run_available():
    checked()
    total = 0
    for seed in range(3):
        for arm in CURVE_ARMS:
            for snapshot in SNAPSHOTS:
                path = TRAIN / f'seed_{seed:03d}' / arm / f'{snapshot}.pt'
                if path.exists():
                    total += len(run_checkpoint(path, seed, arm, snapshot))
        path = TRAIN / f'seed_{seed:03d}' / 'direct' / 'origin.pt'
        if path.exists():
            total += len(run_checkpoint(path, seed, 'short_origin', 'B000', direct=True))
    return total


def audit():
    plan = checked()
    records = []
    groups = {}
    for path in sorted(OUT.glob('seed_*/*.json')):
        if path.name.endswith('.requests.json'):
            continue
        row = read(path)
        assert row['format'] == FORMAT and row['protocol_hash'] == plan['protocol_hash']
        for artifact, expected in row['artifacts'].items():
            assert sha(artifact) == expected
        assert sha(row['checkpoint_path']) == row['checkpoint_sha256']
        states = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        with np.load(path.with_suffix('.npz')) as saved:
            frames = {key: saved[key] for key in saved.files}
        assert np.array_equal(frames['actions'], frames['logits'].argmax(-1))
        assert state_digest(states['initial_world']) == row['start_world_hash']
        assert state_digest(states['end_world']) == row['end_world_hash']
        assert state_digest(states['initial_h']) == row['start_h_hash']
        assert state_digest(states['end_h']) == row['end_h_hash']
        rebuilt = score(frames, states['initial_world'], table=row['mapping_table'],
            history_family=row['history_family'], b_family=row['B_family'],
            fork_tick=row['fork_tick'], regime_id=row['evaluation_regime_id'], raw_groups=row['raw_groups'])
        for key in ('metrics', 'strata', 'raw_groups', 'by_family_raw', 'primary_identity_hash'):
            assert rebuilt[key] == row[key], (path, key)
        assert rebuilt['primary_requests'] == read(path.with_suffix('.requests.json'))['matured']
        assert rebuilt['action_requests'] == read(path.with_suffix('.requests.json'))['actions']
        for key, value in rebuilt['counts'].items():
            assert row['counts'][key] == value
        assert external_digest(frames) == row['external_digest']
        expected = reference(row['seed'], 'related' if row['arm'] in ('short_origin', 'direct') else row['arm'], row['scenario'])
        assert sha(expected['external_artifact']) == expected['external_artifact_sha256']
        with np.load(expected['external_artifact']) as external:
            for key in EXTERNAL_FIELDS:
                assert np.array_equal(frames[key], external[key]), (path, key)
        for key in ('external_digest', 'primary_identity_hash', 'start_scheduler_hash',
                    'end_scheduler_hash', 'start_event_hash', 'end_event_hash'):
            assert row[key] == expected[key], (path, key)
        match = (row['seed'], row['scenario'])
        if match in groups:
            other = groups[match]
            for field in ('start_scheduler_hash', 'end_scheduler_hash',
                'start_event_hash', 'end_event_hash', 'external_digest', 'primary_identity_hash', 'warmup_observation_hash'):
                assert row[field] == other[field], (match, field)
        groups[match] = row
        records.append(row)
    report = dict(format=FORMAT, verified=True, forks=len(records), matched_groups=len(groups),
        protocol_hash=plan['protocol_hash'], checks=['frozen source and checkpoint hashes',
            'actual native eleven-action argmax', 'public raw/context pulses equal scorer identities',
            'semantically reconstructed BEGIN and legal FINISH equal actual receipts and maturation pulses',
            'independent original scorer counts', 'all stages/arms common worlds external streams and primary identities'])
    write(OUT / 'audit.json', report)
    return report


def summarize():
    plan = checked()
    records = [read(p) for p in sorted(OUT.glob('seed_*/*.json')) if not p.name.endswith('.requests.json')]
    summaries = []
    for scenario in ('B', 'history', 'mixed'):
        for arm in (*CURVE_ARMS, 'short_origin'):
            for snapshot in SNAPSHOTS:
                for family_part in ('all', 'B', 'history'):
                    for raw_group in ('all', 'shared', 'conflict'):
                        per_seed = []
                        for seed in range(3):
                            rows = [x for x in records if (x['seed'], x['arm'], x['snapshot'], x['scenario']) == (seed, arm, snapshot, scenario)]
                            if not rows:
                                continue
                            row = rows[0]
                            members = [row['B_family']] if family_part == 'B' else [row['history_family']] if family_part == 'history' else range(4)
                            c = Counter()
                            for f in members:
                                c.update(row['strata'][str(f)][raw_group]['counts'])
                            per_seed.append(dict(seed=seed, counts=dict(c), metrics=rates(c), work=row['checkpoint_work']))
                        if not per_seed:
                            continue
                        summary = dict(scenario=scenario, arm=arm, snapshot=snapshot, family_part=family_part,
                            raw_group=raw_group, per_seed=per_seed, metrics={})
                        for metric in rates({}):
                            values = [x['metrics'][metric] for x in per_seed if x['metrics'][metric] is not None]
                            summary['metrics'][metric] = dict(mean=statistics.mean(values) if values else None,
                                sample_sd=statistics.stdev(values) if len(values) > 1 else None)
                        summaries.append(summary)
    curves = []
    for group in ('all', 'shared', 'conflict'):
        for snapshot in SNAPSHOTS:
            selected = {arm: next((x for x in summaries if (x['scenario'], x['arm'], x['snapshot'], x['family_part'], x['raw_group']) == ('B', arm, snapshot, 'B', group)), None) for arm in ARMS}
            zero = {arm: next((x for x in summaries if (x['scenario'], x['arm'], x['snapshot'], x['family_part'], x['raw_group']) == ('B', arm, 'B000', 'B', group)), None) for arm in ARMS}
            if any(x is None for x in (*selected.values(), *zero.values())):
                continue
            row = dict(raw_group=group, snapshot=snapshot, metrics={})
            for metric in rates({}):
                def values(summary):
                    return {x['seed']: x['metrics'][metric] for x in summary['per_seed']}
                a, c, a0, c0 = (values(selected['related']), values(selected['disjoint']), values(zero['related']), values(zero['disjoint']))
                paired = []
                for seed in sorted(a.keys() & c.keys() & a0.keys() & c0.keys()):
                    if any(v[seed] is None for v in (a, c, a0, c0)):
                        continue
                    paired.append(dict(seed=seed, related_minus_disjoint=a[seed]-c[seed],
                        initial_difference=a0[seed]-c0[seed],
                        related_gain=a[seed]-a0[seed], disjoint_gain=c[seed]-c0[seed],
                        gain_difference=(a[seed]-a0[seed])-(c[seed]-c0[seed])))
                row['metrics'][metric] = dict(per_seed=paired)
                for field in ('related_minus_disjoint', 'initial_difference', 'related_gain', 'disjoint_gain', 'gain_difference'):
                    vv = [x[field] for x in paired]
                    row['metrics'][metric][field] = dict(mean=statistics.mean(vv) if vv else None,
                        sample_sd=statistics.stdev(vv) if len(vv)>1 else None)
            curves.append(row)
    direct_curves = []
    for group in ('all', 'shared', 'conflict'):
        for snapshot in SNAPSHOTS:
            for arm in ARMS:
                chosen = {}
                for who in (arm, 'direct'):
                    for stage in (snapshot, 'B000'):
                        item = next((x for x in summaries if (x['scenario'], x['arm'], x['snapshot'], x['family_part'], x['raw_group']) == ('B', who, stage, 'B', group)), None)
                        chosen[who, stage] = item
                if any(x is None for x in chosen.values()):
                    continue
                row = dict(raw_group=group, snapshot=snapshot, arm=arm, reference='direct', metrics={})
                for metric in rates({}):
                    vs = {key: {p['seed']:p['metrics'][metric] for p in x['per_seed']} for key,x in chosen.items()}
                    seeds = set.intersection(*(set(x) for x in vs.values()))
                    paired = []
                    for seed in sorted(seeds):
                        if any(x[seed] is None for x in vs.values()):
                            continue
                        a,c,a0,c0 = (vs[arm,snapshot][seed],vs['direct',snapshot][seed],vs[arm,'B000'][seed],vs['direct','B000'][seed])
                        paired.append(dict(seed=seed, difference=a-c, initial_difference=a0-c0,
                            arm_gain=a-a0, direct_gain=c-c0, gain_difference=(a-a0)-(c-c0)))
                    stat = dict(per_seed=paired)
                    for field in ('difference', 'initial_difference', 'arm_gain', 'direct_gain', 'gain_difference'):
                        vv = [x[field] for x in paired]
                        stat[field] = dict(mean=statistics.mean(vv) if vv else None,
                            sample_sd=statistics.stdev(vv) if len(vv)>1 else None)
                    row['metrics'][metric] = stat
                direct_curves.append(row)
    value = dict(format=FORMAT, protocol_hash=plan['protocol_hash'], forks=len(records), summaries=summaries,
                 paired_B_curves=curves, paired_direct_curves=direct_curves,
                 interpretation='Initial_difference is B0 transfer; gain_difference subtracts B0 and describes subsequent measured gains, not an isolated causal learning-rate parameter.')
    write(OUT / 'summary.json', value)
    return value


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'available', 'run', 'audit', 'summarize'))
    p.add_argument('--spec', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--arm', choices=(*CURVE_ARMS, 'short_origin'))
    p.add_argument('--snapshot', choices=SNAPSHOTS)
    args = p.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(read(args.spec)), indent=2))
    elif args.command == 'available':
        print(json.dumps(dict(forks=run_available())))
    elif args.command == 'run':
        run_checkpoint(args.checkpoint, args.seed, args.arm, args.snapshot, direct=args.arm=='short_origin')
    elif args.command == 'audit':
        print(json.dumps(audit(), indent=2))
    else:
        v = summarize()
        print(json.dumps(dict(forks=v['forks'], summaries=len(v['summaries']), curves=len(v['paired_B_curves']))))


if __name__ == '__main__':
    main()
