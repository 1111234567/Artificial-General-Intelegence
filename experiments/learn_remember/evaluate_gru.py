"""Matched autonomous acquisition/withdrawal evaluation in the original world.

The controller receives only ContextWorld.observe(). Private fields below are
scorer evidence, never input, action gates, training labels, or state updates.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from collections import Counter
from copy import deepcopy
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
from experiments.lifelong.world import ContextWorld, NEW, REPLACE
from experiments.lifelong.evaluate import CONDITIONS

OUT = ROOT / 'data/learn-remember/evaluation-v1'
PLAN = OUT / 'protocol.json'
FORMAT = 'learn-remember-gru-autonomous-evaluation-v1'
STAGES = ('random', 'short', 'wide', 'withdrawal_mid', 'withdrawal', 'review_mid', 'review')
EPS = 1e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def array(value):
    return value.detach().cpu().numpy().copy()


def identity_digest(rows):
    identities = sorted((r['actor'], r['job'], r['request_id'], r['family'], r['raw'],
                         r['issued_time'], r['duration']) for r in rows)
    return hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()


def rates(counts):
    def ratio(n, d, scale=1.):
        return scale * counts.get(n, 0) / counts[d] if counts.get(d, 0) else None
    return dict(ordinary_completion_rate=ratio('completed', 'matured'),
                strict_completion_rate=ratio('clean_completed', 'matured'),
                idle_actions_per1000=ratio('idle_actions', 'idle_opportunities', 1000.))


def prepare():
    if PLAN.exists():
        raise FileExistsError('immutable evaluation protocol already exists')
    sources = [Path(__file__).resolve(), ROOT / 'experiments/learn_remember/evaluate_gru_test.py',
               ROOT / 'experiments/lifelong/world.py', ROOT / 'experiments/lifelong/evaluate.py',
               ROOT / 'experiments/continuity/world.py']
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen before reading any new GRU autonomous evaluation results; no winner, checkpoint, or threshold selection from this test.',
        source_files={str(p): sha(p) for p in sources}, seeds=[0, 1, 2],
        stages=list(STAGES), families=[0, 1, 2, 3], conditions=['short', 'long_both'],
        steps=2048, withdrawn_family=3, withdrawn_unique_raw=[1, 3],
        common_world=dict(seed_offset=100000, actor_ids=list(range(10000, 10008)),
                          dt=.25, batch_size=8, n_jobs=2, active_jobs=2, warmup_steps=1024),
        warmup='One independent heldout world is born per seed. Execute WAIT in every slot throughout1024 public observations. Switch future family and short/long_both regime every128 ticks. Every snapshot starts with newborn numerical h once, then consumes exactly this same observation/dt history using frozen snapshot weights. Only this explicit deployment birth initializes h; no state reset at requests, blocks, family changes or forks.',
        evaluation='Clone the complete common warmup world and each model-specific warmup h. Change future NEW family and duration/gap only, then run2048 actual observations with free eleven-action argmax. Own actions, receipts, active states and h continue. No target, clock, phase, family mapping or score is supplied beyond the original56 public observation features.',
        matching='All stages within seed/condition/family share exact heldout start world, scheduler, actor identities and public warmup. Require identical future external event digest, end scheduler/context RNG state and primary matured request identity digest. Own observations after first action can differ through public receipts, and hidden states differ by snapshot.',
        primary='Eligible requests issued after fork under the evaluation NEW regime with deadline actually matured. Denominator is never selected by successful BEGIN, active status or action. Inherited, prior-contract and censored requests are separate. Every family and raw symbol is retained.',
        metrics=['matured/completed/clean_completed and corresponding rates',
                 'actual inactive enabled job opportunities and non-WAIT actions per1000; numerator and denominator',
                 'family3 raw1+raw3 mature completion and strict rates; raw-specific idle rates are explicitly stale-retained-raw strata, not request denominators'],
        comparisons='Per seed and condition: wide-minus-random acquisition; withdrawal_mid-minus-wide and withdrawal-minus-wide retention. Supplemental review_mid/review checkpoints use the exact same forks and are compared with matched-duration no-review withdrawal checkpoints. Report all three paired values, mean and sample SD. Four-family aggregate is descriptive; short initially trains family3 and wide trains all4 families. No significance or general lifelong claim.',
        endpoint='Optional secondary real training-world/h forks retain actual endpoint states. Different checkpoint ages do not have matched future cases; never mix them into the common-history primary comparison.',
        continuity='One world and h per birth and per detached evaluation branch. Parameters frozen in all tests; optimizer and training RNG never imported into test. Evaluation does not mutate training checkpoints or invoke world.oracle().',
        evidence='Per fork compressed arrays contain public observations, logits, actual argmax actions, scoring masks and maturation pulses; primary request rows plus complete initial/end worlds/h saved. Audit independently recomputes mature outcomes, raw strata and idle counts from arrays and verifies sources/hashes. Validate against the unmodified lifelong.evaluate on short and long random-model cases before evaluating trained snapshots.',
        limits=['Supervised GRU prototype, not unsupervised or local-rule learning.',
                'Common WAIT history is an explicit matched deployment history, not each snapshot own training history.',
                'Retained state is continuous inside each independent test; frozen forks do not alter the continuously trained subject.',
                'Idle opportunities depend on policy; after terminal events raw is retained, so raw-stratified idle counts refer to that retained slot content.',
                'Three training seeds and one heldout world stream per seed constrain generality.'])
    plan['protocol_hash'] = state_digest(plan)
    write(PLAN, plan)
    for seed in plan['seeds']:
        common_stream(seed, plan, create=True)
    return plan


def checked():
    plan = read(PLAN)
    assert plan['protocol_hash'] == state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'})
    for path, expected in plan['source_files'].items():
        if sha(path) != expected:
            raise ValueError('frozen evaluation source changed: ' + path)
    return plan


def common_stream(seed, plan=None, *, create=False):
    plan = read(PLAN) if plan is None else plan
    path = OUT / f'common_seed_{seed:03d}.pt'
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=False)
        assert value['protocol_hash'] == plan['protocol_hash']
        return value
    if not create:
        raise FileNotFoundError(path)
    spec = plan['common_world']
    world = ContextWorld(seed=spec['seed_offset'] + seed, actor_ids=spec['actor_ids'],
                         batch_size=spec['batch_size'], n_jobs=spec['n_jobs'],
                         active_jobs=spec['active_jobs'], dt=spec['dt'], family_allowed=(0, 1, 2, 3))
    birth = world.state_dict()
    observations, elapsed = [], []
    for tick in range(spec['warmup_steps']):
        if tick % 128 == 0:
            block = tick // 128
            world.set_family(block % 4)
            name = 'short' if block < 4 else 'long_both'
            world.set_regime(**CONDITIONS[name], name=f'evaluation-common/{block}/{name}', dt=.25)
        observations.append(world.observe().clone())
        elapsed.append(world.elapsed)
        world.act(torch.zeros((world.batch_size, world.n_jobs), dtype=torch.long))
    value = dict(protocol_hash=plan['protocol_hash'], seed=seed, birth_world=birth,
                 world=world.state_dict(), observations=torch.stack(observations),
                 elapsed=elapsed, warmup_policy='all WAIT', world_hash=state_digest(world.state_dict()))
    torch.save(value, path)
    return value


@torch.inference_mode()
def warmup(model, common):
    state = model.initial_state(common['world']['spec']['batch_size'], device='cpu')
    for obs, dt in zip(common['observations'], common['elapsed']):
        logits, state = model.step(obs, state, dt)
        if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite common-history state')
    return state.detach().clone()


def score_arrays(frames, initial_world, *, fork_tick, evaluation_regime_id):
    """Scorer independent of model execution; reconstruct from real pulses."""
    tensors = initial_world['tensors']
    context = initial_world['context']['tensors']
    tracked = {name: array(value) for name, value in dict(
        request_id=tensors['request_id'], regime=tensors['current_regime_id'],
        raw=context['raw_payload'], family=context['current_family'],
        issued_time=tensors['issued_time'], duration=tensors['duration']).items()}
    actors = initial_world['spec']['actor_ids']
    jobs = initial_world['spec']['n_jobs']
    enabled = np.broadcast_to(np.arange(jobs) < initial_world['spec']['active_jobs'], tracked['raw'].shape)
    counts = Counter()
    by_raw = {str(raw): Counter() for raw in range(4)}
    cohorts = {name: Counter() for name in ('inherited', 'carryover_contract')}
    rows = []
    for tick in range(len(frames['actions'])):
        mature = frames['matured_mask'][tick]
        ids = frames['matured_request_id'][tick]
        assert not np.any(mature & (ids != tracked['request_id']))
        issued_after = (ids >= 0) & (ids // jobs > fork_tick)
        primary = mature & issued_after & (tracked['regime'] == evaluation_regime_id)
        for name, mask in [('primary', primary), ('inherited', mature & ~issued_after),
                           ('carryover_contract', mature & issued_after & ~primary)]:
            target = counts if name == 'primary' else cohorts[name]
            target['matured'] += int(mask.sum())
            target['completed'] += int((mask & frames['matured_completed'][tick]).sum())
            target['clean_completed'] += int((mask & frames['matured_clean_completed'][tick]).sum())
        for i, j in np.argwhere(primary):
            row = dict(actor=int(actors[i]), job=int(j), request_id=int(ids[i, j]),
                       family=int(tracked['family'][i, j]), raw=int(tracked['raw'][i, j]),
                       issued_time=float(tracked['issued_time'][i, j]), duration=float(tracked['duration'][i, j]),
                       matured_tick=int(frames['tick'][tick]), completed=bool(frames['matured_completed'][tick, i, j]),
                       clean_completed=bool(frames['matured_clean_completed'][tick, i, j]))
            rows.append(row)
            raw = by_raw[str(row['raw'])]
            raw['matured'] += 1
            raw['completed'] += row['completed']
            raw['clean_completed'] += row['clean_completed']
        command = (frames['event_kind'][tick] == NEW) | (frames['event_kind'][tick] == REPLACE)
        for name, frame_name in [('request_id', 'request_id'), ('regime', 'current_regime_id'),
                                 ('raw', 'raw_payload'), ('family', 'current_family'),
                                 ('issued_time', 'issued_time'), ('duration', 'duration')]:
            tracked[name] = np.where(command, frames[frame_name][tick], tracked[name])
        eligible = command & frames['event_eligible'][tick]
        counts['primary_issued'] += int((eligible & (tracked['regime'] == evaluation_regime_id)).sum())
        idle = enabled & ~frames['active'][tick]
        act = frames['actions'][tick]
        for name, mask in [('idle_opportunities', idle), ('idle_actions', idle & (act != 0)),
                           ('idle_begin', idle & (act > 0) & (act < 3)), ('idle_finish', idle & (act >= 3))]:
            counts[name] += int(mask.sum())
            for raw in range(4):
                by_raw[str(raw)][name] += int((mask & (frames['raw_payload'][tick] == raw)).sum())
    counts = dict(counts)
    raw_result = {key: dict(value, **rates(value)) for key, value in by_raw.items()}
    unique = Counter()
    for raw in ('1', '3'):
        unique.update(by_raw[raw])
    return dict(counts=counts, metrics=rates(counts), by_raw=raw_result,
                unique_raw_1_3=dict(unique, **rates(unique)),
                cohorts={key: dict(value) for key, value in cohorts.items()},
                primary_identity_hash=identity_digest(rows), primary_requests=rows)


@torch.inference_mode()
def rollout(model, world, hidden, *, family, condition, steps):
    before = state_digest(dict(model=model.state_dict(), world=world.state_dict(), h=hidden))
    live = ContextWorld.from_state_dict(world.state_dict(), device='cpu')
    initial_world = live.state_dict()
    h = hidden.detach().clone()
    live.set_family(family)
    live.set_regime(**CONDITIONS[condition], name=f'evaluation/{family}/{condition}', dt=.25)
    regime_id = live.regime_id
    frames = {name: [] for name in ('tick', 'obs', 'logits', 'actions', 'result', 'active',
        'request_id', 'current_regime_id', 'raw_payload', 'current_family', 'issued_time', 'duration',
        'event_kind', 'event_eligible', 'matured_mask', 'matured_request_id',
        'matured_completed', 'matured_clean_completed')}
    for _ in range(steps):
        obs = live.observe()
        logits, h = model.step(obs, h, live.elapsed)
        if not torch.isfinite(h).all() or not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite autonomous state')
        actions = logits.argmax(-1)
        for name in frames:
            if name not in ('obs', 'logits', 'actions', 'result', 'tick'):
                frames[name].append(array(getattr(live, name)))
        frames['tick'].append(live.tick)
        frames['obs'].append(array(obs))
        frames['logits'].append(array(logits))
        frames['actions'].append(array(actions))
        live.act(actions)
        frames['result'].append(array(live.last_result))
    frames = {key: np.asarray(value) for key, value in frames.items()}
    scores = score_arrays(frames, initial_world, fork_tick=world.tick, evaluation_regime_id=regime_id)
    end = live.state_dict()
    pending = live._score_pending & (live._score_request_id // live.n_jobs > world.tick) & (live.current_regime_id == regime_id)
    scores['counts']['primary_censored'] = int(pending.sum())
    assert scores['counts']['primary_issued'] == scores['counts']['matured'] + int(pending.sum())
    assert np.array_equal(frames['actions'], frames['logits'].argmax(-1))
    assert before == state_digest(dict(model=model.state_dict(), world=world.state_dict(), h=hidden))
    scores.update(fork_tick=world.tick, end_tick=live.tick, family=family, condition=condition,
        steps=steps, evaluation_regime_id=regime_id,
        start_world_hash=state_digest(initial_world), end_world_hash=state_digest(end),
        start_scheduler_hash=state_digest(initial_world['scheduler']), end_scheduler_hash=state_digest(end['scheduler']),
        start_context_hash=state_digest(initial_world['context']), end_context_hash=state_digest(end['context']),
        start_event_hash=world.event_hash, end_event_hash=live.event_hash,
        start_h_hash=state_digest(hidden), end_h_hash=state_digest(h), model_hash=state_digest(model.state_dict()),
        continuity=dict(world_resets=0, hidden_resets=0, parameter_updates=0, argmax_action_changes=0))
    states = dict(initial_world=initial_world, initial_h=hidden.detach().clone(), end_world=end, end_h=h.detach().clone())
    return scores, frames, states


def run_checkpoint(path, seed, stage, *, endpoint=False):
    plan = checked()
    from experiments.learn_remember.gru_train import load_checkpoint
    model, native_world, native_h, checkpoint = load_checkpoint(path, device='cpu')
    model.eval()
    source = 'endpoint' if endpoint else 'common'
    if endpoint:
        world, hidden = native_world, native_h
    else:
        common = common_stream(seed, plan)
        world = ContextWorld.from_state_dict(common['world'], device='cpu')
        hidden = warmup(model, common)
    records = []
    for condition in plan['conditions']:
        for family in plan['families']:
            stem = OUT / f'seed_{seed:03d}' / f'{stage}_{source}_{condition}_family_{family}'
            result_path = stem.with_suffix('.json')
            if result_path.exists():
                records.append(read(result_path))
                continue
            started = time.perf_counter()
            scores, frames, states = rollout(model, world, hidden, family=family,
                                             condition=condition, steps=plan['steps'])
            stem.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(stem.with_suffix('.npz'), **frames)
            torch.save(states, stem.with_suffix('.pt'))
            rows = scores.pop('primary_requests')
            write(stem.with_suffix('.requests.json'), rows)
            scores.update(format=FORMAT, protocol_hash=plan['protocol_hash'], seed=seed, stage=stage,
                source=source, checkpoint_path=str(Path(path).resolve()), checkpoint_sha256=sha(path),
                checkpoint_stage=checkpoint.get('stage'), checkpoint_work=checkpoint.get('work'),
                model_config=checkpoint.get('model_config'), elapsed_seconds=time.perf_counter()-started,
                artifacts={str(stem.with_suffix(ext)): sha(stem.with_suffix(ext))
                           for ext in ('.npz', '.pt', '.requests.json')})
            write(result_path, scores)
            records.append(scores)
            print(json.dumps({key: scores[key] for key in ('seed', 'stage', 'source', 'condition', 'family', 'counts', 'metrics')}), flush=True)
    return records


def audit():
    plan = checked()
    records = []
    for path in sorted(OUT.glob('seed_*/*_family_?.json')):
        result = read(path)
        assert result['protocol_hash'] == plan['protocol_hash']
        for artifact, expected in result['artifacts'].items():
            assert sha(artifact) == expected, artifact
        assert sha(result['checkpoint_path']) == result['checkpoint_sha256']
        states = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        with np.load(path.with_suffix('.npz')) as saved:
            frames = {key: saved[key] for key in saved.files}
        assert np.array_equal(frames['actions'], frames['logits'].argmax(-1))
        assert state_digest(states['initial_world']) == result['start_world_hash']
        assert state_digest(states['end_world']) == result['end_world_hash']
        assert state_digest(states['initial_h']) == result['start_h_hash']
        assert state_digest(states['end_h']) == result['end_h_hash']
        recomputed = score_arrays(frames, states['initial_world'], fork_tick=result['fork_tick'],
                                  evaluation_regime_id=result['evaluation_regime_id'])
        for field in ('metrics', 'by_raw', 'unique_raw_1_3', 'cohorts', 'primary_identity_hash'):
            assert recomputed[field] == result[field], (path, field)
        for key, value in recomputed['counts'].items():
            assert value == result['counts'][key], (path, key)
        assert recomputed['primary_requests'] == read(path.with_suffix('.requests.json'))
        records.append(result)
    groups = {}
    for row in records:
        if row['source'] == 'common':
            key = row['seed'], row['condition'], row['family']
            if key in groups:
                other = groups[key]
                for field in ('start_world_hash', 'start_scheduler_hash', 'start_context_hash', 'start_event_hash',
                              'end_scheduler_hash', 'end_context_hash', 'end_event_hash', 'primary_identity_hash'):
                    assert row[field] == other[field], (key, field)
            groups[key] = row
    report = dict(format=FORMAT, protocol_hash=plan['protocol_hash'], forks=len(records),
                  matched_common_groups=len(groups), verified=True,
                  checks=['frozen source hashes', 'checkpoint and raw evidence hashes', 'actual actions equal11-way argmax',
                          'independently reconstructed mature identities, outcomes, raw strata and idle counters',
                          'complete world/h artifact hashes', 'common-stage external streams and mature identities match'])
    write(OUT / 'audit.json', report)
    return report


def summarize():
    checked()
    records = [read(path) for path in sorted(OUT.glob('seed_*/*_family_?.json'))]
    summaries = []
    for source in ('common', 'endpoint'):
        for stage in STAGES:
            for condition in ('short', 'long_both'):
                for family in (0, 1, 2, 3, 'all'):
                    for raw_group in ('all', 'unique_raw_1_3'):
                        per_seed = []
                        for seed in range(3):
                            chosen = [r for r in records if (r['source'], r['stage'], r['condition'], r['seed']) ==
                                      (source, stage, condition, seed) and (family == 'all' or r['family'] == family)]
                            if len(chosen) != (4 if family == 'all' else 1):
                                continue
                            count = Counter()
                            for row in chosen:
                                value = row['counts'] if raw_group == 'all' else row['unique_raw_1_3']
                                count.update({k: v for k, v in value.items() if isinstance(v, int)})
                            per_seed.append(dict(seed=seed, counts=dict(count), **rates(count)))
                        if not per_seed:
                            continue
                        row = dict(source=source, stage=stage, condition=condition, family=family,
                                   raw_group=raw_group, per_seed=per_seed)
                        for metric in ('ordinary_completion_rate', 'strict_completion_rate', 'idle_actions_per1000'):
                            values = [r[metric] for r in per_seed if r[metric] is not None]
                            row[metric] = dict(mean=statistics.mean(values) if values else None,
                                              sample_sd=statistics.stdev(values) if len(values) > 1 else None)
                        summaries.append(row)
    differences = []
    for before, after in [('random', 'wide'), ('wide', 'withdrawal_mid'), ('wide', 'withdrawal'),
                          ('withdrawal_mid', 'review_mid'), ('withdrawal', 'review')]:
        for row in summaries:
            if row['source'] != 'common' or row['stage'] != after:
                continue
            parents = [r for r in summaries if r['source'] == 'common' and r['stage'] == before and all(
                r[key] == row[key] for key in ('condition', 'family', 'raw_group'))]
            if not parents:
                continue
            parent = parents[0]
            diff = {key: row[key] for key in ('condition', 'family', 'raw_group')}
            diff.update(source='common', before=before, after=after)
            for metric in ('ordinary_completion_rate', 'strict_completion_rate', 'idle_actions_per1000'):
                a = {r['seed']: r[metric] for r in parent['per_seed']}
                b = {r['seed']: r[metric] for r in row['per_seed']}
                values = [dict(seed=seed, difference=b[seed]-a[seed]) for seed in sorted(a.keys() & b.keys())
                          if a[seed] is not None and b[seed] is not None]
                diff[metric] = dict(per_seed=values,
                    mean=statistics.mean(v['difference'] for v in values) if values else None,
                    sample_sd=statistics.stdev(v['difference'] for v in values) if len(values) > 1 else None)
            differences.append(diff)
    value = dict(format=FORMAT, forks=len(records), summaries=summaries, paired_differences=differences)
    write(OUT / 'summary.json', value)
    return value


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'audit', 'summarize'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed', type=int, choices=range(3))
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--endpoint', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif args.command == 'run':
        if args.checkpoint is None or args.seed is None or args.stage is None:
            parser.error('run requires checkpoint, seed and stage')
        run_checkpoint(args.checkpoint, args.seed, args.stage, endpoint=args.endpoint)
    elif args.command == 'audit':
        print(json.dumps(audit(), indent=2))
    else:
        value = summarize()
        print(json.dumps(dict(forks=value['forks'], rows=len(value['summaries']), paired=len(value['paired_differences']))))


if __name__ == '__main__':
    main()
