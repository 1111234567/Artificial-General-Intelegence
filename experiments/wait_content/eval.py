"""Prospective free eleven-action comparison after equal-budget content training.

Only the scorer sees private world state.  The existing frozen evaluator owns
the actual autonomous trajectory; hooks below observe it without interventions.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.continuity.run import state_digest
from experiments.continuity.evaluate import _preserve_process_state
from experiments.lifelong.world import ContextWorld, NEW, REPLACE
from experiments.lifelong.evaluate import evaluate, _phases

OUT = ROOT/'data/wait-content/eval-v1'
PLAN = OUT/'protocol.json'
DATA = ROOT/'data/wait-content/data-v1'
TRAIN = ROOT/'data/wait-content/train-v1'
ARMS = ('untrained_anchor', 'end_only', 'wait_mixed')
CONDITIONS = ('short', 'long_both')
GROUPS = ('heldout', 'overlap', 'all')
EPS = 1e-6
FORMAT = 'wait-content-autonomous-evaluation-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(normalized(value), indent=2), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compressed_json(path, value):
    # Unstarted-world ready_time=+inf is an intentional private sentinel.  The
    # original full world retains it; portable raw scorer JSON represents it null.
    def portable(item):
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, list):
            return [portable(v) for v in item]
        if isinstance(item, dict):
            return {k: portable(v) for k, v in item.items()}
        return item
    Path(path).write_bytes(gzip.compress(json.dumps(normalized(portable(value)), separators=(',', ':')).encode(),
                                       compresslevel=1, mtime=0))


def prepare():
    if PLAN.exists():
        raise FileExistsError('immutable evaluation protocol already exists')
    sources = [Path(__file__).resolve(), Path(__file__).with_name('eval_test.py'),
               ROOT/'experiments/lifelong/evaluate.py', ROOT/'experiments/lifelong/world.py',
               ROOT/'experiments/continuity/world.py', ROOT/'experiments/continuity/evaluate.py']
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen before either new training arm has completed and before any new free-action result is read. All three seeds and arms are evaluated; no winner selection or subsequent threshold tuning.',
        source_files={str(p): sha(p) for p in sources}, seeds=[0, 1, 2], arms=list(ARMS),
        families=[0, 1, 2, 3], conditions=list(CONDITIONS), steps=2048, forks=72,
        source='Per-seed shared data-v1 real own-policy trajectory starts from the actual CP147456 complete world/h. Training consumes only lanes0..3 and changes supported recurrent edges and conditional FINISH affine parameters. All arms deploy from the identical actual external endpoint after16384 new observations, including scheduler, pending tasks, receipts, clock and context RNG. No original frozen directory is written.',
        hidden='For every arm, load its frozen final model, start with the common actual precollection h, and consume all16384 public observations with their original elapsed dt in order. This is common-history model-specific warmup; it is not an assertion that a new model generated the old actions. No h is reset at model load, request, stage, fork or evaluation. Anchor warmup must exactly reproduce saved collector end_h and all archived executed actions. New trained-arm h is allowed to differ. Training h from lanes0..3 is never padded or used as an8-lane deployment state.',
        policy='Exact same trained group-first controller class and11-action argmax for every arm. Future action receipts and h evolve on each arm own policy. No scorer gates, target actions, teacher forcing, frozen observations or weight updates at deployment.',
        matching='Per seed/family/condition require same start world/scheduler/event/case/actor identities and end scheduler/event streams, same mature primary request identity fingerprints in heldout/overlap/all, and disjoint future primary identities from all collection observations. Start hidden hashes are model-specific and are not falsely required equal.',
        primary=dict(actors=[4, 5, 6, 7], request_denominator='Postfork eligible requests under evaluation NEW contract whose deadlines actually matured. Same identities across arms, never selected by action or successful BEGIN.',
                     metrics=['ordinary_completion_rate', 'strict_completion_rate', 'idle_actions_per1000'],
                     idle='Enabled actually inactive job opportunities before own action; denominator can differ by policy. Always report both numerator and denominator plus BEGIN/FINISH partitions.'),
        secondary=['Feasible accepted BEGIN per primary matured request', 'legal FINISH opportunities, attempts, exact-content correct/wrong, accepted',
                   'active started processing before ready_time early FINISH; unstarted FINISH separately',
                   'cue, nearest processing midpoint and first legal conditional eight-FINISH content accuracy with missing-stage counts and policy-dependent denominators',
                   'all active processing conditional content and action counts, ordinary completion plus errors and repeat actions'],
        aggregation='Sum four family counts inside each seed/condition/arm; preserve each fork and overlap/all diagnostics. Report3 paired seed differences wait_mixed-minus-end_only, mean and sample SD. Anchor is descriptive. No significance or generality claim from3 seeds.',
        artifacts='Bind complete input files and model/source hashes at execution. Save per-fork frozen evaluator record, scorer counts, full primary request table and compressed per-tick preaction facts plus real action/result. Independent reconstruction can recompute heldout denominators and timing partitions.',
        limits=['Externally supervised recurrent and conditional FINISH learning on fixed collector experience; not new self-generated learning experience or learning native action ports from scratch.',
                'Actors4..7 are held out of appended action-head fitting, not historical backbone learning.',
                'One fixed budget and objective comparison; request/stage availability can change with policy.',
                'Warmup history was generated by the common collector, so trained-arm hidden states are same-history states rather than historical on-policy states.'])
    plan = normalized(plan)
    plan['protocol_hash'] = state_digest(plan)
    write(PLAN, plan)
    return plan


def checked():
    plan = read(PLAN)
    if plan['protocol_hash'] != state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'}):
        raise ValueError('evaluation protocol digest differs')
    amendments = OUT/'implementation-amendment.json'
    replacements = {}
    if amendments.exists():
        value = read(amendments)
        if value['protocol_hash'] != plan['protocol_hash']:
            raise ValueError('foreign implementation-only amendment')
        for item in value['changed_sources']:
            if item['original_sha256'] != plan['source_files'][item['path']] or sha(item['original_snapshot']) != item['original_sha256']:
                raise ValueError('amendment did not preserve original frozen source')
            replacements[item['path']] = item['amended_sha256']
    for path, expected in plan['source_files'].items():
        expected = replacements.get(path, expected)
        if sha(path) != expected:
            raise ValueError('bound evaluation source changed: '+path)
    return plan


def groups(lane):
    return ('heldout' if lane >= 4 else 'overlap', 'all')


def identity_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()


class Observer:
    """Read-only scorer for actual8-lane trajectories; no values enter the model."""
    def __init__(self, source, observed_ids=()):
        self.regime = source.regime_id+1
        self.fork_tick = source.tick
        self.cases = source.case_ids
        self.observed_ids = set(tuple(v) for v in observed_ids)
        self.rows = {}
        self.counts = {g: Counter() for g in GROUPS}
        self.last_logits = None
        self.last_obs_hash = None
        self.observations = []
        self.elapsed = []
        self.frames = []
        self.original_act = ContextWorld.act

    def key(self, world, i, j, rid=None):
        return (self.cases[i], j, int(world.request_id[i, j]) if rid is None else int(rid))

    def step(self, original, model, obs, h, dt, max_step=None):
        logits, state = original(model, obs, h, dt, max_step)
        self.last_logits = logits.detach().cpu().clone()
        self.last_obs_hash = state_digest({'observation': obs})
        self.observations.append(obs.detach().cpu().float().numpy().copy())
        self.elapsed.append(float(dt))
        return logits, state

    def sample(self, world, i, j, phase):
        values = self.last_logits[i, j]
        choice = int(values[3:].argmax())
        target = 4*int(world.version[i, j])+int(world.payload[i, j])
        return dict(tick=world.tick, time=world.now, phase=phase, choice=choice, target=target,
                    correct=choice == target, action=int(values.argmax()),
                    wait_minus_best_finish=float(values[0]-values[3:].max()))

    def mature(self, world):
        for i, j in world.matured_mask.nonzero().tolist():
            key = self.key(world, i, j, world.matured_request_id[i, j])
            if key not in self.rows:
                continue
            row = self.rows[key]
            if row['matured']:
                raise AssertionError('primary identity matured twice')
            row.update(matured=True, matured_tick=world.tick,
                       completed=bool(world.matured_completed[i, j]),
                       clean_completed=bool(world.matured_clean_completed[i, j]))
            for group in groups(i):
                c = self.counts[group]
                c['primary_matured'] += 1
                c['primary_completed'] += row['completed']
                c['primary_clean_completed'] += row['clean_completed']
                c['primary_began'] += row['accepted_begin_time'] is not None
                c['primary_feasible_began'] += bool(row['feasible_begin'])
                for phase in ('cue', 'processing_mid', 'first_legal'):
                    sample = row[phase]
                    c['primary_'+phase+'_observed'] += sample is not None
                    c['primary_'+phase+'_correct'] += bool(sample and sample['correct'])
                c['primary_complete_three_stage'] += all(row[p] is not None for p in ('cue', 'processing_mid', 'first_legal'))
                for phase in ('cue', 'processing_mid', 'first_legal'):
                    sample = row[phase]
                    c['primary_three_stage_'+phase+'_correct'] += bool(sample and sample['correct']) if all(row[p] is not None for p in ('cue', 'processing_mid', 'first_legal')) else 0

    def act(self, world, actions):
        if self.last_logits is None or not torch.equal(actions.cpu(), self.last_logits.argmax(-1)):
            raise AssertionError('scorer is not aligned to model11 argmax')
        self.mature(world)
        labels = world.oracle('midpoint')
        enabled = torch.zeros_like(world.active)
        enabled[:, :world.active_jobs] = True
        active = world.active & enabled
        phase = torch.from_numpy(_phases(world)).to(world.device)
        legal = active & phase.eq(3)
        processing = active & phase.eq(2)
        unstarted = active & ~world.started
        idle = enabled & ~active
        begin = (actions == 1) | (actions == 2)
        finish = actions >= 3
        target = 4*world.version+world.payload
        choice = self.last_logits[..., 3:].argmax(-1).to(world.device)
        exact = actions.sub(3).eq(target)
        command = world.event_kind.eq(NEW) | world.event_kind.eq(REPLACE)
        primary_command = command & world.event_eligible & world.event_regime_id.eq(self.regime)
        for i, j in primary_command.nonzero().tolist():
            key = self.key(world, i, j)
            if key in self.rows or key in self.observed_ids or key[2]//world.n_jobs <= self.fork_tick:
                raise AssertionError('future primary request leaked/duplicated')
            self.rows[key] = dict(case_id=key[0], actor_id=world.actor_ids[i], lane=i, job=j,
                request_id=key[2], family=int(world.current_family[i, j]),
                issued_time=float(world.issued_time[i, j]), deadline=float(world.deadline[i, j]),
                duration=float(world.duration[i, j]), raw_payload=int(world.raw_payload[i, j]),
                mapped_payload=int(world.payload[i, j]), version=int(world.version[i, j]),
                matured=False, completed=None, clean_completed=None, accepted_begin_time=None,
                feasible_begin=False, cue=self.sample(world, i, j, 'cue'),
                processing_mid=None, first_legal=None, actions=Counter())
        primary_mask = torch.zeros_like(active)
        for i in range(world.batch_size):
            for j in range(world.active_jobs):
                row = self.rows.get(self.key(world, i, j))
                if row is None:
                    continue
                primary_mask[i, j] = True
                if bool(processing[i, j]):
                    point = self.sample(world, i, j, 'processing_mid')
                    point['distance_from_midpoint'] = abs(world.now-(row['accepted_begin_time']+row['duration']/2))
                    old = row['processing_mid']
                    if old is None or point['distance_from_midpoint'] < old['distance_from_midpoint']:
                        row['processing_mid'] = point
                elif bool(legal[i, j]) and row['first_legal'] is None:
                    row['first_legal'] = self.sample(world, i, j, 'first_legal')
                if bool(active[i, j]):
                    row['actions']['positions'] += 1
                    row['actions']['begin'] += bool(begin[i, j])
                    row['actions']['finish'] += bool(finish[i, j])
                    row['actions']['processing_early_finish'] += bool(processing[i, j] & finish[i, j])
                    row['actions']['legal_finish_correct'] += bool(legal[i, j] & finish[i, j] & exact[i, j])
                    row['actions']['legal_finish_wrong'] += bool(legal[i, j] & finish[i, j] & ~exact[i, j])
        masks = dict(opportunities=enabled, active_opportunities=active,
            idle_opportunities=idle, idle_actions=idle & actions.ne(0),
            idle_begin=idle & begin, idle_finish=idle & finish,
            begin_attempts=enabled & begin, finish_attempts=enabled & finish,
            duplicate_begin=active & begin & world.started & actions.sub(1).eq(world.version),
            wrong_version_begin=active & begin & actions.sub(1).ne(world.version),
            legal_finish_opportunities=legal, legal_finish_attempts=legal & finish,
            legal_finish_correct=legal & finish & exact, legal_finish_wrong=legal & finish & ~exact,
            processing_opportunities=processing, processing_early_finish=processing & finish,
            processing_early_correct_content=processing & finish & exact,
            before_begin_finish=unstarted & finish, processing_wait=processing & actions.eq(0),
            processing_conditional_correct=processing & choice.eq(target),
            teacher_begin_positions=enabled & labels.gt(0) & labels.lt(3),
            teacher_finish_positions=enabled & labels.ge(3),
            primary_processing_opportunities=processing & primary_mask,
            primary_processing_early_finish=processing & primary_mask & finish,
            primary_legal_finish_opportunities=legal & primary_mask,
            primary_legal_finish_attempts=legal & primary_mask & finish,
            primary_legal_finish_correct=legal & primary_mask & finish & exact,
            primary_legal_finish_wrong=legal & primary_mask & finish & ~exact)
        for group, sl in (('heldout', slice(4, 8)), ('overlap', slice(0, 4)), ('all', slice(0, 8))):
            for name, mask in masks.items():
                self.counts[group][name] += int(mask[sl].sum())
        names = ('request_id', 'active', 'started', 'version', 'payload', 'duration',
                 'ready_time', 'deadline', 'event_kind', 'event_regime_id', 'event_eligible',
                 'matured_mask', 'matured_request_id', 'matured_completed', 'matured_clean_completed')
        frame = {name: getattr(world, name).cpu().tolist() for name in names}
        frame.update(tick=world.tick, time=world.now, elapsed=float(world.elapsed),
                     phase=phase.cpu().tolist(), enabled=enabled.cpu().tolist(),
                     primary_current=primary_mask.cpu().tolist(), primary_command=primary_command.cpu().tolist(),
                     conditional_finish_choice=choice.cpu().tolist(), target=target.cpu().tolist(),
                     actions=actions.cpu().tolist(), teacher_actions=labels.cpu().tolist(),
                     idle=idle.cpu().tolist(), observation_hash=self.last_obs_hash)
        result = self.original_act(world, actions)
        accepted = world.last_result.eq(1)
        for i, j in (accepted & begin).nonzero().tolist():
            row = self.rows.get(self.key(world, i, j))
            if row is not None:
                if row['accepted_begin_time'] is not None:
                    raise AssertionError('same request accepted BEGIN twice')
                row['accepted_begin_time'] = world.now
                row['feasible_begin'] = world.now+row['duration'] <= row['deadline']+EPS
        for group, sl in (('heldout', slice(4, 8)), ('overlap', slice(0, 4)), ('all', slice(0, 8))):
            self.counts[group]['accepted_begin'] += int((accepted & begin)[sl].sum())
            self.counts[group]['accepted_finish'] += int((accepted & finish)[sl].sum())
        frame['result'] = world.last_result.cpu().tolist()
        self.frames.append(frame)
        self.last_logits = None
        return result

    def result(self):
        values = {}
        for group in GROUPS:
            rows = [r for r in self.rows.values() if r['matured'] and
                    (group == 'all' or (r['lane'] >= 4) == (group == 'heldout'))]
            ids = [(r['case_id'], r['job'], r['request_id']) for r in rows]
            counts = dict(self.counts[group])
            if counts.get('primary_matured', 0) != len(ids):
                raise AssertionError('matured rows/count mismatch')
            values[group] = dict(counts=counts, primary_request_count=len(ids),
                                 primary_request_set_sha256=identity_hash(ids),
                                 no_collection_request_overlap=True)
        return values


def evaluate_observed(model, world, h, *, family, condition, steps=2048, observed_ids=()):
    observer = Observer(world, observed_ids)
    original_step = type(model).step
    def step(live, obs, state, dt, max_step=None):
        return observer.step(original_step, live, obs, state, dt, max_step)
    def act(live_world, actions):
        return observer.act(live_world, actions)
    with _preserve_process_state(), patch.object(type(model), 'step', step), \
            patch.object(ContextWorld, 'act', act):
        record = evaluate(model, world, h, family=family, condition=condition, steps=steps, device='cpu')
    shadow = observer.result()
    for field, counter in (('matured', 'primary_matured'), ('completed', 'primary_completed'),
                           ('clean_completed', 'primary_clean_completed')):
        if record['primary'][field] != shadow['all']['counts'].get(counter, 0):
            raise AssertionError('scorer/frozen evaluator primary mismatch: '+field)
    c = shadow['all']['counts']
    for field, counter in (('inactive_opportunities', 'idle_opportunities'), ('inactive_actions', 'idle_actions'),
                           ('inactive_begin_actions', 'idle_begin'), ('inactive_finish_actions', 'idle_finish'),
                           ('legally_timed_finish_attempts', 'legal_finish_attempts'),
                           ('legally_timed_wrong_content_attempts', 'legal_finish_wrong')):
        if record['opportunities'][field] != c.get(counter, 0):
            raise AssertionError('scorer/frozen evaluator opportunity mismatch: '+field)
    if len(observer.frames) != steps or record['action_hash'] != record['model_action_hash']:
        raise AssertionError('free-action trajectory was altered')
    return dict(record=record, shadow=shadow), observer


@torch.inference_mode()
def warmup(model, start_h, observations, elapsed, *, expected_h=None, expected_actions=None):
    state = start_h.detach().clone()
    digest = hashlib.sha256()
    for tick, (obs, dt) in enumerate(zip(observations, elapsed)):
        obs = torch.as_tensor(obs, dtype=torch.float64)
        logits, state = model.step(obs, state, float(dt))
        action = logits.argmax(-1)
        if expected_actions is not None and not torch.equal(action, torch.as_tensor(expected_actions[tick]).long()):
            raise AssertionError('anchor archived actual action differs at warmup tick '+str(tick))
        digest.update(action.numpy().astype(np.int64).tobytes())
    if len(observations) != len(elapsed):
        raise ValueError('warmup observation/dt length differs')
    if expected_h is not None and not torch.equal(state, expected_h):
        raise AssertionError('anchor continuous end_h reconstruction differs')
    return state, dict(steps=len(observations), hidden_resets=0,
        start_hidden_hash=state_digest({'hidden': start_h}), end_hidden_hash=state_digest({'hidden': state}),
        replay_action_sha256=digest.hexdigest(), anchor_end_h_exact=expected_h is not None,
        anchor_all_actions_exact=expected_actions is not None)


def load_observed_ids(path):
    value = read(path)
    values = value['observed_requests']
    if isinstance(values, dict):
        values = [item for group in values.values() for item in group]
    return [tuple(item) for item in values]


def run(seed):
    from experiments.wait_content import train
    torch.set_num_threads(1)
    plan = checked()
    training_plan = train.checked()
    folder = OUT/f'seed_{seed:03d}'
    result_path = folder/'result.json'
    if result_path.exists():
        raise FileExistsError('immutable seed evaluation result exists')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = DATA/f'seed_{seed:03d}'
    names = ('stream.npz', 'start.pt', 'endpoint.pt', 'requests.json')
    bindings = {str(data/name): sha(data/name) for name in names}
    bindings[str(Path(train.__file__).resolve())] = sha(train.__file__)
    bindings[str(train.PLAN)] = sha(train.PLAN)
    bindings[str(DATA/'protocol.json')] = sha(DATA/'protocol.json')
    bindings[str(data/'manifest.json')] = sha(data/'manifest.json')
    for source in train.source_info(seed).values():
        bindings[source['path']] = sha(source['path'])
    start = torch.load(data/'start.pt', map_location='cpu', weights_only=False)
    endpoint = torch.load(data/'endpoint.pt', map_location='cpu', weights_only=False)
    stream = np.load(data/'stream.npz', allow_pickle=False)
    observations, elapsed = stream['obs'], stream['elapsed']
    if observations.shape != (16384, 8, 56) or len(elapsed) != 16384:
        raise ValueError('wrong complete common warmup trajectory')
    world = ContextWorld.from_state_dict(endpoint['world'], device='cpu')
    if world.tick-start['world']['tick'] != 16384 or not endpoint['world']['acted']:
        raise ValueError('wrong real common endpoint boundary')
    observed_ids = load_observed_ids(data/'requests.json')
    base, original = train.load_initial(seed)
    if state_digest(base.state_dict()) != state_digest(start['collector_state']):
        raise AssertionError('initial arm differs from real collector model')
    if not torch.equal(start['h'], original.h) or state_digest(start['world']) != state_digest(original.world.state_dict()):
        raise AssertionError('common warmup does not start at real inherited h/world')
    records, warms, models = [], {}, {}
    for arm in ARMS:
        if arm == 'untrained_anchor':
            model = base
        else:
            model_path = TRAIN/f'seed_{seed:03d}'/arm/'checkpoint.pt'
            bindings[str(model_path)] = sha(model_path)
            model, checkpoint = train.load_trained(seed, arm, path=model_path)
            if (checkpoint['seed'], checkpoint['arm'], checkpoint['block'], checkpoint['protocol_hash']) != (seed, arm, 128, training_plan['protocol_hash']):
                raise AssertionError('wrong/incomplete trained checkpoint')
        model.eval()
        model_hash = state_digest(model.state_dict())
        h, warmed = warmup(model, start['h'], observations, elapsed,
            expected_h=endpoint['h'] if arm == 'untrained_anchor' else None,
            expected_actions=stream['actual_actions'] if arm == 'untrained_anchor' else None)
        models[arm] = model_hash
        warmup_path = folder/f'{arm}_warmup.pt'
        torch.save(dict(h=h, model_hash=model_hash, warmup=warmed, protocol_hash=plan['protocol_hash']),
                   warmup_path)
        warmed['artifact'] = dict(path=str(warmup_path), sha256=sha(warmup_path))
        warms[arm] = warmed
        for condition in CONDITIONS:
            for family in range(4):
                value, observer = evaluate_observed(model, world, h, family=family,
                    condition=condition, steps=2048, observed_ids=observed_ids)
                stem = f'{arm}_{condition}_family_{family}'
                trace_path, request_path = folder/(stem+'.trace.json.gz'), folder/(stem+'.requests.json.gz')
                compressed_json(trace_path, observer.frames)
                compressed_json(request_path, sorted(observer.rows.values(), key=lambda r: (r['lane'], r['job'], r['request_id'])))
                value.update(arm=arm, condition=condition, family=family,
                    trace_artifact=dict(path=str(trace_path), sha256=sha(trace_path)),
                    request_artifact=dict(path=str(request_path), sha256=sha(request_path)))
                if arm == 'untrained_anchor':
                    obs_path = folder/(stem+'.public-observations.npz')
                    np.savez_compressed(obs_path, obs=np.stack(observer.observations),
                                        elapsed=np.asarray(observer.elapsed, dtype=np.float64))
                    value['observations_artifact'] = dict(path=str(obs_path), sha256=sha(obs_path))
                records.append(value)
                write(folder/'partial.json', dict(seed=seed, protocol_hash=plan['protocol_hash'],
                      records=records, warmup=warms, seconds=time.perf_counter()-started))
                print(json.dumps(dict(seed=seed, arm=arm, condition=condition, family=family,
                                      done=len(records), seconds=time.perf_counter()-started)), flush=True)
        if model_hash != state_digest(model.state_dict()):
            raise AssertionError('deployment modified learned weights')
    assert_matching(records)
    if state_digest(world.state_dict()) != state_digest(endpoint['world']):
        raise AssertionError('common source world was mutated')
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise AssertionError('evaluation input changed while executing: '+path)
    checked()
    result = normalized(dict(format=FORMAT, seed=seed, protocol_hash=plan['protocol_hash'],
        records=records, warmup=warms, model_hashes=models, inputs=bindings,
        source_unchanged=True, source_world_hash=state_digest(endpoint['world']),
        common_start_hidden_hash=state_digest({'hidden': start['h']}),
        new_forks=len(records), seconds=time.perf_counter()-started))
    result['result_hash'] = state_digest(result)
    write(result_path, result)
    return result


def assert_matching(records):
    for condition in CONDITIONS:
        for family in range(4):
            values = [r for r in records if r['condition'] == condition and r['family'] == family]
            if {r['arm'] for r in values} != set(ARMS) or len(values) != len(ARMS):
                raise AssertionError('missing/duplicate evaluation arm')
            first = values[0]
            for value in values[1:]:
                for field in ('start_world_state_hash', 'start_scheduler_hash', 'end_scheduler_hash',
                              'start_event_hash', 'end_event_hash', 'case_ids', 'actor_ids', 'fork_tick'):
                    if value['record'][field] != first['record'][field]:
                        raise AssertionError('unmatched real external stream: '+field)
                for group in GROUPS:
                    if value['shadow'][group]['primary_request_set_sha256'] != first['shadow'][group]['primary_request_set_sha256']:
                        raise AssertionError('unmatched real matured primary identities')


def rates(counts):
    c = Counter(counts)
    def rate(n, d, scale=1):
        return scale*c[n]/c[d] if c[d] else None
    return dict(ordinary_completion_rate=rate('primary_completed', 'primary_matured'),
        strict_completion_rate=rate('primary_clean_completed', 'primary_matured'),
        feasible_begin_rate=rate('primary_feasible_began', 'primary_matured'),
        idle_actions_per1000=rate('idle_actions', 'idle_opportunities', 1000),
        idle_begin_per1000=rate('idle_begin', 'idle_opportunities', 1000),
        idle_finish_per1000=rate('idle_finish', 'idle_opportunities', 1000),
        processing_early_finish_rate=rate('processing_early_finish', 'processing_opportunities'),
        legal_finish_attempt_rate=rate('legal_finish_attempts', 'legal_finish_opportunities'),
        legal_finish_content_accuracy=rate('legal_finish_correct', 'legal_finish_attempts'),
        processing_conditional_content_accuracy=rate('processing_conditional_correct', 'processing_opportunities'),
        cue_content_accuracy=rate('primary_cue_correct', 'primary_cue_observed'),
        processing_mid_content_accuracy=rate('primary_processing_mid_correct', 'primary_processing_mid_observed'),
        first_legal_content_accuracy=rate('primary_first_legal_correct', 'primary_first_legal_observed'))


def summarize():
    plan = checked()
    rows, paired, bindings = [], [], {}
    for seed in range(3):
        path = OUT/f'seed_{seed:03d}'/'result.json'
        value = read(path)
        if value['protocol_hash'] != plan['protocol_hash'] or value['result_hash'] != state_digest({k: v for k, v in value.items() if k != 'result_hash'}):
            raise ValueError('seed result binding differs')
        bindings[str(path)] = sha(path)
        assert_matching(value['records'])
        for record in value['records']:
            for field in ('request_artifact', 'trace_artifact'):
                artifact = record[field]
                if sha(artifact['path']) != artifact['sha256']:
                    raise AssertionError('raw evidence bytes changed')
        for condition in CONDITIONS:
            per_arm = {}
            for arm in ARMS:
                c = Counter()
                selected = [r for r in value['records'] if r['condition'] == condition and r['arm'] == arm]
                for record in selected:
                    c.update(record['shadow']['heldout']['counts'])
                if len(selected) != 4:
                    raise AssertionError('need all four actual family forks')
                per_arm[arm] = rates(c)
                rows.append(dict(seed=seed, condition=condition, arm=arm, counts=dict(c), rates=per_arm[arm]))
            a, b = per_arm['wait_mixed'], per_arm['end_only']
            paired.append(dict(seed=seed, condition=condition,
                delta={k: a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in a}))
    aggregate = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for key in paired[0]['delta']:
            values = [r['delta'][key] for r in paired if r['condition'] == condition and r['delta'][key] is not None]
            aggregate[condition][key] = dict(paired_seeds=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)
    result = normalized(dict(format=FORMAT, protocol_hash=plan['protocol_hash'], forks=72,
        heldout=rows, paired=paired, paired_aggregate=aggregate, inputs=bindings, limits=plan['limits'],
        audit='All arms/seeds completed; real matured identities and external streams matched; all raw evidence SHA256 bound.'))
    result['summary_hash'] = state_digest(result)
    write(OUT/'summary.json', result)
    return result


def launch():
    checked()
    def one(seed):
        log_path = OUT/f'seed_{seed:03d}.log'
        with log_path.open('a', encoding='utf-8') as stream:
            command = [sys.executable, '-m', 'experiments.wait_content.eval', 'run', '--seed', str(seed)]
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     env=os.environ.copy())
            return dict(seed=seed, pid=child.pid, command=command, log=str(log_path), returncode=child.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(one, range(3)))
    write(OUT/'exit_codes.json', values)
    if any(v['returncode'] for v in values):
        raise RuntimeError('free evaluation failed: '+str(values))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'summarize', 'launch'))
    parser.add_argument('--seed', type=int, choices=range(3))
    args = parser.parse_args()
    value = prepare() if args.command == 'prepare' else run(args.seed) if args.command == 'run' else launch() if args.command == 'launch' else summarize()
    print(json.dumps(dict(format=FORMAT, output=str(OUT))), flush=True)


if __name__ == '__main__':
    main()
