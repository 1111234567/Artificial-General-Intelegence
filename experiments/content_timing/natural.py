"""Read-only content timing on the exact saved own-policy autonomous forks.

No fitting, forced actions, state substitutions or evaluator edits.  The eight
FINISH scores here are the deployed external controller's final action scores.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import torch
from experiments.action_return import readout as b, finish_readout as f, refit_policy as rp

OUT = b.ROOT / 'data/content-timing/natural-v1'
PLAN = OUT / 'protocol.json'
STAGES = ('cue', 'processing_mid', 'first_legal')
META = ('request_id', 'event_kind', 'event_eligible', 'event_regime_id',
        'active', 'started', 'version', 'payload', 'raw_payload', 'current_family',
        'duration', 'deadline', 'ready_time', 'issued_time', 'current_eligible',
        'current_regime_id', 'matured_mask', 'matured_request_id',
        'matured_completed', 'matured_clean_completed')
EPS = 1e-6


def normalized(value):
    return json.loads(json.dumps(value))


def prepare():
    if PLAN.exists():
        raise FileExistsError('immutable natural protocol exists')
    parent = rp.checked()
    inputs = {str(rp.PLAN): b.old.sha(rp.PLAN)}
    for seed in range(3):
        folder = rp.OUT / f'seed_{seed:03d}'
        result = b.old.read(folder / 'result.json')
        if result['protocol_hash'] != parent['protocol_hash']:
            raise ValueError('foreign saved refit result')
        if result['result_hash'] != b.state_digest({k: v for k, v in result.items() if k != 'result_hash'}):
            raise ValueError('refit result digest')
        for path in (folder / 'model_own_policy.pt', folder / 'result.json',
                     Path(parent['seeds'][str(seed)]['source']['endpoint']['path'])):
            inputs[str(path)] = b.old.sha(path)
        if inputs[str(folder / 'model_own_policy.pt')] != result['model_sha256']['own_policy']:
            raise ValueError('own-policy model SHA differs')
    source_paths = [Path(__file__).resolve(), Path(b.__file__).resolve(), Path(f.__file__).resolve(),
                    Path(rp.__file__).resolve(), b.ROOT / 'experiments/lifelong/evaluate.py',
                    b.ROOT / 'experiments/lifelong/world.py', b.ROOT / 'experiments/continuity/world.py']
    plan = dict(format='natural-content-timing-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        prospective='Additional diagnostic frozen before collecting its new traces; saved refit outcomes already exist. No refitting, new thresholds or additional policy arms.',
        source_files={str(p): b.old.sha(p) for p in source_paths}, input_files=inputs,
        parent_protocol_hash=parent['protocol_hash'], seeds=[0, 1, 2], arm='own_policy',
        deployment='Saved own_policy action head; original real147456 complete endpoint h/world for each seed. Unchanged lifelong.evaluate keep_trace=True; eleven-way argmax actions. Read-only class hooks record final logits, observations and pre-action world metadata.',
        forks=dict(families=[0, 1, 2, 3], conditions=['short', 'long_both'], steps=2048, total=24),
        population='Full original8-actor world. Primary tables: decoder-fitting-heldout lanes4..7, event-eligible post-fork requests in evaluation regime, whose deadlines mature within2048 ticks. Separately retain issued but censored requests. Backbone trained on all8 actors.',
        identity='case_id/job/request_id within seed/condition/family fork; NEW/REPLACE create distinct identities; retain actual request family, raw payload, mapped payload and version.',
        phases=dict(cue='First NEW/REPLACE observation consumed by normal step, before that tick action. Eligible primary commands all retained, including those never started/legal.',
            processing_origin='Actual accepted BEGIN time, equivalently ready_time-duration. Request issue time is separately retained and never used as processing start.',
            processing_mid='Among actual active+started observations strictly before ready_time-EPS and after accepted BEGIN, choose the one minimizing absolute distance to accepted_BEGIN_time+duration/2; tie chooses earlier observation. Selection never changes the rollout. Record realized fraction, distance, tick and duration. Missing if no actual processing observation.',
            first_legal='First actual active+started observation at now>=ready_time-EPS, before action. Active already excludes now>deadline+EPS. No counterfactual/legal gating reaches model.'),
        scores='Rank final deployed controller FINISH logits indices3..10 with stable decreasing score and lower action index on exact ties. Top FINISH class in0..7, rank of correct class, full8 scores, correct-minus-best-wrong margin, version=choice//4 and mapped payload=choice%4. These are deployed external action-head scores, not native ports. Actual full11 action and result are separate.',
        pairings='Within identical matured request identities, report cue->processing_mid, cue->first_legal and processing_mid->first_legal four-cell transitions for exact8-way correctness, version and payload. Report all-stage marginal denominators, paired early marginal and late marginal, and complete three-stage patterns. In particular lost=early correct/late wrong; gained=early wrong/late correct. No comparison of unpaired cue rate with survivor late rate.',
        feedback='At each stage retain previous public action/result from actual observation and pre-stage cumulative within-request BEGIN,FINISH,accepted,rejected counts plus accepted BEGIN time. Natural delayed scores can use own actions and feedback, other-slot cues and full recurrent dynamics; this is descriptive and does not isolate passive retention.',
        missing='Every matured primary row reports cue/mid/first-legal availability, accepted BEGIN, too-late BEGIN, completion, strict completion and missing reason. Stage denominators remain request counts; no substitution of teacher paths or all-position counts.',
        equality='Entire evaluator record after removing only returned trace must equal original saved own_policy record after JSON normalization; original shadow result must also match; request-set fingerprints and action hashes must agree. Source semantic state and model state unchanged.',
        aggregation='Sum family counts within each seed/condition. The3 seeds are independent replication units; family forks never counted as seeds. Retain fork tables and seed tables; give mean and sample SD of seed rates without pooled-binomial uncertainty.',
        runtime=dict(workers=3, cpu_threads=1, dtype='float64', device='cpu'),
        limits=['Post-result descriptive diagnostic of3 existing externally supervised heads.',
                'Survival to actual first legal is policy-dependent; complete three-stage subsets are selected.',
                'Natural trajectories retain true receipts and cross-slot information; cannot prove passive memory decay or information absence.',
                'Four family mappings overlap on some raw symbols; preserve raw/mapped labels and family breakdown.'])
    plan = normalized(plan)
    plan['protocol_hash'] = b.state_digest(plan)
    b.old.write(PLAN, plan)
    return plan


def checked():
    plan = b.old.read(PLAN)
    if plan['protocol_hash'] != b.state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'}):
        raise ValueError('natural protocol digest')
    for group in ('source_files', 'input_files'):
        for path, sha in plan[group].items():
            if b.old.sha(path) != sha:
                raise ValueError('bound source/input changed: ' + path)
    return plan


def load_model(seed, trainer):
    saved = torch.load(rp.OUT / f'seed_{seed:03d}/model_own_policy.pt', map_location='cpu', weights_only=False)
    begin_fits = [{k: saved[f'begin_head.j{j}_{k}'] for k in ('mean', 'std', 'coeff', 'bias')} for j in range(2)]
    finish_fits = [{k: saved[f'j{j}_{k}'] for k in ('mean', 'std', 'coef', 'bias')} for j in range(2)]
    model = f.FinishController(b.BeginController(trainer.model, 'non_input24', begin_fits,
                                              saved['begin_head.indices'].tolist()), finish_fits)
    model.load_state_dict(saved, strict=True)
    if b.state_digest(model.state_dict()) != b.state_digest(saved):
        raise ValueError('saved deployed head differs')
    return model


def sample(logits, obs, world, i, j, row):
    scores = logits[i, j, 3:].tolist()
    order = sorted(range(8), key=lambda k: (-scores[k], k))
    target = 4 * int(world.version[i, j]) + int(world.payload[i, j])
    chosen = order[0]
    begin_time = row.get('accepted_begin_time')
    duration = float(world.duration[i, j])
    return dict(tick=world.tick, time=world.now, task_age=world.now-float(world.issued_time[i, j]),
        processing_age=None if begin_time is None else world.now-begin_time,
        processing_fraction=None if begin_time is None else (world.now-begin_time)/duration,
        distance_from_processing_mid=None if begin_time is None else abs(world.now-(begin_time+duration/2)),
        finish_scores=scores, finish_order=order, choice=chosen, target=target,
        correct=chosen == target, version_correct=chosen//4 == target//4,
        payload_correct=chosen%4 == target%4, correct_rank=order.index(target)+1,
        margin=scores[target]-max(v for k, v in enumerate(scores) if k != target),
        action=int(logits[i, j].argmax()), result=None,
        previous_public_action=0 if not bool(obs[i].reshape(2, 28)[j, 11:22].any()) else
            int(obs[i].reshape(2, 28)[j, 11:22].argmax()),
        previous_public_accepted=bool(obs[i].reshape(2, 28)[j, 22]),
        previous_public_rejected=bool(obs[i].reshape(2, 28)[j, 23]),
        prior_request_actions=dict(row['actions']),
        had_error_before=bool(world.had_error[i, j]))


class Observer:
    def __init__(self, source, seed, family, condition):
        self.seed, self.family, self.condition = seed, family, condition
        self.regime = source.regime_id + 1
        self.rows = {}
        self.raw = {k: [] for k in META}
        self.logits, self.observations = [], []
        self.last_logits = self.last_obs = None
        self.shadow = b.Shadow(source, [])
        self.steps = 0

    def step(self, original, model, obs, h, dt, max_step=None):
        logits, state = original(model, obs, h, dt, max_step)
        self.last_logits, self.last_obs = logits.detach().cpu().clone(), obs.detach().cpu().clone()
        self.logits.append(self.last_logits)
        self.observations.append(self.last_obs)
        return logits, state

    def act(self, world, actions):
        if self.last_logits is None or not torch.equal(self.last_logits.argmax(-1), actions.cpu()):
            raise AssertionError('observer/action alignment')
        for k in META:
            self.raw[k].append(getattr(world, k).detach().cpu().clone())
        current = []
        for i in range(4, 8):
            for j in range(world.active_jobs):
                if bool(world.matured_mask[i, j]):
                    mature_key = (world.case_ids[i], j, int(world.matured_request_id[i, j]))
                    if mature_key in self.rows:
                        row = self.rows[mature_key]
                        if row['matured']:
                            raise AssertionError('request matured twice')
                        row.update(matured=True, matured_tick=world.tick,
                            completed=bool(world.matured_completed[i, j]),
                            clean_completed=bool(world.matured_clean_completed[i, j]))
                key = (world.case_ids[i], j, int(world.request_id[i, j]))
                command = int(world.event_kind[i, j]) in (b.NEW, b.REPLACE)
                primary_command = (command and bool(world.event_eligible[i, j]) and
                                   int(world.event_regime_id[i, j]) == self.regime)
                if primary_command:
                    if key in self.rows:
                        raise AssertionError('duplicate eligible command identity')
                    self.rows[key] = dict(seed=self.seed, condition=self.condition, fork_family=self.family,
                        case_id=key[0], actor_id=world.actor_ids[i], lane=i, job=j, request_id=key[2],
                        request_family=int(world.current_family[i, j]), raw_payload=int(world.raw_payload[i, j]),
                        mapped_payload=int(world.payload[i, j]), version=int(world.version[i, j]),
                        target=4*int(world.version[i, j])+int(world.payload[i, j]), event_kind=int(world.event_kind[i, j]),
                        issued_time=float(world.issued_time[i, j]), duration=float(world.duration[i, j]),
                        deadline=float(world.deadline[i, j]), cue=None, processing_mid=None, first_legal=None,
                        accepted_begin_time=None, matured=False, completed=None, clean_completed=None,
                        actions=Counter(), processing_observations=0)
                if key not in self.rows:
                    continue
                row = self.rows[key]
                active, started = bool(world.active[i, j]), bool(world.started[i, j])
                selected = []
                if primary_command:
                    row['cue'] = sample(self.last_logits, self.last_obs, world, i, j, row)
                    selected.append(row['cue'])
                if active and started:
                    if row['accepted_begin_time'] is None:
                        raise AssertionError('primary request started without observed accepted BEGIN')
                    if abs(row['accepted_begin_time']-(float(world.ready_time[i, j])-row['duration'])) > 1e-9:
                        raise AssertionError('processing time origin differs from accepted BEGIN')
                    if world.now < float(world.ready_time[i, j])-EPS:
                        row['processing_observations'] += 1
                        candidate = sample(self.last_logits, self.last_obs, world, i, j, row)
                        old = row['processing_mid']
                        if old is None or candidate['distance_from_processing_mid'] < old['distance_from_processing_mid']:
                            row['processing_mid'] = candidate
                            selected.append(candidate)
                    elif row['first_legal'] is None:
                        row['first_legal'] = sample(self.last_logits, self.last_obs, world, i, j, row)
                        selected.append(row['first_legal'])
                if active:
                    current.append((i, j, row, selected))
        result = self.shadow.act(world, actions)
        for i, j, row, selected in current:
            action, receipt = int(actions[i, j]), int(world.last_result[i, j])
            for point in selected:
                point['result'] = receipt
            counts = row['actions']
            counts['positions'] += 1
            counts['begin'] += int(action in (1, 2))
            counts['finish'] += int(action >= 3)
            counts['accepted'] += int(receipt == 1)
            counts['rejected'] += int(receipt == -1)
            counts['accepted_begin'] += int(receipt == 1 and action in (1, 2))
            counts['accepted_finish'] += int(receipt == 1 and action >= 3)
            if receipt == 1 and action in (1, 2):
                if row['accepted_begin_time'] is not None:
                    raise AssertionError('accepted BEGIN twice')
                row['accepted_begin_time'] = world.now
                row['begin_too_late_for_deadline'] = world.now + row['duration'] > row['deadline'] + EPS
        self.steps += 1
        self.last_logits = None
        return result

    def finish(self):
        values = []
        for key, row in sorted(self.rows.items()):
            if row['cue'] is None:
                reason = 'cue_unobserved'
            elif row['accepted_begin_time'] is None:
                reason = 'never_began'
            elif row['first_legal'] is None:
                reason = 'began_too_late' if row.get('begin_too_late_for_deadline') else 'began_no_observed_legal'
            elif row['processing_mid'] is None:
                reason = 'no_observed_processing_tick'
            else:
                reason = 'complete_three_stage'
            row['stage_status'] = reason
            row['actions'] = dict(row['actions'])
            values.append(row)
        return values


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(seed):
    torch.set_num_threads(1)
    plan = checked()
    out = OUT / f'seed_{seed:03d}'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    parent = b.old.read(rp.PLAN)
    src = parent['seeds'][str(seed)]['source']['endpoint']
    trainer = b.native.NativeTrainer.from_checkpoint(src['path'], device='cpu')
    initial = b.state_digest(trainer.semantic_state())
    if initial != src['semantic_hash']:
        raise ValueError('wrong actual endpoint h/world')
    model = load_model(seed, trainer)
    model_hash = b.state_digest(model.state_dict())
    saved = b.old.read(rp.OUT / f'seed_{seed:03d}/result.json')
    records, rows, bindings = [], [], {}
    for condition in ('short', 'long_both'):
        for family in range(4):
            observer = Observer(trainer.world, seed, family, condition)
            original_step = f.FinishController.step
            def step_wrapper(live_model, obs, h, dt, max_step=None):
                return observer.step(original_step, live_model, obs, h, dt, max_step)
            def act_wrapper(live_world, actions):
                return observer.act(live_world, actions)
            with b._preserve_process_state(), patch.object(f.FinishController, 'step', step_wrapper), \
                    patch.object(b.ContextWorld, 'act', act_wrapper):
                record = b.evaluate(model, trainer.world, trainer.h, family=family,
                                    condition=condition, steps=2048, keep_trace=True, device='cpu')
            trace = record.pop('trace')
            old = next(r for r in saved['records'] if r['arm'] == 'own_policy' and
                       r['condition'] == condition and r['family'] == family)
            if normalized(record) != old['record']:
                differing = [k for k in set(record)|set(old['record']) if normalized(record.get(k)) != old['record'].get(k)]
                raise AssertionError('entire original evaluator result mismatch: ' + str(differing))
            if normalized(observer.shadow.result()) != old['shadow']:
                raise AssertionError('original shadow result mismatch')
            fork_rows = observer.finish()
            matured = [r for r in fork_rows if r['matured']]
            ids = sorted((r['case_id'], r['job'], r['request_id']) for r in matured)
            fingerprint = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            if fingerprint != old['shadow']['heldout']['primary_request_set_sha256']:
                raise AssertionError('heldout primary matured identities mismatch')
            if len(matured) != old['shadow']['heldout']['counts']['primary_matured']:
                raise AssertionError('heldout primary matured count mismatch')
            for key, field in (('primary_completed', 'completed'), ('primary_clean_completed', 'clean_completed')):
                if sum(r[field] for r in matured) != old['shadow']['heldout']['counts'].get(key, 0):
                    raise AssertionError('completion request table differs')
            if len(trace) != observer.steps or observer.steps != 2048:
                raise AssertionError('trace/hook length')
            stem = f'{condition}_family_{family}'
            trace_path, raw_path = out / (stem+'.trace.json.gz'), out / (stem+'.raw.pt')
            trace_path.write_bytes(gzip.compress(json.dumps(trace, separators=(',', ':')).encode(), compresslevel=1, mtime=0))
            torch.save(dict(logits=torch.stack(observer.logits), observations=torch.stack(observer.observations),
                            before={k: torch.stack(v) for k, v in observer.raw.items()},
                            protocol_hash=plan['protocol_hash'], action_hash=record['action_hash']), raw_path)
            for path in (trace_path, raw_path):
                bindings[str(path)] = b.old.sha(path)
            rows.extend(fork_rows)
            records.append(dict(condition=condition, family=family, evaluator_record_exact=True,
                original_shadow_exact=True, action_hash=record['action_hash'],
                request_set_sha256=fingerprint, matured_requests=len(matured), issued_primary_requests=len(fork_rows),
                record=record, shadow=observer.shadow.result()))
            print(seed, condition, family, len(matured), 'exact', flush=True)
    if initial != b.state_digest(trainer.semantic_state()) or model_hash != b.state_digest(model.state_dict()):
        raise AssertionError('source/model mutated')
    request_path = out / 'requests.jsonl'
    with request_path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
    flat = []
    for row in rows:
        if row['matured']:
            for phase in STAGES:
                point = row[phase]
                if point is not None:
                    flat.append({k: row[k] for k in ('seed', 'condition', 'fork_family', 'case_id', 'actor_id', 'job', 'request_id', 'request_family', 'raw_payload', 'mapped_payload', 'version')} |
                        dict(phase=phase, **{k: point[k] for k in ('tick', 'time', 'task_age', 'processing_age', 'processing_fraction', 'correct', 'version_correct', 'payload_correct', 'correct_rank', 'margin', 'action', 'result')},
                             finish_order=json.dumps(point['finish_order']), finish_scores=json.dumps(point['finish_scores'])))
    write_csv(out/'stage_samples.csv', flat)
    bindings[str(request_path)] = b.old.sha(request_path)
    bindings[str(out/'stage_samples.csv')] = b.old.sha(out/'stage_samples.csv')
    checked()
    result = normalized(dict(format='natural-content-timing-result-v1', seed=seed,
        protocol_hash=plan['protocol_hash'], source_semantic_hash=initial, source_unchanged=True,
        model_state_hash=model_hash, records=records, bindings=bindings, seconds=time.perf_counter()-started))
    result['result_hash'] = b.state_digest(result)
    b.old.write(out/'result.json', result)


def aggregate(rows, **identity):
    mature = [r for r in rows if r['matured']]
    result = dict(**identity, issued=len(rows), censored=len(rows)-len(mature), matured=len(mature),
        completed=sum(r['completed'] for r in mature), clean_completed=sum(r['clean_completed'] for r in mature))
    for label in ('never_began', 'began_too_late', 'began_no_observed_legal', 'no_observed_processing_tick', 'cue_unobserved', 'complete_three_stage'):
        result[label] = sum(r['stage_status'] == label for r in mature)
    for phase in STAGES:
        selected = [r[phase] for r in mature if r[phase] is not None]
        result[phase+'_n'] = len(selected)
        for measure in ('correct', 'version_correct', 'payload_correct'):
            count = sum(p[measure] for p in selected)
            result[phase+'_'+measure+'_count'] = count
            result[phase+'_'+measure+'_rate'] = count/len(selected) if selected else None
        result[phase+'_mean_correct_rank'] = statistics.mean(p['correct_rank'] for p in selected) if selected else None
        result[phase+'_mean_margin'] = statistics.mean(p['margin'] for p in selected) if selected else None
        result[phase+'_actual_wait'] = sum(p['action'] == 0 for p in selected)
        result[phase+'_actual_begin'] = sum(p['action'] in (1, 2) for p in selected)
        result[phase+'_actual_finish'] = sum(p['action'] >= 3 for p in selected)
        result[phase+'_prior_rejected_requests'] = sum(p['prior_request_actions'].get('rejected', 0) > 0 for p in selected)
    for early, late in (('cue', 'processing_mid'), ('cue', 'first_legal'), ('processing_mid', 'first_legal')):
        pair = [r for r in mature if r[early] is not None and r[late] is not None]
        prefix = early+'_to_'+late
        result[prefix+'_n'] = len(pair)
        for measure in ('correct', 'version_correct', 'payload_correct'):
            for label, a, z in (('both_correct', True, True), ('lost', True, False), ('gained', False, True), ('both_wrong', False, False)):
                result[prefix+'_'+measure+'_'+label] = sum(r[early][measure] == a and r[late][measure] == z for r in pair)
            for phase in (early, late):
                result[prefix+'_'+measure+'_'+phase+'_rate'] = sum(r[phase][measure] for r in pair)/len(pair) if pair else None
    complete = [r for r in mature if all(r[p] is not None for p in STAGES)]
    for pattern in range(8):
        name = format(pattern, '03b')
        result['three_stage_correct_'+name] = sum(''.join(str(int(r[p]['correct'])) for p in STAGES) == name for r in complete)
    return result


def summarize():
    plan = checked()
    all_rows, bindings = [], {}
    for seed in range(3):
        path = OUT/f'seed_{seed:03d}/result.json'
        result = b.old.read(path)
        if result['protocol_hash'] != plan['protocol_hash'] or result['result_hash'] != b.state_digest({k: v for k, v in result.items() if k != 'result_hash'}):
            raise ValueError('foreign natural result/digest')
        for bound, sha in result['bindings'].items():
            if b.old.sha(bound) != sha:
                raise ValueError('raw artifact changed')
        bindings[str(path)] = b.old.sha(path)
        all_rows.extend(json.loads(line) for line in (path.parent/'requests.jsonl').read_text(encoding='utf-8').splitlines())
    per_seed, per_fork = [], []
    for seed in range(3):
        for condition in ('short', 'long_both'):
            selected = [r for r in all_rows if r['seed'] == seed and r['condition'] == condition]
            per_seed.append(aggregate(selected, seed=seed, condition=condition))
            for family in range(4):
                per_fork.append(aggregate([r for r in selected if r['fork_family'] == family],
                                          seed=seed, condition=condition, family=family))
    seed_rates = {}
    for condition in ('short', 'long_both'):
        selected = [r for r in per_seed if r['condition'] == condition]
        seed_rates[condition] = {k: dict(mean=statistics.mean(r[k] for r in selected),
            sample_sd=statistics.stdev(r[k] for r in selected)) for k in selected[0]
            if k.endswith('_rate') and all(r[k] is not None for r in selected)}
    write_csv(OUT/'per_seed.csv', per_seed)
    write_csv(OUT/'per_fork.csv', per_fork)
    result = normalized(dict(format='natural-content-timing-summary-v1', protocol_hash=plan['protocol_hash'],
        audit='passed', results=bindings, per_seed=per_seed, per_fork=per_fork, seed_rates=seed_rates,
        limits=plan['limits'], checks=['24 entire unchanged evaluator records exactly reproduce historical own_policy',
            '24 original shadow results exact', 'heldout primary matured request identities and completed/strict results exact',
            'source h/world/model unchanged', 'processing origin independently checked against ready_time-duration',
            'all stage rates and pair transitions count distinct matched matured requests', 'independent replication unit remains3 seeds']))
    result['summary_hash'] = b.state_digest(result)
    b.old.write(OUT/'summary.json', result)
    return result


def launch():
    checked()
    def one(seed):
        with (OUT/f'seed_{seed:03d}.log').open('w') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'experiments.content_timing.natural', 'run', '--seed', str(seed)],
                cwd=b.ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            return dict(seed=seed, pid=process.pid, returncode=process.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes = list(pool.map(one, range(3)))
    b.old.write(OUT/'exit_codes.json', codes)
    if any(c['returncode'] for c in codes):
        raise RuntimeError(codes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'launch', 'summarize'))
    parser.add_argument('--seed', type=int, choices=range(3))
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare()['protocol_hash'])
    elif args.command == 'run':
        run(args.seed)
    elif args.command == 'launch':
        launch()
    else:
        print(json.dumps(summarize()['seed_rates'], indent=2))
