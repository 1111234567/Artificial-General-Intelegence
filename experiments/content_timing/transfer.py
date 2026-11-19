"""Frozen paired-probe transfer to unchanged natural autonomous trajectories.

No natural row enters fitting, normalization, parameter selection or thresholds.
The primary test population is the same complete-three-stage matured requests.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.action_return import readout as b, refit_policy as rp
from experiments.content_timing import natural as n, probe

OUT = b.ROOT / 'data/content-timing/transfer-v1'
PLAN = OUT / 'protocol.json'
MATCHED = b.ROOT / 'data/content-timing/matched-v1'
PHASES = (('cue', 'cue'), ('processing_mid', 'mid'), ('first_legal', 'first_legal'))


def prepare():
    if PLAN.exists():
        raise FileExistsError('immutable transfer protocol exists')
    natural_plan = n.checked()
    # Only metadata and file hashes are read here. No matched/probe scores,
    # matched features or labels are read before this transfer declaration.
    paths = [Path(__file__).resolve(), Path(n.__file__).resolve(), Path(probe.__file__).resolve(), probe.PLAN]
    sources = {str(p): b.old.sha(p) for p in paths}
    inputs = {str(n.PLAN): b.old.sha(n.PLAN), str(n.OUT/'summary.json'): b.old.sha(n.OUT/'summary.json'),
              str(rp.PLAN): b.old.sha(rp.PLAN)}
    for seed in range(3):
        folder = n.OUT/f'seed_{seed:03d}'
        result = b.old.read(folder/'result.json')
        if result['protocol_hash'] != natural_plan['protocol_hash']:
            raise ValueError('foreign natural result')
        inputs[str(folder/'result.json')] = b.old.sha(folder/'result.json')
        inputs.update(result['bindings'])
    for path in (MATCHED/'protocol.json', probe.OUTPUT/'protocol.json'):
        if path.exists():
            inputs[str(path)] = b.old.sha(path)
    plan = dict(format='paired-to-natural-transfer-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen after natural action-head content results but before viewing any matched/probe accuracy, scores or matched feature/label arrays. The narrow transfer is chosen to distinguish readability only in locked paired forks from readability in real autonomous trajectories. No fitting on natural labels.',
        source_files=sources, input_files=inputs, natural_protocol_hash=natural_plan['protocol_hash'],
        seeds=[0, 1, 2], conditions=['short', 'long_both'],
        duration_map={'short': 2, 'long_both': 8}, phases=[list(pair) for pair in PHASES],
        replay='For each of the24 saved full8-actor own_policy forks, start at the seed actual147456 h, consume every saved actual public observation continuously with its original elapsed dt, and run the exact saved own_policy model. Never reset by request, stage or actor. Exact equality of all final11 logits and actions plus every existing trace hidden_hash is required. Preserve source state. Extract only24 non-input h coordinates from this replay.',
        training='Call probe.load_matched on the3 fixed seed records.pt artifacts when complete, and probe.fit_scores directly, with no adaptation. For each seed/duration/job/fold/stage and raw/mapped label, train exclusively on matched factorial data at that same stage. x24 and standardization, unpenalized intercept, ridge=.001*n and4-way one-hot targets exactly follow probe.py. No natural labels are passed into fitting or scaling.',
        folds='Same separate-job cross-actor folds as probe.py: predict natural actors4,5 from matched actors6,7 and natural6,7 from matched4,5. Retain all family/raw variants of a matched anchor together; no cross-fold anchor split. All seeds remain separate fits.',
        test='Primary: identical matured decoder-heldout primary request rows with cue,processing_mid,first_legal present in natural-v1. Each request contributes one heldout prediction per stage and label. Diagnostics: all matured primary rows at every available stage, explicitly different denominators. Natural raw/mapped targets score predictions only after fitting.',
        outputs='Per prediction exact identity, fold, phase, label,4 scores,choice,target; per fold and per seed condition/phase/label population counts,correct,total,4x4 confusion,true-class denominators/correct/recall,macro recall and test majority-class proportion. All four families aggregated inside each seed and retained per-family diagnostics; mean and sampleSD across3 seed rates only.',
        matched_artifacts='Exact3 fixed paths matched-v1/seed_000..002/records.pt. Their SHA256s and matched protocol hash are recorded at application time; fail if expected durations, phases,actors,jobs or complete factorial schema are absent. No result-directed alternative source.',
        exclusions='No retraining action head or backbone, no natural readout fit, no cross-phase transfer arm, no probe-score/threshold/hyperparameter selection, no new control policy or model action execution.',
        limits=['Transfer asks access by a paired-data linear decoder, not endogenous action-head access.',
                'A failure can reflect train/test distribution mismatch; it does not prove information absence.',
                'Natural recurrent features include true own actions, feedback and other-slot cues.',
                'Natural duration is a distribution while paired training uses fixed2 or8; this extrapolation is fixed before scores.',
                'The same complete-three-stage subset is policy-dependent; report all missing-stage counts from natural-v1.',
                'Actors4..7 were held out of action-head fitting, not the original backbone.'],
        runtime=dict(workers=3, cpu_threads=1, dtype='float64', device='cpu'))
    plan = n.normalized(plan)
    plan['protocol_hash'] = b.state_digest(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    b.old.write(PLAN, plan)
    return plan


def checked():
    plan = b.old.read(PLAN)
    if plan['protocol_hash'] != b.state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'}):
        raise ValueError('transfer protocol digest')
    for group in ('source_files', 'input_files'):
        for path, sha in plan[group].items():
            if b.old.sha(path) != sha:
                raise ValueError('bound transfer source/input changed: ' + path)
    return plan


def replay(seed):
    torch.set_num_threads(1)
    plan = checked()
    out = OUT/f'seed_{seed:03d}'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    parent = b.old.read(rp.PLAN)
    src = parent['seeds'][str(seed)]['source']['endpoint']
    trainer = b.native.NativeTrainer.from_checkpoint(src['path'], device='cpu')
    original_semantics = b.state_digest(trainer.semantic_state())
    if original_semantics != src['semantic_hash']:
        raise ValueError('wrong real147456 complete endpoint')
    model = n.load_model(seed, trainer)
    model_hash = b.state_digest(model.state_dict())
    natural_result = b.old.read(n.OUT/f'seed_{seed:03d}/result.json')
    checks, artifacts = [], {}
    for condition in ('short', 'long_both'):
        for family in range(4):
            stem = f'{condition}_family_{family}'
            source = n.OUT/f'seed_{seed:03d}'
            raw_path, trace_path = source/(stem+'.raw.pt'), source/(stem+'.trace.json.gz')
            saved = torch.load(raw_path, map_location='cpu', weights_only=False)
            trace = json.loads(gzip.decompress(trace_path.read_bytes()))
            record = next(r['record'] for r in natural_result['records'] if r['condition'] == condition and r['family'] == family)
            state = trainer.h.clone()
            features = []
            digest = hashlib.sha256()
            with torch.inference_mode():
                for t, obs in enumerate(saved['observations']):
                    logits, state = model.step(obs, state, record['dt'])
                    if not torch.equal(logits, saved['logits'][t]):
                        raise AssertionError(f'all11 logits differ on replay {seed}/{stem}/{t}')
                    actions = logits.argmax(-1)
                    if actions.tolist() != trace[t]['executed_action']:
                        raise AssertionError('continuous replay action differs')
                    if b.state_digest({'hidden': state}) != trace[t]['hidden_hash']:
                        raise AssertionError('continuous replay full hidden differs')
                    digest.update(actions.numpy().astype(np.int64).tobytes())
                    features.append(state.index_select(-1, model.begin_head.indices).clone())
            if digest.hexdigest() != record['action_hash'] or len(features) != 2048:
                raise AssertionError('replayed entire action hash/length differs')
            path = out/(stem+'.features.pt')
            torch.save(dict(x24=torch.stack(features), indices=model.begin_head.indices,
                protocol_hash=plan['protocol_hash'], natural_raw_sha256=b.old.sha(raw_path),
                action_hash=digest.hexdigest(), start_hidden_hash=record['start_hidden_hash']), path)
            artifacts[str(path)] = b.old.sha(path)
            checks.append(dict(condition=condition, family=family, steps=2048,
                all11_logits_exact=True, all_actions_exact=True, every_hidden_hash_exact=True,
                action_hash=digest.hexdigest(), source_raw_sha256=b.old.sha(raw_path)))
            print(seed, stem, 'continuous exact', flush=True)
    if original_semantics != b.state_digest(trainer.semantic_state()) or model_hash != b.state_digest(model.state_dict()):
        raise AssertionError('transfer replay mutated original endpoint/model')
    result = n.normalized(dict(format='natural-transfer-continuous-replay-v1', seed=seed,
        protocol_hash=plan['protocol_hash'], checks=checks, artifacts=artifacts,
        source_unchanged=True, source_semantic_hash=original_semantics, seconds=time.perf_counter()-started))
    result['result_hash'] = b.state_digest(result)
    b.old.write(out/'replay.json', result)


def matrix_record(predictions, **identity):
    true = np.array([p['target'] for p in predictions], dtype=np.int64)
    pred = np.array([p['choice'] for p in predictions], dtype=np.int64)
    matrix = np.bincount(4*true+pred, minlength=16).reshape(4, 4)
    denominators = matrix.sum(1)
    correct = matrix.diagonal()
    recalls = [float(c/d) if d else None for c, d in zip(correct, denominators)]
    return identity | dict(correct=int(correct.sum()), total=len(predictions),
        accuracy=float(correct.sum()/len(predictions)) if predictions else None,
        confusion_true_by_pred=matrix.tolist(), class_denominators=denominators.tolist(),
        class_correct=correct.tolist(), class_recall=recalls,
        macro_recall=statistics.mean(r for r in recalls if r is not None) if predictions else None,
        observed_classes=int((denominators>0).sum()),
        majority_class_count=int(denominators.max()),
        majority_class_proportion=float(denominators.max()/len(predictions)) if predictions else None)


def apply(seed):
    plan = checked()
    out = OUT/f'seed_{seed:03d}'
    result_path = out/'result.json'
    if result_path.exists():
        raise FileExistsError('immutable transfer seed result exists')
    replay_result = b.old.read(out/'replay.json')
    if replay_result['protocol_hash'] != plan['protocol_hash']:
        raise ValueError('foreign replay')
    for path, sha in replay_result['artifacts'].items():
        if b.old.sha(path) != sha:
            raise ValueError('continuous features changed')
    path = MATCHED/f'seed_{seed:03d}/records.pt'
    anchors, states, durations, matched_protocol_hash = probe.load_matched(path)
    jobs = np.array([a['job'] for a in anchors])
    lanes = np.array([a['lane'] for a in anchors])
    families = np.array([a['family'] for a in anchors])
    raw_targets = np.broadcast_to(np.arange(4), (len(anchors), 4))
    mapped_targets = np.array(probe.FAMILY_MAPPINGS)[families]
    rows = [json.loads(line) for line in (n.OUT/f'seed_{seed:03d}/requests.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['matured']]
    feature_cache = {}
    # Index observations by saved absolute tick; each fork begins immediately
    # after the inherited endpoint action and never resets within the replay.
    natural_result = b.old.read(n.OUT/f'seed_{seed:03d}/result.json')
    starts = {(r['condition'], r['family']): r['record']['fork_tick']+1 for r in natural_result['records']}
    for condition in ('short', 'long_both'):
        for family in range(4):
            feature_cache[(condition, family)] = torch.load(out/f'{condition}_family_{family}.features.pt',
                map_location='cpu', weights_only=False)['x24'].numpy()
    predictions, folds, fits = [], [], []
    for condition, duration in (('short', 2), ('long_both', 8)):
        if duration not in durations:
            raise ValueError('predeclared paired training duration absent')
        di = durations.index(duration)
        for job in range(2):
            for heldout in ((4, 5), (6, 7)):
                training = (jobs == job) & ~np.isin(lanes, heldout)
                if set(families[training]) != set(range(4)):
                    raise AssertionError('paired training lacks a family')
                for natural_phase, probe_phase in PHASES:
                    pi = probe.PHASES.index(probe_phase)
                    x = states[training, di, :, pi, :].reshape(-1, 24)
                    test_rows = [r for r in rows if r['condition'] == condition and r['job'] == job and
                                 r['lane'] in heldout and r[natural_phase] is not None]
                    xt = np.array([feature_cache[(condition, r['fork_family'])]
                        [r[natural_phase]['tick']-starts[(condition, r['fork_family'])], r['lane']] for r in test_rows], dtype=np.float64)
                    if not test_rows:
                        xt = np.empty((0, 24), dtype=np.float64)
                    for label, paired_targets in (('raw', raw_targets), ('mapped', mapped_targets)):
                        y = paired_targets[training].ravel()
                        scores = probe.fit_scores(x, y, xt)
                        # Natural targets are read only after scores are fixed.
                        current = []
                        for r, value in zip(test_rows, scores):
                            current.append(dict(seed=seed, condition=condition, duration=duration, job=job,
                                test_lanes=list(heldout), lane=r['lane'], actor_id=r['actor_id'],
                                fork_family=r['fork_family'], request_family=r['request_family'],
                                case_id=r['case_id'], request_id=r['request_id'], phase=natural_phase,
                                label=label, target=r['raw_payload' if label == 'raw' else 'mapped_payload'],
                                choice=int(value.argmax()), scores=value.tolist(),
                                complete_three_stage=all(r[p] is not None for p in n.STAGES)))
                        predictions.extend(current)
                        fit = dict(seed=seed, condition=condition, duration=duration, job=job, test_lanes=list(heldout),
                            phase=natural_phase, label=label, training_anchors=int(training.sum()),
                            training_positions=len(x), training_class_counts=np.bincount(y, minlength=4).tolist(),
                            training_feature_hash=hashlib.sha256(x.tobytes()).hexdigest(),
                            training_target_hash=hashlib.sha256(y.tobytes()).hexdigest())
                        fits.append(fit)
                        for population in ('complete_three_stage', 'all_available'):
                            selected = current if population == 'all_available' else [p for p in current if p['complete_three_stage']]
                            folds.append(matrix_record(selected, **fit, population=population))
    seed_rows = []
    family_rows = []
    for condition in ('short', 'long_both'):
        for phase, _ in PHASES:
            for label in ('raw', 'mapped'):
                for population in ('complete_three_stage', 'all_available'):
                    selected = [p for p in predictions if p['condition'] == condition and p['phase'] == phase and
                                p['label'] == label and (population == 'all_available' or p['complete_three_stage'])]
                    keys = [(p['fork_family'], p['case_id'], p['job'], p['request_id']) for p in selected]
                    if len(keys) != len(set(keys)):
                        raise AssertionError('same natural request duplicated across folds')
                    seed_rows.append(matrix_record(selected, seed=seed, condition=condition,
                        phase=phase, label=label, population=population))
                    for family in range(4):
                        family_rows.append(matrix_record([p for p in selected if p['fork_family'] == family],
                            seed=seed, condition=condition, phase=phase, label=label,
                            population=population, family=family))
    pred_path = out/'predictions.jsonl'
    with pred_path.open('x', encoding='utf-8') as stream:
        for value in predictions:
            stream.write(json.dumps(value, separators=(',', ':'))+'\n')
    checked()
    result = n.normalized(dict(format='paired-to-natural-transfer-result-v1', seed=seed,
        protocol_hash=plan['protocol_hash'], matched_protocol_hash=matched_protocol_hash,
        matched_artifact=dict(path=str(path), sha256=b.old.sha(path)),
        replay_result=dict(path=str(out/'replay.json'), sha256=b.old.sha(out/'replay.json')),
        prediction_artifact=dict(path=str(pred_path), sha256=b.old.sha(pred_path)),
        fits=fits, folds=folds, per_seed=seed_rows, per_family=family_rows))
    result['result_hash'] = b.state_digest(result)
    b.old.write(result_path, result)
    print(seed, 'transfer saved', len(predictions), flush=True)


def summarize():
    plan = checked()
    results = []
    bindings = {}
    for seed in range(3):
        path = OUT/f'seed_{seed:03d}/result.json'
        value = b.old.read(path)
        if value['protocol_hash'] != plan['protocol_hash'] or value['result_hash'] != b.state_digest({k: v for k, v in value.items() if k != 'result_hash'}):
            raise ValueError('foreign transfer result')
        for key in ('matched_artifact', 'prediction_artifact', 'replay_result'):
            if b.old.sha(value[key]['path']) != value[key]['sha256']:
                raise ValueError('transfer input/result changed')
        results.append(value)
        bindings[str(path)] = b.old.sha(path)
    all_seed = [r for v in results for r in v['per_seed']]
    all_fold = [r for v in results for r in v['folds']]
    all_family = [r for v in results for r in v['per_family']]
    aggregate = []
    for condition in ('short', 'long_both'):
        for phase, _ in PHASES:
            for label in ('raw', 'mapped'):
                for population in ('complete_three_stage', 'all_available'):
                    selected = [r for r in all_seed if r['condition'] == condition and r['phase'] == phase and r['label'] == label and r['population'] == population]
                    aggregate.append(dict(condition=condition, phase=phase, label=label, population=population,
                        **{metric+'_'+stat: operation([r[metric] for r in selected]) for metric in
                           ('accuracy', 'macro_recall', 'majority_class_proportion') for stat, operation in
                           (('mean', statistics.mean), ('sample_sd', statistics.stdev))}))
    for name, rows in (('per_seed.csv', all_seed), ('per_fold.csv', all_fold),
                       ('per_family.csv', all_family), ('aggregate.csv', aggregate)):
        n.write_csv(OUT/name, [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows])
    result = n.normalized(dict(format='paired-to-natural-transfer-summary-v1', audit='passed',
        protocol_hash=plan['protocol_hash'], results=bindings, per_seed=all_seed, per_fold=all_fold,
        per_family=all_family, aggregate=aggregate, limits=plan['limits']))
    result['summary_hash'] = b.state_digest(result)
    b.old.write(OUT/'summary.json', result)
    return result


def launch(command):
    checked()
    def one(seed):
        with (OUT/f'{command}_seed_{seed:03d}.log').open('w') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'experiments.content_timing.transfer', command, '--seed', str(seed)],
                cwd=b.ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            return dict(seed=seed, pid=process.pid, returncode=process.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes = list(pool.map(one, range(3)))
    b.old.write(OUT/(command+'_exit_codes.json'), codes)
    if any(v['returncode'] for v in codes):
        raise RuntimeError(codes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'replay', 'apply', 'launch_replay', 'launch_apply', 'summarize'))
    parser.add_argument('--seed', type=int, choices=range(3))
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare()['protocol_hash'])
    elif args.command == 'replay':
        replay(args.seed)
    elif args.command == 'apply':
        apply(args.seed)
    elif args.command == 'launch_replay':
        launch('replay')
    elif args.command == 'launch_apply':
        launch('apply')
    else:
        print(summarize()['summary_hash'])
