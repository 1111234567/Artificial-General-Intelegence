"""Serialization-only correction to transfer's continuous replay verification.

The original frozen transfer.py/protocol remain untouched. Its initial replay
matched all2048 logits/actions/hashes in each seed's first fork, then stopped
because its full-action digest omitted the evaluator's format prefix.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.content_timing import transfer as t
from experiments.lifelong.evaluate import TRACE_HASH_VERSION

torch, b, n, rp = t.torch, t.b, t.n, t.rp
AMENDMENT = t.OUT/'replay_correction.json'


def declare():
    plan = t.checked()
    if AMENDMENT.exists():
        raise FileExistsError('immutable correction exists')
    value = dict(format='transfer-replay-serialization-correction-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), protocol_hash=plan['protocol_hash'],
        reason='Original replay incorrectly initialized action digest with empty SHA256 rather than lifelong-own-trace-v1/actions prefix. First2048 steps of all3 seeds had already passed exact all11 logits, actions and hidden_hash checks, then failed only the final prefixed digest comparison. No matched/probe scores or feature arrays have been viewed and no transfer fitting has run.',
        correction='Initialize digest with the exact unchanged evaluator TRACE_HASH_VERSION/actions prefix. Replay all24 forks again. Preserve every original transfer/natural source and protocol. Resume only empty seed directories created by the failed attempts; no failed artifact is overwritten.',
        source=dict(path=str(Path(__file__).resolve()), sha256=b.old.sha(__file__)),
        old_source=dict(path=str(Path(t.__file__).resolve()), sha256=b.old.sha(t.__file__)),
        failed_exit_codes=dict(path=str(t.OUT/'replay_exit_codes.json'), sha256=b.old.sha(t.OUT/'replay_exit_codes.json')),
        unchanged='Actual source h/model, public observations,24 non-input features,train/test folds,matched training rule,duration map,test identities and metrics all unchanged.')
    value['amendment_hash'] = b.state_digest(value)
    b.old.write(AMENDMENT, value)
    return value


def checked():
    t.checked()
    value = b.old.read(AMENDMENT)
    if value['amendment_hash'] != b.state_digest({k: v for k, v in value.items() if k != 'amendment_hash'}):
        raise ValueError('correction digest')
    for key in ('source', 'old_source', 'failed_exit_codes'):
        if b.old.sha(value[key]['path']) != value[key]['sha256']:
            raise ValueError('correction source/evidence changed')
    return value


def replay(seed):
    amendment = checked()
    plan = t.checked()
    torch.set_num_threads(1)
    out = t.OUT/f'seed_{seed:03d}'
    out.mkdir(exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('refuse to overwrite any previous feature/result')
    started = time.perf_counter()
    src = b.old.read(rp.PLAN)['seeds'][str(seed)]['source']['endpoint']
    trainer = b.native.NativeTrainer.from_checkpoint(src['path'], device='cpu')
    initial = b.state_digest(trainer.semantic_state())
    if initial != src['semantic_hash']:
        raise ValueError('wrong actual endpoint')
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
            state, features = trainer.h.clone(), []
            digest = hashlib.sha256(f'{TRACE_HASH_VERSION}/actions'.encode())
            with torch.inference_mode():
                for tick, obs in enumerate(saved['observations']):
                    logits, state = model.step(obs, state, record['dt'])
                    if not torch.equal(logits, saved['logits'][tick]):
                        raise AssertionError('all11 replay logits differ')
                    actions = logits.argmax(-1)
                    if actions.tolist() != trace[tick]['executed_action']:
                        raise AssertionError('replay actions differ')
                    if b.state_digest({'hidden': state}) != trace[tick]['hidden_hash']:
                        raise AssertionError('replay entire hidden differs')
                    digest.update(actions.numpy().astype(t.np.int64).tobytes())
                    features.append(state.index_select(-1, model.begin_head.indices).clone())
            if digest.hexdigest() != record['action_hash'] or len(features) != 2048:
                raise AssertionError('prefixed replay action hash/length differs')
            path = out/(stem+'.features.pt')
            torch.save(dict(x24=torch.stack(features), indices=model.begin_head.indices,
                protocol_hash=plan['protocol_hash'], natural_raw_sha256=b.old.sha(raw_path),
                action_hash=digest.hexdigest(), start_hidden_hash=record['start_hidden_hash']), path)
            artifacts[str(path)] = b.old.sha(path)
            checks.append(dict(condition=condition, family=family, steps=2048,
                all11_logits_exact=True, all_actions_exact=True, every_hidden_hash_exact=True,
                action_hash=digest.hexdigest(), source_raw_sha256=b.old.sha(raw_path)))
            print(seed, stem, 'continuous exact', flush=True)
    if initial != b.state_digest(trainer.semantic_state()) or model_hash != b.state_digest(model.state_dict()):
        raise AssertionError('replay mutated original source')
    checked()
    value = n.normalized(dict(format='natural-transfer-continuous-replay-v1', seed=seed,
        protocol_hash=plan['protocol_hash'], serialization_correction_hash=amendment['amendment_hash'],
        checks=checks, artifacts=artifacts, source_unchanged=True, source_semantic_hash=initial,
        seconds=time.perf_counter()-started))
    value['result_hash'] = b.state_digest(value)
    b.old.write(out/'replay.json', value)


def launch():
    checked()
    def one(seed):
        with (t.OUT/f'replay_corrected_seed_{seed:03d}.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'experiments.content_timing.transfer_replay', 'run', str(seed)],
                cwd=b.ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            return dict(seed=seed, pid=process.pid, returncode=process.wait())
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes = list(pool.map(one, range(3)))
    b.old.write(t.OUT/'replay_corrected_exit_codes.json', codes)
    if any(c['returncode'] for c in codes):
        raise RuntimeError(codes)


if __name__ == '__main__':
    if sys.argv[1:] == ['declare']:
        print(declare()['amendment_hash'])
    elif sys.argv[1:] == ['launch']:
        launch()
    elif len(sys.argv) == 3 and sys.argv[1] == 'run':
        replay(int(sys.argv[2]))
    else:
        raise SystemExit('usage: transfer_replay declare|launch|run SEED')
