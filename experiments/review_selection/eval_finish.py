"""Complete all frozen bindings, parallel independent per-seed evidence audits."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from experiments.review_selection import eval_review as e, eval_runtime, eval_report


def audit_seed(seed):
    torch.set_num_threads(1)
    eval_runtime.install()
    original_records, original_write = e.records, e.write
    all_rows = original_records()
    chosen = [r for r in all_rows if r['seed'] == seed]
    expected = {(a, t, q) for a in e.ARMS for t in e.SNAPSHOTS for q in e.scenarios(t)}
    assert len(chosen) == 56
    assert {(r['arm'], r['snapshot'], r['scenario']) for r in chosen} == expected
    training_plan = e.read(e.TRAIN/'protocol.json')
    assert training_plan['protocol_hash'] == e.state_digest({k:v for k,v in training_plan.items() if k != 'protocol_hash'})
    origin_sha = e.sha(e.checkpoint_path(seed, 'no_review', 'B000'))
    bank_sha = e.sha(e.ROOT/f'data/review-selection/bank-v1/seed_{seed:03d}/bank.pt')
    checkpoints = {}
    for row in chosen:
        path = row['checkpoint_path']
        if path not in checkpoints:
            checkpoints[path] = torch.load(path, map_location='cpu', weights_only=False)
        cp = checkpoints[path]
        arm, snapshot = row['arm'], row['snapshot']
        assert cp['transfer']['arm'] == ('related' if arm == 'no_review' else arm)
        assert cp['stage'] == 'B'
        assert cp['round'] == (1 if snapshot.startswith('Bfirst') else int(snapshot[1:]))
        if arm != 'no_review':
            assert cp['review_selection']['protocol_hash'] == training_plan['protocol_hash']
            assert cp['review_selection']['original_B000_sha256'] == origin_sha
            assert cp['review_selection']['bank_sha256'] == bank_sha
            if snapshot == 'B000':
                assert row['actual_unique_B_NEW'] == 0
            elif snapshot.startswith('Bfirst'):
                if 'B_actual_NEW_including_boundary' in cp['work']:
                    assert cp['work']['B_actual_NEW_including_boundary'] == row['actual_unique_B_NEW']
            else:
                assert cp['work']['B_actual_NEW_including_boundary'] == row['actual_unique_B_NEW']
    e.records = lambda: chosen
    def write(path, value):
        if Path(path) == e.OUT/'audit.json':
            return
        original_write(path, value)
    e.write = write
    report = e.audit()
    report.update(scope='all declared bindings in one seed', seed=seed, seed_complete=True,
                  expected_seed_records=56, raw_audit_source_sha256=e.sha(e.__file__),
                  checkpoint_arm_stage_round_binding=True, training_protocol_hash=training_plan['protocol_hash'],
                  bank_sha256=bank_sha, original_B000_sha256=origin_sha)
    original_write(e.OUT/f'audit_seed_{seed:03d}.json', report)
    return report


def worker(seed):
    path = e.OUT/f'audit_seed_{seed:03d}.log'
    with path.open('w', encoding='utf-8') as log:
        completed = subprocess.run([sys.executable, '-u', '-m', 'experiments.review_selection.eval_finish', 'audit-seed', '--seed', str(seed)],
                                   cwd=e.ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, ('audit shard failed', seed, str(path))
    return e.read(e.OUT/f'audit_seed_{seed:03d}.json')


def finalize():
    plan = e.checked()
    wanted = [e.result_path(s, a, t, q) for s in range(3) for a in e.ARMS for t in e.SNAPSHOTS for q in e.scenarios(t)]
    assert len(wanted) == 168
    while not all(p.exists() for p in wanted):
        time.sleep(5)
    with ThreadPoolExecutor(max_workers=3) as pool:
        shards = list(pool.map(worker, range(3)))
    for seed, shard in enumerate(shards):
        assert shard['seed'] == seed and shard['verified'] and shard['seed_complete']
        assert shard['records'] == 56 and shard['matched_external_groups'] == 3
        assert shard['protocol_hash'] == plan['protocol_hash']
    rows = e.records()
    assert len(rows) == 168
    assert {(r['seed'], r['arm'], r['snapshot'], r['scenario']) for r in rows} == {
        (s, a, t, q) for s in range(3) for a in e.ARMS for t in e.SNAPSHOTS for q in e.scenarios(t)}
    audit = dict(format=e.FORMAT, protocol_hash=plan['protocol_hash'], verified=True, complete=True,
        records=168, matched_external_groups=9, unique_evidence_forks=sum(s['unique_evidence_forks'] for s in shards),
        checks=shards[0]['checks'], shard_reports={str(e.OUT/f'audit_seed_{s:03d}.json'):e.sha(e.OUT/f'audit_seed_{s:03d}.json') for s in range(3)},
        audit_finalizer_source_sha256=e.sha(__file__), serialization_adapter_manifest_sha256=e.sha(e.OUT/'serialization-adapter.json'))
    e.write(e.OUT/'audit.json', audit)
    eval_report.run()
    summary = e.read(e.OUT/'summary.json')
    assert summary['complete'] and summary['records'] == 168
    value = dict(complete=True, format=e.FORMAT, records=168, protocol_hash=plan['protocol_hash'],
        finalizer_source_sha256=e.sha(__file__), checkpoint_binding_count=len(rows),
        artifacts={str(e.OUT/name):e.sha(e.OUT/name) for name in ('audit.json','summary.json','curves.csv','curves.png','report.md','report-binding.json','serialization-adapter.json')})
    e.write(e.OUT/'completion.json', value)
    print(json.dumps(dict(event='review_evaluation_complete', records=168, protocol_hash=plan['protocol_hash'])), flush=True)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('finalize','audit-seed'))
    p.add_argument('--seed', type=int, choices=range(3))
    args = p.parse_args()
    if args.command == 'audit-seed':
        print(json.dumps(audit_seed(args.seed)), flush=True)
    else:
        finalize()
