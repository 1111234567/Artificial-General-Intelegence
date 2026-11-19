"""One explicitly post-hoc static interior-FINISH evaluation; no sweep."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from experiments.finish_position import eval_finish as core

ROOT = core.ROOT
OUT = ROOT / 'data/finish-position/eval-interior-v1'
TRAIN = ROOT / 'data/finish-position/interior-v1'
PLAN = OUT / 'protocol.json'
MAIN = ROOT / 'data/finish-position/eval-v1'
FORMAT = 'finish-position-static-interior-autonomous-supplement-v1'
STAGES = ('B000', 'B001', 'B002', 'B004', 'B008')
ORIGINAL_BIND = core.bind


def scenarios(stage):
    return ('B', 'mixed', 'history') if stage == 'B008' else ('B',)


def expected_keys():
    return {(s, 'interior', t, q) for s in range(3) for t in STAGES for q in scenarios(t)}


def reference_rows():
    rows = []
    for seed in range(3):
        for stage in STAGES:
            for scenario in scenarios(stage):
                for arm, folder in (('first', MAIN), ('midpoint', MAIN), ('last', core.LAST_EVAL), ('no_review', core.MAIN_EVAL)):
                    rows.append(folder / f'seed_{seed:03d}' / f'{arm}_{stage}_{scenario}.json')
    return rows


def checked():
    plan = core.read(PLAN)
    assert plan['protocol_hash'] == core.state_digest({k: v for k, v in plan.items() if k != 'protocol_hash'})
    core.base.checked()
    for path, digest in plan['frozen_files'].items():
        assert core.sha(path) == digest, ('interior frozen dependency changed', path)
    return plan


def prepare():
    assert not PLAN.exists()
    main = core.checked()
    assert core.read(MAIN / 'completion.json')['complete']
    train = core.read(TRAIN / 'protocol.json')
    files = [Path(__file__).resolve(), Path(__file__).with_name('eval_interior_test.py'),
        Path(core.__file__), Path(core.timing.__file__), Path(core.base.__file__), Path(core.base.original.__file__),
        MAIN / 'protocol.json', MAIN / 'completion.json', MAIN / 'audit.json', MAIN / 'summary.json',
        MAIN / 'reporting-correction.json', TRAIN / 'protocol.json']
    bindings = {}
    for path in reference_rows():
        row = core.read(path)
        files.append(path)
        bindings[str(path)] = dict(sha256=core.sha(path), artifacts=row['artifacts'], checkpoint_path=row['checkpoint_path'], checkpoint_sha256=row['checkpoint_sha256'])
    for seed in range(3):
        files.append(core.base.OUT / f'common_seed_{seed:03d}_related.pt')
        for scenario in ('B', 'mixed', 'history'):
            ref = core.base.reference(seed, 'related', scenario)
            files += [core.base.OUT / f'reference_seed_{seed:03d}_related_{scenario}.json', Path(ref['external_artifact'])]
    plan = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Unique post-hoc mechanistic supplement, selected after seeing the first-versus-original-teacher FINISH tradeoff. Static floor((first_legal_tick + original_teacher_FINISH_tick)/2); same final-eligible WAIT. Frozen before any interior autonomous outcome. No sweep, tuning or later position search.',
        arms=['interior'], seeds=[0, 1, 2], snapshots=list(STAGES), expected_records=21,
        steps=main['steps'], batch=main['batch'], spec=main['spec'],
        training_protocol_hash=train['protocol_hash'], main_protocol_hash=main['protocol_hash'],
        frozen_files={str(path): core.sha(path) for path in dict.fromkeys(files)}, historical_bindings=bindings,
        tests='Exactly21 bindings: B-only long heldout B0/1/2/4/8 plus B8 random-mixed and A-only history, three seeds. Same public1024 WAIT warmup, external events, scheduler and matured request identities; no gate, oracle, learning or intra-fork reset.',
        timing_early=main['timing_early'], timing_legal=main['timing_legal'], timing_denominator=main['timing_denominator'],
        analysis=main['analysis'], write_policy=main['write_policy'],
        comparison='Bind exact main first/midpoint and historical last/no_review checkpoints at matching snapshot/scenario and actual B NEW exposures. Keep A118 and B117 B8 mixed conflict cohorts separate, actual early timing versus legal-window content and no-BEGIN/no-window/no-FINISH distinct.')
    plan['protocol_hash'] = core.state_digest(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    core.old.write(PLAN, plan)
    return plan


def install():
    core.OUT, core.TRAIN, core.PLAN, core.FORMAT = OUT, TRAIN, PLAN, FORMAT
    core.ARMS, core.STAGES = ('interior',), STAGES
    core.scenarios, core.expected_keys = scenarios, expected_keys
    core.reference_rows, core.checked = reference_rows, checked
    core.bind = bind


def bind(row, path, cp, seed, arm, stage, scenario, plan):
    ORIGINAL_BIND(row, path, cp, seed, arm, stage, scenario, plan)
    assert cp['interior_position']['protocol_hash'] == plan['training_protocol_hash']
    assert cp['interior_position']['single_post_endpoint_supplement'] is True
    row['checkpoint_interior_position'] = core.old.json_metadata(cp['interior_position'])
    return row


def audit_worker(seed):
    path = OUT / f'audit_seed_{seed:03d}.json'
    if not path.exists():
        with (OUT / f'audit_seed_{seed:03d}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u', '-m', 'experiments.finish_position.eval_interior', 'audit', '--seed', str(seed)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, ('interior audit failed', seed)
    return core.read(path)


def report():
    plan = checked()
    assert core.read(OUT / 'audit.json')['complete']
    own = core.records()
    assert len(own) == 21
    rows = own + [core.read(path) for path in reference_rows()]
    flat = []
    for row in rows:
        for family, cue in (('A', row['history_family']), ('B', row['B_family'])):
            for group in ('all', 'shared', 'conflict'):
                counts = row['strata'][str(cue)][group]['counts']
                if counts.get('matured'):
                    flat.append(dict(seed=row['seed'], arm=row['arm'], snapshot=row['snapshot'], scenario=row['scenario'], family=family,
                        raw_group=group, counts=counts, metrics=core.base.rates(counts), actual_unique_B_NEW=row['actual_unique_B_NEW']))
    summaries = []
    for arm, stage, scenario, family, group in sorted({tuple(r[k] for k in ('arm', 'snapshot', 'scenario', 'family', 'raw_group')) for r in flat}):
        per = [r for r in flat if (r['arm'], r['snapshot'], r['scenario'], r['family'], r['raw_group']) == (arm, stage, scenario, family, group)]
        assert len(per) == 3
        summaries.append(dict(arm=arm, snapshot=stage, scenario=scenario, family=family, raw_group=group, per_seed=per,
            metrics={m: core.describe(r['metrics'][m] for r in per) for m in core.base.rates({})}))
    timing_rows = []
    for path in sorted((OUT / 'timing').glob('seed_*/*_mixed.json')) + sorted((OUT / 'reference-timing').glob('seed_*/*_mixed.json')):
        diagnostic = core.read(path)
        for group in diagnostic['by_family']:
            timing_rows.append(dict(seed=diagnostic['seed'], arm=diagnostic['arm'], snapshot=diagnostic['snapshot'], **group))
    timing_summaries = []
    for arm in ('interior', 'first', 'midpoint', 'last', 'no_review'):
        for family in ('A', 'B'):
            per = [r for r in timing_rows if r['arm'] == arm and r['family'] == family]
            assert len(per) == 3
            timing_summaries.append(dict(arm=arm, family=family, per_seed=per,
                metrics={m: core.describe(r['metrics'][m] for r in per) for m in per[0]['metrics']},
                early_lead_seconds=core.describe(r['first_error_actual_early_lead_seconds']['mean'] for r in per)))
    value = dict(format=FORMAT, protocol_hash=plan['protocol_hash'], complete=True, records=21,
        summaries=summaries, mixed_timing=timing_summaries, historical_bindings=plan['historical_bindings'],
        per_seed_raw=[dict(seed=r['seed'], arm=r['arm'], snapshot=r['snapshot'], scenario=r['scenario'], by_family_raw=r['by_family_raw']) for r in rows])
    core.write(OUT / 'summary.json', value)
    def pick(arm, stage, scenario, family, group='all'):
        return next(r for r in summaries if (r['arm'], r['snapshot'], r['scenario'], r['family'], r['raw_group']) == (arm, stage, scenario, family, group))['metrics']
    def pct(value):
        return 'NA' if value is None else f'{100*value:.2f}%'
    lines = ['# 唯一中间tick FINISH事后补测', '', plan['declaration'], '',
        '21/21独立绑定，所有比率为3seed等权均值。原始自由行动、回执、合法窗口和成熟请求身份已重建审计。', '',
        '|快照|组|B普通|B严格|idle动作/千|', '|---|---|---:|---:|---:|']
    for stage in STAGES:
        for arm in ('interior', 'first', 'midpoint', 'no_review'):
            m = pick(arm, stage, 'B', 'B')
            lines.append(f"|{stage}|{arm}|{pct(m['ordinary_completion_rate']['mean'])}|{pct(m['strict_completion_rate']['mean'])}|{m['idle_actions_per1000']['mean']:.3f}|")
    lines += ['', '|B8随机交错|A冲突普通|A冲突严格|B冲突普通|B冲突严格|', '|---|---:|---:|---:|---:|']
    for arm in ('interior', 'first', 'midpoint', 'last', 'no_review'):
        a, b = (pick(arm, 'B008', 'mixed', family, 'conflict') for family in ('A', 'B'))
        lines.append(f"|{arm}|{pct(a['ordinary_completion_rate']['mean'])}|{pct(a['strict_completion_rate']['mean'])}|{pct(b['ordinary_completion_rate']['mean'])}|{pct(b['strict_completion_rate']['mean'])}|")
    lines += ['', '|组|seed|家族|成熟|普通|严格|首错实际early|未开始|已开始无窗口|首次legal对|首次legal错|窗口未发|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in timing_rows:
        c = r['counts']
        lines.append('|' + '|'.join(map(str, [r['arm'], r['seed'], r['family'], *[c[k] for k in ('matured', 'completed', 'strict', 'first_error_actual_early_FINISH', 'not_started', 'started_never_legal', 'first_legal_FINISH_correct', 'first_legal_FINISH_wrong', 'legal_window_no_FINISH')]])) + '|')
    lines += ['', plan['timing_early'], '', plan['timing_legal'], '', plan['analysis'], '']
    path = core.owned(OUT / 'report.md')
    text = '\n'.join(lines)
    if path.exists():
        assert path.read_text(encoding='utf-8') == text
    else:
        path.write_text(text, encoding='utf-8')
    core.write(OUT / 'completion.json', dict(format=FORMAT, protocol_hash=plan['protocol_hash'], complete=True, records=21,
        main_protocol_hash=plan['main_protocol_hash'], training_protocol_hash=plan['training_protocol_hash'],
        artifacts={str(OUT / n): core.sha(OUT / n) for n in ('audit.json', 'summary.json', 'report.md')}))
    return value


def finalize():
    plan = checked()
    while not all(core.result_path(*key).exists() for key in expected_keys()):
        time.sleep(5)
    with ThreadPoolExecutor(max_workers=3) as pool:
        shards = list(pool.map(audit_worker, range(3)))
    for seed, shard in enumerate(shards):
        assert shard['verified'] and shard['complete'] and shard['seed'] == seed and shard['records'] == 7
        assert shard['protocol_hash'] == plan['protocol_hash']
    core.write(OUT / 'audit.json', dict(format=FORMAT, protocol_hash=plan['protocol_hash'], verified=True, complete=True, records=21,
        expected_records=21, checks=shards[0]['checks'], timing_artifacts=[r for s in shards for r in s['timing_artifacts']],
        reference_timing_artifacts=[r for s in shards for r in s['reference_timing_artifacts']],
        shard_reports={str(OUT / f'audit_seed_{s:03d}.json'): core.sha(OUT / f'audit_seed_{s:03d}.json') for s in range(3)}))
    return report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'watch', 'audit', 'report', 'finalize'))
    parser.add_argument('--seed', type=int, choices=range(3))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'prepare':
        plan = prepare()
        print(json.dumps(dict(protocol_hash=plan['protocol_hash'], expected_records=21)), flush=True)
        return
    install()
    if args.command == 'watch':
        print(json.dumps(core.available(args.seed, True)), flush=True)
    elif args.command == 'audit':
        result = core.audit(True, args.seed)
        print(json.dumps(dict(records=result['records'], complete=result['complete'])), flush=True)
    elif args.command == 'report':
        print(json.dumps(dict(records=report()['records'])), flush=True)
    else:
        print(json.dumps(dict(event='interior_eval_complete', records=finalize()['records'])), flush=True)


if __name__ == '__main__':
    main()
