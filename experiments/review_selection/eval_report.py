"""Export complete autonomous curves and separate old/new conflict retention."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.review_selection import eval_review as e


def fmt(value, percent=False):
    if value is None:
        return 'NA'
    return f'{100*value:.1f}%' if percent else f'{value:.2f}'


def run():
    summary = e.summarize()
    rows = summary['summaries']
    def pick(scenario, arm, snapshot, part, group):
        return next((r for r in rows if (r['scenario'], r['arm'], r['snapshot'], r['family_part'], r['raw_group']) == (scenario, arm, snapshot, part, group)), None)
    def mean(row, metric):
        return row['metrics'][metric]['mean'] if row else None
    lines = [
        '# Independent review-selection behavior evaluation', '',
        f"Protocol `{summary['protocol_hash']}`; {summary['records']} / 168 bindings; complete={summary['complete']}.", '',
        'All tests use native11-action argmax, no parameter updates and no action gate. Each checkpoint consumes the same independent public WAIT warmup before an autonomous long-heldout test. The same external cases and eligible matured-request denominators are checked against pre-existing WAIT references.', '',
        'The mixed stream samples A/B independently at each NEW and job; it is random interleaving, not strict alternation. A and B can be pending simultaneously. Explicit same-slot/raw alternation is separately reported by the paired-control evaluation.', '',
        'Every mean below weights the three seed rates equally. Per-seed counts, numerators, denominators, all raw groups and each conflicting symbol appear in curves.csv and summary.json. Ordinary completion allows a prior rejected action; strict completion does not. Idle non-WAIT is per1000 inactive slot opportunities and retains the previous slot family/raw.', '',
        'Actual B NEW counts come from the serialized real world, minus original B0. They include the no-loss boundary exposure. work.B_unique_new_commands is separately retained and is lower by2/1/3 for seed0/1/2 in the original experiment. Repeated review or temporal replay does not add independent cases.', '',
        '## Actual independent B NEW cases', '', '| Snapshot | seed0 | seed1 | seed2 | mean |', '|---|---:|---:|---:|---:|']
    for stage in e.SNAPSHOTS:
        row = pick('B', 'no_review', stage, 'B', 'all')
        if not row:
            continue
        vals = {x['seed']: x['actual_unique_B_NEW'] for x in row['per_seed']}
        lines.append('| '+stage+' | '+' | '.join(str(vals.get(s, 'NA')) for s in range(3))+f" | {statistics.mean(vals.values()):.1f} |")
    lines += ['', '## Independent long B curve', '', '| Snapshot | Condition | Ordinary | Strict | Idle /1000 |', '|---|---|---:|---:|---:|']
    for stage in e.SNAPSHOTS:
        for arm in e.ARMS:
            row = pick('B', arm, stage, 'B', 'all')
            if row:
                lines.append(f"| {stage} | {arm} | {fmt(mean(row,'ordinary_completion_rate'),True)} | {fmt(mean(row,'strict_completion_rate'),True)} | {fmt(mean(row,'idle_actions_per1000'))} |")
    lines += ['', '## Same-world random-interleaving conflict curves', '',
        'These two family columns must be assessed separately: a good B rate does not compensate for failed old A requests.', '',
        '| Snapshot | Condition | A ordinary | A strict | A idle /1000 | B ordinary | B strict | B idle /1000 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for stage in e.MIXED:
        for arm in e.ARMS:
            a = pick('mixed', arm, stage, 'history', 'conflict')
            b = pick('mixed', arm, stage, 'B', 'conflict')
            if a and b:
                vals = []
                for row in (a, b):
                    vals += [fmt(mean(row, 'ordinary_completion_rate'), True), fmt(mean(row, 'strict_completion_rate'), True), fmt(mean(row, 'idle_actions_per1000'))]
                lines.append('| '+stage+' | '+arm+' | '+' | '.join(vals)+' |')
    lines += ['', '## B8 mixed conflicting-symbol detail by seed', '',
        '| seed | Condition | Family | Raw | Matured | Completed | Strict completed | Ordinary | Strict | Idle actions / opportunities | Idle /1000 |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for seed in range(3):
        for arm in e.ARMS:
            for part in ('history', 'B'):
                for group in ('conflict_first', 'conflict_second'):
                    row = pick('mixed', arm, 'B008', part, group)
                    if not row:
                        continue
                    x = next((x for x in row['per_seed'] if x['seed'] == seed), None)
                    if not x:
                        continue
                    c, m = x['counts'], x['metrics']
                    lines.append(f"| {seed} | {arm} | {'A' if part=='history' else 'B'} | {x['raws'][0]} | {c.get('matured',0)} | {c.get('completed',0)} | {c.get('clean_completed',0)} | {fmt(m['ordinary_completion_rate'],True)} | {fmt(m['strict_completion_rate'],True)} | {c.get('idle_actions',0)} / {c.get('idle_opportunities',0)} | {fmt(m['idle_actions_per1000'])} |")
    lines += ['', '## Endpoint differences from no review', '',
        'Paired differences are percentage points, each seed compared with itself on the identical external cases. B gain is a measured difference between snapshots; it is not an isolated learning-rate parameter.', '',
        '| Condition | Long B strict delta | Mixed A-conflict strict delta | Mixed B-conflict strict delta |', '|---|---:|---:|---:|']
    for arm in e.ARMS[:3]:
        vals = []
        for scenario, part, group in [('B','B','all'), ('mixed','history','conflict'), ('mixed','B','conflict')]:
            row = next((r for r in summary['paired'] if (r['scenario'],r['arm'],r['snapshot'],r['family_part'],r['raw_group']) == (scenario,arm,'B008',part,group)), None)
            val = row['metrics']['strict_completion_rate']['review_minus_no_review']['mean'] if row else None
            vals.append('NA' if val is None else f'{100*val:+.2f}')
        lines.append('| '+arm+' | '+' | '.join(vals)+' |')
    lines += ['', 'Selection policies use only old candidate experience; error-based selection can favor timing/control errors rather than a conflicting semantic symbol. The selection-audit report identifies which examples were actually chosen. Shared budget and case identity do not establish that one policy will generalize outside these3 fixed seeds and one heldout stream per seed.', '',
        'Original no-review results and original training files remain unchanged. audit.json checks every checkpoint/evidence binding, reconstructed request outcomes, public raw/context pulse identity, frozen external references and equal actual B exposure counts.', '']
    (e.OUT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    colors = dict(uniform='#2274A5', conflict='#D9841E', error='#4A8F55', no_review='#777777')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, scenario, part, group, stages, title in [
        (axes[0], 'B', 'B', 'all', e.SNAPSHOTS, 'Independent long B'),
        (axes[1], 'mixed', 'history', 'conflict', e.MIXED, 'Random A/B mixed: A conflict'),
        (axes[2], 'mixed', 'B', 'conflict', e.MIXED, 'Random A/B mixed: B conflict')]:
        for arm in e.ARMS:
            selected = [pick(scenario,arm,t,part,group) for t in stages]
            xx = list(range(len(stages)))
            yy = [mean(r,'strict_completion_rate') for r in selected]
            if any(v is None for v in yy):
                continue
            sd = [r['metrics']['strict_completion_rate']['sample_sd'] or 0 for r in selected]
            ax.errorbar(xx, yy, yerr=sd, color=colors[arm], label=arm, marker='o', capsize=2)
            for seed in range(3):
                ss = [next((p['metrics']['strict_completion_rate'] for p in r['per_seed'] if p['seed']==seed), None) for r in selected]
                ax.plot(xx, ss, color=colors[arm], alpha=.2, lw=.7)
        ax.set_title(title); ax.set_ylim(0,1); ax.grid(alpha=.2)
        ax.set_xticks(range(len(stages)), ['0','2u','8u','1r','2r','4r','8r'] if len(stages)==7 else ['0','1r','2r','4r','8r'])
    axes[0].set_ylabel('Strict completion rate'); axes[0].legend(fontsize=8)
    fig.suptitle('Equal-weight seed means +/- sample SD; faint lines = individual seeds')
    fig.supxlabel('Frozen snapshots, categorical spacing: u = first-epoch updates; r = complete B round')
    fig.tight_layout(rect=(0,.04,1,.94)); fig.savefig(e.OUT/'curves.png', dpi=180); plt.close(fig)
    e.write(e.OUT/'report-binding.json', dict(protocol_hash=summary['protocol_hash'], source_sha256=e.sha(__file__),
        summary_sha256=e.sha(e.OUT/'summary.json'), artifacts={str(e.OUT/name): e.sha(e.OUT/name) for name in ('report.md','curves.csv','curves.png')}))
    print(json.dumps(dict(records=summary['records'], complete=summary['complete'], report=str(e.OUT/'report.md'))))


if __name__ == '__main__':
    run()
