"""Compact, source-bound report of the original three-arm selection logs."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.wait_position import bank
from experiments.review_selection import bank as old


def report():
    path=bank.OUT/'mechanism_diagnostics.json'
    value=json.loads(path.read_text(encoding='utf-8'))
    assert not value['partial']
    assert old.sha(ROOT/'experiments/wait_position/bank_diagnostics.py')==value['hashes']['implementation']
    for log,digest in value['hashes']['logs'].items():assert old.sha(ROOT/log)==digest
    small={key:value[key] for key in ('format','definitions','hashes','eligible_bank_phase_coverage','pooled')}
    small.update(full_diagnostics=dict(path=str(path.relative_to(ROOT)),sha256=old.sha(path)),
        seeds={seed:{arm:{phase:record[phase] for phase in
            ('actual_logged_review_updates','all','first_round_updates_0to31','last_round_updates_224to255','by_round')}
            for arm,record in arms.items()} for seed,arms in value['seeds'].items()},
        report_implementation_sha256=old.sha(__file__))
    old.write(bank.OUT/'mechanism_summary.json',small)
    lines=['# Actual selected WAIT positions','',
        'This report reads retained actual trainer selections only. It does not rescore policies or change any choice. '
        'The three original arms each have 3 seeds x 256 updates x 64 selected requests = 49,152 selected WAIT rows. '
        'Physical times use the native world unit (source dt=0.25).','',
        '| Arm | Ready distance median | Processing fraction median | Last eligible tick | True FINISH argmax | Early phase <25% | Late phase >=75% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('midpoint','hard','random'):
        r=value['pooled'][arm]['all'];p=r['phase_coverage']
        lines.append(f"| {arm} | {r['time_to_ready']['median']:.6f} | {r['processing_fraction']['median']:.6f} | {100*r['selected_is_last_eligible_fraction']:.4f}% | {100*r['selected_finish_intent_fraction']:.4f}% | {100*p['early_processing_0to25pct']['fraction']:.4f}% | {100*p['late_processing_75to100pct']['fraction']:.4f}% |")
    lines+=['','Hard uses maximum FINISH-vs-nonFINISH margin among admissible WAIT ticks. '
        'A true FINISH intent means max(FINISH logits)>max(WAIT/BEGIN logits); otherwise the allowed protocol still chooses the largest margin.','',
        '| Hard subset | Rows | Ready distance median | Ready distance mean | Last eligible tick |',
        '|---|---:|---:|---:|---:|']
    for key,label in (('selected_FINISH_gt_nonFINISH','True FINISH > nonFINISH'),
                      ('entire_eligible_window_no_FINISH','No FINISH anywhere in eligible window')):
        r=value['pooled']['hard']['all']['timing_by_intent'][key]
        lines.append(f"| {label} | {r['n']} | {r['time_to_ready']['median']:.6f} | {r['time_to_ready']['mean']:.6f} | {100*r['selected_is_last_eligible_fraction']:.4f}% |")
    lines+=['','| Hard training phase | Rows | Last eligible tick | True FINISH argmax | Ready distance median | Early processing <25% |',
        '|---|---:|---:|---:|---:|---:|']
    for key,label in (('first_round_updates_0to31','First 32 review updates'),('last_round_updates_224to255','Last 32 review updates')):
        r=value['pooled']['hard'][key]
        lines.append(f"| {label} | {r['n']} | {100*r['selected_is_last_eligible_fraction']:.4f}% | {100*r['selected_finish_intent_fraction']:.4f}% | {r['time_to_ready']['median']:.6f} | {100*r['phase_coverage']['early_processing_0to25pct']['fraction']:.4f}% |")
    hard=value['pooled']['hard']['all']['phase_coverage']
    random=value['pooled']['random']['all']['phase_coverage']
    early,late='early_processing_0to25pct','late_processing_75to100pct'
    last_by_seed=', '.join(f"seed {seed}: {100*arms['hard']['all']['selected_is_last_eligible_fraction']:.4f}%"
                          for seed,arms in value['seeds'].items())
    lines+=['',f"All {hard[early]['requests_with_eligible_candidate_in_phase']} retained requests have admissible early (<25%) and late (>=75%) processing candidates. "
        f"Over the full run, hard supervises early processing for {hard[early]['unique_requests']}/768 requests and late processing for {hard[late]['unique_requests']}/768; "
        f"random covers early/late phases in {random[early]['unique_requests']}/{random[late]['unique_requests']} requests. Hard last-eligible coincidences are {last_by_seed}.",'',
        'These describe the realized training positions and intents. They do not by themselves identify an outcome effect '
        'of model-based hardness separately from concentrating feedback near readiness.','',
        f"Source: `mechanism_diagnostics.json`, SHA256 `{old.sha(path)}`. Compact exact values and source log hashes are in `mechanism_summary.json`."]
    (bank.OUT/'mechanism_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=str(bank.OUT/'mechanism_summary.json'),report=str(bank.OUT/'mechanism_report.md'))),flush=True)


if __name__=='__main__':report()
