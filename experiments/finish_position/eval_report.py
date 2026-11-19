"""Post-audit reporting-only correction; frozen evaluator remains unchanged.

Conflict raw numbers rotate across seed-specific mapping tables. Requiring all
three seeds inside a particular numeric raw group is invalid. The all/shared/
conflict summaries still have three seeds; numeric-raw summaries explicitly
retain their actual contributing seeds. No rollout or scoring is changed.
"""
from __future__ import annotations
from datetime import datetime, timezone
import inspect
from pathlib import Path
from experiments.finish_position import eval_finish as e


def main():
    plan = e.checked()
    audited = e.read(e.OUT / 'audit.json')
    assert audited['verified'] and audited['complete'] and audited['records'] == 84
    correction = dict(format='finish-position-reporting-correction-v1',
        protocol_hash=plan['protocol_hash'], audit_sha256=e.sha(e.OUT / 'audit.json'),
        frozen_evaluator_sha256=e.sha(e.__file__), report_wrapper_sha256=e.sha(__file__),
        reason='Specific numeric conflict raws vary across seed mapping tables. Two report-only all-three-seed assertions were invalid. Retain every per-seed numeric-raw stratum and explicitly expose contributing_seeds. Overall conflict groups remain three-seed equal-weight means.',
        unchanged='All84 raw rollouts, source evaluator, timing analyses, checkpoint bindings, and passed raw audits are unchanged.')
    e.write(e.OUT / 'reporting-correction.json', correction)
    source = inspect.getsource(e.report)
    assert source.count('assert len(per) == 3') == 2
    source = source.replace('def report():', 'def corrected_report():', 1)
    source = source.replace('assert len(per) == 3', "assert 1 <= len(per) <= 3 and len({r['seed'] for r in per}) == len(per)")
    source = source.replace('per_seed=per,', 'per_seed=per, contributing_seeds=len(per),')
    source = source.replace('summaries=summaries, mixed_timing=timing_summaries,',
        'summaries=summaries, reporting_correction=read(OUT / \'reporting-correction.json\'), mixed_timing=timing_summaries,')
    exec(compile(source, str(Path(__file__).resolve()) + ':frozen-report-adapter', 'exec'), e.__dict__)
    result = e.__dict__['corrected_report']()
    print(dict(event='finish_position_eval_complete', records=result['records'], reporting_correction=True), flush=True)


if __name__ == '__main__':
    main()
