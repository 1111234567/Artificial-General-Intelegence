"""Timing regression: actual early is independent of content and version."""
from experiments.wait_position import eval_position as e,eval_timing as t

def test_prespecified210_bindings():
    assert sum(len(e.scenarios(s)) for _ in range(3) for a in e.ARMS for s in e.SNAPSHOTS)==210

def test_first_error_early_does_not_hide_behind_wrong_version():
    assert t.category(dict(early_FINISH=True,wrong_version=True))=='early_FINISH'
    assert t.category(dict(early_FINISH=False,wrong_version=True))=='wrong_version'

def test_summary_matured_denominator_and_actual_window_denominator_stay_distinct():
    common=dict(completed=False,strict=False,accepted_BEGIN=True,first_actual_early_FINISH=None,first_error=None)
    rows=[common|dict(reached_legal_window=True,first_legal_window_FINISH=None,first_legal_observation=dict(now=2),mechanism='legal_window_no_FINISH'),
          common|dict(reached_legal_window=False,first_legal_window_FINISH=None,first_legal_observation=None,mechanism='started_but_never_legal')]
    r=t.summarize(rows)
    assert r['counts']['matured']==2 and r['counts']['reached_legal_window']==1
    assert r['metrics']['legal_window_no_FINISH_per_matured']==.5
    assert r['metrics']['first_legal_FINISH_emitted_per_window']==0
