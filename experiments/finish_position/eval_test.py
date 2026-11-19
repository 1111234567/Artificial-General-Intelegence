"""Independent safety and timing denominator regression checks."""
import pytest
from experiments.finish_position import eval_finish as e


def test_write_guard_rejects_every_historical_output():
    for target in (e.MAIN_EVAL / 'audit.json', e.LAST_EVAL / 'audit.json', e.base.OUT / 'audit.json', e.base.original.OUT / 'audit.json'):
        with pytest.raises(ValueError):
            e.owned(target)
    assert e.owned(e.OUT / 'audit.json').is_relative_to(e.OUT.resolve())


def test_original_source_output_and_writes_are_restored():
    saved = e.base.write, e.base.original.write, e.base.original.OUT, e.base.original.PLAN
    with e.source_write_guard():
        assert e.base.original.OUT.is_relative_to(e.OUT)
        with pytest.raises(RuntimeError):
            e.base.write(e.OUT / 'probe.json', {})
        with pytest.raises(RuntimeError):
            e.base.original.write(e.OUT / 'probe.json', {})
    assert (e.base.write, e.base.original.write, e.base.original.OUT, e.base.original.PLAN) == saved


def test_all_declared_forks_and_first_snapshots():
    assert len(e.expected_keys()) == 84
    assert e.scenarios('Bfirst002') == ('B',)
    assert e.scenarios('B008') == ('B', 'mixed', 'history')


def test_actual_early_precedes_version_and_content_errors():
    assert e.timing.category(dict(early_FINISH=True, wrong_version=True, legal_other_context_payload=True)) == 'early_FINISH'
    assert e.timing.category(dict(early_FINISH=False, wrong_version=True)) == 'wrong_version'
