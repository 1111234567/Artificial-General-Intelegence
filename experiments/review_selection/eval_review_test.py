"""Counter and condition checks for the independent review behavior evaluation."""
import torch
from experiments.review_selection import eval_review as e


def test_true_new_counter_includes_no_loss_boundary_and_not_replay_work():
    def cp(count):
        v = torch.zeros((2, 2, 4), dtype=torch.long)
        v[0, 0, 1] = count
        return dict(world=dict(context=dict(tensors=dict(family_new_counts=v))), work=dict(B_unique_new_commands=10, review_positions=5000))
    assert e.actual_B_new(cp(15), cp(2), 1) == 13
    assert e.actual_B_new(cp(15), cp(2), 0) == 0


def test_family_denominators_never_hide_old_failure_with_new_success():
    row = dict(history_family=0, B_family=1, strata={
        '0': {'conflict': {'counts': dict(matured=100, completed=0, clean_completed=0)}},
        '1': {'conflict': {'counts': dict(matured=10, completed=10, clean_completed=10)}}})
    a, b = (e.family_counts(row, part, 'conflict') for part in ('history', 'B'))
    assert e.base.rates(a)['strict_completion_rate'] == 0
    assert e.base.rates(b)['strict_completion_rate'] == 1
    assert a['matured'] == 100 and b['matured'] == 10


def test_full_prespecified_curves_have_168_bindings():
    assert sum(len(e.scenarios(t)) for _ in range(3) for _ in e.ARMS for t in e.SNAPSHOTS) == 168
    assert all('mixed' in e.scenarios(t) for t in ('B000', 'B001', 'B002', 'B004', 'B008'))
