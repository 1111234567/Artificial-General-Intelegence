"""Additive supervised statistics require fixed features and correct partitioning."""
import numpy as np
import pytest

from experiments.credit.statistics import (
    AdditiveRidge, Normalizer, actor_safe_blocks, audit_arrays, target_association_counterexample,
)


def _data():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(120, 8)).astype(np.float32)
    x[:, -1] = 1
    y = rng.integers(0, 4, len(x))
    columns = dict(actor=np.repeat(np.arange(6), 20), job=np.zeros(120, int),
                   request=np.tile(np.repeat(np.arange(5), 4), 6),
                   condition=np.asarray(["iid"] * 120), tick=np.tile(np.arange(20), 6))
    return x, y, columns


def test_chunked_augmented_statistics_match_direct_batch_with_intercept():
    x, y, columns = _data()
    result = audit_arrays(x, y, columns, columns["actor"] < 4, columns["actor"] >= 4, classes=4, block_rows=9)
    assert result["status"] == "passed_float64_equivalence"
    assert result["train_samples"] == 80
    assert result["heldout_samples"] == 40
    assert result["parameter_max_abs_difference"] < 1e-12
    assert result["heldout"]["prediction_agreement"] == 1
    assert result["heldout"]["disagreements_outside_numerical_tie_bound"] == 0
    assert not result["intercept_penalized"]
    assert result["memory"]["retained_statistic_state_bytes"] == ((9 * 9 + 9 * 4 + 8 * 2) * 8 + 8)


def test_chronological_blocks_never_cross_actor_domain_job_or_request_boundary():
    _, _, columns = _data()
    blocks = list(actor_safe_blocks(columns, np.ones(120, bool), block_rows=7))
    seen = np.concatenate(blocks)
    assert np.array_equal(seen, np.arange(120))
    request_blocks = {}
    for block_index, block in enumerate(blocks):
        assert len(set(columns["actor"][block])) == 1
        assert np.all(np.diff(columns["tick"][block]) >= 0)
        for i in block:
            request = int(columns["actor"][i]), int(columns["request"][i])
            assert request not in request_blocks or request_blocks[request] == block_index
            request_blocks[request] = block_index


def test_heldout_values_and_labels_do_not_enter_normalization_or_statistics():
    x, y, columns = _data()
    train, test = columns["actor"] < 4, columns["actor"] >= 4
    first = audit_arrays(x, y, columns, train, test, classes=4)
    changed = x.copy()
    changed[test] += 1e6
    changed_y = y.copy()
    changed_y[test] = (changed_y[test] + 1) % 4
    second = audit_arrays(changed, changed_y, columns, train, test, classes=4)
    for key in ("xtx_max_abs_difference", "xty_max_abs_difference", "parameter_max_abs_difference", "parameter_relative_l2_difference"):
        assert first[key] == second[key]
    assert first["heldout"]["direct"] != second["heldout"]["direct"]


def test_same_feature_correlation_and_class_frequency_do_not_fix_target_solution():
    example = target_association_counterexample(.01)
    assert example["same_xtx"] and example["same_label_marginals"]
    assert example["xty_max_abs_difference"] > 0
    assert example["coefficient_max_abs_difference"] > 0
    assert example["predictions_a"] == [0, 1]
    assert example["predictions_b"] == [1, 0]


def test_constant_features_missing_classes_and_no_test_samples_remain_defined_honestly():
    x, y, columns = _data()
    x[:] = 1
    y[:] = 0
    result = audit_arrays(x, y, columns, columns["actor"] < 4, np.zeros(120, bool), classes=4)
    assert result["status"] == "passed_float64_equivalence"
    assert result["heldout"]["n"] == 0
    assert result["heldout"]["prediction_agreement"] is None
    assert result["heldout"]["direct"]["accuracy"] is None
    normalizer = Normalizer.from_training(x)
    assert np.all(normalizer.scale == 1)


def test_invalid_overlap_target_or_intercept_is_rejected():
    x, y, columns = _data()
    with pytest.raises(ValueError, match="overlap"):
        audit_arrays(x, y, columns, np.ones(120, bool), np.ones(120, bool), classes=4)
    accumulator = AdditiveRidge(2, 2)
    with pytest.raises(ValueError, match="intercept"):
        accumulator.add([[0, 1, 2]], [0])
    with pytest.raises(ValueError, match="target"):
        accumulator.add([[1, 1, 2]], [2])
    with pytest.raises(ValueError):
        list(actor_safe_blocks(columns, np.ones(120, bool), block_rows=0))
