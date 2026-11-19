"""Meaningful checks for request grouping, frozen preprocessing and exports."""
import unittest

import numpy as np

from experiments.timing_boundary import probe


class BoundaryMetricsTest(unittest.TestCase):
    def rows(self):
        # Intentionally interleaved and reversed request order.
        return dict(pair_id=np.array([22, 11, 22, 11]), offset=np.array([0, -1, -1, 0]),
                    legal=np.array([True, False, False, True]), tick=np.array([8, 4, 7, 5]),
                    job=np.array([1, 0, 1, 0]), actor=np.array([3, 2, 3, 2]), rid=np.array([7, 6, 7, 6]),
                    family=np.array([1, 0, 1, 0]), raw=np.array([2, 1, 2, 1]), conflict=np.ones(4, bool),
                    target=np.array([4, 3, 4, 3]), duration=np.array([13., 9., 13., 9.]),
                    remaining_ticks=np.array([0., 1., 1., 0.]))

    def test_pairs_are_grouped_by_identity(self):
        rows = self.rows()
        self.assertEqual(probe.boundary_indices(rows).tolist(), [0, 1, 2, 3])
        last, first = probe.pair_indices(rows)
        self.assertEqual(last.tolist(), [1, 2])
        self.assertEqual(first.tolist(), [3, 0])
        # Request11: both correct, request22: tie predicting legal twice.
        metric = probe.timing_metrics(rows, np.array([0.7, 0.2, 0.7, 0.8]))
        self.assertEqual(metric["correct_rows"], 3)
        self.assertEqual(metric["both_correct"], 1)
        self.assertEqual(metric["rank_wins"], 1)
        self.assertEqual(metric["rank_ties"], 1)
        self.assertEqual(metric["rank_strict_accuracy"], 0.5)
        self.assertEqual(metric["rank_accuracy"], 0.75)
        self.assertEqual(metric["false_early"], 1)
        self.assertEqual(metric["missed_first_legal"], 0)

    def test_threshold_is_fixed_inclusive_half(self):
        metric = probe.timing_metrics(self.rows(), np.full(4, 0.5))
        self.assertEqual(metric["both_correct"], 0)
        self.assertEqual(metric["false_early"], 2)
        self.assertEqual(metric["missed_first_legal"], 0)
        self.assertEqual(metric["rank_accuracy"], 0.5)

    def test_missing_or_duplicate_endpoint_rejected(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            probe.pair_indices(rows, [0, 1, 2])
        rows["tick"][0] += 1
        with self.assertRaisesRegex(ValueError, "consecutive"):
            probe.boundary_indices(rows)

    def test_pairflip_preserves_pairs_and_balance(self):
        rows = self.rows()
        y, flipped = probe.pairflip_targets(rows, np.arange(4), seed=1, slot=0)
        self.assertEqual(len(flipped), 1)
        self.assertEqual(int(y.sum()), 2)
        for pair in (11, 22):
            self.assertEqual(int(y[rows["pair_id"] == pair].sum()), 1)
        again, other = probe.pairflip_targets(rows, np.arange(4), seed=1, slot=0)
        np.testing.assert_array_equal(y, again)
        np.testing.assert_array_equal(flipped, other)

    def test_remaining_sign_and_offsets(self):
        rows = self.rows()
        metric = probe.remaining_metrics(rows, np.array([-0.1, 0.9, 0.3, -0.2]))
        self.assertEqual(metric["boundary_both_correct"], 2)
        self.assertEqual(metric["by_offset"]["-1"]["rows"], 2)
        self.assertAlmostEqual(metric["mae_ticks"], 0.275)


class FrozenReadoutTest(unittest.TestCase):
    def test_no_test_scaler_use_and_numpy_binary_reproduction(self):
        from sklearn.linear_model import LogisticRegression
        train = np.array([[-2., 7.], [-1., 7.], [1., 7.], [2., 7.]])
        target = np.array([0, 0, 1, 1])
        model, info = probe.fit_readout(train, target, "timing")
        np.testing.assert_array_equal(model["mean"], [0., 7.])
        self.assertEqual(model["scale"][1], 1.)
        heldout = np.array([[10000., -10000.], [-10000., 10000.]])
        saved_mean, saved_scale = model["mean"].copy(), model["scale"].copy()
        predicted = probe.numpy_predict(model, heldout, "timing")
        np.testing.assert_array_equal(model["mean"], saved_mean)
        np.testing.assert_array_equal(model["scale"], saved_scale)
        self.assertEqual(predicted.tolist(), [1., 0.])
        reference = LogisticRegression(C=1., l1_ratio=0., solver="lbfgs", max_iter=2000, tol=1e-8).fit(
            (train - model["mean"]) / model["scale"], target)
        query = np.array([[-0.2, 7.], [0.8, 7.]])
        np.testing.assert_allclose(probe.numpy_predict(model, query, "timing"),
                                   reference.predict_proba((query - model["mean"]) / model["scale"])[:, 1], atol=1e-14)
        self.assertEqual(info["train_rows"], 4)

    def test_multiclass_and_ridge_numpy_exports(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        rng = np.random.default_rng(41)
        train = rng.normal(size=(60, 4))
        query = rng.normal(size=(7, 4))
        y = np.tile([0, 2, 5], 20)
        model, _ = probe.fit_readout(train, y, "content")
        transformed = (train - model["mean"]) / model["scale"]
        zquery = (query - model["mean"]) / model["scale"]
        reference = LogisticRegression(C=1., l1_ratio=0., solver="lbfgs", max_iter=2000, tol=1e-8).fit(transformed, y)
        actual = probe.numpy_predict(model, query, "content")
        np.testing.assert_allclose(actual[:, [0, 2, 5]], reference.predict_proba(zquery), atol=1e-14)
        np.testing.assert_array_equal(actual[:, [1, 3, 4, 6, 7]], 0.)
        regression_y = train[:, 0] * 2 - train[:, 1] + rng.normal(size=60)
        model, _ = probe.fit_readout(train, regression_y, "remaining")
        reference = Ridge(alpha=1., solver="cholesky").fit(transformed, regression_y)
        np.testing.assert_allclose(probe.numpy_predict(model, query, "remaining"), reference.predict(zquery), atol=1e-14)

    def test_mlp_numpy_export_layer_order(self):
        from sklearn.neural_network import MLPClassifier
        rng = np.random.default_rng(43)
        x = rng.normal(size=(30, 4))
        y = (x[:, 0] > 0).astype(int)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = MLPClassifier(hidden_layer_sizes=(64,), activation="tanh", max_iter=3, random_state=2).fit(x, y)
        model = dict(mean=np.zeros(4), scale=np.ones(4), classes=estimator.classes_,
                     w0=estimator.coefs_[0], b0=estimator.intercepts_[0], w1=estimator.coefs_[1], b1=estimator.intercepts_[1])
        np.testing.assert_allclose(probe.numpy_predict(model, x, "timing"), estimator.predict_proba(x)[:, 1], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
