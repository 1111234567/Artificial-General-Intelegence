"""Small regression checks for independent audit arithmetic."""
import unittest

import numpy as np

from experiments.timing_boundary import audit


class AuditArithmeticTests(unittest.TestCase):
    def test_row_pair_and_ranking_are_distinct(self):
        result = audit.boundary_metrics([3, 3, 8, 8], [-1, 0, -1, 0],
                                        [0, 1, 0, 1], [-3, -2, 2, 3])
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["both_correct"], 0)
        self.assertEqual(result["ranking_accuracy"], 1.0)

    def test_ranking_tie_counts_half(self):
        result = audit.boundary_metrics([3, 3], [-1, 0], [0, 1], [0, 0])
        self.assertEqual(result["ranking_accuracy"], .5)
        self.assertEqual(result["ranking_ties"], 1)
        self.assertEqual(result["correct"], 1)

    def test_duplicate_or_incomplete_pair_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            audit.boundary_metrics([3, 3], [-1, -1], [0, 0], [-1, -2])
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            audit.boundary_metrics([3], [-1], [0], [-1])

    def test_receipt_respects_readiness_content_version_and_begin(self):
        shape = (1, 1, 7)
        facts = dict(actions=np.array([[[3, 3, 7, 4, 3, 1, 0]]]),
                     version=np.zeros(shape, dtype=np.int64), payload=np.zeros(shape, dtype=np.int64),
                     active=np.ones(shape, dtype=bool), started=np.array([[[1, 1, 1, 1, 0, 0, 0]]], dtype=bool),
                     ready=np.array([[[1.1, 1, 1, 1, 1, np.inf, np.inf]]]),
                     deadline=np.full(shape, 2.), now=np.array([1.]))
        receipt, begins, finishes = audit.receipts(facts)
        np.testing.assert_array_equal(receipt, [[[-1, 1, -1, -1, -1, 1, 0]]])
        self.assertEqual(begins.sum(), 1)
        self.assertEqual(finishes.sum(), 1)

    def test_coefficient_replay_and_slot_specific_features(self):
        rows = dict(job=np.array([0, 1]), logits=np.arange(44).reshape(2, 2, 11))
        selected = audit.feature_matrix(rows, "targetslot_logits11")
        np.testing.assert_array_equal(selected, np.vstack([np.arange(11), np.arange(33, 44)]))
        fitted = dict(mean=np.array([1., 2.]), scale=np.array([2., 4.]),
                      coef=np.array([3., 5.]), intercept=np.array(-2.))
        np.testing.assert_allclose(audit.affine_prediction([[3, 6], [1, 2]], fitted), [6, -2])

    def test_maturity_matches_origin_before_simultaneous_new(self):
        shape = (517, 1, 1)
        facts = {name: np.zeros(shape, dtype=np.int64) for name in
                 ("event_kind", "event_id", "mature_id", "mature_family", "family", "raw", "version", "payload", "rid")}
        facts.update({name: np.zeros(shape, dtype=bool) for name in
                      ("event_eligible", "mature", "active", "started", "eligible")})
        facts.update({name: np.zeros(shape, dtype=np.float64) for name in
                      ("ready", "deadline", "issued_time", "duration", "mature_issued", "mature_duration")})
        facts["actor_ids"] = np.array([91000])
        facts["now"] = np.arange(517) / 4
        facts["event_kind"][512, 0, 0] = facts["event_kind"][516, 0, 0] = 1
        facts["event_id"][512, 0, 0] = 1024
        facts["event_id"][516, 0, 0] = 1032
        facts["event_eligible"][512, 0, 0] = True
        facts["rid"][512:516, 0, 0] = 1024
        facts["rid"][516, 0, 0] = 1032
        facts["active"][512:516, 0, 0] = facts["eligible"][512:516, 0, 0] = True
        facts["started"][513:516, 0, 0] = True
        facts["issued_time"][512:516, 0, 0] = 128
        facts["ready"][512:516, 0, 0] = 128.5
        facts["deadline"][512:516, 0, 0] = 128.75
        facts["duration"][512:516, 0, 0] = .5
        facts["mature"][516, 0, 0] = True
        facts["mature_id"][516, 0, 0] = 1024
        facts["mature_issued"][516, 0, 0] = 128
        facts["mature_duration"][516, 0, 0] = .5
        facts["family"][516, 0, 0] = 1
        expected, cohort, counts = audit.rebuild_rows(facts, 0)
        self.assertEqual(counts["matured"], 1)
        self.assertEqual(counts["pairs"], 1)
        self.assertEqual(cohort[0]["family"], 0)
        self.assertEqual({key[-1] for key in expected}, {-1, 0, 1})


if __name__ == "__main__":
    unittest.main()
