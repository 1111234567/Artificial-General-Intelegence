"""Small native suffix and past-clock corruption checks; no model/results."""
import unittest

import numpy as np
import torch

from experiments.episodic_review import retention_data as d


class RetentionSuffixDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = torch.load(d.source_world_path(0), map_location="cpu", weights_only=False)["final"]
        cls.facts, cls.context = d.collect_native(0, total=128, initial_state=cls.initial)
        cls.prefix = d.source_prefix(0)

    def test_fixed_budget_and_unchanged_complete_native_boundary(self):
        self.assertEqual((d.WARM, d.TOTAL, d.BLOCK, d.UPDATES, d.BATCH), (0, 8192, 128, 64, 32))
        self.assertEqual(d.state_digest(self.initial), d.state_digest(self.context["initial"]))
        self.assertEqual(self.facts["world_tick"][0], 66048)
        self.assertEqual(self.facts["world_tick"][-1], 66175)
        np.testing.assert_array_equal(self.facts["now"], np.arange(66048, 66176)*.25)
        self.assertEqual(self.context["final"]["tick"], 66175)

    def test_actual_B_only_labels_and_inherited_tasks_are_retained(self):
        counts = d.verify(self.facts, 0, self.initial)
        self.assertTrue(all(value == 0 for key, value in counts.items() if key.startswith("A_")))
        self.assertGreater(counts["B_inherited_time_labels"], 0)
        report = d.accepted_begin_ledger(self.facts, self.initial, self.prefix)
        self.assertEqual(report["time_rows"], int(self.facts["time_mask"].sum()))
        self.assertGreater(report["inherited_finite_ready_clocks_verified"], 0)
        self.assertFalse(report["eligibility_used"])
        self.assertFalse(report["ready_from_new_archive_used_to_construct_labels"])

    def test_independent_ledger_rejects_time_and_native_receipt_corruption(self):
        index = tuple(np.argwhere(self.facts["time_mask"])[0])
        for name in ("ready", "time_target"):
            corrupted = {**self.facts, name: self.facts[name].copy()}
            corrupted[name][index] += .125
            with self.assertRaises(AssertionError):
                d.accepted_begin_ledger(corrupted, self.initial, self.prefix)
        begin = tuple(np.argwhere(self.facts["actions"] != 0)[0])
        corrupted = {**self.facts, "receipts": self.facts["receipts"].copy()}
        corrupted["receipts"][begin] = -1
        with self.assertRaises(AssertionError):
            d.accepted_begin_ledger(corrupted, self.initial, self.prefix)

    def test_source_finite_clock_requires_actual_past_accepted_BEGIN(self):
        prefix = {key: value.copy() for key, value in self.prefix.items()}
        prefix["receipts"].fill(0)
        with self.assertRaises(AssertionError):
            d.accepted_begin_ledger(self.facts, self.initial, prefix)


if __name__ == "__main__":
    unittest.main()
