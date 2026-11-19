"""Finite signed labels, causal inclusion and independent native-clock ledger."""
import unittest
import warnings

import numpy as np
import torch

from experiments.action_split import data as source
from experiments.remaining_time import targets as d


class RemainingTimeTargetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = source.config(0)
        cfg.update(seed=1001, batch_size=2, actor_ids=(95000, 95001),
                   duration_range=(1., 2.), gap_range=(.5, 1.))
        cls.facts, cls.context = source.collect_native(0, total=384, world_config=cfg)

    def test_sign_boundary_and_physical_scale(self):
        ready = np.asarray([[[7.75, 8., 8.25]]], np.float64)
        active = started = np.ones(ready.shape, bool)
        target, mask = d.derive_targets(ready, np.asarray([8.]), active, started)
        np.testing.assert_array_equal(target, np.asarray([[[-.25/16, 0., .25/16]]], np.float32))
        self.assertTrue(mask.all())
        self.assertEqual(target.dtype, np.float32)

    def test_nonfinite_ready_never_enters_arithmetic(self):
        ready = np.asarray([np.inf, -np.inf, np.nan, 8., 9., 10.], np.float64)
        active = np.asarray([True, True, True, False, True, True])
        started = np.asarray([True, True, True, True, False, True])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(all="raise"):
                target, mask = d.derive_targets(ready, 11., active, started)
        np.testing.assert_array_equal(mask, [False, False, False, False, False, True])
        np.testing.assert_array_equal(target, [0., 0., 0., 0., 0., -1/16])
        self.assertTrue(np.isfinite(target).all())

    def test_no_eligibility_selection_or_future_outcome(self):
        f = self.facts
        value, mask = d.derive_targets(f["ready"], f["now"], f["active"], f["started"])
        np.testing.assert_array_equal(mask, f["active"] & f["started"] & np.isfinite(f["ready"]))
        self.assertGreater(int((mask & ~f["eligible"]).sum()), 0)
        changed = {**f, "eligible": ~f["eligible"]}
        changed_value, changed_mask = d.derive_targets(changed["ready"], changed["now"], changed["active"], changed["started"])
        np.testing.assert_array_equal(value, changed_value)
        np.testing.assert_array_equal(mask, changed_mask)
        self.assertTrue(np.all(value[~mask] == 0.))

    def test_accepted_begin_duration_reconstruction_every_row(self):
        f = self.facts
        actual, mask = d.derive_targets(f["ready"], f["now"], f["active"], f["started"])
        independent, audit = d.accepted_begin_ledger(f)
        np.testing.assert_array_equal(actual, independent["time_target"])
        np.testing.assert_array_equal(mask, independent["time_mask"])
        self.assertEqual(audit["checked_ticks"], 384)
        self.assertGreater(audit["accepted_BEGIN"], 0)
        self.assertGreater(audit["accepted_FINISH"], 0)
        self.assertFalse(audit["eligibility_used"])
        broken = {**f, "ready": f["ready"].copy()}
        tick, lane, job = np.argwhere(mask)[0]
        broken["ready"][tick, lane, job] += .125
        with self.assertRaises(AssertionError):
            d.accepted_begin_ledger(broken)

    def test_independent_targets_match_model_helper(self):
        from experiments.remaining_time.model import remaining_time_targets
        f = self.facts
        value, mask = d.derive_targets(f["ready"], f["now"], f["active"], f["started"])
        helper_value, helper_mask = remaining_time_targets(torch.from_numpy(f["ready"]),
            torch.from_numpy(f["now"]), torch.from_numpy(f["active"]), torch.from_numpy(f["started"]))
        np.testing.assert_array_equal(value, helper_value.float().numpy())
        np.testing.assert_array_equal(mask, helper_mask.numpy())

    def test_counts_and_256block_budget(self):
        self.assertEqual((d.WARM, d.TRAIN_STEPS, d.TOTAL, d.BLOCK), (512, 32768, 33280, 128))
        self.assertEqual(d.TRAIN_STEPS // d.BLOCK, 256)
        f = self.facts
        value, mask = d.derive_targets(f["ready"], f["now"], f["active"], f["started"])
        targets = dict(time_target=value, time_mask=mask)
        total = d.counts(f, targets)
        blocks = [d.counts(f, targets, i, i+128) for i in range(0, 384, 128)]
        for key, val in total.items():
            self.assertEqual(sum(row[key] for row in blocks), val, key)
        self.assertEqual(total["valid"], total["negative"]+total["zero"]+total["positive"])
        self.assertGreater(total["ineligible_mask_diagnostic"], 0)


if __name__ == "__main__":
    unittest.main()
