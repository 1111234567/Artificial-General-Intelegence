"""Causal native teacher targets and the full active supervised denominator."""
import unittest

import numpy as np
import torch

from experiments.action_split import data as d


class ActionSplitDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kwargs = d.config(0)
        kwargs.update(seed=1001, batch_size=2, actor_ids=(95000, 95001),
                      duration_range=(1., 2.), gap_range=(.5, 1.))
        cls.facts, cls.context = d.collect_native(0, total=384, world_config=kwargs)

    def test_declared_budget_and_native_world(self):
        self.assertEqual(d.TOTAL, 33280)
        self.assertEqual(d.TRAIN_STEPS // d.BLOCK, 256)
        for seed in range(3):
            kwargs = d.config(seed)
            self.assertEqual(kwargs["seed"], 850000+seed)
            self.assertEqual(kwargs["actor_ids"], tuple(range(95000, 95032)))
            self.assertEqual(kwargs["family_allowed"],
                             (d.SEEDS[seed]["history_cue"], d.SEEDS[seed]["B_cue"]))
            self.assertEqual(kwargs["duration_range"], (8., 16.))
            self.assertEqual(kwargs["gap_range"], (16., 32.))

    def test_native_earliest_receipts_and_public_fields(self):
        report = d.verify_facts(self.facts)
        self.assertEqual(report["native_earliest_actions"], "exact")
        self.assertEqual(report["checked_ticks"], 384)
        self.assertEqual(self.facts["obs"].dtype, np.float32)
        self.assertEqual(self.facts["obs"].shape, (384, 2, 56))
        self.assertEqual(self.facts["actions"].shape, (384, 2, 2))
        self.assertGreater(int(self.facts["mature"].sum()), 0)
        self.assertTrue(np.all(self.facts["mature_strict"] == self.facts["mature_done"]))

    def test_current_targets_include_later_canceled_and_replaced(self):
        f = self.facts
        ineligible_active = f["active"] & ~f["eligible"]
        self.assertGreater(int(ineligible_active.sum()), 0)
        np.testing.assert_array_equal(f["finish_target"][ineligible_active],
                                     (4*f["version"] + f["payload"])[ineligible_active])
        self.assertTrue(np.all(f["finish_target"][~f["active"]] == -1))
        c = d.counts(f)
        self.assertEqual(c["active_content"], int(f["active"].sum()))
        self.assertGreater(c["active_ineligible_content"], 0)
        self.assertGreater(c["begin_ineligible"], 0)
        self.assertEqual(c["active_content"], c["active_eligible_content"] + c["active_ineligible_content"])
        # Native earliest starts the initial NEW despite its hidden future
        # cancellation/replacement, rather than teaching a clairvoyant WAIT.
        doomed_new = (f["event_kind"] == 1) & ~f["event_eligible"]
        self.assertGreater(int(doomed_new.sum()), 0)
        self.assertTrue(np.all(f["mode_target"][doomed_new] == 1))
        self.assertTrue(np.any(f["event_kind"] == 2))
        self.assertTrue(np.any(f["event_kind"] == 3))

    def test_counts_partition_and_block_totals(self):
        f = self.facts
        full = d.counts(f)
        self.assertEqual(full["mode_WAIT"] + full["mode_BEGIN"] + full["mode_FINISH"], 384*2*2)
        self.assertEqual(full["begin_version_0"] + full["begin_version_1"], full["mode_BEGIN"])
        blocks = [d.counts(f, start, start+128) for start in range(0, 384, 128)]
        for key, value in full.items():
            self.assertEqual(sum(b[key] for b in blocks), value, key)

    def test_replay_restored_native_world_matches_every_tick(self):
        f = self.facts
        with d.mapping_scope(d.mapping_table(0, "related")):
            world = d.ContextWorld.from_state_dict(self.context["initial_world"], device="cpu")
            for tick in range(384):
                np.testing.assert_array_equal(world.observe().numpy(), f["obs"][tick])
                np.testing.assert_array_equal(world.oracle("earliest").numpy(), f["actions"][tick])
                world.act(torch.from_numpy(f["actions"][tick]))
                np.testing.assert_array_equal(world.last_result.numpy(), f["receipts"][tick])
            self.assertEqual(world.event_hash, self.context["events_sha256"])
            self.assertEqual(world.metrics(), self.context["metrics"])


if __name__ == "__main__":
    unittest.main()
