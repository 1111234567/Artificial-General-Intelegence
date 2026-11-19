"""Request-level checks for the immutable paired past-experience library."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.episodic_review import bank as b
from experiments.experience_transfer.train_transfer import SEEDS


class PairedPastBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is the real already-learned archive, not regenerated synthetic
        # future data. One fixture also exercises long prehistory and REPLACE.
        cls.facts = b.load_source_facts(0)
        cls.rows, cls.selection = b.select_pairs(cls.facts, 0)
        cls.bank = b.materialize(cls.facts, cls.rows, 0)

    def test_fixed_selection_is_unique_raw_balanced_and_physically_paired(self):
        again, report = b.select_pairs(self.facts, 0)
        self.assertEqual(again, self.rows)
        self.assertEqual(report, self.selection)
        self.assertEqual(report["selected_pairs"], 256)
        for family in ("A", "B"):
            rows = self.rows[family]
            self.assertEqual(len({(r["actor_lane"], r["target_slot"], r["rid"]) for r in rows}), 256)
            self.assertEqual([sum(r["raw"] == raw for r in rows) for raw in range(4)], [64]*4)
            self.assertTrue(all(r["cue_tick"] >= 512 and r["anchor_start"]+128 <= 33280 for r in rows))
            self.assertTrue(any(r["cue_kind"] == 2 for r in rows))
        for left, right in zip(self.rows["A"], self.rows["B"]):
            self.assertEqual(tuple(left[k] for k in b.PAIR_KEY), tuple(right[k] for k in b.PAIR_KEY))
            self.assertEqual(left["pair_id"], right["pair_id"])

    def test_native_observations_and_labels_are_not_counterfactual_edits(self):
        f = self.facts
        for family in ("A", "B"):
            g = self.bank[family]
            for index, row in enumerate(self.rows[family]):
                take, lane = slice(row["anchor_start"], row["anchor_start"]+128), row["actor_lane"]
                np.testing.assert_array_equal(g["obs"][:, index], f["obs"][take, lane])
                np.testing.assert_array_equal(g["actions"][:, index], f["actions"][take, lane])
                slot, cue, finish = row["target_slot"], row["cue_tick"], row["finish_tick"]
                self.assertEqual(f["event_id"][cue, lane, slot], row["rid"])
                self.assertEqual(f["receipts"][cue, lane, slot], 1)
                self.assertEqual(f["receipts"][finish, lane, slot], 1)
                expected_payload = SEEDS[0][family][row["raw"]]
                self.assertEqual(f["actions"][finish, lane, slot], 3+4*row["version"]+expected_payload)

    def test_all_96_postcue_rows_in_named_slot_are_supervised_with_real_idle_tail(self):
        for family in ("A", "B"):
            g = self.bank[family]
            for index, row in enumerate(self.rows[family]):
                mask = g["action_mask"][:, index]
                self.assertFalse(mask[:32].any())
                self.assertFalse(mask[:, 1-row["target_slot"]].any())
                self.assertEqual(int(mask.sum()), 96)
                time_mask = g["time_mask"][:, index]
                self.assertFalse(time_mask[32].any())
                self.assertEqual(int(time_mask.sum()), row["supervised_length"]-1)
                expected = mask.copy()
                expected[32] = False
                expected[32+row["supervised_length"]:] = False
                np.testing.assert_array_equal(time_mask, expected)
                slot, lane = row["target_slot"], row["actor_lane"]
                tail = slice(row["finish_tick"]+1, row["anchor_start"]+128)
                self.assertFalse(self.facts["active"][tail, lane, slot].any())
                self.assertFalse(self.facts["event_kind"][tail, lane, slot].any())
                self.assertFalse(self.facts["actions"][tail, lane, slot].any())
            self.assertFalse(g["time_target"][~g["time_mask"]].any())
        np.testing.assert_array_equal(self.bank["A"]["action_mask"], self.bank["B"]["action_mask"])
        np.testing.assert_array_equal(self.bank["A"]["time_mask"], self.bank["B"]["time_mask"])
        for key in ("action_rows", "WAIT", "BEGIN", "FINISH", "action_weight_sum", "time_rows",
                    "time_positive", "time_zero", "time_negative"):
            self.assertEqual(b.count_rows(self.bank["A"])[key], b.count_rows(self.bank["B"])[key])

    def test_canceled_or_noncompleted_past_requests_cannot_enter_library(self):
        accepted_finish = {(int(lane), int(slot), int(self.facts["rid"][tick, lane, slot]))
            for tick, lane, slot in zip(*np.nonzero((self.facts["actions"] >= 3) & (self.facts["receipts"] == 1)))}
        for rows in self.rows.values():
            self.assertTrue(all((r["actor_lane"], r["target_slot"], r["rid"]) in accepted_finish for r in rows))
        # Remove one selected actual successful receipt: its command is no
        # longer an eligible completed memory, without consulting future flags.
        selected = self.rows["A"][0]
        key = selected["finish_tick"], selected["actor_lane"], selected["target_slot"]
        original = self.facts["receipts"][key]
        try:
            self.facts["receipts"][key] = -1
            pool = b.candidates(self.facts, 0)
            identities = {(r["actor_lane"], r["target_slot"], r["rid"])
                for group in pool["A"].values() for r in group}
            self.assertNotIn((selected["actor_lane"], selected["target_slot"], selected["rid"]), identities)
        finally:
            self.facts["receipts"][key] = original

    def test_ledger_rejects_wrong_identity_ready_and_action(self):
        row = self.rows["A"][0]
        wrong = {**row, "rid": row["rid"]+2}
        with self.assertRaises(AssertionError):
            b.request_ledger(self.facts, wrong, 0, "A")
        for name, tick, amount in (("ready", row["cue_tick"]+1, .125),
                                   ("actions", row["finish_tick"], -1)):
            key = tick, row["actor_lane"], row["target_slot"]
            original = self.facts[name][key]
            try:
                self.facts[name][key] += amount
                with self.assertRaises(AssertionError):
                    b.request_ledger(self.facts, row, 0, "A")
            finally:
                self.facts[name][key] = original

    def test_schedule_has_four_presentations_and_is_family_independent(self):
        schedule = self.bank["schedule"]
        self.assertEqual(schedule.shape, (32, 32))
        np.testing.assert_array_equal(np.bincount(schedule.ravel(), minlength=256), np.full(256, 4))
        for cycle in schedule.reshape(4, 256):
            np.testing.assert_array_equal(np.sort(cycle), np.arange(256))
        self.assertFalse(np.array_equal(schedule, b.replay_schedule(1)))

    def test_conflict_finish_labels_differ_and_shared_finish_labels_agree(self):
        for index, left in enumerate(self.rows["A"]):
            right = self.rows["B"][index]
            a = self.bank["A"]["actions"][32+left["supervised_length"]-1, index, left["target_slot"]]
            z = self.bank["B"]["actions"][32+right["supervised_length"]-1, index, right["target_slot"]]
            self.assertEqual(a == z, left["raw"] in SEEDS[0]["shared_raw"])

    def test_public_history_loader_and_bank_contain_no_hidden_anchor(self):
        history = b.load_history(0)
        self.assertEqual(history.dtype, np.float32)
        self.assertEqual(history.shape, (33280, 32, 56))
        np.testing.assert_array_equal(history, self.facts["obs"])
        for family in ("A", "B"):
            self.assertEqual(set(self.bank[family]), set(b.ARRAY_FIELDS+b.VECTOR_FIELDS))
            self.assertFalse(any("hidden" in key for key in self.bank[family]))


if __name__ == "__main__":
    unittest.main()
