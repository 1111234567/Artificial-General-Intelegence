"""Small invariance tests; full real-source checks live in bank_audit.py."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from experiments.review_selection import bank
from experiments.experience_transfer import train_transfer as source


class BankTests(unittest.TestCase):
    def test_priority_cannot_select_by_model_loss_or_completion(self):
        row = dict(source_round=8, lane=1, slot=0, request_id=55)
        changed = dict(row, strict_completed=True, had_error=False, B_score=1e6, NLL=-1e6,
                       raw=3, finish_tick=99, actual_cue_action=0)
        self.assertEqual(bank.stable_priority(0, row), bank.stable_priority(0, changed))
        self.assertNotEqual(bank.stable_priority(0, row), bank.stable_priority(1, row))

    def test_private_labels_do_not_enter_hidden_reconstruction(self):
        torch.manual_seed(1207)
        model = source.plain.GRUAgent(width=9)
        public = torch.randn(17, 2, 56)
        anchor = torch.randn(2, 9)
        one = dict(obs=public, anchor=anchor, labels=torch.zeros(17, 2, 2, dtype=torch.long))
        two = deepcopy(one)
        two["labels"].fill_(10)
        first = bank.public_hidden(model, one["obs"], one["anchor"])
        second = bank.public_hidden(model, two["obs"], two["anchor"])
        self.assertTrue(torch.equal(first, second))
        h = anchor
        expected = [h]
        with torch.no_grad():
            for obs in public:
                _, h = model.step(obs, h)
                expected.append(h)
        # Tiny synthetic GEMM shapes can round differently from a time batch;
        # the production-width archived anchors have a separate exact audit.
        self.assertTrue(torch.allclose(first, torch.stack(expected), atol=2e-6, rtol=2e-6))

    def test_every_retained_example_has_only_its_request_three_rows(self):
        for seed in range(3):
            data = bank.load(seed)
            self.assertEqual(data["obs"].shape, (128, 256, 56))
            self.assertEqual(data["mask"].sum().item(), 768)
            for i, meta in enumerate(data["metadata"]):
                slot = meta["slot"]
                self.assertFalse(data["mask"][:, i, 1-slot].any())
                supervised = data["labels"][:, i][data["mask"][:, i]].tolist()
                self.assertEqual(supervised, [1+meta["version"], 0, 3+4*meta["version"]+meta["A_target"]])
                self.assertEqual(meta["cue_tick"] - meta["start_tick"], 32)
                self.assertEqual(meta["stop_tick"] - meta["start_tick"], 128)

    def test_conflict_definition_uses_real_old_new_answer_change(self):
        for seed, spec in source.SEEDS.items():
            data = bank.load(seed)
            for i, raw in enumerate(data["raw"].tolist()):
                is_conflict = spec["A"][raw] != spec["B"][raw]
                self.assertEqual(bool(data["conflict"][i]), is_conflict)
                if is_conflict:
                    self.assertEqual(spec["A"][raw], spec["C"][raw])
            self.assertEqual(torch.bincount(data["raw"]).tolist(), [64,64,64,64])


if __name__ == "__main__":
    unittest.main()
