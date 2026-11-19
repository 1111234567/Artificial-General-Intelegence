"""Temporary random-model tests; no trained experiment artifact is loaded."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.shared import SharedAssociationMemory
from experiments.learn_remember.gru_train import GRUAgent
from experiments.learned_binding import common, data
from experiments.learned_binding.interaction import Subject, probe
from experiments.learned_binding.model import NativeMemoryPolicy


def same_tree(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            same_tree(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            same_tree(test, x, y)
    else:
        test.assertEqual(left, right)


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(8765)
        self.native = NativeMemoryPolicy(original_mappings=common.mapping_table(0, "related")).eval()
        self.executor = deepcopy(self.native.executor)
        manual = SharedAssociationMemory(kind="delta")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "manual.pt"
        torch.save(dict(model=manual.state_dict(),
                        metadata=dict(model_config=dict(kind="delta", key_dim=16, n_keys=8, n_values=4))), path)
        settings = dict(common.SETTINGS, actors=2, support_duration=[1., 1.], support_gap=[1., 1.])
        self.patches = [patch.dict(common.SETTINGS, settings),
                        patch.object(common, "load_native", side_effect=lambda seed: deepcopy(self.native)),
                        patch.object(common, "executor", side_effect=lambda seed: deepcopy(self.executor)),
                        patch.object(common, "manual_checkpoint", return_value=path)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def subject(self, kind="learned"):
        return Subject(0, kind)

    def alternate_wait_and_rejected_finish(self, subject):
        original = subject.model.step

        def step(*args, **kwargs):
            logits, state, diagnostics = original(*args, **kwargs)
            logits = torch.full_like(logits, -1000.)
            # FINISH before any BEGIN is always rejected; WAIT has no receipt.
            logits[..., 0 if subject.work["model_steps"] % 2 == 0 else 10] = 0.
            return logits, state, diagnostics

        subject.model.step = step

    def test_last_action_reward_and_pending_block_continuity(self):
        subject = self.subject()
        initial = subject.snapshot()
        self.alternate_wait_and_rejected_finish(subject)
        first = subject.rollout_block(2, sample=False)
        self.assertTrue(torch.equal(first["actions"][:, 0, 0], torch.tensor([0, 10])))
        self.assertTrue(torch.equal(first["rewards"][0], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(first["rewards"][-1], torch.full((2, 2), -.1)))
        self.assertTrue(torch.equal(first["rewards"][-1], data.reward_from_observation(first["pending_obs"])))
        self.assertEqual(subject.world.tick, 2)
        self.assertEqual(subject.work["ticks"], 2)
        self.assertEqual(subject.work["model_steps"], 2)
        second = subject.rollout_block(2, sample=False)
        self.assertTrue(torch.equal(second["obs"][0], first["pending_obs"]))
        self.assertEqual(subject.world.tick, 4)
        self.assertEqual(subject.work["ticks"], 4)
        final = subject.snapshot()
        subject.restore(initial)
        continuous = subject.rollout_block(4, sample=False)
        self.assertTrue(torch.equal(torch.cat((first["obs"], second["obs"])), continuous["obs"]))
        self.assertTrue(torch.equal(torch.cat((first["actions"], second["actions"])), continuous["actions"]))
        self.assertTrue(torch.equal(torch.cat((first["rewards"], second["rewards"])), continuous["rewards"]))
        same_tree(self, final, subject.snapshot())
        pending_tick = subject.world.tick
        before = subject.work["ticks"]
        closed = subject.close_pending(sample=False)
        self.assertIsNotNone(closed)
        self.assertEqual(subject.world.tick, pending_tick)
        self.assertEqual(subject.work["ticks"], before + 1)
        self.assertTrue(subject.world.state_dict()["acted"])
        self.assertIsNone(subject.pending_obs)
        self.assertIsNone(subject.close_pending())

    def test_snapshot_restore_replays_rng_actions_parameters_and_all_states(self):
        for kind in ("learned", "manual_hard", "manual_soft"):
            with self.subTest(kind=kind):
                subject = self.subject(kind)
                subject.support(7, sample=True)
                subject.rollout_block(3, sample=True)
                saved = subject.snapshot()
                expected = subject.rollout_block(9, sample=True)
                expected_final = subject.snapshot()
                subject.memory.add_(7.)
                subject.restore(saved)
                actual = subject.rollout_block(9, sample=True)
                same_tree(self, expected, actual)
                same_tree(self, expected_final, subject.snapshot())
                # Snapshot tensors cannot be aliased by subsequent controller updates.
                self.assertTrue(torch.equal(saved["state"]["memory"], expected["start"]["state"]["memory"]))

    def test_self_support_rollout_close_and_probe_never_call_oracle(self):
        for kind in ("learned", "manual_hard", "manual_soft"):
            with self.subTest(kind=kind):
                subject = self.subject(kind)
                with patch.object(subject.world, "oracle", side_effect=AssertionError("oracle forbidden")) as oracle:
                    subject.support(8, teacher=False, sample=True)
                    subject.rollout_block(5, sample=True)
                    subject.close_pending(sample=True)
                    probe(subject, ticks=12)
                    oracle.assert_not_called()
                self.assertEqual(subject.work["teacher_action_rows"], 0)

    def test_teacher_diagnostic_is_explicit_and_ends_after_action(self):
        subject = self.subject()
        with patch.object(subject.world, "oracle", wraps=subject.world.oracle) as oracle:
            result = subject.support(12, teacher=True, sample=True)
        self.assertEqual(oracle.call_count, 12)
        self.assertEqual(subject.work["sampled_action_rows"], 0)
        self.assertEqual(subject.work["teacher_action_rows"], 12 * 2 * 2)
        self.assertTrue(result["final"]["world"]["acted"])
        self.assertIsNone(subject.pending_obs)
        subject.world.set_family_policy((subject.contexts[1],))

    def test_no_write_keeps_M_for_all_subjects_while_other_state_advances(self):
        for kind in ("learned", "manual_hard", "manual_soft"):
            with self.subTest(kind=kind):
                subject = self.subject(kind)
                subject.memory.normal_(std=.1)
                memory, hidden = subject.memory.clone(), subject.hidden.clone()
                subject.support(13, teacher=True, write_enabled=False)
                subject.rollout_block(7, write_enabled=False)
                result = probe(subject, ticks=25, write_enabled=False)
                self.assertTrue(torch.equal(memory, subject.memory))
                self.assertFalse(torch.equal(hidden, subject.hidden))
                self.assertEqual(result["report"]["effective_gate_sum"], 0.)
                self.assertTrue(result["report"]["memory_readonly"])
                self.assertTrue(np.all(result["arrays"]["effective_gate"] == 0))

    def test_manual_hard_and_soft_share_teacher_evidence_but_keep_interfaces(self):
        hard, soft = self.subject("manual_hard"), self.subject("manual_soft")
        left = hard.support(64, teacher=True)
        right = soft.support(64, teacher=True)
        self.assertTrue(torch.equal(left["obs"], right["obs"]))
        self.assertTrue(torch.equal(left["actions"], right["actions"]))
        self.assertTrue(torch.equal(hard.memory, soft.memory))
        self.assertTrue(torch.equal(left["read_probs"].sum(-1), torch.ones_like(left["read_probs"][..., 0])))
        self.assertTrue(bool(((left["read_probs"] == 0) | (left["read_probs"] == 1)).all()))
        self.assertTrue(bool(((right["read_probs"] > 0) & (right["read_probs"] < 1)).all()))
        self.assertGreater(hard.work["manual_public_examples"], 0)
        self.assertIsNone(self.subject().binder)

    def test_collection_is_detached_and_native_replay_retains_gradients(self):
        subject = self.subject()
        block = subject.rollout_block(8, sample=True)
        self.assertFalse(block["logits"].requires_grad)
        logits, _ = subject.model(block["obs"], block["start"]["state"])
        self.assertTrue(logits.requires_grad)
        log_probability = logits.log_softmax(-1).gather(-1, block["actions"][..., None]).squeeze(-1)
        (-log_probability.mean()).backward()
        self.assertIsNotNone(subject.model.gate.weight.grad)
        self.assertGreater(float(subject.model.gate.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in subject.executor.parameters()))

    def test_probe_counts_mature_eligible_requests_that_never_begin(self):
        subject = self.subject()
        with torch.no_grad():
            subject.executor.head.weight.zero_()
            subject.executor.head.bias.fill_(-1000.)
            subject.executor.head.bias.reshape(2, 11)[:, 0] = 0.
        result = probe(subject, ticks=96)
        arrays, report = result["arrays"], result["report"]
        book, completed, seen_first = {}, [], set()
        for tick in range(96):
            for lane, job in np.argwhere(arrays["mature_mask"][tick]):
                key = int(lane), int(job), int(arrays["mature_rid"][tick, lane, job])
                if key in book:
                    completed.append(book.pop(key))
            for lane, job in np.argwhere(arrays["event_eligible"][tick]):
                key = int(lane), int(job), int(arrays["event_rid"][tick, lane, job])
                identity = int(lane), int(arrays["event_family"][tick, lane, job]), int(arrays["raw"][tick, lane, job])
                book[key] = identity not in seen_first
                seen_first.add(identity)
        self.assertGreater(report["groups"]["all"]["n"], 10)
        self.assertEqual(report["groups"]["all"]["n"], len(completed))
        self.assertEqual(len(report["right_censored"]), len(book))
        self.assertEqual(report["groups"]["all"]["ordinary"], 0)
        self.assertEqual(report["groups"]["all"]["strict"], 0)
        self.assertTrue(np.all(arrays["actions"] == 0))
        self.assertEqual(report["first_groups"]["all"]["n"], sum(completed))
        self.assertEqual(report["gate_statistics"]["all"]["count"], 96 * 2 * 2)
        self.assertEqual(report["gate_statistics"]["success_finish"]["count"], 0)
        self.assertEqual(report["idle_actions"], 0)
        self.assertEqual(report["final_work"]["ticks"] - report["start_work"]["ticks"], 96)
        self.assertEqual(report["groups"]["A"]["n"] + report["groups"]["B"]["n"], len(completed))

    def test_pending_observation_is_scored_once_by_probe(self):
        subject = self.subject()
        block = subject.rollout_block(3, sample=False)
        pending = block["pending_obs"].numpy()
        before = subject.work["model_steps"]
        result = probe(subject, ticks=5)
        self.assertTrue(np.array_equal(result["arrays"]["obs"][0], pending))
        self.assertTrue(result["report"]["start_had_pending_observation"])
        self.assertEqual(subject.work["model_steps"] - before, 5)
        self.assertIsNone(subject.pending_obs)
        self.assertTrue(subject.world.state_dict()["acted"])


if __name__ == "__main__":
    unittest.main()
