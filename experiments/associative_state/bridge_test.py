"""Causal-boundary and public-information tests for the symbol bridge."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.bridge import (
    PublicReceiptBinder, RandomMappingWorld, translate,
)
from experiments.lifelong.world import ContextWorld, FAMILY_MAPPINGS


def blank(batch=1):
    return torch.zeros(batch, 56)


def cue(obs, *, job=0, kind=1, raw=0, family=0, version=0):
    features = obs.reshape(len(obs), 2, 28)
    features[:, job, kind - 1] = 1
    features[:, job, 3 + version] = 1
    if kind != 3:
        features[:, job, 5 + raw] = 1
        features[:, job, 24 + family] = 1
    return obs


def receipt(obs, action, *, job=0, success=True):
    features = obs.reshape(len(obs), 2, 28)
    features[:, job, 11 + action] = 1
    features[:, job, 22 if success else 23] = 1
    return obs


def mappings(batch=2):
    return torch.tensor([[[2, 3, 0, 1], [3, 2, 1, 0]],
                         [[1, 2, 3, 0], [2, 0, 3, 1]]])[:batch]


def same_tree(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
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


class ReceiptTests(unittest.TestCase):
    def test_receipt_precedes_same_tick_new_cue(self):
        binder = PublicReceiptBinder(1)
        binder.consume(cue(blank(), raw=2, family=0, version=1))
        simultaneous = receipt(cue(blank(), raw=1, family=3, version=0), 10)
        writes = binder.consume(simultaneous)
        self.assertEqual([(k.item(), v.item(), m.item()) for k, v, m in writes],
                         [(2, 3, True), (0, 0, False)])
        keys, values, mask = binder.consume(receipt(blank(), 5))[0]
        self.assertEqual((keys.item(), values.item(), mask.item()), (5, 2, True))
        self.assertFalse(binder.valid.any())

    def test_rejected_finish_and_successful_begin_do_not_write(self):
        binder = PublicReceiptBinder(1)
        binder.consume(cue(blank(), raw=3))
        self.assertFalse(binder.consume(receipt(blank(), 6, success=False))[0][2].any())
        self.assertTrue(binder.valid[0, 0])
        self.assertFalse(binder.consume(receipt(blank(), 1))[0][2].any())
        self.assertTrue(binder.consume(receipt(blank(), 6))[0][2].all())
        self.assertFalse(binder.consume(receipt(blank(), 6))[0][2].any())

    def test_cancel_clears_and_replace_changes_version(self):
        binder = PublicReceiptBinder(1)
        binder.consume(cue(blank(), raw=0, version=0))
        binder.consume(cue(blank(), kind=2, raw=2, family=3, version=1))
        self.assertFalse(binder.consume(receipt(blank(), 4))[0][2].any())
        keys, values, mask = binder.consume(receipt(blank(), 9))[0]
        self.assertEqual((keys.item(), values.item(), mask.item()), (6, 2, True))
        binder.consume(cue(blank(), raw=1))
        binder.consume(cue(blank(), kind=3))
        self.assertFalse(binder.consume(receipt(blank(), 4))[0][2].any())

    def test_batched_job_order_and_restore_without_aliases(self):
        binder = PublicReceiptBinder(2)
        binder.consume(cue(cue(blank(2), job=0, raw=1), job=1, raw=2, family=3))
        state = binder.state_dict()
        restored = PublicReceiptBinder.from_state_dict(state)
        obs = receipt(receipt(blank(2), 4, job=0), 5, job=1)
        actual, expected = binder.consume(obs), restored.consume(obs)
        for a, b in zip(actual, expected):
            same_tree(self, a, b)
        self.assertEqual(actual[0][0].tolist(), [1, 1])
        self.assertEqual(actual[1][0].tolist(), [6, 6])
        self.assertTrue(state["tensors"]["valid"].all())
        self.assertEqual(binder.state_report()["extra_state_scalars_per_actor"], 8)
        self.assertEqual(binder.state_report()["extra_state_bytes_per_actor"], 50)

    def test_only_designated_contexts_are_bound(self):
        with self.assertRaises(ValueError):
            PublicReceiptBinder(1).consume(cue(blank(), family=1))


class TranslationTests(unittest.TestCase):
    def test_inverse_prevents_double_mapping_and_keeps_other_features(self):
        obs = cue(cue(blank(), job=0, raw=3, family=3),
                  job=1, kind=2, raw=0, family=1, version=1)
        obs[0, 9:11] = torch.tensor([12., 2.])
        original = obs.clone()
        translated = translate(obs, torch.tensor([[0, 1]]))
        selected = translated.reshape(1, 2, 28)[..., 5:9].argmax(-1)
        self.assertEqual(selected.tolist(), [[3, 0]])
        self.assertEqual([FAMILY_MAPPINGS[f][c] for f, c in zip((3, 1), selected[0])],
                         [0, 1])
        unchanged = torch.ones(56, dtype=torch.bool)
        unchanged[5:9] = False
        unchanged[33:37] = False
        self.assertTrue(torch.equal(translated[:, unchanged], obs[:, unchanged]))
        self.assertTrue(torch.equal(obs, original))

    def test_idle_and_cancel_are_bitwise_unchanged(self):
        obs = cue(receipt(blank(), 10), job=1, kind=3, version=1)
        self.assertTrue(torch.equal(obs, translate(obs, torch.tensor([[3, 2]]))))


class WorldTests(unittest.TestCase):
    def make_world(self, **kwargs):
        spec = dict(seed=71, batch_size=2, n_jobs=2, dt=.25,
                    family_allowed=(0, 3), mappings=mappings())
        spec.update(kwargs)
        return RandomMappingWorld(**spec)

    def test_dynamic_targets_do_not_leak_into_public_cues(self):
        original_rules = deepcopy(FAMILY_MAPPINGS)
        dynamic = self.make_world()
        original = ContextWorld(seed=71, batch_size=2, n_jobs=2, dt=.25,
                                family_allowed=(0, 3))
        changed = 0
        for _ in range(256):
            obs, reference = dynamic.observe(), original.observe()
            self.assertTrue(torch.equal(obs, reference))
            for actor, job in ((dynamic.event_kind == 1) | (dynamic.event_kind == 2)).nonzero():
                context_index = int(dynamic.event_family[actor, job] == 3)
                raw = dynamic.raw_event_payload[actor, job]
                expected = mappings()[actor, context_index, raw]
                self.assertEqual(dynamic.payload[actor, job], expected)
                changed += int(dynamic.payload[actor, job] != original.payload[actor, job])
            actions = torch.zeros(2, 2, dtype=torch.long)
            dynamic.act(actions)
            original.act(actions)
        self.assertGreater(changed, 0)
        self.assertEqual(original_rules, FAMILY_MAPPINGS)

    def test_actual_success_receipts_recover_private_mapping(self):
        world = self.make_world(cancel=0., replace=0.)
        binder = PublicReceiptBinder(2)
        seen = 0
        for _ in range(256):
            obs = world.observe()
            for keys, values, mask in binder.consume(obs):
                for actor in mask.nonzero().flatten():
                    key = int(keys[actor])
                    self.assertEqual(values[actor], mappings()[actor, key // 4, key % 4])
                    seen += 1
            world.act(world.oracle())
        self.assertGreater(seen, 10)

    def test_actions_do_not_change_external_stream(self):
        teacher, idle = self.make_world(), self.make_world()
        for _ in range(256):
            left, right = teacher.observe().reshape(2, 2, 28), idle.observe().reshape(2, 2, 28)
            self.assertTrue(torch.equal(left[..., :11], right[..., :11]))
            self.assertTrue(torch.equal(left[..., 24:], right[..., 24:]))
            self.assertEqual(teacher.event_hash, idle.event_hash)
            teacher.act(teacher.oracle())
            idle.act(torch.full((2, 2), 10, dtype=torch.long))

    def test_checkpoint_and_deepcopy_resume_both_causal_boundaries(self):
        for after_action in (False, True):
            world = self.make_world()
            for _ in range(36):
                world.observe()
                world.act(world.oracle())
            world.observe()
            if after_action:
                world.act(world.oracle())
            restored = RandomMappingWorld.from_state_dict(world.state_dict())
            cloned, copied = world.clone(), deepcopy(world)
            for alternative in (restored, cloned, copied):
                same_tree(self, world.state_dict(), alternative.state_dict())
            if not after_action:
                actions = world.oracle()
                for candidate in (world, restored, cloned, copied):
                    candidate.act(actions)
            for _ in range(40):
                expected = world.observe()
                for alternative in (restored, cloned, copied):
                    self.assertTrue(torch.equal(expected, alternative.observe()))
                actions = world.oracle()
                for candidate in (world, restored, cloned, copied):
                    candidate.act(actions)
            for alternative in (restored, cloned, copied):
                same_tree(self, world.state_dict(), alternative.state_dict())

    def test_invalid_mapping_and_policy_are_rejected(self):
        invalid = mappings()
        invalid[0, 0, 0] = invalid[0, 0, 1]
        with self.assertRaises(ValueError):
            self.make_world(mappings=invalid)
        with self.assertRaises(ValueError):
            self.make_world(family_allowed=(0, 1))
        world = self.make_world()
        world.observe()
        world.act(torch.zeros(2, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            world.set_family_policy((1,))

    def test_actor_mapping_is_independent_of_batch_placement(self):
        batched = self.make_world(actor_ids=(19, 7))
        alone = self.make_world(batch_size=1, actor_ids=(7,), mappings=mappings()[1:])
        for _ in range(128):
            left, right = batched.observe(), alone.observe()
            self.assertTrue(torch.equal(left[1:], right))
            self.assertTrue(torch.equal(batched.payload[1:], alone.payload))
            batched.act(torch.zeros(2, 2, dtype=torch.long))
            alone.act(torch.zeros(1, 2, dtype=torch.long))

    def test_default_world_keeps_56_inputs_and_tampered_target_is_rejected(self):
        world = RandomMappingWorld(batch_size=1)
        self.assertEqual(tuple(world.observe().shape), (1, 56))
        while not (world.current_family >= 0).any():
            world.act(torch.zeros(1, 2, dtype=torch.long))
            world.observe()
        state = world.state_dict()
        actor, job = (world.current_family >= 0).nonzero()[0]
        state["tensors"]["payload"][actor, job] = (state["tensors"]["payload"][actor, job] + 1) % 4
        with self.assertRaises(ValueError):
            RandomMappingWorld.from_state_dict(state)

    def test_revision_preserves_pending_and_restore_then_changes_future_command(self):
        world = self.make_world(cancel=0., replace=1.)
        while True:
            world.observe()
            world.act(torch.zeros(2, 2, dtype=torch.long))
            if (world.event_kind == 1).any():
                break
        payload, applied = world.payload.clone(), world.applied_mappings.clone()
        public_hash = world.event_hash
        old_scheduler = world.state_dict()["scheduler"]
        revised = (world.private_mappings + 1) % 4
        world.set_mappings(revised)
        self.assertTrue(torch.equal(world.payload, payload))
        self.assertTrue(torch.equal(world.applied_mappings, applied))
        self.assertEqual(world.event_hash, public_hash)
        same_tree(self, world.state_dict()["scheduler"], old_scheduler)
        restored = RandomMappingWorld.from_state_dict(world.state_dict())
        same_tree(self, world.state_dict(), restored.state_dict())
        changed = 0
        for _ in range(128):
            self.assertTrue(torch.equal(world.observe(), restored.observe()))
            for actor, job in ((world.event_kind == 1) | (world.event_kind == 2)).nonzero():
                index = int(world.event_family[actor, job] == 3)
                self.assertTrue(torch.equal(world.applied_mappings[actor, job], revised[actor, index]))
                self.assertEqual(world.payload[actor, job], revised[actor, index, world.raw_payload[actor, job]])
                changed += 1
            actions = world.oracle()
            world.act(actions)
            restored.act(actions)
        self.assertGreater(changed, 0)
        same_tree(self, world.state_dict(), restored.state_dict())

    def test_revision_does_not_reinterpret_pending_cancel(self):
        world = self.make_world(cancel=1., replace=0.)
        saw_cancel = False
        for _ in range(128):
            world.observe()
            if (world.event_kind == 3).any():
                for actor, job in (world.event_kind == 3).nonzero():
                    expected = world.applied_mappings[actor, job, world.raw_event_payload[actor, job]]
                    self.assertEqual(world.event_payload[actor, job], expected)
                    self.assertEqual(world.payload[actor, job], expected)
                saw_cancel = True
            world.act(torch.zeros(2, 2, dtype=torch.long))
            world.set_mappings((world.private_mappings + 1) % 4)
        self.assertTrue(saw_cancel)


if __name__ == "__main__":
    unittest.main()
