"""Causal public56, actual receipt and independent-stream tests."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.associative_state.bridge import RandomMappingWorld
from experiments.learned_binding.data import generate, make_world, reward_from_observation


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


class RewardTests(unittest.TestCase):
    def test_reward_uses_public_action_and_result_only(self):
        obs = torch.zeros(11, 56, requires_grad=True)
        with torch.no_grad():
            features = obs.reshape(11, 2, 28)
            for action in range(1, 11):
                features[action, 0, 11 + action] = 1
                features[action, 0, 22] = 1
                features[action, 1, 11 + action] = 1
                features[action, 1, 23] = 1
        reward = reward_from_observation(obs)
        self.assertEqual(tuple(reward.shape), (11, 2))
        self.assertTrue(torch.equal(reward[:, 0], torch.tensor([0., 0., 0.] + [1.] * 8)))
        self.assertTrue(torch.equal(reward[:, 1], torch.tensor([0.] + [-.1] * 10)))
        self.assertFalse(reward.requires_grad)
        # Arbitrary cue/version/context features cannot change the reward.
        modified = obs.detach().clone().reshape(11, 2, 28)
        modified[..., :11] = torch.rand(11, 2, 11)
        modified[..., 24:] = torch.rand(11, 2, 4)
        self.assertTrue(torch.equal(reward, reward_from_observation(modified.reshape(11, 56))))

    def test_absent_and_begin_receipts_are_zero_and_leading_dims_work(self):
        obs = torch.zeros(3, 4, 56)
        obs[0, 1, 11 + 2] = 1
        obs[0, 1, 22] = 1
        self.assertEqual(tuple(reward_from_observation(obs).shape), (3, 4, 2))
        self.assertEqual(float(reward_from_observation(obs).sum()), 0.)
        for bad in (torch.zeros(56), torch.zeros(1, 55), torch.zeros(1, 56, dtype=torch.long)):
            with self.assertRaises(ValueError):
                reward_from_observation(bad)

    def test_real_rejection_and_success_reward_previous_action(self):
        world = make_world(0, batch=2)
        world.observe()
        action = torch.full((2, 2), 10, dtype=torch.long)
        world.act(action)
        receipt = world.observe()
        self.assertTrue(torch.equal(reward_from_observation(receipt), torch.full((2, 2), -.1)))
        world.act(world.oracle("midpoint"))
        saw_finish = False
        for _ in range(128):
            receipt = world.observe()
            public = receipt.reshape(2, 2, 28)
            code = public[..., 11:22].argmax(-1)
            expected = ((code >= 3) & (public[..., 22] > .5)).float()
            self.assertTrue(torch.equal(reward_from_observation(receipt), expected))
            saw_finish |= bool(expected.any())
            world.act(world.oracle("midpoint"))
        self.assertTrue(saw_finish)


class TeacherStreamTests(unittest.TestCase):
    def test_stream_reproducible_and_randomness_domains_are_independent(self):
        first = generate(1, batch=3, ticks=64, stream_id=4, random_seed=17)
        same_tree(self, first, generate(1, batch=3, ticks=64, stream_id=4, random_seed=17))
        second = generate(1, batch=3, ticks=64, stream_id=5, random_seed=17)
        self.assertFalse(torch.equal(first["obs"], second["obs"]))
        self.assertNotEqual(first["metadata"]["world_seed"], second["metadata"]["world_seed"])
        self.assertNotEqual(first["metadata"]["actor_ids"], second["metadata"]["actor_ids"])
        alternate = generate(1, batch=3, ticks=64, stream_id=4, random_seed=18)
        self.assertFalse(torch.equal(first["obs"], alternate["obs"]))
        evaluation = generate(1, batch=3, ticks=64, stream_id=4, random_seed=17, split="eval")
        self.assertEqual(first["metadata"]["world_seed"] >> 62, 1)
        self.assertEqual(evaluation["metadata"]["world_seed"] >> 62, 2)
        self.assertTrue(set(first["metadata"]["actor_ids"]).isdisjoint(evaluation["metadata"]["actor_ids"]))

    def test_new_maps_have_exactly_two_shared_and_two_conflicting_symbols(self):
        world = make_world(2, batch=32, stream_id=8)
        rules = world.private_mappings
        self.assertTrue(torch.equal(rules[:, 0].sort(-1).values, torch.arange(4).expand(32, 4)))
        self.assertTrue(torch.equal(rules[:, 1], rules[:, 0][:, [1, 0, 2, 3]]))
        self.assertTrue(torch.equal((rules[:, 0] == rules[:, 1]).sum(-1), torch.full((32,), 2)))
        self.assertGreater(len(torch.unique(rules[:, 0], dim=0)), 8)
        self.assertFalse(torch.equal(rules, make_world(2, batch=32, stream_id=9).private_mappings))
        prefix = make_world(2, batch=3, stream_id=8)
        self.assertEqual(world.actor_ids[:3], prefix.actor_ids)
        self.assertTrue(torch.equal(rules[:3], prefix.private_mappings))

    def test_labels_are_executed_and_align_with_next_public_receipt(self):
        stream = generate(0, batch=4, ticks=192)
        obs, actions = stream["obs"], stream["actions"]
        self.assertEqual(tuple(obs.shape), (192, 4, 56))
        self.assertEqual(tuple(actions.shape), (192, 4, 2))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.long)
        public = obs.reshape(192, 4, 2, 28)
        self.assertFalse(bool(public[0, ..., 11:24].any()))
        previous = actions[:-1]
        receipt_code = public[1:, ..., 11:22].argmax(-1)
        self.assertTrue(torch.equal(receipt_code, previous))
        self.assertTrue(torch.equal(public[1:, ..., 22] > .5, previous != 0))
        self.assertFalse(bool(public[..., 23].any()))
        self.assertTrue(torch.equal(reward_from_observation(obs[1:]), (previous >= 3).float()))
        self.assertGreater(int((previous >= 3).sum()), 10)
        # Every archived observation is obtained before consulting its label.
        replay = RandomMappingWorld.from_state_dict(stream["initial_world"])
        for observation, label in zip(obs, actions):
            self.assertTrue(torch.equal(replay.observe(), observation))
            before = deepcopy(replay.state_dict())
            self.assertTrue(torch.equal(replay.oracle("midpoint"), label))
            same_tree(self, before, replay.state_dict())
            replay.act(label)
        same_tree(self, replay.state_dict(), stream["final_world"])

    def test_private_or_future_targets_cannot_change_unexecuted_public_cues(self):
        original = make_world(1, batch=3, stream_id=3)
        changed = original.clone()
        changed.set_mappings((original.private_mappings + 1) % 4)
        unequal_targets = 0
        for _ in range(96):
            # Labels/private rules differ, but public cues remain raw and causal.
            left, right = original.observe(), changed.observe()
            self.assertTrue(torch.equal(left, right))
            unequal_targets += int((original.payload != changed.payload).sum())
            original.oracle("midpoint")
            changed.oracle("midpoint")
            same_action = torch.zeros(3, 2, dtype=torch.long)
            original.act(same_action)
            changed.act(same_action)
        self.assertGreater(unequal_targets, 0)

    def test_both_halves_keep_ab_and_repeat_experiences_without_a_reset(self):
        stream = generate(2, batch=4, ticks=1024)
        contexts = stream["metadata"]["contexts"]
        public = stream["obs"].reshape(1024, 4, 2, 28)
        for segment in (public[:512], public[512:]):
            command = (segment[..., 0] > .5) | (segment[..., 1] > .5)
            family = segment[..., 24:].argmax(-1)
            self.assertEqual(set(family[command].tolist()), set(contexts))
            self.assertGreater(int((reward_from_observation(segment.reshape(512, 4, 56)) > 0).sum()), 50)
        self.assertEqual(stream["initial_world"]["tick"], -1)
        self.assertEqual(stream["final_world"]["tick"], 1023)
        self.assertTrue(torch.equal(stream["initial_world"]["random_mapping"]["mappings"],
                                    stream["final_world"]["random_mapping"]["mappings"]))

    def test_saved_initial_and_final_worlds_resume_actual_stream(self):
        stream = generate(0, batch=2, ticks=48)
        archive = BytesIO()
        torch.save(stream, archive)
        archive.seek(0)
        loaded = torch.load(archive, weights_only=False)
        same_tree(self, stream, loaded)
        resumed = RandomMappingWorld.from_state_dict(loaded["final_world"])
        continuous = RandomMappingWorld.from_state_dict(stream["initial_world"])
        for action in stream["actions"]:
            continuous.observe()
            continuous.act(action)
        for _ in range(32):
            self.assertTrue(torch.equal(resumed.observe(), continuous.observe()))
            action = continuous.oracle("midpoint")
            resumed.act(action)
            continuous.act(action)
        same_tree(self, resumed.state_dict(), continuous.state_dict())

    def test_invalid_stream_specs_fail(self):
        for kwargs in (dict(seed=3), dict(seed=True), dict(seed=0, batch=0),
                       dict(seed=0, ticks=0), dict(seed=0, stream_id=-1),
                       dict(seed=0, random_seed=-1), dict(seed=0, split="test")):
            with self.assertRaises(ValueError):
                generate(**kwargs)


if __name__ == "__main__":
    unittest.main()
