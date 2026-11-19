"""Small checks of genome lineage, CE isolation and actual-world scoring."""
from copy import deepcopy
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from experiments.learned_binding import audit, common, data, pretrain
from experiments.learned_binding.frozen_trace import FrozenTracePolicy
from experiments.seed_evolution import core


class GenomeTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_genome_seed_does_not_choose_executor_or_inherit_adjacent_seed(self):
        rng_before = torch.random.get_rng_state()
        first = core.new_genome(0, 78100)
        next_seed = core.new_genome(0, 78101)
        other_source = core.new_genome(1, 78100)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertEqual(core.genome_hash(first), core.genome_hash(core.new_genome(0, 78100)))
        self.assertNotEqual(core.genome_hash(first), core.genome_hash(next_seed))
        self.assertNotEqual(core.genome_hash(first), core.genome_hash(other_source))
        for name in first.keys() - {"original_mappings"}:
            self.assertTrue(torch.equal(first[name], other_source[name]), name)

        with torch.random.fork_rng(devices=[]):
            executor = common.executor(0)
            torch.random.default_generator.manual_seed(78101)
            independent = FrozenTracePolicy(executor=executor, original_mappings=common.mapping_table(0, "related"))
        model = core.model_from_genome(0, next_seed)
        common.assert_tree_equal(model.state_dict(), independent.state_dict())
        common.assert_tree_equal(model.executor.state_dict(), core.model_from_genome(0, first).executor.state_dict())
        self.assertEqual(model.parameter_report()["trainable_parameters"], 2258)
        with self.assertRaisesRegex(ValueError, "original_mappings"):
            core.model_from_genome(0, other_source)

    def test_mutation_only_perturbs_random_coordinates_and_copies_parent(self):
        parent = core.new_genome(0, 78100)
        before = deepcopy(parent)
        rng_before = torch.random.get_rng_state()
        child = core.mutate_genome(parent, 78101)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        common.assert_tree_equal(parent, before)
        common.assert_tree_equal(child, core.mutate_genome(parent, 78101))
        common.assert_tree_equal(parent, core.mutate_genome(parent, 78101, sigma=0))
        adjacent = core.new_genome(0, 78101)
        generator = torch.Generator().manual_seed(78101)
        mutable = core.ENCODER_WEIGHTS + core.ENCODER_BIASES + core.HEAD_WEIGHTS
        for name in mutable:
            noise = torch.randn(parent[name].shape, generator=generator)
            sd = core.ENCODER_INIT_SD if name.startswith("encoder.") else core.HEAD_INIT_SD
            if name in core.ENCODER_BIASES:
                noise[32:64] = 0
                self.assertTrue(torch.equal(child[name][32:64], torch.full((32,), 2.)))
            self.assertTrue(torch.equal(child[name], parent[name].add(noise, alpha=.15 * sd)), name)
            self.assertGreater(float((child[name] - parent[name]).norm()), 0., name)
            self.assertGreater(float((adjacent[name] - parent[name]).norm()),
                               4 * float((child[name] - parent[name]).norm()), name)
        for name in parent:
            self.assertNotEqual(parent[name].data_ptr(), child[name].data_ptr(), name)
            if name not in mutable:
                self.assertTrue(torch.equal(parent[name], child[name]), name)

    def test_fresh_adam_and_zero_state_reproduce_original_head_ce(self):
        genome = core.new_genome(0, 78100)
        before = deepcopy(genome)
        fixtures = []
        for index in range(2):
            sample = data.generate(0, batch=3, ticks=48, duration_range=(1., 2.),
                                   gap_range=(1., 2.), stream_id=index)
            fixtures.append((sample["obs"], sample["actions"]))
        originals = deepcopy(fixtures)
        with patch.object(core, "_training_streams", return_value=tuple(fixtures)):
            measured = core.inner_train(0, genome, steps=2, checkpoints=(0, 1, 2))
            rerun = core.inner_train(0, genome, steps=2, checkpoints=(0, 2))
            next_genome = core.mutate_genome(genome, 78101)
            next_run = core.inner_train(0, next_genome, steps=0, checkpoints=(0,))

        reference = core.model_from_genome(0, genome).train()
        trainable = [parameter for parameter in reference.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=.001)
        snapshots = measured["checkpoints"]
        self.assertEqual(snapshots[0]["optimizer"]["state"], {})
        self.assertEqual(next_run["checkpoints"][0]["optimizer"]["state"], {})
        for name, value in next_genome.items():
            self.assertTrue(torch.equal(next_run["checkpoints"][0]["model"][name], value), name)
        self.assertEqual(len(snapshots[0]["optimizer"]["param_groups"][0]["params"]), 9)
        for step, (obs, actions) in enumerate(fixtures, 1):
            logits, _ = reference(obs, reference.initial_state(obs.shape[1]))
            loss = pretrain.loss_for(logits, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.)
            optimizer.step()
            common.assert_tree_equal(reference.state_dict(), snapshots[step]["model"])
            common.assert_tree_equal(optimizer.state_dict(), snapshots[step]["optimizer"])
        common.assert_tree_equal(snapshots[2], rerun["checkpoints"][2])
        self.assertFalse(torch.equal(snapshots[0]["model"]["query.weight"], snapshots[2]["model"]["query.weight"]))
        for name, value in snapshots[0]["model"].items():
            if name.startswith(("encoder.", "executor.")):
                self.assertTrue(torch.equal(value, snapshots[2]["model"][name]), name)
        self.assertEqual(measured["metadata"]["stream_indices"], [0, 1])
        self.assertEqual(measured["metadata"]["initial_state"], dict(h=0, z=0, memory=0))
        common.assert_tree_equal(genome, before)
        common.assert_tree_equal(fixtures, originals)

    def test_original_v1_schedule_switches_after_128_dense_updates(self):
        self.assertEqual([core._stream_index(step) for step in range(64)], list(range(8)) * 8)
        self.assertEqual([core._stream_index(step) for step in range(64, 128)], list(range(8)) * 8)
        self.assertEqual([core._stream_index(step) for step in range(128, 512)], list(range(16)) * 24)


class ScoreTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.genome = core.new_genome(0, 78100)
        cls.model_state = core.model_from_genome(0, cls.genome).state_dict()
        cls.settings = dict(stream_ids=(901, 902), support_ticks=24, probe_ticks=480, actors=4)

    def test_equal_weight_per_conflict_family_and_stream(self):
        groups = [dict(A_conflict=dict(n=1, strict=1), B_conflict=dict(n=99, strict=0)),
                  dict(A_conflict=dict(n=10, strict=9), B_conflict=dict(n=5, strict=2))]
        reports = [(dict(fitness=core._conflict_fitness(row)), {}) for row in groups]
        self.assertEqual(reports[0][0]["fitness"], .5)
        self.assertEqual(reports[1][0]["fitness"], .65)
        with patch.object(core, "_score_stream", side_effect=reports):
            score = core.score_model(0, self.model_state, **self.settings)
        self.assertEqual(score["fitness"], .575)
        with self.assertRaisesRegex(ValueError, "B_conflict"):
            core._conflict_fitness(dict(A_conflict=dict(n=1, strict=0), B_conflict=dict(n=0, strict=0)))

    def test_own_scoring_reproducible_with_fixed_exogenous_worlds_and_real_actions(self):
        before = deepcopy(self.model_state)
        rng_before = torch.random.get_rng_state()
        with patch.object(data.RandomMappingWorld, "oracle", side_effect=AssertionError("unexpected oracle")):
            result = core.score_model(0, self.model_state, archive=True, **self.settings)
            compact = core.score_model(0, self.model_state, **self.settings)
        self.assertEqual({key: value for key, value in result.items() if key != "archives"}, compact)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        common.assert_tree_equal(self.model_state, before)
        self.assertNotEqual(compact["streams"][0]["sampling_seed"], compact["streams"][1]["sampling_seed"])
        self.assertTrue(set(compact["streams"][0]["actor_ids"]).isdisjoint(compact["streams"][1]["actor_ids"]))

        child = core.model_from_genome(0, core.mutate_genome(self.genome, 78101)).state_dict()
        alternative = core.score_model(0, child, **self.settings)
        self.assertNotEqual(result["parameter_hash"], alternative["parameter_hash"])
        for stream, other in zip(result["streams"], alternative["streams"]):
            for key in ("world_start_hash", "world_support_end_hash", "world_end_hash",
                        "support_exogenous_hash", "probe_exogenous_hash", "private_mapping_hash",
                        "sampling_seed", "actor_ids", "contexts"):
                self.assertEqual(stream[key], other[key], key)
            trace = result["archives"][stream["stream_id"]]
            support, probe = trace["support"], trace["probe"]
            common.assert_tree_equal(support["final"], probe["start"])
            support_arrays = {key: value.numpy() for key, value in support.items() if isinstance(value, torch.Tensor)}
            audit.check_block({}, support_arrays, support)
            recount = audit.recount_probe(probe["report"], probe["arrays"], probe)
            self.assertEqual(recount["groups"], stream["groups"])
            self.assertEqual(core._conflict_fitness(recount["groups"]), stream["fitness"])
            self.assertEqual(support["final"]["work"]["sampled_action_rows"], 24 * 4 * 2)
            self.assertEqual(support["final"]["work"]["teacher_action_rows"], 0)
            self.assertEqual(probe["start"]["work"]["sampled_action_rows"], probe["final"]["work"]["sampled_action_rows"])

        executor_key = next(name for name in before if name.startswith("executor."))
        before[executor_key].view(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "fixed source tensor"):
            core.score_model(0, before, **self.settings)

    def test_no_read_applies_during_support_and_probe_and_b_delay_preserves_state(self):
        settings = dict(self.settings, stream_ids=(901,))
        result = core.score_model(0, self.model_state, archive=True, read_enabled=False, b_ticks=40, **settings)
        trace = result["archives"][901]
        support, b_support, probe = (trace[name] for name in ("support", "b_support", "probe"))
        self.assertTrue(torch.equal(support["read_probs"], torch.full_like(support["read_probs"], .25)))
        self.assertTrue(torch.equal(b_support["read_probs"], torch.full_like(b_support["read_probs"], .25)))
        self.assertTrue(np.all(probe["arrays"]["read_probs"] == .25))
        self.assertGreater(float(b_support["final"]["state"]["memory"].norm()), 0.)
        for before, after in ((support["final"], b_support["start"]), (b_support["final"], probe["start"])):
            for name in ("state", "rng", "model", "work"):
                common.assert_tree_equal(before[name], after[name])
            self.assertEqual(before["world"]["tick"], after["world"]["tick"])
        contexts = support["start"]["world"]["random_mapping"]["contexts"]
        self.assertEqual(b_support["start"]["world"]["context"]["family_allowed"], contexts[1:])
        self.assertEqual(probe["start"]["world"]["context"]["family_allowed"], contexts)

    def test_teacher_support_uses_actual_midpoint_actions_and_leaves_probe_unsampled(self):
        settings = dict(self.settings, stream_ids=(901,))
        result = core.score_model(0, self.model_state, archive=True, teacher_support=True, **settings)
        trace = result["archives"][901]
        support = trace["support"]
        arrays = {key: value.numpy() for key, value in support.items() if isinstance(value, torch.Tensor)}
        audit.check_block({}, arrays, support, teacher=True)
        self.assertEqual(trace["probe"]["final"]["work"]["sampled_action_rows"], 0)
        self.assertEqual(trace["probe"]["final"]["work"]["teacher_action_rows"], 24 * 4 * 2)


if __name__ == "__main__":
    main()
