"""Small continuation and intervention checks; these do not run a study."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from experiments.continuity.run import (
    TrainConfig, Trainer, Variant, rng_state, state_digest, tensor_digest,
)
from experiments.integration.run import (
    ARMS, FORMAT_VERSION, IntegrationConfig, IntegrationTrainer,
    load_checkpoint, run_to_target, write_run_artifacts,
)


LOCAL_MODES = ("coact_live", "coact_mix", "coact_mix_shuffled", "coact_only_mix")


def old_trainer(device="cpu"):
    config = TrainConfig(seed=173, batch_size=3, tbptt=8, updates=4, device=device,
                         train_dts=(0.25,), dt_every=16, replay_capacity=2,
                         replay_block_ticks=16, replay_burnin=8, log_every=2)
    trainer = Trainer(Variant("integration-test", width=12, policy="online"), config)
    trainer.run_updates(4)
    return trainer


def tiny_config(mode="bp64", **changes):
    live_ticks = 16 if mode in {"accum128", "bp128"} else 8
    values = dict(mode=mode, live_ticks=live_ticks,
                  gradient_horizon=8 if mode == "accum128" else live_ticks)
    values.update(changes)
    return IntegrationConfig(**values)


def disturb_rng():
    random.random()
    np.random.random(7)
    torch.rand(7)
    if torch.cuda.is_available():
        torch.rand(7, device="cuda")


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_defaults_encode_the_frozen_arm_budgets(self):
        self.assertEqual(tuple(ARMS), (
            "frozen", "bp64", "repeat64", "replay64", "accum128", "bp128",
            "coact_live", "coact_mix", "coact_mix_shuffled", "coact_only_mix",
            "coverage_long_dense", "coverage_short_sparse"))
        for name in ARMS:
            config = IntegrationConfig(mode=name)
            self.assertEqual(config.live_ticks, 128 if name in {"accum128", "bp128"} else 64)
            self.assertEqual(config.gradient_horizon, 128 if name == "bp128" else 64)
            self.assertEqual(config.duration_range, (1., 4.) if name == "coverage_short_sparse" else (4., 8.))
            self.assertEqual(config.gap_range, (.5, 2.) if name == "coverage_long_dense" else (4., 12.))
        for changes in (dict(live_ticks=True), dict(live_ticks=0),
                        dict(live_ticks=8, gradient_horizon=16),
                        dict(live_ticks=8, gradient_horizon=3),
                        dict(local_lr=float("nan")), dict(duration_range=(2, 1)),
                        dict(gap_range=(0, 1))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                IntegrationConfig(**changes)

    def test_stage_transition_preserves_inherited_dynamic_state(self):
        base = old_trainer()
        before = base.semantic_state()
        stage = IntegrationTrainer(base, tiny_config("coact_mix"))
        after = stage.base.semantic_state()
        for name in ("model", "optimizer", "hidden", "replay", "rng", "ticks", "updates",
                     "gradient_updates", "dt_index", "dt_history", "action_hash", "data_hash"):
            self.assertEqual(state_digest(before[name]), state_digest(after[name]), name)
        self.assertEqual(stage.origin["semantic_hash"], state_digest(before))
        self.assertEqual(stage.world.regime.duration_range, (4., 8.))
        self.assertEqual(stage.world.regime.gap_range, (4., 12.))
        old_world, new_world = before["world"], after["world"]
        for key in old_world:
            if key not in {"regime", "regime_id", "regime_history"}:
                self.assertEqual(state_digest(old_world[key]), state_digest(new_world[key]), key)
        self.assertEqual(stage.stage_ticks, 0)
        self.assertEqual(stage.stage_updates, 0)
        self.assertEqual(stage.hidden_resets, 0)

    def test_report_text_matches_the_actual_update_mechanism(self):
        for mode, arm in ARMS.items():
            with self.subTest(mode=mode):
                trainer = IntegrationTrainer(old_trainer(), tiny_config(mode))
                semantic_before = state_digest(trainer.semantic_state())
                report = trainer.report()
                self.assertEqual(semantic_before, state_digest(trainer.semantic_state()))
                steps = int(arm.current_gradient) + int(arm.extra_gradient is not None)
                self.assertIn(f"Adam performs {steps} updates", report["gradient_semantics"])
                if arm.current_gradient:
                    self.assertIn(f"graphs detach every {trainer.integration_config.gradient_horizon}",
                                  report["gradient_semantics"])
                    self.assertIn("supervised", report["learning"])
                else:
                    self.assertIn("no label-gradient credit", report["gradient_semantics"])
                    self.assertNotIn("graphs detach", report["gradient_semantics"])
                    self.assertNotIn("hybrid", report["learning"])
                if mode == "frozen":
                    self.assertIn("Weights remain frozen", report["learning"])
                if mode == "coact_only_mix":
                    self.assertIn("no target-label gradients", report["learning"])
                    self.assertIn("supervised pretrained", report["learning"])
                if arm.extra_gradient is not None:
                    self.assertIn("second Adam pass", report["learning"])

    def test_all_first_live_chunks_share_state_and_extras_cannot_overwrite_it(self):
        base = old_trainer()
        old_hash = state_digest(base.replay.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.pt"
            base.save_checkpoint(path)
            states = []
            gradient_counts = {"frozen": 0, "bp64": 1, "repeat64": 2, "replay64": 2,
                               "coact_live": 1, "coact_mix": 1, "coact_mix_shuffled": 1,
                               "coact_only_mix": 0}
            for mode, gradients in gradient_counts.items():
                with self.subTest(mode=mode):
                    trainer = IntegrationTrainer.from_base_checkpoint(path, tiny_config(mode))
                    point = trainer.train_update()
                    self.assertEqual(point["gradient_updates_this_chunk"], gradients)
                    self.assertEqual(point["local_updates_this_chunk"], int(mode in LOCAL_MODES))
                    if gradients:
                        self.assertGreater(point["bp_recurrent_delta_frob"], 0.)
                        self.assertGreaterEqual(point["bp_all_delta_frob"], point["bp_recurrent_delta_frob"])
                    else:
                        self.assertIsNone(point["bp_recurrent_delta_frob"])
                    self.assertEqual(point["live_hidden_hash"], state_digest(trainer.h))
                    self.assertEqual(state_digest(trainer.replay.state_dict()), old_hash)
                    self.assertEqual(trainer.ticks, base.ticks + 8)
                    self.assertEqual(trainer.world.tick, base.world.tick + 8)
                    self.assertEqual(trainer.hidden_resets, 0)
                    if mode in LOCAL_MODES:
                        self.assertEqual(point["work"]["old_forward_ticks"], 16)
                        self.assertEqual(point["work"]["old_burnin_ticks"], 8)
                        self.assertEqual(trainer.local_live.observed_ticks, 8)
                        self.assertEqual(trainer.local_replay.observed_ticks, 16)
                        self.assertEqual(trainer.local_live.pending_ticks, 0)
                        self.assertEqual(trainer.local_replay.pending_ticks, 0)
                    states.append((state_digest(trainer.h), state_digest(trainer.world.state_dict()),
                                   trainer.action_hash, trainer.data_hash))
            self.assertEqual(len(set(states)), 1)

    def test_all_local_arms_use_the_same_private_old_sampling_sequence(self):
        base = old_trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.pt"
            base.save_checkpoint(path)
            selections, generator_states = [], []
            for mode in LOCAL_MODES:
                trainer = IntegrationTrainer.from_base_checkpoint(path, tiny_config(mode))
                points = trainer.run_updates(4)
                selections.append(tuple(point["replay_start_tick"] for point in points))
                generator_states.append(state_digest(trainer.replay_rng.getstate()))
                self.assertEqual(trainer.local_live.observed_ticks, 32)
                self.assertEqual(trainer.local_replay.history_restarts, 4)
                self.assertGreater(float(trainer.local_live.trace.square().sum()), 0.)
                self.assertEqual(trainer.work["old_samples"], 4)
            self.assertEqual(len(set(selections)), 1)
            self.assertEqual(len(set(generator_states)), 1)

    def test_zero_local_step_is_exactly_the_bp64_controller(self):
        base = old_trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.pt"
            base.save_checkpoint(path)
            expected = IntegrationTrainer.from_base_checkpoint(path, tiny_config())
            expected.run_updates(3)
            for mode in LOCAL_MODES[:-1]:
                with self.subTest(mode=mode):
                    actual = IntegrationTrainer.from_base_checkpoint(path, tiny_config(mode, local_lr=0.))
                    actual.run_updates(3)
                    for name in ("model", "optimizer", "hidden", "world", "rng", "replay"):
                        self.assertEqual(state_digest(expected.base.semantic_state()[name]),
                                         state_digest(actual.base.semantic_state()[name]), name)
                    self.assertEqual(actual.action_hash, expected.action_hash)
                    self.assertEqual(actual.data_hash, expected.data_hash)
                    self.assertEqual(actual.local_live.matrix_writes, 0)

    def test_pure_coactivity_updates_only_recurrent_and_keeps_adam(self):
        trainer = IntegrationTrainer(old_trainer(), tiny_config("coact_only_mix"))
        original_model = deepcopy(trainer.model.state_dict())
        original_adam = state_digest(trainer.optimizer.state_dict())
        start_gradients = trainer.gradient_updates
        trainer.run_updates(3)
        for name, parameter in trainer.model.state_dict().items():
            if name == "recurrent":
                self.assertFalse(torch.equal(original_model[name], parameter))
            else:
                self.assertTrue(torch.equal(original_model[name], parameter), name)
        self.assertEqual(state_digest(trainer.optimizer.state_dict()), original_adam)
        self.assertEqual(trainer.gradient_updates, start_gradients)
        self.assertEqual(trainer.work["supervised_ticks"], 0)
        self.assertEqual(trainer.local_updates, 3)

    def test_pure_coactivity_update_does_not_read_live_or_old_target_values(self):
        base = old_trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.pt"
            base.save_checkpoint(path)
            expected = IntegrationTrainer.from_base_checkpoint(path, tiny_config("coact_only_mix"))
            expected.run_updates(2)
            actual = IntegrationTrainer.from_base_checkpoint(path, tiny_config("coact_only_mix"))
            oracle = actual.world.oracle
            actual.world.oracle = lambda policy: (oracle(policy) + 1) % actual.model.n_actions
            for block in actual.replay.blocks:
                block["targets"] = (block["targets"] + 1) % actual.model.n_actions
            actual.run_updates(2)
            for name in ("model", "optimizer", "hidden", "world"):
                self.assertEqual(state_digest(expected.base.semantic_state()[name]),
                                 state_digest(actual.base.semantic_state()[name]), name)
            self.assertEqual(expected.action_hash, actual.action_hash)
            self.assertNotEqual(expected.data_hash, actual.data_hash)

    def test_accumulated64_and_full128_have_equal_values_and_different_credit(self):
        base = old_trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.pt"
            base.save_checkpoint(path)
            values, parameter_hashes, state_grad_flags = [], [], []
            for mode in ("accum128", "bp128"):
                trainer = IntegrationTrainer.from_base_checkpoint(path, tiny_config(mode))
                original_step, flags = trainer.model.step, []

                def observed_step(observation, state, dt):
                    flags.append(state.requires_grad)
                    return original_step(observation, state, dt)

                trainer.model.step = observed_step
                point = trainer.train_update()
                self.assertEqual(point["gradient_updates_this_chunk"], 1)
                self.assertEqual(point["work"]["live_forward_ticks"], 16)
                self.assertEqual(point["work"]["supervised_ticks"], 16)
                values.append((state_digest(trainer.h), state_digest(trainer.world.state_dict())))
                parameter_hashes.append(tensor_digest(trainer.model.state_dict()))
                state_grad_flags.append(flags)
            self.assertEqual(values[0], values[1])
            self.assertNotEqual(parameter_hashes[0], parameter_hashes[1])
            self.assertFalse(state_grad_flags[0][8])
            self.assertTrue(state_grad_flags[1][8])

    def test_exact_resume_all_arms_with_complete_new_state(self):
        base = old_trainer()
        with tempfile.TemporaryDirectory() as directory:
            origin = Path(directory) / "A.pt"
            continuation = Path(directory) / "new.pt"
            base.save_checkpoint(origin)
            for mode in ARMS:
                with self.subTest(mode=mode):
                    full = IntegrationTrainer.from_base_checkpoint(origin, tiny_config(mode))
                    full.run_updates(3)
                    expected = full.semantic_state()
                    split = IntegrationTrainer.from_base_checkpoint(origin, tiny_config(mode))
                    split.train_update()
                    saved = split.semantic_state()
                    split.save_checkpoint(continuation)
                    disturb_rng()
                    resumed = IntegrationTrainer.from_checkpoint(continuation)
                    self.assertEqual(state_digest(saved), state_digest(resumed.semantic_state()))
                    resumed.run_updates(2)
                    actual = resumed.semantic_state()
                    self.assertEqual(state_digest(expected), state_digest(actual))
                    for name in expected:
                        self.assertEqual(state_digest(expected[name]), state_digest(actual[name]), name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_exact_cuda_resume_for_shuffled_local_mixture(self):
        base = old_trainer("cuda")
        config = tiny_config("coact_mix_shuffled")
        with tempfile.TemporaryDirectory() as directory:
            origin, continuation = Path(directory) / "A.pt", Path(directory) / "new.pt"
            base.save_checkpoint(origin)
            full = IntegrationTrainer.from_base_checkpoint(origin, config, device="cuda")
            full.run_updates(3)
            expected = state_digest(full.semantic_state())
            split = IntegrationTrainer.from_base_checkpoint(origin, config, device="cuda")
            split.train_update()
            split.save_checkpoint(continuation)
            disturb_rng()
            resumed = IntegrationTrainer.from_checkpoint(continuation, device="cuda")
            resumed.run_updates(2)
            self.assertEqual(expected, state_digest(resumed.semantic_state()))

    def test_dispatch_and_source_guards(self):
        base = old_trainer()
        with tempfile.TemporaryDirectory() as directory:
            origin, continuation = Path(directory) / "A.pt", Path(directory) / "new.pt"
            base.save_checkpoint(origin)
            loaded = load_checkpoint(origin)
            self.assertIsInstance(loaded, Trainer)
            self.assertEqual(state_digest(base.semantic_state()), state_digest(loaded.semantic_state()))
            trainer = IntegrationTrainer.from_base_checkpoint(origin, tiny_config("coact_mix"))
            trainer.train_update()
            trainer.save_checkpoint(continuation)
            dispatched = load_checkpoint(continuation)
            self.assertIsInstance(dispatched, IntegrationTrainer)
            self.assertEqual(state_digest(trainer.semantic_state()), state_digest(dispatched.semantic_state()))
            with self.assertRaisesRegex(ValueError, "unsupported continuity"):
                Trainer.from_checkpoint(continuation)
            state = torch.load(continuation, map_location="cpu", weights_only=False)
            state["source_hashes"]["run.py"] = "0" * 64
            torch.save(state, continuation)
            with self.assertRaisesRegex(ValueError, "snapshot/hash mismatch"):
                IntegrationTrainer.from_checkpoint(continuation)
            IntegrationTrainer.from_checkpoint(continuation, strict_sources=False)
            state["source_hashes"] = trainer.source_hashes
            state["continuation"]["source_hashes"]["world.py"] = "0" * 64
            torch.save(state, continuation)
            with self.assertRaisesRegex(ValueError, "inherited checkpoint source mismatch"):
                IntegrationTrainer.from_checkpoint(continuation)

    def test_artifacts_preserve_both_source_generations_and_full_traces(self):
        trainer = IntegrationTrainer(old_trainer(), tiny_config("coact_mix_shuffled"))
        trainer.train_update()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run_artifacts(trainer, folder)
            for relative in ("checkpoint.pt", "report.json", "sources/integration/run.py",
                             "sources/integration/local.py", "sources/continuity/run.py",
                             "sources/continuity/world.py", "sources/continuity/model.py"):
                self.assertTrue((folder / relative).is_file(), relative)
            state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["format"], FORMAT_VERSION)
            for name in ("integration_config", "origin", "local_live", "local_replay", "replay_rng",
                         "local_updates", "work", "update_norms", "sources", "source_hashes", "continuation"):
                self.assertIn(name, state)
            self.assertEqual(state_digest(state["continuation"]["rng"]), state_digest(rng_state()))

    def test_tick_target_matches_experiences_for_different_update_frequencies(self):
        with tempfile.TemporaryDirectory() as directory:
            base = old_trainer()
            origin = Path(directory) / "A.pt"
            base.save_checkpoint(origin)
            for mode, updates in (("bp64", 4), ("accum128", 2), ("bp128", 2)):
                trainer = IntegrationTrainer.from_base_checkpoint(origin, tiny_config(mode))
                folder = Path(directory) / mode
                with redirect_stdout(io.StringIO()):
                    run_to_target(trainer, 32, folder)
                self.assertEqual(trainer.stage_ticks, 32)
                self.assertEqual(trainer.stage_updates, updates)
                self.assertEqual(trainer.work["supervised_ticks"], 32)
                self.assertEqual(trainer.work["live_forward_ticks"], 32)
                with self.assertRaises(ValueError):
                    run_to_target(trainer, 33, folder)

    def test_online_actions_use_own_current_world_labels(self):
        trainer = IntegrationTrainer(old_trainer(), tiny_config("bp64"))
        with torch.no_grad():
            trainer.model.head.weight.zero_()
            trainer.model.head.bias.zero_()
            trainer.model.head.bias.view(4, 11)[:, 0] = 20
        point = trainer.train_update()
        self.assertEqual(point["actual_action_counts"], [8 * 3 * 4] + [0] * 10)
        self.assertGreater(sum(point["target_action_counts"][1:]), 0)


if __name__ == "__main__":
    unittest.main()
