"""Continuation and causal-training regression tests; no long training runs."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from experiments.continuity.run import (
    ExperienceReservoir, TrainConfig, Trainer, VARIANTS, Variant,
    rng_state, state_digest, tensor_digest, write_run_artifacts,
)


def tiny_config(device: str = "cpu", **changes) -> TrainConfig:
    values = dict(seed=173, batch_size=3, tbptt=8, updates=5,
                  device=device, train_dts=(0.125, 0.25, 0.5), dt_every=16,
                  replay_capacity=2, replay_block_ticks=16, replay_burnin=8,
                  log_every=2)
    values.update(changes)
    return TrainConfig(**values)


def disturb_random_generators() -> None:
    random.random()
    np.random.random(7)
    torch.rand(7)
    if torch.cuda.is_available():
        torch.rand(7, device="cuda")


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_resume_exact(self, variant: Variant, device: str = "cpu") -> None:
        config = tiny_config(device)
        uninterrupted = Trainer(variant, config)
        uninterrupted.run_updates(5)
        expected = uninterrupted.semantic_state()
        expected_hash = state_digest(expected)

        interrupted = Trainer(variant, config)
        interrupted.run_updates(3)
        saved = interrupted.semantic_state()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "interrupted.pt"
            interrupted.save_checkpoint(path)
            disturb_random_generators()
            resumed = Trainer.from_checkpoint(path, device=device)
            self.assertEqual(state_digest(saved), state_digest(resumed.semantic_state()))
            resumed.run_updates(2)
        actual = resumed.semantic_state()
        self.assertEqual(expected_hash, state_digest(actual))
        for key in ("model", "optimizer", "hidden", "world", "rng", "stats",
                    "replay", "dt_history", "trace"):
            self.assertEqual(state_digest(expected[key]), state_digest(actual[key]), key)
        self.assertTrue(torch.equal(uninterrupted.h, resumed.h))
        self.assertEqual(uninterrupted.action_hash, resumed.action_hash)
        self.assertEqual(uninterrupted.data_hash, resumed.data_hash)
        self.assertEqual(resumed.ticks, 40)
        self.assertEqual(resumed.world.tick, 39)
        self.assertEqual(resumed.gradient_updates, 5)
        self.assertEqual(resumed.dt_history,
                         [{"tick": 0, "dt": 0.125}, {"tick": 16, "dt": 0.25},
                          {"tick": 32, "dt": 0.5}])
        self.assertEqual(resumed.world.now, 16 * 0.125 + 16 * 0.25 + 8 * 0.5 - 0.125)

    def test_exact_resume_after_arbitrary_optimizer_boundary(self):
        for variant in (Variant("teacher-test", width=12),
                        Variant("online-test", width=12, policy="online"),
                        Variant("reset-test", width=12, reset_every=16),
                        Variant("reservoir-test", width=12, kind="reservoir"),
                        Variant("gru-test", width=12, kind="gru")):
            with self.subTest(variant=variant.name):
                self.assert_resume_exact(variant)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_exact_cuda_resume_with_cpu_world(self):
        self.assert_resume_exact(Variant("cuda-online-test", width=12, policy="online"), "cuda")

    def test_online_executes_own_actions_and_labels_own_world(self):
        trainer = Trainer(Variant("always-wait-test", width=12, policy="online"),
                          tiny_config(train_dts=(0.25,)))
        with torch.no_grad():
            trainer.model.head.weight.zero_()
            trainer.model.head.bias.zero_()
            trainer.model.head.bias.view(4, 11)[:, 0] = 20
        point = trainer.train_update()
        self.assertEqual(point["actual_action_counts"], [8 * 3 * 4] + [0] * 10)
        self.assertGreater(sum(point["target_action_counts"][1:]), 0)
        self.assertEqual(trainer.world.metrics()["attempted"], 0)
        self.assertEqual(trainer.world.metrics()["accepted"], 0)
        self.assertGreater(trainer.world.metrics()["commands"], 0)
        partial = trainer.replay.partial
        observations = torch.stack(partial["observations"]).view(8, 3, 4, 24)
        self.assertTrue(torch.equal(observations[..., 11:24],
                                    torch.zeros_like(observations[..., 11:24])))

    def test_same_width_initialization_and_data_do_not_depend_on_control_name(self):
        config = tiny_config()
        continuous = Trainer(Variant("continuous-test", width=12), config)
        initial_hash = continuous.initial_parameter_hash
        continuous.run_updates(2)
        continuous_world_hash = state_digest(continuous.world.state_dict())
        continuous_model_hash = tensor_digest(continuous.model.state_dict())
        continuous_data_hash = continuous.data_hash
        control = Trainer(Variant("reset-test", width=12, reset_every=16), config)
        self.assertEqual(control.initial_parameter_hash, initial_hash)
        control.run_updates(2)
        self.assertEqual(state_digest(control.world.state_dict()), continuous_world_hash)
        self.assertEqual(tensor_digest(control.model.state_dict()), continuous_model_hash)
        self.assertEqual(control.data_hash, continuous_data_hash)
        self.assertEqual(control.hidden_resets, 0)
        control.train_update()
        self.assertEqual(control.hidden_resets, 1)
        self.assertEqual(control.world.tick, 23)

    def test_default_controls_and_budget(self):
        self.assertEqual(list(VARIANTS), ["c64", "c128", "c256", "reset128",
                                         "reservoir128", "gru128", "online128"])
        self.assertEqual([VARIANTS[name].width for name in ("c64", "c128", "c256")], [64, 128, 256])
        self.assertEqual(VARIANTS["reset128"].reset_every, 512)
        self.assertEqual(VARIANTS["online128"].policy, "online")
        config = TrainConfig()
        self.assertEqual((config.updates, config.tbptt, config.batch_size, config.n_jobs),
                         (2048, 64, 64, 4))
        self.assertEqual((config.replay_capacity, config.replay_block_ticks, config.replay_burnin),
                         (32, 128, 64))

    def test_artifacts_include_full_continuation_and_sources(self):
        trainer = Trainer(Variant("artifacts-test", width=12), tiny_config())
        trainer.train_update()
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "run"
            write_run_artifacts(trainer, destination)
            self.assertTrue((destination / "report.json").is_file())
            self.assertTrue((destination / "sources" / "run.py").is_file())
            self.assertTrue((destination / "sources" / "model.py").is_file())
            state = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=False)
            for name in ("model", "optimizer", "hidden", "world", "rng", "replay", "dt_history",
                         "ticks", "updates", "trace", "sources", "runtime", "source_hashes"):
                self.assertIn(name, state)
            self.assertEqual(state["runtime"]["numpy"], np.__version__)
            self.assertEqual(state_digest(state["rng"]), state_digest(rng_state()))


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_branch_preserves_subject_and_extra_updates_leave_live_state_alone(self):
        base = Trainer(Variant("adaptation-test", width=12, policy="online"), tiny_config())
        base.run_updates(4)
        start_model_hash = tensor_digest(base.model.state_dict())
        start_optimizer_hash = state_digest(base.optimizer.state_dict())
        start_hidden = base.h.clone()
        start_world_tick = base.world.tick
        start_world_now = base.world.now
        stored_blocks_hash = state_digest(base.replay.blocks)
        live_hidden_hashes, live_world_hashes = [], []
        final_models = {}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "A.pt"
            base.save_checkpoint(checkpoint)
            for mode, expected_updates in (("frozen", 0), ("current", 1),
                                           ("current_repeat", 2), ("replay", 2)):
                with self.subTest(mode=mode):
                    branch = Trainer.from_checkpoint(checkpoint)
                    branch.begin_adaptation(mode)
                    self.assertEqual(tensor_digest(branch.model.state_dict()), start_model_hash)
                    self.assertEqual(state_digest(branch.optimizer.state_dict()), start_optimizer_hash)
                    self.assertTrue(torch.equal(branch.h, start_hidden))
                    self.assertEqual(branch.world.tick, start_world_tick)
                    self.assertEqual(branch.world.now, start_world_now)
                    self.assertEqual(branch.world.regime.duration_range, (4.0, 8.0))
                    point = branch.train_update()
                    self.assertEqual(point["gradient_updates_this_chunk"], expected_updates)
                    self.assertEqual(state_digest(branch.replay.blocks), stored_blocks_hash)
                    self.assertEqual(branch.replay.blocks_seen, base.replay.blocks_seen)
                    self.assertEqual(branch.ticks, base.ticks + 8)
                    self.assertEqual(branch.hidden_resets, 0)
                    self.assertEqual(branch.stage_updates, 1)
                    self.assertEqual(branch.gradient_updates - branch.phase_start_gradient, expected_updates)
                    live_hidden_hashes.append(tensor_digest({"h": branch.h}))
                    live_world_hashes.append(state_digest(branch.world.state_dict()))
                    final_models[mode] = tensor_digest(branch.model.state_dict())
        # All first chunks execute under identical inherited parameters. Second
        # optimization passes must not replace that shared live hidden value.
        self.assertEqual(len(set(live_hidden_hashes)), 1)
        self.assertEqual(len(set(live_world_hashes)), 1)
        self.assertEqual(final_models["frozen"], start_model_hash)
        self.assertNotEqual(final_models["current"], final_models["current_repeat"])
        self.assertNotEqual(final_models["current_repeat"], final_models["replay"])

    def test_replay_adaptation_resume_is_exact(self):
        config = tiny_config()
        variant = Variant("adapt-resume-test", width=12, policy="online")
        base = Trainer(variant, config)
        base.run_updates(4)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "A.pt"
            continuation = Path(folder) / "B.pt"
            base.save_checkpoint(checkpoint)
            full = Trainer.from_checkpoint(checkpoint)
            full.begin_adaptation("replay")
            full.run_updates(3)
            expected = full.semantic_state()
            split = Trainer.from_checkpoint(checkpoint)
            split.begin_adaptation("replay")
            split.train_update()
            split.save_checkpoint(continuation)
            disturb_random_generators()
            resumed = Trainer.from_checkpoint(continuation)
            resumed.run_updates(2)
            self.assertEqual(state_digest(expected), state_digest(resumed.semantic_state()))

    def test_replay_reprocesses_past_h_with_burnin(self):
        trainer = Trainer(Variant("burnin-test", width=12, policy="online"), tiny_config())
        trainer.run_updates(2)
        block = trainer.replay.blocks[0]
        self.assertEqual((block["start_tick"], block["end_tick"]), (0, 16))
        self.assertTrue(torch.equal(block["start_h"], torch.zeros_like(block["start_h"])))
        self.assertEqual(block["elapsed"][0], 0)
        h = block["start_h"].to(trainer.device)
        x = block["observations"].to(trainer.device)
        for index in range(8):
            with torch.no_grad():
                _, h = trainer.model.step(x[index], h, block["elapsed"][index])
        expected, expected_h = trainer._forward_sequence(x[8:], h.detach(),
                                                        block["elapsed"][8:], block["resets"][8:])
        logits, actual_h = trainer._forward_sequence(x, block["start_h"].to(trainer.device),
                                                     block["elapsed"], block["resets"], burnin=8)
        self.assertTrue(torch.equal(logits, expected))
        self.assertTrue(torch.equal(actual_h, expected_h))
        self.assertEqual(logits.shape, (8, 3, 4, 11))


class ReservoirTests(unittest.TestCase):
    def test_finite_uniform_reservoir_continues_exactly(self):
        reservoir = ExperienceReservoir(capacity=2, block_ticks=4, seed=1)
        for tick in range(13):
            reservoir.add_tick(torch.full((2, 3), float(tick)), torch.zeros(2, 1, dtype=torch.long),
                               0.25, torch.full((2, 5), float(tick)), tick, False)
        state = reservoir.state_dict()
        clone = ExperienceReservoir(capacity=2, block_ticks=4, seed=99)
        clone.load_state_dict(state)
        for tick in range(13, 41):
            arguments = (torch.full((2, 3), float(tick)), torch.zeros(2, 1, dtype=torch.long),
                         0.25, torch.full((2, 5), float(tick)), tick, False)
            reservoir.add_tick(*arguments)
            clone.add_tick(*arguments)
        self.assertEqual(state_digest(reservoir.state_dict()), state_digest(clone.state_dict()))
        self.assertEqual(reservoir.blocks_seen, 10)
        self.assertEqual(len(reservoir.blocks), 2)
        self.assertEqual(reservoir.summary()["partial_ticks"], 1)
        for block in reservoir.blocks:
            self.assertEqual(block["end_tick"] - block["start_tick"], 4)
            self.assertTrue(torch.equal(block["start_h"],
                                        torch.full((2, 5), float(block["start_tick"]))))


if __name__ == "__main__":
    unittest.main()
