"""Short regression checks for the continuous credit runner, not a study."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.continuity.run import TrainConfig, Trainer, Variant, state_digest
from experiments.credit.run import (
    CreditConfig, CreditTrainer, MODES, load_checkpoint, run_to_target, write_run_artifacts,
)


def source_trainer(device="cpu", **changes):
    values = dict(seed=173, batch_size=3, tbptt=4, updates=4, device=device,
                  train_dts=(.25,), replay_capacity=2, replay_block_ticks=8,
                  replay_burnin=4, log_every=2)
    values.update(changes)
    trainer = Trainer(Variant("credit-test", width=8, policy="online"), TrainConfig(**values))
    trainer.run_updates(4)
    return trainer


def config(mode="bp64", **changes):
    values = dict(mode=mode, chunk_ticks=4)
    values.update(changes)
    return CreditConfig(**values)


def disturb_rng():
    random.random()
    np.random.rand(5)
    torch.rand(5)
    if torch.cuda.is_available():
        torch.rand(5, device="cuda")


class CreditRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_roster_and_configuration(self):
        self.assertEqual(tuple(MODES), ("frozen", "readout_only", "bp64", "bp_replay", "diag_sym",
            "diag_sym_replay", "diag_random", "diag_random_replay", "diag_reset64", "diag_instant", "diag_shuffle"))
        default = CreditConfig()
        self.assertEqual((default.chunk_ticks, default.duration_range, default.gap_range), (64, (4., 8.), (4., 12.)))
        for changes in (dict(mode="oops"), dict(chunk_ticks=True), dict(chunk_ticks=0),
                        dict(duration_range=(2, 1)), dict(gap_range=(0, 1)), dict(regime_name="")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CreditConfig(**changes)
        with self.assertRaisesRegex(ValueError, "same tick budget"):
            CreditTrainer(source_trainer(), config("bp_replay", chunk_ticks=8))

    def test_initial_branch_keeps_complete_source_state(self):
        base = source_trainer()
        before = base.semantic_state()
        trainer = CreditTrainer(base, config("diag_random_replay"))
        after = trainer.base.semantic_state()
        for name in ("model", "optimizer", "hidden", "rng", "replay", "ticks", "updates", "dt_history"):
            self.assertEqual(state_digest(before[name]), state_digest(after[name]), name)
        self.assertEqual(trainer.origin["semantic_hash"], state_digest(before))
        self.assertTrue(torch.equal(trainer.live_credit.fixed_feedback, trainer.old_credit.fixed_feedback))
        self.assertEqual(trainer.stage_ticks, 0)
        self.assertEqual(len(trainer.regime_changes), 1)

    def test_every_mode_has_same_first_live_state_and_one_or_zero_adam_step(self):
        base = source_trainer()
        old_bank = state_digest(base.replay.state_dict())
        states = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A.pt"
            base.save_checkpoint(path)
            for name, mode in MODES.items():
                with self.subTest(mode=name):
                    trainer = CreditTrainer.from_base_checkpoint(path, config(name))
                    point = trainer.train_update()
                    self.assertEqual(point["optimizer_steps_this_chunk"], int(mode.method != "frozen"))
                    self.assertEqual(trainer.work["optimizer_steps"], int(mode.method != "frozen"))
                    self.assertEqual(trainer.work["old_forward_ticks"], 8 if mode.replay else 0)
                    self.assertEqual(trainer.work["old_warmup_ticks"], 4 if mode.replay else 0)
                    self.assertEqual(trainer.work["old_loss_ticks"], 4 if mode.replay else 0)
                    self.assertEqual(state_digest(trainer.replay.state_dict()), old_bank)
                    self.assertEqual(point["live_hidden_hash"], state_digest(trainer.h))
                    states.append((state_digest(trainer.h), state_digest(trainer.world.state_dict()),
                                   trainer.action_hash, trainer.data_hash))
        self.assertEqual(len(set(states)), 1)

    def test_bp64_reproduces_original_current_control(self):
        base = source_trainer()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A.pt"
            base.save_checkpoint(path)
            original = Trainer.from_checkpoint(path)
            original.begin_adaptation("current")
            original.run_updates(3)
            credit = CreditTrainer.from_base_checkpoint(path, config(gap_range=(.5, 2.)))
            credit.run_updates(3)
            for name in ("model", "optimizer", "hidden"):
                self.assertEqual(state_digest(original.semantic_state()[name]),
                                 state_digest(credit.base.semantic_state()[name]), name)
            self.assertEqual(original.action_hash, credit.action_hash)
            self.assertEqual(original.data_hash, credit.data_hash)

    def test_manual_modes_do_not_call_backward_or_autograd_grad(self):
        for name, mode in MODES.items():
            if mode.method not in {"diag", "readout"}:
                continue
            with self.subTest(mode=name):
                trainer = CreditTrainer(source_trainer(), config(name))
                with patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward called")), \
                     patch("torch.autograd.grad", side_effect=AssertionError("autograd.grad called")):
                    trainer.run_updates(2)
                self.assertEqual(trainer.work["backward_calls"], 0)
                self.assertFalse(trainer.h.requires_grad)
                self.assertTrue(all(parameter.grad is None for parameter in trainer.model.parameters()))

    def test_readout_only_skips_core_adam_moments_instead_of_zero_gradient_drift(self):
        trainer = CreditTrainer(source_trainer(), config("readout_only"))
        before = deepcopy(trainer.model.state_dict())
        old_adam = {name: deepcopy(trainer.optimizer.state[parameter])
                    for name, parameter in trainer.model.named_parameters()}
        trainer.run_updates(3)
        for name, parameter in trainer.model.named_parameters():
            if name.startswith("head."):
                self.assertFalse(torch.equal(before[name], parameter))
                self.assertEqual(float(trainer.optimizer.state[parameter]["step"]), float(old_adam[name]["step"]) + 3)
            else:
                self.assertTrue(torch.equal(before[name], parameter), name)
                self.assertEqual(state_digest(old_adam[name]), state_digest(trainer.optimizer.state[parameter]), name)
        report = trainer.report()
        self.assertEqual(report["effective_updated_parameters"], 44 * 8 + 44)
        self.assertIn("not a spiking LSM", report["learning"])

    def test_replay_uses_preupdate_weights_full_old_bp_graph_and_half_normalized_gradients(self):
        trainer = CreditTrainer(source_trainer(wait_weight=.17), config("bp_replay"))
        reference = deepcopy(trainer.model)
        start_h = trainer.h.clone()
        initial_weights = state_digest(trainer.model.state_dict())
        observations, targets, elapsed, weight_hashes, grad_flags = [], [], [], [], []
        original_step, original_observe, original_oracle = trainer.model.step, trainer.world.observe, trainer.world.oracle
        captured = {}

        def step(observation, state, dt):
            weight_hashes.append(state_digest(trainer.model.state_dict()))
            grad_flags.append(state.requires_grad)
            return original_step(observation, state, dt)

        def observe():
            value = original_observe()
            observations.append(value.clone())
            elapsed.append(trainer.world.elapsed)
            return value

        def oracle(policy):
            value = original_oracle(policy)
            targets.append(value.clone())
            return value

        original_apply = trainer._apply_gradients
        def apply(gradients):
            captured.update({name: value.clone() for name, value in gradients.items()})
            return original_apply(gradients)

        trainer.model.step, trainer.world.observe, trainer.world.oracle, trainer._apply_gradients = step, observe, oracle, apply
        point = trainer.train_update()
        self.assertEqual(set(weight_hashes), {initial_weights})
        self.assertEqual(len(weight_hashes), 12)
        self.assertTrue(grad_flags[8])  # old suffix begins with the warm-up graph attached
        self.assertEqual(trainer.work["bp_graph_ticks"], 12)
        self.assertEqual(trainer.work["backward_calls"], 2)
        self.assertEqual(trainer.work["optimizer_steps"], 1)
        h, live_outputs = start_h, []
        for x, dt in zip(observations, elapsed):
            output, h = reference.step(x, h, dt)
            live_outputs.append(output)
        live_loss = trainer.base._loss(torch.stack(live_outputs), torch.stack(targets))
        block = trainer.replay.blocks[point["old"]["block_index"]]
        h, old_outputs = block["start_h"], []
        for index, (x, dt) in enumerate(zip(block["observations"], block["elapsed"])):
            output, h = reference.step(x, h, dt)
            if index >= 4:
                old_outputs.append(output)
        old_loss = trainer.base._loss(torch.stack(old_outputs), block["targets"][4:])
        (.5 * live_loss + .5 * old_loss).backward()
        for name, parameter in reference.named_parameters():
            self.assertTrue(torch.allclose(parameter.grad, captured[name], rtol=2e-6, atol=2e-7), name)

    def test_explicit_reset_controls_do_not_reset_hidden_or_accumulated_gradient(self):
        for mode, expected in (("diag_sym", 0), ("diag_reset64", 2), ("diag_instant", 8)):
            with self.subTest(mode=mode):
                trainer = CreditTrainer(source_trainer(), config(mode))
                trainer.run_updates(2)
                self.assertEqual(trainer.work["live_trace_resets"], expected)
                self.assertEqual(trainer.live_credit.eligibility_resets, expected)
                self.assertEqual(trainer.live_credit.loss_ticks, 8)
                self.assertEqual(trainer.live_credit.pending_loss_ticks, 0)
                self.assertGreater(float(trainer.live_credit.F.abs().sum()), 0)
                self.assertEqual(trainer.hidden_resets, 0)

    def test_all_replay_modes_share_private_sampling_and_leave_old_bank_untouched(self):
        base = source_trainer()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A.pt"
            base.save_checkpoint(path)
            choices = []
            for name in ("bp_replay", "diag_sym_replay", "diag_random_replay"):
                trainer = CreditTrainer.from_base_checkpoint(path, config(name))
                points = trainer.run_updates(4)
                choices.append(tuple(point["old"]["block_index"] for point in points))
                self.assertEqual(state_digest(trainer.replay.state_dict()), state_digest(base.replay.state_dict()))
                if trainer.old_credit is not None:
                    self.assertEqual(trainer.old_credit.eligibility_resets, 4)
                    self.assertEqual(trainer.live_credit.eligibility_resets, 0)
            self.assertEqual(len(set(choices)), 1)

    def test_change_regime_preserves_live_state_traces_optimizer_and_rng(self):
        trainer = CreditTrainer(source_trainer(), config("diag_random_replay"))
        trainer.run_updates(2)
        before = trainer.semantic_state()
        trainer.change_regime((1, 4), (.5, 2), "familiar-A")
        after = trainer.semantic_state()
        for name in ("live_credit", "old_credit", "sampler_rng", "work", "update_norms"):
            self.assertEqual(state_digest(before[name]), state_digest(after[name]), name)
        for name in ("model", "optimizer", "hidden", "rng", "replay", "ticks", "updates"):
            self.assertEqual(state_digest(before["continuation"][name]), state_digest(after["continuation"][name]), name)
        self.assertEqual(trainer.stage_ticks, 8)
        self.assertEqual(trainer.regime_changes[-1]["stage_tick"], 8)
        valid = state_digest(trainer.semantic_state())
        with self.assertRaises(ValueError):
            trainer.change_regime((3, 1), (.5, 2), "invalid")
        self.assertEqual(valid, state_digest(trainer.semantic_state()))
        trainer.base._at_boundary = False
        with self.assertRaises(RuntimeError):
            trainer.change_regime((1, 4), (.5, 2), "during-update")

    def test_exact_cpu_resume_all_modes(self):
        base = source_trainer()
        with tempfile.TemporaryDirectory() as folder:
            origin, split_path = Path(folder) / "A.pt", Path(folder) / "credit.pt"
            base.save_checkpoint(origin)
            for mode in MODES:
                with self.subTest(mode=mode):
                    full = CreditTrainer.from_base_checkpoint(origin, config(mode))
                    full.run_updates(3)
                    expected = state_digest(full.semantic_state())
                    split = CreditTrainer.from_base_checkpoint(origin, config(mode))
                    split.train_update()
                    saved = state_digest(split.semantic_state())
                    split.save_checkpoint(split_path)
                    disturb_rng()
                    resumed = CreditTrainer.from_checkpoint(split_path)
                    self.assertEqual(saved, state_digest(resumed.semantic_state()))
                    resumed.run_updates(2)
                    self.assertEqual(expected, state_digest(resumed.semantic_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_exact_cuda_resume_symmetric_random_replay_and_shuffle(self):
        base = source_trainer("cuda")
        with tempfile.TemporaryDirectory() as folder:
            origin, split_path = Path(folder) / "A.pt", Path(folder) / "credit.pt"
            base.save_checkpoint(origin)
            for mode in ("diag_sym", "diag_random_replay", "diag_shuffle", "bp_replay"):
                with self.subTest(mode=mode):
                    full = CreditTrainer.from_base_checkpoint(origin, config(mode), device="cuda")
                    full.run_updates(3)
                    expected = state_digest(full.semantic_state())
                    split = CreditTrainer.from_base_checkpoint(origin, config(mode), device="cuda")
                    split.train_update()
                    split.save_checkpoint(split_path)
                    disturb_rng()
                    resumed = CreditTrainer.from_checkpoint(split_path, device="cuda")
                    resumed.run_updates(2)
                    self.assertEqual(expected, state_digest(resumed.semantic_state()))

    def test_resume_across_explicit_alternating_regimes(self):
        base = source_trainer()
        with tempfile.TemporaryDirectory() as folder:
            origin, split_path = Path(folder) / "A.pt", Path(folder) / "credit.pt"
            base.save_checkpoint(origin)
            full = CreditTrainer.from_base_checkpoint(origin, config("diag_sym_replay"))
            full.run_updates(2)
            full.change_regime((1, 4), (.5, 2), "A")
            full.run_updates(2)
            full.change_regime((4, 8), (4, 12), "B")
            full.train_update()
            expected = state_digest(full.semantic_state())
            split = CreditTrainer.from_base_checkpoint(origin, config("diag_sym_replay"))
            split.run_updates(2)
            split.change_regime((1, 4), (.5, 2), "A")
            split.train_update()
            split.save_checkpoint(split_path)
            resumed = CreditTrainer.from_checkpoint(split_path)
            resumed.train_update()
            resumed.change_regime((4, 8), (4, 12), "B")
            resumed.train_update()
            self.assertEqual(expected, state_digest(resumed.semantic_state()))
            self.assertEqual(len(resumed.regime_changes), 3)

    def test_checkpoint_dispatch_sources_and_artifacts(self):
        base = source_trainer()
        with tempfile.TemporaryDirectory() as folder:
            origin = Path(folder) / "A.pt"
            base.save_checkpoint(origin)
            self.assertIsInstance(load_checkpoint(origin), Trainer)
            trainer = CreditTrainer.from_base_checkpoint(origin, config("diag_random_replay"))
            trainer.train_update()
            destination = Path(folder) / "run"
            write_run_artifacts(trainer, destination)
            path = destination / "checkpoint.pt"
            self.assertIsInstance(load_checkpoint(path), CreditTrainer)
            for relative in ("report.json", "sources/credit/run.py", "sources/credit/local.py",
                             "sources/continuity/model.py", "sources/dependencies/integration/run.py"):
                self.assertTrue((destination / relative).is_file(), relative)
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["source_hashes"]["run.py"] = "0" * 64
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "credit checkpoint source"):
                CreditTrainer.from_checkpoint(path)
            CreditTrainer.from_checkpoint(path, strict_sources=False)
            state["source_hashes"] = trainer.source_hashes
            state["dependency_hashes"]["integration/run.py"] = "0" * 64
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "dependency source"):
                CreditTrainer.from_checkpoint(path)

    def test_tick_target_and_checkpoint_boundaries(self):
        trainer = CreditTrainer(source_trainer(), config("readout_only"))
        with tempfile.TemporaryDirectory() as folder:
            with redirect_stdout(io.StringIO()):
                run_to_target(trainer, 12, folder, checkpoint_every=2)
            self.assertEqual(trainer.stage_ticks, 12)
            self.assertTrue((Path(folder) / "checkpoint_000002.pt").is_file())
            with self.assertRaises(ValueError):
                run_to_target(trainer, 13, folder)
            trainer.base._at_boundary = False
            with self.assertRaises(RuntimeError):
                trainer.save_checkpoint(Path(folder) / "partial.pt")


if __name__ == "__main__":
    unittest.main()
