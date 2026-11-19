"""Toy CPU continuation checks, including complete semantic resume equality."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from experiments.continuity.run import TrainConfig, Trainer, Variant, rng_state, state_digest
from experiments.credit.run import CreditConfig, CreditTrainer
from experiments.loops.model import ARMS
from experiments.loops.train import LoopTrainer, main, optimizer_parameter_names


def tiny_source():
    base = Trainer(Variant("loops-toy", width=8, policy="online"),
        TrainConfig(seed=137, batch_size=2, tbptt=4, updates=2, device="cpu",
            train_dts=(.25,), replay_capacity=2, replay_block_ticks=8, replay_burnin=4, log_every=2))
    base.run_updates(2)
    return base


class LoopTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_graft_preserves_old_parameters_optimizer_hidden_history_and_rng(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                base = tiny_source()
                before = deepcopy(base.semantic_state())
                old_parameters = dict(base.model.named_parameters())
                optimizer = base.optimizer
                old_groups = deepcopy(optimizer.state_dict())
                old_world = before["world"]
                trainer = LoopTrainer(base, arm, chunk_ticks=4)
                after = base.semantic_state()
                self.assertIs(trainer.optimizer, optimizer)
                for name, parameter in old_parameters.items():
                    self.assertIs(dict(trainer.model.named_parameters())[name], parameter)
                    self.assertTrue(torch.equal(parameter, before["model"][name]))
                for key in ("hidden", "rng", "replay", "ticks", "updates", "dt_history"):
                    self.assertEqual(state_digest(before[key]), state_digest(after[key]), key)
                self.assertEqual(state_digest(old_groups["state"]), state_digest(optimizer.state_dict()["state"]))
                self.assertEqual(old_groups["param_groups"][0], optimizer.state_dict()["param_groups"][0])
                for key in old_world:
                    if key not in {"regime", "regime_id", "regime_history"}:
                        self.assertEqual(state_digest(old_world[key]), state_digest(after["world"][key]), key)
                if arm != "baseline":
                    self.assertNotIn(trainer.model.branch, optimizer.state)
                    self.assertEqual(optimizer_parameter_names(trainer.model, optimizer)[-1], ["branch"])
                self.assertEqual(trainer.stage_ticks, 0)
                self.assertEqual(trainer.credit_config.mode, "bp64")

    def test_resume_matches_entire_semantic_state_for_each_arm(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as folder:
                trainer = LoopTrainer(tiny_source(), arm, chunk_ticks=4)
                trainer.run_updates(2)
                checkpoint = Path(folder) / "checkpoint.pt"
                trainer.save_checkpoint(checkpoint)
                boundary = state_digest(trainer.semantic_state())
                trainer.run_updates(2)
                expected = state_digest(trainer.semantic_state())
                random.random()
                np.random.random(3)
                torch.rand(3)
                resumed = LoopTrainer.from_checkpoint(checkpoint, device="cpu")
                self.assertEqual(boundary, state_digest(resumed.semantic_state()))
                resumed.run_updates(2)
                self.assertEqual(expected, state_digest(resumed.semantic_state()))
                self.assertEqual(resumed.stage_ticks, 16)
                if arm != "baseline":
                    self.assertGreater(float(resumed.model.branch.detach().norm()), 0.)
                    self.assertEqual(float(resumed.optimizer.state[resumed.model.branch]["step"]), 4.)
                    self.assertGreater(resumed.branch_update_l2_sum, 0.)

    def test_optimizer_names_and_source_hash_are_validated(self):
        trainer = LoopTrainer(tiny_source(), "interaction", chunk_ticks=4)
        trainer.train_update()
        state = trainer.state_dict()
        broken = deepcopy(state)
        broken["optimizer_parameter_names"][0][0] = "encoder.weight"
        with self.assertRaises(ValueError):
            LoopTrainer._from_state(broken)
        broken = deepcopy(state)
        broken["sources"]["model.py"] += "\n# corruption\n"
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            LoopTrainer._from_state(broken)
        broken = deepcopy(state)
        broken["credit"]["continuation"]["optimizer"]["param_groups"][0]["params"][0:2] = [1, 0]
        with self.assertRaisesRegex(ValueError, "parameter IDs"):
            LoopTrainer._from_state(broken)
        self.assertEqual(trainer.report()["model"]["state_scalars"], 8)
        self.assertEqual(trainer.report()["branch_metrics"]["branch_weight_l2"], float(trainer.model.branch.detach().double().norm()))

    def test_baseline_exactly_matches_old_credit_bptt_for_four_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A.pt"
            tiny_source().save_checkpoint(path)
            base = Trainer.from_checkpoint(path)
            loop = LoopTrainer(base, "baseline", chunk_ticks=64)
            loop.run_updates(4)
            expected = deepcopy(CreditTrainer.semantic_state(loop))
            old = CreditTrainer.from_base_checkpoint(path, CreditConfig(mode="bp64", chunk_ticks=64,
                duration_range=(4., 8.), gap_range=(4., 12.), regime_name="loops-B"))
            old.run_updates(4)
            actual = old.semantic_state()
            for key in ("model", "optimizer", "hidden", "world", "rng", "replay", "stats", "ticks", "updates"):
                self.assertEqual(state_digest(expected["continuation"][key]), state_digest(actual["continuation"][key]), key)
            self.assertEqual(loop.action_hash, old.action_hash)
            self.assertEqual(loop.data_hash, old.data_hash)

    def test_invalid_gradients_rejected_before_any_parameter_or_optimizer_mutation(self):
        trainer = LoopTrainer(tiny_source(), "interaction", chunk_ticks=4)
        before_model = state_digest(trainer.model.state_dict())
        before_adam = state_digest(trainer.optimizer.state_dict())
        gradients = {name: torch.zeros_like(value) for name, value in trainer.model.named_parameters()}
        missing = dict(gradients)
        missing.pop("branch")
        with self.assertRaisesRegex(ValueError, "cover every"):
            trainer._apply_gradients(missing)
        gradients["branch"][0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "invalid gradient"):
            trainer._apply_gradients(gradients)
        self.assertEqual(before_model, state_digest(trainer.model.state_dict()))
        self.assertEqual(before_adam, state_digest(trainer.optimizer.state_dict()))

    def test_corrupt_model_hidden_and_adam_checkpoint_tensors_are_rejected(self):
        trainer = LoopTrainer(tiny_source(), "interaction", chunk_ticks=4)
        trainer.train_update()
        state = trainer.state_dict()
        for target in ("branch", "hidden", "adam-nan", "adam-shape"):
            with self.subTest(target=target):
                broken = deepcopy(state)
                continuation = broken["credit"]["continuation"]
                if target == "branch":
                    continuation["model"]["branch"][0, 0] = float("inf")
                elif target == "hidden":
                    continuation["hidden"][0, 0] = float("nan")
                elif target == "adam-nan":
                    continuation["optimizer"]["state"][4]["exp_avg"][0, 0] = float("nan")
                else:
                    continuation["optimizer"]["state"][4]["exp_avg"] = torch.zeros(1)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    LoopTrainer._from_state(broken)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_tiny_cuda_complete_resume_for_all_arms(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as folder:
                base = Trainer(Variant("loops-cuda-toy", width=8, policy="online"),
                    TrainConfig(seed=137, batch_size=2, tbptt=4, updates=2, device="cuda",
                        train_dts=(.25,), replay_capacity=2, replay_block_ticks=8, replay_burnin=4, log_every=2))
                base.run_updates(2)
                trainer = LoopTrainer(base, arm, chunk_ticks=4)
                trainer.train_update()
                path = Path(folder) / "checkpoint.pt"
                trainer.save_checkpoint(path)
                boundary = state_digest(trainer.semantic_state())
                trainer.run_updates(2)
                expected = state_digest(trainer.semantic_state())
                torch.rand(3, device="cuda")
                resumed = LoopTrainer.from_checkpoint(path, device="cuda")
                self.assertEqual(boundary, state_digest(resumed.semantic_state()))
                resumed.run_updates(2)
                self.assertEqual(expected, state_digest(resumed.semantic_state()))

    def test_cli_zero_tick_start_and_resume_do_not_train_or_change_source(self):
        base = Trainer(Variant("loops-cli-toy", width=128, policy="online"),
            TrainConfig(seed=137, batch_size=1, tbptt=2, updates=1, device="cpu",
                train_dts=(.25,), replay_capacity=1, replay_block_ticks=4, replay_burnin=2, log_every=2))
        base.run_updates(1)
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "A.pt", Path(folder) / "new"
            base.save_checkpoint(source)
            before = source.read_bytes()
            main(["--source", str(source), "--arm", "interaction", "--output", str(output),
                  "--target-ticks", "0", "--device", "cpu"])
            checkpoint = output / "checkpoint.pt"
            trainer = LoopTrainer.from_checkpoint(checkpoint)
            initial = state_digest(trainer.semantic_state())
            self.assertEqual(trainer.stage_ticks, 0)
            main(["--output", str(output), "--resume", "--target-ticks", "0", "--device", "cpu"])
            self.assertEqual(initial, state_digest(LoopTrainer.from_checkpoint(checkpoint).semantic_state()))
            self.assertEqual(before, source.read_bytes())


if __name__ == "__main__":
    unittest.main()
