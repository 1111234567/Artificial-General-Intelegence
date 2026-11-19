"""Factor continuation mechanics and deterministic resume, not a formal study."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.continuity import run as continuity
from experiments.continuity.run import TrainConfig, Trainer, Variant, state_digest
from experiments.factors.run import (
    BUCKETS, FactorTrainer, chunk_diagnostics, factor_loss, run_to_target,
)
from experiments.factors.structure import FactorConfig, MODES
from experiments.rewiring.plasticity import RewireConfig
from experiments.rewiring.run import RewireTrainer
from experiments.rewiring.test_run import disturb_rng


FORMAL_SOURCE = Path(continuity.ROOT) / "data/rewiring/main-v1/training/seed_000/retire_high/checkpoint_B.pt"
CORE_KEYS = ("model", "optimizer", "hidden", "world", "rng", "replay", "stats", "ticks",
             "updates", "gradient_updates", "hidden_resets", "dt_history", "action_hash", "data_hash")


def tiny_b(device="cpu"):
    base = Trainer(Variant("factor-test", width=6, policy="online"),
        TrainConfig(seed=179, batch_size=2, tbptt=4, updates=2, device=device,
            train_dts=(.25,), replay_capacity=2, replay_block_ticks=8,
            replay_burnin=4, log_every=2))
    base.run_updates(2)
    with torch.no_grad():
        base.model.recurrent[::2, ::2] = 0.
    source = RewireTrainer(base, RewireConfig(arm="retire", shrink_step=2e-4,
        window_updates=2, structural_seed=315), chunk_ticks=4, phase_ticks=8)
    source.run_updates(2)
    return source


def factor_for(mode="retire", idle_lambda=0., device="cpu"):
    return FactorTrainer(tiny_b(device), FactorConfig(mode=mode, idle_lambda=idle_lambda))


class FactorLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_lambda_zero_passes_the_exact_original_ce_tensor_and_gradients(self):
        logits = torch.tensor([[[[.2, -.1, 1.]]]], requires_grad=True)
        targets = torch.tensor([[[0]]])
        sentinel = torch.nn.functional.cross_entropy(logits.flatten(0, 2), targets.flatten(),
                                                     weight=torch.tensor([.1, 1., 1.]))

        class Base:
            def _loss(self, z, y):
                self_z, self_y = z, y
                assert self_z is logits and self_y is targets
                return sentinel

        loss, base, idle = factor_loss(Base(), logits, targets, torch.ones_like(targets, dtype=torch.bool), 0.)
        self.assertIs(loss, sentinel)
        self.assertIs(base, sentinel)
        self.assertGreater(float(idle.detach()), 0)
        expected = torch.autograd.grad(sentinel, logits, retain_graph=True)[0]
        actual = torch.autograd.grad(loss, logits)[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_empty_idle_and_active_wait_add_no_penalty(self):
        trainer = factor_for(idle_lambda=1.)
        z = torch.randn(2, 2, 4, 11, requires_grad=True)
        y = torch.zeros((2, 2, 4), dtype=torch.long)
        idle = torch.zeros_like(y, dtype=torch.bool)  # Every WAIT has an active job.
        loss, original, penalty = factor_loss(trainer.base, z, y, idle, 1.)
        self.assertIs(loss, original)
        self.assertEqual(float(penalty), 0.)
        active, enabled = torch.ones_like(idle), torch.ones_like(idle)
        diagnostic = chunk_diagnostics(z, y, z.detach().argmax(-1), idle, active, enabled, trainer.loss_weights)
        self.assertEqual(diagnostic["idle_count"], 0)
        self.assertEqual(diagnostic["buckets"]["active_wait"]["count"], y.numel())

    def test_idle_normalization_is_independent_of_weighted_ce_mass(self):
        trainer = factor_for(idle_lambda=.1)
        z = torch.tensor([[[[0., 2.] + [0.] * 9, [1., 0.] + [0.] * 9]]], requires_grad=True)
        y = torch.tensor([[[0, 1]]])
        idle = torch.tensor([[[True, False]]])
        loss, base, idle_nll = factor_loss(trainer.base, z, y, idle, .1)
        expected = -z.log_softmax(-1)[0, 0, 0, 0]
        self.assertTrue(torch.equal(idle_nll, expected))
        self.assertTrue(torch.equal(loss, base + .1 * expected))


class FactorRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_graft_preserves_inherited_objects_counters_and_pending_contracts(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                source = tiny_b()
                before = deepcopy(source.semantic_state())
                model, optimizer, world, hidden = source.model, source.optimizer, source.world, source.h
                factor = FactorTrainer(source, FactorConfig(mode=mode, idle_lambda=.1))
                self.assertIs(factor.model, model)
                self.assertIs(factor.optimizer, optimizer)
                self.assertIs(factor.world, world)
                self.assertIs(factor.h, hidden)
                self.assertEqual(factor.stage_ticks, 8)
                self.assertEqual(factor.stage_updates, 2)
                self.assertEqual((factor.factor_ticks, factor.factor_updates), (0, 0))
                self.assertEqual((factor.factor_start_tick, factor.factor_start_update), (16, 4))
                self.assertEqual(factor.factor_origin["inherited_semantic_hash"], state_digest(before))
                self.assertEqual(factor.operator.updates, 2)
                after = factor.base.semantic_state()
                old = before["credit"]["continuation"]
                for key in CORE_KEYS:
                    if key != "world":
                        self.assertEqual(state_digest(old[key]), state_digest(after[key]), key)
                for key in old["world"]:
                    if key not in {"regime", "regime_id", "regime_history"}:
                        self.assertEqual(state_digest(old["world"][key]), state_digest(after["world"][key]), key)
                self.assertEqual(len(factor.regime_changes), 2)
                self.assertEqual(factor.regime_changes[-1]["stage_tick"], 8)
                self.assertEqual(factor.world.regime.name, "rewiring-A2")

    def _resume_case(self, mode, idle_lambda, device):
        with tempfile.TemporaryDirectory() as folder:
            trainer = factor_for(mode, idle_lambda, device)
            trainer.train_update()
            checkpoint = Path(folder) / "middle.pt"
            trainer.save_checkpoint(checkpoint)
            boundary = state_digest(trainer.semantic_state())
            trainer.run_updates(5)
            expected = state_digest(trainer.semantic_state())
            disturb_rng()
            resumed = FactorTrainer.from_checkpoint(checkpoint, device=device)
            self.assertEqual(boundary, state_digest(resumed.semantic_state()))
            resumed.run_updates(5)
            self.assertEqual(expected, state_digest(resumed.semantic_state()))
            self.assertEqual((resumed.factor_ticks, resumed.factor_updates), (24, 6))
            self.assertEqual((resumed.stage_ticks, resumed.stage_updates), (32, 8))
            self.assertEqual(len(resumed.regime_changes), 2)

    def test_complete_cpu_resume_all_eighteen_modes_and_losses(self):
        for mode in MODES:
            for idle_lambda in (0., .1, 1.):
                with self.subTest(mode=mode, idle_lambda=idle_lambda):
                    self._resume_case(mode, idle_lambda, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_complete_cuda_resume_all_eighteen_modes_and_losses(self):
        for mode in MODES:
            for idle_lambda in (0., .1, 1.):
                with self.subTest(mode=mode, idle_lambda=idle_lambda):
                    self._resume_case(mode, idle_lambda, "cuda")

    def _retire_matches_old(self, device, formal=False):
        with tempfile.TemporaryDirectory() as folder:
            path = FORMAL_SOURCE if formal else Path(folder) / "B.pt"
            if not formal:
                tiny_b(device).save_checkpoint(path)
            old = RewireTrainer.from_checkpoint(path, device=device)
            old.run_updates(4)
            expected = deepcopy(old.semantic_state())
            old_points = deepcopy(old.trace[-4:])
            if formal:
                factor = FactorTrainer.from_source_checkpoint(path, FactorConfig(mode="retire"), device=device)
            else:
                factor = FactorTrainer(RewireTrainer.from_checkpoint(path, device=device), FactorConfig(mode="retire"))
            factor.run_updates(4)
            actual = factor.semantic_state()
            for key in CORE_KEYS:
                self.assertEqual(state_digest(expected["credit"]["continuation"][key]),
                                 state_digest(actual["credit"]["continuation"][key]), key)
            for key in ("work", "update_norms", "sampler_rng", "regime_changes"):
                self.assertEqual(state_digest(expected["credit"][key]), state_digest(actual["credit"][key]), key)
            self.assertEqual(state_digest(expected["rewiring"]), state_digest(factor.operator.rewire.state_dict()))
            for a, b in zip(old_points, factor.trace[-4:]):
                for key in ("action_hash", "data_hash", "event_hash", "live_hidden_hash", "action_chunk_hash",
                            "data_chunk_hash", "actual_action_counts", "target_action_counts", "loss"):
                    self.assertEqual(a[key], b[key], key)
                for key in ("gradient_norm", "clip_scale", "all_delta", "recurrent_delta", "encoder_delta", "head_delta",
                            "potential_recurrent_gradient_norm", "raw_parameter_gradient_norms"):
                    self.assertEqual(a["parameter_update"][key], b["parameter_update"][key], key)

    def test_retire_lambda_zero_matches_old_four_chunks_cpu(self):
        self._retire_matches_old("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_retire_lambda_zero_matches_old_four_chunks_cuda(self):
        self._retire_matches_old("cuda")

    @unittest.skipUnless(FORMAL_SOURCE.is_file(), "formal sparse B source unavailable")
    def test_actual_sparse_B_retire_lambda_zero_matches_old_four_chunks_cpu(self):
        self._retire_matches_old("cpu", formal=True)

    @unittest.skipUnless(FORMAL_SOURCE.is_file() and torch.cuda.is_available(), "CUDA/source unavailable")
    def test_actual_sparse_B_retire_lambda_zero_matches_old_four_chunks_cuda(self):
        self._retire_matches_old("cuda", formal=True)

    def test_idle_mask_is_captured_before_action_can_clear_active(self):
        trainer = factor_for(idle_lambda=1.)
        world = trainer.world
        original_act = world.act
        active_before, enabled_before, targets = [], [], []
        original_oracle = world.oracle

        def oracle(policy):
            result = original_oracle(policy)
            targets.append(result.clone())
            return result

        def act(action):
            active_before.append(world.active.clone())
            enabled_before.append((torch.arange(world.n_jobs) < world.active_jobs).expand_as(world.active).clone())
            original_act(action)
            # A successful FINISH also clears active in the real world. Clearing
            # all active flags makes any post-action alias/capture obvious.
            world.active.zero_()

        world.oracle, world.act = oracle, act
        _, metrics, chunk = trainer._live_chunk()
        expected_idle = torch.stack(enabled_before) & ~torch.stack(active_before)
        self.assertGreater(int(torch.stack(active_before).sum()), 0)
        self.assertEqual(metrics["diagnostics"]["idle_count"], int(expected_idle.sum()))
        self.assertLess(metrics["diagnostics"]["idle_count"], expected_idle.numel())
        self.assertTrue(torch.equal(chunk["targets"], torch.stack(targets)))
        self.assertEqual(sum(metrics["diagnostics"]["buckets"][name]["count"] for name in BUCKETS), expected_idle.numel())

    def test_live_gradient_mask_precedes_clip_and_reported_delta_is_actual(self):
        trainer = factor_for("fixed_shrink")
        gradient = {name: torch.full_like(value, .01) for name, value in trainer.model.named_parameters()}
        gradient["recurrent"][~trainer.operator.allowed_mask] = 1e6
        expected = torch.cat([value.flatten() if name != "recurrent" else
                              trainer.operator.before_gradient(value).flatten() for name, value in gradient.items()]).norm()
        before = trainer.model.recurrent.detach().double().clone()
        result = trainer._apply_gradients(gradient)
        self.assertAlmostEqual(result["gradient_norm"], float(expected), places=5)
        self.assertGreater(result["potential_recurrent_gradient_norm"], 1e6)
        self.assertEqual(result["recurrent_delta"], float((trainer.model.recurrent.detach().double() - before).norm()))

    def test_run_target_counts_only_new_ticks_and_writes_full_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = factor_for()
            with redirect_stdout(io.StringIO()):
                run_to_target(trainer, 12, folder, checkpoint_every=2, stop_after_updates=1)
            path = Path(folder)
            self.assertEqual((trainer.factor_ticks, trainer.stage_ticks), (4, 12))
            resumed = FactorTrainer.from_checkpoint(path / "checkpoint.pt")
            with redirect_stdout(io.StringIO()):
                run_to_target(resumed, 12, folder, checkpoint_every=2)
            self.assertEqual((resumed.factor_ticks, resumed.stage_ticks), (12, 20))
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["factor_updates"], 3)
            self.assertEqual(report["diagnostics"]["chunks"], 3)
            self.assertTrue((path / "checkpoint_000002.pt").is_file())
            for name in ("run.py", "structure.py", "test_run.py", "test_structure.py"):
                self.assertTrue((path / "sources/factors" / name).is_file(), name)
            self.assertTrue((path / "sources/dependencies/rewiring/plasticity.py").is_file())

    def test_resume_rejects_counter_or_objective_mismatch(self):
        trainer = factor_for()
        trainer.train_update()
        for key in ("factor_ticks", "factor_updates", "factor_start_tick", "factor_start_update"):
            with self.subTest(key=key):
                state = trainer.state_dict()
                state[key] += 1
                with self.assertRaises(ValueError):
                    FactorTrainer._from_state(state)
        state = trainer.state_dict()
        state["factor_config"]["idle_lambda"] = 1.
        with self.assertRaises(ValueError):
            FactorTrainer._from_state(state)


if __name__ == "__main__":
    unittest.main()
