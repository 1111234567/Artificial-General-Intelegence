"""Frozen random traces: algebra, causal gradients, workers and audit routing.

Every artifact written by these tests is temporary.  The small actual-world PG
fixture exercises the frozen v1 objective and its independent Adam recomputation;
no formal training or evaluation is started.
"""
from contextlib import ExitStack, redirect_stdout
from concurrent.futures import Future
from copy import deepcopy
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.associative_state.shared import SharedAssociationMemory
from experiments.learn_remember.gru_train import GRUAgent
from experiments.learned_binding import audit, common, frozen_trace, pretrain, run
from experiments.learned_binding.audit_test import make_fixture
from experiments.learned_binding.model_test import public_observations
from experiments.learned_binding.read_first import ReadFirstPolicy
from experiments.learned_binding.frozen_trace import FrozenTracePolicy


class FrozenTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9173)

    def activation_patches(self, stack):
        for module, name in ((common, "OUT"), (common, "NativeMemoryPolicy"),
                             (pretrain, "OUT"), (run, "OUT"), (audit, "NativeMemoryPolicy")):
            stack.enter_context(patch.object(module, name))

    def test_same_seed_full_state_dict_initialization_config_and_state_are_identical(self):
        for seed in (0, 1, 2):
            torch.manual_seed(372000 + seed)
            old = ReadFirstPolicy()
            torch.manual_seed(372000 + seed)
            new = FrozenTracePolicy()
            self.assertEqual(old.config(), new.config())
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for name, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
            for name, value in old.initial_state(3).items():
                self.assertTrue(torch.equal(value, new.initial_state(3)[name]), name)
            before, after = old.parameter_report(), new.parameter_report()
            self.assertTrue(after.pop("encoder_frozen"))
            self.assertEqual(before["trainable_parameters"], 8210)
            self.assertEqual(after["trainable_parameters"], 2258)
            before.pop("trainable_parameters")
            after.pop("trainable_parameters")
            self.assertEqual(before, after)
            self.assertEqual(after["encoder_parameters"], 5952)
            self.assertEqual(after["state_scalars"], 320)
            frozen_trace._check_trainability(new)
        self.assertIs(FrozenTracePolicy.forward, ReadFirstPolicy.forward)
        self.assertIs(FrozenTracePolicy.step, ReadFirstPolicy.step)
        self.assertEqual(inspect.signature(FrozenTracePolicy.__init__),
                         inspect.signature(ReadFirstPolicy.__init__))

    def test_forward_is_bitwise_equal_to_read_first_for_all_ablation_flags(self):
        old = ReadFirstPolicy().train()
        new = FrozenTracePolicy(**old.config()).train()
        new.load_state_dict(old.state_dict(), strict=True)
        obs, state = public_observations(13, 3), old.initial_state(3)
        for value in state.values():
            value.normal_(std=.2)
        for write, read in ((True, True), (False, True), (True, False), (False, False)):
            with self.subTest(write=write, read=read), torch.no_grad():
                expected = old(obs, state, write_enabled=write, read_enabled=read, return_diagnostics=True)
                actual = new(obs, state, write_enabled=write, read_enabled=read, return_diagnostics=True)
                common.assert_tree_equal(expected, actual)
                if not write:
                    self.assertIs(actual[1]["memory"], state["memory"])

    def test_future_action_ce_reaches_old_memory_writes_with_encoder_gradients_absent(self):
        model = FrozenTracePolicy().train()
        obs = public_observations(2, 2).reshape(2, 2, 2, 28)
        # Only the final tick is a command: earlier writes reach this CE via M.
        obs[0, ..., :3] = 0.
        obs[1, ..., 0] = 1.
        logits, _, diagnostic = model(obs.reshape(2, 2, 56), model.initial_state(2), return_diagnostics=True)
        for name in ("k", "value", "gate", "q"):
            diagnostic[name].retain_grad()
        F.cross_entropy(logits[1].reshape(-1, 11), torch.tensor([0, 3, 6, 9])).backward()
        for name in ("k", "value", "gate"):
            self.assertGreater(float(diagnostic[name].grad[0].abs().sum()), 0., name)
            self.assertEqual(float(diagnostic[name].grad[1].abs().sum()), 0., name)
        self.assertGreater(float(diagnostic["q"].grad[1].abs().sum()), 0.)
        for head in (model.query, model.key, model.value, model.gate):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0.)
        self.assertGreater(float(model.read_scale_raw.grad.abs()), 0.)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.executor.parameters()))

    def test_ce_adam_changes_heads_but_preserves_random_encoder_and_executor_bitwise(self):
        model = FrozenTracePolicy().train()
        initial = deepcopy(model.state_dict())
        obs = public_observations(7, 2)
        targets = torch.arange(7 * 2 * 2).remainder(11).reshape(7, 2, 2)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=common.SETTINGS["ce_lr"])
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 9)
        for _ in range(3):
            logits, _ = model(obs, model.initial_state(2))
            loss = pretrain.loss_for(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for name, value in initial.items():
            if name.startswith(("encoder.", "executor.")):
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        for head in ("query", "key", "value", "gate"):
            self.assertFalse(torch.equal(initial[head + ".weight"], model.state_dict()[head + ".weight"]), head)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_freezing_weights_keeps_input_and_initial_trace_jacobians(self):
        model = FrozenTracePolicy()
        obs = public_observations(5, 2).requires_grad_()
        state = model.initial_state(2)
        state["z"].requires_grad_()
        logits, _ = model(obs, state)
        logits[-1].square().sum().backward()
        self.assertGreater(float(obs.grad[0].abs().sum()), 0.)
        self.assertGreater(float(state["z"].grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def _step_sequence(self, device):
        model = FrozenTracePolicy().to(device).train()
        obs, state = public_observations(17, 3).to(device), model.initial_state(3)
        for write, read in ((True, True), (False, True), (True, False), (False, False)):
            with torch.no_grad():
                expected, final, diag = model(obs, state, write_enabled=write, read_enabled=read, return_diagnostics=True)
                current, rows, reads = state, [], []
                for frame in obs:
                    logits, current, step_diag = model.step(frame, current, write_enabled=write,
                        read_enabled=read, return_diagnostics=True)
                    rows.append(logits)
                    reads.append(step_diag["read_probs"])
                torch.testing.assert_close(torch.stack(rows), expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(torch.stack(reads), diag["read_probs"], rtol=2e-5, atol=2e-6)
                for name in state:
                    torch.testing.assert_close(current[name], final[name], rtol=2e-5, atol=2e-6)

    def test_cpu_training_mode_step_sequence(self):
        self._step_sequence("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_step_sequence_and_cpu_gradients(self):
        previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            self._step_sequence("cuda")
            cpu = FrozenTracePolicy().train()
            cuda = deepcopy(cpu).cuda().train()
            obs = public_observations(13, 2)
            targets = torch.arange(13 * 2 * 2).remainder(11)
            cpu_logits, cpu_state = cpu(obs, cpu.initial_state(2))
            cuda_logits, cuda_state = cuda(obs.cuda(), cuda.initial_state(2))
            torch.testing.assert_close(cuda_logits.cpu(), cpu_logits, rtol=2e-5, atol=2e-6)
            for name in cpu_state:
                torch.testing.assert_close(cuda_state[name].cpu(), cpu_state[name], rtol=2e-5, atol=2e-6)
            F.cross_entropy(cpu_logits.reshape(-1, 11), targets).backward()
            F.cross_entropy(cuda_logits.reshape(-1, 11), targets.cuda()).backward()
            for (name, a), (other, b) in zip(cpu.named_parameters(), cuda.named_parameters()):
                self.assertEqual(name, other)
                if a.requires_grad:
                    torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=2e-4, atol=2e-6)
                else:
                    self.assertIsNone(a.grad)
                    self.assertIsNone(b.grad)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous

    def test_activate_factory_checkpoint_loader_and_audit_use_frozen_trace(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            folder = Path(temporary)
            self.activation_patches(stack)
            stack.enter_context(patch.object(frozen_trace, "OUT", folder))
            stack.enter_context(patch.object(common, "executor", side_effect=lambda seed: GRUAgent().eval()))
            self.assertEqual(frozen_trace.activate(), folder)
            self.assertEqual((common.OUT, pretrain.OUT, run.OUT), (folder,) * 3)
            self.assertIs(common.NativeMemoryPolicy, FrozenTracePolicy)
            self.assertIs(audit.NativeMemoryPolicy, FrozenTracePolicy)
            model = common.new_native(0)
            self.assertIsInstance(model, FrozenTracePolicy)
            path = common.native_checkpoint(0)
            path.parent.mkdir(parents=True)
            torch.save(dict(model_config=model.config(), model=model.state_dict()), path)
            loaded = common.load_native(0)
            common.assert_tree_equal(model.state_dict(), loaded.state_dict())
            frozen_trace._check_trainability(loaded)
            rebuilt = audit.model_from_snapshot(dict(kind="learned", model=model.state_dict()))
            self.assertIsInstance(rebuilt, FrozenTracePolicy)
            frozen_trace._check_trainability(rebuilt)

    def test_windows_worker_wrappers_reactivate_before_running(self):
        with patch.object(frozen_trace, "activate") as activate, \
             patch.object(pretrain, "train_one", side_effect=lambda job: job[0]) as train, \
             patch.object(run, "worker", return_value=dict(seed=2, kind="learned")) as behavior:
            self.assertEqual(frozen_trace._train_worker((1, "cpu")), 1)
            activate.assert_called_once_with()
            train.assert_called_once_with((1, "cpu"))
            activate.reset_mock()
            self.assertEqual(frozen_trace._behavior_worker(2), dict(seed=2, kind="learned"))
            activate.assert_called_once_with()
            behavior.assert_called_once_with((2, "learned"))

    def test_train_driver_keeps_mixed_devices_and_freezes_only_after_all_endpoints(self):
        submitted, activations = [], []

        class TemporaryPool:
            def __init__(self, max_workers, initializer):
                activations.append(initializer)
                initializer()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, job):
                submitted.append((fn, job))
                future = Future()
                try:
                    future.set_result(fn(job))
                except Exception as error:
                    future.set_exception(error)
                return future

        def fake_train(job):
            seed, device = job
            self.assertFalse((frozen_trace.OUT / "training-freeze.json").exists())
            model = common.new_native(seed)
            path = common.native_checkpoint(seed)
            path.parent.mkdir(parents=True)
            torch.save(dict(model_config=model.config(), model=model.state_dict(),
                updates=512, settings=common.SETTINGS), path)
            common.write_json(path.with_name("training.json"), dict(
                checkpoint_sha256=common.sha(path), executor_unchanged=True, device=device))
            return seed

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            output = root / "frozen-trace-v3"
            output.mkdir()
            common.write_json(output / "protocol.json", {})
            self.activation_patches(stack)
            stack.enter_context(patch.object(frozen_trace, "ROOT", root))
            stack.enter_context(patch.object(frozen_trace, "OUT", output))
            stack.enter_context(patch.object(common, "executor", side_effect=lambda seed: GRUAgent().eval()))
            stack.enter_context(patch.object(common, "check_protocol"))
            stack.enter_context(patch.object(frozen_trace, "ProcessPoolExecutor", TemporaryPool))
            stack.enter_context(patch.object(pretrain, "train_one", side_effect=fake_train))
            with redirect_stdout(io.StringIO()):
                freeze = frozen_trace.train_all()
            self.assertEqual([job for _, job in submitted], [(0, "cuda"), (1, "cpu"), (2, "cpu")])
            self.assertTrue(all(fn is frozen_trace._train_worker for fn, _ in submitted))
            self.assertEqual(activations, [frozen_trace.activate])
            self.assertTrue(freeze["all_complete_before_scoring"])
            self.assertTrue(freeze["encoder_matches_seeded_initialization"])
            self.assertEqual(freeze["architecture"], "FrozenTracePolicy")
            self.assertEqual(len(freeze["checkpoints"]), 3)
            self.assertEqual(freeze["devices"], {"0": "cuda", "1": "cpu", "2": "cpu"})
            self.assertEqual(freeze["trainable_parameters"], 2258)
            with self.assertRaises(FileExistsError):
                frozen_trace.train_all()
            with self.assertRaisesRegex(ValueError, "seed0 CUDA"):
                frozen_trace.train_all("cpu")

    def test_small_actual_world_pg_and_independent_audit_use_only_trainable_heads(self):
        native = FrozenTracePolicy(original_mappings=common.mapping_table(0, "related")).eval()
        initial_encoder = deepcopy(native.encoder.state_dict())
        manual = SharedAssociationMemory(kind="delta")
        settings = dict(common.SETTINGS, seeds=[0], actors=2, support_ticks=16,
            probe_ticks=40, pg_blocks=2, pg_ticks=4, support_duration=[1., 1.], support_gap=[1., 1.])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manual_path = root / "manual.pt"
            torch.save(dict(model=manual.state_dict(), metadata=dict(model_config=dict(
                kind="delta", key_dim=16, n_keys=8, n_values=4))), manual_path)
            with patch.dict(common.SETTINGS, settings), \
                 patch.object(common, "load_native", side_effect=lambda seed: deepcopy(native)), \
                 patch.object(common, "executor", side_effect=lambda seed: deepcopy(native.executor)), \
                 patch.object(common, "manual_checkpoint", return_value=manual_path), \
                 patch.object(audit, "NativeMemoryPolicy", FrozenTracePolicy), redirect_stdout(io.StringIO()):
                make_fixture(root, settings)
                result = audit.audit_all(root, settings=settings, strict_sources=False)
            self.assertEqual(result["probe_count"], 13)
            self.assertEqual(result["B_block_count"], 8)
            self.assertEqual(result["native_replay_count"], 3)
            self.assertEqual(result["independently_recomputed_pg_updates"], 2)
            for phase in result["B_phases"]:
                for check in phase["pg_checks"]:
                    self.assertTrue(check["model_and_adam_bitwise_equal"])
            for path in (root / "seed_000/behavior/learned/outcome_B").glob("block_*.pt"):
                block = torch.load(path, weights_only=False)
                common.assert_tree_equal(initial_encoder, frozen_trace._encoder_state(block["after_update"]["model"]))
                self.assertEqual(len(block["optimizer_after"]["param_groups"][0]["params"]), 9)

    def test_manual_copy_records_v1_audit_and_every_file_hash_and_rejects_damage(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            old, new = repository / "v1", repository / "v3"
            for seed in range(3):
                for kind in ("manual_hard", "manual_soft"):
                    folder = old / f"seed_{seed:03d}/behavior" / kind
                    folder.mkdir(parents=True)
                    common.write_json(folder / "completion.json", dict(seed=seed, kind=kind))
                    (folder / "fixture.bin").write_bytes(b"unchanged control")
            with patch.object(frozen_trace, "ROOT", repository), patch.object(frozen_trace, "OUT", new), \
                 patch.object(frozen_trace, "V1_OUT", old), patch.object(frozen_trace, "activate"), \
                 patch.object(frozen_trace, "_parent_audit", return_value=dict(path="v1/audit.json", sha256="original audit")), \
                 patch.object(frozen_trace, "_check_training_barrier"):
                manifest = frozen_trace.reuse_manual_controls()
                self.assertEqual(len(manifest["copies"]), 6)
                self.assertFalse(manifest["controls_reexecuted"])
                self.assertEqual(manifest["parent_audit"]["path"], "v1/audit.json")
                for row in manifest["copies"]:
                    self.assertEqual(frozen_trace._tree_hashes(repository / row["source"]),
                                     frozen_trace._tree_hashes(repository / row["destination"]))
                    self.assertEqual(len(row["files_sha256"]), 2)
                (new / "seed_000/behavior/manual_hard/fixture.bin").write_bytes(b"damaged")
                with self.assertRaisesRegex(AssertionError, "differs from v1"):
                    frozen_trace.reuse_manual_controls()

    def test_declare_freezes_correct_sources_plan_and_identical_v1_manifest(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            repository = Path(temporary)
            old, second, new = (repository / name for name in ("v1", "v2", "v3"))
            for folder in (old, second):
                folder.mkdir()
            sources = {name: repository / name for name in
                       ("frozen.py", "frozen_trace.py", "frozen-trace-plan.md", "read_first.py", "audit.py")}
            for path in sources.values():
                path.write_text("temporary fixture\n", encoding="utf-8")
            prototype = dict(settings=common.SETTINGS, sources={"frozen.py": common.sha(sources["frozen.py"])}, origins={})
            complete = dict(probe_count=39, B_block_count=384, native_replay_count=9,
                independently_recomputed_pg_updates=6, all_world_actions_and_public_rewards_replayed=True,
                all_controller_live_states_preserved_across_optimizer_updates=True, all_source_executors_unchanged=True)
            for folder in (old, second):
                common.write_json(folder / "protocol.json", prototype)
                common.write_json(folder / "training-freeze.json", {})
                common.write_json(folder / "audit.json", complete)
            original_data = b'{"streams":[{"path":"v1/original.pt"}]}\n'
            (old / "data-freeze.json").write_bytes(original_data)
            self.activation_patches(stack)
            for name, value in dict(ROOT=repository, OUT=new, V1_OUT=old, READ_FIRST_OUT=second,
                SOURCE=sources["frozen_trace.py"], PLAN=sources["frozen-trace-plan.md"],
                READ_FIRST_SOURCE=sources["read_first.py"]).items():
                stack.enter_context(patch.object(frozen_trace, name, value))
            stack.enter_context(patch.object(audit, "__file__", str(sources["audit.py"])))
            stack.enter_context(patch.object(common, "declare", side_effect=lambda: common.write_json(new / "protocol.json", prototype)))
            protocol = frozen_trace.declare()
            self.assertEqual((new / "data-freeze.json").read_bytes(), original_data)
            self.assertEqual(protocol["experiment"], "learned-binding-frozen-trace-v3")
            self.assertTrue(protocol["followup"]["declared_after_v1_and_v2_results_inspected"])
            self.assertTrue(protocol["followup"]["identical_initial_state_dict_and_forward_to_read_first"])
            self.assertFalse(protocol["followup"]["binding_labels_used_for_policy_training"])
            self.assertNotIn("only_policy_computation_change", protocol["followup"])
            self.assertEqual(protocol["trainable_parameters"], 2258)
            for name in ("frozen_trace.py", "frozen-trace-plan.md", "read_first.py", "audit.py"):
                self.assertEqual(protocol["sources"][name], common.sha(sources[name]))
            self.assertFalse(protocol["manual_behavior_reuse"]["controls_reexecuted"])
            with self.assertRaises(FileExistsError):
                frozen_trace.declare()

    def test_behavior_refuses_missing_or_incomplete_three_seed_training_barrier(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(frozen_trace, "OUT", root))
            stack.enter_context(patch.object(frozen_trace, "ROOT", root))
            stack.enter_context(patch.object(common, "OUT", root))
            stack.enter_context(patch.object(run, "check_freeze"))
            with self.assertRaises(FileNotFoundError):
                frozen_trace._check_training_barrier()
            common.write_json(root / "protocol.json", {})
            common.write_json(root / "training-freeze.json", dict(checkpoints={}))
            with self.assertRaisesRegex(ValueError, "all three"):
                frozen_trace._check_training_barrier()
            freeze = dict(checkpoints={str(common.native_checkpoint(seed).relative_to(root)): "fixture" for seed in range(3)},
                seeds=[0, 1, 2], architecture="FrozenTracePolicy", encoder_matches_seeded_initialization=True,
                source_protocol_sha256=common.sha(root / "protocol.json"))
            common.write_json(root / "training-freeze.json", freeze)
            frozen_trace._check_training_barrier()
            freeze["architecture"] = "ReadFirstPolicy"
            common.write_json(root / "training-freeze.json", freeze)
            with self.assertRaisesRegex(ValueError, "all three"):
                frozen_trace._check_training_barrier()

    def test_audit_cli_reconstructs_frozen_model_and_checks_every_learned_snapshot(self):
        report = dict(probe_count=39, B_block_count=384, native_replay_count=9,
                      independently_recomputed_pg_updates=6)
        initial = deepcopy(FrozenTracePolicy().encoder.state_dict())
        model_state = {"encoder." + name: value for name, value in initial.items()}
        original_check = audit.check_snapshot

        def fake_audit(root):
            self.assertIs(audit.NativeMemoryPolicy, FrozenTracePolicy)
            audit.check_snapshot(dict(kind="learned", seed=0, model=model_state))
            return dict(report)

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            folder = Path(temporary)
            self.activation_patches(stack)
            stack.enter_context(patch.object(frozen_trace, "OUT", folder))
            stack.enter_context(patch.object(frozen_trace, "_check_training_barrier"))
            stack.enter_context(patch.object(frozen_trace, "_initial_encoder", side_effect=lambda seed: deepcopy(initial)))
            stack.enter_context(patch.object(audit, "load_pt", return_value=dict(model=model_state)))
            stack.enter_context(patch.object(audit, "check_snapshot"))
            stack.enter_context(patch.object(audit, "audit_all", side_effect=fake_audit))
            result = frozen_trace.audit_all()
            self.assertEqual(result["probe_count"], 39)
            self.assertEqual(result["learned_snapshot_encoder_checks"], 1)
            saved = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["variant"], "frozen-trace-v3")
            self.assertEqual(saved["frozen_trace_source_sha256"], common.sha(frozen_trace.SOURCE))
            model_state["encoder.weight_ih_l0"] = model_state["encoder.weight_ih_l0"] + .01
            with self.assertRaisesRegex(audit.AuditError, "CE changed random encoder"):
                frozen_trace.audit_all()
        self.assertIs(audit.check_snapshot, original_check)


if __name__ == "__main__":
    unittest.main()
