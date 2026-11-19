"""Shared-gradient and orchestration checks without formal training/scoring."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from experiments.associative_state.model import AssociationMemory, KINDS
from experiments.associative_state import shared


class SharedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_keys_and_operations_exactly_match_the_original_seed(self):
        keys, values = torch.arange(8), torch.arange(8).remainder(4)
        for kind in KINDS:
            torch.manual_seed(113)
            original = AssociationMemory(kind)
            torch.manual_seed(113)
            model = shared.SharedAssociationMemory(kind)
            self.assertTrue(torch.equal(model.key_embedding.codes, original.key_embedding.weight))
            self.assertTrue(torch.equal(model._key(keys), original._key(keys)))
            if kind == "recurrent":
                for name, value in original.core.state_dict().items():
                    self.assertTrue(torch.equal(value, model.core.state_dict()[name]))
            state = original.initial_state(8)
            expected = original.write(state, keys, values)
            actual = model.write(state, keys, values)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(model.read(actual, keys), original.read(expected, keys)))

    def test_parameter_and_fixed_constant_budgets_are_separate(self):
        for kind in KINDS:
            model = shared.SharedAssociationMemory(kind)
            report = model.parameter_report()
            expected = 16769 if kind == "recurrent" else 258
            self.assertEqual(report["total_parameters"], expected)
            self.assertEqual(report["trainable_parameters"], expected)
            self.assertEqual(report["shared_encoder_parameters"], 256)
            self.assertEqual(report["fixed_code_scalars"], 128)
            self.assertEqual(report["fixed_code_bytes"], 512)
            self.assertEqual(report["buffers_by_name"], {"key_embedding.codes": 128})
            self.assertFalse(report["fixed_codes_trainable"])
            self.assertFalse(report["fixed_codes_in_parameter_count"])
            self.assertFalse(report["fixed_codes_in_fast_state_count"])
            self.assertEqual(report["state_scalars"], 64)
            self.assertEqual(report["encoder"], shared.ENCODER)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertNotIn("key_embedding.codes", dict(model.named_parameters()))
            self.assertFalse(model.key_embedding.codes.requires_grad)

    def test_add_delta_are_parameter_and_buffer_isomorphic(self):
        torch.manual_seed(114)
        additive = shared.SharedAssociationMemory("add")
        torch.manual_seed(114)
        delta = shared.SharedAssociationMemory("delta")
        self.assertEqual(set(dict(additive.named_parameters())),
                         {"eta_logit", "read_scale_raw", "key_embedding.projection.weight"})
        self.assertEqual(set(additive.state_dict()), set(delta.state_dict()))
        for name, value in additive.state_dict().items():
            self.assertTrue(torch.equal(value, delta.state_dict()[name]))
        delta.load_state_dict(additive.state_dict(), strict=True)

    def test_b_only_online_step_can_change_a_queries_without_changing_memory(self):
        for kind in KINDS:
            torch.manual_seed(115)
            model = shared.SharedAssociationMemory(kind).double()
            memory = torch.randn(8, 64, dtype=torch.float64)
            before_memory = memory.clone()
            before_codes = model.key_embedding.codes.clone()
            a_keys = torch.arange(8).remainder(4)
            b_keys = 4 + torch.arange(8).remainder(4)
            before_queries = model._key(a_keys).detach().clone()
            before_scores = model.read(memory, a_keys).detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=.03)
            loss = F.cross_entropy(model.read(memory.detach(), b_keys),
                                   torch.tensor([1, 2, 3, 0, 2, 3, 0, 1]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = model.key_embedding.projection.weight.grad
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)
            optimizer.step()
            self.assertFalse(torch.equal(model._key(a_keys), before_queries))
            self.assertFalse(torch.equal(model.read(memory, a_keys), before_scores))
            self.assertTrue(torch.equal(memory, before_memory))
            self.assertTrue(torch.equal(model.key_embedding.codes, before_codes))
            # Keep the v1 structural diagnosis explicit: B-only gradients do
            # not reach A rows in its independent embedding lookup table.
            torch.manual_seed(115)
            original = AssociationMemory(kind).double()
            old_a = original._key(a_keys).detach().clone()
            old_labels = original.read(memory, a_keys).argmax(-1)
            old_optimizer = torch.optim.Adam(original.parameters(), lr=.03)
            F.cross_entropy(original.read(memory.detach(), b_keys),
                            torch.tensor([1, 2, 3, 0, 2, 3, 0, 1])).backward()
            self.assertEqual(int(original.key_embedding.weight.grad[:4].count_nonzero()), 0)
            old_optimizer.step()
            self.assertTrue(torch.equal(original._key(a_keys), old_a))
            self.assertTrue(torch.equal(original.read(memory, a_keys).argmax(-1), old_labels))

    def test_checkpoint_restores_fixed_codes_projection_config_and_state(self):
        for kind in KINDS:
            torch.manual_seed(116)
            model = shared.SharedAssociationMemory(kind).double()
            with torch.no_grad():
                model.key_embedding.projection.weight.add_(.01 * torch.randn(16, 16))
            state = model.write(model.initial_state(2), torch.tensor([1, 6]), torch.tensor([2, 0]))
            buffer = io.BytesIO()
            torch.save(dict(config=model.config(), model=model.state_dict(), state=state), buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
            restored = shared.SharedAssociationMemory(**checkpoint["config"]).double()
            restored.load_state_dict(checkpoint["model"], strict=True)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))
            self.assertEqual(model.config(), restored.config())
            self.assertTrue(torch.equal(model.read(state, torch.tensor([1, 6])),
                                        restored.read(checkpoint["state"], torch.tensor([1, 6]))))


class SharedOrchestrationTest(unittest.TestCase):
    def setUp(self):
        # Activation is intentionally process-local; restore the old module
        # bindings so this test cannot redirect unrelated v1 runner tests.
        self.bindings = [(shared.base_train, "OUT"), (shared.base_evaluate, "OUT"),
                         (shared.base_launch, "OUT"), (shared.base_train, "AssociationMemory"),
                         (shared.base_evaluate, "AssociationMemory")]
        self.original = [getattr(module, name) for module, name in self.bindings]

    def tearDown(self):
        for (module, name), value in zip(self.bindings, self.original):
            setattr(module, name, value)

    def test_activation_redirects_every_runner_and_is_idempotent(self):
        self.assertEqual(shared.OUT.name, "shared-v2")
        for _ in range(2):
            self.assertEqual(shared.activate(), shared.OUT)
            for module in (shared.base_train, shared.base_evaluate, shared.base_launch):
                self.assertEqual(module.OUT, shared.OUT)
            self.assertIs(shared.base_train.AssociationMemory, shared.SharedAssociationMemory)
            self.assertIs(shared.base_evaluate.AssociationMemory, shared.SharedAssociationMemory)

    def test_declaration_extends_original_protocol_and_discloses_post_v1_design(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(shared, "OUT", Path(directory)), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(shared.base_train, "sha", return_value="source-digest"):
            def original_declare():
                shared.base_train.write_json(shared.OUT / "protocol.json",
                                             dict(settings=deepcopy(shared.base_train.SETTINGS),
                                                  arms=list(shared.base_train.ARMS), sources={}, origins={}))
            with patch.object(shared.base_train, "declare", side_effect=original_declare) as declaration:
                protocol = shared.declare()
            declaration.assert_called_once_with()
            self.assertEqual(protocol["settings"], shared.base_train.SETTINGS)
            self.assertIn(str(shared.SOURCE.relative_to(shared.ROOT)), protocol["sources"])
            self.assertIn(str(shared.PLAN.relative_to(shared.ROOT)), protocol["sources"])
            self.assertEqual(protocol["encoder"], shared.ENCODER)
            self.assertTrue(protocol["followup"]["declared_after_inspection_of_v1_results"])
            self.assertFalse(protocol["followup"]["parameter_budget_matched_to_v1"])
            saved = json.loads((shared.OUT / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, protocol)

    def test_workers_activate_before_delegating_and_eval_checks_freeze(self):
        events = []
        with patch.object(shared, "activate", side_effect=lambda: events.append("activate")), \
                patch.object(shared, "check_protocol", side_effect=lambda: events.append("protocol")), \
                patch.object(shared, "check_freeze", side_effect=lambda: events.append("freeze")), \
                patch.object(shared.base_launch, "train_one", side_effect=lambda job: events.append(("train", job))), \
                patch.object(shared.base_launch, "eval_one", side_effect=lambda job: events.append(("eval", job))):
            shared.train_one((0, "delta"))
            shared.eval_one((1, "add"))
        self.assertEqual(events, ["activate", "protocol", ("train", (0, "delta")),
                                  "activate", "freeze", ("eval", (1, "add"))])

    def test_freeze_requires_all_nine_endpoints_and_verifies_every_digest(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(shared, "OUT", Path(directory)), \
                patch.object(shared, "check_protocol"), \
                patch.object(shared.base_train, "sha", return_value="checkpoint-digest") as digest:
            checkpoints = {str(Path(f"seed_{seed:03d}") / arm / "final.pt"): "checkpoint-digest"
                           for seed in range(3) for arm in shared.base_train.ARMS}
            frozen = dict(all_training_complete_before_scoring=True, paired_train_samples=True,
                          paired_add_delta_initial_parameters=True, checkpoints=checkpoints)
            shared.base_train.write_json(shared.OUT / "training-freeze.json", frozen)
            self.assertEqual(shared.check_freeze(), frozen)
            self.assertEqual(digest.call_count, 9)
            frozen["checkpoints"].pop(next(iter(checkpoints)))
            shared.base_train.write_json(shared.OUT / "training-freeze.json", frozen)
            with self.assertRaises(AssertionError):
                shared.check_freeze()

    def test_scheduler_freezes_only_after_all_training_futures_succeed(self):
        events = []

        class Pool:
            def __init__(self, *, max_workers, initializer):
                self.initializer = initializer
                events.append(("pool", max_workers))
            def __enter__(self):
                self.initializer()
                return self
            def __exit__(self, *args):
                events.append("pool_exit")
            def submit(self, worker, job):
                events.append(("submit", worker.__name__, job))
                return Mock(result=lambda: events.append(("done", job)))

        with patch.object(shared, "activate", side_effect=lambda: events.append("activate")), \
                patch.object(shared, "check_protocol", side_effect=lambda: events.append("protocol")), \
                patch.object(shared, "check_freeze", side_effect=lambda: events.append("checked_freeze")), \
                patch.object(shared, "ProcessPoolExecutor", Pool), \
                patch.object(shared, "as_completed", side_effect=lambda values: values), \
                patch.object(shared.base_launch, "freeze", side_effect=lambda: events.append("frozen")), \
                redirect_stdout(io.StringIO()):
            shared.launch("train", workers=2)
        self.assertEqual(sum(isinstance(item, tuple) and item[0] == "done" for item in events), 9)
        self.assertLess(events.index("pool_exit"), events.index("frozen"))
        self.assertEqual(events[-2:], ["frozen", "checked_freeze"])


if __name__ == "__main__":
    unittest.main()
