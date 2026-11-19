"""Differentiable binding, native executor, and explicit-state invariants."""
import inspect
import unittest

import torch
from torch.nn import functional as F

from experiments.learn_remember.gru_train import GRUAgent
from experiments.learned_binding.model import (
    NativeMemoryPolicy, detach_state, soft_translate,
)
from experiments.lifelong.world import FAMILY_MAPPINGS


def public_observations(ticks, batch):
    """Synthetic public features; deliberately no world/target object exists."""
    obs = torch.randn(ticks, batch, 2, 28) * .1
    obs[..., :3] = 0
    obs[..., 5:9] = F.one_hot(torch.arange(ticks * batch * 2).reshape(
        ticks, batch, 2).remainder(4), 4).float()
    obs[..., 24:28] = F.one_hot(torch.arange(ticks * batch * 2).reshape(
        ticks, batch, 2).remainder(4), 4).float()
    obs[..., 0] = (torch.arange(ticks)[:, None, None] % 3 == 0).float()
    obs[..., 1] = (torch.arange(ticks)[:, None, None] % 3 == 1).float()
    return obs.reshape(ticks, batch, 56)


class LearnedBindingModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1837)

    def test_soft_translation_exact_inverse_mapping_and_gradient(self):
        obs = public_observations(3, 4)
        scores = torch.randn(3, 4, 2, 4, requires_grad=True)
        probabilities = scores.softmax(-1)
        result = soft_translate(obs, probabilities).reshape(3, 4, 2, 28)
        original = obs.reshape(3, 4, 2, 28)
        mappings = torch.tensor(FAMILY_MAPPINGS)
        for tick in range(3):
            for batch in range(4):
                for job in range(2):
                    context = int(original[tick, batch, job, 24:28].argmax())
                    expected = (probabilities[tick, batch, job, mappings[context]]
                                if tick < 2 else original[tick, batch, job, 5:9])
                    torch.testing.assert_close(result[tick, batch, job, 5:9], expected)
        self.assertTrue(torch.equal(result[..., :5], original[..., :5]))
        self.assertTrue(torch.equal(result[..., 9:], original[..., 9:]))
        (result[..., 5:9] * torch.arange(4.)).sum().backward()
        self.assertGreater(float(scores.grad[:2].abs().sum()), 0.)
        self.assertEqual(float(scores.grad[2].abs().sum()), 0.)
        # One-hot targets recover the exact raw symbol for every context/value.
        for context in range(4):
            for target in range(4):
                cue = torch.zeros(1, 2, 28)
                cue[..., 0], cue[..., 24 + context] = 1., 1.
                probs = F.one_hot(torch.full((1, 2), target), 4).float()
                actual = soft_translate(cue.reshape(1, 56), probs).reshape(1, 2, 28)
                raw = list(FAMILY_MAPPINGS[context]).index(target)
                self.assertTrue(torch.equal(actual[..., 5:9],
                    F.one_hot(torch.full((1, 2), raw), 4).float()))

    def test_step_sequence_and_chunk_equivalence_fp32(self):
        model = NativeMemoryPolicy().eval()
        obs = public_observations(17, 3)
        state = model.initial_state(3)
        for name in state:
            state[name].normal_(std=.1)
        with torch.no_grad():
            for write, read in ((True, True), (False, True), (True, False), (False, False)):
                expected, sequence_state, diagnostics = model(obs, state,
                    write_enabled=write, read_enabled=read, return_diagnostics=True)
                current = state
                outputs = []
                step_reads = []
                for frame in obs:
                    logits, current, diag = model.step(frame, current,
                        write_enabled=write, read_enabled=read, return_diagnostics=True)
                    outputs.append(logits)
                    step_reads.append(diag["read_probs"])
                torch.testing.assert_close(torch.stack(outputs), expected,
                                           rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(torch.stack(step_reads), diagnostics["read_probs"],
                                           rtol=2e-5, atol=2e-6)
                for name in state:
                    torch.testing.assert_close(current[name], sequence_state[name],
                                               rtol=2e-5, atol=2e-6)
                first, middle = model(obs[:7], state, write_enabled=write, read_enabled=read)
                second, chunk_state = model(obs[7:], middle,
                                           write_enabled=write, read_enabled=read)
                torch.testing.assert_close(torch.cat((first, second)), expected,
                                           rtol=2e-5, atol=2e-6)
                for name in state:
                    torch.testing.assert_close(chunk_state[name], sequence_state[name],
                                               rtol=2e-5, atol=2e-6)

    def test_future_action_loss_reaches_early_writes_and_frozen_executor_input(self):
        model = NativeMemoryPolicy()
        obs = public_observations(6, 2)
        # No earlier command reaches the executor: the only route from these
        # early write activations to future CE is persistent explicit memory.
        obs = obs.reshape(6, 2, 2, 28)
        obs[:-1, ..., :3] = 0.
        obs[-1, ..., 0] = 1.
        obs = obs.reshape(6, 2, 56)
        before = {name: value.clone() for name, value in model.executor.state_dict().items()}
        logits, _, diagnostics = model(obs, model.initial_state(2), return_diagnostics=True)
        for name in ("q", "k", "value", "gate", "read_probs", "mapped_obs"):
            diagnostics[name].retain_grad()
        F.cross_entropy(logits[-1].reshape(-1, 11), torch.tensor([0, 3, 6, 9])).backward()
        for name in ("k", "value", "gate"):
            gradient = diagnostics[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            self.assertGreater(float(gradient[0].abs().sum()), 0., name)
        for name in ("q", "read_probs", "mapped_obs"):
            self.assertGreater(float(diagnostics[name].grad[-1].abs().sum()), 0., name)
        for parameter in model.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in model.executor.parameters()))
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad))
        optimizer.step()
        for name, value in model.executor.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_delta_update_order_matches_explicit_algebra_and_no_receipt_mask(self):
        model = NativeMemoryPolicy().double()
        obs = torch.zeros(1, 2, 56, dtype=torch.float64)
        state = model.initial_state(2)
        state["memory"].normal_(std=.1)
        original = {name: value.clone() for name, value in state.items()}
        _, updated, diag = model(obs, state, return_diagnostics=True)
        matrix = state["memory"].reshape(2, 16, 4)
        for job in range(2):
            key = diag["k"][0, :, job]
            predicted = torch.einsum("bk,bkv->bv", key, matrix)
            matrix = matrix + (diag["gate"][0, :, job, :, None]
                * key[..., None] * (diag["value"][0, :, job] - predicted)[:, None])
        torch.testing.assert_close(updated["memory"], matrix.reshape(2, 64),
                                   rtol=1e-12, atol=1e-12)
        self.assertFalse(torch.equal(updated["memory"], state["memory"]))
        expected_read = (model.read_scale * torch.einsum(
            "bjk,bkv->bjv", diag["q"][0], matrix)).softmax(-1)
        torch.testing.assert_close(diag["read_probs"][0], expected_read)
        for name in state:
            self.assertTrue(torch.equal(state[name], original[name]))

    def test_disabled_writes_are_bitwise_unchanged_and_disabled_reads_uniform(self):
        model = NativeMemoryPolicy()
        obs = public_observations(9, 4)
        state = model.initial_state(4)
        state["memory"].normal_()
        for read in (True, False):
            _, updated, diag = model(obs, state, write_enabled=False,
                                     read_enabled=read, return_diagnostics=True)
            self.assertIs(updated["memory"], state["memory"])
            self.assertTrue(torch.equal(updated["memory"], state["memory"]))
            if not read:
                self.assertTrue(torch.equal(diag["read_probs"],
                                            torch.full((9, 4, 2, 4), .25)))
                torch.testing.assert_close(diag["mapped_obs"],
                    soft_translate(obs, torch.full((9, 4, 2, 4), .25)))

    def test_parameter_counts_state_budget_config_and_public_only_signature(self):
        executor = GRUAgent()
        before = {name: value.clone() for name, value in executor.state_dict().items()}
        model = NativeMemoryPolicy(executor=executor)
        self.assertIs(model.executor, executor)
        for name, value in executor.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        state = model.initial_state(5)
        self.assertEqual({name: tuple(value.shape) for name, value in state.items()},
                         dict(h=(5, 192), z=(5, 2, 32), memory=(5, 64)))
        self.assertEqual(sum(value.numel() for value in state.values()), 5 * 320)
        self.assertTrue(all(int(value.count_nonzero()) == 0 for value in state.values()))
        report = model.parameter_report()
        self.assertEqual(report["executor_parameters"], 148246)
        self.assertEqual(report["encoder_parameters"], 5952)
        self.assertEqual(report["projection_parameters"], 2257)
        self.assertEqual(report["trainable_parameters"], 8210)
        self.assertEqual(report["total_parameters"], 156456)
        self.assertEqual(report["state_scalars"], 320)
        self.assertEqual(report["memory_state_scalars"], 64)
        self.assertEqual(sum(report["parameters_by_name"].values()), 156456)
        self.assertEqual(report["head_input_features"], 60)
        self.assertAlmostEqual(report["read_scale"], 5., places=6)
        self.assertTrue(torch.equal(model.gate.bias, torch.tensor([-5.])))
        self.assertEqual(tuple(model.original_mappings.shape), (4, 4))
        self.assertNotIn("original_mappings", dict(model.named_parameters()))
        restored = NativeMemoryPolicy(**model.config())
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(restored.config(), model.config())
        expected_parameters = ["self", "obs", "state", "write_enabled",
                               "read_enabled", "return_diagnostics"]
        for method in (NativeMemoryPolicy.forward, NativeMemoryPolicy.step):
            self.assertEqual(list(inspect.signature(method).parameters), expected_parameters)
        with self.assertRaises(TypeError):
            model(public_observations(2, 5), state, write_enabled=torch.ones(5).bool())
        with self.assertRaises(TypeError):
            model(public_observations(2, 5), state, private_target=torch.zeros(5))
        with self.assertRaises(ValueError):
            NativeMemoryPolicy(executor_config=dict(width=32))

    def test_learned_retention_initialization_and_direct_current_public_path(self):
        model = NativeMemoryPolicy()
        for bias in (model.encoder.bias_ih_l0, model.encoder.bias_hh_l0):
            self.assertTrue(bias.requires_grad)
            self.assertTrue(torch.equal(bias[32:64], torch.full((32,), 2.)))
        obs = public_observations(3, 2)
        _, _, diag = model(obs, model.initial_state(2), return_diagnostics=True)
        joined = torch.cat((diag["z"], obs.reshape(3, 2, 2, 28)), -1)
        for head in (model.query, model.key, model.value, model.gate):
            self.assertEqual(head.in_features, 60)
        torch.testing.assert_close(diag["q"], F.normalize(model.query(joined), dim=-1))
        torch.testing.assert_close(diag["k"], F.normalize(model.key(joined), dim=-1))
        torch.testing.assert_close(diag["value"], model.value(joined).softmax(-1))
        torch.testing.assert_close(diag["gate"], model.gate(joined).sigmoid())
        (diag["gate"].sum() + (diag["value"] * torch.arange(4.)).sum()).backward()
        for bias in (model.encoder.bias_ih_l0, model.encoder.bias_hh_l0):
            self.assertIsNotNone(bias.grad)
            self.assertGreater(float(bias.grad[32:64].abs().sum()), 0.)
        # Sever observation-to-encoder dependence in this test only.  Head
        # gradients on current public receipt features must still be present.
        with torch.no_grad():
            model.encoder.weight_ih_l0.zero_()
        receipt_obs = public_observations(1, 2).requires_grad_()
        _, state, direct = model(receipt_obs, model.initial_state(2),
                                 return_diagnostics=True)
        objective = direct["gate"].sum() + (
            direct["value"] * torch.arange(4.)).sum()
        gradients = torch.autograd.grad(objective, receipt_obs)[0].reshape(1, 2, 2, 28)
        self.assertGreater(float(gradients[..., 11:24].abs().sum()), 0.)
        self.assertEqual(set(state), {"h", "z", "memory"})
        self.assertEqual(sum(value.numel() for value in state.values()), 2 * 320)

    def test_detach_state_preserves_values_and_removes_history(self):
        model = NativeMemoryPolicy()
        _, state = model(public_observations(3, 2), model.initial_state(2))
        detached = detach_state(state)
        self.assertEqual(set(detached), {"h", "z", "memory"})
        for name in state:
            self.assertTrue(torch.equal(detached[name], state[name]))
            self.assertFalse(detached[name].requires_grad)
            self.assertIsNone(detached[name].grad_fn)


if __name__ == "__main__":
    unittest.main()
