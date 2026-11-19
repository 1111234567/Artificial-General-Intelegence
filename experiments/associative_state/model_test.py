"""Algebra, state isolation, and differentiability of association controls."""
import unittest

import torch
from torch.nn import functional as F

from experiments.associative_state.model import AssociationMemory, KINDS


def orthogonal_model(kind):
    model = AssociationMemory(kind=kind).double()
    with torch.no_grad():
        model.key_embedding.weight.copy_(torch.eye(16, dtype=torch.float64)[:8])
    return model


class AssociationMemoryTest(unittest.TestCase):
    def test_default_state_and_exact_parameter_counts(self):
        for kind in KINDS:
            torch.manual_seed(31)
            model = AssociationMemory(kind=kind)
            state = model.initial_state(3)
            self.assertEqual(tuple(state.shape), (3, 64))
            self.assertEqual(state.dtype, model.key_embedding.weight.dtype)
            self.assertEqual(int(state.count_nonzero()), 0)
            self.assertEqual(tuple(model.read(state, torch.tensor([0, 1, 7])).shape),
                             (3, 4))
            self.assertTrue(torch.equal(model.read(state, torch.tensor([0, 1, 7])),
                                        torch.zeros(3, 4)))
            report = model.parameter_report()
            expected = 16641 if kind == "recurrent" else 130
            self.assertEqual(report["total_parameters"], expected)
            self.assertEqual(report["trainable_parameters"], expected)
            self.assertEqual(report["state_scalars"], 64)
            self.assertEqual(sum(report["parameters_by_name"].values()), expected)
            self.assertAlmostEqual(report["read_scale"], 5., places=6)
            if kind != "recurrent":
                self.assertEqual(report["eta"], .5)
            else:
                self.assertIsNone(report["eta"])
            restored = AssociationMemory(**model.config())
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(restored.config(), model.config())

    def test_add_and_delta_have_identical_seeded_parameters_and_names(self):
        torch.manual_seed(170)
        additive = AssociationMemory("add")
        torch.manual_seed(170)
        delta = AssociationMemory("delta")
        self.assertEqual(list(dict(additive.named_parameters())),
                         ["eta_logit", "read_scale_raw", "key_embedding.weight"])
        self.assertEqual(list(additive.state_dict()), list(delta.state_dict()))
        for name, value in additive.state_dict().items():
            self.assertTrue(torch.equal(value, delta.state_dict()[name]))
        delta.load_state_dict(additive.state_dict(), strict=True)

    def test_delta_update_matches_analytic_residual_for_arbitrary_matrix(self):
        torch.manual_seed(171)
        model = AssociationMemory("delta").double()
        with torch.no_grad():
            model.eta_logit.fill_(torch.logit(torch.tensor(.3)).item())
        before = torch.randn(3, 64, dtype=torch.float64)
        original = before.clone()
        keys, values = torch.tensor([0, 3, 7]), torch.tensor([2, 1, 0])
        matrix = before.reshape(3, 16, 4)
        key = F.normalize(model.key_embedding(keys), dim=-1)
        value = F.one_hot(values, 4).double()
        prediction = torch.einsum("bkv,bk->bv", matrix, key)
        expected = matrix + model.eta * torch.einsum("bk,bv->bkv", key,
                                                     value - prediction)
        actual = model.write(before, keys, values)
        torch.testing.assert_close(actual.reshape(3, 16, 4), expected,
                                   rtol=1e-13, atol=1e-13)
        self.assertTrue(torch.equal(before, original))
        expected_read = model.read_scale * torch.einsum("bkv,bk->bv", expected, key)
        torch.testing.assert_close(model.read(actual, keys), expected_read)

    def test_orthogonal_key_write_preserves_unrelated_read(self):
        for kind in ("add", "delta"):
            model = orthogonal_model(kind)
            state = model.write(model.initial_state(1), torch.tensor([0]),
                                torch.tensor([2]))
            before = model.read(state, torch.tensor([0]))
            after = model.write(state, torch.tensor([1]), torch.tensor([3]))
            self.assertTrue(torch.equal(model.read(after, torch.tensor([0])), before))
            torch.testing.assert_close(model.read(after, torch.tensor([1])),
                                       torch.tensor([[0., 0., 0., 2.5]],
                                                    dtype=torch.float64))

    def test_repeats_accumulate_for_add_and_saturate_for_delta(self):
        for kind, expected in (("add", 5.), ("delta", 3.75)):
            model = orthogonal_model(kind)
            state = model.initial_state(1)
            for _ in range(2):
                state = model.write(state, torch.tensor([2]), torch.tensor([1]))
            torch.testing.assert_close(model.read(state, torch.tensor([2])),
                                       torch.tensor([[0., expected, 0., 0.]],
                                                    dtype=torch.float64))

    def test_override_residual_reduces_old_value_only_for_delta(self):
        for kind, old_value in (("add", 2.5), ("delta", 1.25)):
            model = orthogonal_model(kind)
            state = model.write(model.initial_state(1), torch.tensor([4]),
                                torch.tensor([0]))
            state = model.write(state, torch.tensor([4]), torch.tensor([3]))
            torch.testing.assert_close(model.read(state, torch.tensor([4])),
                                       torch.tensor([[old_value, 0., 0., 2.5]],
                                                    dtype=torch.float64))

    def test_mask_preserves_inactive_lanes_and_ignores_sentinel_indices(self):
        torch.manual_seed(172)
        for kind in KINDS:
            model = AssociationMemory(kind).double()
            state = torch.randn(3, 64, dtype=torch.float64)
            original = state.clone()
            result = model.write(state, torch.tensor([0, -1, 7]),
                                 torch.tensor([1, -1, 3]),
                                 torch.tensor([True, False, True]))
            self.assertTrue(torch.equal(result[1], state[1]))
            self.assertTrue(torch.equal(state, original))
            expected = model.write(state[[0, 2]], torch.tensor([0, 7]),
                                   torch.tensor([1, 3]))
            torch.testing.assert_close(result[[0, 2]], expected)
            no_op = model.write(state, torch.full((3,), -1),
                                torch.full((3,), -1), torch.zeros(3, dtype=torch.bool))
            self.assertIs(no_op, state)

    def test_finite_gradients_reach_first_write_and_embedding_through_second_write(self):
        for kind in KINDS:
            torch.manual_seed(173)
            model = AssociationMemory(kind).double()
            initial = model.initial_state(2).requires_grad_()
            first = model.write(initial, torch.tensor([0, 3]), torch.tensor([1, 2]))
            first.retain_grad()
            second = model.write(first, torch.tensor([1, 4]), torch.tensor([2, 0]))
            logits = model.read(second, torch.tensor([2, 5]))
            F.cross_entropy(logits, torch.tensor([1, 2])).backward()
            for gradient in (initial.grad, first.grad, model.key_embedding.weight.grad):
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.abs().sum()), 0.)
            # Keys 0 and 3 occur only in the first write.  Keys 1 and 4 occur
            # only in the second, and 2 and 5 only in the query.  Each path
            # must reach its own embedding rows without a query shortcut.
            for rows in ([0, 3], [1, 4], [2, 5]):
                self.assertGreater(float(model.key_embedding.weight.grad[rows].abs().sum()), 0.)
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_recurrent_write_is_exact_gru_on_key_and_public_one_hot(self):
        torch.manual_seed(174)
        model = AssociationMemory("recurrent").double()
        state = torch.randn(2, 64, dtype=torch.float64)
        keys, values = torch.tensor([1, 6]), torch.tensor([3, 0])
        inputs = torch.cat((F.normalize(model.key_embedding(keys), dim=-1),
                            F.one_hot(values, 4).double()), dim=-1)
        torch.testing.assert_close(model.write(state, keys, values),
                                   model.core(inputs, state))

    def test_input_validation_and_nondefault_state_size(self):
        model = AssociationMemory("delta", key_dim=3, n_keys=2, n_values=2).double()
        self.assertEqual(tuple(model.initial_state(2).shape), (2, 6))
        self.assertEqual(model.parameter_report()["state_scalars"], 6)
        with self.assertRaises(ValueError):
            AssociationMemory("oracle")
        with self.assertRaises(ValueError):
            AssociationMemory(key_dim=0)
        with self.assertRaises(ValueError):
            model.read(torch.zeros(2, 64, dtype=torch.float64), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            model.write(model.initial_state(2), torch.tensor([0, 1]), torch.tensor([0, 1]),
                        torch.tensor([1., 0.]))


if __name__ == "__main__":
    unittest.main()
