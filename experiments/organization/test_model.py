"""Organization, matched birth, temporal derivatives and serialization checks."""
import io
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.continuity.model import AgentModel
from experiments.organization.model import FIELD_MODES, OrganizationModel


def block_diagonal_birth(model, fan_in=2):
    """A private toy equivalent of the runner's shared newborn mask."""
    generator = torch.Generator().manual_seed(318)
    width, half = model.width, model.width // 2
    mask = torch.zeros(width, width, dtype=torch.bool)
    for row in range(width):
        start = 0 if row < half else half
        mask[row, start + torch.randperm(half, generator=generator)[:fan_in]] = True
    with torch.no_grad():
        model.recurrent.mul_(mask).mul_(math.sqrt(width / fan_in))
    return mask


class _Step(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observation, state, dt):
        logits, hidden = self.model.step(observation, state, dt)
        return torch.cat((logits.flatten(1), hidden), dim=1)


class OrganizationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dimensions_parameter_budget_and_config(self):
        for mode in FIELD_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(8)
                model = OrganizationModel(field_mode=mode)
                report = model.parameter_report()
                self.assertEqual(report["state_scalars"], 128)
                self.assertEqual(report["total_parameters"], 36396)
                self.assertEqual(report["trainable_parameters"], 36396)
                self.assertEqual(report["recurrent_parameters"], 16384)
                self.assertEqual(report["recurrent_mac_per_field"], 16384)
                self.assertEqual(report["recurrent_matrix_products_per_field"], 4)
                self.assertEqual(report["tanh_scalars_per_field"], 128 if mode == "plain" else 256)
                self.assertEqual(report["bank_width"], 64)
                self.assertEqual(OrganizationModel(**model.config()).config(), model.config())
                self.assertEqual(set(dict(model.named_parameters())),
                                 {"recurrent", "encoder.weight", "head.weight", "head.bias"})
                self.assertEqual(tuple(model.recurrent.shape), (128, 128))
        for kwargs in ({"width": 7}, {"width": 0}, {"width": 8.0}, {"width": True},
                       {"field_mode": "unknown"}, {"kind": "gru"}, {"kind": "reservoir"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OrganizationModel(**kwargs)

    def test_plain_field_and_solver_equal_original_dense_dynamics(self):
        torch.manual_seed(5)
        model = OrganizationModel(input_size=3, n_jobs=1, n_actions=2, width=8).double()
        old = AgentModel(**{key: value for key, value in model.config().items() if key != "field_mode"}).double()
        old.load_state_dict(model.state_dict())
        hidden, observation = torch.randn(2, 8, dtype=torch.double), torch.randn(2, 3, dtype=torch.double)
        torch.testing.assert_close(model.vector_field(hidden), -hidden + F.linear(hidden.tanh(), old.recurrent),
                                   rtol=1e-13, atol=1e-13)
        for elapsed in (0., .25, .375, .7):
            with self.subTest(elapsed=elapsed):
                expected, old_h = old.step(observation, hidden, elapsed)
                actual, new_h = model.step(observation, hidden, elapsed)
                torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(new_h, old_h, rtol=1e-13, atol=1e-13)

    def test_all_modes_have_exact_same_multistep_birth_function(self):
        models = []
        for mode in FIELD_MODES:
            torch.manual_seed(19)
            model = OrganizationModel(input_size=3, n_jobs=1, n_actions=2, width=8,
                                      field_mode=mode).double()
            mask = block_diagonal_birth(model)
            self.assertEqual(int(mask.sum()), 16)
            self.assertEqual(int(model.recurrent[:4, 4:].count_nonzero()), 0)
            self.assertEqual(int(model.recurrent[4:, :4].count_nonzero()), 0)
            models.append(model)
        for model in models[1:]:
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))
        observation = torch.linspace(-.4, .6, 6, dtype=torch.double).reshape(2, 3)
        states = [torch.linspace(-.7, .5, 16, dtype=torch.double).reshape(2, 8) for _ in models]
        for elapsed in (0., .25, .375, .125):
            outputs = [model.step(observation, hidden, elapsed) for model, hidden in zip(models, states)]
            for logits, hidden in outputs[1:]:
                self.assertTrue(torch.equal(logits, outputs[0][0]))
                self.assertTrue(torch.equal(hidden, outputs[0][1]))
            states = [hidden for _, hidden in outputs]

    def test_zero_cross_blocks_receive_gradients_at_birth(self):
        for mode in FIELD_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(11)
                model = OrganizationModel(input_size=3, n_jobs=1, n_actions=2, width=8,
                                          field_mode=mode).double()
                block_diagonal_birth(model)
                hidden = torch.linspace(-.7, .9, 16, dtype=torch.double).reshape(2, 8)
                model.vector_field(hidden).square().mean().backward()
                self.assertGreater(float(model.recurrent.grad[:4, 4:].norm()), 0.)
                self.assertGreater(float(model.recurrent.grad[4:, :4].norm()), 0.)

    def test_coupled_has_cross_bank_mixed_partial_but_parallel_is_separable(self):
        hidden = torch.tensor([.2, -.3, .7, -.4], dtype=torch.double)
        for mode in FIELD_MODES:
            with self.subTest(mode=mode):
                model = OrganizationModel(input_size=2, n_jobs=1, n_actions=2, width=4,
                                          field_mode=mode).double()
                with torch.no_grad():
                    model.recurrent.copy_(torch.tensor(
                        [[.7, -.1, .3, .4], [.2, .5, -.5, .2],
                         [.6, -.2, -.1, .8], [.1, .3, -.4, .5]], dtype=torch.double))
                hessian = torch.autograd.functional.hessian(lambda value: model.vector_field(value)[0], hidden)
                if mode == "coupled":
                    self.assertGreater(float(hessian[0, 2].abs()), .01)
                    self.assertGreater(float(hessian[:2, 2:].abs().sum()), .01)
                else:
                    self.assertTrue(torch.equal(hessian[:2, 2:], torch.zeros(2, 2, dtype=torch.double)))

    def test_heun_uses_selected_field_and_pulse_is_injected_once(self):
        for mode in FIELD_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(17)
                model = OrganizationModel(input_size=3, n_jobs=1, n_actions=2, width=8,
                                          field_mode=mode).double()
                hidden = torch.linspace(-.3, .6, 16, dtype=torch.double).reshape(2, 8)
                expected = hidden
                delta = .375 / 2
                for _ in range(2):
                    first = model.vector_field(expected)
                    second = model.vector_field(expected + delta * first)
                    expected = expected + .5 * delta * (first + second)
                self.assertTrue(torch.equal(model.flow(hidden, .375), expected))
                self.assertTrue(torch.equal(model.flow(hidden, 0.), hidden))
                observation = torch.linspace(-.2, .4, 6, dtype=torch.double).reshape(2, 3)
                for limit in (.25, .0625):
                    _, actual = model.step(observation, model.initial_state(2), .5, limit)
                    self.assertTrue(torch.equal(actual, model.encoder(observation)))
                for elapsed, limit in ((-.1, .25), (float("nan"), .25), (.1, 0.), (.1, float("inf"))):
                    with self.assertRaises(ValueError):
                        model.flow(hidden, elapsed, limit)

    def test_heun_pulse_derivatives_include_observation_hidden_and_all_parameters(self):
        for mode in FIELD_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(31)
                model = OrganizationModel(input_size=2, n_jobs=1, n_actions=2, width=4,
                                          field_mode=mode).double()
                wrapped = _Step(model)
                named = tuple(wrapped.named_parameters())
                keys = [key for key, _ in named]
                values = [value.detach().clone().requires_grad_() for _, value in named]
                observation = torch.tensor([[.2, -.4]], dtype=torch.double, requires_grad=True)
                hidden = torch.tensor([[.13, -.27, .41, -.19]], dtype=torch.double, requires_grad=True)

                def evaluate(observation, hidden, *parameters):
                    return torch.func.functional_call(wrapped, dict(zip(keys, parameters)),
                                                      (observation, hidden, .375))

                self.assertTrue(torch.autograd.gradcheck(evaluate, (observation, hidden, *values),
                                                        eps=1e-6, atol=2e-5, rtol=2e-4))

    def _serialization_roundtrip(self, device):
        for mode in FIELD_MODES:
            with self.subTest(mode=mode, device=device):
                torch.manual_seed(29)
                model = OrganizationModel(input_size=3, n_jobs=1, n_actions=2, width=8,
                                          field_mode=mode).double().to(device)
                hidden = torch.linspace(-.3, .4, 16, dtype=torch.double, device=device).reshape(2, 8)
                observation = torch.linspace(-.2, .5, 6, dtype=torch.double, device=device).reshape(2, 3)
                expected = model.step(observation, hidden, .375)
                buffer = io.BytesIO()
                torch.save({"model_config": model.config(), "model": model.state_dict(),
                            "hidden": hidden}, buffer)
                buffer.seek(0)
                checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
                restored = OrganizationModel(**checkpoint["model_config"]).double()
                restored.load_state_dict(checkpoint["model"])
                self.assertEqual(restored.config(), model.config())
                actual = restored.step(observation.cpu(), checkpoint["hidden"], .375)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference.cpu(), rtol=1e-12, atol=1e-12)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value.cpu(), restored.state_dict()[key]))

    def test_cpu_checkpoint_and_model_config_roundtrip(self):
        self._serialization_roundtrip("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_checkpoint_loads_on_cpu_with_model_config_roundtrip(self):
        self._serialization_roundtrip("cuda")


if __name__ == "__main__":
    unittest.main()
