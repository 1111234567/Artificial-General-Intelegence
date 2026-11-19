"""Small CPU derivative and exact-graft checks; no architecture study."""
from copy import deepcopy
import unittest

import torch
from torch import nn

from experiments.continuity.model import AgentModel
from experiments.continuity.run import rng_state, state_digest
from experiments.loops.model import ARMS, LoopModel


class _Step(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observation, state, dt):
        logits, hidden = self.model.step(observation, state, dt)
        return torch.cat((logits.flatten(1), hidden), dim=1)


class LoopModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_graft_preserves_parameters_rng_and_exact_multistep_output(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                old = AgentModel(input_size=3, n_jobs=1, n_actions=3, width=5).double()
                reference = deepcopy(old)
                parameters = dict(old.named_parameters())
                before = state_digest(rng_state())
                model = LoopModel.attach(old, arm)
                self.assertEqual(before, state_digest(rng_state()))
                for name, value in parameters.items():
                    self.assertIs(value, dict(model.named_parameters())[name])
                h = torch.linspace(-.8, .9, 10, dtype=torch.float64).reshape(2, 5)
                old_h = h.clone()
                observation = torch.linspace(-.2, .4, 6, dtype=torch.float64).reshape(2, 3)
                for dt in (0., .25, .7, .125):
                    actual, h = model.step(observation, h, dt)
                    expected, old_h = reference.step(observation, old_h, dt)
                    self.assertTrue(torch.equal(h, old_h))
                    self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(model.parameter_report()["branch_parameters"], 0 if arm == "baseline" else 25)

    def test_exact_heun_and_pulse_gradients_include_all_parameters(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = LoopModel.attach(AgentModel(input_size=2, n_jobs=1, n_actions=2, width=3).double(), arm)
                if arm != "baseline":
                    with torch.no_grad():
                        model.branch.copy_(torch.linspace(-.25, .3, 9).reshape(3, 3))
                wrapped = _Step(model)
                named = tuple(wrapped.named_parameters())
                keys = [key for key, _ in named]
                values = [value.detach().clone().requires_grad_() for _, value in named]
                observation = torch.tensor([[.2, -.4]], dtype=torch.float64, requires_grad=True)
                hidden = torch.tensor([[.13, -.27, .41]], dtype=torch.float64, requires_grad=True)

                def evaluate(observation, hidden, *parameters):
                    return torch.func.functional_call(wrapped, dict(zip(keys, parameters)), (observation, hidden, .4))

                self.assertTrue(torch.autograd.gradcheck(evaluate, (observation, hidden, *values),
                    eps=1e-6, atol=2e-5, rtol=2e-4))

    def test_interaction_depends_on_recipient_and_source_state(self):
        source = LoopModel.attach(AgentModel(input_size=2, n_jobs=1, n_actions=2, width=2).double(), "source_square")
        interaction = LoopModel.attach(AgentModel(input_size=2, n_jobs=1, n_actions=2, width=2).double(), "interaction")
        for model in (source, interaction):
            with torch.no_grad():
                model.branch.zero_()
                model.branch[0, 1] = 2.
        a = torch.tensor([[.2, .7]], dtype=torch.float64)
        b = torch.tensor([[-.2, .7]], dtype=torch.float64)
        self.assertTrue(torch.equal(source.branch_forcing(a), source.branch_forcing(b)))
        self.assertTrue(torch.equal(interaction.branch_forcing(a), -interaction.branch_forcing(b)))
        self.assertGreater(float(source.branch_forcing(a)[0, 0].detach()), 0.)
        with self.assertRaises(ValueError):
            LoopModel.attach(interaction, "baseline")

    def test_instantaneous_interaction_has_cross_neuron_mixed_partial(self):
        for arm in ("source_square", "interaction"):
            with self.subTest(arm=arm):
                model = LoopModel.attach(AgentModel(input_size=2, n_jobs=1, n_actions=2, width=2).double(), arm)
                with torch.no_grad():
                    model.branch.zero_()
                    model.branch[0, 1] = 2.
                h = torch.tensor([.2, .7], dtype=torch.float64, requires_grad=True)
                hessian = torch.autograd.functional.hessian(lambda x: model.vector_field(x[None, :])[0, 0], h)
                expected = 0. if arm == "source_square" else 2. * (1. - h[0].tanh().square()) * (1. - h[1].tanh().square())
                torch.testing.assert_close(hessian[0, 1], torch.as_tensor(expected).to(h))
                if arm == "interaction":
                    self.assertGreater(float(hessian[0, 1]), 1.)

    def test_pair_feature_metrics_match_explicit_actor_outer_products(self):
        model = LoopModel.attach(AgentModel(input_size=2, n_jobs=1, n_actions=2, width=3).double(), "interaction")
        h = torch.tensor([[.2, -.7, .4], [-.3, .8, -.6]], dtype=torch.float64)
        rates = h.tanh()
        products = rates[:, :, None] * rates[:, None, :]
        mask = ~torch.eye(3, dtype=torch.bool)
        metrics = model.branch_metrics(h)
        self.assertAlmostEqual(metrics["recipient_source_product_mean"], float(products.mean()), places=14)
        self.assertAlmostEqual(metrics["recipient_source_product_rms"], float(products.square().mean().sqrt()), places=14)
        self.assertAlmostEqual(metrics["off_diagonal_recipient_source_product_mean"], float(products[:, mask].mean()), places=14)
        self.assertAlmostEqual(metrics["off_diagonal_recipient_source_product_rms"], float(products[:, mask].square().mean().sqrt()), places=14)


if __name__ == "__main__":
    unittest.main()
