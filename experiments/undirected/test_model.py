"""Graph sharing, potential-graph gradients, transforms, and birth controls."""
import io
import math
import random
import unittest

import torch
from torch.nn import functional as F

from experiments.undirected.model import (
    MODES, PortGraphModel, UndirectedModel, connected_regular_mask, contextual_input_scale,
)


def _model(mode="undirected", width=6, *, dtype=torch.double):
    torch.manual_seed(181)
    result = UndirectedModel(input_size=3, n_jobs=1, n_actions=2,
                             width=width, mode=mode).to(dtype=dtype)
    result.initialize_birth(mask_seed=617, weight_seed=883, degree=2)
    return result


def _energy(model, hidden):
    output = hidden.tanh()
    logcosh = torch.logaddexp(hidden, -hidden) - math.log(2.)
    return -.5 * torch.einsum("...i,ij,...j->...", output, model.recurrent_matrix(), output) + (
        hidden * output - logcosh).sum(-1)


class UndirectedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_dimensions_and_true_independent_parameter_counts(self):
        for mode in MODES:
            model = UndirectedModel(mode=mode)
            report = model.parameter_report()
            expected = 2016 if mode == "undirected" else 4032
            self.assertEqual(report["state_scalars"], 64)
            self.assertEqual(report["recurrent_parameters"], expected)
            self.assertEqual(report["total_parameters"], 10028 + expected)
            self.assertEqual(report["recurrent_mac_per_field"], 64 ** 2)
            self.assertEqual(model.edge_shared.shape, (2016,))
            self.assertEqual(set(dict(model.named_buffers())), {"pair_index", "input_scale"})
            parameters = {"edge_shared", "encoder.weight", "head.weight", "head.bias"}
            if mode == "untied":
                parameters.add("edge_skew")
            self.assertEqual(set(dict(model.named_parameters())), parameters)
            self.assertEqual(UndirectedModel(**model.config()).config(), model.config())
            large = UndirectedModel(width=128, mode=mode).parameter_report()
            self.assertEqual(large["potential_pairs"], 8128)
            self.assertEqual(large["recurrent_parameters"], 8128 if mode == "undirected" else 16256)
            self.assertEqual(large["total_parameters"], 28140 if mode == "undirected" else 36268)

    def test_invalid_configuration_graph_and_shapes(self):
        for kwargs in ({"width": 0}, {"width": True}, {"width": 4.0}, {"input_size": 0},
                       {"mode": "directed"}, {"max_step": 0.}, {"max_step": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                UndirectedModel(**kwargs)
        for args in ((1, 1, 0), (6, 1, 0), (5, 3, 0), (4, 4, 0), (4, 2, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                connected_regular_mask(*args)
        model = _model()
        for args in ((torch.zeros(2, 4), torch.zeros(2, 6), .25),
                     (torch.zeros(2, 3), torch.zeros(3, 6), .25),
                     (torch.zeros(2, 3), torch.zeros(2, 6), -.25)):
            with self.assertRaises(ValueError):
                model(*args)

    def test_connected_regular_reproducible_graph_without_rng_side_effects(self):
        python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
        for width, degree in ((2, 1), (5, 2), (8, 3), (16, 4), (64, 16), (128, 32), (8, 7)):
            with self.subTest(width=width, degree=degree):
                mask = connected_regular_mask(width, degree, 935)
                self.assertTrue(torch.equal(mask, connected_regular_mask(width, degree, 935)))
                index = torch.triu_indices(width, width, 1)
                adjacency = torch.zeros(width, width, dtype=torch.bool)
                adjacency[index[0], index[1]] = mask
                adjacency |= adjacency.T.clone()
                self.assertEqual(int(mask.sum()), width * degree // 2)
                self.assertTrue(torch.equal(adjacency.sum(0), torch.full((width,), degree)))
                self.assertEqual(int(adjacency.diag().sum()), 0)
                reached = {0}
                while True:
                    expanded = reached | {other for node in reached for other in adjacency[node].nonzero().flatten().tolist()}
                    if expanded == reached:
                        break
                    reached = expanded
                self.assertEqual(len(reached), width)
        self.assertFalse(torch.equal(connected_regular_mask(16, 4, 935), connected_regular_mask(16, 4, 936)))
        self.assertEqual(random.getstate(), python_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_paired_birth_is_bitwise_identical_in_matrix_io_and_multistep_function(self):
        models = [_model(mode) for mode in MODES]
        masks, metadata = zip(*(model.initialize_birth(29, 31, degree=2) for model in models))
        self.assertTrue(torch.equal(masks[0], masks[1]))
        self.assertEqual(metadata[0], metadata[1])
        self.assertTrue(torch.equal(models[0].recurrent_matrix(), models[1].recurrent_matrix()))
        self.assertTrue(torch.equal(models[0].edge_shared, models[1].edge_shared))
        self.assertEqual(int(models[1].edge_skew.count_nonzero()), 0)
        for name in ("encoder.weight", "head.weight", "head.bias"):
            self.assertTrue(torch.equal(dict(models[0].named_parameters())[name], dict(models[1].named_parameters())[name]))
        hidden = [torch.linspace(-.4, .7, 12, dtype=torch.double).reshape(2, 6) for _ in models]
        obs = torch.linspace(-.6, .5, 6, dtype=torch.double).reshape(2, 3)
        for dt in (0., .25, .375, .7):
            result = [model(obs, state, dt) for model, state in zip(models, hidden)]
            self.assertTrue(torch.equal(result[0][0], result[1][0]))
            self.assertTrue(torch.equal(result[0][1], result[1][1]))
            hidden = [value[1] for value in result]

    def test_birth_global_frobenius_scale_and_rng_independence(self):
        model = _model(width=8)
        before = torch.get_rng_state().clone()
        mask, metadata = model.initialize_birth(mask_seed=19, weight_seed=97, degree=4, gain=1.2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        matrix = model.recurrent_matrix()
        torch.testing.assert_close(matrix.norm(), matrix.new_tensor(1.2 * math.sqrt(8)), rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(metadata["initial_mean_row_squared_norm"], 1.2 ** 2)
        self.assertEqual(int(mask.sum()), 16)
        self.assertEqual(metadata["initial_oriented_couplings"], 32)
        self.assertAlmostEqual(metadata["initial_operator_norm"], float(torch.linalg.matrix_norm(matrix, 2).detach()))

    def test_one_true_shared_parameter_affects_both_directions_and_no_others(self):
        model = _model(width=4)
        with torch.no_grad():
            model.edge_shared.zero_()
            model.edge_shared[0] = math.sqrt(2.) * .7
        matrix = model.recurrent_matrix()
        expected = torch.zeros(4, 4, dtype=torch.double)
        expected[0, 1] = expected[1, 0] = .7
        torch.testing.assert_close(matrix, expected, rtol=1e-15, atol=1e-15)
        hidden = torch.tensor([.2, -.4, .7, -.3], dtype=torch.double)
        jacobian = torch.func.jacrev(lambda shared: torch.func.functional_call(
            model, {"edge_shared": shared}, (torch.zeros(3, dtype=torch.double), hidden, .25))[1])(model.edge_shared)
        self.assertGreater(float(jacobian[0, 0].abs().detach()), 0.)
        self.assertGreater(float(jacobian[1, 0].abs().detach()), 0.)
        self.assertEqual(float(jacobian[2:, 0].abs().sum().detach()), 0.)
        self.assertTrue(torch.equal(matrix, matrix.T))
        self.assertEqual(int(matrix.diag().count_nonzero()), 0)

    def test_untied_control_can_change_one_physical_direction(self):
        model = _model("untied", width=4)
        before = model.recurrent_matrix().detach().clone()
        with torch.no_grad():
            model.edge_shared[0] += .2
            model.edge_skew[0] += .2
        delta = model.recurrent_matrix().detach() - before
        self.assertGreater(float(delta[0, 1]), .28)
        self.assertEqual(float(delta[1, 0]), 0.)
        self.assertEqual(int(delta.count_nonzero()), 1)

    def test_orthonormal_coordinates_match_matrix_frobenius_norm_and_gradients(self):
        gradient = torch.arange(16, dtype=torch.double).reshape(4, 4) / 7
        for mode in MODES:
            model = _model(mode, width=4)
            if mode == "untied":
                with torch.no_grad():
                    model.edge_skew.copy_(torch.linspace(-.5, .8, model.pair_count, dtype=torch.double))
            matrix = model.recurrent_matrix()
            coordinate_squared = model.edge_shared.square().sum()
            parameters = [model.edge_shared]
            if mode == "untied":
                coordinate_squared = coordinate_squared + model.edge_skew.square().sum()
                parameters.append(model.edge_skew)
            torch.testing.assert_close(matrix.square().sum(), coordinate_squared, rtol=1e-14, atol=1e-14)
            actual = torch.autograd.grad((matrix * gradient).sum(), parameters)
            projected = model.project_matrix_gradient(gradient)
            for value, expected in zip(actual, projected.values()):
                torch.testing.assert_close(value, expected, rtol=1e-14, atol=1e-14)

    def test_inactive_potential_has_true_derivative_then_operator_projects_both_coordinates(self):
        for mode in MODES:
            model = _model(mode, width=6)
            mask, _ = model.initialize_birth(17, 19, degree=2)
            inactive = (~mask).nonzero().flatten()[0]
            model.validate_eligibility(mask)
            hidden = torch.linspace(-.7, .9, 6, dtype=torch.double)
            sensitivity = torch.autograd.grad(model.vector_field(hidden).square().sum(), model.edge_shared)[0]
            self.assertNotEqual(float(sensitivity[inactive]), 0.)
            old = model.edge_shared[inactive].detach().clone()
            epsilon = 1e-6
            with torch.no_grad():
                model.edge_shared[inactive] = old + epsilon
                plus = model.vector_field(hidden).square().sum()
                model.edge_shared[inactive] = old - epsilon
                minus = model.vector_field(hidden).square().sum()
                model.edge_shared[inactive] = old
            torch.testing.assert_close(sensitivity[inactive], (plus - minus) / (2 * epsilon), rtol=2e-6, atol=2e-7)
            with torch.no_grad():
                model.edge_shared[inactive] = .3
                if mode == "untied":
                    model.edge_skew[inactive] = .2
            with self.assertRaises(ValueError):
                model.validate_eligibility(mask)
            model.project_inactive_(mask)
            model.validate_eligibility(mask)
            u, v = model.pair_index[:, inactive]
            self.assertEqual(float(model.recurrent_matrix()[u, v].detach()), 0.)
            self.assertEqual(float(model.recurrent_matrix()[v, u].detach()), 0.)
            expanded = mask.clone()
            expanded[inactive] = True
            with torch.no_grad():
                model.edge_shared[inactive] = .3
            model.validate_eligibility(expanded)
            self.assertNotEqual(float(model.recurrent_matrix()[u, v].detach()), 0.)
            for invalid in (mask.float(), mask.reshape(1, -1)):
                with self.assertRaises(ValueError):
                    model.validate_eligibility(invalid)

    def test_heun_selected_field_zero_time_and_single_observation_pulse(self):
        model = _model()
        hidden = torch.linspace(-.4, .6, 12, dtype=torch.double).reshape(2, 6)
        expected = hidden
        dt = .375 / 2
        for _ in range(2):
            first = -expected + F.linear(expected.tanh(), model.recurrent_matrix())
            second = model.vector_field(expected + dt * first)
            expected = expected + .5 * dt * (first + second)
        self.assertTrue(torch.equal(model.flow(hidden, .375), expected))
        self.assertTrue(torch.equal(model.flow(hidden, 0.), hidden))
        observation = torch.linspace(-.3, .7, 6, dtype=torch.double).reshape(2, 3)
        for max_step in (.25, .0625):
            _, actual = model.step(observation, model.initial_state(2), .5, max_step)
            self.assertTrue(torch.equal(actual, model.encoder(observation)))
        expected = model(observation.float().double(), hidden, .25)
        actual = model(observation.float(), hidden, .25)
        self.assertEqual(actual[0].dtype, torch.double)
        for value, reference in zip(actual, expected):
            self.assertTrue(torch.equal(value, reference))
        for dt, limit in ((-.1, .25), (float("nan"), .25), (.1, 0.), (.1, float("inf"))):
            with self.assertRaises(ValueError):
                model.flow(hidden, dt, limit)

    def test_functional_call_jacrev_vmap_and_complete_gradcheck(self):
        for mode in MODES:
            model = _model(mode, width=4)
            parameters = dict(model.named_parameters())
            observations = torch.linspace(-.5, .4, 9, dtype=torch.double).reshape(3, 3)
            hidden = torch.linspace(-.6, .7, 12, dtype=torch.double).reshape(3, 4)
            single = lambda params, obs, state: torch.func.functional_call(model, params, (obs, state, .375))[1]
            transformed = torch.vmap(torch.func.jacrev(single, argnums=(0, 2)), in_dims=(None, 0, 0))(
                parameters, observations, hidden)
            self.assertEqual(transformed[0]["edge_shared"].shape, (3, 4, 6))
            self.assertEqual(transformed[1].shape, (3, 4, 4))
            for index in range(3):
                expected = torch.func.jacrev(single, argnums=(0, 2))(parameters, observations[index], hidden[index])
                for name in parameters:
                    torch.testing.assert_close(transformed[0][name][index], expected[0][name], rtol=1e-13, atol=1e-13)
            keys, values = tuple(parameters), tuple(value.detach().clone().requires_grad_() for value in parameters.values())
            observation = observations[0].clone().requires_grad_()
            state = hidden[0].clone().requires_grad_()

            def evaluate(observation, state, *coordinates):
                logits, next_state = torch.func.functional_call(model, dict(zip(keys, coordinates)), (observation, state, .375))
                return torch.cat((logits.flatten(), next_state))

            self.assertTrue(torch.autograd.gradcheck(evaluate, (observation, state, *values),
                                                    eps=1e-6, atol=2e-5, rtol=2e-4))

    def test_symmetric_flow_energy_derivative_and_generally_asymmetric_jacobian(self):
        model = _model(width=4)
        hidden = torch.tensor([.1, -.4, .8, -.2], dtype=torch.double, requires_grad=True)
        field = model.vector_field(hidden)
        energy_gradient = torch.autograd.grad(_energy(model, hidden), hidden)[0]
        actual = (energy_gradient * field).sum()
        expected = -((1 - hidden.tanh().square()) * field.square()).sum()
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        self.assertLess(float(actual.detach()), 0.)
        jacobian = torch.func.jacrev(model.vector_field)(hidden)
        analytic = -torch.eye(4, dtype=torch.double) + model.recurrent_matrix() * (1 - hidden.tanh().square())[None, :]
        torch.testing.assert_close(jacobian, analytic, rtol=1e-14, atol=1e-14)
        self.assertGreater(float((jacobian - jacobian.T).abs().sum().detach()), .01)
        # Step refinement is checked on this finite example, not asserted as a
        # general Lyapunov guarantee for the finite-step Heun integrator.
        refined = model.flow(hidden, .25, max_step=.005)
        self.assertLess(float(_energy(model, refined).detach()), float(_energy(model, hidden).detach()))

    def _serialization_roundtrip(self, device):
        for mode in MODES:
            model = _model(mode).to(device)
            if mode == "untied":
                with torch.no_grad():
                    model.edge_skew[0] = .12
            hidden = torch.linspace(-.3, .4, 12, dtype=torch.double, device=device).reshape(2, 6)
            observation = torch.linspace(-.2, .5, 6, dtype=torch.double, device=device).reshape(2, 3)
            expected = model(observation, hidden, .375)
            buffer = io.BytesIO()
            torch.save({"model_config": model.config(), "model": model.state_dict(), "hidden": hidden}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
            restored = UndirectedModel(**checkpoint["model_config"]).double()
            restored.load_state_dict(checkpoint["model"])
            actual = restored(observation.cpu(), checkpoint["hidden"], .375)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference.cpu(), rtol=1e-12, atol=1e-12)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), restored.state_dict()[key]))
            damaged = dict(checkpoint["model"])
            damaged["pair_index"] = damaged["pair_index"].flip(1)
            with self.assertRaises(RuntimeError):
                restored.load_state_dict(damaged)

    def test_cpu_checkpoint_config_and_canonical_edge_indices(self):
        self._serialization_roundtrip("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_checkpoint_loads_on_cpu(self):
        self._serialization_roundtrip("cuda")


class PortGraphModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_no_trainable_interfaces_and_all_port_nodes_belong_to_one_graph(self):
        for mode in MODES:
            model = PortGraphModel(mode=mode)
            expected_names = ("edge_shared", "edge_skew") if mode == "untied" else ("edge_shared",)
            self.assertEqual(tuple(dict(model.named_parameters())), expected_names)
            self.assertEqual(model.dynamic_parameter_names(), expected_names)
            self.assertFalse(hasattr(model, "encoder"))
            self.assertFalse(hasattr(model, "head"))
            self.assertEqual(model.input_ports.tolist(), list(range(56)))
            self.assertEqual(model.output_ports.tolist(), list(range(56, 78)))
            report = model.parameter_report()
            self.assertEqual(report["total_parameters"], 3160 if mode == "undirected" else 6320)
            self.assertEqual(report["input_interface_parameters"], 0)
            self.assertEqual(report["readout_interface_parameters"], 0)
            self.assertEqual(report["non_port_state_nodes"], 2)
            self.assertEqual(report["interface"], "fixed_graph_ports")
            core = UndirectedModel(input_size=56, n_jobs=2, width=80, mode=mode)
            self.assertEqual(core.dynamic_parameter_names(), (*expected_names, "encoder.weight"))
            self.assertEqual(core.parameter_report()["total_parameters"], report["total_parameters"] + 6262)

    def test_initializers_preserve_ambient_random_streams(self):
        before_python, before_cpu = random.getstate(), torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        for mode in MODES:
            for constructor in (PortGraphModel, lambda **kw: UndirectedModel(input_size=56, n_jobs=2, width=80, **kw)):
                model = constructor(mode=mode).double()
                if isinstance(model, PortGraphModel):
                    pass
                else:
                    model.initialize_port_matched_interfaces_()
                model.initialize_birth(183, 281, degree=16)
        self.assertEqual(before_python, random.getstate())
        self.assertTrue(torch.equal(before_cpu, torch.get_rng_state()))
        for before, after in zip(before_cuda, torch.cuda.get_rng_state_all() if before_cuda else []):
            self.assertTrue(torch.equal(before, after))

    def test_four_arm_birth_has_exactly_the_same_full_function(self):
        models, births = [], []
        for mode in MODES:
            core = UndirectedModel(input_size=56, n_jobs=2, width=80, mode=mode).double()
            interface_birth = core.initialize_port_matched_interfaces_()
            self.assertEqual(interface_birth["readout"], "linear")
            port = PortGraphModel(mode=mode).double()
            for model in (core, port):
                _, metadata = model.initialize_birth(mask_seed=99, weight_seed=183, degree=16)
                models.append(model)
                births.append(metadata)
        self.assertTrue(all(birth == births[0] for birth in births))
        for model in models[1:]:
            self.assertTrue(torch.equal(model.recurrent_matrix(), models[0].recurrent_matrix()))
        obs = torch.linspace(-.4, 2., 112, dtype=torch.float32).reshape(2, 56)
        hidden = [torch.linspace(-.6, .8, 160, dtype=torch.double).reshape(2, 80) for _ in models]
        for dt in (0., .25, .375, .7):
            outputs = [model(obs, state, dt) for model, state in zip(models, hidden)]
            for logits, state in outputs[1:]:
                self.assertTrue(torch.equal(state, outputs[0][1]))
                self.assertTrue(torch.equal(logits, outputs[0][0]))
            hidden = [value[1] for value in outputs]

    def test_public_input_units_change_only_duration_and_window_without_labels(self):
        model = PortGraphModel().double()
        obs = torch.arange(1., 113., dtype=torch.double).reshape(2, 56)
        original = obs.clone()
        normalized = model.normalized_observation(obs)
        expected = obs.clone()
        expected[..., [9, 37]] /= 16.
        expected[..., [10, 38]] /= 2.5
        torch.testing.assert_close(normalized, expected, rtol=1e-15, atol=1e-15)
        self.assertTrue(torch.equal(obs, original))
        self.assertEqual(model.input_scale.tolist(), contextual_input_scale(2))
        selector = torch.zeros(56, dtype=torch.double)
        selector[[9, 10, 37, 38]] = 1
        self.assertTrue(torch.equal(normalized[:, selector.eq(0)], obs[:, selector.eq(0)]))
        pulse = model.observation_pulse(obs)
        self.assertTrue(torch.equal(pulse[:, :56], normalized * .25))
        self.assertEqual(float(pulse[:, 56:].abs().sum()), 0.)

    def test_output_port_sends_shared_feedback_to_input_and_internal_nodes(self):
        model = PortGraphModel(input_size=2, n_jobs=1, n_actions=1, width=4,
                               input_scale=[1., 1.]).double()
        with torch.no_grad():
            model.edge_shared.zero_()
            for pair, value in (((0, 2), .8), ((2, 3), .6)):
                index = ((model.pair_index[0] == pair[0]) & (model.pair_index[1] == pair[1])).nonzero().item()
                model.edge_shared[index] = value * math.sqrt(2.)
        hidden = torch.tensor([0., 0., .7, 0.], dtype=torch.double)
        field = model.vector_field(hidden)
        self.assertGreater(float(field[0].detach()), .4)
        self.assertGreater(float(field[3].detach()), .3)
        self.assertTrue(torch.equal(model.recurrent_matrix(), model.recurrent_matrix().T))
        obs = torch.zeros(2, dtype=torch.double)
        jacobian = torch.func.jacrev(lambda state: model(obs, state, .25)[1])(hidden)
        self.assertGreater(float(jacobian[0, 2].detach()), 0.)
        self.assertGreater(float(jacobian[3, 2].detach()), 0.)
        # The output is an ordinary persistent state; it is neither clamped nor
        # a one-way learned output layer, and logits are not tanh-bounded.
        large = hidden.clone()
        large[2] = 12.
        logits, unchanged = model(obs, large, 0.)
        self.assertEqual(float(logits.detach()), 12.)
        self.assertTrue(torch.equal(unchanged, large))
        _, evolved = model(obs, hidden, .25)
        self.assertNotEqual(float(evolved[2].detach()), 0.)
        self.assertNotEqual(float(evolved[2].detach()), float(hidden[2]))

    def test_ports_functional_call_float64_vmap_and_all_parameter_gradcheck(self):
        for mode in MODES:
            model = PortGraphModel(input_size=3, n_jobs=1, n_actions=2, width=8,
                                   input_scale=[1., .5, .25], mode=mode).double()
            model.initialize_birth(19, 71, degree=2)
            parameters = dict(model.named_parameters())
            obs = torch.linspace(-.5, .7, 9, dtype=torch.double).reshape(3, 3)
            hidden = torch.linspace(-.6, .8, 24, dtype=torch.double).reshape(3, 8)
            single = lambda params, observation, state: torch.func.functional_call(model, params, (observation, state, .375))[1]
            jacobians = torch.vmap(torch.func.jacrev(single, argnums=(0, 2)), in_dims=(None, 0, 0))(
                parameters, obs.float(), hidden)
            self.assertEqual(jacobians[0]["edge_shared"].shape, (3, 8, 28))
            self.assertEqual(jacobians[1].shape, (3, 8, 8))
            names = tuple(parameters)
            values = tuple(value.detach().clone().requires_grad_() for value in parameters.values())

            def evaluate(observation, state, *coordinates):
                logits, following = torch.func.functional_call(model, dict(zip(names, coordinates)), (observation, state, .375))
                return torch.cat((logits.flatten(), following))

            self.assertTrue(torch.autograd.gradcheck(evaluate, (obs[0].requires_grad_(), hidden[0].requires_grad_(), *values),
                                                    eps=1e-6, atol=2e-5, rtol=2e-4))

    def test_invalid_or_overlapping_ports_and_fixed_scale_configuration(self):
        for kwargs in ({"width": 77}, {"input_ports": [0] * 56}, {"pulse_gain": 0.},
                       {"input_scale": [1.] * 55}, {"input_scale": [0.] * 56},
                       {"input_size": 3, "n_jobs": 1, "n_actions": 2, "width": 8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PortGraphModel(**kwargs)
        model = PortGraphModel()
        with self.assertRaises(ValueError):
            model.initialize_port_matched_interfaces_()

    def _serialization_roundtrip(self, device):
        for mode in MODES:
            models = [PortGraphModel(mode=mode).double().to(device),
                      UndirectedModel(input_size=56, n_jobs=2, width=80, mode=mode).double().to(device)]
            models[1].initialize_port_matched_interfaces_()
            for model in models:
                model.initialize_birth(31, 97, degree=16)
                obs = torch.linspace(-.2, .5, 56, dtype=torch.float32, device=device)
                hidden = torch.linspace(-.3, .4, 80, dtype=torch.double, device=device)
                expected = model(obs, hidden, .375)
                buffer = io.BytesIO()
                torch.save({"config": model.config(), "model": model.state_dict(), "hidden": hidden}, buffer)
                buffer.seek(0)
                state = torch.load(buffer, map_location="cpu", weights_only=False)
                restored = type(model)(**state["config"]).double()
                restored.load_state_dict(state["model"])
                self.assertEqual(restored.config(), model.config())
                for actual, reference in zip(restored(obs.cpu(), state["hidden"], .375), expected):
                    torch.testing.assert_close(actual, reference.cpu(), rtol=1e-12, atol=1e-12)
                damaged = dict(state["model"])
                if isinstance(model, PortGraphModel):
                    damaged["output_ports"] = damaged["output_ports"].flip(0)
                else:
                    damaged["input_scale"] = damaged["input_scale"] * 2
                with self.assertRaises(RuntimeError):
                    restored.load_state_dict(damaged)

    def test_cpu_port_and_core_configuration_roundtrip(self):
        self._serialization_roundtrip("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_port_and_core_checkpoint_load_on_cpu(self):
        self._serialization_roundtrip("cuda")


if __name__ == "__main__":
    unittest.main()
