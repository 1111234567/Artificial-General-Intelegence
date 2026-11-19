"""Exact finite-trajectory derivatives, long credit and continuation checks."""
from copy import deepcopy
import io
import math
import unittest

import torch
from torch.func import functional_call
from torch.nn import functional as F

from experiments.undirected.credit import (
    CREDIT_METHODS, CreditConfig, TemporalCreditEngine, snapshot_parameters,
    state_digest, weighted_loss, stable_l2, diagnostic_inner_product,
)
from experiments.undirected.model import UndirectedModel, PortGraphModel


def tiny(mode="undirected", width=4, n_jobs=2, n_actions=3):
    torch.manual_seed(193)
    model = UndirectedModel(input_size=3, n_jobs=n_jobs, n_actions=n_actions,
                            width=width, mode=mode).double()
    with torch.no_grad():
        model.edge_shared.normal_(0., .15)
        if mode == "untied":
            model.edge_skew.normal_(0., .07)
    return model


def tiny_ports(mode="undirected"):
    model = PortGraphModel(input_size=3, n_jobs=1, n_actions=2, width=6,
                           mode=mode, input_scale=(1., 1., 1.), pulse_gain=.25).double()
    generator = torch.Generator().manual_seed(621)
    with torch.no_grad():
        model.edge_shared.copy_(torch.randn(model.edge_shared.shape, generator=generator, dtype=torch.double) * .15)
        if mode == "untied":
            model.edge_skew.copy_(torch.randn(model.edge_skew.shape, generator=generator, dtype=torch.double) * .07)
    return model


def block(model, h, start, ticks, *, observations=None, targets=None, elapsed=None):
    generator = torch.Generator().manual_seed(400 + start)
    if observations is None:
        observations = torch.randn(ticks, len(h), model.input_size, generator=generator,
                                   dtype=h.dtype, device="cpu").to(h.device) * .15
    if targets is None:
        targets = torch.randint(model.n_actions, (ticks, len(h), model.n_jobs), generator=generator).to(h.device)
    elapsed = tuple(.25 if index % 2 else .375 for index in range(ticks)) if elapsed is None else tuple(elapsed)
    start_h = h.detach().clone()
    parameters = snapshot_parameters(model)
    with torch.no_grad():
        for observation, dt in zip(observations, elapsed):
            _, h = model(observation, h, dt)
    return {"observations": observations.detach().clone(), "targets": targets.detach().clone(),
            "elapsed": elapsed, "start_h": start_h, "end_h": h.detach().clone(),
            "start_tick": start, "end_tick": start + ticks}, parameters


def standard_gradients(model, chunks, wait_weight=.1):
    h, outputs = chunks[0]["start_h"].detach().clone(), []
    for index, chunk in enumerate(chunks):
        for observation, dt in zip(chunk["observations"], chunk["elapsed"]):
            output, h = model(observation, h, dt)
            if index == len(chunks) - 1:
                outputs.append(output)
    loss, _, _ = weighted_loss(torch.stack(outputs), chunks[-1]["targets"], wait_weight)
    return {name: value.detach() for (name, _), value in zip(model.named_parameters(),
            torch.autograd.grad(loss, tuple(model.parameters())))}


def apply_gradients(optimizer, model, gradients):
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.grad = gradients[name].clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def flatten(values):
    return torch.cat([value.reshape(-1) for value in values.values()])


class TemporalCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_gradients(self, actual, expected, *, atol=2e-12, rtol=2e-11):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol)

    def test_configuration_validation_and_horizon(self):
        self.assertEqual(CreditConfig("recorded512").credit_ticks, 512)
        self.assertEqual(CreditConfig("recorded1024").credit_ticks, 1024)
        self.assertIsNone(CreditConfig("rtrl").credit_ticks)
        self.assertEqual(CreditConfig("rtrl_reset").credit_ticks, 128)
        for changes in ({"method": "diagonal"}, {"update_ticks": 0}, {"update_ticks": True},
                        {"jacrev_chunk_size": 0}, {"wait_weight": 0}, {"wait_weight": float("nan")},
                        {"primal_atol": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CreditConfig(**changes)

    def test_scaled_l2_preserves_finite_large_and_tiny_vectors_without_modification(self):
        for scale in (1e200, 1e-200):
            value = torch.tensor([3. * scale, 4. * scale], dtype=torch.float64)
            before = value.clone()
            self.assertAlmostEqual(stable_l2(value) / scale, 5., places=13)
            self.assertTrue(torch.equal(value, before))
        maximum = torch.finfo(torch.float64).max
        with self.assertRaisesRegex(FloatingPointError, "true diagnostic L2"):
            stable_l2(torch.tensor([maximum, maximum], dtype=torch.float64))

    def test_inner_product_diagnostic_marks_overflow_instead_of_inf_json(self):
        value, overflow = diagnostic_inner_product(torch.tensor([1e200], dtype=torch.float64),
                                                  torch.tensor([1e200], dtype=torch.float64))
        self.assertIsNone(value)
        self.assertTrue(overflow)
        self.assertEqual(diagnostic_inner_product(torch.tensor([2., 3.]), torch.tensor([4., -1.])), (5., False))

    def test_dynamic_parameter_declaration_is_explicit_and_validated(self):
        for declaration in ((), ("missing",), ("edge_shared", "edge_shared"), "edge_shared"):
            model = tiny()
            model.dynamic_parameter_names = lambda declaration=declaration: declaration
            with self.subTest(declaration=declaration), self.assertRaisesRegex(ValueError, "dynamic parameter"):
                TemporalCreditEngine(model)

    def test_weighted_loss_uses_one_global_denominator(self):
        logits = torch.tensor([[[[2., -1., 0.], [1., 0., -1.]]],
                               [[[0., 2., -1.], [-1., 0., 3.]]]], dtype=torch.double, requires_grad=True)
        targets = torch.tensor([[[0, 0]], [[1, 2]]])
        loss, numerator, denominator = weighted_loss(logits, targets, .1)
        reference = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1),
                                    weight=logits.new_tensor([.1, 1., 1.]))
        torch.testing.assert_close(loss, reference)
        self.assertAlmostEqual(float(denominator), 2.2)
        torch.testing.assert_close(numerator, reference * 2.2)
        wrong = sum(weighted_loss(logits[index], targets[index], .1)[0] for index in range(2)) / 2
        self.assertGreater(float((loss - wrong).detach().abs()), .01)

    def test_first_block_and_reset_match_standard_tbptt_for_both_graph_modes(self):
        for mode in ("undirected", "untied"):
            model = tiny(mode)
            h = model.initial_state(2)
            chunk, snapshot = block(model, h, 0, 4)
            expected = standard_gradients(model, [chunk])
            before_model, before_chunk = state_digest(model.state_dict()), state_digest(chunk)
            for method in CREDIT_METHODS:
                with self.subTest(mode=mode, method=method):
                    engine = TemporalCreditEngine(model, CreditConfig(method, update_ticks=4))
                    actual, metrics = engine.gradients(chunk, snapshot)
                    self.assert_gradients(actual, expected)
                    self.assertEqual(metrics["loss_ticks"], 4)
                    self.assertEqual(metrics["end_h_hash"], state_digest(chunk["end_h"]))
                    self.assertEqual(metrics["work"]["optimizer_steps"], 0)
                    self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                    self.assertEqual(state_digest(model.state_dict()), before_model)
                    self.assertEqual(state_digest(chunk), before_chunk)

    def test_fixed_parameters_recorded_and_rtrl_equal_full_bptt(self):
        for mode in ("undirected", "untied"):
            model, chunks = tiny(mode), []
            engines = {method: TemporalCreditEngine(model, CreditConfig(method, update_ticks=4))
                       for method in ("recorded512", "recorded1024", "rtrl")}
            h = model.initial_state(2)
            for index in range(3):
                chunk, snapshot = block(model, h, 4 * index, 4)
                h = chunk["end_h"]
                chunks.append(chunk)
                expected = standard_gradients(model, chunks)
                for method, engine in engines.items():
                    with self.subTest(mode=mode, method=method, block=index):
                        actual, metrics = engine.gradients(chunk, snapshot)
                        self.assert_gradients(actual, expected)
                        self.assertEqual(metrics["credit_path_ticks"], 4 * (index + 1))
                        if method == "rtrl":
                            self.assertEqual(metrics["work"]["jacobian_reverse_rows"], 8)
                            self.assertEqual(engine.sensitivity.shape,
                                             (2, 4, engine.dynamic_parameter_count))

    def test_recorded_parameter_changes_match_finite_difference(self):
        model = tiny("untied")
        engine = TemporalCreditEngine(model, CreditConfig("recorded512", update_ticks=3))
        h, chunks, snapshots = model.initial_state(1), [], []
        for index in range(3):
            if index:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.add_(.013 * (index if name == "edge_shared" else -index))
            chunk, snapshot = block(model, h, 3 * index, 3)
            h = chunk["end_h"]
            chunks.append(chunk)
            snapshots.append(snapshot)
            gradients, _ = engine.gradients(chunk, snapshot)
        generator = torch.Generator().manual_seed(817)
        directions = {name: torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                      for name, parameter in model.named_parameters()}
        norm = flatten(directions).norm()
        directions = {name: direction / norm for name, direction in directions.items()}

        def evaluate(amount):
            hidden = chunks[0]["start_h"].clone()
            outputs = []
            for index, (chunk, snapshot) in enumerate(zip(chunks, snapshots)):
                values = {name: parameter + amount * directions[name] for name, parameter in snapshot.items()}
                for observation, dt in zip(chunk["observations"], chunk["elapsed"]):
                    output, hidden = functional_call(model, values, (observation, hidden, dt))
                    if index == len(chunks) - 1:
                        outputs.append(output)
            loss = weighted_loss(torch.stack(outputs), chunks[-1]["targets"])[0]
            return float(loss), hidden

        eps = 1e-5
        finite = (evaluate(eps)[0] - evaluate(-eps)[0]) / (2 * eps)
        analytic = sum(float((gradients[name] * direction).sum()) for name, direction in directions.items())
        self.assertAlmostEqual(analytic, finite, places=9)
        torch.testing.assert_close(evaluate(0)[1], h, atol=0, rtol=0)
        self.assertEqual(engine.update_age, 3)

    def test_delayed_cue_beyond_128_has_zero_short_and_nonzero_long_encoder_gradient(self):
        model = tiny(width=4, n_jobs=1, n_actions=2)
        with torch.no_grad():
            # A near-neutral collective mode makes this a useful gradient-path
            # test; ordinary leaking random networks can erase the derivative.
            model.edge_shared.fill_(math.sqrt(2) / 3)
            model.encoder.weight.fill_(.01)
            model.head.weight[0].fill_(.25)
            model.head.weight[1].fill_(-.25)
            model.head.bias.zero_()
        engines = {method: TemporalCreditEngine(model, CreditConfig(method))
                   for method in ("tbptt128", "recorded512", "rtrl", "rtrl_reset")}
        h = model.initial_state(1)
        for index in range(2):
            if index:
                with torch.no_grad():
                    model.edge_shared.add_(.0005)
            observations = torch.zeros(128, 1, 3, dtype=torch.double)
            if not index:
                observations[0, 0, 0] = 1.
            chunk, snapshot = block(model, h, 128 * index, 128, observations=observations,
                                    targets=torch.ones(128, 1, 1, dtype=torch.long), elapsed=[.25] * 128)
            h = chunk["end_h"]
            results = {method: engine.gradients(chunk, snapshot) for method, engine in engines.items()}
        short, long, rtrl, reset = (results[method][0] for method in engines)
        self.assertEqual(float(short["encoder.weight"].norm()), 0.)
        self.assertGreater(float(long["encoder.weight"].norm()), .1)
        self.assert_gradients(long, rtrl)
        self.assert_gradients(short, reset)
        self.assertEqual(results["rtrl"][1]["credit_path_ticks"], 256)
        self.assertIsNone(results["rtrl"][1]["credit_hard_cap_ticks"])
        for field in ("jacobian_forward_calls", "jacobian_reverse_rows", "jacobian_reverse_chunks",
                      "sensitivity_matmul_mac", "trace_bytes"):
            self.assertEqual(results["rtrl"][1]["work"][field], results["rtrl_reset"][1]["work"][field])

    def test_reset_control_matches_128_across_exogenous_parameter_changes(self):
        model = tiny("untied")
        engines = {method: TemporalCreditEngine(model, CreditConfig(method, update_ticks=4))
                   for method in ("tbptt128", "rtrl_reset", "rtrl")}
        h = model.initial_state(2)
        for index in range(3):
            chunk, snapshot = block(model, h, 4 * index, 4)
            h = chunk["end_h"]
            results = {method: engine.gradients(chunk, snapshot) for method, engine in engines.items()}
            self.assert_gradients(results["tbptt128"][0], results["rtrl_reset"][0])
            with torch.no_grad():
                model.edge_shared.mul_(.7)  # Structural value changes do not reset S.
                model.encoder.weight.add_(.01)
        self.assertGreater(float((flatten(results["rtrl"][0]) - flatten(results["tbptt128"][0])).norm()), 1e-5)

    def test_rtrl_gradient_and_trace_diagnostics_are_exact_without_extra_derivatives(self):
        model = tiny("untied")
        engines = {method: TemporalCreditEngine(model, CreditConfig(method, update_ticks=4))
                   for method in ("tbptt128", "rtrl_reset", "rtrl")}
        h = model.initial_state(2)
        for index in range(3):
            previous = engines["rtrl"].sensitivity
            expected_previous_norm = 0. if previous is None else float(previous.norm())
            chunk, snapshot = block(model, h, 4 * index, 4)
            h = chunk["end_h"]
            results = {method: engine.gradients(chunk, snapshot) for method, engine in engines.items()}
            local, full = results["tbptt128"][0], results["rtrl"][0]
            correction = flatten(full) - flatten(local)
            diagnostics = results["rtrl"][1]["gradient_diagnostics"]
            self.assertAlmostEqual(diagnostics["local_gradient_l2"], float(flatten(local).norm()), places=12)
            self.assertAlmostEqual(diagnostics["full_gradient_l2"], float(flatten(full).norm()), places=12)
            self.assertAlmostEqual(diagnostics["cross_boundary_gradient_l2"], float(correction.norm()), places=12)
            self.assertAlmostEqual(diagnostics["local_cross_inner_product"], float(flatten(local) @ correction), places=12)
            self.assertAlmostEqual(diagnostics["previous_sensitivity_l2"], expected_previous_norm, places=12)
            self.assertAlmostEqual(diagnostics["next_sensitivity_l2"], float(engines["rtrl"].sensitivity.norm()), places=12)
            self.assertEqual(diagnostics["cross_boundary_by_parameter_l2"]["head.weight"], 0.)
            self.assertEqual(diagnostics["cross_boundary_by_parameter_l2"]["head.bias"], 0.)
            self.assertEqual(results["rtrl_reset"][1]["gradient_diagnostics"]["cross_boundary_gradient_l2"], 0.)
            self.assertEqual(results["rtrl_reset"][1]["gradient_diagnostics"]["previous_sensitivity_l2"], 0.)
            for method in ("rtrl", "rtrl_reset"):
                work = results[method][1]["work"]
                self.assertEqual(work["diagnostic_forward_calls"], 0)
                self.assertEqual(work["diagnostic_reverse_sweeps"], 0)
                self.assertEqual(work["loss_reverse_sweeps"], 1)
                self.assertEqual(work["jacobian_reverse_rows"], 8)
            self.assertEqual(results["rtrl"][1]["work"]["diagnostic_norm_elements"],
                             results["rtrl_reset"][1]["work"]["diagnostic_norm_elements"])
            with torch.no_grad():
                model.edge_shared.add_(.01)

    def test_recorded_long_window_diagnostics_mark_local_reference_uncomputed(self):
        model = tiny_ports()
        engine = TemporalCreditEngine(model, CreditConfig("recorded512", update_ticks=3))
        h = model.initial_state(2)
        for index in range(2):
            chunk, snapshot = block(model, h, 3 * index, 3)
            h = chunk["end_h"]
            gradients, metrics = engine.gradients(chunk, snapshot)
        diagnostics = metrics["gradient_diagnostics"]
        self.assertAlmostEqual(diagnostics["full_gradient_l2"], float(flatten(gradients).norm()), places=12)
        for name in ("local_gradient_l2", "local_by_parameter_l2", "cross_boundary_gradient_l2",
                     "cross_boundary_by_parameter_l2", "previous_sensitivity_l2", "next_sensitivity_l2"):
            self.assertIsNone(diagnostics[name])
        self.assertTrue(diagnostics["split_status"].startswith("not_computed"))
        self.assertEqual(metrics["work"]["diagnostic_forward_calls"], 0)
        self.assertEqual(metrics["work"]["diagnostic_reverse_sweeps"], 0)
        self.assertEqual(metrics["work"]["loss_reverse_sweeps"], 1)

    def test_save_resume_restores_tape_trace_and_next_adam_update(self):
        for method in CREDIT_METHODS:
            with self.subTest(method=method):
                model = tiny("untied")
                config = CreditConfig(method, update_ticks=3)
                engine = TemporalCreditEngine(model, config)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                h = model.initial_state(2)
                for index in range(2):
                    chunk, snapshot = block(model, h, 3 * index, 3)
                    h = chunk["end_h"]
                    gradients, _ = engine.gradients(chunk, snapshot)
                    apply_gradients(optimizer, model, gradients)
                stream = io.BytesIO()
                torch.save({"credit": engine.state_dict(), "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "h": h}, stream)
                stream.seek(0)
                saved = torch.load(stream, weights_only=False)
                restored = tiny("untied")
                restored.load_state_dict(saved["model"])
                resumed = TemporalCreditEngine(restored, config)
                resumed.load_state_dict(saved["credit"])
                resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
                resumed_optimizer.load_state_dict(saved["optimizer"])
                self.assertEqual(state_digest(engine.state_dict()), state_digest(resumed.state_dict()))
                first, first_snapshot = block(model, h, 6, 3)
                second, second_snapshot = block(restored, saved["h"], 6, 3)
                self.assertEqual(state_digest(first), state_digest(second))
                ga, ma = engine.gradients(first, first_snapshot)
                gb, mb = resumed.gradients(second, second_snapshot)
                self.assert_gradients(ga, gb, atol=0, rtol=0)
                self.assertEqual(ma, mb)
                apply_gradients(optimizer, model, ga)
                apply_gradients(resumed_optimizer, restored, gb)
                self.assertEqual(state_digest(model.state_dict()), state_digest(restored.state_dict()))
                self.assertEqual(state_digest(optimizer.state_dict()), state_digest(resumed_optimizer.state_dict()))
                self.assertEqual(state_digest(engine.state_dict()), state_digest(resumed.state_dict()))

    def test_state_is_copied_and_invalid_snapshots_or_hidden_replacement_are_rejected(self):
        model = tiny()
        engine = TemporalCreditEngine(model, CreditConfig("recorded512", update_ticks=3))
        chunk, snapshot = block(model, model.initial_state(2), 0, 3)
        engine.gradients(chunk, snapshot)
        state = engine.state_dict()
        before = state_digest(engine.state_dict())
        state["tape"][0]["observations"].zero_()
        self.assertEqual(before, state_digest(engine.state_dict()))
        with self.assertRaisesRegex(ValueError, "hash"):
            TemporalCreditEngine(model, engine.config).load_state_dict(state)
        next_chunk, next_snapshot = block(model, chunk["end_h"], 3, 3)
        wrong_h = deepcopy(next_chunk)
        wrong_h["start_h"].add_(.1)
        with self.assertRaisesRegex(ValueError, "replaced"):
            engine.gradients(wrong_h, next_snapshot)
        next_snapshot["edge_shared"].add_(.1)
        with self.assertRaisesRegex(ValueError, "pre-update"):
            engine.gradients(next_chunk, next_snapshot)
        self.assertEqual(before, state_digest(engine.state_dict()))

    def test_tape_bound_and_recorded_window_excludes_older_chunks(self):
        model = tiny()
        engine = TemporalCreditEngine(model, CreditConfig("recorded512", update_ticks=2))
        h, chunks = model.initial_state(1), []
        for index in range(6):
            chunk, snapshot = block(model, h, 2 * index, 2)
            h = chunk["end_h"]
            chunks.append(chunk)
            gradients, metrics = engine.gradients(chunk, snapshot)
        self.assertEqual(len(engine.tape), 4)
        self.assertEqual(metrics["credit_path_ticks"], 8)
        self.assertEqual(metrics["credit_hard_cap_ticks"], 8)
        self.assert_gradients(gradients, standard_gradients(model, chunks[-4:]))

    def test_zero_potential_edges_have_counterfactual_gradients(self):
        model = tiny()
        with torch.no_grad():
            model.edge_shared.zero_()
        engine = TemporalCreditEngine(model, CreditConfig("rtrl", update_ticks=3))
        chunk, snapshot = block(model, model.initial_state(2), 0, 3)
        gradients, _ = engine.gradients(chunk, snapshot)
        self.assertGreater(float(gradients["edge_shared"].norm()), 1e-6)
        self.assertGreater(float(engine.sensitivity[..., :model.pair_count].norm()), 1e-6)
        self.assertEqual(float(model.edge_shared.detach().norm()), 0.)

    def test_float32_world_observation_is_preserved_for_float64_model(self):
        model = tiny()
        engine = TemporalCreditEngine(model, CreditConfig("rtrl", update_ticks=3))
        observation = torch.randn(3, 2, 3, dtype=torch.float32)
        chunk, snapshot = block(model, model.initial_state(2), 0, 3, observations=observation)
        gradients, _ = engine.gradients(chunk, snapshot)
        self.assertEqual(engine.tape[0]["observations"].dtype, torch.float32)
        self.assertEqual(engine.sensitivity.dtype, torch.float64)
        self.assert_gradients(gradients, standard_gradients(model, [chunk]))

    def test_port_graph_has_no_local_parameters_and_matches_full_bptt(self):
        for mode in ("undirected", "untied"):
            model = tiny_ports(mode)
            self.assertFalse(hasattr(model, "encoder"))
            self.assertFalse(hasattr(model, "head"))
            engines = {method: TemporalCreditEngine(model, CreditConfig(method, update_ticks=4))
                       for method in ("recorded512", "recorded1024", "rtrl")}
            h, chunks = model.initial_state(2), []
            for index in range(3):
                chunk, snapshot = block(model, h, index * 4, 4)
                h = chunk["end_h"]
                chunks.append(chunk)
                expected = standard_gradients(model, chunks)
                for method, engine in engines.items():
                    with self.subTest(mode=mode, method=method, block=index):
                        self.assertEqual(engine.local_names, ())
                        self.assertEqual(set(engine.dynamic_names), set(dict(model.named_parameters())))
                        actual, metrics = engine.gradients(chunk, snapshot)
                        self.assert_gradients(actual, expected)
                        self.assertEqual(metrics["primal_max_abs_error"], 0.)
                        self.assertEqual(engine.state_dict()["local_names"], ())

    def test_port_graph_parameter_changes_match_recorded_rtrl_and_reset_tbptt(self):
        for mode in ("undirected", "untied"):
            model = tiny_ports(mode)
            engines = {method: TemporalCreditEngine(model, CreditConfig(method, update_ticks=3))
                       for method in ("tbptt128", "recorded512", "rtrl", "rtrl_reset")}
            h = model.initial_state(2)
            for index in range(3):
                if index:
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(.01 * index)
                chunk, snapshot = block(model, h, index * 3, 3)
                h = chunk["end_h"]
                results = {method: engine.gradients(chunk, snapshot)[0] for method, engine in engines.items()}
                with self.subTest(mode=mode, block=index):
                    self.assert_gradients(results["recorded512"], results["rtrl"])
                    self.assert_gradients(results["tbptt128"], results["rtrl_reset"])

    def test_port_matched_learned_interfaces_have_same_initial_edge_credit(self):
        for mode in ("undirected", "untied"):
            ports = tiny_ports(mode)
            core = UndirectedModel(input_size=3, n_jobs=1, n_actions=2, width=6, mode=mode).double()
            core.initialize_port_matched_interfaces_(pulse_gain=.25, input_scale=(1., 1., 1.))
            with torch.no_grad():
                core.edge_shared.copy_(ports.edge_shared)
                if mode == "untied":
                    core.edge_skew.copy_(ports.edge_skew)
            engines = [TemporalCreditEngine(model, CreditConfig("rtrl", update_ticks=4)) for model in (ports, core)]
            h = ports.initial_state(2)
            for index in range(3):
                chunk, snapshot = block(ports, h, index * 4, 4)
                h = chunk["end_h"]
                port_gradient, _ = engines[0].gradients(chunk, snapshot)
                core_gradient, _ = engines[1].gradients(chunk, snapshot_parameters(core))
                for name in ports.dynamic_parameter_names():
                    with self.subTest(mode=mode, block=index, parameter=name):
                        torch.testing.assert_close(port_gradient[name], core_gradient[name], atol=2e-12, rtol=2e-11)

    def test_port_graph_retains_gradient_paths_across_128_ticks(self):
        model = tiny_ports()
        with torch.no_grad():
            # A signed collective mode is visible in the *difference* of the
            # two raw output ports.  A uniform mode would cancel in softmax.
            signs = torch.tensor([1., 1., 1., 1., -1., -1.], dtype=torch.double)
            first, second = model.pair_index
            model.edge_shared.copy_(math.sqrt(2) * signs[first] * signs[second] / (model.width - 1))
        engines = {method: TemporalCreditEngine(model, CreditConfig(method))
                   for method in ("tbptt128", "recorded512", "rtrl", "rtrl_reset")}
        h = model.initial_state(1)
        for index in range(2):
            if index:
                with torch.no_grad():
                    model.edge_shared.add_(.0001)
            observations = torch.zeros(128, 1, model.input_size, dtype=torch.double)
            if not index:
                observations[0, 0, 0] = 1.
            chunk, snapshot = block(model, h, index * 128, 128, observations=observations,
                                    targets=torch.ones(128, 1, 1, dtype=torch.long), elapsed=(.25,) * 128)
            h = chunk["end_h"]
            results = {method: engine.gradients(chunk, snapshot)[0] for method, engine in engines.items()}
        self.assert_gradients(results["recorded512"], results["rtrl"])
        self.assert_gradients(results["tbptt128"], results["rtrl_reset"])
        self.assertGreater(float((results["rtrl"]["edge_shared"] - results["tbptt128"]["edge_shared"]).norm()), 1e-3)
        self.assertEqual(engines["rtrl"].last_metrics["credit_path_ticks"], 256)

    def test_port_graph_save_resume_has_exact_next_gradient_and_trace(self):
        for mode in ("undirected", "untied"):
            for method in CREDIT_METHODS:
                with self.subTest(mode=mode, method=method):
                    model = tiny_ports(mode)
                    config = CreditConfig(method, update_ticks=3)
                    engine = TemporalCreditEngine(model, config)
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    h = model.initial_state(2)
                    for index in range(2):
                        chunk, snapshot = block(model, h, index * 3, 3)
                        h = chunk["end_h"]
                        gradients, _ = engine.gradients(chunk, snapshot)
                        apply_gradients(optimizer, model, gradients)
                    stream = io.BytesIO()
                    torch.save({"credit": engine.state_dict(), "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(), "h": h}, stream)
                    stream.seek(0)
                    saved = torch.load(stream, weights_only=False)
                    restored = tiny_ports(mode)
                    restored.load_state_dict(saved["model"])
                    resumed = TemporalCreditEngine(restored, config)
                    resumed.load_state_dict(saved["credit"])
                    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
                    restored_optimizer.load_state_dict(saved["optimizer"])
                    next_chunk, snapshot = block(model, h, 6, 3)
                    same_chunk, same_snapshot = block(restored, saved["h"], 6, 3)
                    first, first_metrics = engine.gradients(next_chunk, snapshot)
                    second, second_metrics = resumed.gradients(same_chunk, same_snapshot)
                    self.assert_gradients(first, second, atol=0, rtol=0)
                    self.assertEqual(first_metrics, second_metrics)
                    self.assertEqual(state_digest(engine.state_dict()), state_digest(resumed.state_dict()))
                    apply_gradients(optimizer, model, first)
                    apply_gradients(restored_optimizer, restored, second)
                    self.assertEqual(state_digest(model.state_dict()), state_digest(restored.state_dict()))
                    self.assertEqual(state_digest(optimizer.state_dict()), state_digest(restored_optimizer.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_block_rtrl_has_same_derivative_as_recorded_reconstruction(self):
        model = tiny("untied").cuda()
        methods = ("recorded512", "rtrl")
        engines = {method: TemporalCreditEngine(model, CreditConfig(method, update_ticks=3)) for method in methods}
        h = model.initial_state(2)
        for index in range(2):
            chunk, snapshot = block(model, h, 3 * index, 3)
            h = chunk["end_h"]
            results = {method: engine.gradients(chunk, snapshot)[0] for method, engine in engines.items()}
            self.assert_gradients(results["recorded512"], results["rtrl"], atol=2e-11, rtol=2e-10)
            with torch.no_grad():
                model.edge_shared.add_(.01)


if __name__ == "__main__":
    unittest.main()
