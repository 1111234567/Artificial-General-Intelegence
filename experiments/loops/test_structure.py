"""Tiny analytic references; no checkpoint reads, trajectories, or training."""
import json
import math
import unittest

import torch

from experiments.loops.structure import analyze_states, analyze_weight


def walk(summary, length):
    return next(row for row in summary["closed_walks"]["by_length"] if row["length"] == length)


def assert_nested_close(test, left, right):
    if isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_close(test, left[key], right[key])
    elif isinstance(left, list):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_close(test, a, b)
    elif isinstance(left, float):
        test.assertTrue(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12), (left, right))
    else:
        test.assertEqual(left, right)


class StructureTests(unittest.TestCase):
    def test_chain_is_acyclic_and_has_no_closed_walks(self):
        weight = torch.zeros(5, 5, dtype=torch.float64)
        weight[torch.arange(1, 5), torch.arange(4)] = torch.tensor([.5, -2., .25, 1.], dtype=torch.float64)
        audit = analyze_weight(weight)
        graph = audit["graph"]["exact"]
        self.assertEqual(graph["offdiagonal_edges"], 4)
        self.assertEqual(graph["scc_sizes"], [1] * 5)
        self.assertFalse(graph["has_any_cycle"])
        self.assertEqual(graph["simple_cycle_lengths_guaranteed"], [])
        self.assertIsNone(graph["minimum_multineuron_cycle_length"])
        self.assertEqual(audit["offdiagonal"]["spectral_radius"], 0.)
        for row in audit["offdiagonal"]["closed_walks"]["by_length"]:
            self.assertEqual(row["signed_trace"], 0.)
            self.assertEqual(row["absolute_weight_trace"], 0.)
            self.assertIsNone(row["signed_to_absolute_ratio"])

    def test_feedforward_dag_remains_acyclic_with_skip_edges(self):
        weight = torch.tril(torch.ones(6, 6, dtype=torch.float64), diagonal=-1)
        audit = analyze_weight(weight)
        graph = audit["graph"]["exact"]
        self.assertEqual(graph["offdiagonal_edges"], 15)
        self.assertFalse(graph["has_multineuron_cycle"])
        self.assertEqual(graph["two_node_simple_cycles"], 0)
        self.assertEqual(graph["three_node_simple_cycles"], 0)
        self.assertTrue(all(row["absolute_weight_trace"] == 0 for row in audit["weight"]["closed_walks"]["by_length"]))

    def test_directed_ring_has_one_cycle_but_repeated_closed_walks(self):
        weight = torch.zeros(3, 3, dtype=torch.float64)
        weight[1, 0], weight[2, 1], weight[0, 2] = 2., -3., .5
        audit = analyze_weight(weight)
        graph = audit["graph"]["exact"]
        self.assertEqual(graph["scc_sizes"], [3])
        self.assertEqual(graph["minimum_multineuron_cycle_length"], 3)
        self.assertEqual(graph["simple_cycle_lengths_guaranteed"], [3])
        self.assertEqual(graph["three_node_simple_cycles"], 1)
        self.assertEqual(graph["multineuron_simple_cycle_count_lower_bound"], 1)
        self.assertFalse(graph["at_least_two_distinct_multineuron_cycles"])
        self.assertAlmostEqual(walk(audit["offdiagonal"], 3)["signed_trace"], -9.)
        self.assertAlmostEqual(walk(audit["offdiagonal"], 3)["absolute_weight_trace"], 9.)
        self.assertAlmostEqual(walk(audit["offdiagonal"], 6)["signed_trace"], 27.)
        self.assertEqual(walk(audit["offdiagonal"], 6)["signed_to_absolute_ratio"], 1.)
        for length in (2, 4, 5, 7, 8):
            self.assertEqual(walk(audit["offdiagonal"], length)["absolute_weight_trace"], 0.)
        self.assertAlmostEqual(audit["offdiagonal"]["spectral_radius"], 3**(1/3))

    def test_long_ring_scc_girth_and_complete_graph_length_guarantees(self):
        ring = torch.roll(torch.eye(5, dtype=torch.float64), shifts=1, dims=0)
        graph = analyze_weight(ring)["graph"]["exact"]
        self.assertEqual(graph["minimum_multineuron_cycle_length"], 5)
        self.assertEqual(graph["simple_cycle_lengths_guaranteed"], [5])
        self.assertFalse(graph["simple_cycle_lengths_guaranteed_are_exhaustive"])
        self.assertEqual(graph["multineuron_simple_cycle_count_lower_bound"], 1)
        complete = analyze_weight(torch.ones(5, 5, dtype=torch.float64))["graph"]["exact"]
        self.assertTrue(complete["complete_offdiagonal_support"])
        self.assertEqual(complete["simple_cycle_lengths_guaranteed"], [2, 3, 4, 5])
        self.assertTrue(complete["simple_cycle_lengths_guaranteed_are_exhaustive"])
        self.assertEqual(complete["two_node_simple_cycles"], 10)
        self.assertEqual(complete["three_node_simple_cycles"], 20)
        self.assertTrue(complete["at_least_two_distinct_multineuron_cycles"])

    def test_signed_cancellation_is_not_abs_of_signed_trace(self):
        # Two disjoint two-neuron loops have opposite product signs.
        weight = torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.],
                               [0., 0., 0., -1.], [0., 0., 1., 0.]], dtype=torch.float64)
        audit = analyze_weight(weight)
        row = walk(audit["offdiagonal"], 2)
        self.assertAlmostEqual(row["signed_trace"], 0.)
        self.assertAlmostEqual(row["absolute_weight_trace"], 4.)
        self.assertAlmostEqual(row["cancellation_fraction"], 1.)
        self.assertAlmostEqual(row["normalized_absolute_weight_per_neuron"], 1.)
        self.assertEqual(audit["graph"]["exact"]["multineuron_cyclic_scc_sizes"], [2, 2])
        self.assertAlmostEqual(audit["offdiagonal"]["reciprocity"]["weight_transpose_cosine"], 0.)
        self.assertAlmostEqual(audit["offdiagonal"]["reciprocity"]["absolute_weight_transpose_cosine"], 1.)

    def test_self_edges_are_separate_and_thresholds_do_not_replace_exact_support(self):
        weight = torch.tensor([[2., 1e-5], [1., -2.]], dtype=torch.float64)
        audit = analyze_weight(weight, thresholds=(.1, .1, 0.))
        exact = audit["graph"]["exact"]
        thresholded = audit["graph"]["diagnostic_thresholds"]
        self.assertEqual(exact["self_loop_count"], 2)
        self.assertTrue(exact["has_multineuron_cycle"])
        self.assertEqual(len(thresholded), 1)
        self.assertFalse(thresholded[0]["has_multineuron_cycle"])
        self.assertTrue(thresholded[0]["has_any_cycle"])
        diagonal = analyze_weight(torch.diag(torch.tensor([2., -2.])))
        self.assertTrue(diagonal["graph"]["exact"]["has_any_cycle"])
        self.assertFalse(diagonal["graph"]["exact"]["has_multineuron_cycle"])
        self.assertEqual(walk(diagonal["offdiagonal"], 2)["absolute_weight_trace"], 0.)
        self.assertAlmostEqual(walk(diagonal["weight"], 2)["absolute_weight_trace"], 8.)

    def test_state_jacobian_matches_autograd_and_scales_source_columns(self):
        weight = torch.tensor([[.2, -.7, .1], [.4, -.2, .5], [-.3, .8, .15]], dtype=torch.float64)
        hidden = torch.tensor([.2, -.4, .8], dtype=torch.float64)
        derivative = 1-hidden.tanh().square()
        expected_recurrent = weight * derivative[None, :]
        expected_jacobian = torch.autograd.functional.jacobian(lambda h: -h + weight @ h.tanh(), hidden)
        torch.testing.assert_close(expected_jacobian, expected_recurrent-torch.eye(3, dtype=torch.float64))
        result = analyze_states(weight, hidden)["samples"][0]
        self.assertNotIn("closed_walks", result["jacobian"])
        self.assertAlmostEqual(result["recurrent"]["frobenius_norm"], float(expected_recurrent.norm()))
        self.assertAlmostEqual(result["jacobian"]["operator_norm_2"], float(torch.linalg.matrix_norm(expected_jacobian, ord=2)))
        self.assertAlmostEqual(result["jacobian"]["spectral_abscissa"], float(torch.linalg.eigvals(expected_jacobian).real.max()))
        for length in range(2, 9):
            self.assertAlmostEqual(walk(result["recurrent"], length)["signed_trace"],
                                   float(torch.trace(torch.linalg.matrix_power(expected_recurrent, length))))

    def test_saturation_and_zero_state_reference(self):
        weight = torch.tensor([[.3, .4], [-.2, .5]], dtype=torch.float64)
        audit = analyze_states(weight, torch.tensor([[0., 0.], [30., -30.]], dtype=torch.float64))
        first, saturated = audit["samples"]
        self.assertEqual(first["saturation"]["derivative_mean"], 1.)
        self.assertEqual(first["effective_gain"]["offdiagonal_frobenius_ratio"], 1.)
        self.assertEqual(first["recurrent"], analyze_weight(weight)["weight"])
        self.assertEqual(saturated["saturation"]["numerically_zero_derivative_fraction"], 1.)
        self.assertEqual(saturated["effective_gain"]["recurrent_operator_ratio"], 0.)
        self.assertEqual(saturated["recurrent"]["spectral_radius"], 0.)
        self.assertEqual(saturated["jacobian"]["spectral_radius"], 1.)
        self.assertEqual(saturated["jacobian"]["spectral_abscissa"], -1.)
        self.assertEqual(audit["aggregate"]["saturation"]["derivative_mean"]["mean"], .5)

    def test_permutation_invariance_for_weight_and_state_summaries(self):
        weight = torch.tensor([[.2, -.7, 0., .1], [.4, -.2, .5, 0.],
                               [0., .8, .15, .6], [-.2, 0., .3, -.1]], dtype=torch.float64)
        hidden = torch.tensor([[[.2, -.4, .8, .1]], [[-1., .3, .4, .8]]], dtype=torch.float64)
        permutation = torch.tensor([2, 0, 3, 1])
        moved = weight[permutation][:, permutation]
        assert_nested_close(self, analyze_weight(weight), analyze_weight(moved))
        assert_nested_close(self, analyze_states(weight, hidden), analyze_states(moved, hidden[..., permutation]))

    def test_zero_width_one_inputs_json_finite_read_only_and_rng(self):
        weight = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
        hidden = torch.zeros(2, 1, dtype=torch.float32, requires_grad=True)
        before_weight, before_hidden, rng = weight.detach().clone(), hidden.detach().clone(), torch.get_rng_state().clone()
        for audit in (analyze_weight(weight), analyze_states(weight, hidden)):
            json.dumps(audit, allow_nan=False)
            self.assertEqual(audit["dtype"], "float64")
            self.assertEqual(audit["device"], "cpu")
        audit = analyze_states(weight, hidden)
        self.assertIsNone(audit["samples"][0]["effective_gain"]["recurrent_frobenius_ratio"])
        self.assertEqual(audit["aggregate"]["effective_gain"]["recurrent_frobenius_ratio"]["count"], 0)
        self.assertTrue(torch.equal(weight, before_weight))
        self.assertTrue(torch.equal(hidden, before_hidden))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(weight.grad)
        self.assertIsNone(hidden.grad)

    def test_invalid_inputs_and_walk_length_selection(self):
        for weight in ([], [[1., 2.]], [[math.nan]], [[math.inf]], [[1j]]):
            with self.assertRaises(ValueError):
                analyze_weight(weight)
        for thresholds in ((-.1,), (math.nan,), (math.inf,)):
            with self.assertRaises(ValueError):
                analyze_weight([[1.]], thresholds=thresholds)
        for states in ([], [[1., 2.]], [[math.inf]]):
            with self.assertRaises(ValueError):
                analyze_states([[1.]], states)
        for low, high in ((1, 8), (2, 1), (2., 8), (True, 8)):
            with self.assertRaises(ValueError):
                analyze_weight([[1.]], k_min=low, k_max=high)
        for cutoff in (0., 1., math.nan):
            with self.assertRaises(ValueError):
                analyze_states([[1.]], [[1.]], saturation_derivative_threshold=cutoff)
        audit = analyze_weight([[1.]], k_min=4, k_max=5)
        self.assertEqual([row["length"] for row in audit["weight"]["closed_walks"]["by_length"]], [4, 5])

    def test_python_float_precision_and_raw_moment_overflow(self):
        value = .1234567890123456
        audit = analyze_weight([[value]])
        self.assertEqual(audit["weight"]["operator_norm_2"], value)
        weight = [[0., 1e50], [1e50, 0.]]
        audit = analyze_weight(weight)
        row = walk(audit["offdiagonal"], 8)
        self.assertIsNone(row["signed_trace"])
        self.assertIsNone(row["absolute_weight_trace"])
        self.assertEqual(row["normalized_signed_per_neuron"], 1.)
        self.assertEqual(row["normalized_absolute_weight_per_neuron"], 1.)
        json.dumps(audit, allow_nan=False)
        states = analyze_states(weight, [[0., 0.], [.1, .1]])
        json.dumps(states, allow_nan=False)
        self.assertEqual(states["aggregate"]["offdiagonal"]["closed_walks_by_length"][-1]["signed_trace"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
