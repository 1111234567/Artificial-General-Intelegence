"""Read-only graph and instantaneous linearization diagnostics for tanh ODEs.

For dh/dt = -h + W tanh(h), W[i,j] sends neuron j to neuron i. At one
sampled state, D = diag(1-tanh(h)^2), R = W D, K = offdiag(R), and J = -I+R.
Graph cycles establish possible structural feedback. tr(M^k) is a weighted
sum of rooted closed walks, which may revisit vertices and repeat shorter
cycles; it does not count simple cycles. Frozen-state powers are neither a
Heun transition derivative nor an autonomous multi-tick sensitivity measure.

Only PyTorch (already used by the model) is required. Inputs are detached,
copied to float64 CPU, and never changed. No model, checkpoint, file, or RNG
is read or written by these functions.
"""
from __future__ import annotations

from collections import deque
import math

import torch


DEFAULT_THRESHOLDS = (0.0, 1e-3, 1e-2, 1e-1)
_MATRIX_METRICS = ("spectral_radius", "spectral_abscissa", "operator_norm_2", "frobenius_norm")
_WALK_METRICS = ("signed_trace", "absolute_weight_trace", "normalized_signed_per_neuron",
                 "normalized_absolute_weight_per_neuron", "signed_to_absolute_ratio", "cancellation_fraction")


def _real_tensor(value, name):
    try:
        tensor = torch.as_tensor(value)
        if tensor.is_complex() or tensor.layout != torch.strided:
            raise ValueError(f"{name} must be a real dense tensor")
        # Re-read array-like input with the requested dtype: converting an
        # inferred float32 temporary would lose Python-float precision first.
        tensor = torch.as_tensor(value, device="cpu", dtype=torch.float64).detach().clone()
    except (TypeError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _weight(value):
    tensor = _real_tensor(value, "W")
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1] or tensor.shape[0] < 1:
        raise ValueError("W must be a nonempty square matrix")
    return tensor


def _lengths(k_min, k_max):
    if (isinstance(k_min, bool) or isinstance(k_max, bool)
            or not isinstance(k_min, int) or not isinstance(k_max, int)
            or not 2 <= k_min <= k_max):
        raise ValueError("integer walk lengths with 2 <= k_min <= k_max required")
    return range(k_min, k_max + 1)


def _thresholds(values):
    try:
        result = sorted({float(value) for value in values})
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("thresholds must be finite nonnegative numbers") from exc
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("thresholds must be finite nonnegative numbers")
    return [value for value in result if value > 0]


def _offdiagonal(matrix):
    result = matrix.clone()
    result.diagonal().zero_()
    return result


def _describe(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "population_std": None}
    mean = math.fsum(value / len(values) for value in values)
    return {"count": len(values), "mean": mean, "min": min(values), "max": max(values),
            "population_std": math.hypot(*(value / math.sqrt(len(values)) - mean / math.sqrt(len(values))
                                           for value in values))}


def _sccs(outgoing):
    """Iterative Kosaraju, avoiding a width-dependent Python recursion limit."""
    width = len(outgoing)
    incoming = [[] for _ in range(width)]
    for source, targets in enumerate(outgoing):
        for target in targets:
            incoming[target].append(source)
    seen, finish = set(), []
    for start in range(width):
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, iter(outgoing[start]))]
        while stack:
            current, targets = stack[-1]
            target = next(targets, None)
            if target is None:
                finish.append(current)
                stack.pop()
            elif target not in seen:
                seen.add(target)
                stack.append((target, iter(outgoing[target])))
    seen, components = set(), []
    for start in reversed(finish):
        if start in seen:
            continue
        seen.add(start)
        component, stack = [], [start]
        while stack:
            current = stack.pop()
            component.append(current)
            for target in incoming[current]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        components.append(component)
    return components


def _girth(outgoing, component):
    """Shortest directed cycle of length >= 2 in a nontrivial SCC."""
    members, shortest = set(component), len(component) + 1
    for start in component:
        distance, queue = {start: 0}, deque([start])
        while queue:
            current = queue.popleft()
            if distance[current] + 1 >= shortest:
                continue
            for target in outgoing[current]:
                if target not in members or target == current:
                    continue
                if target == start:
                    shortest = min(shortest, distance[current] + 1)
                    if shortest == 2:
                        return 2
                elif target not in distance:
                    distance[target] = distance[current] + 1
                    queue.append(target)
    if shortest > len(component):
        raise AssertionError("nontrivial strongly connected component lacked a cycle")
    return shortest


def _graph(support, threshold):
    width = len(support)
    diagonal = support.diagonal()
    off = support.clone()
    off.diagonal().zero_()
    # Source j is column j. A transpose would retain SCC/cycle summaries but
    # would reverse individual paths, so keep the model's actual convention.
    outgoing = [torch.nonzero(off[:, source], as_tuple=False).flatten().tolist() for source in range(width)]
    components = _sccs(outgoing)
    cyclic = [part for part in components if len(part) > 1]
    reciprocal = int((off & off.T).sum()) // 2
    integer_support = off.to(torch.int64)
    triangle_count = int(torch.trace(integer_support @ integer_support @ integer_support)) // 3
    girths, complete_sizes, guaranteed = [], [], set()
    for part in cyclic:
        shortest = _girth(outgoing, part)
        girths.append(shortest)
        guaranteed.add(shortest)
        size = len(part)
        if int(off[part][:, part].sum()) == size * (size - 1):
            complete_sizes.append(size)
            guaranteed.update(range(2, size + 1))
    if reciprocal:
        guaranteed.add(2)
    if triangle_count:
        guaranteed.add(3)
    # Two- and three-node cycle counts are exact because self edges were
    # excluded. A >=4 girth contributes one additional distinct cycle per SCC.
    lower_bound = reciprocal + triangle_count + sum(length >= 4 for length in girths)
    edges, possible = int(off.sum()), width * (width - 1)
    self_loops = int(diagonal.sum())
    return {"absolute_weight_threshold_strictly_greater_than": threshold,
            "offdiagonal_edges": edges, "possible_offdiagonal_edges": possible,
            "offdiagonal_density": edges / possible if possible else 0.0,
            "complete_offdiagonal_support": edges == possible,
            "self_loop_count": self_loops,
            "scc_count": len(components), "scc_sizes": sorted((len(part) for part in components), reverse=True),
            "strongly_connected": len(components) == 1,
            "multineuron_cyclic_scc_sizes": sorted((len(part) for part in cyclic), reverse=True),
            "cyclic_scc_count_including_self_loops": len(cyclic) + sum(
                len(part) == 1 and bool(diagonal[part[0]]) for part in components),
            "has_any_cycle": bool(cyclic or self_loops), "has_multineuron_cycle": bool(cyclic),
            "reciprocal_pairs": reciprocal, "two_node_simple_cycles": reciprocal,
            "three_node_simple_cycles": triangle_count,
            "minimum_multineuron_cycle_length": min(girths) if girths else None,
            "complete_multineuron_scc_sizes": sorted(complete_sizes, reverse=True),
            "simple_cycle_lengths_guaranteed": sorted(guaranteed),
            "simple_cycle_lengths_guaranteed_are_exhaustive": not cyclic or len(complete_sizes) == len(cyclic),
            "multineuron_simple_cycle_count_lower_bound": lower_bound,
            "at_least_two_distinct_multineuron_cycles": lower_bound >= 2}


def _reciprocity(matrix):
    off = _offdiagonal(matrix)
    support = off != 0
    edges = int(support.sum())
    pairs = int((support & support.T).sum()) // 2
    squared_norm = float(off.square().sum())
    product = off * off.T
    return {"reciprocal_pairs": pairs,
            "bidirectional_edge_fraction": 2 * pairs / edges if edges else None,
            "signed_pair_product_sum": float(product.sum()) / 2,
            "absolute_pair_product_sum": float(product.abs().sum()) / 2,
            "weight_transpose_cosine": float(product.sum()) / squared_norm if squared_norm else None,
            "absolute_weight_transpose_cosine": float(product.abs().sum()) / squared_norm if squared_norm else None}


def _rescale_trace(trace, scale, length):
    if trace == 0:
        return 0.0
    try:
        value = trace * scale**length
        if math.isfinite(value):
            return value
    except OverflowError:
        pass
    # Raw moments may overflow even though normalized moments are bounded.
    log_magnitude = math.log(abs(trace)) + length * math.log(scale)
    try:
        value = math.copysign(math.exp(log_magnitude), trace)
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _closed_walks(matrix, lengths):
    width = len(matrix)
    absolute = matrix.abs()
    scale = float(torch.linalg.matrix_norm(absolute, ord=2))
    if not math.isfinite(scale):
        raise FloatingPointError("closed-walk normalization overflowed; rescale input weights")
    # One common scale retains the signed/absolute cancellation comparison.
    divisor = scale if scale else 1.0
    signed_base, absolute_base = matrix / divisor, absolute / divisor
    signed_power, absolute_power = signed_base, absolute_base
    rows = []
    for length in range(2, lengths.stop):
        signed_power = signed_power @ signed_base
        absolute_power = absolute_power @ absolute_base
        if length < lengths.start:
            continue
        signed, mass = float(torch.trace(signed_power)), float(torch.trace(absolute_power))
        ratio = max(-1., min(1., signed / mass)) if mass else None
        rows.append({"length": length,
                     "signed_trace": _rescale_trace(signed, divisor, length),
                     "absolute_weight_trace": _rescale_trace(mass, divisor, length),
                     "normalized_signed_per_neuron": signed / width,
                     "normalized_absolute_weight_per_neuron": mass / width,
                     "signed_to_absolute_ratio": ratio,
                     "cancellation_fraction": 1-abs(ratio) if ratio is not None else None})
    return {"common_scale_absolute_operator_norm_2": scale,
            "normalization": "trace((M/s)^k)/width, s=||abs(M)||_2; use s=1 for M=0",
            "absolute_definition": "trace(abs(M)^k), summing absolute products of rooted closed walks",
            "raw_overflow_convention": "raw trace is null if not representable in float64; normalized values remain available",
            "by_length": rows}


def _matrix_summary(matrix, lengths, *, walks=True):
    eigenvalues = torch.linalg.eigvals(matrix)
    result = {"spectral_radius": float(eigenvalues.abs().max()),
              "spectral_abscissa": float(eigenvalues.real.max()),
              "operator_norm_2": float(torch.linalg.matrix_norm(matrix, ord=2)),
              "frobenius_norm": float(torch.linalg.matrix_norm(matrix, ord="fro")),
              "reciprocity": _reciprocity(matrix)}
    if any(not math.isfinite(result[key]) for key in _MATRIX_METRICS):
        raise FloatingPointError("nonfinite matrix summary; rescale input weights")
    if walks:
        result["closed_walks"] = _closed_walks(matrix, lengths)
    return result


@torch.no_grad()
def analyze_weight(W, *, thresholds=DEFAULT_THRESHOLDS, k_min=2, k_max=8):
    """Summarize exact weight support and explicitly diagnostic thresholds.

    Thresholds are absolute weight magnitudes, with edges retained iff
    abs(W[i,j]) > threshold. Exact support is always included separately even
    if zero is absent from thresholds. No thresholded matrix replaces W.
    Guaranteed simple-cycle lengths are lower-bound evidence; unless marked
    exhaustive, omitted lengths have not been ruled out.
    """
    weight = _weight(W)
    lengths, diagnostic = _lengths(k_min, k_max), _thresholds(thresholds)
    return {"format": "loops-weight-structure-v1", "width": len(weight), "dtype": "float64", "device": "cpu",
            "weight": _matrix_summary(weight, lengths),
            "offdiagonal": _matrix_summary(_offdiagonal(weight), lengths),
            "graph": {"exact": _graph(weight != 0, 0.0),
                      "diagnostic_thresholds": [_graph(weight.abs() > threshold, threshold) for threshold in diagnostic]},
            "conventions": {
                "orientation": "W[i,j] maps source j to target i",
                "graph": "exact support uses W!=0; nonzero magnitude thresholds are descriptive diagnostics only",
                "walks": "trace powers sum rooted closed walks, including repeated vertices and repeated shorter cycles",
                "simple_cycles": "only graph summaries explicitly named simple cycles or guaranteed lengths concern simple cycles",
                "limitation": "structural cycles and weight powers do not establish nonlinear autonomous multi-tick feedback, memory, or task necessity"}}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _aggregate(samples, lengths):
    result = {"saturation": {}, "effective_gain": {}, "recurrent": {}, "offdiagonal": {}, "jacobian": {}}
    for family in ("saturation", "effective_gain"):
        for key in samples[0][family]:
            result[family][key] = _describe(sample[family][key] for sample in samples)
    for family in ("recurrent", "offdiagonal", "jacobian"):
        result[family].update({key: _describe(sample[family][key] for sample in samples) for key in _MATRIX_METRICS})
        if family == "jacobian":
            continue
        result[family]["closed_walks_by_length"] = [
            {"length": length, **{key: _describe(sample[family]["closed_walks"]["by_length"][offset][key]
                                               for sample in samples) for key in _WALK_METRICS}}
            for offset, length in enumerate(lengths)]
    return result


@torch.no_grad()
def analyze_states(W, states, *, k_min=2, k_max=8, saturation_derivative_threshold=0.1):
    """Instantaneous diagnostics at supplied states, without advancing time.

    states has shape [..., width]; leading dimensions are flattened in row
    order. The derivative is evaluated literally as 1-tanh(h)^2 in float64,
    so numerical zero derivatives are reported. Saturation means derivative
    < saturation_derivative_threshold, a descriptive cutoff. J's spectral
    abscissa concerns a frozen continuous-time local linearization; its
    spectral radius is not a discrete transition gain or stability test.
    """
    weight, hidden = _weight(W), _real_tensor(states, "states")
    lengths = _lengths(k_min, k_max)
    cutoff = float(saturation_derivative_threshold)
    if not math.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError("saturation_derivative_threshold must lie strictly between zero and one")
    if hidden.ndim < 1 or hidden.shape[-1] != len(weight) or hidden.numel() == 0:
        raise ValueError("states must have nonempty shape [..., width]")
    original_shape = list(hidden.shape)
    hidden = hidden.reshape(-1, len(weight))
    baseline_off = _offdiagonal(weight)
    baseline = {"recurrent_frobenius": float(torch.linalg.matrix_norm(weight, ord="fro")),
                "offdiagonal_frobenius": float(torch.linalg.matrix_norm(baseline_off, ord="fro")),
                "recurrent_operator": float(torch.linalg.matrix_norm(weight, ord=2)),
                "offdiagonal_operator": float(torch.linalg.matrix_norm(baseline_off, ord=2))}
    identity, samples = torch.eye(len(weight), dtype=torch.float64), []
    for index, state in enumerate(hidden):
        rate = state.tanh()
        derivative = 1-rate.square()
        recurrent = weight * derivative.unsqueeze(0)
        offdiagonal = _offdiagonal(recurrent)
        recurrent_summary = _matrix_summary(recurrent, lengths)
        off_summary = _matrix_summary(offdiagonal, lengths)
        saturation = {"derivative_mean": float(derivative.mean()),
                      "derivative_rms": float(derivative.square().mean().sqrt()),
                      "derivative_min": float(derivative.min()), "derivative_max": float(derivative.max()),
                      "fraction_derivative_below_threshold": float((derivative < cutoff).double().mean()),
                      "numerically_zero_derivative_fraction": float((derivative == 0).double().mean()),
                      "mean_abs_tanh": float(rate.abs().mean())}
        gain = {"recurrent_frobenius_ratio": _ratio(recurrent_summary["frobenius_norm"], baseline["recurrent_frobenius"]),
                "offdiagonal_frobenius_ratio": _ratio(off_summary["frobenius_norm"], baseline["offdiagonal_frobenius"]),
                "recurrent_operator_ratio": _ratio(recurrent_summary["operator_norm_2"], baseline["recurrent_operator"]),
                "offdiagonal_operator_ratio": _ratio(off_summary["operator_norm_2"], baseline["offdiagonal_operator"])}
        samples.append({"sample_index": index, "saturation": saturation, "effective_gain": gain,
                        "recurrent": recurrent_summary, "offdiagonal": off_summary,
                        "jacobian": _matrix_summary(recurrent-identity, lengths, walks=False)})
    return {"format": "loops-instantaneous-states-v1", "width": len(weight), "sample_count": len(samples),
            "state_shape": original_shape, "dtype": "float64", "device": "cpu",
            "saturation_derivative_threshold": cutoff, "samples": samples, "aggregate": _aggregate(samples, lengths),
            "conventions": {
                "equations": "D=diag(1-tanh(h)^2); R=W D; K=offdiag(R); J=-I+R",
                "state_order": "leading state dimensions are flattened in row order; each sample is a frozen state",
                "effective_gain": "tanh derivative and induced matrix-norm retention relative to W or offdiag(W); ratios are null for zero denominator",
                "jacobian": "J is the instantaneous continuous-time vector-field Jacobian; spectral abscissa=max Re(eig(J)); spectral radius is not a transition gain",
                "walks": "R/K powers are weighted frozen-state closed walks, not simple-cycle counts or time-ordered Heun tangent products",
                "numerics": "literal float64 1-tanh(h)^2 may round to zero at strong saturation; the mathematical derivative is positive for every finite h",
                "sampling": "aggregates describe supplied states, not independent trained-model replicates",
                "limitation": "no state was advanced; these diagnostics alone cannot establish autonomous multi-tick feedback or task necessity"}}
