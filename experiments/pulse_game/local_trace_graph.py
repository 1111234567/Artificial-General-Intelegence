"""A pulse graph whose retained node activity drives local synaptic writes."""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import self_param_writer as native
from .self_param_writer import Graph


NODES = 20
TRACE_RANK = 2
REACTION_TICKS = 5
CONTACT_TICKS = 3
TASTE_TICKS = 3
PRE_WRITE_TICKS = 4
POST_WRITE_TICKS = 2
WEIGHT_LIMIT = 1.15
BIAS_LIMIT = 1.15
MIN_WEIGHT = 1e-7
EDIT_DRAWS = 4
SHUFFLE_DRAWS = NODES * NODES + NODES


@dataclass(frozen=True)
class _Coefficients:
    trace_input: torch.Tensor
    trace_keep: torch.Tensor
    modulation: torch.Tensor
    scales: torch.Tensor
    gates: torch.Tensor
    ranking: torch.Tensor


class Rule(nn.Module):
    """Coefficients shared by nodes and edges, with independent rows for fit roots."""

    def __init__(self, seeds=0):
        super().__init__()
        seeds = [seeds] if isinstance(seeds, Integral) else list(seeds)
        if not seeds:
            raise ValueError("Expected at least one independent rule root")
        mean = _Coefficients(
            trace_input=torch.tensor([[[1.0, 0.1, 0.25, 0.0], [0.2, 0.7, -0.3, 0.0]],
                                      [[0.1, 0.9, 0.2, 0.0], [-0.4, 0.8, -0.2, 0.0]]]),
            trace_keep=torch.tensor([[1.8, 1.05], [1.25, 0.7]]),
            modulation=torch.tensor([[0.0, 0.7, 1.5, 0.0], [0.0, -0.5, 0.7, 0.0]]),
            scales=torch.tensor([-1.15, -1.15, -0.55]),
            gates=torch.tensor([[-2.7, 0.2, 0.4], [-2.6, -0.15, 0.3]]),
            ranking=torch.tensor([[0.2, 1.0], [-0.3, -1.0]]),
        )
        rngs = [np.random.default_rng(np.random.SeedSequence([int(seed), 416820])) for seed in seeds]
        for name, value in vars(mean).items():
            noise = torch.from_numpy(np.stack([rng.normal(0, 0.07, value.shape) for rng in rngs]))
            setattr(self, name, nn.Parameter(value + noise.float()))

    def expand(self, batch_size):
        """Create a differentiable root-major view for one optimizer iteration."""
        return _Coefficients(**{
            name: value.repeat_interleave(batch_size, dim=0) for name, value in self.named_parameters()
        })


@dataclass(frozen=True)
class State:
    graph: Graph
    hidden: torch.Tensor
    sender_trace: torch.Tensor
    receiver_trace: torch.Tensor

    def subset(self, indices):
        if isinstance(indices, Integral) or isinstance(indices, torch.Tensor) and indices.ndim == 0:
            indices = [int(indices)]
        graph = self.graph
        return State(Graph(graph.weights[indices], graph.bias[indices], graph.present[indices]),
                     self.hidden[indices], self.sender_trace[indices], self.receiver_trace[indices])

    def detach(self):
        return State(self.graph.detach(), self.hidden.detach(), self.sender_trace.detach(),
                     self.receiver_trace.detach())


@dataclass(frozen=True)
class Action:
    state: State
    side: torch.Tensor
    motor_logit: torch.Tensor
    motor_log_prob: torch.Tensor


@dataclass(frozen=True)
class Experience:
    state: State
    side: torch.Tensor
    taste: torch.Tensor
    feedback: torch.Tensor


@dataclass(frozen=True)
class Receipt:
    proposed_birth: torch.Tensor
    proposed_retirement: torch.Tensor
    birth: torch.Tensor
    retired: torch.Tensor
    weight_delta: torch.Tensor
    bias_delta: torch.Tensor
    birth_weight: torch.Tensor
    retired_weight: torch.Tensor
    match: "ShuffleMatch | None" = None


@dataclass(frozen=True)
class ShuffleMatch:
    reference: Receipt
    birth_relocated: torch.Tensor
    retirement_relocated: torch.Tensor
    weight_matches: torch.Tensor
    bias_matches: torch.Tensor
    retirement_matches: torch.Tensor


@dataclass(frozen=True)
class Commit:
    state: State
    receipt: Receipt
    edit_log_prob: torch.Tensor


def spawn(seed):
    """Use the same writable 20-node birth graph as the archival addressed writer."""
    if isinstance(seed, Integral):
        return native.spawn(int(seed), NODES - native.BODY)
    births = [spawn(root) for root in seed]
    if not births:
        raise ValueError("Expected at least one graph root")
    return Graph(*(torch.cat([getattr(g, name) for g in births]) for name in ("weights", "bias", "present")))


def start(graph, batch_size=None):
    """Start cold activity and traces, repeating graph roots in root-major order."""
    if graph.width != NODES:
        raise ValueError(f"Expected a {NODES}-node graph")
    if batch_size is not None:
        graph = graph.repeat(batch_size) if len(graph.bias) == 1 else Graph(*(
            value.repeat_interleave(batch_size, dim=0) for value in (graph.weights, graph.bias, graph.present)
        ))
    trace = graph.bias.new_zeros((*graph.bias.shape, TRACE_RANK))
    return State(graph, torch.zeros_like(graph.bias), trace, trace)


def swap(state, donor, components):
    """Exchange selected physical state with a shape-matched donor lane."""
    components = (components,) if isinstance(components, str) else tuple(components)
    if set(components) - {"hidden", "trace", "graph"} or state.hidden.shape != donor.hidden.shape:
        raise ValueError("Expected shape-matched lanes and hidden, trace, or graph components")
    return State(donor.graph if "graph" in components else state.graph,
                 donor.hidden if "hidden" in components else state.hidden,
                 donor.sender_trace if "trace" in components else state.sender_trace,
                 donor.receiver_trace if "trace" in components else state.receiver_trace)


def _advance(rule, state, stimulus, ticks):
    graph, hidden = state.graph, state.hidden
    trace = torch.cat((state.sender_trace, state.receiver_trace), dim=2)
    coeff = rule.trace_input.flatten(1, 2).transpose(1, 2)
    keep = rule.trace_keep.flatten(1, 2).sigmoid()[:, None, :]
    drive = graph.bias + stimulus
    weights = graph.weights * graph.present
    for _ in range(ticks):
        current = torch.bmm(weights, hidden.unsqueeze(2)).squeeze(2) + drive
        next_hidden = (1.0 - native.LEAK) * hidden + native.LEAK * current.tanh()
        local = torch.stack((hidden, next_hidden, hidden * next_hidden, torch.ones_like(hidden)), dim=2)
        sample = torch.matmul(local, coeff).tanh()
        trace = keep * trace + (1 - keep) * sample
        hidden = next_hidden
    return State(graph, hidden, *trace.split(TRACE_RANK, dim=2))


def act(rule, state, left, right, draw=None, *, stochastic=True, side_override=None):
    """Sense a menu and emit the motor choice before observing the contacted taste."""
    state = _advance(rule, state, native.sensory(state.graph, left, right), REACTION_TICKS)
    logit = 2.5 * (state.hidden[:, 11] - state.hidden[:, 10])
    if side_override is not None:
        side = torch.as_tensor(side_override, device=logit.device, dtype=logit.dtype).expand_as(logit)
        if not ((side == -1) | (side == 1)).all():
            raise ValueError("Expected physical choices of -1 or +1")
    else:
        if stochastic and draw is None:
            raise ValueError("Stochastic choices require a caller-owned motor draw")
        rightward = draw < logit.detach().sigmoid() if stochastic else logit > 0
        side = 2 * rightward.to(logit.dtype) - 1
    return Action(state, side, logit, F.logsigmoid(side * logit))


def observe(rule, action, left, right, taste):
    """Route the chosen side and the observed taste through native sensory ports."""
    taste = torch.as_tensor(taste, device=action.side.device, dtype=action.side.dtype).expand_as(action.side)
    graph, side = action.state.graph, action.side
    contact = _advance(rule, action.state, native.sensory(graph, left, right, side), CONTACT_TICKS)
    post = _advance(rule, contact, native.sensory(graph, left, right, side, taste), TASTE_TICKS)
    response = post.hidden - contact.hidden
    feedback = torch.stack((contact.hidden, post.hidden, response, torch.ones_like(response)), dim=2)
    return Experience(post, side, taste, feedback)


def wait(rule, state, ticks, *, max_ticks=None):
    """Advance native neutral activity for the elapsed ticks and hold both traces."""
    graph, hidden = state.graph, state.hidden
    neutral = torch.zeros_like(hidden)
    if isinstance(ticks, Integral) and not isinstance(ticks, (bool, np.bool_)) and max_ticks is None:
        if ticks < 0:
            raise ValueError("Neutral durations must be nonnegative integers")
        return replace(state, hidden=native.advance(graph, hidden, neutral, ticks))
    durations = torch.as_tensor(ticks, device=hidden.device).expand(len(hidden))
    if durations.dtype.is_floating_point or durations.dtype == torch.bool or (durations < 0).any():
        raise ValueError("Neutral durations must be nonnegative integers")
    if max_ticks is None:
        max_ticks = int(durations.max())
    elif (durations > max_ticks).any():
        raise ValueError("Neutral duration exceeds max_ticks")
    weights = graph.weights * graph.present
    for tick in range(max_ticks):
        current = torch.bmm(weights, hidden.unsqueeze(2)).squeeze(2) + graph.bias
        advanced = (1.0 - native.LEAK) * hidden + native.LEAK * current.tanh()
        hidden = torch.where((durations > tick)[:, None], advanced, hidden)
    return replace(state, hidden=hidden)


def _sample_edit(gate, score, available, draws):
    """Sample a Bernoulli proposal and a conditional eligible edge address."""
    eligible = available.any(dim=1)
    proposed = eligible & (draws[:, 0] < gate.detach().sigmoid())
    logits = score.masked_fill(~(available | ~eligible[:, None]), -torch.inf)
    log_probs = logits.log_softmax(dim=1)
    cdf = log_probs.detach().exp().cumsum(dim=1)
    address_draw = draws[:, 1, None].to(cdf.dtype).contiguous()
    index = torch.searchsorted(cdf, address_draw, right=True).squeeze(1)
    # Roundoff in the last CDF bucket must not create an invalid physical proposal.
    last = available.flip(1).to(torch.uint8).argmax(dim=1)
    last = available.shape[1] - 1 - last
    valid = (index < available.shape[1]) & available.gather(1, index.clamp_max(available.shape[1] - 1)[:, None]).squeeze(1)
    index = torch.where(valid, index, last)
    conditional = log_probs.gather(1, index[:, None]).squeeze(1)
    log_prob = F.logsigmoid(torch.where(proposed, gate, -gate)) + torch.where(proposed, conditional, 0.0)
    return torch.where(proposed, index, -1), torch.where(eligible, log_prob, 0.0)


def _replace_edge(values, address, replacement):
    """Differentiable per-lane replacement. A missing (-1) address is a no-op."""
    batch = len(values)
    flat = values.flatten(1)
    index = address.clamp_min(0).reshape(batch, 1)
    old = flat.gather(1, index).squeeze(1)
    new = torch.where(address >= 0, replacement, old)
    return flat.scatter(1, index, new[:, None]).reshape_as(values)


def _read_edge(values, address):
    at = values.flatten(1).gather(1, address.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(address >= 0, at, 0)


def _after_retirement(graph, retired):
    lanes = (retired >= 0).nonzero(as_tuple=True)[0]
    if len(lanes) == 0:
        return retired
    # Births are already in the support. No scaffold arc is protected a priori.
    allowed = native.connected_after_retirement(graph, lanes, retired // NODES, retired % NODES)
    return retired.scatter(0, lanes, torch.where(allowed, retired[lanes], -1))


def _nonzero(value):
    tiny = torch.copysign(torch.full_like(value, MIN_WEIGHT), value)
    return torch.where(value.abs() < MIN_WEIGHT, tiny, value)


def _edge_effects(graph, weight_delta, bias_delta, birth, retired, birth_weight):
    present = _replace_edge(graph.present, retired, torch.zeros_like(retired, dtype=torch.bool))
    present = _replace_edge(present, birth, torch.ones_like(birth, dtype=torch.bool))
    weight = _replace_edge(graph.weights + weight_delta, retired, torch.zeros_like(birth_weight))
    weight = _replace_edge(weight, birth, birth_weight)
    return Graph(weight, graph.bias + bias_delta, present)


def _actual_receipt(graph, next_graph, proposed_birth, proposed_retirement, birth, retired):
    common = graph.present & next_graph.present
    return Receipt(proposed_birth, proposed_retirement, birth, retired,
                   torch.where(common, next_graph.weights - graph.weights, 0.0),
                   next_graph.bias - graph.bias, _read_edge(next_graph.weights, birth),
                   _read_edge(graph.weights, retired))


def _bounded_update(old, delta, limit):
    lower = old.clamp_max(-limit)
    upper = old.clamp_min(limit)
    return (old + delta).clamp(min=lower, max=upper)


def _shuffle_support(present, weight, born, retired, priorities):
    """Find randomized destination changes, favoring equal-sign retirements of similar magnitude."""
    if born < 0 and retired < 0:
        return -1, -1
    original = present.flatten()
    missing = np.flatnonzero(~original)
    birth_order = missing[np.argsort(priorities[missing])]
    if born < 0:
        birth_order = [-1]
    else:
        birth_order = [*birth_order[birth_order != born], born]
    if retired < 0:
        return birth_order[0], -1
    options = np.flatnonzero(original)
    options = options[options != retired]
    flat_weight = weight.flatten()
    sign_match = np.signbit(flat_weight[options]) == np.signbit(flat_weight[retired])
    difference = np.abs(np.abs(flat_weight[options]) - abs(flat_weight[retired]))
    retire_order = options[np.lexsort((priorities[options], difference, ~sign_match))]
    for new_born in birth_order:
        amended = original.copy()
        if new_born >= 0:
            amended[new_born] = True
        for new_retired in (*retire_order, retired):
            amended[new_retired] = False
            if native.strongly_connected(amended.reshape(NODES, NODES)):
                return new_born, new_retired
            amended[new_retired] = True
    raise RuntimeError("Native structural edits had no connected replay")


def _shuffle_deltas(old, delta, targets, priorities, limit, *, nonzero):
    """Place signed additive values on permuted destinations within their numeric bounds."""
    result = np.zeros_like(old)
    sources = np.flatnonzero(delta)
    eligible = targets
    targets = np.flatnonzero(targets)
    targets = targets[np.argsort(priorities[targets], kind="stable")]
    lower = np.minimum(old, -limit)
    upper = np.maximum(old, limit)

    def feasible(index, payload):
        updated = old[index] + payload
        safe = (updated >= lower[index]) & (updated <= upper[index]) & (updated != old[index])
        return safe & (np.abs(updated) >= MIN_WEIGHT) if nonzero else safe

    # Most steps fit a complete permutation without individual bound checks.
    chosen = targets[:len(sources)]
    if len(chosen) == len(sources) and feasible(chosen, delta[sources]).all():
        result[chosen] = delta[sources]
        return result

    # Large magnitudes have fewer possible destinations and get first choice.
    sources = sources[np.argsort(-np.abs(delta[sources]), kind="stable")]
    unused = np.ones(len(targets), dtype=bool)
    for source in sources:
        payload = delta[source]
        choices = np.flatnonzero(unused & feasible(targets, payload))
        if len(choices):
            distinct = choices[targets[choices] != source]
            choice = distinct[0] if len(distinct) else choices[0]
            result[targets[choice]] = payload
            unused[choice] = False
    # A dense or saturated graph can force constrained writes back to their own locations.
    if np.count_nonzero(result) != len(sources) and not np.any(delta[~eligible]):
        return delta.copy()
    return result


def _same_deltas(reference, realized):
    left = reference.flatten(1).sort(dim=1).values
    right = realized.flatten(1).sort(dim=1).values
    return ((left.sign() == right.sign()).all(dim=1)
            & torch.isclose(left, right, rtol=2e-5, atol=1e-7).all(dim=1))


def _shuffled(graph, reference, priorities, *, keep_topology):
    expected = (len(graph.bias), SHUFFLE_DRAWS)
    if priorities is None or tuple(priorities.shape) != expected:
        raise ValueError(f"Location shuffling requires caller-owned priorities of shape {expected}")
    values = [value.detach().cpu().numpy() for value in (
        graph.present, graph.weights, graph.bias, reference.weight_delta, reference.bias_delta,
        reference.birth, reference.retired, priorities,
    )]
    birth, retired, weight_delta, bias_delta = [], [], [], []
    for present, weight, bias, dw, db, born, removed, draws in zip(*values):
        edge_draws, bias_draws = np.split(draws, [NODES * NODES])
        new_birth, new_retirement = (born, removed) if keep_topology else _shuffle_support(
            present, weight, born, removed, edge_draws)
        targets = present.flatten().copy()
        if new_retirement >= 0:
            targets[new_retirement] = False
        next_dw = _shuffle_deltas(weight.flatten(), dw.flatten(), targets, edge_draws, WEIGHT_LIMIT, nonzero=True)
        # If a shuffled retirement blocks a complete numerical match, retain the native support edit
        # and prioritize reproducing the physical dose. Location changes remain visible in the receipt.
        if new_retirement != removed and np.count_nonzero(next_dw) != np.count_nonzero(dw):
            new_birth, new_retirement = born, removed
            targets = present.flatten().copy()
            if removed >= 0:
                targets[removed] = False
            next_dw = _shuffle_deltas(weight.flatten(), dw.flatten(), targets, edge_draws, WEIGHT_LIMIT, nonzero=True)
        birth.append(new_birth)
        retired.append(new_retirement)
        weight_delta.append(next_dw.reshape(NODES, NODES))
        bias_delta.append(_shuffle_deltas(bias, db, np.ones(NODES, dtype=bool), bias_draws, BIAS_LIMIT, nonzero=False))
    birth = reference.birth.new_tensor(birth)
    retired = reference.retired.new_tensor(retired)
    next_graph = _edge_effects(graph, graph.weights.new_tensor(np.stack(weight_delta)),
                               graph.bias.new_tensor(np.stack(bias_delta)), birth, retired, reference.birth_weight)
    receipt = _actual_receipt(graph, next_graph, reference.proposed_birth, reference.proposed_retirement, birth, retired)
    match = ShuffleMatch(
        reference, birth != reference.birth, retired != reference.retired,
        _same_deltas(reference.weight_delta, receipt.weight_delta),
        _same_deltas(reference.bias_delta, receipt.bias_delta),
        (reference.retired_weight.sign() == receipt.retired_weight.sign())
        & torch.isclose(reference.retired_weight, receipt.retired_weight, rtol=2e-5, atol=1e-7),
    )
    return next_graph, replace(receipt, match=match)


def commit(rule, experience, draws, *, intervention="active", shuffle=None):
    """Attempt local feedback-driven graph writes, with a score for the sampled proposals.

    Numeric writes use endpoint traces and the receiving node's observed response.
    Birth and retirement proposals are independent. Birth is applied first and a
    retirement that would disconnect the graph leaves no effective support edit.
    The score covers the sampled latent proposals, including rejected retirements.
    """
    modes = {"active", "sham", "disabled", "shuffle", "shuffle_values", "weights_only", "bias_only",
             "weights_and_bias", "topology_only"}
    if intervention not in modes:
        raise ValueError(f"Unknown intervention {intervention!r}")
    ready = wait(rule, experience.state, PRE_WRITE_TICKS)
    graph = ready.graph
    if draws.shape != (len(graph.bias), EDIT_DRAWS):
        raise ValueError("Expected per-lane birth gate/address and retirement gate/address draws")

    modulation = torch.matmul(experience.feedback, rule.modulation.transpose(1, 2)).tanh()
    receiving = ready.receiver_trace * modulation
    local = torch.bmm(receiving, ready.sender_trace.transpose(1, 2)) / TRACE_RANK
    scales = rule.scales.sigmoid()
    weight_next = _nonzero(_bounded_update(graph.weights, 0.24 * scales[:, 0, None, None] * local, WEIGHT_LIMIT))
    weight_delta = torch.where(graph.present, weight_next - graph.weights, 0.0)
    bias_next = _bounded_update(graph.bias, 0.20 * scales[:, 1, None] * receiving.mean(dim=2), BIAS_LIMIT)
    bias_delta = bias_next - graph.bias

    # A learned rate fixes proposal sparsity, while endpoint data determines eligible locations.
    magnitude = local.abs().mean(dim=(1, 2))
    response = modulation.abs().mean(dim=(1, 2))
    gate_features = torch.stack((torch.ones_like(magnitude), magnitude, response), dim=1)
    gate = (rule.gates * gate_features[:, None, :]).sum(dim=2)
    abs_weight = graph.weights.abs()
    birth_score = rule.ranking[:, 0, 0, None, None] * local + rule.ranking[:, 0, 1, None, None] * local.abs()
    retire_score = (rule.ranking[:, 1, 0, None, None] * (local * graph.weights)
                    + rule.ranking[:, 1, 1, None, None] * abs_weight)
    present = graph.present.flatten(1)
    proposed_birth, birth_prob = _sample_edit(gate[:, 0], birth_score.flatten(1), ~present, draws[:, :2])
    proposed_retire, retire_prob = _sample_edit(gate[:, 1], retire_score.flatten(1), present, draws[:, 2:])
    birth = proposed_birth
    born_support = _replace_edge(graph.present, birth, torch.ones_like(birth, dtype=torch.bool))
    retired = _after_retirement(replace(graph, present=born_support), proposed_retire)
    birth_weight = _nonzero(0.40 * scales[:, 2] * (4.0 * _read_edge(local, birth)).tanh())

    if intervention in {"sham", "disabled"}:
        next_graph = graph
        birth = torch.full_like(birth, -1)
        retired = torch.full_like(retired, -1)
    else:
        if intervention in {"weights_only", "bias_only", "weights_and_bias"}:
            birth = torch.full_like(birth, -1)
            retired = torch.full_like(retired, -1)
        if intervention in {"bias_only", "topology_only"}:
            weight_delta = torch.zeros_like(weight_delta)
        if intervention in {"weights_only", "topology_only"}:
            bias_delta = torch.zeros_like(bias_delta)
        next_graph = _edge_effects(graph, weight_delta, bias_delta, birth, retired, birth_weight)
    receipt = _actual_receipt(graph, next_graph, proposed_birth, proposed_retire, birth, retired)
    if intervention in {"shuffle", "shuffle_values"}:
        next_graph, receipt = _shuffled(graph, receipt, shuffle, keep_topology=intervention == "shuffle_values")
    current = wait(rule, replace(ready, graph=next_graph), POST_WRITE_TICKS)
    return Commit(current, receipt, birth_prob + retire_prob)
