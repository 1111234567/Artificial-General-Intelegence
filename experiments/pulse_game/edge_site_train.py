"""Choose which existing synapse defers part of a native, signed receipt write."""

from dataclasses import dataclass, replace

import torch

from . import interleave_interference as environment
from . import local_trace_graph as net
from . import local_trace_train as storage
from . import noon_ticket


GROUP_SIZE = 4
DEFER_FRACTION = 0.75
MIN_PROPOSAL = 2e-5
CURRENT_FEATURES = (
    "change_magnitude", "weight_with_change_sign", "weight_magnitude",
    "sender_activity", "receiver_activity", "sender_receiver_activity",
    "receiver_taste_response", "sender_to_receiver_response",
)
OLD_FEATURES = (
    "old_change_magnitude", "old_change_with_proposal_sign",
    "old_sender_activity", "old_receiver_activity", "old_endpoint_interaction",
    "sender_reactivation", "receiver_reactivation",
    "old_change_with_sender_reactivation", "old_change_with_receiver_reactivation",
    "old_change_with_endpoint_reactivation",
)
FEATURES = CURRENT_FEATURES + OLD_FEATURES


@dataclass(frozen=True)
class OldTrace:
    change: torch.Tensor
    send: torch.Tensor
    receive: torch.Tensor

    def subset(self, index):
        return OldTrace(self.change[index], self.send[index], self.receive[index])


@dataclass(frozen=True)
class Candidates:
    address: torch.Tensor
    amount: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class Prefix:
    rule: object
    old: OldTrace
    before_b: net.State
    action_b: net.Action
    experience_b: net.Experience
    native_b: net.Commit
    candidates: Candidates
    before_a: net.State | None = None
    action_a: net.Action | None = None
    experience_a: net.Experience | None = None
    after_a: net.Commit | None = None


@dataclass(frozen=True)
class Selection:
    index: torch.Tensor
    log_prob: torch.Tensor
    features: torch.Tensor | None = None
    score_gradient: torch.Tensor | None = None


@dataclass(frozen=True)
class Probe:
    side: torch.Tensor
    correct: torch.Tensor
    payout: torch.Tensor
    logit: torch.Tensor


def load_parent(birth):
    cohort = "original" if birth in noon_ticket.births("original") else "independent"
    if birth not in noon_ticket.births(cohort):
        raise ValueError(f"No retention-objective parent for birth {birth}")
    return storage.load_checkpoint(noon_ticket.source(cohort), "odor", "uniform", birth, 8000)


def _contact(rule, state, vector, law, world, index):
    orient = world.orders[:, index]
    right = vector * orient[:, None]
    action = net.act(rule, state, -right, right, world.draws[:, index, 0])
    taste = (action.side * orient) * law
    return action, net.observe(rule, action, -right, right, taste)


def group_proposals(delta):
    """Group consecutive native magnitudes separately for each signed payload."""
    flat = delta.flatten(1)
    ranked = []
    for sign in (-1, 1):
        magnitude = flat * sign
        eligible = magnitude >= MIN_PROPOSAL
        width = int(eligible.sum(1).max()) // GROUP_SIZE * GROUP_SIZE
        values, addresses = magnitude.topk(width, sorted=True)
        selected = addresses.unflatten(1, (-1, GROUP_SIZE)) if width else addresses.reshape(len(flat), 0, GROUP_SIZE)
        smallest = values[:, GROUP_SIZE - 1::GROUP_SIZE]
        valid = smallest >= MIN_PROPOSAL
        amount = torch.where(valid, sign * DEFER_FRACTION * smallest, 0)
        ranked.append(Candidates(selected, amount, valid))
    return Candidates(*(torch.cat([getattr(part, key) for part in ranked], dim=1)
                        for key in ("address", "amount", "valid")))


@torch.no_grad()
def prepare_b(rule, state_after_A, old, world, *, gap=4):
    before = net.wait(rule, state_after_A, gap)
    action, observed = _contact(rule, before, world.b, world.law_b, world, 1)
    native = net.commit(rule, observed, world.draws[:, 1, 1:])
    # The native call includes two neutral ticks. Only its graph and receipt
    # are consumed by selected policies. The native successor takes no action.
    group = group_proposals(native.receipt.weight_delta)
    return Prefix(rule, old, before, action, observed, native, group)


@torch.no_grad()
def prepare(g, r, world, *, gap=4):
    lanes = len(world.a)
    if lanes % len(g.bias):
        raise ValueError("Expected a root-major world batch with equal lanes per birth")
    rep = lanes // len(g.bias)
    rule = r.expand(rep)
    initial = net.start(g, rep)
    action, observed = _contact(rule, initial, world.a, world.law_a, world, 0)
    written = net.commit(rule, observed, world.draws[:, 0, 1:])
    activity = action.state.hidden
    old = OldTrace(written.receipt.weight_delta, activity, activity)
    prefix = prepare_b(rule, written.state, old, world, gap=gap)
    return replace(prefix, before_a=initial, action_a=action, experience_a=observed, after_a=written)


def _edge(values, addresses):
    return values.flatten(1).gather(1, addresses.flatten(1)).reshape_as(addresses)


def _node(values, indices):
    return values.gather(1, indices.flatten(1)).reshape_as(indices)


def _cycle(priorities):
    # A single cycle moves all four histories while preserving their joint values.
    order = priorities.argsort(dim=-1)
    return order.scatter(-1, order, order.roll(1, dims=-1))


def _feature_components(prefix, old, permutation):
    addresses = prefix.candidates.address
    receiver, sender = addresses // net.NODES, addresses % net.NODES
    new = _edge(prefix.native_b.receipt.weight_delta, addresses)
    change_sign = prefix.candidates.amount.sign()[..., None]
    weight = _edge(prefix.before_b.graph.weights, addresses)
    activity = prefix.action_b.state.hidden
    send = _node(activity, sender)
    receive = _node(activity, receiver)
    response = _node(prefix.experience_b.feedback[:, :, 2], receiver)
    current = (40 * new.abs(), change_sign * weight, weight.abs(), send, receive,
               send * receive, response, send * response)
    if old is None:
        return current, (torch.zeros_like(new),) * len(OLD_FEATURES)
    history = _edge(old.change, addresses)
    past_send = _node(old.send, sender)
    past_receive = _node(old.receive, receiver)
    if permutation is not None:
        history, past_send, past_receive = (values.gather(-1, permutation)
                                           for values in (history, past_send, past_receive))
    signed = 40 * history * change_sign
    send_similarity = send * past_send
    receive_similarity = receive * past_receive
    remembered = (40 * history.abs(), signed, past_send, past_receive, past_send * past_receive,
                  send_similarity, receive_similarity, signed * send_similarity,
                  signed * receive_similarity, signed * send_similarity * receive_similarity)
    return current, remembered


@torch.no_grad()
def choose(prefix, weights=None, *, mode="full", draw=None, priority=None, donor=None,
           gradient=False):
    """Select one address per feasible group from local data and caller-owned randomness."""
    group = prefix.candidates
    columns = group.address.shape
    feature_matrix = None
    if draw is not None and draw.shape != group.valid.shape:
        raise ValueError("Expected one uniform variate per lane and group")
    if priority is not None and priority.shape != columns:
        raise ValueError("Expected one priority per candidate address")
    if mode == "random":
        if draw is None:
            raise ValueError("Random address choice requires caller-owned draws")
        logits = group.amount.new_zeros(columns)
    elif mode == "fixed":
        previous = _edge(prefix.old.change, group.address)
        logits = (-previous * group.amount.sign()[..., None]).clamp_min(0)
        if priority is not None:
            logits = logits + 1e-8 * priority
    elif mode in {"full", "no_old", "blank", "shuffled", "donor"}:
        if weights is None:
            raise ValueError("A fitted policy requires its coefficient row")
        if mode == "shuffled" and priority is None:
            raise ValueError("Shuffled histories require caller-owned priorities")
        if mode == "donor" and donor is None:
            raise ValueError("The donor mode requires an explicit old trace")
        old = None if mode in {"no_old", "blank"} else (donor if mode == "donor" else prefix.old)
        permutation = _cycle(priority) if mode == "shuffled" else None
        current, history = _feature_components(prefix, old, permutation)
        feature_matrix = torch.stack((*current, *history), dim=-1)
        weights = torch.as_tensor(weights, dtype=feature_matrix.dtype, device=feature_matrix.device)
        if weights.ndim == 2 and len(weights) in (1, len(group.valid)):
            weights = weights[:, None, :, None]
            logits = (feature_matrix @ weights).squeeze(-1)
        elif weights.shape == (len(FEATURES),):
            logits = feature_matrix @ weights
        else:
            raise ValueError("Expected the shared coefficient vector or one per lane")
    else:
        raise ValueError(f"Unknown edge selector {mode!r}")
    if gradient and (feature_matrix is None or draw is None):
        raise ValueError("Policy gradients require sampled fitted choices")

    log_prob = logits.log_softmax(dim=-1)
    probability = log_prob.exp()
    if draw is None or mode == "fixed":
        index = logits.argmax(dim=-1)
    else:
        index = (probability.cumsum(dim=-1)[..., :-1] <= draw[..., None]).sum(dim=-1)
    chosen = log_prob.gather(-1, index[..., None]).squeeze(-1)
    grad = None
    if gradient:
        n = len(FEATURES)
        selected = feature_matrix.gather(-2, index[..., None, None].expand(-1, -1, 1, n)).squeeze(-2)
        mean = (probability[..., None] * feature_matrix).sum(-2)
        grad = ((selected - mean) * group.valid[..., None]).sum(1)
    return Selection(index, torch.where(group.valid, chosen, 0), feature_matrix, grad)


@torch.no_grad()
def apply(prefix, selection=None, *, mode="selected"):
    """Apply the chosen numerical change, preserving native timing, bias and support."""
    realized = prefix.native_b.state
    if mode == "native":
        return realized
    group = prefix.candidates
    graph = realized.graph
    if mode == "no_numeric":
        before = prefix.before_b.graph
        weights = torch.where(before.present & graph.present, before.weights, graph.weights)
    elif mode == "selected":
        index = selection.index if isinstance(selection, Selection) else selection
        if index is None or index.shape != group.valid.shape:
            raise ValueError("Expected one option index per lane and candidate group")
        address = group.address.gather(2, index[..., None]).squeeze(-1)
        weights = graph.weights.flatten(1).scatter_add(1, address, -group.amount).reshape_as(graph.weights)
    elif mode == "uniform":
        reduction = (group.amount[..., None] / GROUP_SIZE).expand_as(group.address)
        weights = graph.weights.flatten(1).scatter_add(
            1, group.address.flatten(1), -reduction.flatten(1)).reshape_as(graph.weights)
    else:
        raise ValueError(f"Unknown B numeric treatment {mode!r}")
    ready = net.wait(prefix.rule, prefix.experience_b.state, net.PRE_WRITE_TICKS)
    return net.wait(prefix.rule, replace(ready, graph=replace(graph, weights=weights)), net.POST_WRITE_TICKS)


@torch.no_grad()
def probe(rule, state, world, *, wait=4, order=None):
    """Act in two serial revisits, read actual environmental payouts, and report A then B."""
    first = (world.draws[:, 2, 0] < 0.5).long() if order is None else torch.as_tensor(
        order, dtype=torch.long, device=state.hidden.device).expand(len(state.hidden))
    canonical = torch.stack((first, 1 - first), dim=1)
    rows = torch.arange(len(first), device=first.device)
    vectors = torch.stack((world.a, world.b), dim=1)
    law = torch.stack((world.law_a, world.law_b), dim=1)
    state = net.wait(rule, state, wait)
    sides, payouts, logits = [], [], []
    for time in (0, 1):
        address = canonical[:, time]
        index = 3 + address
        orientation = world.orders[rows, index]
        right = vectors[rows, address] * orientation[:, None]
        action = net.act(rule, state, -right, right, world.draws[rows, index, 0])
        sides.append(action.side)
        payouts.append(action.side * orientation * law[rows, address])
        logits.append(action.motor_logit)
        if time == 0:
            contact = net.observe(rule, action, -right, right, torch.zeros_like(action.side))
            state = net.wait(rule, contact.state, net.PRE_WRITE_TICKS + net.POST_WRITE_TICKS)

    def by_object(values):
        return torch.stack(values, dim=1).gather(1, canonical)

    payout = by_object(payouts)
    return Probe(by_object(sides), payout > 0, payout, by_object(logits))


def realized_delta(prefix, state):
    """Read the physical signed weight changes on edges that existed before B."""
    before = prefix.before_b.graph
    after = state.graph
    return torch.where(before.present & after.present, after.weights - before.weights, 0)


def candidates(prefix):
    return prefix.candidates
