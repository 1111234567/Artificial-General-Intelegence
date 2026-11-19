"""A closed pulse graph that addresses and writes its own graph parameters."""

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np
import torch
import torch.nn.functional as F


BODY = 12
WRITER_WIDTHS = (4, 8)
LIMIT = 2.5
LEAK = 0.7


class Op(IntEnum):
    NONE = 0
    ADJUST = 1
    BIRTH = 2
    RETIRE = 3


class Status(IntEnum):
    DECLINED = 0
    APPLIED = 1
    SUPPORT_MISMATCH = 2
    CONNECTIVITY = 3
    DISABLED = 4
    WRITER_PROTECTED = 5
    UNCHANGED = 6
    NUMERIC_RANGE = 7


@dataclass(frozen=True)
class Graph:
    weights: torch.Tensor
    bias: torch.Tensor
    present: torch.Tensor

    @property
    def width(self):
        return self.bias.shape[1]

    def repeat(self, batch):
        if len(self.bias) != 1:
            raise ValueError("Only a one-lane birth graph can be copied into a batch")
        return Graph(self.weights.expand(batch, -1, -1), self.bias.expand(batch, -1),
                     self.present.expand(batch, -1, -1))

    def detach(self):
        return Graph(self.weights.detach(), self.bias.detach(), self.present)


@dataclass(frozen=True)
class Command:
    address: torch.Tensor
    op: torch.Tensor
    payload: torch.Tensor


@dataclass(frozen=True)
class Receipt:
    status: torch.Tensor
    value_before: torch.Tensor
    value_after: torch.Tensor
    present_before: torch.Tensor
    present_after: torch.Tensor


@dataclass(frozen=True)
class Proposal:
    command: Command
    receipt: Receipt
    next_graph: Graph
    address_log_prob: torch.Tensor
    op_log_prob: torch.Tensor


def spawn(seed, writers):
    """Shared body initialization across widths, with a writable spanning cycle."""
    if writers not in WRITER_WIDTHS:
        raise ValueError(f"Expected writer widths {WRITER_WIDTHS}")
    rng = np.random.default_rng(seed)
    maximum = BODY + max(WRITER_WIDTHS)
    width = BODY + writers
    present = (rng.random((maximum, maximum)) < 0.48)[:width, :width].copy()
    ids = np.arange(width)
    present[(ids + 1) % width, ids] = True
    present[10:12, :10] = True
    values = rng.normal(0.0, 0.13, (maximum, maximum))[:width, :width].astype(np.float32)
    bias = rng.normal(0.0, 0.04, maximum)[:width].astype(np.float32)
    return Graph(torch.from_numpy((values * present)[None]), torch.from_numpy(bias[None]),
                 torch.from_numpy(present[None]))


def strongly_connected(present):
    """Check one Boolean adjacency in receiver-by-sender order."""
    def visit(arcs):
        reached = np.zeros(len(arcs), dtype=bool)
        reached[0] = True
        frontier = [0]
        while frontier:
            neighbors = np.flatnonzero(arcs[frontier.pop()] & ~reached)
            reached[neighbors] = True
            frontier.extend(neighbors)
        return bool(reached.all())

    return visit(present) and visit(present.T)


def connected_after_retirement(graph, lanes, receiver, sender):
    """Check proposed single-edge removals in parallel, including on disconnected input."""
    support = graph.present[lanes].detach().cpu().numpy().copy()
    rows = np.arange(len(support))
    support[rows, receiver[lanes].cpu().numpy(), sender[lanes].cpu().numpy()] = False

    def reachable(arcs):
        seen = np.zeros(arcs.shape[:2], dtype=bool)
        seen[:, 0] = True
        for _ in range(arcs.shape[1] - 1):
            grown = seen | (arcs & seen[:, None, :]).any(axis=2)
            if np.array_equal(seen, grown):
                break
            seen = grown
        return seen.all(axis=1)

    both = reachable(support) & reachable(support.transpose(0, 2, 1))
    return torch.as_tensor(both, device=lanes.device)


def decode(address, width):
    edge = address < width * width
    receiver = torch.where(edge, address // width, address - width * width)
    sender = torch.where(edge, address % width, 0)
    return edge, receiver, sender


def read_address(graph, address, view="own"):
    """Read presence, edge/bias value, and receiver bias at an emitted address."""
    edge, receiver, sender = decode(address, graph.width)
    rows = torch.arange(len(address), device=address.device)
    b = graph.bias[rows, receiver]
    present = torch.where(edge, graph.present[rows, receiver, sender], True)
    edge_value = torch.where(present, graph.weights[rows, receiver, sender], 0.0)
    value = torch.where(edge, edge_value, b)
    if view == "own":
        return present, value, b
    if view == "blind":
        return torch.zeros_like(present), torch.zeros_like(value), torch.zeros_like(b)
    raise ValueError(f"Unknown readback view {view!r}")


def commit(graph, command, *, keep_connected=True, intervention="active", value_limit=LIMIT):
    """Validate an already emitted command, then apply it once to that address."""
    if intervention not in ("active", "disabled", "protect_writer"):
        raise ValueError(f"Unknown intervention {intervention!r}")
    address, op, payload = command.address, command.op, command.payload
    width = graph.width
    if (address.shape != graph.bias.shape[:1] or op.shape != address.shape
            or payload.shape != address.shape or (address < 0).any()
            or (address >= width * width + width).any() or (op < Op.NONE).any()
            or (op > Op.RETIRE).any() or not torch.isfinite(payload).all()):
        raise ValueError("Expected one finite, in-range command per graph lane")
    edge, receiver, sender = decode(address, width)
    present, before, _ = read_address(graph, address)
    adjust, birth, retire = op == Op.ADJUST, op == Op.BIRTH, op == Op.RETIRE
    status = torch.full_like(address, Status.SUPPORT_MISMATCH)
    status[op == Op.NONE] = Status.DECLINED
    valid = (adjust & present) | (birth & edge & ~present) | (retire & edge & present)
    if intervention == "disabled":
        status[op != Op.NONE] = Status.DISABLED
        valid = torch.zeros_like(valid)
    elif intervention == "protect_writer":
        protected = (receiver >= BODY) | (edge & (sender >= BODY))
        status[protected & (op != Op.NONE)] = Status.WRITER_PROTECTED
        valid = valid & ~protected
    if keep_connected:
        lanes = torch.where(valid & retire)[0]
        if lanes.numel():
            rejected = lanes[~connected_after_retirement(graph, lanes, receiver, sender)]
            status[rejected] = Status.CONNECTIVITY
            valid[rejected] = False
    value = torch.where(adjust, before + payload, torch.where(birth, payload, 0.0))
    if value_limit is not None:
        if not math.isfinite(value_limit) or value_limit <= 0:
            raise ValueError("A finite value limit must be positive")
        value = value.clamp(-value_limit, value_limit)
    overflowing = valid & ~value.isfinite()
    status[overflowing] = Status.NUMERIC_RANGE
    valid = valid & ~overflowing
    changed = valid & (~adjust | (value != before))
    status[valid & ~changed] = Status.UNCHANGED
    status[changed] = Status.APPLIED

    rows = torch.arange(len(address), device=address.device)
    next_present = torch.where(changed, ~retire, present)
    after = torch.where(changed, value, before)
    edge_index = torch.where(edge, address, 0).unsqueeze(1)
    bias_index = torch.where(edge, 0, receiver).unsqueeze(1)
    raw_edge = graph.weights[rows, torch.where(edge, receiver, 0), sender]
    edge_value = torch.where(edge & changed, after, raw_edge)
    bias_value = torch.where(~edge & changed, after,
                             graph.bias[rows, torch.where(edge, 0, receiver)])
    presence = torch.where(edge, next_present, graph.present[rows, 0, 0])
    weights = graph.weights.flatten(1).scatter(1, edge_index, edge_value.unsqueeze(1))
    bias = graph.bias.scatter(1, bias_index, bias_value.unsqueeze(1))
    support = graph.present.flatten(1).scatter(1, edge_index, presence.unsqueeze(1))
    return Graph(weights.reshape_as(graph.weights), bias, support.reshape_as(graph.present)), Receipt(
        status, before, after, present, next_present)


def advance(graph, hidden, stimulus, ticks):
    drive = graph.bias + stimulus
    weights = graph.weights * graph.present
    for _ in range(ticks):
        update = torch.bmm(weights, hidden.unsqueeze(2)).squeeze(2) + drive
        hidden = (1.0 - LEAK) * hidden + LEAK * torch.tanh(update)
    return hidden


def sensory(graph, left, right, side=None, taste=None):
    stimulus = torch.zeros_like(graph.bias)
    stimulus[:, :3] = left
    stimulus[:, 3:6] = right
    if side is not None:
        stimulus[:, 6] = side < 0
        stimulus[:, 7] = side > 0
    if taste is not None:
        stimulus[:, 8] = taste.relu()
        stimulus[:, 9] = (-taste).relu()
    return stimulus


def motor(graph, left, right):
    stimulus = sensory(graph, left, right)
    hidden = advance(graph, torch.zeros_like(graph.bias), stimulus, 5)
    return hidden, 2.5 * (hidden[:, 11] - hidden[:, 10])


def sample(logits, uniform, *, greedy=False):
    """Inverse CDF sampling; the score keeps gradients while the draw stops them."""
    log_probs = logits.log_softmax(dim=1)
    if greedy:
        chosen = logits.argmax(dim=1)
    else:
        cdf = log_probs.detach().exp().cumsum(dim=1)
        eligible = logits.isfinite()
        crossed = (cdf > uniform[:, None]) & eligible
        last_eligible = logits.shape[1] - 1 - eligible.flip(1).to(torch.uint8).argmax(dim=1)
        chosen = torch.where(crossed.any(dim=1), crossed.to(torch.uint8).argmax(dim=1),
                             last_eligible)
    return chosen, log_probs.gather(1, chosen[:, None])[:, 0]


def address_logits(graph, before, post):
    """Score all ordered pairs, including absent edges, and every node bias."""
    writer = post[:, BODY:]
    wide = writer.shape[1] == 8
    source_gain = writer[:, 0]
    receive_gain = writer[:, 6] if wide else writer[:, 1]
    bias_gate = writer[:, 4] if wide else writer[:, 0]
    source_score = 3.0 * before.abs() + 4.0 * source_gain[:, None] * before
    receiver_score = 4.0 * post.abs() + 6.0 * receive_gain[:, None] * post
    edge_density = F.log_softmax(receiver_score, dim=1)[:, :, None] + F.log_softmax(
        source_score, dim=1)[:, None, :]
    edge_density = edge_density + F.logsigmoid(-4.0 * bias_gate)[:, None, None]
    bias_density = F.log_softmax(receiver_score, dim=1) + F.logsigmoid(4.0 * bias_gate)[:, None]
    return torch.cat((edge_density.flatten(1), bias_density), dim=1)


def operation_logits(graph, post, address, *, view, return_hidden=False):
    """Query the chosen address through fixed sensory ports, using the same graph."""
    edge, receiver, sender = decode(address, graph.width)
    present, value, receiver_bias = read_address(graph, address, view)
    rows = torch.arange(len(address), device=address.device)
    stimulus = torch.zeros_like(graph.bias)
    stimulus[:, 0] = present
    stimulus[:, 1] = value
    stimulus[:, 2] = receiver_bias
    stimulus[rows, receiver] = stimulus[rows, receiver] + 0.4
    stimulus[rows, sender] = stimulus[rows, sender] + 0.4 * edge.to(stimulus.dtype)
    local = advance(graph, post, stimulus, 2)
    writer = local[:, BODY:]
    wide = writer.shape[1] == 8
    none = writer[:, 5] if wide else -0.5 * writer[:, 1]
    adjust, structural = writer[:, 1], writer[:, 2]
    logits = 4.0 * torch.stack((none, adjust, structural, -structural), dim=1)
    structural_addresses = op_availability(edge)
    logits = logits.masked_fill(~structural_addresses, -torch.inf)
    source = torch.where(edge, post[rows, sender], 1.0)
    baseline = (writer[:, 7] if wide else 0.5 * structural) * 0.8
    payload = 0.80 * torch.tanh(3.0 * writer[:, 3] * source + baseline)
    if return_hidden:
        return logits, payload, local
    return logits, payload


def op_availability(edge):
    """A bias is an always-present scalar; existence never filters edge logits."""
    eligible = torch.ones_like(edge, dtype=torch.bool)
    return torch.stack((eligible, eligible, edge, edge), dim=1)


def propose(graph, hidden, left, right, side, taste, draws, *, view="own",
            intervention="active", keep_connected=True, greedy=False, value_limit=LIMIT):
    contact = advance(graph, hidden, sensory(graph, left, right, side), 3)
    post = advance(graph, contact, sensory(graph, left, right, side, taste), 3)
    address, address_log_prob = sample(address_logits(graph, contact, post), draws[:, 0],
                                       greedy=greedy)
    logits, payload = operation_logits(graph, post, address, view=view)
    op, op_log_prob = sample(logits, draws[:, 1], greedy=greedy)
    command = Command(address, op, payload)
    next_graph, receipt = commit(graph, command, keep_connected=keep_connected,
                                 intervention=intervention, value_limit=value_limit)
    return Proposal(command, receipt, next_graph, address_log_prob, op_log_prob)


def reward_to_go(tastes, horizon=None):
    """Reward available from each motor decision onward, and from each later write."""
    future = tastes.flip(1).cumsum(1).flip(1)
    if horizon is None or horizon >= tastes.shape[1]:
        return future
    if horizon < 1:
        raise ValueError("A credit horizon must include at least one contact")
    return future - F.pad(future[:, horizon:], (0, horizon))


def policy_loss(motor_log_prob, address_log_prob, op_log_prob, tastes,
                motor_baseline, write_baseline, *, write_horizon=None):
    """Previous-iteration baselines remain independent of current sampled choices."""
    future = reward_to_go(tastes)
    motor_advantage = (future - motor_baseline).detach()
    write_future = reward_to_go(tastes, write_horizon) if write_horizon is not None else future
    write_advantage = (write_future[:, 1:] - write_baseline[1:]).detach()
    return -(motor_log_prob * motor_advantage
             + F.pad((address_log_prob[:, :-1] + op_log_prob[:, :-1]) * write_advantage,
                     (0, 1))).mean()
