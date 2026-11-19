"""Native prediction and gating for edits on the continuous self-writing graph."""

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

import torch
from torch.nn import functional as F

from . import self_param_writer as fw


GATE_GAIN = 4.0
LOCAL_RECEIPT_PULSE = 0.4
WITHHELD = max(fw.Status) + 1
RECEIPT_WIDTH = 10


class EditKind(IntEnum):
    NOOP = 0
    APPLIED = 1
    REJECTED = 2
    WITHHELD = 3


@dataclass(frozen=True)
class State:
    graph: fw.Graph
    hidden: torch.Tensor

    def subset(self, indices):
        if isinstance(indices, Integral):
            indices = [indices]
        graph = self.graph
        return State(fw.Graph(graph.weights[indices], graph.bias[indices], graph.present[indices]),
                     self.hidden[indices])

    def repeat(self, batch):
        return State(self.graph.repeat(batch), self.hidden.expand(batch, -1))

    def detach(self):
        return State(self.graph.detach(), self.hidden.detach())


@dataclass(frozen=True)
class Proposal:
    command: fw.Command
    hidden: torch.Tensor
    side: torch.Tensor
    taste: torch.Tensor
    motor_logit: torch.Tensor
    motor_log_prob: torch.Tensor
    address_log_prob: torch.Tensor
    op_log_prob: torch.Tensor
    gate_logit: torch.Tensor
    benefit: torch.Tensor
    effect_prediction: torch.Tensor
    sense: torch.Tensor
    sensing: str
    donor: torch.Tensor | None


@dataclass(frozen=True)
class Receipt:
    status: torch.Tensor
    value_before: torch.Tensor
    value_after: torch.Tensor
    present_before: torch.Tensor
    present_after: torch.Tensor
    activation_before: torch.Tensor
    activation_after: torch.Tensor

    @property
    def kind(self):
        kind = torch.full_like(self.status, EditKind.REJECTED)
        kind[(self.status == fw.Status.DECLINED) | (self.status == fw.Status.UNCHANGED)] = EditKind.NOOP
        kind[self.status == fw.Status.APPLIED] = EditKind.APPLIED
        kind[self.status == WITHHELD] = EditKind.WITHHELD
        return kind

    @property
    def value_delta(self):
        return self.value_after - self.value_before

    @property
    def topology_delta(self):
        return self.present_after.to(torch.int8) - self.present_before.to(torch.int8)

    @property
    def activation_delta(self):
        return self.activation_after - self.activation_before

    @property
    def effect_target(self):
        """Normalized receiver activity trend, parameter value and topology changes."""
        return torch.stack((self.activation_delta[:, 0] / 2, self.value_delta.tanh(),
                            self.topology_delta.to(self.value_before.dtype)), dim=1)

    @property
    def features(self):
        """Status one-hot, effect target, receiver before/after, and sender delta."""
        kinds = F.one_hot(self.kind, num_classes=len(EditKind)).to(self.value_before.dtype)
        return torch.cat((kinds, self.effect_target, self.activation_before[:, :1],
                          self.activation_after[:, :1], self.activation_delta[:, 1:2]), dim=1)


@dataclass(frozen=True)
class CommitResult:
    state: State
    receipt: Receipt
    accepted: torch.Tensor
    gate_log_prob: torch.Tensor
    sensed_receipt: torch.Tensor


def start(initial, batch):
    graph = initial.repeat(batch)
    return State(graph, torch.zeros_like(graph.bias))


def motor(state, left, right):
    """Compute the motor activity and right-side logit before feedback."""
    graph = state.graph
    hidden = fw.advance(graph, state.hidden, fw.sensory(graph, left, right), 5)
    return hidden, 2.5 * (hidden[:, 11] - hidden[:, 10])


def _sensed(values, mode, donor):
    if mode == "own":
        return values
    if mode in ("blind", "zero"):
        return torch.zeros_like(values)
    if mode not in ("yoked", "shuffled"):
        raise ValueError(f"Unknown sensing mode {mode!r}")
    lanes = torch.arange(len(values), device=values.device)
    if len(lanes) < 2:
        raise ValueError(f"{mode} sensing requires at least two graph lanes")
    if donor is None:
        if mode == "shuffled":
            raise ValueError("Shuffled sensing requires a matched donor permutation")
        donor = lanes ^ 1 if len(lanes) % 2 == 0 else lanes.roll(1)
    donor = torch.as_tensor(donor, device=values.device, dtype=torch.long)
    if donor.shape != lanes.shape or not torch.equal(donor.sort().values, lanes) or (donor == lanes).any():
        raise ValueError("Expected a donor permutation with no self matches")
    return values[donor].detach()


def _local_activation(hidden, address):
    edge, receiver, sender = fw.decode(address, hidden.shape[1])
    rows = torch.arange(len(address), device=address.device)
    return torch.stack((hidden[rows, receiver], torch.where(edge, hidden[rows, sender], 0.0)), dim=1)


def propose(state, left, right, outcome, draws, *, stochastic=True, sensing="own", donor=None,
            side_override=None):
    """Emit an address, op, payload and prospective effects before calling the executor.

    Draws are ordered address, operation, motor, gate. Only the selected side's
    taste enters the graph. Address and op remain sampled with a greedy motor.
    """
    graph = state.graph
    motor_hidden, motor_logit = motor(state, left, right)
    go = draws[:, 2] < motor_logit.sigmoid().detach() if stochastic else motor_logit > 0
    side = go.to(motor_logit.dtype) * 2 - 1
    if side_override is not None:
        side = torch.as_tensor(side_override, device=side.device, dtype=side.dtype).expand_as(side)
    taste = outcome * side
    contact = fw.advance(graph, motor_hidden, fw.sensory(graph, left, right, side), 3)
    post = fw.advance(graph, contact, fw.sensory(graph, left, right, side, taste), 3)
    address, address_log_prob = fw.sample(fw.address_logits(graph, contact, post), draws[:, 0])
    logits, payload, inspected = fw.operation_logits(graph, post, address, view="own", return_hidden=True)
    op, op_log_prob = fw.sample(logits, draws[:, 1])

    present, value, receiver_bias = fw.read_address(graph, address)
    local = _local_activation(inspected, address)
    sense = _sensed(torch.cat((present[:, None].to(value.dtype), value.tanh()[:, None],
                              receiver_bias.tanh()[:, None], local), dim=1), sensing, donor)
    edge, _, _ = fw.decode(address, graph.width)
    operation = torch.stack((edge, op == fw.Op.ADJUST, op == fw.Op.BIRTH, op == fw.Op.RETIRE), dim=1)
    stimulus = torch.zeros_like(graph.bias)
    stimulus[:, :10] = torch.cat((sense, operation.to(value.dtype), payload.tanh()[:, None]), dim=1)
    predicted = fw.advance(graph, inspected, stimulus, 1)
    effect = predicted[:, fw.BODY:fw.BODY + 3]
    hidden = fw.advance(graph, predicted, stimulus, 1)
    benefit = hidden[:, fw.BODY + 3]
    return Proposal(fw.Command(address, op, payload), hidden, side, taste, motor_logit,
                    F.logsigmoid(side * motor_logit), address_log_prob, op_log_prob,
                    GATE_GAIN * benefit, benefit, effect, sense, sensing, donor)


def commit(state, proposal, draws, *, gate="learned", stochastic_gate=True, accept_override=None,
           sensing=None, donor=None, intervention="active", keep_connected=True, value_limit=None,
           receipt_override=None):
    """Apply at most one permitted edit, then deliver a receipt to the new graph.

    Receipt overrides replace the ten inputs delivered after commit, leaving the
    true receipt and the graph produced by the executor intact. Gate overrides
    select a branch without changing the candidate, its prediction or the prefix
    activity. No-op proposals have no gate score.
    """
    if gate not in ("open", "learned"):
        raise ValueError(f"Unknown gate {gate!r}")
    graph = state.graph
    command = proposal.command
    wants_edit = command.op != fw.Op.NONE
    logit = proposal.gate_logit
    if accept_override is not None:
        accepted = torch.as_tensor(accept_override, device=logit.device, dtype=torch.bool).expand_as(wants_edit)
    elif gate == "open":
        accepted = wants_edit
    elif stochastic_gate:
        accepted = draws[:, 3] < logit.sigmoid().detach()
    else:
        accepted = logit > 0
    accepted = accepted & wants_edit
    gate_log_prob = torch.zeros_like(logit)
    if gate == "learned" and accept_override is None:
        gate_log_prob = torch.where(wants_edit, F.logsigmoid(torch.where(accepted, logit, -logit)), 0.0)

    op = torch.where(accepted, command.op, fw.Op.NONE)
    next_graph, result = fw.commit(graph, fw.Command(command.address, op, command.payload),
                                   keep_connected=keep_connected, intervention=intervention,
                                   value_limit=value_limit)
    status = torch.where(wants_edit & ~accepted, WITHHELD, result.status)
    # Observe successive states on the actual path, so activity can drift on no-op writes.
    before = proposal.hidden
    after = fw.advance(next_graph, before, torch.zeros_like(graph.bias), 1)
    receipt = Receipt(status, result.value_before, result.value_after, result.present_before,
                      result.present_after, _local_activation(before, command.address),
                      _local_activation(after, command.address))
    if sensing is None:
        sensing = proposal.sensing
    if donor is None and sensing == proposal.sensing:
        donor = proposal.donor
    if receipt_override is None:
        visible = _sensed(receipt.features, sensing, donor)
    else:
        visible = torch.as_tensor(receipt_override, device=after.device, dtype=after.dtype)
    if visible.shape != (len(after), RECEIPT_WIDTH):
        raise ValueError(f"Expected a receipt with {RECEIPT_WIDTH} features per graph lane")

    stimulus = torch.zeros_like(graph.bias)
    stimulus[:, :RECEIPT_WIDTH] = visible
    edge, receiver, sender = fw.decode(command.address, graph.width)
    pulse = LOCAL_RECEIPT_PULSE * visible[:, EditKind.APPLIED]
    stimulus = stimulus.scatter_add(1, torch.stack((receiver, sender), dim=1),
                                   torch.stack((pulse, pulse * edge), dim=1))
    hidden = fw.advance(next_graph, after, stimulus, 1)
    return CommitResult(State(next_graph, hidden), receipt, accepted, gate_log_prob, visible)
