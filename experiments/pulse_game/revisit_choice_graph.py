"""A continuous self-writing pulse graph with two candidate edits and a skip action."""

from dataclasses import dataclass
from numbers import Integral

import torch
from torch.nn import functional as F

from . import edit_receipt_graph as receipt_graph, self_param_writer as fw
from .edit_receipt_graph import State, motor, start


WITHHOLD = 2
WITHHELD = receipt_graph.WITHHELD
RECEIPT_WIDTH = receipt_graph.RECEIPT_WIDTH
CHOICE_GAIN = 4.0
ADDRESS_READ_TICKS = 2  # fw.operation_logits
FORECAST_TICKS = 2


def _indices(indices, reference):
    if isinstance(indices, Integral) or isinstance(indices, torch.Tensor) and indices.ndim == 0:
        return torch.as_tensor(indices, device=reference.device).reshape(1)
    return indices


def _fork_indices(reference, count):
    return torch.arange(len(reference), device=reference.device).repeat(count)


@dataclass(frozen=True)
class Action:
    state: State
    side: torch.Tensor
    motor_logit: torch.Tensor
    motor_log_prob: torch.Tensor

    def subset(self, indices):
        indices = _indices(indices, self.side)
        return Action(self.state.subset(indices), self.side[indices], self.motor_logit[indices],
                      self.motor_log_prob[indices])


@dataclass(frozen=True)
class Experience:
    state: State
    contact_hidden: torch.Tensor
    side: torch.Tensor
    taste: torch.Tensor
    sensed_taste: torch.Tensor
    motor_logit: torch.Tensor
    motor_log_prob: torch.Tensor

    def subset(self, indices):
        indices = _indices(indices, self.side)
        return Experience(self.state.subset(indices), self.contact_hidden[indices], self.side[indices],
                          self.taste[indices], self.sensed_taste[indices], self.motor_logit[indices],
                          self.motor_log_prob[indices])

    def fork_batch(self, count):
        return self.subset(_fork_indices(self.side, count))


@dataclass(frozen=True)
class Proposals:
    command: fw.Command
    hidden: torch.Tensor
    withheld_hidden: torch.Tensor
    address_log_prob: torch.Tensor
    op_log_prob: torch.Tensor
    benefit: torch.Tensor
    effect_prediction: torch.Tensor
    logits: torch.Tensor
    sense: torch.Tensor

    def subset(self, indices):
        indices = _indices(indices, self.logits)
        command = self.command
        return Proposals(fw.Command(command.address[indices], command.op[indices], command.payload[indices]),
                         self.hidden[indices], self.withheld_hidden[indices], self.address_log_prob[indices],
                         self.op_log_prob[indices], self.benefit[indices], self.effect_prediction[indices],
                         self.logits[indices], self.sense[indices])

    def fork_batch(self, count):
        return self.subset(_fork_indices(self.logits, count))


@dataclass(frozen=True)
class Decision:
    index: torch.Tensor
    log_prob: torch.Tensor

    def subset(self, indices):
        indices = _indices(indices, self.index)
        return Decision(self.index[indices], self.log_prob[indices])


@dataclass(frozen=True)
class CommitResult:
    state: State
    receipt: receipt_graph.Receipt
    raw_receipt: fw.Receipt
    command: fw.Command
    accepted: torch.Tensor
    gate_log_prob: torch.Tensor
    sensed_receipt: torch.Tensor


def act(state, left, right, draw=None, *, stochastic=True, side_override=None):
    """Emit a physical side choice from sensory activity before taste arrives."""
    hidden, motor_logit = motor(state, left, right)
    if side_override is not None:
        side = torch.as_tensor(side_override, device=motor_logit.device,
                               dtype=motor_logit.dtype).expand_as(motor_logit)
        if not ((side == -1) | (side == 1)).all():
            raise ValueError("A physical side must be -1 or 1")
    else:
        rightward = draw < motor_logit.detach().sigmoid() if stochastic else motor_logit > 0
        side = rightward.to(motor_logit.dtype) * 2 - 1
    return Action(State(state.graph, hidden), side, motor_logit, F.logsigmoid(side * motor_logit))


def taste(action, left, right, observed_taste, *, taste_mode="own", donor=None):
    """Advance from the selected side and its physically observed taste."""
    graph = action.state.graph
    observed = torch.as_tensor(observed_taste, device=action.side.device, dtype=action.side.dtype)
    observed = observed.expand_as(action.side)
    sensed = -observed if taste_mode == "reverse" else receipt_graph._sensed(observed, taste_mode, donor)
    contact = fw.advance(graph, action.state.hidden, fw.sensory(graph, left, right, action.side), 3)
    post = fw.advance(graph, contact, fw.sensory(graph, left, right, action.side, sensed), 3)
    return Experience(State(graph, post), contact, action.side, observed, sensed,
                      action.motor_logit, action.motor_log_prob)


def observe(state, left, right, outcome, motor_draw=None, *, stochastic=True, side_override=None,
            taste_mode="own", donor=None):
    """Environment convenience wrapper for menus with opposite tastes on the two sides."""
    action = act(state, left, right, motor_draw, stochastic=stochastic, side_override=side_override)
    return taste(action, left, right, action.side * outcome, taste_mode=taste_mode, donor=donor)


def propose(experience, draws, *, sensing="own", donor=None):
    """Sample two commands independently from one post-taste graph state."""
    graph, post = experience.state.graph, experience.state.hidden
    batch = len(post)
    if draws.shape != (batch, 2, 2):
        raise ValueError("Expected address and operation draws for each of two candidates per lane")
    # Candidate-major batching gives both proposals identical sensory timing and parameters.
    both = experience.state.subset(_fork_indices(post, 2))
    draws = draws.transpose(0, 1).reshape(2 * batch, 2)
    address_logits = fw.address_logits(graph, experience.contact_hidden, post).repeat(2, 1)
    address, address_log_prob = fw.sample(address_logits, draws[:, 0])
    op_logits, payload, inspected = fw.operation_logits(both.graph, both.hidden, address,
                                                         view="own", return_hidden=True)
    op, op_log_prob = fw.sample(op_logits, draws[:, 1])
    present, value, receiver_bias = fw.read_address(both.graph, address)
    local = receipt_graph._local_activation(inspected, address)

    def by_lane(values):
        return values.unflatten(0, (2, batch)).transpose(0, 1)

    sense = by_lane(torch.cat((present[:, None].to(value.dtype), value.tanh()[:, None],
                               receiver_bias.tanh()[:, None], local), dim=1))
    sense = receipt_graph._sensed(sense, sensing, donor)
    edge, _, _ = fw.decode(address, graph.width)
    operation = torch.stack((edge, op == fw.Op.ADJUST, op == fw.Op.BIRTH, op == fw.Op.RETIRE), dim=1)
    stimulus = torch.zeros_like(both.graph.bias)
    stimulus[:, :10] = torch.cat((sense.transpose(0, 1).flatten(0, 1), operation.to(value.dtype),
                                  payload.tanh()[:, None]), dim=1)
    predicted = fw.advance(both.graph, inspected, stimulus, 1)
    hidden = fw.advance(both.graph, predicted, stimulus, 1)
    withheld = fw.advance(graph, post, torch.zeros_like(graph.bias), ADDRESS_READ_TICKS + FORECAST_TICKS)

    effect = by_lane(predicted[:, fw.BODY:fw.BODY + 3])
    benefit = by_lane(hidden[:, fw.BODY + 3])
    # A four-node writer shares effect neurons across ticks. An eight-node writer has
    # independent activity ports for attempted-edit and withheld scores.
    select_port, withhold_port = (fw.BODY + 4, fw.BODY + 5) if graph.width > fw.BODY + 4 else (fw.BODY, fw.BODY + 1)
    logits = CHOICE_GAIN * torch.cat((by_lane(hidden[:, select_port]), withheld[:, withhold_port, None]), dim=1)
    command = fw.Command(by_lane(address), by_lane(op), by_lane(payload))
    return Proposals(command, by_lane(hidden), withheld, by_lane(address_log_prob), by_lane(op_log_prob),
                     benefit, effect, logits, sense)


def choose(proposals, draw=None, *, stochastic=True, override=None):
    """Choose one candidate or withhold based on the dedicated native policy outputs."""
    logits = proposals.logits
    if override is None:
        index, log_prob = fw.sample(logits, draw, greedy=not stochastic)
    else:
        raw = torch.as_tensor(override, device=logits.device).expand(logits.shape[0])
        if not ((raw == 0) | (raw == 1) | (raw == WITHHOLD)).all():
            raise ValueError("Selection must be candidate 0, candidate 1, or WITHHOLD (2)")
        index = raw.to(torch.long)
        log_prob = logits.new_zeros(len(logits))
    return Decision(index, log_prob)


def commit(experience, proposals, decision, *, intervention="active", keep_connected=True,
           value_limit=None, receipt_override=None, receipt_sensing="own", donor=None):
    """Execute the chosen command once, observe it, and carry the resulting graph and activity."""
    graph = experience.state.graph
    selected = decision.index != WITHHOLD
    rows = torch.arange(len(selected), device=selected.device)
    candidate = torch.where(selected, decision.index, 0)
    emitted = proposals.command
    chosen_op = emitted.op[rows, candidate]
    command = fw.Command(emitted.address[rows, candidate], torch.where(selected, chosen_op, fw.Op.NONE),
                         emitted.payload[rows, candidate])
    next_graph, raw = fw.commit(graph, command, keep_connected=keep_connected, intervention=intervention,
                                value_limit=value_limit)
    status = torch.where(selected, raw.status, WITHHELD)
    before = torch.where(selected[:, None], proposals.hidden[rows, candidate], proposals.withheld_hidden)
    after = fw.advance(next_graph, before, torch.zeros_like(graph.bias), 1)
    activation_before = receipt_graph._local_activation(before, command.address)
    activation_after = receipt_graph._local_activation(after, command.address)
    # Withhold has a status and no selected address, so it has no local activation receipt.
    receipt = receipt_graph.Receipt(status, raw.value_before, raw.value_after, raw.present_before,
                                    raw.present_after, activation_before * selected[:, None],
                                    activation_after * selected[:, None])
    if receipt_override is None:
        visible = receipt_graph._sensed(receipt.features, receipt_sensing, donor)
    else:
        visible = torch.as_tensor(receipt_override, device=after.device, dtype=after.dtype)
    if visible.shape != (len(after), RECEIPT_WIDTH):
        raise ValueError(f"Expected a receipt with {RECEIPT_WIDTH} features per graph lane")
    stimulus = torch.zeros_like(graph.bias)
    stimulus[:, :RECEIPT_WIDTH] = visible
    edge, receiver, sender = fw.decode(command.address, graph.width)
    pulse = receipt_graph.LOCAL_RECEIPT_PULSE * visible[:, receipt_graph.EditKind.APPLIED] * selected
    stimulus = stimulus.scatter_add(1, torch.stack((receiver, sender), dim=1),
                                   torch.stack((pulse, pulse * edge), dim=1))
    hidden = fw.advance(next_graph, after, stimulus, 1)
    return CommitResult(State(next_graph, hidden), receipt, raw, command, selected & (chosen_op != fw.Op.NONE),
                        decision.log_prob, visible)


def wait(state, ticks, *, max_ticks=None):
    """Advance neutral activity for each lane's elapsed ticks without sensory feedback."""
    graph, hidden = state.graph, state.hidden
    neutral = torch.zeros_like(hidden)
    durations = torch.as_tensor(ticks, device=hidden.device).expand(len(hidden))
    if durations.dtype.is_floating_point or durations.dtype == torch.bool or (durations < 0).any():
        raise ValueError("Neutral durations must be nonnegative integers")
    if isinstance(ticks, Integral) and max_ticks is None:
        return State(graph, fw.advance(graph, hidden, neutral, ticks))
    if max_ticks is None:
        max_ticks = int(durations.max())
    elif (durations > max_ticks).any():
        raise ValueError("Neutral duration exceeds max_ticks")
    # Compute the recurrence's constant terms once, matching fw.advance.
    weights = graph.weights * graph.present
    for tick in range(max_ticks):
        update = torch.bmm(weights, hidden.unsqueeze(2)).squeeze(2) + graph.bias
        advanced = (1.0 - fw.LEAK) * hidden + fw.LEAK * torch.tanh(update)
        hidden = torch.where((durations > tick)[:, None], advanced, hidden)
    return State(graph, hidden)
