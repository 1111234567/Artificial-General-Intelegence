"""Two doors, two rooms, and a locally observed change before the goal is known."""

from dataclasses import dataclass, fields, replace
from itertools import permutations, product

import numpy as np
import torch
from torch.nn import functional as F

from .fork_lesson_bit_relay import primary_tick
from .fork_lesson_native_request import NativeRequest, READOUT, Selection
from .fork_lesson_whatif import bridge_current, pick
from .fork_lesson_world import PORTS, REGION, State


FLIP, STAY, UNCHANGED, TIE = range(4)
CASE_NAMES = ("flip", "stay", "unchanged", "tie")
GOAL_ORDERS = tuple(permutations((-1., 0., 1.)))
DOOR_PORTS = slice(0, 2)
ROOM_PORTS = slice(2, 4)
ENDPOINT_PORTS = slice(4, 7)
GOAL_PORT = 7
REWARD_PORTS = slice(8, 10)
SHADOW_MOTOR = slice(10, 12)
MID_READOUT = 15
VISIT_TICKS = 6
INSPECTION_TICKS = 6
HISTORY_TICKS = 2 * VISIT_TICKS + INSPECTION_TICKS
GOAL_TICKS = 3
PREPARATION_TICKS = 3
MID_TICK = 3
FORECAST_TICK = 5
TRIAL_TICKS = 7
COMMIT_TICKS = 8
PANEL_REPEATS = 2 * 2 * 3 * len(GOAL_ORDERS) * 2


class RerouteGraph(NativeRequest):
    """Use the two original motor nodes and two original primary forecast nodes."""


@dataclass(frozen=True)
class Episode:
    door_rooms: torch.Tensor
    prior_endpoints: torch.Tensor
    inspected_room: torch.Tensor
    observed_endpoint: torch.Tensor
    visit_order: torch.Tensor
    goals: torch.Tensor

    def __len__(self):
        return len(self.inspected_room)

    def __getitem__(self, index):
        return Episode(*(getattr(self, field.name)[index] for field in fields(self)))


@dataclass(frozen=True)
class Walk:
    door: torch.Tensor | None
    room: torch.Tensor
    endpoint: torch.Tensor


@dataclass(frozen=True)
class History:
    visits: tuple[Walk, ...]
    inspection: Walk
    inputs: torch.Tensor


@dataclass(frozen=True)
class Outcome:
    door: torch.Tensor
    room: torch.Tensor
    endpoint: torch.Tensor
    paid: torch.Tensor


@dataclass(frozen=True)
class Reference:
    rooms: torch.Tensor
    endpoints: torch.Tensor
    payoffs: torch.Tensor
    prior_payoffs: torch.Tensor


@dataclass(frozen=True)
class Trial:
    final: State
    forecast: torch.Tensor
    mid: torch.Tensor
    mid_activity: torch.Tensor
    pulse: torch.Tensor
    first_primary: torch.Tensor
    returns: torch.Tensor
    activities: torch.Tensor
    states: tuple[State, ...]


def _episode(roles, room, room_side, inspect_first, update, goal, device):
    """Assign physical identities to the three endpoint roles and both room roles."""
    rows = np.arange(len(room))
    prior = np.empty((len(room), 2), dtype=np.int64)
    prior[rows, room], prior[rows, 1 - room] = roles[:, 0], roles[:, 1]
    door_rooms = np.empty_like(prior)
    door_rooms[rows, room_side], door_rooms[rows, 1 - room_side] = room, 1 - room
    first = np.where(inspect_first, room_side, 1 - room_side)

    def as_long(value):
        return torch.as_tensor(value, dtype=torch.long, device=device)

    return Episode(as_long(door_rooms), as_long(prior), as_long(room), as_long(roles[rows, update]),
                   as_long(np.stack((first, 1 - first), axis=1)),
                   torch.as_tensor(np.asarray(GOAL_ORDERS)[goal], dtype=torch.float32, device=device))


def scenes(seed, count, *, device="cpu", easy=False):
    """Generate independent physical roles and goal assignments for real single-door episodes."""
    rng = np.random.default_rng(seed)
    roles = rng.permuted(np.tile(np.arange(3), (count, 1)), axis=1)
    room, room_side, inspect_first = rng.integers(2, size=(3, count))
    # Index 0 retains the old exit, 1 joins the other route, 2 reaches a third endpoint.
    weights = (.75, .125, .125) if easy else (.25, .25, .5)
    update = rng.choice(3, count, p=weights)
    goal = rng.integers(len(GOAL_ORDERS), size=count)
    return _episode(roles, room, room_side, inspect_first, update, goal, device)


def panel(seed, roots, *, device="cpu"):
    """Cross each history with all goals and both assignments of physical doors."""
    if roots < 1:
        raise ValueError("An evaluation panel needs at least one root")
    rng = np.random.default_rng(seed)
    roles = rng.permuted(np.tile(np.arange(3), (roots, 1)), axis=1)
    levels = tuple(product(range(2), range(2), range(3), range(len(GOAL_ORDERS)), range(2)))
    room, inspect_first, update, goal, room_side = np.tile(np.asarray(levels), (roots, 1)).T
    episode = _episode(np.repeat(roles, PANEL_REPEATS, axis=0), room, room_side, inspect_first, update, goal, device)
    truth = oracle(episode)
    changes_best = truth.payoffs.argmax(1) != truth.prior_payoffs.argmax(1)
    case = torch.where(truth.endpoints[:, 0] == truth.endpoints[:, 1], TIE,
                       torch.where(torch.as_tensor(update == 0, device=device), UNCHANGED,
                                   torch.where(changes_best, FLIP, STAY)))
    return episode, case, torch.arange(roots, device=device).repeat_interleave(PANEL_REPEATS)


def swap_room_labels(e):
    """Exchange visible room identities while retaining every door's endpoints and payoffs."""
    return replace(e, door_rooms=1 - e.door_rooms, prior_endpoints=e.prior_endpoints.flip(1),
                   inspected_room=1 - e.inspected_room)


def _by_row(table, index):
    return table.gather(1, index[:, None])[:, 0]


def _current_endpoints(e):
    return e.prior_endpoints.scatter(1, e.inspected_room[:, None], e.observed_endpoint[:, None])


def oracle(e):
    """Compute references outside the physical rollout for verification and test stratification."""
    prior = e.prior_endpoints.gather(1, e.door_rooms)
    endpoint = torch.where(e.door_rooms == e.inspected_room[:, None], e.observed_endpoint[:, None], prior)
    return Reference(e.door_rooms, endpoint, e.goals.gather(1, endpoint), e.goals.gather(1, prior))


def _senses(state, *, door=None, room=None, endpoint=None, goal=None, paid=None):
    pulse = state.h.new_zeros((len(state.h), PORTS))
    if door is not None:
        pulse[:, DOOR_PORTS] = F.one_hot(door, num_classes=2)
    if room is not None:
        pulse[:, ROOM_PORTS] = F.one_hot(room, num_classes=2)
    if endpoint is not None:
        pulse[:, ENDPOINT_PORTS] = F.one_hot(endpoint, num_classes=3)
    if goal is not None:
        pulse[:, ENDPOINT_PORTS] = goal
        pulse[:, GOAL_PORT] = 1.
    if paid is not None:
        pulse[:, REWARD_PORTS] = torch.stack((paid.relu(), (-paid).relu()), dim=1)
    return pulse


def history(model, e, *, last_only=False, return_receipts=False):
    """Actually cross both initial doors, then visit just the inspected room from inside."""
    state = model.start(len(e))
    zero = _senses(state)
    visits, inputs = [], []

    def advance(pulse, ticks):
        nonlocal state
        state = model.advance(state, pulse, ticks)
        if return_receipts:
            inputs.extend([pulse] * ticks)

    if last_only:
        advance(zero, 2 * VISIT_TICKS)
    else:
        for door in e.visit_order.unbind(1):
            advance(zero, 1)
            advance(_senses(state, door=door), 1)
            room = _by_row(e.door_rooms, door)
            advance(_senses(state, door=door, room=room), 2)
            endpoint = _by_row(e.prior_endpoints, room)
            advance(_senses(state, room=room, endpoint=endpoint), 2)
            visits.append(Walk(door, room, endpoint))

    # The inspection starts inside this room. It never reveals a new door crossing.
    advance(zero, 1)
    room = e.inspected_room
    advance(_senses(state, room=room), 2)
    endpoint = _by_row(_current_endpoints(e), room)
    advance(_senses(state, room=room, endpoint=endpoint), 2)
    advance(zero, 1)
    if return_receipts:
        return state, History(tuple(visits), Walk(None, room, endpoint), torch.stack(inputs))
    return state


def prepare(model, e, shared=None):
    """Reveal the goal only at the common start, after both visits and the local inspection."""
    state = history(model, e) if shared is None else shared
    state = model.advance(state, _senses(state, goal=e.goals), GOAL_TICKS)
    return model.advance(state, _senses(state), PREPARATION_TICKS)


def select(model, state, draws, exploration, forced=None, *, greedy=False):
    logit = model.logit(state)
    side, probability, score = pick(logit, draws, exploration, forced, greedy=greedy)
    return Selection(side, probability, score, logit)


def _trial(states, returns, pulse):
    activities = torch.stack([state.h for state in states])
    return Trial(states[-1], activities[FORECAST_TICK - 1, :, READOUT], activities[MID_TICK - 1, :, MID_READOUT],
                 activities[MID_TICK - 1, :, REGION:], pulse, activities[0, :, :REGION], torch.stack(returns),
                 activities, tuple(states))


def try_side(model, prepared, query, *, mode="natural", feedback=None, mid_override=None,
             mid_primary_room=None):
    """Drive the replica's motor sites on tick one and run the original body in silence."""
    if mode not in ("natural", "blank", "cut_return", "isolate_return"):
        raise ValueError(f"Unknown trial mode {mode!r}")
    if feedback is not None and feedback.shape != (TRIAL_TICKS, len(prepared.h), REGION):
        raise ValueError("Returning current must be indexed by trial tick, episode, and primary neuron")
    if mid_override is not None and mid_override.shape != prepared.h[:, REGION:].shape:
        raise ValueError("The midpoint replacement needs just the replica's activity")
    if mid_primary_room is not None and mid_primary_room.shape != prepared.h[:, MID_READOUT].shape:
        raise ValueError("The primary room replacement needs one activity value per episode")
    pulse = model.probe_current(query)
    if mode == "blank":
        pulse = torch.zeros_like(pulse)
    state = prepared
    zero = _senses(state)
    states, returns = [], []
    for tick in range(TRIAL_TICKS):
        outward = None
        if tick == 0:
            outward = bridge_current(model.return_weight * model.mask[REGION:, :REGION],
                                     state.fast[:, REGION:, :REGION], state.h[:, :REGION])
            outward = outward + F.pad(pulse, (SHADOW_MOTOR.start, REGION - SHADOW_MOTOR.stop))
        state, back = model.tick(state, zero, outbound_override=outward,
                                 feedback_override=None if feedback is None else feedback[tick],
                                 disable_feedback=mode in ("cut_return", "isolate_return"),
                                 adoption_off=mode == "isolate_return")
        if tick == MID_TICK - 1:
            if mid_override is not None:
                state = replace(state, h=torch.cat((state.h[:, :REGION], mid_override), dim=1))
            if mid_primary_room is not None:
                activity = state.h.clone()
                activity[:, MID_READOUT] = mid_primary_room
                state = replace(state, h=activity)
        states.append(state)
        returns.append(back)
    return _trial(states, returns, pulse)


def fixed_side(model, prepared, blank, template):
    """Keep the replica on its blank trajectory and add an independent side template to the return."""
    if template.shape == (TRIAL_TICKS, REGION):
        template = template[:, None, :]
    if template.shape not in ((TRIAL_TICKS, 1, REGION), blank.returns.shape):
        raise ValueError("The side template must match the trial ticks and primary region")
    state = prepared
    received = blank.returns + template
    states = []
    for blank_state, current in zip(blank.states, received, strict=True):
        state = primary_tick(model, state, blank_state, current)
        states.append(state)
    return _trial(states, received.unbind(0), blank.pulse)


def settle(model, state, e, decision, *, return_state=False):
    """Walk only the committed door and advance the graph through its observed room and payoff."""
    door = decision.side
    state = model.advance(state, _senses(state, door=door), 1)
    room = _by_row(e.door_rooms, door)
    state = model.advance(state, _senses(state, room=room), 2)
    state = model.advance(state, _senses(state), 3)
    endpoint = _by_row(_current_endpoints(e), room)
    paid = _by_row(e.goals, endpoint)
    state = model.advance(state, _senses(state, room=room, endpoint=endpoint, paid=paid), 2)
    outcome = Outcome(door, room, endpoint, paid)
    return (state, outcome) if return_state else outcome
