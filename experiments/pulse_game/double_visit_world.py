"""Two ordered visits to room exits, with the goal revealed only afterward."""

from dataclasses import dataclass, fields, replace
from itertools import permutations, product

import numpy as np
import torch

from . import reroute_world as reroute
from .reroute_world import (
    CASE_NAMES,
    COMMIT_TICKS,
    DOOR_PORTS,
    ENDPOINT_PORTS,
    FLIP,
    FORECAST_TICK,
    GOAL_ORDERS,
    GOAL_PORT,
    GOAL_TICKS,
    INSPECTION_TICKS,
    MID_READOUT,
    MID_TICK,
    PORTS,
    PREPARATION_TICKS,
    READOUT,
    REGION,
    REWARD_PORTS,
    ROOM_PORTS,
    SHADOW_MOTOR,
    STAY,
    TIE,
    TRIAL_TICKS,
    UNCHANGED,
    VISIT_TICKS,
    Outcome,
    Reference,
    State,
    Walk,
    select,
)
from .reroute_world import _by_row, _senses


HISTORY_TICKS = 2 * VISIT_TICKS + 2 * INSPECTION_TICKS
ENDPOINT_READOUT = slice(16, 19)
ENDPOINT_TICK = FORECAST_TICK
ENDPOINT_GAIN = 4.
OLD_MAPS = tuple(permutations(range(3)))
EVEN_MAP_IDS = tuple(
    index for index, roles in enumerate(OLD_MAPS)
    if sum(roles[i] > roles[j] for i, j in ((0, 1), (0, 2), (1, 2))) % 2 == 0
)
senses = _senses


class DoubleVisitGraph(reroute.RerouteGraph):
    """Use the original graph and report activities already present in the primary region."""


@dataclass(frozen=True)
class Episode:
    door_rooms: torch.Tensor
    prior_endpoints: torch.Tensor
    inspected_rooms: torch.Tensor
    observed_endpoints: torch.Tensor
    visit_order: torch.Tensor
    goals: torch.Tensor

    def __len__(self):
        return len(self.inspected_rooms)

    def __getitem__(self, index):
        return Episode(*(getattr(self, field.name)[index] for field in fields(self)))


@dataclass(frozen=True)
class History:
    visits: tuple[Walk, ...]
    inspections: tuple[Walk, ...]
    inputs: torch.Tensor


@dataclass(frozen=True)
class Trial(reroute.Trial):
    @property
    def endpoint_activity(self):
        return self.activities[ENDPOINT_TICK - 1, :, ENDPOINT_READOUT]

    @property
    def endpoint_logits(self):
        return ENDPOINT_GAIN * self.endpoint_activity

    @property
    def endpoint_probability(self):
        return self.endpoint_logits.softmax(dim=1)


def _make_episode(map_ids, door, order, inspection, observed, goal, device):
    def index(value):
        return torch.as_tensor(value, dtype=torch.long, device=device)

    prior = np.asarray(OLD_MAPS)[map_ids, :2]
    return Episode(
        index(np.stack((door, 1 - door), axis=1)), index(prior), index(inspection), index(observed),
        index(np.stack((order, 1 - order), axis=1)),
        torch.as_tensor(np.asarray(GOAL_ORDERS)[goal], dtype=torch.float32, device=device),
    )


def scenes(seed, count, *, device="cpu", easy=False):
    """Draw training episodes using only even permutations of the entire old room map."""
    rng = np.random.default_rng(seed)
    map_ids = rng.choice(EVEN_MAP_IDS, size=count)
    door, order = rng.integers(2, size=(2, count))
    inspection = rng.integers(2, size=(count, 2))
    observed = rng.integers(3, size=(count, 2))
    if easy:
        # Each easy visit usually finds the exit still in place immediately before that visit.
        rows = np.arange(count)
        current = np.asarray(OLD_MAPS)[map_ids, :2].copy()
        retained = rng.random((count, 2)) < .625
        for rank in range(2):
            room = inspection[:, rank]
            observed[:, rank] = np.where(retained[:, rank], current[rows, room], observed[:, rank])
            current[rows, room] = observed[:, rank]
    goal = rng.integers(len(GOAL_ORDERS), size=count)
    return _make_episode(map_ids, door, order, inspection, observed, goal, device)


def panel(*, device="cpu"):
    """Enumerate each unique old graph, physical history, and late goal assignment exactly once."""
    levels = np.asarray(list(product(
        range(len(OLD_MAPS)), range(2), range(2), range(2), range(2), range(3), range(3),
        range(len(GOAL_ORDERS)),
    )))
    map_ids, door, order, first_room, second_room, first_exit, second_exit, goal = levels.T
    episode = _make_episode(
        map_ids, door, order, np.column_stack((first_room, second_room)),
        np.column_stack((first_exit, second_exit)), goal, device,
    )
    return episode, cases(episode), torch.as_tensor(map_ids, device=device)


def heldout_map(e):
    """An odd permutation of (room 0's old exit, room 1's old exit, unused exit)."""
    first, second = e.prior_endpoints.unbind(1)
    unused = 3 - first - second
    inversions = (first > second).long() + (first > unused).long() + (second > unused).long()
    return inversions.remainder(2).bool()


def _observe(current, room, endpoint):
    return current.scatter(1, room[:, None], endpoint[:, None])


def current_endpoints(e):
    """Current physical exits by room after both local observations in sequence."""
    current = e.prior_endpoints
    for room, endpoint in zip(e.inspected_rooms.unbind(1), e.observed_endpoints.unbind(1), strict=True):
        current = _observe(current, room, endpoint)
    return current


def oracle(e):
    """Compute door-indexed references outside the body's training and physical histories."""
    prior = e.prior_endpoints.gather(1, e.door_rooms)
    endpoint = current_endpoints(e).gather(1, e.door_rooms)
    return Reference(e.door_rooms, endpoint, e.goals.gather(1, endpoint), e.goals.gather(1, prior))


def cases(e):
    truth = oracle(e)
    tied = truth.endpoints[:, 0] == truth.endpoints[:, 1]
    unchanged = (truth.payoffs == truth.prior_payoffs).all(1)
    flip = truth.payoffs.argmax(1) != truth.prior_payoffs.argmax(1)
    return torch.where(tied, TIE, torch.where(unchanged, UNCHANGED, torch.where(flip, FLIP, STAY)))


def first_observation_pairs(e):
    """Match full histories except the first observed endpoint, with opposite optimal doors.

    The first column favors the early-inspected room and the second favors the late-inspected room.
    When a scene occurs more than once, keep its first index.
    """
    early, late = e.inspected_rooms.unbind(1)
    first, last = e.observed_endpoints.unbind(1)
    first_pay, last_pay = _by_row(e.goals, first), _by_row(e.goals, last)
    candidate = ((early != late) & (last_pay == 0) & (first_pay != 0)).nonzero().flatten()
    key = torch.column_stack((e.door_rooms, e.prior_endpoints, e.visit_order, e.inspected_rooms, last, e.goals))
    pairs = {}
    for row, context, payoff in zip(
        candidate.tolist(), key[candidate].int().tolist(), first_pay[candidate].tolist(), strict=True,
    ):
        pairs.setdefault(tuple(context), {}).setdefault(0 if payoff > 0 else 1, row)
    indices = [(pair[0], pair[1]) for pair in pairs.values() if len(pair) == 2]
    return torch.tensor(indices, dtype=torch.long, device=e.goals.device).reshape(-1, 2)


def swap_room_labels(e):
    """Rename physical rooms consistently while preserving each route's endpoints and payoff."""
    return replace(e, door_rooms=1 - e.door_rooms, prior_endpoints=e.prior_endpoints.flip(1),
                   inspected_rooms=1 - e.inspected_rooms)


def history(model, e, *, last_only=False, omit_first=False, return_receipts=False):
    """Take both old doors, then actually follow each local exit from inside its room."""
    state = model.start(len(e))
    zero = senses(state)
    visits, inspections, inputs = [], [], []

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
            advance(senses(state, door=door), 1)
            room = _by_row(e.door_rooms, door)
            advance(senses(state, door=door, room=room), 2)
            endpoint = _by_row(e.prior_endpoints, room)
            advance(senses(state, room=room, endpoint=endpoint), 2)
            visits.append(Walk(door, room, endpoint))

    current = e.prior_endpoints
    for rank, (room, observed) in enumerate(zip(
        e.inspected_rooms.unbind(1), e.observed_endpoints.unbind(1), strict=True,
    )):
        current = _observe(current, room, observed)
        if rank == 0 and (last_only or omit_first):
            advance(zero, INSPECTION_TICKS)
            continue
        advance(zero, 1)
        advance(senses(state, room=room), 2)
        endpoint = _by_row(current, room)
        advance(senses(state, room=room, endpoint=endpoint), 2)
        advance(zero, 1)
        inspections.append(Walk(None, room, endpoint))

    if return_receipts:
        return state, History(tuple(visits), tuple(inspections), torch.stack(inputs))
    return state


def prepare(model, e, shared=None):
    """Fork an existing history, reveal the late goal, and advance to the common start state."""
    state = history(model, e) if shared is None else shared
    return reroute.prepare(model, e, shared=state)


def _trial(receipt):
    return Trial(*(getattr(receipt, field.name) for field in fields(reroute.Trial)))


def try_side(model, prepared, query, *, mode="natural", feedback=None, mid_override=None, mid_primary_room=None):
    """Try an internal door with the original silent seven-tick dynamics."""
    return _trial(reroute.try_side(
        model, prepared, query, mode=mode, feedback=feedback, mid_override=mid_override,
        mid_primary_room=mid_primary_room,
    ))


def fixed_side(model, prepared, blank, template):
    return _trial(reroute.fixed_side(model, prepared, blank, template))


def continue_trial(model, trial, *, mid_override=None, mid_primary_room=None):
    """Replace activity at the room report and finish with silent natural dynamics."""
    states = list(trial.states[:MID_TICK])
    state = states[-1]
    if mid_override is not None or mid_primary_room is not None:
        primary, replica = state.h[:, :REGION], state.h[:, REGION:]
        if mid_primary_room is not None:
            if mid_primary_room.shape != primary[:, MID_READOUT].shape:
                raise ValueError("The primary room replacement needs one activity value per episode")
            primary = primary.clone()
            primary[:, MID_READOUT] = mid_primary_room
        if mid_override is not None:
            if mid_override.shape != replica.shape:
                raise ValueError("The midpoint replacement needs just the replica's activity")
            replica = mid_override
        state = replace(state, h=torch.cat((primary, replica), dim=1))
        states[-1] = state

    returns = list(trial.returns[:MID_TICK].unbind(0))
    zero = senses(state)
    for _ in range(MID_TICK, TRIAL_TICKS):
        state, current = model.tick(state, zero)
        states.append(state)
        returns.append(current)
    return _trial(reroute._trial(states, returns, trial.pulse))


def settle(model, state, e, decision, *, return_state=False):
    """Follow only the chosen physical door and reveal endpoint identity and signed payout on arrival."""
    door = decision.side
    state = model.advance(state, senses(state, door=door), 1)
    room = _by_row(e.door_rooms, door)
    state = model.advance(state, senses(state, room=room), 2)
    state = model.advance(state, senses(state), 3)
    endpoint = _by_row(current_endpoints(e), room)
    paid = _by_row(e.goals, endpoint)
    state = model.advance(state, senses(state, room=room, endpoint=endpoint, paid=paid), 2)
    outcome = Outcome(door, room, endpoint, paid)
    return (state, outcome) if return_state else outcome
