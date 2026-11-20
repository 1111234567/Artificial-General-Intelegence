"""Check what can be observed before an action and after a local change."""

from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from experiments.pulse_game import reroute_world as world


class RecordedGraph(world.RerouteGraph):
    def __init__(self):
        super().__init__(731)
        self.pulses = []

    def tick(self, s, x, **kwargs):
        self.pulses.append(x.clone())
        return super().tick(s, x, **kwargs)


def assert_equal_state(received, expected):
    for field in fields(world.State):
        assert torch.equal(getattr(received, field.name), getattr(expected, field.name)), field.name


@torch.no_grad()
def test_real_contact_order_goal_fork_and_local_only_history():
    torch.set_num_threads(1)
    model = RecordedGraph()
    e, _, _ = world.panel(558107, 1)
    e = e[:36]
    shared, log = world.history(model, e, return_receipts=True)
    tape = torch.stack(model.pulses)
    assert torch.equal(tape, log.inputs)
    assert len(tape) == world.HISTORY_TICKS
    assert not tape[:, :, world.GOAL_PORT:].any()
    for rank, visit in enumerate(log.visits):
        start = rank * world.VISIT_TICKS
        assert torch.equal(visit.door, e.visit_order[:, rank])
        assert torch.equal(visit.room, e.door_rooms.gather(1, visit.door[:, None]).squeeze(1))
        assert torch.equal(visit.endpoint, e.prior_endpoints.gather(1, visit.room[:, None]).squeeze(1))
        assert not tape[start + 1, :, world.ROOM_PORTS].any()
        assert not tape[start + 1:start + 4, :, world.ENDPOINT_PORTS].any()
        assert torch.equal(tape[start + 2, :, world.ROOM_PORTS].argmax(1), visit.room)
        assert torch.equal(tape[start + 4, :, world.ENDPOINT_PORTS].argmax(1), visit.endpoint)
    assert log.inspection.door is None
    assert torch.equal(log.inspection.room, e.inspected_room)
    assert torch.equal(log.inspection.endpoint, e.observed_endpoint)
    assert not tape[2 * world.VISIT_TICKS:, :, world.DOOR_PORTS].any()
    assert not tape[-1].any()

    alternate = replace(e, goals=-e.goals)
    assert_equal_state(shared, world.history(model, alternate))
    model.pulses.clear()
    ready = world.prepare(model, alternate, shared=shared)
    post_history = torch.stack(model.pulses)
    assert len(post_history) == world.GOAL_TICKS + world.PREPARATION_TICKS
    assert torch.equal(post_history[0, :, world.ENDPOINT_PORTS], alternate.goals)
    assert (post_history[:world.GOAL_TICKS, :, world.GOAL_PORT] == 1).all()
    assert not post_history[world.GOAL_TICKS:].any()
    assert not torch.equal(ready.h, world.prepare(model, e, shared).h)

    last, last_log = world.history(model, e, last_only=True, return_receipts=True)
    assert not last_log.visits
    assert not last_log.inputs[:2 * world.VISIT_TICKS].any()
    assert torch.equal(last_log.inputs[2 * world.VISIT_TICKS:], tape[2 * world.VISIT_TICKS:])
    donor = torch.arange(len(e)) ^ 1
    assert_equal_state(last, world.history(model, e[donor], last_only=True))


def test_panel_balances_door_side_and_recent_room_within_every_case():
    roots = 6
    e, case, root = world.panel(187307, roots)
    truth = world.oracle(e)
    before, after = truth.prior_payoffs.argmax(1), truth.payoffs.argmax(1)
    older = before != after
    inspected_side = (e.door_rooms == e.inspected_room[:, None]).long().argmax(1)
    last_room = e.door_rooms.gather(1, e.visit_order[:, 1:])[:, 0]
    assert torch.equal(case == world.FLIP, older & (case != world.TIE))
    assert torch.equal(case == world.UNCHANGED, torch.eq(truth.prior_payoffs, truth.payoffs).all(1))
    assert torch.equal(case == world.TIE, truth.endpoints[:, 0] == truth.endpoints[:, 1])
    assert (e.prior_endpoints[:, 0] != e.prior_endpoints[:, 1]).all()
    for k in range(roots):
        counts = case[root == k].bincount(minlength=len(world.CASE_NAMES))
        assert counts.tolist() == [16, 32, 48, 48]
        for group in (world.FLIP, world.STAY, world.UNCHANGED):
            keep = (root == k) & (case == group)
            for factor in (before, after, inspected_side, last_room):
                assert int(factor[keep].sum()) * 2 == int(keep.sum())
            assert int((after[keep] == inspected_side[keep]).sum()) * 2 == int(keep.sum())
            assert int((after[keep] == e.visit_order[keep, 1]).sum()) * 2 == int(keep.sum())
        kept_ties = (root == k) & (case == world.TIE)
        assert torch.unique(truth.payoffs[kept_ties, 0], return_counts=True)[1].tolist() == [16, 16, 16]

    donor = torch.arange(len(e)) ^ 1
    for name in ("prior_endpoints", "inspected_room", "observed_endpoint", "goals"):
        assert torch.equal(getattr(e, name), getattr(e[donor], name))
    assert torch.equal(e.door_rooms, e[donor].door_rooms.flip(1))
    assert torch.equal(e.visit_order, 1 - e[donor].visit_order)
    assert torch.equal(case, case[donor])
    assert torch.equal(truth.payoffs, truth.payoffs[donor].flip(1))

    relabeled = world.oracle(world.swap_room_labels(e))
    assert torch.equal(truth.rooms, 1 - relabeled.rooms)
    for name in ("endpoints", "payoffs", "prior_payoffs"):
        assert torch.equal(getattr(truth, name), getattr(relabeled, name))


@torch.no_grad()
def test_both_physical_routes_agree_with_the_reference_after_delayed_settlement():
    torch.set_num_threads(1)
    model = RecordedGraph()
    e, _, _ = world.panel(91307, 1)
    truth = world.oracle(e)
    state = world.prepare(model, e)
    for door in (0, 1):
        decision = world.select(model, state, None, 0., forced=door)
        model.pulses.clear()
        _, outcome = world.settle(model, state, e, decision, return_state=True)
        tape = torch.stack(model.pulses)
        assert len(tape) == world.COMMIT_TICKS
        assert (outcome.door == door).all()
        assert torch.equal(outcome.room, truth.rooms[:, door])
        assert torch.equal(outcome.endpoint, truth.endpoints[:, door])
        assert torch.equal(outcome.paid, truth.payoffs[:, door])
        assert (tape[0, :, world.DOOR_PORTS].argmax(1) == door).all()
        assert not tape[0, :, world.ROOM_PORTS].any()
        assert torch.equal(tape[1, :, world.ROOM_PORTS].argmax(1), outcome.room)
        assert not tape[:6, :, world.ENDPOINT_PORTS].any()
        assert not tape[:6, :, world.REWARD_PORTS].any()
        assert not tape[3:6].any()
        assert torch.equal(tape[6, :, world.ENDPOINT_PORTS].argmax(1), outcome.endpoint)
        assert torch.equal(tape[6, :, 8] - tape[6, :, 9], outcome.paid)


@torch.no_grad()
def test_request_isolation_return_replay_and_activity_only_replacement():
    torch.set_num_threads(1)
    model = RecordedGraph()
    e = world.scenes(559083, 12)
    state = world.prepare(model, e)
    left = torch.zeros(len(e), dtype=torch.long)
    model.pulses.clear()
    natural = world.try_side(model, state, left)
    assert not torch.stack(model.pulses).any()
    opposite = world.try_side(model, state, 1 - left)
    assert torch.equal(natural.first_primary, opposite.first_primary)
    assert not torch.equal(natural.activities[0, :, world.REGION:], opposite.activities[0, :, world.REGION:])
    assert torch.equal(natural.mid, natural.activities[world.MID_TICK - 1, :, world.MID_READOUT])
    assert torch.equal(natural.forecast, natural.activities[world.FORECAST_TICK - 1, :, world.READOUT])
    assert world.MID_TICK < world.FORECAST_TICK < world.TRIAL_TICKS
    for mode in ("blank", "isolate_return"):
        i = world.try_side(model, state, left, mode=mode)
        j = world.try_side(model, state, 1 - left, mode=mode)
        assert torch.equal(i.forecast, j.forecast)
        assert torch.equal(i.final.h[:, :world.REGION], j.final.h[:, :world.REGION])

    blank = world.try_side(model, state, left, mode="blank")
    fixed = world.fixed_side(model, state, blank, torch.zeros(world.TRIAL_TICKS, world.REGION))
    replay = world.try_side(model, state, left, feedback=natural.returns, mid_override=natural.mid_activity,
                            mid_primary_room=natural.mid)
    for index in range(world.TRIAL_TICKS):
        assert_equal_state(fixed.states[index], blank.states[index])
        assert_equal_state(replay.states[index], natural.states[index])
    template = torch.full((world.TRIAL_TICKS, world.REGION), 0.1)
    fixed = world.fixed_side(model, state, blank, template)
    for fixed_state, blank_state in zip(fixed.states, blank.states, strict=True):
        assert torch.equal(fixed_state.h[:, world.REGION:], blank_state.h[:, world.REGION:])
        assert torch.equal(fixed_state.trace[:, world.REGION:], blank_state.trace[:, world.REGION:])
        assert torch.equal(fixed_state.fast[:, world.REGION:], blank_state.fast[:, world.REGION:])

    donor = world.try_side(model, world.prepare(model, world.swap_room_labels(e)), left)
    spliced = world.try_side(model, state, left, mid_override=donor.mid_activity)
    cut = world.MID_TICK - 1
    for index in range(cut):
        assert_equal_state(spliced.states[index], natural.states[index])
    assert torch.equal(spliced.states[cut].h[:, :world.REGION], natural.states[cut].h[:, :world.REGION])
    assert torch.equal(spliced.states[cut].h[:, world.REGION:], donor.mid_activity)
    for name in ("fast", "trace", "candidate"):
        assert torch.equal(getattr(spliced.states[cut], name), getattr(natural.states[cut], name))
    assert not torch.equal(spliced.final.h, natural.final.h)
    for entry in (natural, spliced):
        action = world.select(model, entry.final, None, 0., forced=left)
        assert torch.equal(world.settle(model, entry.final, e, action).paid, world.oracle(e).payoffs[:, 0])


class SensoryTape:
    def __init__(self):
        self.pulses = []

    def start(self, batch):
        return SimpleNamespace(h=torch.zeros(batch, 2 * world.REGION))

    def advance(self, state, inputs, ticks):
        self.pulses.extend([inputs] * ticks)
        return state

    def logit(self, state):
        return state.h[:, 0]


def test_disjoint_prior_maps_have_shared_sensory_vocabulary_and_recoverable_routes():
    torch.set_num_threads(1)
    e, case, _ = world.panel(90317, 96)
    body = SensoryTape()
    shared = world.history(body, e)
    ready = world.prepare(body, e, shared=shared)
    seen = torch.stack(body.pulses)
    rows = torch.arange(len(e))
    room_by_door = torch.full((len(e), 2), -1, dtype=torch.long)
    prior_by_room = room_by_door.clone()
    for start in (0, world.VISIT_TICKS):
        crossed = seen[start + 1, :, :2].argmax(1)
        entered = seen[start + 2, :, 2:4].argmax(1)
        arrived = seen[start + 4, :, 4:7].argmax(1)
        room_by_door[rows, crossed] = entered
        prior_by_room[rows, entered] = arrived
    inspected = seen[2 * world.VISIT_TICKS + 1, :, 2:4].argmax(1)
    found = seen[2 * world.VISIT_TICKS + 3, :, 4:7].argmax(1)
    goals = seen[world.HISTORY_TICKS, :, 4:7].long()
    current = prior_by_room.clone()
    current[rows, inspected] = found
    destinations = current.gather(1, room_by_door)
    previous = prior_by_room.gather(1, room_by_door)
    predicted = goals.gather(1, destinations)
    before = goals.gather(1, previous)

    reference = world.oracle(e)
    assert torch.equal(room_by_door, reference.rooms)
    assert torch.equal(destinations, reference.endpoints)
    assert torch.equal(predicted, reference.payoffs)
    assert torch.equal(before, reference.prior_payoffs)
    chosen = world.select(body, ready, None, 0., forced=predicted.argmax(1))
    assert torch.equal(world.settle(body, ready, e, chosen).paid, predicted.max(1).values)

    first, second = prior_by_room.unbind(1)
    unused = 3 - first - second
    inversions = (first > second).long() + (first > unused).long() + (second > unused).long()
    odd = inversions.remainder(2).bool()
    assert torch.equal(odd, odd[rows ^ 1])
    older = torch.cat((room_by_door, prior_by_room), dim=1)
    changed = torch.column_stack((older, inspected, found))
    episodes = torch.cat((changed, e.visit_order, goals), dim=1)

    def support(tensor, mask):
        return set(map(tuple, tensor[mask].unique(dim=0).tolist()))

    for tape, size in ((older, 6), (changed, 36), (episodes, 432)):
        train, heldout = support(tape, ~odd), support(tape, odd)
        assert len(train) == len(heldout) == size
        assert train.isdisjoint(heldout)
    for name, size in ((world.FLIP, 48), (world.STAY, 96), (world.UNCHANGED, 144), (world.TIE, 144)):
        assert len(support(episodes, odd & (case == name))) == size
    new_maps = torch.cat((room_by_door, destinations), dim=1)
    assert support(new_maps, odd) == support(new_maps, ~odd)
    assert len(support(new_maps, odd)) == 18

    for door in (0, 1):
        for endpoints in (previous, destinations):
            visible = torch.column_stack((room_by_door[:, door], endpoints[:, door], goals))
            assert support(visible, ~odd) == support(visible, odd)
            assert len(support(visible, odd)) == 2 * 3 * len(world.GOAL_ORDERS)
    visible = torch.column_stack((inspected, found, goals))
    assert support(visible, ~odd) == support(visible, odd)
    assert len(support(visible, odd)) == 2 * 3 * len(world.GOAL_ORDERS)
