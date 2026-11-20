"""Check the physical contact sequence, finite split, and native reports."""

from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from experiments.pulse_game import double_visit_world as world
from experiments.pulse_game import reroute_world as reroute


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


def support(values, mask):
    return set(map(tuple, values[mask].unique(dim=0).tolist()))


def assert_equal_state(received, expected):
    for field in fields(world.State):
        torch.testing.assert_close(getattr(received, field.name), getattr(expected, field.name), rtol=0, atol=0)


def test_physical_senses_reconstruct_both_routes_and_targets_without_oracle_inputs():
    torch.set_num_threads(1)
    e, _, _ = world.panel()
    body = SensoryTape()
    shared, history = world.history(body, e, return_receipts=True)
    assert torch.equal(history.inputs, torch.stack(body.pulses))
    assert len(history.visits) == len(history.inspections) == 2
    world.prepare(body, e, shared=shared)
    tape = torch.stack(body.pulses)
    assert len(tape) == world.HISTORY_TICKS + world.GOAL_TICKS + world.PREPARATION_TICKS
    assert not tape[:world.HISTORY_TICKS, :, world.GOAL_PORT:].any()

    rows = torch.arange(len(e))
    room_by_door = torch.full((len(e), 2), -1, dtype=torch.long)
    prior_by_room = room_by_door.clone()
    for rank, visit in enumerate(history.visits):
        start = rank * world.VISIT_TICKS
        door = tape[start + 1, :, world.DOOR_PORTS].argmax(1)
        room = tape[start + 2, :, world.ROOM_PORTS].argmax(1)
        endpoint = tape[start + 4, :, world.ENDPOINT_PORTS].argmax(1)
        assert not tape[start].any()
        assert not tape[start + 1, :, world.ROOM_PORTS].any()
        assert not tape[start:start + 4, :, world.ENDPOINT_PORTS].any()
        assert torch.equal(door, visit.door)
        assert torch.equal(room, visit.room)
        assert torch.equal(endpoint, visit.endpoint)
        room_by_door[rows, door] = room
        prior_by_room[rows, room] = endpoint

    current = prior_by_room.clone()
    for rank, inspection in enumerate(history.inspections):
        start = 2 * world.VISIT_TICKS + rank * world.INSPECTION_TICKS
        local = tape[start:start + world.INSPECTION_TICKS]
        assert inspection.door is None
        assert not local[:, :, world.DOOR_PORTS].any()
        assert not local[0].any() and not local[-1].any()
        assert not local[:3, :, world.ENDPOINT_PORTS].any()
        room = local[1, :, world.ROOM_PORTS].argmax(1)
        endpoint = local[3, :, world.ENDPOINT_PORTS].argmax(1)
        assert torch.equal(room, inspection.room)
        assert torch.equal(endpoint, inspection.endpoint)
        current[rows, room] = endpoint

    goals = tape[world.HISTORY_TICKS, :, world.ENDPOINT_PORTS]
    goal_window = tape[world.HISTORY_TICKS:world.HISTORY_TICKS + world.GOAL_TICKS]
    assert (goal_window[:, :, world.GOAL_PORT] == 1).all()
    assert not goal_window[:, :, world.DOOR_PORTS].any() and not goal_window[:, :, world.ROOM_PORTS].any()
    assert not tape[world.HISTORY_TICKS + world.GOAL_TICKS:].any()
    assert (goals.sort(1).values == torch.tensor((-1, 0, 1))).all()
    endpoints = current.gather(1, room_by_door)
    payoffs = goals.gather(1, endpoints)
    truth = world.oracle(e)
    assert torch.equal(room_by_door, truth.rooms)
    assert torch.equal(endpoints, truth.endpoints)
    assert torch.equal(payoffs, truth.payoffs)
    assert torch.equal(goals.gather(1, prior_by_room.gather(1, room_by_door)), truth.prior_payoffs)

    same_room = e.inspected_rooms[:, 0] == e.inspected_rooms[:, 1]
    overwritten = same_room & (e.observed_endpoints[:, 0] != e.observed_endpoints[:, 1])
    check = e.inspected_rooms[overwritten, 1:]
    assert overwritten.sum() == 1728
    assert torch.equal(current[overwritten].gather(1, check).flatten(), e.observed_endpoints[overwritten, 1])
    other = 1 - check
    assert torch.equal(current[overwritten].gather(1, other), prior_by_room[overwritten].gather(1, other))

    # Another goal ranking leaves the exits intact, including after both observations hit one room.
    goal_swap = replace(e, goals=e.goals.roll(1, dims=1))
    goal_body = SensoryTape()
    _, goal_history = world.history(goal_body, goal_swap, return_receipts=True)
    assert torch.equal(goal_history.inputs, history.inputs)
    assert torch.equal(world.oracle(goal_swap).endpoints, endpoints)
    assert torch.equal(world.oracle(goal_swap).payoffs, goal_swap.goals.gather(1, endpoints))

    for door in (0, 1):
        body.pulses.clear()
        action = world.select(body, shared, None, 0., forced=door)
        outcome = world.settle(body, shared, e, action)
        tape = torch.stack(body.pulses)
        assert len(tape) == world.COMMIT_TICKS
        assert (outcome.door == door).all()
        assert torch.equal(outcome.room, room_by_door[:, door])
        assert torch.equal(outcome.endpoint, endpoints[:, door])
        assert torch.equal(outcome.paid, payoffs[:, door])
        assert (tape[0, :, world.DOOR_PORTS].argmax(1) == door).all()
        assert not tape[0, :, world.ROOM_PORTS].any()
        assert not tape[1:, :, world.DOOR_PORTS].any()
        assert torch.equal(tape[1, :, world.ROOM_PORTS].argmax(1), outcome.room)
        assert not tape[:6, :, world.ENDPOINT_PORTS].any() and not tape[:6, :, world.REWARD_PORTS].any()
        assert not tape[3:6].any()
        assert torch.equal(tape[6, :, world.ENDPOINT_PORTS].argmax(1), outcome.endpoint)
        assert torch.equal(tape[6, :, 8] - tape[6, :, 9], outcome.paid)


def test_even_to_odd_partition_exhausts_local_senses_and_balances_cases():
    torch.set_num_threads(1)
    e, case, map_id = world.panel()
    heldout = world.heldout_map(e)
    assert len(e) == 5184 and heldout.sum() == len(e) // 2
    assert tuple(map_id[~heldout].unique().tolist()) == world.EVEN_MAP_IDS
    for easy in (False, True):
        sampled = world.scenes(98413, 400, easy=easy)
        assert not world.heldout_map(sampled).any()
        assert len(sampled.prior_endpoints.unique(dim=0)) == 3

    truth = world.oracle(e)
    older = torch.column_stack((e.door_rooms, e.prior_endpoints))
    changed = torch.column_stack((older, e.inspected_rooms, e.observed_endpoints))
    scenes = torch.column_stack((changed, e.visit_order, e.goals))
    for table, size in ((older, 6), (changed, 216), (scenes, 2592)):
        even, odd = support(table, ~heldout), support(table, heldout)
        assert len(even) == len(odd) == size and even.isdisjoint(odd)

    older_by_door = e.prior_endpoints.gather(1, e.door_rooms)
    observations = torch.column_stack((e.inspected_rooms, e.observed_endpoints, e.goals))
    current = torch.column_stack((e.door_rooms, truth.endpoints))
    fragments = [(observations, 216), (current, 18)]
    for door in (0, 1):
        fragments.append((torch.column_stack((e.door_rooms[:, door], older_by_door[:, door], observations)), 1296))
        fragments.append((torch.column_stack((e.door_rooms[:, door], truth.endpoints[:, door], e.goals)), 36))
    for fragment, size in fragments:
        even, odd = support(fragment, ~heldout), support(fragment, heldout)
        assert even == odd and len(even) == size

    for split in (~heldout, heldout):
        assert case[split].bincount().tolist() == [576, 576, 576, 864]
        tie_pay = truth.payoffs[split & (case == world.TIE), 0]
        assert tie_pay.unique(return_counts=True)[1].tolist() == [288, 288, 288]
        for category in range(len(world.CASE_NAMES)):
            keep = split & (case == category)
            for labels, levels in (
                (e.door_rooms[:, 0], 2), (e.visit_order[:, 0], 2), (e.inspected_rooms[:, 0], 2),
                (e.inspected_rooms[:, 1], 2), (e.observed_endpoints[:, 0], 3), (e.observed_endpoints[:, 1], 3),
            ):
                assert labels[keep].bincount(minlength=levels).tolist() == [int(keep.sum()) // levels] * levels
            if category != world.TIE:
                assert truth.prior_payoffs[keep].argmax(1).sum() * 2 == keep.sum()
                assert truth.payoffs[keep].argmax(1).sum() * 2 == keep.sum()

    relabeled_scene = world.swap_room_labels(e)
    relabeled = world.oracle(relabeled_scene)
    assert torch.equal(world.heldout_map(relabeled_scene), ~heldout)
    assert torch.equal(relabeled.rooms, 1 - truth.rooms)
    assert torch.equal(relabeled.endpoints, truth.endpoints)
    assert torch.equal(relabeled.payoffs, truth.payoffs)
    assert torch.equal(relabeled.prior_payoffs, truth.prior_payoffs)


@torch.no_grad()
def test_critical_pairs_differ_only_on_first_endpoint_and_ablation_erases_that_difference():
    torch.set_num_threads(1)
    e, _, _ = world.panel()
    pairs = world.first_observation_pairs(e)
    assert pairs.shape == (288, 2)
    heldout = world.heldout_map(e)
    assert heldout[pairs[:, 0]].sum() == 144
    assert torch.equal(heldout[pairs[:, 0]], heldout[pairs[:, 1]])
    first, second = e[pairs[:, 0]], e[pairs[:, 1]]
    for field in ("door_rooms", "prior_endpoints", "visit_order", "inspected_rooms", "goals"):
        assert torch.equal(getattr(first, field), getattr(second, field))
    assert torch.equal(first.observed_endpoints[:, 1], second.observed_endpoints[:, 1])
    assert (first.observed_endpoints[:, 0] != second.observed_endpoints[:, 0]).all()
    assert (first.inspected_rooms[:, 0] != first.inspected_rooms[:, 1]).all()
    assert (first.goals.gather(1, first.observed_endpoints[:, 1:]) == 0).all()
    favored = world.oracle(first).payoffs.argmax(1)
    disfavored = world.oracle(second).payoffs.argmax(1)
    assert torch.equal(favored, 1 - disfavored)
    assert torch.equal(first.door_rooms.gather(1, favored[:, None]), first.inspected_rooms[:, :1])
    assert torch.equal(second.door_rooms.gather(1, disfavored[:, None]), second.inspected_rooms[:, 1:])

    for split in (~heldout[pairs[:, 0]], heldout[pairs[:, 0]]):
        for choice in (favored, first.door_rooms[:, 0], first.visit_order[:, 0], first.inspected_rooms[:, 0]):
            assert choice[split].bincount().tolist() == [72, 72]
        for label in (*first.observed_endpoints.unbind(1), second.observed_endpoints[:, 0]):
            assert label[split].bincount().tolist() == [48, 48, 48]

    _, initial = world.history(SensoryTape(), first, return_receipts=True)
    _, alternative = world.history(SensoryTape(), second, return_receipts=True)
    changed_ticks = (initial.inputs != alternative.inputs).any(dim=(1, 2)).nonzero().flatten().tolist()
    assert changed_ticks == [15, 16]
    assert torch.equal(initial.inputs[:, :, :world.ENDPOINT_PORTS.start],
                       alternative.inputs[:, :, :world.ENDPOINT_PORTS.start])

    graph = world.DoubleVisitGraph(731)
    for keep_only_last in (False, True):
        states = [world.history(graph, sample[:8], last_only=keep_only_last, omit_first=True, return_receipts=True)
                  for sample in (first, second)]
        assert_equal_state(states[0][0], states[1][0])
        tape = states[0][1].inputs
        assert torch.equal(tape, states[1][1].inputs)
        assert len(tape) == world.HISTORY_TICKS
        assert not tape[12:18].any()
        assert len(states[0][1].visits) == (0 if keep_only_last else 2)
        assert len(states[0][1].inspections) == 1
        assert torch.equal(tape[18:], initial.inputs[18:, :8])
        assert_equal_state(world.prepare(graph, first[:8], states[0][0]),
                           world.prepare(graph, second[:8], states[1][0]))


@torch.no_grad()
def test_three_native_reports_and_midpoint_replay_keep_the_original_graph():
    torch.set_num_threads(1)
    graph = world.DoubleVisitGraph(731)
    parent = reroute.RerouteGraph(731)
    assert sum(parameter.numel() for parameter in graph.parameters()) == 3062
    assert {name: value.shape for name, value in graph.state_dict().items()} == {
        name: value.shape for name, value in parent.state_dict().items()
    }
    e = world.scenes(42103, 8)
    ready = world.prepare(graph, e)
    assert ready.h.shape[1] == 40
    query = torch.arange(len(e)).remainder(2)
    natural = world.try_side(graph, ready, query)
    opposite = world.try_side(graph, ready, 1 - query)
    assert torch.equal(natural.first_primary, opposite.first_primary)
    assert not torch.equal(natural.activities[0, :, world.REGION:], opposite.activities[0, :, world.REGION:])
    assert torch.equal(natural.mid, natural.activities[world.MID_TICK - 1, :, world.MID_READOUT])
    assert torch.equal(natural.forecast, natural.activities[world.FORECAST_TICK - 1, :, world.READOUT])
    assert torch.equal(natural.endpoint_logits, 4 * natural.activities[world.ENDPOINT_TICK - 1, :, 16:19])
    torch.testing.assert_close(natural.endpoint_probability.sum(1), torch.ones(len(e)))

    for mode in ("blank", "isolate_return"):
        replay = world.try_side(graph, ready, query, mode=mode)
        other = world.try_side(graph, ready, 1 - query, mode=mode)
        assert torch.equal(replay.final.h[:, :world.REGION], other.final.h[:, :world.REGION])
    blank = world.try_side(graph, ready, query, mode="blank")
    fixed = world.fixed_side(graph, ready, blank, torch.zeros(world.TRIAL_TICKS, world.REGION))
    for a, b in zip(blank.states, fixed.states, strict=True):
        assert_equal_state(a, b)

    repeated = world.continue_trial(graph, natural)
    fed = world.try_side(graph, ready, query, feedback=natural.returns,
                         mid_override=natural.mid_activity, mid_primary_room=natural.mid)
    for a, b, c in zip(natural.states, repeated.states, fed.states, strict=True):
        assert_equal_state(a, b)
        assert_equal_state(a, c)
    assert torch.equal(natural.returns, repeated.returns)

    donor_room = (2 * e.door_rooms.gather(1, (1 - query)[:, None])[:, 0] - 1).float()
    continued = world.continue_trial(graph, natural, mid_primary_room=donor_room)
    from_start = world.try_side(graph, ready, query, mid_primary_room=donor_room)
    for a, b in zip(continued.states, from_start.states, strict=True):
        assert_equal_state(a, b)
    midpoint, original = continued.states[world.MID_TICK - 1], natural.states[world.MID_TICK - 1]
    expected = original.h.clone()
    expected[:, world.MID_READOUT] = donor_room
    assert torch.equal(midpoint.h, expected)
    for name in ("fast", "trace", "candidate"):
        assert torch.equal(getattr(midpoint, name), getattr(original, name))
    action = world.select(graph, continued.final, None, 0., forced=query)
    outcome = world.settle(graph, continued.final, e, action)
    assert torch.equal(outcome.endpoint, world.oracle(e).endpoints.gather(1, query[:, None])[:, 0])
