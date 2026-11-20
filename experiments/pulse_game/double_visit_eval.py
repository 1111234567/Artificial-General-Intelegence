"""Freeze both curricula and replay the finite two-inspection world with matched tapes."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import double_visit_world as world
from . import interleave_revisit_train as storage
from . import reroute_eval as reroute


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/double-visit-20261120"
DATA = ROOT / "data/double-visit-20261120"
PLAN = EVIDENCE / "eval-plan.md"
SEEDS = (734590621, 734590687)
DRAWS = 32
MODES = ("natural", "blank", "omit_first", "wrong_return")
CONDITIONS = ("baseline", "identity")
BIRTHS = (731, 732, 733, 734, 735, 736)
BATCH = 216


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fingerprints():
    paths = (PLAN, Path(__file__), *(Path(__file__).with_name(name + ".py") for name in
              ("double_visit_world", "double_visit_fit", "double_visit_eval_report", "reroute_eval",
               "reroute_world", "fork_lesson_world", "fork_lesson_whatif", "fork_lesson_native_request",
               "fork_lesson_bit_relay")))
    return {path.name: digest(path) for path in paths}


def data_path(condition, birth):
    return DATA / f"eval-{condition}-{birth}.npz"


def episode_keys(e):
    return list(map(tuple, torch.column_stack([getattr(e, field.name) for field in fields(e)]).int().tolist()))


def layout():
    e, case, map_id = world.panel()
    expected = world.oracle(e)
    # Rebuild the physical graph with scalar writes independently of both
    # the world's scatter-based oracle and the settlement implementation.
    independent = []
    for old, rooms, exits, doors in zip(e.prior_endpoints.tolist(), e.inspected_rooms.tolist(),
                                       e.observed_endpoints.tolist(), e.door_rooms.tolist(), strict=True):
        for room, endpoint in zip(rooms, exits, strict=True):
            old[room] = endpoint
        independent.append(tuple(old[room] for room in doors))
    np.testing.assert_array_equal(expected.endpoints, independent)
    keys = episode_keys(e)
    index = {key: i for i, key in enumerate(keys)}
    if len(index) != len(e):
        raise AssertionError("The exhaustive panel contains duplicate histories")

    def locate(variation):
        return np.array([index[key] for key in episode_keys(variation)])

    alternate = e.observed_endpoints.clone()
    alternate[:, 0] = (alternate[:, 0] + 1) % 3
    donor = locate(replace(e, observed_endpoints=alternate))
    reverse = locate(replace(e, inspected_rooms=e.inspected_rooms.flip(1),
                             observed_endpoints=e.observed_endpoints.flip(1)))
    reassigned = e.goals.scatter(1, expected.endpoints, expected.payoffs.flip(1))
    goal_swap = locate(replace(e, goals=reassigned))
    room_swap = locate(world.swap_room_labels(e))
    strict = (case != world.TIE).numpy()
    same_room = e.inspected_rooms[:, 0].eq(e.inspected_rooms[:, 1]).numpy()
    row = np.arange(len(e))
    first = world.first_observation_pairs(e).numpy()
    pairs = {
        "first_only_changes": first,
        "last_only_changes": reverse[first],
        "goal_swap": np.column_stack((row, goal_swap))[strict & (row < goal_swap)],
        "room_relabel": np.column_stack((row, room_swap))[strict & (row < room_swap)],
        "overwritten_first": np.column_stack((row, donor))[strict & same_room],
    }
    if len(first) != 288:
        raise AssertionError("The exhaustive critical-support denominator changed")
    for name, changed in (("first_only_changes", 0), ("last_only_changes", 1)):
        a, b = pairs[name].T
        for field in fields(e):
            value = getattr(e, field.name).numpy()
            if field.name == "observed_endpoints":
                np.testing.assert_array_equal(value[a, 1 - changed], value[b, 1 - changed])
                np.testing.assert_array_equal(value[a, changed] != value[b, changed], True)
            else:
                np.testing.assert_array_equal(value[a], value[b])
    np.testing.assert_array_equal(expected.payoffs[reverse[first]], expected.payoffs[first])
    for pair_name, selected in pairs.items():
        a, b = selected.T
        best = expected.payoffs.argmax(1).numpy()
        opposite = pair_name in {"first_only_changes", "last_only_changes", "goal_swap"}
        np.testing.assert_array_equal(best[a] != best[b], opposite)
        if pair_name != "room_relabel":
            np.testing.assert_array_equal(map_id[a], map_id[b])

    # Ignore room names and chronological update order in the tape grouping so
    # each design contrast shares both draws, including the time-reversed pair.
    groups = torch.column_stack((e.prior_endpoints.sort(1).values, e.visit_order)).numpy()
    _, group_id = np.unique(groups, axis=0, return_inverse=True)
    for selected in pairs.values():
        a, b = selected.T
        np.testing.assert_array_equal(group_id[a], group_id[b])
    heldout = world.heldout_map(e).numpy()
    np.testing.assert_array_equal(heldout, ~np.isin(map_id, world.EVEN_MAP_IDS))
    return e, case.numpy(), map_id.numpy(), heldout, donor, pairs, group_id


def random_tape(seed, group_id):
    """One self-query draw and two independent motor draws, fixed across contrasts."""
    streams = np.random.SeedSequence(seed).spawn(3)
    shape = (int(group_id.max()) + 1, DRAWS)
    return np.stack([np.random.default_rng(stream).random(shape, dtype=np.float32)[group_id]
                     for stream in streams], axis=-1)


def check_history(e, full, omission):
    if (len(full.visits), len(full.inspections), len(omission.visits), len(omission.inspections)) != (2, 2, 2, 1):
        raise AssertionError("Unexpected real visits in the decision prehistory")
    for rank, walk in enumerate(full.visits):
        door = e.visit_order[:, rank]
        room = e.door_rooms.gather(1, door[:, None]).squeeze(1)
        endpoint = e.prior_endpoints.gather(1, room[:, None]).squeeze(1)
        if any(not torch.equal(got, want) for got, want in zip((walk.door, walk.room, walk.endpoint),
                                                             (door, room, endpoint), strict=True)):
            raise AssertionError("An old traversal does not match its physical map")
    for rank, walk in enumerate(full.inspections):
        if (walk.door is not None or not torch.equal(walk.room, e.inspected_rooms[:, rank])
                or not torch.equal(walk.endpoint, e.observed_endpoints[:, rank])):
            raise AssertionError("A local inspection did not start in its room and follow its current exit")
    first = 2 * world.VISIT_TICKS
    last = first + world.INSPECTION_TICKS
    if (len(full.inputs) != world.HISTORY_TICKS or full.inputs.shape != omission.inputs.shape
            or full.inputs[first:, :, world.DOOR_PORTS].any() or full.inputs[:, :, world.REWARD_PORTS].any()
            or full.inputs[:, :, world.GOAL_PORT].any()
            or not torch.equal(full.inputs[:first], omission.inputs[:first])
            or not torch.equal(full.inputs[last:], omission.inputs[last:]) or omission.inputs[first:last].any()):
        raise AssertionError("History or its matched first-inspection ablation leaked inputs or changed time")


def check_settle(reference, decision, receipt):
    if not torch.equal(receipt.door, decision.side):
        raise AssertionError("The action receipt used a different door than the native motor command")
    for measured, truth in ((receipt.room, reference.rooms), (receipt.endpoint, reference.endpoints),
                            (receipt.paid, reference.payoffs)):
        if not torch.equal(measured, truth.gather(1, decision.side[:, None]).squeeze(1)):
            raise AssertionError("Physical settlement differs from the reference route")


def collect_mode(model, branches, query_probability):
    side = torch.zeros_like(query_probability)
    return {
        "query_p": query_probability,
        "action_p": torch.stack([world.select(model, tr.final, side, 0.).probability for tr in branches], 1),
        "room_read": torch.stack([tr.mid for tr in branches], 1),
        "exit_read": torch.stack([tr.endpoint_logits for tr in branches], 1),
        "value_read": torch.stack([tr.forecast for tr in branches], 1),
    }


def _store_receipts(output, key, receipts):
    for name in ("door", "room", "endpoint", "paid"):
        output[f"{key}/{name}"] = torch.stack([getattr(item, name) for item in receipts], 1)


@torch.inference_mode()
def collect_batch(model, e, donor, tapes, *, audit):
    full, visits = world.history(model, e, return_receipts=True)
    short, omissions = world.history(model, e, omit_first=True, return_receipts=True)
    check_history(e, visits, omissions)
    prepared = world.prepare(model, e, shared=full)
    abbreviated = world.prepare(model, e, shared=short)
    friend = world.prepare(model, donor)
    left = torch.zeros_like(e.door_rooms[:, 0])
    sides = left, left + 1
    draws = torch.zeros_like(left, dtype=e.goals.dtype)
    query = world.select(model, prepared, draws, 0.).probability
    short_query = world.select(model, abbreviated, draws, 0.).probability
    truth = world.oracle(e)

    natural = [world.try_side(model, prepared, side) for side in sides]
    blank = world.try_side(model, prepared, left, mode="blank")
    omitted = [world.try_side(model, abbreviated, side) for side in sides]
    donor_blank = world.try_side(model, friend, left, mode="blank")
    wrong, fallback, norm_error = [], 0, 0.
    for side, original in zip(sides, natural, strict=True):
        donor_delta = world.try_side(model, friend, side).returns - donor_blank.returns
        delta = original.returns - blank.returns
        replacement, count, _ = reroute.match_return(delta, donor_delta)
        received = blank.returns + replacement
        applied = (received.double() - blank.returns.double()).norm(dim=-1)
        norm_error = max(norm_error, (applied - delta.double().norm(dim=-1)).abs().max().item())
        fallback += count
        wrong.append(world.try_side(model, prepared, side, feedback=received))
        if audit:
            reroute.check_trial(original, world.try_side(model, prepared, side, feedback=original.returns))
    if norm_error > 2e-5:
        raise AssertionError("Norm matching altered the energy of the return increment")
    if audit:
        reroute.check_trial(blank, world.try_side(model, prepared, left + 1, mode="blank"))
        requested = (tapes[0][:, 0] < query).long()
        direct = world.try_side(model, prepared, requested)
        gathered = reroute.select_state(requested, [tr.final for tr in natural])
        if not reroute.equal_state(direct.final, gathered):
            raise AssertionError("Enumerating natural branches changed the self-selected trial")
        if any(not torch.equal(tr.first_primary, blank.first_primary) for tr in natural):
            raise AssertionError("The trial pulse reached the main region directly")

    output = {"room": truth.rooms, "endpoint": truth.endpoints,
              "paid": truth.payoffs, "past_paid": truth.prior_payoffs}
    # Traverse and audit the two deterministic physical payoff branches on trial
    # copies. Each sampled command below has exactly one of these outcomes.
    physical = []
    for side in sides:
        decision = world.select(model, blank.final, draws, 0., forced=side)
        actual = world.settle(model, blank.final, e, decision)
        check_settle(truth, decision, actual)
        physical.append(actual)
    _store_receipts(output, "physical", physical)

    configurations = zip(MODES, (natural, [blank, blank], omitted, wrong),
                         (query, query, short_query, query), strict=True)
    for mode, branches, q in configurations:
        values = collect_mode(model, branches, q)
        output.update({f"{mode}/{key}": value for key, value in values.items()})
        # On each independent tape, also run the actual sampled first
        # query and decision through physical contact and compare to the receipt.
        sampled = []
        for tape in tapes:
            q_side = (tape[:, 0] < q).long()
            final = reroute.select_state(q_side, [entry.final for entry in branches])
            command = world.select(model, final, tape[:, 1], 0.)
            p_action = values["action_p"].gather(1, q_side[:, None]).squeeze(1)
            torch.testing.assert_close(command.probability, p_action, rtol=0, atol=0)
            actual = world.settle(model, final, e, command)
            check_settle(truth, command, actual)
            sampled.append(actual)
        _store_receipts(output, f"{mode}/tape_contact", sampled)
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise AssertionError("A frozen model produced a nonfinite evaluation")
    return {key: value.cpu().numpy() for key, value in output.items()}, fallback, norm_error


def collect(condition, birth, device="cpu"):
    from . import double_visit_fit as fit

    model, _, checkpoint = fit.load_final(birth, condition, device=device)
    model.eval().requires_grad_(False)
    if model.width != 40 or len(list(model.modules())) != 1:
        raise AssertionError("Expected a single 40-node experimental body")
    e, case, map_id, heldout, donor, _, group_id = layout()
    tapes = [torch.from_numpy(random_tape(seed, group_id)[:, 0].copy()).to(device) for seed in SEEDS]
    stored = {}
    count, error = 0, 0.
    for start in range(0, len(e), BATCH):
        stop = start + BATCH
        block = slice(start, stop)
        def on_device(scene):
            return replace(scene, **{field.name: getattr(scene, field.name).to(device) for field in fields(scene)})

        raw, extra, norm = collect_batch(model, on_device(e[block]), on_device(e[donor[block]]),
                                         [draw[block] for draw in tapes], audit=start == 0)
        for key, value in raw.items():
            stored.setdefault(key, []).append(value)
        count += extra
        error = max(error, norm)
    provenance = fingerprints()
    storage._write_raw(data_path(condition, birth), **{key: np.concatenate(value) for key, value in stored.items()},
                       **{field.name: getattr(e, field.name).numpy() for field in fields(e)},
                       case=case, map_id=map_id, heldout=heldout, group_id=group_id, donor=donor,
                       fallback=count, norm_error=error, checkpoint=checkpoint, birth=birth, condition=condition,
                       source_names=list(provenance), source_sha256=list(provenance.values()), seeds=SEEDS, draws=DRAWS)
    print(f"{condition} {birth} across all {len(e)} finite worlds", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--birth", type=int, choices=BIRTHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.condition is None and args.birth is None:
        from .double_visit_eval_report import summarize
        summarize()
    elif args.condition is None or args.birth is None:
        parser.error("--condition and --birth must be given together")
    else:
        collect(args.condition, args.birth, args.device)


if __name__ == "__main__":
    main()
