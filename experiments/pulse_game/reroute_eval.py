"""Freeze native reroute bodies and execute their first post-observation action."""

from argparse import ArgumentParser
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as storage
from . import reroute_fit as fit
from . import reroute_world as world


TESTS = ((287416031, 128), (287416099, 128))
CALIBRATION = (287415979, 64)
BATCH_ROOTS = 2
REPEATS = world.PANEL_REPEATS
PROBE_MODES = ("natural", "blank", "fixed_side", "wrong_return", "last_only")
SHORTCUTS = ("old_best", "pretrial_motor")
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/reroute-20261120"
DATA = ROOT / "data/reroute-20261120"
PLAN = EVIDENCE / "eval-plan.md"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fingerprints():
    directory = Path(__file__).parent
    sources = [PLAN, Path(__file__), Path(world.__file__), Path(fit.__file__),
               directory / "reroute_eval_stats.py", directory / "reroute_eval_report.py",
               directory / "fork_lesson_world.py", directory / "fork_lesson_whatif.py",
               directory / "fork_lesson_native_request.py", directory / "fork_lesson_bit_relay.py"]
    return {path.name: digest(path) for path in sources}


def batches(episode, *, device):
    for start in range(0, len(episode.door_rooms), BATCH_ROOTS * REPEATS):
        end = start + BATCH_ROOTS * REPEATS
        yield start, type(episode)(**{
            field.name: getattr(episode, field.name)[start:end].to(device) for field in fields(episode)
        })


def zeros_and_ones(episode):
    zeros = torch.zeros_like(episode.inspected_room)
    return zeros, 1 - zeros


def finish(model, episode, final, draw, reference, *, forced=None):
    decision = world.select(model, final, draw, exploration=0., forced=forced)
    actual = world.settle(model, final, episode, decision)
    for field, expected in (("paid", reference.payoffs), ("room", reference.rooms), ("endpoint", reference.endpoints)):
        if not torch.equal(getattr(actual, field), expected.gather(1, decision.side[:, None]).squeeze(1)):
            raise AssertionError(f"Real {field} differs from independently computed route truth")
    if not torch.equal(actual.door, decision.side):
        raise AssertionError("Settled door differs from the original motion command")
    return {
        "action": decision.side, "action_p": decision.probability, "room": actual.room,
        "endpoint": actual.endpoint, "pay": actual.paid,
    }


def equal_state(left, right):
    return all(torch.equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left))


def select_state(query, options):
    left, right = options
    result = {}
    for field in fields(left):
        value = getattr(left, field.name)
        mask = query.bool().view(-1, *([1] * (value.ndim - 1)))
        result[field.name] = torch.where(mask, getattr(right, field.name), value)
    return type(left)(**result)


def predictions_and_action(model, episode, reference, branches, requested, action_draw):
    final = select_state(requested, [trial.final for trial in branches])
    out = finish(model, episode, final, action_draw, reference)
    out["forecast"] = torch.stack([trial.forecast for trial in branches], dim=1)
    out["mid"] = torch.stack([trial.mid for trial in branches], dim=1)
    out["requested"] = requested
    if not all(torch.isfinite(value).all() for value in out.values()):
        raise AssertionError("A frozen evaluation produced a nonfinite readout")
    return out


def match_return(delta, donor):
    """Use the donor direction with the original life and tick's increment norm."""
    target = delta.double().norm(dim=-1, keepdim=True)
    vector = donor.double()
    size = vector.norm(dim=-1, keepdim=True)
    fixed_axis = torch.zeros_like(vector)
    fixed_axis[..., 0] = 1
    direction = torch.where(size > 0, vector / size.clamp_min(1e-100), fixed_axis)
    replacement = (direction * target).to(delta.dtype)
    norm_error = (replacement.double().norm(dim=-1, keepdim=True) - target).abs().max()
    return replacement, (size.eq(0) & target.gt(0)).sum().item(), norm_error.item()


def check_trial(expected, replay):
    if (not all(equal_state(a, b) for a, b in zip(expected.states, replay.states, strict=True))
            or not torch.equal(expected.forecast, replay.forecast)
            or not torch.equal(expected.mid, replay.mid)
            or not torch.equal(expected.returns, replay.returns)):
        raise AssertionError("Replayed trial failed its bitwise readout and trajectory check")


def audit_history(episode, receipts):
    if len(receipts.visits) != 2:
        raise AssertionError("Physical history must begin with two traversals only")
    for tick, visit in enumerate(receipts.visits):
        if not torch.equal(visit.door, episode.visit_order[:, tick]):
            raise AssertionError("A physical history visit used the wrong initial door")
        if not torch.equal(visit.room, episode.door_rooms.gather(1, visit.door[:, None]).squeeze(1)):
            raise AssertionError("Initial physical route differs from the old map")
        if not torch.equal(visit.endpoint, episode.prior_endpoints.gather(1, visit.room[:, None]).squeeze(1)):
            raise AssertionError("Initial physical endpoint differs from the old map")
    local = receipts.inspection
    if (local.door is not None or not torch.equal(local.room, episode.inspected_room)
            or not torch.equal(local.endpoint, episode.observed_endpoint)):
        raise AssertionError("The last physical observation revisited a door or the wrong room")
    final_tape = receipts.inputs[-world.INSPECTION_TICKS :]
    if (final_tape[:, :, world.DOOR_PORTS].any()
            or receipts.inputs[:, :, world.REWARD_PORTS].any()
            or receipts.inputs[:, :, world.GOAL_PORT].any()):
        raise AssertionError("Physical history exposed a door during inspection, a reward, or a goal")


def check_donors(episode, donor):
    if not torch.equal(episode.door_rooms[donor], episode.door_rooms.flip(1)):
        raise AssertionError("A return donor must reverse the hidden door-to-room map")
    for name in ("inspected_room", "observed_endpoint", "goals", "prior_endpoints"):
        value = getattr(episode, name)
        if not torch.equal(value[donor], value):
            raise AssertionError(f"A return donor changed {name}")


def append_mode(out, mode, values):
    out.update({f"{mode}/{key}": value for key, value in values.items()})


@torch.inference_mode()
def batch(model, episode, query_draw, action_draw, templates, *, audit):
    history, receipts = world.history(model, episode, return_receipts=True)
    audit_history(episode, receipts)
    prepared = world.prepare(model, episode, shared=history)
    query = world.select(model, prepared, query_draw, exploration=0.)
    first_motor = world.select(model, prepared, action_draw, exploration=0.)
    reference = world.oracle(episode)
    sides = zeros_and_ones(episode)
    donor = torch.arange(len(query.side), device=query.side.device) ^ 1
    check_donors(episode, donor)

    blank = world.try_side(model, prepared, sides[0], mode="blank")
    native = [world.try_side(model, prepared, side) for side in sides]
    if audit:
        check_trial(blank, world.try_side(model, prepared, sides[1], mode="blank"))
        check_trial(blank, world.fixed_side(model, prepared, blank, torch.zeros_like(templates[0])))
        for side, original in zip(sides, native):
            check_trial(original, world.try_side(model, prepared, side, feedback=original.returns))
            finish(model, episode, blank.final, action_draw, reference, forced=side)
        direct = world.try_side(model, prepared, query.side)
        gathered = select_state(query.side, [trial.final for trial in native])
        if not equal_state(gathered, direct.final):
            raise AssertionError("Gathered natural trial did not reproduce the self-selected trial")

    out = {
        "truth": reference.payoffs, "prior_truth": reference.prior_payoffs,
        "truth_room": reference.rooms, "truth_endpoint": reference.endpoints,
        "door_rooms": episode.door_rooms, "prior_endpoints": episode.prior_endpoints,
        "inspected_room": episode.inspected_room,
        "observed_endpoint": episode.observed_endpoint, "goals": episode.goals,
        "query_draw": query_draw, "action_draw": action_draw, "query_p": query.probability,
        "post_change_door_walks_before_decision": torch.zeros_like(query.side),
    }
    for mode, branches in (("natural", native), ("blank", [blank, blank])):
        append_mode(out, mode, predictions_and_action(model, episode, reference, branches, query.side, action_draw))

    fixed = [world.fixed_side(model, prepared, blank, template) for template in templates]
    for trial in fixed:
        for actual, expected in zip(trial.states, blank.states, strict=True):
            n = world.REGION
            if not (torch.equal(actual.h[:, n:], expected.h[:, n:])
                    and torch.equal(actual.trace[:, n:], expected.trace[:, n:])
                    and torch.equal(actual.fast[:, n:, :], expected.fast[:, n:, :])):
                raise AssertionError("The side-only controller altered the replica's blank trajectory")
    append_mode(out, "fixed_side", predictions_and_action(model, episode, reference, fixed, query.side, action_draw))

    wrong = []
    norm_error = 0.
    fallback = 0
    for side, original in zip(sides, native):
        delta = original.returns - blank.returns
        replacement, count, error = match_return(delta, delta[:, donor])
        norm_error = max(norm_error, error)
        fallback += count
        wrong.append(world.try_side(model, prepared, side, feedback=blank.returns + replacement))
    if norm_error > 2e-5:
        raise AssertionError("The mismatched return did not preserve incremental current norms")
    append_mode(out, "wrong_return", predictions_and_action(model, episode, reference, wrong, query.side, action_draw))

    short_history, short_receipts = world.history(model, episode, last_only=True, return_receipts=True)
    inspection_start = 2 * world.VISIT_TICKS
    torch.testing.assert_close(short_receipts.inputs[inspection_start:], receipts.inputs[inspection_start:],
                               atol=0, rtol=0)
    if (short_receipts.visits or short_receipts.inputs.shape != receipts.inputs.shape
            or short_receipts.inputs[:inspection_start].any()):
        raise AssertionError("The last-only baseline must remove both door walks without changing time")
    short_prepared = world.prepare(model, episode, shared=short_history)
    short_query = world.select(model, short_prepared, query_draw, exploration=0.)
    short = [world.try_side(model, short_prepared, side) for side in sides]
    append_mode(out, "last_only", predictions_and_action(model, episode, reference, short, short_query.side, action_draw))

    if not (reference.prior_payoffs[:, 0] != reference.prior_payoffs[:, 1]).all():
        raise AssertionError("The initial optimum must be strict throughout this test panel")
    old = reference.prior_payoffs.argmax(dim=1)
    for mode, forced in (("old_best", old), ("pretrial_motor", first_motor.side)):
        append_mode(out, mode, finish(model, episode, blank.final, action_draw, reference, forced=forced))
    return {name: value.cpu().numpy() for name, value in out.items()}, norm_error, fallback


@torch.inference_mode()
def calibrate(model, device):
    episode, _, _ = world.panel(*CALIBRATION)
    total = None
    for _, batch_episode in batches(episode, device=device):
        prepared = world.prepare(model, batch_episode)
        sides = zeros_and_ones(batch_episode)
        blank = world.try_side(model, prepared, sides[0], mode="blank").returns
        native = [world.try_side(model, prepared, side).returns for side in sides]
        block = torch.stack([(entry.double() - blank.double()).sum(1) for entry in native])
        total = block if total is None else total + block
    return (total / len(episode.door_rooms)).float()


def path_for(kind, birth, seed):
    return DATA / f"eval-{seed}-{kind}-{birth}.npz"


def cal_for(kind, birth):
    return DATA / f"eval-cal-{kind}-{birth}.npz"


def run(kind, birth, device):
    model, _, checkpoint = fit.load_final(kind, birth)
    model.eval().requires_grad_(False).to(device)
    if model.width != 40 or len(list(model.modules())) != 1:
        raise AssertionError("Frozen evaluation expects one native 40-node body")
    provenance = fingerprints()
    signature = {"checkpoint": checkpoint, "source_names": list(provenance),
                 "source_sha256": list(provenance.values()), "kind": kind, "birth": birth}
    cal_path = cal_for(kind, birth)
    templates = calibrate(model, device)
    storage._write_raw(cal_path, templates=templates.cpu().numpy(), seed=CALIBRATION[0],
                       roots=CALIBRATION[1], **signature)
    for seed, roots in TESTS:
        episode, case_ids, root_ids = world.panel(seed, roots)
        np.testing.assert_array_equal(root_ids, np.repeat(np.arange(roots), REPEATS))
        rng = np.random.default_rng([seed, 5614053])
        draws = torch.from_numpy(np.repeat(rng.random((2, roots), dtype=np.float32), REPEATS, axis=1))
        rows, norm_error, fallback = [], 0., 0
        for start, scene in batches(episode, device=device):
            end = start + len(scene.door_rooms)
            raw, error, count = batch(model, scene, *draws[:, start:end].to(device), templates, audit=start == 0)
            rows.append(raw)
            norm_error = max(norm_error, error)
            fallback += count
        values = {name: np.concatenate([row[name] for row in rows]) for name in rows[0]}
        storage._write_raw(path_for(kind, birth, seed), case=case_ids, root=root_ids,
                           seed=seed, roots=roots, norm_error=norm_error, fallback=fallback,
                           calibration_sha256=digest(cal_path), **values, **signature)
        print(kind, birth, seed, "completed", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=fit.KINDS)
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.kind is None and args.birth is None:
        from . import reroute_eval_report

        reroute_eval_report.summarize()
    elif args.kind is None or args.birth is None:
        parser.error("--kind and --birth must be given together")
    else:
        run(args.kind, args.birth, torch.device(args.device))


if __name__ == "__main__":
    main()
