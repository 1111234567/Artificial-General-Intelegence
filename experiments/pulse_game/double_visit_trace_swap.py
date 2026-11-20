"""Follow the older observation through matched state swaps in frozen bodies."""

from argparse import ArgumentParser
from dataclasses import fields
from unittest.mock import patch

import numpy as np
import torch

from . import double_visit_fit as fit
from . import double_visit_world as world
from . import reroute_eval as reroute


PLAN = fit.EVIDENCE / "trace-swap-plan.md"
STAGES = ("after_first", "after_second")
MODES = ("natural", "self", "activity_trace", "fast_candidate", "all")
SAMPLE_SEED = 274321911
BOOT_SEED = 274321951
PAIR_BATCH = 72


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def layout():
    panel, _, maps = world.panel()
    pairs = world.first_observation_pairs(panel)
    held = world.heldout_map(panel)[pairs]
    require(len(pairs) == 288 and int(held.sum()) == 288 and torch.equal(held[:, 0], held[:, 1]),
            "Old map or critical pair support changed")
    random = np.random.default_rng(SAMPLE_SEED).random((len(pairs), 2), dtype=np.float32)
    return panel, pairs, maps[pairs[:, 0]], held[:, 0], torch.from_numpy(random)


def other(tensor):
    return tensor.reshape(-1, 2, *tensor.shape[1:]).flip(1).reshape_as(tensor)


def equal_state(left, right):
    return all(torch.equal(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))


def swap(state, mode):
    if mode == "natural":
        return state
    if mode == "self":
        return type(state)(**{f.name: getattr(state, f.name).clone() for f in fields(state)})
    transfer = {"activity_trace": ("h", "trace"), "fast_candidate": ("fast", "candidate"),
                "all": ("h", "trace", "fast", "candidate")}[mode]
    return type(state)(**{f.name: other(getattr(state, f.name)) if f.name in transfer else getattr(state, f.name)
                          for f in fields(state)})


def history_snapshots(model, e):
    tick_first = 2 * world.VISIT_TICKS + world.INSPECTION_TICKS
    stages = (tick_first, world.HISTORY_TICKS)
    final_advance = model.advance
    snapshots, clock = {}, 0

    def observe(state, pulse, ticks):
        nonlocal clock
        state = final_advance(state, pulse, ticks)
        clock += ticks
        if clock in stages:
            snapshots[clock] = state
        return state

    with patch.object(model, "advance", side_effect=observe):
        final, receipts = world.history(model, e, return_receipts=True)
    require(clock == world.HISTORY_TICKS and final is snapshots[clock], "Incorrect native history timing")
    require(len(receipts.inspections) == 2, "Missing physical inspection")
    require(torch.equal(e.goals, other(e.goals)), "Pair has different late goals")
    for tick, pulse in enumerate(receipts.inputs):
        if tick < 2 * world.VISIT_TICKS or tick >= tick_first:
            require(torch.equal(pulse, other(pulse)), "Paired sensory input differs outside the first inspection")
    return snapshots, receipts.inputs


def continue_after(model, e, state, remaining):
    for pulse in remaining:
        state = model.advance(state, pulse, 1)
    return world.prepare(model, e, shared=state)


def decision(model, e, prepared, draws):
    query = world.select(model, prepared, draws[:, 0], 0.)
    types = torch.zeros_like(query.side), torch.ones_like(query.side)
    trials = tuple(world.try_side(model, prepared, side) for side in types)
    action_probability = torch.stack([world.select(model, trial.final, draws[:, 1], 0.).probability
                                      for trial in trials], dim=1)
    final = reroute.select_state(query.side, [trial.final for trial in trials])
    command = world.select(model, final, draws[:, 1], 0.)
    require(torch.equal(command.probability, world._by_row(action_probability, query.side)),
            "Selected trial branch did not produce the native action distribution")
    real = world.settle(model, final, e, command)
    truth = world.oracle(e)
    reroute_fixture = (truth.rooms, truth.endpoints, truth.payoffs)
    for observed, expected in zip((real.room, real.endpoint, real.paid), reroute_fixture, strict=True):
        require(torch.equal(observed, world._by_row(expected, real.door)), "Receiver settled in the wrong world")
    return {"query_p": query.probability, "action_p": action_probability, "trial_side": query.side,
            "real_door": real.door, "real_room": real.room, "real_endpoint": real.endpoint, "real_paid": real.paid}


@torch.inference_mode()
def collect_batch(model, e, draws):
    snapshots, pulses = history_snapshots(model, e)
    draws = draws.repeat_interleave(2, dim=0)
    reference = world.prepare(model, e, shared=snapshots[world.HISTORY_TICKS])
    native = decision(model, e, reference, draws)
    values = {"query_p": [], "action_p": [], "trial_side": [], "real_door": [], "real_room": [],
              "real_endpoint": [], "real_paid": []}
    for tick, original in snapshots.items():
        stage = {key: [] for key in values}
        for mode in MODES:
            if mode == "natural":
                result = native
            else:
                start = swap(original, mode)
                continued = continue_after(model, e, start, pulses[tick:])
                result = decision(model, e, continued, draws)
                if mode == "self":
                    require(equal_state(continued, reference), "Self-state return failed bitwise replay")
                    require(all(torch.equal(value, native[key]) for key, value in result.items()),
                            "Self-state return changed a native choice or settlement")
                elif mode == "all":
                    opposite = swap(reference, mode)
                    require(equal_state(continued, opposite), "Full swap did not replay the other history")
                    for key in ("query_p", "action_p", "trial_side", "real_door"):
                        require(torch.equal(result[key], other(native[key])), "Full swap changed the donor policy")
            for key, value in result.items():
                stage[key].append(value.numpy().reshape(-1, 2, *value.shape[1:]))
        for key, entries in stage.items():
            values[key].append(np.stack(entries))
    return {key: np.stack(entries) for key, entries in values.items()}


def path(condition, birth):
    return fit.FIT_ROOT / f"trace-swap-{condition}-{birth}.npz"


def collect(condition, birth):
    model, _, checkpoint = fit.load_final(birth, condition)
    panel, pairs, maps, heldout, tape = layout()
    members = pairs.flatten()
    expected = world.oracle(panel[members]).payoffs.argmax(1).reshape(-1, 2)
    require(expected[:, 0].ne(expected[:, 1]).all(), "Correct door must be opposite within a pair")
    partial = {}
    for start in range(0, len(pairs), PAIR_BATCH):
        block = slice(start, start + PAIR_BATCH)
        rows = pairs[block].flatten()
        calculations = collect_batch(model, panel[rows], tape[block])
        for key, value in calculations.items():
            partial.setdefault(key, []).append(value)
    payload = {key: np.concatenate(values, axis=2) for key, values in partial.items()}
    fit.prior_fit.storage._write_raw(
        path(condition, birth), **payload, birth=birth, condition=condition, checkpoint_sha256=checkpoint,
        plan_sha256=fit.digest(PLAN), source_sha256=fit.digest(__file__), pairs=pairs.numpy(),
        old_map=maps.numpy(), heldout=heldout.numpy(), tape=tape.numpy(), wanted=expected.numpy(),
        stages=STAGES, modes=MODES,
    )
    print(condition, birth, len(pairs), "pairs", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=fit.CONDITIONS)
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS, action="append")
    args = parser.parse_args()
    fit.prior_fit.configure("cpu")
    for birth in args.birth or fit.BIRTHS:
        collect(args.condition, birth)


if __name__ == "__main__":
    main()
