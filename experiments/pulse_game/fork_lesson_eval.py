"""Freeze fork-lesson checkpoints and log real delayed choices under paired interventions."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

from . import fork_lesson_fit as fit
from . import fork_lesson_world as world
from . import interleave_revisit_train as storage
from .native_rehearsal_fit import digest


ROOT = Path(__file__).resolve().parents[2]
DATA = fit.DATA.parent / "eval"
EVIDENCE = ROOT / "docs" / "evidence" / "fork-lesson-20261120"
PLAN = EVIDENCE / "eval-plan.md"
WORLDS = {"private_a": 98633071, "private_b": 98633107}
ANGLE_BOUNDS = (55, 85, 125, 180.01)
ROOTS_PER_ANGLE = 48
ROOTS_PER_BATCH = 4
FACTORS = tuple(product((0, 1), repeat=6))
REPEATS = len(FACTORS)
INTERVENTION_KINDS = ("whole", "bridge")
MODES = ("natural", "own", "cut_out", "cut_back", "cut_both", "donor_out", "donor_back", "donor_both",
         "shuffle_out", "shuffle_back", "shuffle_both", "forecast_swap", "no_history", "no_menu")


def active_modes(kind):
    return MODES if kind in INTERVENTION_KINDS else MODES[:1]


def angle_degrees(vectors):
    directions = torch.nn.functional.normalize(vectors.double(), dim=2)
    cosine = (directions[:, 0] * directions[:, 1]).sum(1).clamp(-1, 1)
    return torch.rad2deg(cosine.acos())


def index_scene(scene, rows):
    return replace(scene, **{field.name: getattr(scene, field.name)[rows] for field in fields(scene)})


def heldout(seed, roots_per_angle):
    """Preserve unseen sensory geometries and cross all private values and presentation factors."""
    base = world.scenes(seed, ROOTS_PER_ANGLE * 96)
    angles = angle_degrees(base.vectors)
    chosen = []
    for lo, hi in zip(ANGLE_BOUNDS[:-1], ANGLE_BOUNDS[1:], strict=True):
        available = ((angles >= lo) & (angles < hi)).nonzero(as_tuple=True)[0]
        if len(available) < roots_per_angle:
            raise AssertionError(f"Insufficient private roots in angle stratum [{lo}, {hi})")
        chosen.append(available[:roots_per_angle])
    geometry = torch.cat(chosen)
    roots = len(geometry)
    factual = index_scene(base, geometry.repeat_interleave(REPEATS))
    factors = torch.tensor(FACTORS).repeat(roots, 1)
    law = (2 * factors[:, :2] - 1).to(factual.values)
    order = factors[:, 3, None] ^ torch.arange(2)[None, :]
    flips = (2 * factors[:, 4:] - 1).to(factual.flips)
    paired = replace(factual, values=law, first=factors[:, 2], door_order=order, flips=flips)
    row = torch.arange(len(factors))
    donor = row ^ 48
    mirror = (row ^ 4) ^ ((factors[:, 4] ^ factors[:, 5]) * 3)
    if not torch.equal(world.door_cues(paired), world.door_cues(paired)[donor]):
        raise AssertionError("Opposite-history donors changed visible door cues")
    swapped = world.swap_doors(paired)
    if any(not torch.equal(getattr(swapped, f.name), getattr(paired, f.name)[mirror]) for f in fields(paired)):
        raise AssertionError("The mirror did not exchange both complete doors within their paired world")
    return paired, angles[geometry], donor, mirror, geometry


def states_equal(left, right):
    return all(torch.equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left))


def shuffled(currents):
    """Change the receiving addresses while keeping same-row and same-tick current magnitudes."""
    mixed = currents.roll(currents.shape[-1] // 2, dims=-1)
    reference = currents.double().square().sum(-1)
    changed = mixed.double().square().sum(-1)
    torch.testing.assert_close(reference, changed, rtol=3e-15, atol=1e-15)
    return mixed, float((reference - changed).abs().max())


def run_window(model, state, **kwargs):
    return world.choice_window(model, state, return_outgoing=True, **kwargs)


def swap_forecast(model, cue, natural):
    horizon = fit.AFTER_FORECAST_TICKS
    prefix, history, _, _ = run_window(model, cue, ticks=world.BLANK_TICKS - horizon)
    continued = run_window(model, prefix, ticks=horizon)
    if not states_equal(continued[0], natural[0]) or not torch.equal(prefix.h, natural[1][-1 - horizon]):
        raise AssertionError("Splitting at the native forecast time changed the natural decision")
    new_activity = prefix.h.clone()
    new_activity[:, 12:14] = prefix.h[:, 12:14].flip(1)
    revised = replace(prefix, h=new_activity)
    finish, suffix, inbound, outbound = run_window(model, revised, ticks=horizon)
    return finish, torch.cat((history[:-1], new_activity[None], suffix)), inbound, outbound


def selected_rollout(model, state, scene, side=None, *, greedy=False):
    choice = world.choose(model, state, scene, side=side, greedy=greedy)
    _, outcome = world.commit(model, state, scene, choice)
    if choice.immediate.ne(0).any() or not torch.isin(outcome.payoff, outcome.payoff.new_tensor((-1, 1))).all():
        raise AssertionError("Every evaluated gate must give zero at selection and a later signed payoff")
    if side is not None and not torch.equal(choice.side, side):
        raise AssertionError("Counterfactual route executed the wrong motor choice")
    return choice, outcome


@torch.no_grad()
def run_batch(model, scene, kind, donor, mirror):
    history = world.support(model, scene)
    flags = fit.window_flags(kind)
    cue = world.cue(model, history, scene, **flags)
    natural = run_window(model, cue, **flags)
    final, activities, inbound, outbound = natural
    batch = len(scene.values)
    both_real = torch.stack([selected_rollout(model, final, scene, side=torch.full((batch,), side, dtype=torch.long))[1].payoff
                             for side in (0, 1)], dim=1)
    if not torch.equal(both_real, -both_real[donor]) or not torch.equal(both_real, both_real[mirror].flip(1)):
        raise AssertionError("Actual settlements failed the opposite-history or left/right pairing")

    conditions = {"natural": natural}
    energy_error = 0.
    if kind in INTERVENTION_KINDS:
        own = run_window(model, cue, override=inbound, outbound_override=outbound)
        if not states_equal(own[0], final) or any(not torch.equal(x, y) for x, y in zip(own[1:], natural[1:], strict=True)):
            raise AssertionError("Putting back both actual currents changed the same world's evolution")
        conditions["own"] = own
        for name, in_kw, out_kw in (("cut", {"cut": True}, {"cut_outbound": True}),
                                    ("donor", {"override": inbound[:, donor]}, {"outbound_override": outbound[:, donor]})):
            conditions[f"{name}_out"] = run_window(model, cue, **out_kw)
            conditions[f"{name}_back"] = run_window(model, cue, **in_kw)
            conditions[f"{name}_both"] = run_window(model, cue, **in_kw, **out_kw)
        shuffle_in, in_error = shuffled(inbound)
        shuffle_out, out_error = shuffled(outbound)
        energy_error = max(in_error, out_error)
        conditions["shuffle_out"] = run_window(model, cue, outbound_override=shuffle_out)
        conditions["shuffle_back"] = run_window(model, cue, override=shuffle_in)
        conditions["shuffle_both"] = run_window(model, cue, override=shuffle_in, outbound_override=shuffle_out)
        conditions["forecast_swap"] = swap_forecast(model, cue, natural)
        no_history = model.advance(model.start(batch), scene.vectors.new_zeros((batch, world.PORTS)),
                                   world.SUPPORT_TICKS)
        conditions["no_history"] = run_window(model, world.cue(model, no_history, scene))
        conditions["no_menu"] = run_window(model, world.cue(model, history, scene, show=False))

    modes = active_modes(kind)
    readings = []
    for mode in modes:
        state, trial_activity, _, _ = conditions[mode]
        choice, outcome = selected_rollout(model, state, scene)
        greedy, greedy_outcome = selected_rollout(model, state, scene, greedy=True)
        predicted = fit.forecast_scores(trial_activity)
        if predicted.shape != both_real.shape or not torch.isfinite(predicted).all() or not torch.isfinite(choice.logit).all():
            raise AssertionError("Invalid logged motor or native forecast values")
        rows = torch.arange(batch)
        if not torch.equal(outcome.payoff, both_real[rows, choice.side]):
            raise AssertionError("Perturbed natural route failed its matched real payout")
        if not torch.equal(greedy_outcome.payoff, both_real[rows, greedy.side]):
            raise AssertionError("Perturbed greedy route failed its matched real payout")
        readings.append(dict(side=choice.side, immediate=choice.immediate, payoff=outcome.payoff, score=predicted,
                             greedy_side=greedy.side, greedy_payoff=greedy_outcome.payoff, motor_logit=choice.logit))
    if kind in INTERVENTION_KINDS and any(not torch.equal(readings[0][name], readings[1][name]) for name in readings[0]):
        raise AssertionError("Same-world current reinjection changed a prediction or action")
    return ({key: torch.stack([metrics[key] for metrics in readings], dim=1).cpu().numpy() for key in readings[0]}
            | {"forced_payoff": both_real.cpu().numpy(), "max_energy_error": np.array([energy_error])})


def evaluate(model, kind, seed, roots_per_angle):
    scene, angles, donor, mirror, geometry = heldout(seed, roots_per_angle)
    state_before = {name: val.detach().cpu().clone() for name, val in model.state_dict().items()}
    model.requires_grad_(False).eval()
    blocks = []
    step = ROOTS_PER_BATCH * REPEATS
    for first in range(0, len(scene.values), step):
        block = slice(first, first + step)
        blocks.append(run_batch(model, index_scene(scene, block), kind, donor[block] - first, mirror[block] - first))
    if any(not torch.equal(value.cpu(), state_before[name]) for name, value in model.state_dict().items()):
        raise AssertionError("Frozen evaluation changed fitted graph parameters or persistent buffers")
    measured = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    return measured | dict(angle=angles.numpy(), geometry=geometry.numpy(), donor=donor.numpy(), mirror=mirror.numpy(),
                           private_values=scene.values.numpy(), vectors=scene.vectors[::REPEATS].numpy(),
                           draws=scene.draws.numpy(),
                           max_energy_error=measured["max_energy_error"].max())


def artifact(kind, birth, split):
    return DATA / kind / f"{split}-birth-{birth}.npz"


def sources():
    return dict(eval_source=digest(__file__), world_source=digest(world.__file__), fit_source=digest(fit.__file__),
                plan=digest(PLAN))


def signature(kind, birth, split, checkpoint):
    return sources() | dict(checkpoint=checkpoint, kind=kind, birth=birth, split=split, seed=WORLDS[split],
                            roots_per_angle=ROOTS_PER_ANGLE)


def run(kind, birth, split, *, smoke=False):
    if smoke:
        model = world.ForkGraph(birth).eval()
        results = evaluate(model, kind, WORLDS[split], 2)
        print("Smoke", kind, birth, split, {name: value.shape for name, value in results.items()}, flush=True)
        return
    model, _, checkpoint = fit.load_final(kind, birth)
    output = artifact(kind, birth, split)
    inputs = signature(kind, birth, split, checkpoint)
    if output.is_file():
        with np.load(output) as old:
            if all(old.get(key, np.array(None)).item() == value for key, value in inputs.items()):
                return
    measurements = evaluate(model, kind, WORLDS[split], ROOTS_PER_ANGLE)
    storage._write_raw(output, **measurements, **inputs, modes=active_modes(kind))
    print(kind, birth, split, len(measurements["forced_payoff"]), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=fit.KINDS, action="append")
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS, action="append")
    parser.add_argument("--split", choices=WORLDS, action="append")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for kind, birth, split in product(args.kind or fit.KINDS, args.birth or fit.BIRTHS, args.split or WORLDS):
        run(kind, birth, split, smoke=args.smoke)


if __name__ == "__main__":
    main()
