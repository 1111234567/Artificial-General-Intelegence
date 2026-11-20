"""Measure what a frozen native graph's silent activity predicts about B writes."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as storage
from . import native_rehearsal_fit as fit
from . import native_rehearsal_graph as graph


DATA = fit.DATA / "prospect"
EVIDENCE = fit.EVIDENCE
PLAN = EVIDENCE / "prospect-plan.md"
KINDS = ("copy", "guide_a", "guide_c")
BIRTHS = fit.BIRTHS
SPLITS = {"calibration": (82617281, 128), "heldout": (82617289, 192)}
OBJECTS = ("a", "b")
REPEATS = 16
ROOTS_PER_BATCH = 12
SAMPLE_TICKS = (3, 6, 9, 12)
SWAP_OLD_HISTORY = 12  # A mirror and polarity change within a root; B and C stay paired.
MODES = ("native", "frozen", "plus", "minus", "swap_native", "swap_frozen")


def extract_activity(activities):
    indices = [t - 1 for t in SAMPLE_TICKS]
    return activities[indices].transpose(0, 1).flatten(1)


def dose(delta):
    values = delta.flatten(1)
    return torch.stack((values.abs().sum(1), values.norm(dim=1), values.clamp_min(0).sum(1),
                        values.clamp_max(0).sum(1), values.abs().amax(1), values.ne(0).sum(1)), dim=1)


def linear_features(before, after, change, modified, reference):
    base = torch.cat((before.h, before.trace, after.h, after.trace, dose(change), extract_activity(reference)), 1)
    return base, extract_activity(modified - reference)


def opposite_writes(before, after, rng):
    delta = after.fast - before.fast
    mask = torch.from_numpy(rng.random(delta.shape, dtype=np.float32) < 0.5)
    room = (1.15 - before.fast.abs()).clamp_min(0)
    change = delta.sign() * torch.minimum(delta.abs(), room) * mask
    plus, minus = (replace(after, fast=before.fast + direction * change) for direction in (1, -1))
    magnitude_error = (plus.fast - before.fast).abs() - (minus.fast - before.fast).abs()
    return plus, minus, float(magnitude_error.abs().max())


def matching_donor(scene):
    index = torch.arange(len(scene.laws)) ^ SWAP_OLD_HISTORY
    if not torch.equal(scene.laws[index, 1:], scene.laws[:, 1:]):
        raise AssertionError("The wrong-history donor shifted B or C laws")
    if not torch.equal(scene.vectors[index, 1:], scene.vectors[:, 1:]):
        raise AssertionError("The wrong-history donor shifted B or C cues")
    if not torch.equal(scene.laws[index, 0], -scene.laws[:, 0]):
        raise AssertionError("The donor did not reverse the older A law")
    return index


def counterfactual_draw(scene, first_side):
    draws = scene.draws.clone()
    # Sigmoid motor probabilities are strictly between 0 and 1 in this model.
    draws[:, 3] = 0 if first_side == 1 else 1
    return replace(scene, draws=draws)


def future(model, state, scene):
    """Use actual future menu presentations and integrate each possible first motor outcome."""
    rows = torch.arange(len(state.h))
    first = scene.first_query
    action, _, payout, _ = graph.queries(model, state, scene)
    possible = [graph.queries(model, state, counterfactual_draw(scene, side)) for side in (1, -1)]
    right, left = (rollout[3] for rollout in possible)
    if not torch.equal(right[rows, first], left[rows, first]):
        raise AssertionError("First motor response changed before its choice")
    first_factor = scene.laws[rows, 2 * first] * scene.orientation[:, 3]
    rightward = (right[rows, first] * first_factor).sigmoid()
    correct_prob = torch.lerp(left.sigmoid(), right.sigmoid(), rightward[:, None])
    # Check the logged choices and payouts against the sampled two-question route.
    if not (torch.isin(action, action.new_tensor((-1, 1))) & torch.isin(payout, payout.new_tensor((-1, 1)))).all():
        raise AssertionError("The future motor log contains a nonphysical act or payout")
    return correct_prob, action.to(torch.int8), payout.to(torch.int8)


def slice_scene(scene, start, end):
    return replace(scene, **{field.name: getattr(scene, field.name)[start:end] for field in fields(scene)})


@torch.no_grad()
def run_batch(model, scene, rng, *, intervention):
    before, after, _, _ = graph.support(model, scene)
    frozen = replace(after, fast=before.fast)
    successor, activities, currents = graph.silence(model, after)
    quiet_frozen, activity_frozen, current_frozen = graph.silence(model, frozen)
    base, change = linear_features(before, after, after.fast - before.fast, activities, activity_frozen)
    donor = matching_donor(scene)
    states = [successor, quiet_frozen]
    features = dict(base=base, activity_delta=change)
    maximum_dose_error = 0.0
    if intervention:
        plus, minus, maximum_dose_error = opposite_writes(before, after, rng)
        if maximum_dose_error > 2e-7:
            raise AssertionError("Opposite-write conditions failed absolute dose matching")
        quiet_plus, activity_plus, _ = graph.silence(model, plus)
        quiet_minus, activity_minus, _ = graph.silence(model, minus)
        swap_native, activity_swap_native, _ = graph.silence(model, after, override=currents[:, donor])
        swap_frozen, activity_swap_frozen, _ = graph.silence(model, frozen, override=current_frozen[:, donor])
        base_sign, delta_sign = linear_features(before, after, plus.fast - minus.fast, activity_plus, activity_minus)
        base_swap, delta_swap = linear_features(before, after, after.fast - before.fast,
                                                activity_swap_native, activity_swap_frozen)
        features.update(base_sign=base_sign, activity_sign=delta_sign, base_swap=base_swap,
                        activity_swap=delta_swap, activity_donor=change[donor])
        states.extend((quiet_plus, quiet_minus, swap_native, swap_frozen))
    future_prob, action, payout = (torch.stack(part, dim=1) for part in zip(
        *(future(model, state, scene) for state in states), strict=True))
    return dict(**features, future_prob=future_prob, actions=action, payouts=payout, angle=scene.angle,
                first_query=scene.first_query, dose_error=torch.tensor([maximum_dose_error]))


def deploy(model, split, *, smoke=False):
    seed, roots = SPLITS[split]
    if smoke:
        roots = 2
    scene = graph.scenes(seed, roots, evaluation=True)
    random_mask = np.random.default_rng([seed, 706713])
    rows = [run_batch(model, slice_scene(scene, start, start + ROOTS_PER_BATCH * REPEATS), random_mask,
                      intervention=split == "heldout")
            for start in range(0, len(scene.laws), ROOTS_PER_BATCH * REPEATS)]
    return {key: torch.cat([row[key] for row in rows], dim=0).numpy() for key in rows[0]}


def checkpoint(kind, birth):
    return fit.DATA / kind / f"seed-{birth}.pt"


def artifact(kind, birth, split):
    return DATA / kind / f"{split}-seed-{birth}.npz"


def run(kind, birth, *, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if smoke:
        model = graph.NativeGraph(birth, "copy").eval()
        data = deploy(model, "heldout", smoke=True)
        print({name: value.shape for name, value in data.items()}, flush=True)
        return
    source = fit.digest(__file__)
    plan = fit.digest(PLAN)
    body = checkpoint(kind, birth)
    checkpoint_sha = fit.digest(body)
    payload = torch.load(body, map_location="cpu", weights_only=True)
    if (payload["kind"], payload["birth"], payload["step"]) != (kind, birth, fit.STEPS):
        raise ValueError(f"Unexpected checkpoint identity: {body}")
    if payload["signature"]["graph"] != fit.digest(graph.__file__):
        raise ValueError(f"Checkpoint was trained on different graph code: {body}")
    model = graph.NativeGraph(birth, payload["layout"]).eval()
    model.load_state_dict(payload["state"])
    for split, (seed, roots) in SPLITS.items():
        output = artifact(kind, birth, split)
        if output.exists():
            with np.load(output) as old:
                if (str(old["source"]), str(old["plan"]), str(old["checkpoint"])) == (source, plan, checkpoint_sha):
                    continue
        data = deploy(model, split)
        storage._write_raw(output, **data, roots=roots, seed=seed, kind=kind, birth=birth, split=split,
                           source=source, plan=plan, checkpoint=checkpoint_sha,
                           modes=MODES if split == "heldout" else MODES[:2])
        print(kind, birth, split, roots, flush=True)


if __name__ == "__main__":
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("birth", type=int, choices=BIRTHS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.kind, args.birth, smoke=args.smoke)
