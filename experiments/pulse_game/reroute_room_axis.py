"""Read and perturb a cross-map linear room direction in frozen reroute replicas."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from itertools import product

import numpy as np
import torch

from . import noon_market_stats as stats
from . import reroute_fit as fit
from . import reroute_fit_transfer as transfer
from . import reroute_novel as novel
from . import reroute_world as world


DATA = fit.DATA / "axis"
PLAN = fit.EVIDENCE / "axis-plan.md"
TIMES = (0, 1, world.MID_TICK, world.FORECAST_TICK, world.TRIAL_TICKS)
DRAW_SEEDS = (538241, 538297)
STAT_SEED = 538337
KINDS = transfer.KINDS
PROFILES = ("bounded", "unbounded")


def episode_keys(e):
    horizon = [getattr(e, field.name).reshape(len(e), -1) for field in fields(e)]
    return list(map(tuple, torch.cat(horizon, dim=1).long().tolist()))


def setup():
    episodes, _, case, heldout = novel.panel()
    truth = world.oracle(episodes)
    dictionary = {key: index for index, key in enumerate(episode_keys(episodes))}
    assert len(dictionary) == len(episodes)
    mirror = torch.tensor([dictionary[k] for k in episode_keys(world.swap_room_labels(episodes))])
    assert torch.equal(mirror[mirror], torch.arange(len(episodes)))
    assert torch.equal(truth.rooms, 1 - truth.rooms[mirror])
    previous = episodes.prior_endpoints.gather(1, episodes.door_rooms)
    assert torch.equal(previous, previous[mirror])
    for reference in (truth.endpoints, truth.payoffs, truth.prior_payoffs):
        assert torch.equal(reference, reference[mirror])
    fold = 3 - episodes.prior_endpoints.sum(1)
    unseen = torch.from_numpy(heldout)
    assert torch.equal(fold, fold[mirror]) and torch.equal(unseen, ~unseen[mirror])

    ends = truth.endpoints.gather(1, truth.rooms.argsort(1))
    room_values = episodes.goals.gather(1, ends)
    changed = torch.from_numpy(np.isin(case, (world.FLIP, world.STAY)))
    recipient = torch.where(unseen & changed & (room_values.abs() == 1).all(1))[0]
    e = episodes[recipient]
    old = e.prior_endpoints.gather(1, e.inspected_room[:, None])[:, 0]
    other = e.prior_endpoints.gather(1, 1 - e.inspected_room[:, None])[:, 0]
    neutral_goals = torch.zeros_like(e.goals)
    neutral_goals.scatter_(1, old[:, None], -1.)
    neutral_goals.scatter_(1, e.observed_endpoint[:, None], 1.)
    joined = replace(e, observed_endpoint=other)
    donors = [dictionary[k] for k in episode_keys(replace(joined, goals=neutral_goals))]
    assert not world.oracle(episodes[donors]).payoffs.any()
    assert (room_values[recipient, 0] == -room_values[recipient, 1]).all()
    return episodes, mirror, fold, unseen, recipient, torch.tensor(donors), joined, room_values[recipient, 0]


def trajectories(model, e):
    collected = []
    for start in range(0, len(e), 144):
        scene = e[start:start + 144]
        ready = world.prepare(model, scene)
        left = torch.zeros(len(scene), dtype=torch.long)
        truth = world.oracle(scene)
        activities = []
        for side, query in enumerate((left, 1 - left)):
            trial = world.try_side(model, ready, query)
            physical = world.settle(model, trial.final, scene, world.select(model, trial.final, None, 0., forced=query))
            for observed, expected in ((physical.room, truth.rooms), (physical.endpoint, truth.endpoints),
                                       (physical.paid, truth.payoffs)):
                assert torch.equal(observed, expected[:, side])
            activities.append(torch.stack([ready.h[:, world.REGION:]] +
                                          [trial.activities[t - 1, :, world.REGION:] for t in TIMES[1:]], dim=1))
        collected.append(torch.stack(activities, dim=1))
    return torch.cat(collected)


def linear_readout(activity, mirror, rooms, fold, unseen):
    """Fit a mean paired difference, with a fixed intercept learned on calibration worlds."""
    orientations = 2 * rooms - 1
    difference = (activity - activity[mirror]) * orientations[:, :, None, None]
    midpoint = (activity + activity[mirror]) / 2
    axes, intercepts = [], []
    for excluded in range(3):
        calibration = (~unseen) & (fold != excluded)
        d = difference[calibration].mean((0, 1))
        axes.append(d / d.norm(dim=1, keepdim=True).clamp_min(1e-10))
        intercepts.append((midpoint[calibration].mean((0, 1)) * axes[-1]).sum(-1))
    axes, intercepts = torch.stack(axes), torch.stack(intercepts)
    projection = (activity * axes[fold, None]).sum(-1) - intercepts[fold, None]
    paired = ((projection - projection[mirror]) * orientations[:, :, None])
    signed = projection * orientations[:, :, None]
    return axes, intercepts, signed, paired


def pair_perturbation(direction, donor_difference):
    scalar = (direction * donor_difference).sum(1, keepdim=True)
    along = scalar * direction / 2
    transverse = donor_difference - scalar * direction
    length = transverse.norm(dim=1, keepdim=True)
    if (length < 1e-8).any():
        raise AssertionError("Degenerate natural donor contrast in the orthogonal control")
    control = transverse * (scalar.abs() / (2 * length))
    torch.testing.assert_close(along.norm(dim=1), control.norm(dim=1), atol=2e-7, rtol=2e-6)
    torch.testing.assert_close((control * direction).sum(1), torch.zeros(len(along)), atol=3e-7, rtol=0)
    return along, control


def outcomes(model, trial, e, q, draws, payoffs):
    logit = model.logit(trial.final)
    right = logit.sigmoid()
    results = {"forecast": trial.forecast, "query_motor": torch.where(q.bool(), right, 1 - right),
               "expected_pay": (1 - right) * payoffs[:, 0] + right * payoffs[:, 1]}
    for seed, tape in draws.items():
        action = world.select(model, trial.final, tape, exploration=0.)
        actual = world.settle(model, trial.final, e, action)
        assert torch.equal(actual.paid, payoffs.gather(1, action.side[:, None])[:, 0])
        results[f"{seed}/query_chosen"] = (action.side == q).float()
        results[f"{seed}/real_pay"] = actual.paid
    return results


def intervene(model, e, joined, donors, r, fold, mirror, activity, axes):
    result = {}
    time = TIMES.index(world.MID_TICK)
    direction = axes[fold[r], time]
    q_draws = {seed: torch.from_numpy(np.random.default_rng(seed).random((len(r), 2), dtype=np.float32))
               for seed in DRAW_SEEDS}
    for layout, scene in (("separate", e[r]), ("joined", joined)):
        ready = world.prepare(model, scene)
        payoffs = world.oracle(scene).payoffs
        if layout == "joined":
            assert torch.equal(payoffs[:, 0], payoffs[:, 1])
        for side in (0, 1):
            q = torch.full((len(r),), side)
            chosen_room = e.door_rooms[donors, side]
            raw = activity[donors, side, time] - activity[mirror[donors], side, time]
            donor_difference = raw * (2 * chosen_room - 1)[:, None]
            along, control = pair_perturbation(direction, donor_difference)
            result[f"{layout}/{side}/norm"] = along.norm(dim=1)
            natural = world.try_side(model, ready, q)
            draws = {seed: tape[:, side] for seed, tape in q_draws.items()}
            peak_step = torch.maximum(along.abs(), control.abs()).clamp_min(1e-12)
            scale = ((1 - natural.mid_activity.abs()) / peak_step).amin(1).clamp(0., 1.)
            result[f"{layout}/{side}/scale"] = scale
            for profile in PROFILES:
                for arm, change in (("axis", along), ("orthogonal", control)):
                    step = change * scale[:, None] if profile == "bounded" else change
                    for donor_room in (0, 1):
                        delta = (2 * donor_room - 1) * step
                        midpoint = natural.mid_activity + delta
                        torch.testing.assert_close(midpoint - natural.mid_activity, delta, atol=7e-8, rtol=3e-6)
                        if profile == "bounded":
                            assert (midpoint.abs() <= 1.).all()
                        trial = world.try_side(model, ready, q, mid_override=midpoint)
                        before, after = natural.states[world.MID_TICK - 1], trial.states[world.MID_TICK - 1]
                        assert torch.equal(before.h[:, :world.REGION], after.h[:, :world.REGION])
                        for name in ("trace", "fast", "candidate"):
                            assert torch.equal(getattr(before, name), getattr(after, name))
                        key = f"{profile}/{layout}/{arm}/{side}/{donor_room}"
                        result[f"{key}/max_abs_activity"] = midpoint.abs().amax(1)
                        for metric, value in outcomes(model, trial, scene, q, draws, payoffs).items():
                            result[f"{key}/{metric}"] = value
    return {key: value.numpy() for key, value in result.items()}


@torch.no_grad()
def collect(kind, birth):
    model, _, checkpoint = transfer.load_final(kind, birth)
    e, mirror, fold, unseen, r, donors, joined, sign = setup()
    activity = trajectories(model, e)
    axes, intercepts, signed, paired = linear_readout(activity, mirror, e.door_rooms, fold, unseen)
    assert torch.equal(signed[:, 0, 0], -signed[:, 1, 0])
    raw = intervene(model, e, joined, donors, r, fold, mirror, activity, axes)
    fit.storage._write_raw(DATA / f"{kind}-{birth}.npz", activity=activity.numpy(), axes=axes.numpy(),
                           intercepts=intercepts.numpy(), signed=signed.numpy(), paired=paired.numpy(),
                           recipient=r.numpy(), donors=donors.numpy(), sign=sign.numpy(),
                           checkpoint=checkpoint, source=fit.digest(__file__), plan=fit.digest(PLAN), **raw)


def summaries():
    e, mirror, fold, unseen, r, donors, _, sign = setup()
    groups, hashes, activity_bounds = {}, {}, {}

    def save(group, name, value, maps):
        series = groups.setdefault(group, {}).setdefault(name, [])
        series.append([float(value[maps == i].mean()) for i in range(3)])

    def accuracy(x):
        return (x > 0) + .5 * (x == 0)

    measures = ("forecast", "query_motor", "expected_pay") + tuple(
        f"{seed}/{metric}" for seed in DRAW_SEEDS for metric in ("query_chosen", "real_pay"))
    for kind, birth in product(KINDS, transfer.BIRTHS):
        path = DATA / f"{kind}-{birth}.npz"
        hashes[path.name] = fit.digest(path)
        with np.load(path, allow_pickle=False) as z:
            assert z["checkpoint"].item() == fit.digest(transfer.destination(kind, birth))
            assert z["source"].item() == fit.digest(__file__) and z["plan"].item() == fit.digest(PLAN)
            np.testing.assert_array_equal(z["recipient"], r.numpy())
            np.testing.assert_array_equal(z["donors"], donors.numpy())
            reconstructed = linear_readout(torch.from_numpy(z["activity"]), mirror, e.door_rooms, fold, unseen)
            for key, value in zip(("axes", "intercepts", "signed", "paired"), reconstructed, strict=True):
                np.testing.assert_array_equal(z[key], value.numpy())
            bounds = activity_bounds.setdefault(path.name, {})
            for profile in PROFILES:
                magnitudes = np.stack([z[key] for key in z.files
                                       if key.startswith(profile + "/") and key.endswith("/max_abs_activity")])
                bounds[profile] = {"max_abs": float(magnitudes.max()), "rows_above_one": int((magnitudes > 1).sum()),
                                   "row_count": int(magnitudes.size)}
            scaling = np.stack([z[key] for key in z.files if key.endswith("/scale")])
            bounds["scale"] = {"median": float(np.median(scaling)), "zero": int((scaling == 0).sum()),
                               "row_count": int(scaling.size)}
            for t, tick in enumerate(TIMES):
                for key in ("signed", "paired"):
                    value = accuracy(z[key][unseen.numpy(), :, t]).mean(1)
                    save(kind + "/decode", f"{key}/t{tick}", value, fold[unseen].numpy())
            for profile, layout, arm in product(PROFILES, ("separate", "joined"), ("axis", "orthogonal")):
                for metric in measures:
                    values = []
                    for side in (0, 1):
                        entry = f"{profile}/{layout}/{arm}/{side}"
                        gap = z[f"{entry}/0/{metric}"] - z[f"{entry}/1/{metric}"]
                        if metric.endswith("pay"):
                            gap *= (1 - 2 * e.door_rooms[r, side].numpy())
                        else:
                            gap *= sign.numpy()
                        values.append(gap)
                    save(f"{kind}/{profile}/{layout}/{arm}", metric, np.mean(values, axis=0), fold[r].numpy())
    arrays = {kind: {name: np.asarray(s) for name, s in measures.items()} for kind, measures in groups.items()}
    comparisons = [("next_room/decode", "terminal/decode")]
    for kind, profile in product(KINDS, PROFILES):
        key = f"{kind}/{profile}"
        comparisons.extend(((f"{key}/separate/axis", f"{key}/joined/axis"),
                            (f"{key}/separate/axis", f"{key}/separate/orthogonal")))
    reports = stats.summarize(arrays, 3, transfer.BIRTHS, STAT_SEED, comparisons)
    fit.storage._write_json(fit.EVIDENCE / "axis-results.json", dict(
        reports=reports, raw_sha256=hashes, activity_bounds=activity_bounds,
        donor_queries=2 * int(unseen.sum()), recipient_queries=2 * len(r),
        test_prior_maps=[list(p) for p in novel.PERMS if tuple(p[:2]) in {(0, 2), (1, 0), (2, 1)}],
        times=TIMES, seeds=DRAW_SEEDS, source=fit.digest(__file__), plan=fit.digest(PLAN),
        fit_source=fit.digest(transfer.__file__), world_source=fit.digest(world.__file__),
        panel_source=fit.digest(novel.__file__)))
    for name, measures in reports.items():
        print(name, {k: [round(v["mean"], 4), *np.round(v["ci95"], 4)] for k, v in measures.items()}, flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--birth", type=int, choices=transfer.BIRTHS)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.kind is not None and args.birth is not None:
        collect(args.kind, args.birth)
    elif args.summarize:
        summaries()
    elif args.kind is None and args.birth is None:
        for kind, birth in product(KINDS, transfer.BIRTHS):
            collect(kind, birth)
            print(kind, birth, flush=True)
        summaries()
    else:
        p.error("Supply both --kind and --birth")


if __name__ == "__main__":
    main()
