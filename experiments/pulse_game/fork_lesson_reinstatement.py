"""Probe a property of the true earlier contact that the current menu never shows."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from itertools import product
from pathlib import Path
import json

import numpy as np
import torch

from . import fork_lesson_fit as fit
from . import fork_lesson_world as world


KINDS = ("whole", "reward", "sham")
CALIBRATION = 83179473
WORLDS = (83179603, 83179639)
CALIBRATION_ROOTS = 128
ROOTS = 96
BATCH = 16
RESAMPLES = 2000
SIDES = tuple(product((0, 1), repeat=2))
SIGNALS = ("shadow_activity", "return_current")
TIMES = ("before_menu", "last_menu", "first_blank", "late_blank")
VALUES = ("target", "other", "target_minus_other")
DATA = fit.DATA.parent / "reinstatement"
PLAN = fit.EVIDENCE / "reinstatement-plan.md"


class Trace:
    """Record ordinary post-tick shadow activity and the current that reached the main region."""

    def __init__(self, model):
        self.model = model
        self.tape = []

    def start(self, batch):
        return self.model.start(batch)

    def tick(self, state, sensory, **controls):
        state, back = self.model.tick(state, sensory, **controls)
        self.tape.append(torch.stack((state.h[:, world.REGION:], back), dim=1))
        return state, back

    def advance(self, state, sensory, ticks):
        for _ in range(ticks):
            state, _ = self.tick(state, sensory)
        return state


def root_batches(seed, count):
    roots = world.scenes(seed, count)
    for start in range(0, count, BATCH):
        stop = start + BATCH
        yield replace(roots, **{f.name: getattr(roots, f.name)[start:stop] for f in fields(roots)})


def distinct_rails(roots):
    e = replace(roots, **{f.name: getattr(roots, f.name).repeat_interleave(len(SIDES), 0) for f in fields(roots)})
    sides = torch.tensor(SIDES).repeat(len(roots.first), 1)
    order = torch.stack((e.first, 1 - e.first), dim=1)
    return replace(e, approach_sides=sides.gather(1, order)), 2 * sides - 1


def gather(values, columns):
    return values.gather(1, columns[:, None]).squeeze(1)


def old_contacts(model, e, labels):
    trace = Trace(model)
    state, visits = world.support(trace, e, return_receipts=True)
    history = torch.stack(trace.tape)
    phase = world.SUPPORT_TICKS // 2
    signatures = []
    for rank, visit in enumerate(visits):
        side = visit.inputs[-1, :, 6:8].argmax(1)
        if not torch.equal(2 * side - 1, gather(labels, visit.object_id)):
            raise AssertionError("The historical contact label disagrees with the actual physical pulse")
        end = (rank + 1) * phase
        signatures.append(history[end - world.TASTE_TICKS:end].mean(0))
    rank_vectors = torch.stack(signatures, dim=1)
    row = torch.arange(len(e.first))
    canonical = torch.stack((rank_vectors[row, e.first], rank_vectors[row, 1 - e.first]), dim=1)
    return state, canonical, history[-1]


def contact_probe(model, roots=CALIBRATION_ROOTS):
    sums = []
    for geometry in root_batches(CALIBRATION, roots):
        e, labels = distinct_rails(geometry)
        _, contact, _ = old_contacts(model, e, labels)
        sums.append((contact * labels[:, :, None, None]).sum(dim=(0, 1)))
    direction = torch.stack(sums).sum(0)
    scale = direction.norm(dim=1, keepdim=True)
    if (scale == 0).any():
        raise ValueError("No actual contact direction was found for this frozen model")
    return direction / scale


def project(activity, direction):
    return (activity * direction).sum(dim=-1)


def time_points(model, body, earlier, e, *, show=True):
    monitor = Trace(model)
    menu = world.cue(monitor, body, e, show=show)
    ready, activity, back = world.choice_window(model, menu)
    both = torch.stack((activity[:, :, world.REGION:], back), dim=2)
    return ready, torch.stack((earlier, monitor.tape[-1], both[0], both[-3:].mean(0)))


def average_rails(values):
    # values dimensions are time, row, signal, statistic.
    return values.reshape(values.shape[0], -1, len(SIDES), *values.shape[2:]).mean(dim=2).transpose(0, 1)


def sensitivity(projected, labels, target):
    sign = gather(labels, target)
    other = gather(labels, 1 - target)
    target = projected * sign[None, :, None]
    another = projected * other[None, :, None]
    return average_rails(torch.stack((target, another, target - another), dim=-1))


def test_geometry(model, geometry, direction):
    e, labels = distinct_rails(geometry)
    body, contacts, earlier = old_contacts(model, e, labels)
    contact = project(contacts, direction) * labels[:, :, None]
    contact = contact.reshape(len(geometry.first), len(SIDES), 2, len(SIGNALS)).mean(dim=1)
    _, absent = time_points(model, body, earlier, e, show=False)
    absent = project(absent, direction)
    truth_parts, choice_parts, choice_weights, absent_parts, correct = [], [], [], [], []
    dose = e.values.gather(1, e.door_order)
    row = torch.arange(len(e.first))
    for winner in (0, 1):
        payoff = torch.tensor((1., -1.) if winner == 0 else (-1., 1.))
        scene = replace(e, flips=payoff * dose)
        ready, temporal = time_points(model, body, earlier, scene)
        paid = torch.stack([world.commit(model, ready, scene, world.choose(model, ready, scene, side=s))[1].payoff
                            for s in (0, 1)], dim=1)
        if not torch.equal(paid, payoff.expand_as(paid)):
            raise AssertionError("A route's actual settlement disagrees with the planned strict choice")
        natural = world.choose(model, ready, scene)
        _, realized = world.commit(model, ready, scene, natural)
        if not torch.equal(realized.payoff, gather(paid, natural.side)):
            raise AssertionError("Natural and forced walks on the same route settled differently")
        correct.append((realized.payoff > 0).reshape(-1, len(SIDES)).float().mean(1))
        target = scene.door_order[:, winner]
        projected = project(temporal, direction)
        truth_parts.append(sensitivity(projected, labels, target))
        absent_parts.append(sensitivity(absent, labels, target))
        chosen = natural.side.reshape(-1, len(SIDES))
        stable = (chosen == chosen[:, :1]).all(1)
        choice_weights.append(stable.long())
        chosen_stats = sensitivity(projected, labels, scene.door_order[row, natural.side])
        choice_parts.append(chosen_stats * stable[:, None, None, None])
    return {
        "actual_contact_projection": contact.numpy(),
        "truth": torch.stack(truth_parts).mean(0).numpy(),
        "no_menu": torch.stack(absent_parts).mean(0).numpy(),
        "chosen_sum": torch.stack(choice_parts).sum(0).numpy(),
        "chosen_weights": torch.stack(choice_weights).sum(0).numpy(),
        "natural_right_outcome": torch.stack(correct).mean(0).numpy(),
    }


@torch.no_grad()
def evaluate(model, seed, roots):
    direction = contact_probe(model)
    records = [test_geometry(model, geometry, direction) for geometry in root_batches(seed, roots)]
    combined = {name: np.concatenate([record[name] for record in records]) for name in records[0]}
    return {"contact_probe": direction.numpy(), **combined}


def signature(checkpoint):
    sources = (__file__, PLAN, fit.__file__, fit.PROTOCOL, world.__file__)
    return {"checkpoint": checkpoint, "sources": {Path(p).name: fit.digest(p) for p in sources}}


def destination(kind, birth, seed):
    return DATA / f"{kind}-{birth}-{seed}.npz"


def run(kind, birth, seed, *, smoke=False):
    model, _, checkpoint = fit.load_final(kind, birth)
    provenance = json.dumps(signature(checkpoint), sort_keys=True)
    path = destination(kind, birth, seed)
    if path.is_file() and not smoke:
        with np.load(path, allow_pickle=False) as cached:
            if cached["signature"].item() != provenance:
                raise ValueError(f"The reinstatement artifact differs from active inputs at {path}")
        return
    result = evaluate(model, seed, BATCH if smoke else ROOTS)
    print(kind, birth, seed, result["truth"][:, :, 0, 2].mean(0).round(5).tolist(), flush=True)
    if not smoke:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, signature=provenance, **result)
        temporary.replace(path)


def statistics(rows, weights, rng):
    n, m = rows.shape[:2]
    births = rng.integers(n, size=(RESAMPLES, n, 1))
    roots = rng.integers(m, size=(RESAMPLES, 1, m))
    picked = rows[births, roots].sum(axis=(1, 2))
    amount = weights[births, roots].sum(axis=(1, 2))
    return {"mean": float(rows.sum() / weights.sum()), "ci95": np.quantile(picked / amount, (.025, .975)).tolist(),
            "birth": (rows.sum(axis=1) / weights.sum(axis=1)).tolist()}


def summarize_time(values, weights, seed):
    generator = np.random.default_rng(seed)
    to_result = lambda vals: statistics(vals, weights, generator)
    return {
        "sequence": {time: {k: to_result(values[:, :, t, j]) for j, k in enumerate(VALUES)}
                     for t, time in enumerate(TIMES)},
        "late_gain": {TIMES[t]: {VALUES[j]: to_result(values[:, :, -1, j] - values[:, :, t, j]) for j in (0, 2)}
                      for t in (1, 2)},
    }


def publish():
    output = {"calibration_seed": CALIBRATION, "calibration_roots": CALIBRATION_ROOTS, "roots_per_holdout": ROOTS,
              "rows_per_root": len(SIDES) * 2, "resamples": RESAMPLES, "results": {}}
    for seed in WORLDS:
        for kind in KINDS:
            raw = []
            for birth in fit.BIRTHS:
                path = destination(kind, birth, seed)
                with np.load(path, allow_pickle=False) as data:
                    if json.loads(data["signature"].item()) != signature(fit.digest(fit.destination(kind, birth))):
                        raise ValueError(f"The report has stale provenance at {path}")
                    raw.append({k: data[k] for k in data.files if k != "signature"})
            values = {k: np.stack([r[k] for r in raw]) for k in raw[0]}
            all_roots = np.ones_like(values["chosen_weights"])
            counts = values["chosen_weights"]
            models = {s: {
                "actual_contact": statistics(values["actual_contact_projection"][:, :, :, axis].mean(2),
                                             all_roots, np.random.default_rng(seed)),
                "true_best": summarize_time(values["truth"][:, :, :, axis, :], all_roots, seed),
                "absent_menu": summarize_time(values["no_menu"][:, :, :, axis, :], all_roots, seed),
                "subsequently_chosen": summarize_time(values["chosen_sum"][:, :, :, axis, :], counts, seed),
            } for axis, s in enumerate(SIGNALS)}
            output["results"].setdefault(str(seed), {})[kind] = {
                "actual_choice_consensus": (counts.mean(axis=1) / 2).tolist(),
                "natural_choice_accuracy": values["natural_right_outcome"].mean(axis=1).tolist(),
                "signals": models,
            }
    path = fit.EVIDENCE / "reinstatement-results.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS)
    parser.add_argument("--seed", type=int, choices=WORLDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    kinds = KINDS if args.kind is None else (args.kind,)
    births = fit.BIRTHS if args.birth is None else (args.birth,)
    seeds = WORLDS if args.seed is None else (args.seed,)
    for condition in product(kinds, births, seeds):
        run(*condition, smoke=args.smoke)
    if not args.smoke and all(destination(*condition).is_file() for condition in product(KINDS, fit.BIRTHS, WORLDS)):
        publish()


if __name__ == "__main__":
    main()
