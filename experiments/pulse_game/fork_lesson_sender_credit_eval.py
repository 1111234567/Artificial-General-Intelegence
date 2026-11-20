"""Settle both routes in two new worlds for the matched sender credit comparison."""

from argparse import ArgumentParser
from dataclasses import fields, replace
from itertools import product

import numpy as np
import torch

from . import edge_site_eval_stats as inference
from . import fork_lesson_sender_credit as credit


SENDER = credit.sender
FIT = credit.FIT
WORLD = credit.WORLD
SEEDS = (69253111, 69253189)
ARMS = ("start", "window", "full")
ROOTS = SENDER.ROOTS
REPEATS = 64
METRICS = ("two_door_mse", "unchosen_mse", "fixed_anchor_unchosen_mse", "strict_preferred", "net_return")
PAIRS = (("full", "window"), ("full", "start"), ("window", "start"))
LEDGER_FIELDS = ("draws", "probability", "action_right", "realized_payoff", "teacher_target", "chosen_forecast", "grad_norm")


def artifact(kind, birth, seed):
    return credit.DATA / "eval" / f"{seed}-{kind}-{birth}.npz"


def load_models(kind, birth):
    parent, sha = SENDER.load_parent(kind, birth)
    state = {name: val.detach().clone() for name, val in parent.state_dict().items()}
    old, prior = credit.prior_run(kind, birth, sha)
    checkpoint = credit.path(kind, birth)
    full = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if full["signature"] != credit.signature(kind, birth, sha, prior):
        raise ValueError(f"Wrong extended-credit identity: {checkpoint}")
    models = {"start": parent}
    for arm, payload in (("window", old), ("full", full)):
        model = WORLD.ForkGraph(birth).requires_grad_(False).eval()
        model.load_state_dict(payload["body"])
        credit.verify_unchanged(model, state)
        models[arm] = model
    return models, dict(start=sha, window=prior, full=FIT.digest(checkpoint)), full, old


def raw_signature(kind, birth, seed, checkpoints):
    return dict(kind=kind, birth=birth, seed=seed, roots=ROOTS, plan=FIT.digest(credit.PLAN),
                source=FIT.digest(__file__), **{f"checkpoint_{key}": value for key, value in checkpoints.items()})


def scene_block(scene, subset):
    return replace(scene, **{field.name: getattr(scene, field.name)[subset] for field in fields(scene)})


@torch.no_grad()
def execute(model, scene):
    prefix = WORLD.cue(model, WORLD.support(model, scene), scene)
    ready, activity, _ = WORLD.choice_window(model, prefix)
    scores = FIT.forecast_scores(activity)
    action = WORLD.choose(model, ready, scene)
    _, actual = WORLD.commit(model, ready, scene, action)
    outcomes = []
    for side in (0, 1):
        forced = WORLD.choose(model, ready, scene, side=side)
        _, settled = WORLD.commit(model, ready, scene, forced)
        if forced.immediate.ne(0).any() or not torch.equal(forced.side, torch.full_like(forced.side, side)):
            raise AssertionError("A forced route did not commit its assigned door at zero immediate payoff")
        outcomes.append(settled.payoff)
    truth = torch.stack(outcomes, dim=1)
    if action.immediate.ne(0).any() or not torch.equal(actual.payoff, truth.gather(1, action.side[:, None])[:, 0]):
        raise AssertionError("Autonomous choice failed its real, delayed settlement")
    if not torch.equal(truth, WORLD.oracle(scene)) or not torch.isfinite(scores).all():
        raise AssertionError("Forked true settlements or forecast values failed validation")
    return dict(score=scores.numpy(), side=action.side.numpy(), paid=actual.payoff.numpy(), truth=truth.numpy(),
                logit=action.logit.numpy())


def evaluate(kind, birth):
    models, checkpoints, _, _ = load_models(kind, birth)
    for seed in SEEDS:
        location = artifact(kind, birth, seed)
        expected = raw_signature(kind, birth, seed, checkpoints)
        if location.is_file():
            with np.load(location) as old:
                if all(old[key].item() == value for key, value in expected.items()):
                    continue
        scene = SENDER.factorial(seed)
        raw = {}
        for first in range(0, len(scene.first), 256):
            batch = scene_block(scene, slice(first, first + 256))
            for arm, model in models.items():
                for name, value in execute(model, batch).items():
                    raw.setdefault(f"{arm}_{name}", []).append(value)
        FIT.storage._write_raw(location, **{key: np.concatenate(value) for key, value in raw.items()}, **expected)
        print("deploy", kind, birth, seed, len(scene.first), flush=True)


def measures(raw, expected, scene):
    for key, value in expected.items():
        if raw[key].item() != value:
            raise ValueError(f"Unverified frozen rollout signature: {key}")
    y = WORLD.oracle(scene).numpy()
    eligible = y[:, 0] != y[:, 1]
    if not (eligible.reshape(ROOTS, REPEATS).sum(1) == REPEATS // 2).all():
        raise AssertionError("The paired worlds did not balance strict cases within each root")
    rows = np.arange(len(y))
    anchor = raw["start_side"]
    arms = {}
    for arm in ARMS:
        score, side, paid, truth, logit = (raw[f"{arm}_{name}"] for name in ("score", "side", "paid", "truth", "logit"))
        np.testing.assert_array_equal(truth, y)
        np.testing.assert_array_equal(paid, y[rows, side])
        np.testing.assert_array_equal(side, scene.draws.numpy() < torch.from_numpy(logit).sigmoid().numpy())
        error = (score.astype(np.float64) - truth) ** 2
        strict = (paid[eligible] > 0).reshape(ROOTS, REPEATS // 2)

        def by_root(metric):
            return metric.reshape(ROOTS, REPEATS).mean(1)

        values = np.column_stack((error.mean(1).reshape(ROOTS, REPEATS).mean(1), by_root(error[rows, 1 - side]),
                                  by_root(error[rows, 1 - anchor]), strict.mean(1), by_root(paid)))
        arms[arm] = dict(measures=values, strict_successes=int(strict.sum()), strict_denominator=int(strict.size))
    return arms


def estimate(matrix, bootstrap):
    births = matrix.mean(1)
    bounds = bootstrap.intervals(matrix.astype(np.float32))
    return {metric: dict(mean=float(births[:, i].mean()), ci95=bounds[:, i].tolist(), by_birth=births[:, i].tolist())
            for i, metric in enumerate(METRICS)}


def report(kind, seed, identity):
    scene = SENDER.factorial(seed)
    arrays, paths = {}, {}
    for birth in FIT.BIRTHS:
        location = artifact(kind, birth, seed)
        with np.load(location) as raw:
            groups = measures(raw, raw_signature(kind, birth, seed, identity[birth]), scene)
        for arm in ARMS:
            arrays.setdefault(arm, []).append(groups[arm])
        paths[location.relative_to(FIT.ROOT).as_posix()] = FIT.digest(location)
    values = {arm: np.stack([birth["measures"] for birth in entries]) for arm, entries in arrays.items()}
    bootstrap = inference.CrossedBootstrap(len(FIT.BIRTHS), ROOTS)
    levels = {arm: estimate(matrix, bootstrap) for arm, matrix in values.items()}
    for arm, data in levels.items():
        reference = data["strict_preferred"]
        reference.update(successes_by_birth=[entry["strict_successes"] for entry in arrays[arm]],
                         denominator=int(sum(entry["strict_denominator"] for entry in arrays[arm])),
                         successes=int(sum(entry["strict_successes"] for entry in arrays[arm])))
    differences = {f"{a}_minus_{b}": estimate(values[a] - values[b], bootstrap) for a, b in PAIRS}
    return dict(levels=levels, paired_differences=differences, raw_sha256=paths)


def publish():
    cohorts = {}
    for kind in SENDER.KINDS:
        checks, sources = {}, {}
        for birth in FIT.BIRTHS:
            _, sources[birth], full, old = load_models(kind, birth)
            check = credit.audit_ledger(kind, birth, full["ledger"], old["ledger"])
            if check != full["audit"]:
                raise AssertionError("Recomputation disagreed with the extended-credit checkpoint audit")
            checks[birth] = dict(changed=full["changed"], possible=full["possible"], max_change=full["max_change"],
                                 initial_gradient=full["initial_gradient"], ledger=check,
                                 actions_diverging_from_window=int(full["ledger"][:, :, 2].ne(old["ledger"][:, :, 2]).sum()))
        cohorts[kind] = dict(worlds={seed: report(kind, seed, sources) for seed in SEEDS},
                             checkpoint_sha256=sources, checks=checks)
    summary = dict(plan=FIT.digest(credit.PLAN), fit_source=FIT.digest(credit.__file__), eval_source=FIT.digest(__file__),
                   seeds=SEEDS, roots_per_seed=ROOTS, factor_cells=REPEATS, births=FIT.BIRTHS,
                   training_domain=credit.DOMAIN, training_steps=SENDER.STEPS, routes_per_step=credit.BATCH,
                   ledger_columns=LEDGER_FIELDS, bootstrap_draws=inference.N_BOOT, cohorts=cohorts)
    FIT.storage._write_json(credit.EVIDENCE / "sender-credit-results.json", summary)
    for kind, seed in product(SENDER.KINDS, SEEDS):
        world = cohorts[kind]["worlds"][seed]
        print(kind, seed, {arm: {k: round(v["mean"], 5) for k, v in row.items()}
                           for arm, row in world["levels"].items()}, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=SENDER.KINDS)
    parser.add_argument("--birth", type=int, choices=FIT.BIRTHS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.kind is None and args.birth is None:
        publish()
    elif args.kind is not None and args.birth is not None:
        evaluate(args.kind, args.birth)
    else:
        parser.error("Kind and birth must be supplied together")


if __name__ == "__main__":
    main()
