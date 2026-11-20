"""Audit frozen native request ranking and actual actions in independent factorial worlds."""

from argparse import ArgumentParser
from itertools import product

import numpy as np
import torch

from . import fork_lesson_eval as heldout
from . import fork_lesson_native_request_fit as fit
from . import edge_site_eval_stats as bootstrap


NATIVE = fit.native
WORLD = fit.WORLD
SEEDS = (69278519, 69278653)
ROOTS_PER_ANGLE = 32
MODES = ("natural", "blank", "cut_return", "isolate_return")
METRICS = ("signed_accuracy", "rank_accuracy", "rank_margin", "two_door_mse", "unchosen_mse",
           "actual_preferred", "net_return", "motor_margin", "forced_motor_move")


def artifact(kind, birth, seed):
    return fit.DATA / "eval" / f"{kind}-{birth}-{seed}.npz"


def signature(kind, birth, seed, checkpoint):
    return dict(kind=kind, birth=birth, seed=seed, checkpoint=checkpoint, plan=fit.prior.digest(fit.PLAN),
                source=fit.prior.digest(__file__), roots_per_angle=ROOTS_PER_ANGLE, modes=MODES)


def generate(seed, roots_per_angle):
    scenes, angles, donor, mirror, geometry = heldout.heldout(seed, roots_per_angle)
    generators = np.random.default_rng([seed, 701])
    query = torch.from_numpy(generators.random(len(angles), dtype=np.float32)).repeat_interleave(heldout.REPEATS)
    return scenes, angles, donor, mirror, geometry, query


@torch.no_grad()
def deploy_batch(model, scene, query_draw):
    prepared = NATIVE.prepare(model, scene)
    natural = NATIVE.select(model, prepared, query_draw, exploration=0.)
    samples = []
    truth = []
    ordinal = tuple(torch.full_like(natural.side, value) for value in (0, 1))
    for mode in MODES:
        forced = tuple(NATIVE.try_side(model, prepared, side, mode=mode) for side in ordinal)
        left, right = forced
        if not torch.equal(left.first_primary, right.first_primary):
            raise AssertionError("The query reached the primary neurons on its injection tick")
        if mode in ("blank", "isolate_return") and not torch.equal(left.forecast, right.forecast):
            raise AssertionError("A blocked query changed the native forecast")
        hypotheses = torch.stack((left.forecast, right.forecast), dim=1)
        forced_motor = torch.stack([model.logit(trial.final) for trial in forced], dim=1)
        if mode == "natural":
            for side, trial in enumerate(forced):
                act = NATIVE.select(model, trial.final, scene.draws, exploration=0., forced=side)
                truth.append(NATIVE.settle(model, trial.final, scene, act))
            truth = torch.stack(truth, dim=1)
            if not torch.equal(truth, WORLD.oracle(scene)):
                raise AssertionError("Actual forced routes disagree with the independent world reference")
        rollout = NATIVE.try_side(model, prepared, natural.side, mode=mode)
        if not torch.allclose(rollout.forecast, hypotheses.gather(1, natural.side[:, None])[:, 0], atol=1e-6, rtol=0):
            raise AssertionError("The free trial did not follow its native forced-query counterpart")
        action = NATIVE.select(model, rollout.final, scene.draws, exploration=0.)
        paid = NATIVE.settle(model, rollout.final, scene, action)
        if not torch.equal(paid, truth.gather(1, action.side[:, None])[:, 0]):
            raise AssertionError("Actual choice did not reach the matched settled consequence")
        samples.append(dict(prediction=hypotheses, query=natural.side, query_logit=natural.logit,
                            action=action.side, motor=action.logit, settled=paid, forced_motor=forced_motor))
    return {key: torch.stack([row[key] for row in samples], dim=1).numpy() for key in samples[0]} | {"truth": truth.numpy()}


def evaluate(kind, birth, *, smoke=False):
    model, _, checkpoint = fit.load(kind, birth) if not smoke else (*fit.make_model(birth), None)
    if len(list(model.modules())) != 1 or model.width != 40 or model.trial_gain.numel() != 2:
        raise AssertionError("Request body added a module, a hidden unit, or more than two pulse gains")
    for seed in SEEDS[:1] if smoke else SEEDS:
        scene, angles, _, _, geometry, query = generate(seed, 1 if smoke else ROOTS_PER_ANGLE)
        collected = []
        for first in range(0, len(scene.first), 256):
            idx = slice(first, first + 256)
            collected.append(deploy_batch(model, heldout.index_scene(scene, idx), query[idx]))
        raw = {key: np.concatenate([sample[key] for sample in collected]) for key in collected[0]}
        if not smoke:
            fit.prior.storage._write_raw(artifact(kind, birth, seed), **raw, angles=angles.numpy(), geometry=geometry.numpy(),
                                         **signature(kind, birth, seed, checkpoint))
        print(kind, birth, seed, len(scene.first), flush=True)


def rate_of_positive(value):
    return np.where(value == 0, .5, value > 0)


def records(raw, scene, angles, geometry, draw, expected):
    for key, value in expected.items():
        if key == "modes":
            np.testing.assert_array_equal(raw[key], value)
        elif raw[key].item() != value:
            raise AssertionError(f"Unexpected provenance of native evaluation field {key}")
    np.testing.assert_array_equal(raw["angles"], angles.numpy())
    np.testing.assert_array_equal(raw["geometry"], geometry.numpy())
    true = WORLD.oracle(scene).numpy()
    np.testing.assert_array_equal(raw["truth"], true)
    scores = raw["prediction"].astype(np.float64)
    if not np.isfinite(scores).all() or (np.abs(scores) > 1).any():
        raise AssertionError("Native prediction has an invalid scale")
    query_logit, motor_logit, query, action = (raw[key] for key in ("query_logit", "motor", "query", "action"))
    np.testing.assert_array_equal(query, draw.numpy()[:, None] < torch.from_numpy(query_logit).sigmoid().numpy())
    np.testing.assert_array_equal(action, scene.draws.numpy()[:, None] < torch.from_numpy(motor_logit).sigmoid().numpy())
    np.testing.assert_array_equal(raw["settled"], np.take_along_axis(true, action, axis=1))
    ideal = (true[:, 1] - true[:, 0]) / 2
    rank = (scores[:, :, 1] - scores[:, :, 0]) * ideal[:, None]
    signed = rate_of_positive(scores * true[:, None, :]).mean(2)
    error = (scores - true[:, None, :]) ** 2
    holdout = np.take_along_axis(error, (1 - action)[:, :, None], axis=2)[:, :, 0]
    actual = raw["settled"]
    motor_move = np.diff(raw["forced_motor"], axis=2).squeeze(2)
    measured = np.stack((signed, rate_of_positive(rank), rank, error.mean(2), holdout, actual > 0, actual,
                         motor_logit * ideal[:, None], motor_move), axis=2)
    strict = ideal != 0
    close = np.repeat(angles.numpy() < 85, heldout.REPEATS)
    masks = dict(strict=strict, close_strict=strict & close,
                 close_opposite_private=strict & close & (scene.values[:, 0] != scene.values[:, 1]).numpy())
    return measured, masks


def group(values, mask):
    chunked = values.reshape(len(values), -1, heldout.REPEATS, len(MODES), len(METRICS))
    selected = mask.reshape(-1, heldout.REPEATS)
    alive = selected.any(1)
    kept = selected[alive]
    return np.einsum("brnmk,rn->brmk", chunked[:, alive], kept) / kept.sum(1)[None, :, None, None]


def summarize(matrix, keys):
    engine = bootstrap.CrossedBootstrap(*matrix.shape[:2])
    ci = engine.intervals(matrix.astype(np.float32))
    births = matrix.mean(1)
    return {name: {metric: dict(mean=float(births[:, mode, field].mean()),
                                 ci95=ci[:, mode, field].tolist(), by_birth=births[:, mode, field].tolist())
                   for field, metric in enumerate(METRICS)} for mode, name in enumerate(keys)}


def summarize_seed(seed, identities):
    scene, angles, _, _, geometry, draw = generate(seed, ROOTS_PER_ANGLE)
    paths, runs = {}, {}
    for kind in fit.KINDS:
        cohort = []
        for birth in fit.BIRTHS:
            body = artifact(kind, birth, seed)
            with np.load(body) as data:
                values, masks = records(data, scene, angles, geometry, draw, signature(kind, birth, seed, identities[kind, birth]))
                cohort.append(values)
            paths[body.relative_to(fit.prior.ROOT).as_posix()] = fit.prior.digest(body)
        runs[kind] = np.stack(cohort)
    result = {}
    for subset, mask in masks.items():
        means = {kind: group(cohort, mask) for kind, cohort in runs.items()}
        within = {kind: summarize(mean[:, :, :1] - mean[:, :, 1:], MODES[1:]) for kind, mean in means.items()}
        between = summarize(means["binding"] - means["misbound"], MODES)
        result[subset] = dict(rows_per_birth=int(mask.sum()), rates={kind: summarize(mean, MODES) for kind, mean in means.items()},
                              binding_minus_misbound=between, natural_minus_control=within)
    return dict(subsets=result, raw_sha256=paths)


def positive(metric, ref=0.):
    return metric["ci95"][0] > ref and min(metric["by_birth"]) > ref


def publish():
    fitted = {}
    identities = {}
    for kind, birth in product(fit.KINDS, fit.BIRTHS):
        model, saved, digest = fit.load(kind, birth)
        audited = fit.audit(kind, birth, saved["ledger"])
        if audited != saved["audit"] or saved["changed_entries"]["mask"] != 0 or len(list(model.modules())) != 1:
            raise AssertionError("Native fit failed training audit or changed architectural support")
        fitted[f"{kind}-{birth}"] = dict(**audited, sha256=digest, changes=saved["changed_entries"],
                                          trial_gain=model.trial_gain.detach().tolist(), units=model.width,
                                          parameters=sum(p.numel() for p in model.parameters()),
                                          last64=dict(zip(fit.CURVE, saved["curve"][-64:].mean(0).tolist(), strict=True)))
        identities[kind, birth] = digest
    results = {seed: summarize_seed(seed, identities) for seed in SEEDS}
    gates = {}
    for seed, result in results.items():
        m = result["subsets"]["strict"]
        rates = m["rates"]["binding"]["natural"]
        controls = m["natural_minus_control"]["binding"]
        gates[seed] = dict(rank_above_chance=positive(rates["rank_accuracy"], .5),
                           rank_above_misbound=positive(m["binding_minus_misbound"]["natural"]["rank_accuracy"]),
                           rank_depends_on_return=positive(controls["cut_return"]["rank_accuracy"]),
                           rank_depends_on_pulse=positive(controls["blank"]["rank_accuracy"]),
                           action_depends_on_return=positive(controls["cut_return"]["actual_preferred"]),
                           action_depends_on_pulse=positive(controls["blank"]["actual_preferred"]))
    report = dict(plan=fit.prior.digest(fit.PLAN), fit_source=fit.prior.digest(fit.__file__), source=fit.prior.digest(__file__),
                  training_steps=fit.STEPS, batch=fit.BATCH, birth=fit.BIRTHS, roots_per_angle=ROOTS_PER_ANGLE,
                  boot_draws=bootstrap.N_BOOT, fits=fitted, gates=gates, worlds=results)
    fit.prior.storage._write_json(fit.EVIDENCE / "native-request-results.json", report)
    for seed, result in results.items():
        m = result["subsets"]["strict"]
        print(seed, gates[seed], {kind: {mode: [round(m["rates"][kind][mode][metric]["mean"], 4)
                                                      for metric in ("rank_accuracy", "actual_preferred", "unchosen_mse")]
                                         for mode in MODES} for kind in fit.KINDS}, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=fit.KINDS)
    parser.add_argument("--birth", type=int, choices=fit.BIRTHS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.kind is None and args.birth is None and not args.smoke:
        publish()
    elif args.kind is not None and args.birth is not None:
        evaluate(args.kind, args.birth, smoke=args.smoke)
    else:
        parser.error("Supply both kind and birth to deploy, or omit both to publish")


if __name__ == "__main__":
    main()
