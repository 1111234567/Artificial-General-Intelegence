"""Cross learned and starting sender weights before and during a frozen deliberation."""

from argparse import ArgumentParser
from itertools import product

import numpy as np
import torch

from . import fork_lesson_sender_credit_eval as ev


PLAN = ev.credit.EVIDENCE / "sender-credit-probe-plan.md"
DATA = ev.credit.DATA / "probe"
SEEDS = (ev.SEEDS[0], 69253917)
LEVELS = ("pp", "pf", "fp", "ff")
CONTRASTS = {"window_after_parent": ("pf", "pp"), "window_after_full": ("ff", "fp"),
             "prefix_before_parent": ("fp", "pp"), "prefix_before_full": ("ff", "pf"),
             "full_minus_parent": ("ff", "pp")}


def artifact(kind, birth, seed):
    return DATA / f"{seed}-{kind}-{birth}.npz"


def signature(kind, birth, seed, checkpoints):
    return ev.raw_signature(kind, birth, seed, checkpoints) | dict(probe_plan=ev.FIT.digest(PLAN),
                                                                  probe_source=ev.FIT.digest(__file__))


@torch.no_grad()
def cross(models, scene):
    stages = {key: ev.WORLD.cue(model, ev.WORLD.support(model, scene), scene) for key, model in models.items()}
    rows = {}
    for prefix, ending in LEVELS:
        model = models[ending]
        ready, activities, _ = ev.WORLD.choice_window(model, stages[prefix])
        action = ev.WORLD.choose(model, ready, scene)
        _, outcome = ev.WORLD.commit(model, ready, scene, action)
        endpoints = []
        for side in (0, 1):
            forced = ev.WORLD.choose(model, ready, scene, side=side)
            _, settlement = ev.WORLD.commit(model, ready, scene, forced)
            if forced.immediate.ne(0).any():
                raise AssertionError("The forced decision paid before the final world event")
            endpoints.append(settlement.payoff)
        truth = torch.stack(endpoints, dim=1)
        if action.immediate.ne(0).any() or not torch.equal(truth.gather(1, action.side[:, None])[:, 0], outcome.payoff):
            raise AssertionError("Crossed frozen choice did not settle its own door in the world")
        values = dict(score=ev.FIT.forecast_scores(activities), side=action.side, paid=outcome.payoff, truth=truth,
                      logit=action.logit)
        rows.update({f"{prefix}{ending}_{name}": value.numpy() for name, value in values.items()})
    return rows


def reference_match(values, kind, birth, seed):
    if seed in ev.SEEDS:
        with np.load(ev.artifact(kind, birth, seed)) as previous:
            for former, current in (("start", "pp"), ("full", "ff")):
                for key in ("score", "side", "paid", "truth", "logit"):
                    np.testing.assert_array_equal(previous[f"{former}_{key}"], values[f"{current}_{key}"])


def evaluate(kind, birth):
    models, checkpoints, _, _ = ev.load_models(kind, birth)
    models = dict(p=models["start"], f=models["full"])
    for seed in SEEDS:
        scene = ev.SENDER.factorial(seed)
        results = {}
        for i in range(0, len(scene.first), 256):
            batch = ev.scene_block(scene, slice(i, i + 256))
            for key, values in cross(models, batch).items():
                results.setdefault(key, []).append(values)
        rows = {key: np.concatenate(values) for key, values in results.items()}
        reference_match(rows, kind, birth, seed)
        ev.FIT.storage._write_raw(artifact(kind, birth, seed), **rows, **signature(kind, birth, seed, checkpoints))
        print("cross", kind, birth, seed, flush=True)


def measures(raw, scene):
    truth = ev.WORLD.oracle(scene).numpy()
    strict = truth[:, 0] != truth[:, 1]
    anchors = raw["pp_side"]
    rows = np.arange(len(truth))
    prepared = {}
    for level in LEVELS:
        prediction, side, paid, actual, logit = (raw[f"{level}_{metric}"] for metric in ("score", "side", "paid", "truth", "logit"))
        np.testing.assert_array_equal(truth, actual)
        np.testing.assert_array_equal(paid, truth[rows, side])
        np.testing.assert_array_equal(side, scene.draws.numpy() < torch.from_numpy(logit).sigmoid().numpy())
        error = (prediction.astype(np.float64) - truth) ** 2
        components = (error.mean(1), error[rows, 1 - side], error[rows, 1 - anchors], paid[strict] > 0, paid)
        prepared[level] = np.stack([metric.reshape(ev.ROOTS, -1).mean(1) for metric in components], axis=1)
    return prepared


def report(kind, seed, checkpoints):
    comparisons = {level: [] for level in LEVELS}
    raw_paths = {}
    scene = ev.SENDER.factorial(seed)
    for birth in ev.FIT.BIRTHS:
        path = artifact(kind, birth, seed)
        with np.load(path) as raw:
            metadata = signature(kind, birth, seed, checkpoints[birth])
            if any(raw[key].item() != value for key, value in metadata.items()):
                raise ValueError(f"The frozen factorial run has the wrong signature: {path}")
            reference_match(raw, kind, birth, seed)
            for level, matrix in measures(raw, scene).items():
                comparisons[level].append(matrix)
        raw_paths[path.relative_to(ev.FIT.ROOT).as_posix()] = ev.FIT.digest(path)
    levels = {key: np.stack(value) for key, value in comparisons.items()}
    changes = {name: levels[left] - levels[right] for name, (left, right) in CONTRASTS.items()}
    changes["interaction"] = changes["window_after_full"] - changes["window_after_parent"]
    bootstrap = ev.inference.CrossedBootstrap(len(ev.FIT.BIRTHS), ev.ROOTS)
    return dict(levels={name: ev.estimate(matrix, bootstrap) for name, matrix in levels.items()},
                paired_differences={name: ev.estimate(matrix, bootstrap) for name, matrix in changes.items()},
                raw_sha256=raw_paths)


def publish():
    results = {}
    for kind in ev.SENDER.KINDS:
        identities = {birth: ev.load_models(kind, birth)[1] for birth in ev.FIT.BIRTHS}
        results[kind] = {seed: report(kind, seed, identities) for seed in SEEDS}
        for seed, world in results[kind].items():
            print(kind, seed, {key: [round(values[name]["mean"], 5) for name in ("unchosen_mse", "strict_preferred")]
                               for key, values in world["levels"].items()}, flush=True)
    ev.FIT.storage._write_json(ev.credit.EVIDENCE / "sender-credit-probe-results.json",
                               dict(plan=ev.FIT.digest(PLAN), source=ev.FIT.digest(__file__), worlds=results,
                                    roots=ev.ROOTS, repeats=ev.REPEATS, seeds=SEEDS, births=ev.FIT.BIRTHS))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=ev.SENDER.KINDS)
    parser.add_argument("--birth", type=int, choices=ev.FIT.BIRTHS)
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
