"""Train a small exploration gate from real net returns and compare held controls."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import active_symmetry_agent as policy, active_symmetry_world as world
from .spike_analysis import estimates


def subset(panel, index):
    seeds = tuple(panel.seeds[i] for i in index)
    markers = world.body_region(panel.markers, index, 0, panel.ticks, 0)
    protocols = []
    for original in panel.protocols:
        base = original.menu
        child = replace(base, body=world.body_region(base.body, index, 0, base.body.trials, 0),
                        chemicals=base.chemicals[index], probability=base.probability[index], realized=base.realized[index])
        protocols.append(replace(original, menu=child, family=original.family[index], related_side=original.related_side[index],
                                 query=original.query[index], anchor_sign=original.anchor_sign[index], pairmates=original.pairmates[index]))
    return replace(panel, seeds=seeds, arena=panel.arena[index], horizon=panel.horizon[index], costs=panel.costs[index],
                   probe_side=panel.probe_side[index], markers=markers, protocols=tuple(protocols))


def first_blind(record):
    started = record["harvest"].any(1)
    by_world = np.arange(len(started))
    first = record["rewards"][by_world, record["harvest"].argmax(1)]
    return (first > 0) & started, started


def row_output(record):
    correct, exists = first_blind(record)
    return {"net": record["net"].tolist(), "each_probe": record["probe"].tolist(),
            "probe_counts": record["counts"].tolist(), "gross": record["rewards"].sum(1).tolist(),
            "fees": record["paid"].sum(1).tolist(), "first": correct.tolist(), "had_harvest": exists.tolist(),
            "gate_arrived": record["landed"].tolist(), "gain_before_gate": record["confidence"].tolist()}


def train_group(train, validate, build_seed, count, batch):
    rng = np.random.default_rng(build_seed)
    dimension = policy.WIDTH
    actor = .03 * rng.normal(size=dimension)
    critic_gram = np.eye(dimension)
    critic_signal = np.zeros(dimension)
    critic = np.zeros(dimension)
    average, square = np.zeros(dimension), np.zeros(dimension)
    history = []
    best_params = None
    best_score = -np.inf
    started = perf_counter()
    for t in range(1, count + 1):
        chosen = rng.choice(len(train.seeds), size=batch, replace=False)
        fragment = subset(train, chosen)
        result = policy.lifetime(fragment, policy.Gate("learned", actor, noisy=True), (101, 137)[t % 2],
                                 toss=rng.random((len(chosen), train.ticks)))
        visible = result["active"]
        x = result["traces"][visible].astype(np.float64)
        action = result["intentions"][visible]
        p = result["probabilities"][visible].astype(np.float64)
        future = (result["rewards"] - result["paid"])[:, ::-1].cumsum(1)[:, ::-1][visible]
        regret = future - x @ critic
        # The critic and policy only receive actual payoffs of actually reached foods.
        # No unchosen outcome or family is evaluated for the learner.
        entropy = -p * (1 - p) * np.log(np.maximum(p, 1e-7) / np.maximum(1 - p, 1e-7))
        gradient = (x.T @ ((action - p) * regret + .02 * entropy)) / len(p)
        average = .90 * average + .10 * gradient
        square = .99 * square + .01 * gradient**2
        step = .034 / (1 + t / max(count, 1))
        actor += step * (average / (1 - .90**t)) / (np.sqrt(square / (1 - .99**t)) + .0001)
        critic_gram = .93 * critic_gram + x.T @ x / len(x)
        critic_signal = .93 * critic_signal + x.T @ future / len(x)
        critic = np.linalg.solve(critic_gram + .015 * np.eye(dimension), critic_signal)
        if t == 1 or t % 12 == 0 or t == count:
            scores = [policy.lifetime(validate, policy.Gate("learned", actor), sample) for sample in (101, 137)]
            score = float(np.mean([v["net"].mean() for v in scores]))
            probes = float(np.mean([v["probe"].sum(1).mean() for v in scores]))
            if score > best_score:
                best_params, best_score = actor.copy(), score
            history.append({"step": t, "dev_net": score, "dev_probes": probes, "mean_advantage": float(np.mean(regret))})
            print(f"seed {build_seed}, step {t:3d} dev net {score:+.3f} probes {probes:.2f} best {best_score:+.3f}; "
                  f"{perf_counter()-started:.1f}s", flush=True)
    return {"selected_params": best_params.tolist(), "development_best": best_score, "history": history}


def matched_schedule(panel, counts, salt):
    result = np.zeros_like(panel.arena, dtype=bool)
    for i, seed in enumerate(panel.seeds):
        rng = np.random.default_rng(np.random.SeedSequence((seed, salt, 0xF859C)))
        for where in range(2):
            subset_ = np.flatnonzero((panel.arena[i] == where) & (np.arange(panel.ticks) < panel.horizon[i]))
            result[i, rng.choice(subset_, counts[i, where], replace=False)] = True
    return result


def assessment(result, journey):
    n = len(journey.seeds)
    correct, grazed = first_blind(result)
    return {"net": result["net"], "net_per_ticket": result["net"] / journey.horizon,
            "raw_harvest": result["rewards"].sum(1), "paid": result["paid"].sum(1),
            "first_correct": correct.astype(float), "first_harvest_coverage": grazed.astype(float),
            "probes": result["probe"].sum(1), "contacts": result["landed"].sum(1) / journey.horizon,
            "endgame_probe": result["probe"][np.arange(n), journey.horizon - 1].astype(float)}


def average_world(rows):
    return {k: np.mean([x[k] for x in rows], axis=0) for k in rows[0]}


def train(args):
    train_worlds = range(args.start, args.start + args.train_worlds)
    dev_worlds = range(args.start + 20_000, args.start + 20_000 + args.dev_worlds)
    data = world.build(train_worlds)
    validate = world.build(dev_worlds)
    frozen = train_group(data, validate, args.optimizer_seed, args.steps, args.batch)
    folder = Path(__file__).resolve().parent
    frozen.update(optimizer_seed=args.optimizer_seed, training_worlds=list(train_worlds), development_worlds=list(dev_worlds),
                  method="Score-function gradient of actual return to go; no oracle, phase label or family provided to gate",
                  sources={name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in (
                      "active_symmetry_world.py", "active_symmetry_agent.py", "active_symmetry_train.py", "symmetry_spike.py", "symmetry_world.py")})
    args.artifact.write_text(json.dumps(frozen, separators=(",", ":")) + "\n", encoding="utf-8")


def compare(args):
    rules = [json.loads(p.read_text(encoding="utf-8")) for p in args.checkpoints]
    seeds = tuple(range(args.start, args.start + args.worlds))
    for r in rules:
        if set(seeds) & (set(r["training_worlds"]) | set(r["development_worlds"])):
            raise ValueError("A held world intersects the gate's prior data")
    panel = world.build(seeds)
    results = {}
    matched = []
    n_checks = 3
    for i, settings in enumerate(rules):
        for founder in (173, 211):
            optimized = policy.Gate("learned", np.asarray(settings["selected_params"]))
            trained = policy.lifetime(panel, optimized, founder)
            results.setdefault("learned", []).append((i, founder, trained))
            hidden_confidence = policy.lifetime(panel, policy.Gate("learned_blind", optimized.params), founder)
            results.setdefault("learned_no_confidence", []).append((i, founder, hidden_confidence))
            for salt in range(n_checks):
                shuffled = matched_schedule(panel, trained["counts"], salt)
                baseline = policy.lifetime(panel, policy.Gate("matched", quantum=shuffled), founder)
                np.testing.assert_array_equal(trained["counts"], baseline["counts"])
                np.testing.assert_allclose(trained["paid"].sum(1), baseline["paid"].sum(1), atol=1e-12, rtol=0)
                matched.append(assessment(baseline, panel))
            print("learned", settings["optimizer_seed"], founder, float(trained["net"].mean()), flush=True)
    for label in ("none", "once", "twice", "heuristic", "public"):
        results[label] = []
        for founder in (173, 211):
            reference = policy.lifetime(panel, policy.Gate(label), founder)
            results[label].append((None, founder, reference))
        print(label, [float(a[2]["net"].mean()) for a in results[label]], flush=True)
    data = {key: average_world([assessment(track, panel) for _, _, track in rows]) for key, rows in results.items()}
    data["count_matched_random"] = average_world(matched)
    summary = {}
    target = data["learned"]
    for name, vals in data.items():
        summary[name] = {**{k: estimates(v) for k, v in vals.items()},
                         "paired_learned_minus": {k: estimates(target[k] - vals[k]) for k in vals},
                         "from_horizon": {int(h): {k: estimates(v[panel.horizon == h]) for k, v in vals.items()}
                                           for h in np.unique(panel.horizon)}}
    decisions = {}
    for label, configs in results.items():
        active_ = configs[0][2]["active"]
        all_probes = np.stack([track["probe"] for _, _, track in configs])
        q = np.stack([track["confidence"] for _, _, track in configs])
        credit = panel.horizon[:, None] - np.arange(panel.ticks)
        c = panel.costs[np.arange(len(seeds))[:, None], panel.arena]
        conditions = {"last_ticket": credit == 1, "at_least_6": credit >= 6,
                      "low_cost": c == min(world.COSTS), "high_cost": c == max(world.COSTS),
                      "arena_zero": panel.arena == 0, "arena_one": panel.arena == 1,
                      "confident_and_time": (q >= .60) & (credit >= 4), "uncertain_and_time": (q < .15) & (credit >= 4)}
        decisions[label] = {}
        for name, condition in conditions.items():
            sample = np.broadcast_to(condition, all_probes.shape) & active_[None]
            decisions[label][name] = {"mean": float(all_probes[sample].mean()) if sample.any() else None,
                                      "total_opportunities": int(sample.sum())}
    # Same internal gain values, only the currently offered fees/credits are changed.
    grid = []
    from .active_symmetry_agent import features
    for r in rules:
        params = np.asarray(r["selected_params"])
        for credits in (1, 2, 4, 8, 14):
            for fee in world.COSTS:
                for gain in (0., .45, 1., 1.8):
                    view = world.Offer(np.array([0]), np.array([[fee, .36]]), np.array([credits]), np.array([0]))
                    f = features(view, np.array([[gain, .5]]))
                    pro = 1 / (1 + np.exp(-(f @ params)[0]))
                    grid.append({"seed": r["optimizer_seed"], "remaining": credits, "cost": fee, "gain_abs": gain,
                                 "probe_probability": float(pro)})
    out = {"held_worlds": seeds, "checkpoints": tuple(map(str, args.checkpoints)),
           "optimizer_seeds": [r["optimizer_seed"] for r in rules], "held_births": [173, 211],
           "worlds_per_horizon": {int(h): int(np.count_nonzero(panel.horizon == h)) for h in np.unique(panel.horizon)},
           "results": summary, "actual_decisions": decisions, "counterfactual_public_offers": grid,
           "learned_per_world": [{"optimizer_index": i, "birth": founder, **row_output(track)}
                                  for i, founder, track in results["learned"]],
           "public_ideal_expected_net": estimates([policy.ideal(*map(float, panel.costs[i]), max(world.HORIZONS))[2][int(h)]
                                                   for i, h in enumerate(panel.horizon)])}
    args.artifact.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="mode", required=True)
    a = command.add_parser("train")
    a.add_argument("artifact", type=Path)
    a.add_argument("--start", type=int, required=True)
    a.add_argument("--optimizer-seed", type=int, required=True)
    a.add_argument("--steps", type=int, default=144)
    a.add_argument("--batch", type=int, default=96)
    a.add_argument("--train-worlds", type=int, default=768)
    a.add_argument("--dev-worlds", type=int, default=144)
    b = command.add_parser("confirm")
    b.add_argument("artifact", type=Path)
    b.add_argument("--start", type=int, required=True)
    b.add_argument("--worlds", type=int, default=384)
    b.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
