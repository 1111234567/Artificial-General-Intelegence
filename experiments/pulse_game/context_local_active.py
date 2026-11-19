"""A small paid-probe decision head on one continuously running frozen brain."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import active_symmetry_train, active_symmetry_world as game, adaptive_local, independent_births
from . import habitat_world, mixture_flash, mixture_world as food, unified_audit, unified_physical, unified_world, world
from .attractor_search import OUT
from .context_local_summary import difference


BIRTH = 374853277
WIDTH = 10
SEARCH_WIDTH = 14
SPLIT = {"train": 3830000, "development": 3850000, "confirmation": 3870000}


def frozen_brain():
    base = OUT / "independent-lifelong-births-v1"
    registration = json.loads((base / "registered.json").read_text())
    folder = base / f"birth-{BIRTH}"
    front, route, motor = independent_births.resources(folder)
    frozen, = folder.glob("*--full-*/frozen.json")
    checkpoint = json.loads(frozen.read_text())
    front, route, motor = unified_physical.materialize(front, route, motor, np.asarray(checkpoint["best"]["genome"]), "full")
    route = {**route, "release": registration["reopening_rule"]}
    used = [base / "registered.json", frozen, folder / "receptor.pt", *folder.glob("rls-learned-s*.pt")]
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in used}
    return (front, route, motor), inputs


class SharedMotor(adaptive_local.LocalAdaptive):
    """Add the observed plate-side drive to the already-running actuator."""

    gate_drive = None

    def _noise(self):
        noise = super()._noise()
        if self.gate_drive is None:
            return noise
        return noise + self.gate_drive / self.motion[3]


def appearances(seeds):
    cues = []
    for s in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([s, 954371]))
        a = rng.uniform(0, .43)
        theta = np.asarray([a, a + rng.uniform(.67, .98)])[rng.permutation(2)]
        cues.append(np.column_stack((np.cos(theta), np.sin(theta))))
    return np.asarray(cues, dtype=np.float32)


def earlier_experience(brain, seeds):
    """Run the unchanged long conflict protocol before the paid scene."""
    episode = unified_world.make(seeds, same_prior=.5, nonce=84571321, **unified_audit.SETTINGS["long_competing_run"])
    sim = habitat_world.Body(episode.arena, silent_probes=True)
    obs = sim.observe()
    while obs is not None:
        obs = sim.step(brain.act(obs))
    result = unified_world.measures(episode, sim.result())
    # New territories carry real distinct observable vectors. Nothing about
    # their unknown taste rules is picked or provided by this codebook.
    previous = episode.arena.tags
    theta = np.linspace(0, np.pi / 2, 101)
    vectors = np.column_stack((np.cos(theta), np.sin(theta)))
    current = []
    for _ in range(2):
        occupied = (previous @ vectors.T).max(1)
        selected = vectors[occupied.argmin(1)].astype(np.float32)
        current.append(selected)
        previous = np.concatenate((previous, selected[:, None]), axis=1)
    measured = {key: result[key] for key in ("return_A", "B_after_48", "both")}
    measured["conflict"] = episode.arena.relation[:, 0] != episode.arena.relation[:, 1]
    return np.stack(current, axis=1), measured


def observable_head(brain, price, remaining):
    row = np.arange(brain.n)
    selected = np.maximum(brain.places.current, 0)
    confidence = np.abs(np.tanh(brain.route["amp"] * brain.g[row, selected]))
    covariance = brain.cov[row, selected, selected]
    uncertainty = np.log1p(np.maximum(covariance, 0)) / 5
    time = remaining / max(game.HORIZONS)
    x = np.column_stack((np.ones(len(row)), confidence, uncertainty, time, price,
                         confidence * time, uncertainty * time, price * time,
                         price * uncertainty, remaining <= 1))
    if x.shape[1] != WIDTH:
        raise ValueError("head input width changed")
    return x, confidence


def gate_frame(brain, journey, command, active, tick):
    n = len(active)
    tape = game.body_region(journey.markers, np.arange(n), tick, 1, 9247 * (tick + 1))
    sim = world.PulseWorld(tape, feedback="none")
    obs = sim.observe()
    target_side = np.where(command, journey.probe_side[:, tick], journey.probe_side[:, tick] ^ 1)
    # The physical marker-to-side pathway is wired. The same membrane and
    # spikes integrate this strong command and both forms of subsequent motion.
    plate_current = np.where(np.arange(2)[None, :] == target_side[:, None], 16., -16.).astype(np.float32)
    while obs is not None:
        obs[:, food.CTX] = 0
        obs[:, food.LEFT.start:food.RIGHT.stop] = .5
        obs[~active] = 0
        obs[~active, food.LEFT.start:food.RIGHT.stop] = .5
        brain.gate_drive = plate_current * (obs[:, world.ACTIVE, None] > 0)
        movement = brain.act(obs).copy()
        movement[~active] = 0
        obs = sim.step(movement)
    brain.gate_drive = None
    side = sim.result().contact_side[:, 0]
    arrived = active & (side >= 0)
    buy = arrived & (side == journey.probe_side[:, tick])
    return arrived, buy


def taste_pack(journey, where, tick, bought, arrived):
    n = len(where)
    base = game.assembly(journey, 0, tick, np.arange(n), True)
    dims = (n, game.BLOCK)
    fields = ("ctx", "cue", "left_id", "good_id", "distances", "wind")
    values = {name: np.zeros_like(getattr(base.body, name)) for name in fields}
    values["distances"].fill(1)
    scents, payoff = np.full_like(base.chemicals, .5), np.zeros_like(base.probability)
    signs = np.zeros_like(base.realized)
    seeds = list(base.body.seeds)
    available = np.zeros(dims, dtype=bool)
    for location in (0, 1):
        for explore in (False, True):
            part = np.flatnonzero(arrived & (where == location) & (bought == explore))
            if not part.size:
                continue
            source = game.assembly(journey, location, tick, part, explore)
            duration = source.body.trials
            for name in fields:
                values[name][part, :duration] = getattr(source.body, name)
            scents[part, :duration], payoff[part, :duration], signs[part, :duration] = source.chemicals, source.probability, source.realized
            available[part, :duration] = True
            for i, value in zip(part, source.body.seeds, strict=True):
                seeds[i] = value
    body = replace(base.body, **values, seeds=tuple(seeds))
    return replace(base, body=body, chemicals=scents, probability=payoff, realized=signs), available


def consume(brain, menu, available, cue):
    sim = mixture_flash.FlashBody(menu)
    obs = sim.observe()
    while obs is not None:
        present = available[:, sim.trial_index]
        flashed = obs[:, food.CTX].any(axis=1) & present
        obs[flashed, food.CTX] = cue[flashed]
        obs[~present] = 0
        obs[~present, food.LEFT.start:food.RIGHT.stop] = .5
        motion = brain.act(obs).copy()
        motion[~present] = 0
        obs = sim.step(motion)
    return sim.result()


def lifetime(config, journey, *, method, parameters=None, random=None, quantum=None, old_prefix=False, force_tick=None, force_probe=False):
    front, route, motor = config
    brain = SharedMotor(front, route, journey.seeds, motor=motor)
    n, tickets = len(journey.seeds), journey.ticks
    brain.reset(n)
    if old_prefix:
        tags, old = earlier_experience(brain, journey.seeds)
    else:
        tags, old = appearances(journey.seeds), None
    rows = np.arange(n)
    active = np.arange(tickets)[None, :] < journey.horizon[:, None]
    shape = (n, tickets)
    paid = np.zeros(shape)
    harvest = np.zeros(shape)
    purchases = np.zeros(shape, dtype=bool)
    goal = np.zeros_like(purchases)
    touches = np.zeros_like(purchases)
    exposures = np.zeros(shape, dtype=np.int8)
    width = len(parameters) if parameters is not None else WIDTH
    if width not in (WIDTH, SEARCH_WIDTH):
        raise ValueError("unexpected gate parameter shape")
    traces = np.zeros((*shape, width))
    probabilities = np.zeros(shape)
    certainty = np.zeros(shape)
    counts = np.zeros((n, 2), dtype=np.int16)
    chosen_slots = np.zeros((n, brain.slots), dtype=np.int16)
    for tick in range(tickets):
        loc = journey.arena[:, tick]
        available = active[:, tick]
        actual_scene = tags[rows, loc]
        sense = np.zeros((n, world.OBS_SIZE), np.float32)
        sense[:, food.LEFT.start:food.RIGHT.stop] = .5
        sense[available, food.CTX] = actual_scene[available]
        brain.act(sense)
        left = journey.horizon - tick
        fare = journey.costs[rows, loc]
        memory_slot = np.maximum(brain.places.current, 0)
        design, confidence = observable_head(brain, fare, np.maximum(left, 0))
        if width == SEARCH_WIDTH:
            already = chosen_slots[rows, memory_slot] > 0
            seen = brain.places.counts[rows, memory_slot].clip(1)
            design = np.column_stack((design, already, 1 / np.sqrt(seen), fare * already, design[:, 3] * already))
        if method in ("learned", "state_hidden", "gain_cov_hidden", "familiarity_hidden"):
            if method in ("state_hidden", "gain_cov_hidden"):
                design[:, 1] = design[:, 5] = 0
                design[:, 2] = np.log1p(1 / brain.route["ridge"]) / 5
                design[:, 6] = design[:, 2] * design[:, 3]
                design[:, 8] = design[:, 2] * design[:, 4]
            if width == SEARCH_WIDTH and method in ("state_hidden", "familiarity_hidden"):
                design[:, 11] = 1
            prop = 1 / (1 + np.exp(-np.clip(design @ parameters, -16, 16)))
            want = (prop >= .5) if random is None else (random[:, tick] < prop)
        elif method in ("once", "twice"):
            want = (chosen_slots[rows, memory_slot] < (1 if method == "once" else 2)) & (left > 1)
            prop = want.astype(float)
        elif method == "matched":
            want = quantum[:, tick]
            prop = want.astype(float)
        else:
            want = np.zeros(n, bool)
            prop = want.astype(float)
        if force_tick is not None:
            want = np.where(force_tick == tick, force_probe, want)
        want = want & available
        traces[:, tick] = design
        probabilities[:, tick] = prop
        certainty[:, tick] = confidence
        goal[:, tick] = want
        arrived, bought = gate_frame(brain, journey, want, available, tick)
        touches[:, tick] = arrived
        purchases[:, tick] = bought
        counts[rows[bought], loc[bought]] += 1
        chosen_slots[rows[bought], memory_slot[bought]] += 1
        paid[:, tick] = bought * fare
        menu, valid = taste_pack(journey, loc, tick, bought, arrived)
        result = consume(brain, menu, valid, actual_scene)
        harvest[:, tick] = result.reward[:, 1] * (arrived & ~bought)
        exposures[:, tick] = (result.observed_reward != 0).sum(axis=(1, 2))
    return {"net": (harvest - paid).sum(1), "rewards": harvest, "paid": paid, "probe": purchases, "counts": counts,
            "active": active, "traces": traces, "probabilities": probabilities, "intentions": goal,
            "landed": touches, "confidence": certainty, "actual_receipts": exposures,
            "addresses": (brain.places.counts > 0).sum(axis=1), "continuous_neural_ticks": brain.it, "old": old}


def fit(config, train, development, *, iterations, batch):
    generator = np.random.default_rng(265324)
    theta = 0.02 * generator.normal(size=WIDTH)
    average, square = np.zeros(WIDTH), np.zeros(WIDTH)
    gram, signal = np.eye(WIDTH), np.zeros(WIDTH)
    critic = np.zeros(WIDTH)
    best, best_value = None, -np.inf
    traces = []
    for step in range(1, iterations + 1):
        rows = generator.choice(len(train.seeds), batch, replace=False)
        view = active_symmetry_train.subset(train, rows)
        out = lifetime(config, view, method="learned", parameters=theta, random=generator.random((batch, train.ticks)))
        active = out["active"]
        design = out["traces"][active]
        p, y = out["probabilities"][active], out["intentions"][active]
        returns = (out["rewards"] - out["paid"])[:, ::-1].cumsum(axis=1)[:, ::-1][active]
        residual = returns - design @ critic
        delta = design.T @ ((y - p) * residual + .02 * (-p * (1-p) * np.log(p/(1-p)))) / len(p)
        average = .90 * average + .10 * delta
        square = .99 * square + .01 * delta**2
        theta += .045 * average / (1 - .9**step) / (np.sqrt(square / (1 - .99**step)) + .0001)
        gram = .93 * gram + design.T @ design / len(p)
        signal = .93 * signal + design.T @ returns / len(p)
        critic = np.linalg.solve(gram + .015 * np.eye(WIDTH), signal)
        if step == 1 or step % 12 == 0 or step == iterations:
            check = lifetime(config, development, method="learned", parameters=theta)
            net = float(check["net"].mean())
            if net > best_value:
                best, best_value = theta.copy(), net
            row = {"step": step, "development_net": net, "development_probes": float(check["probe"].sum(1).mean()), "sampled_net": float(out["net"].mean())}
            traces.append(row)
            print(json.dumps(row), flush=True)
    return best, traces


def search_head(config, train, development, *, generations, population, batch):
    """Select and perturb tiny threshold heads using only actual net payoff."""
    rng = np.random.default_rng(776008)
    once = np.zeros(SEARCH_WIDTH)
    once[[0, 9, 10]] = [1., -2., -2.]
    no = np.zeros(SEARCH_WIDTH)
    no[0] = -10.
    center = once.copy()
    width = np.full(SEARCH_WIDTH, 0.7)
    retained = once
    best_dev = -np.inf
    history = []
    for stage in range(generations):
        roots = rng.choice(len(train.seeds), batch, replace=False)
        panel = active_symmetry_train.subset(train, roots)
        candidates = np.vstack((once, no, retained, center, center + width * rng.normal(size=(population - 4, SEARCH_WIDTH))))
        observed = [lifetime(config, panel, method="learned", parameters=x)["net"].mean() for x in candidates]
        sorted_indices = np.argsort(-np.asarray(observed), kind="stable")
        elite = candidates[sorted_indices[:max(4, population // 4)]]
        center = 0.65 * center + 0.35 * elite.mean(axis=0)
        width = np.maximum(.12, 0.73 * width + 0.27 * elite.std(axis=0))
        check = []
        for which in sorted_indices[:3]:
            current = lifetime(config, development, method="learned", parameters=candidates[which])
            score = float(current["net"].mean())
            check.append([int(which), score, float(current["probe"].sum(axis=1).mean())])
            if score > best_dev:
                best_dev, retained = score, candidates[which].copy()
        row = {"generation": stage, "training_nets": np.asarray(observed).tolist(), "best_development_net": best_dev, "ranked_development": check}
        history.append(row)
        print(json.dumps({"generation": stage, "best_development_net": best_dev, "ranked_development": check}), flush=True)
    return retained, history


def fit_paid_forks(config, train, development, *, rounds):
    """Fit the differential net of two separately acted, physical continuations."""
    rng = np.random.default_rng(484789)
    once = np.zeros(SEARCH_WIDTH)
    once[[0, 9, 10]] = [1., -2., -2.]
    payload, returns = [], []
    n = len(train.seeds)
    row = np.arange(n)
    history = []
    for batch in range(rounds):
        point = rng.integers(train.horizon)
        buy, harvest = (lifetime(config, train, method="learned", parameters=once, force_tick=point, force_probe=choice)
                        for choice in (True, False))
        before = np.arange(train.ticks)[None, :] < point[:, None]
        for key in ("paid", "rewards", "probe"):
            np.testing.assert_array_equal(buy[key][before], harvest[key][before])
        np.testing.assert_array_equal(buy["traces"][row, point], harvest["traces"][row, point])
        observed = buy["net"] - harvest["net"]
        payload.append(buy["traces"][row, point])
        returns.append(observed)
        history.append({"physical_pair_batch": batch, "worlds": n, "actually_bought_rate": float(buy["probe"][row,point].mean()),
                        "actual_bought_minus_harvest_net": float(observed.mean())})
        print(json.dumps(history[-1]), flush=True)
    observations, outcomes = np.concatenate(payload), np.concatenate(returns)
    normal = observations.T @ observations / len(outcomes)
    fit = np.linalg.solve(normal + .006 * np.eye(SEARCH_WIDTH), observations.T @ outcomes / len(outcomes))
    best, value = None, -np.inf
    check = []
    candidates = [once, *(fit + np.eye(1, SEARCH_WIDTH)[0] * shift for shift in (-.55, -.36, -.22, -.1, 0., .1, .25))]
    for ix, proposal in enumerate(candidates):
        trial = lifetime(config, development, method="learned", parameters=proposal)
        current = float(trial["net"].mean())
        check.append({"candidate": ix, "development_net": current, "development_buys": float(trial["probe"].sum(1).mean())})
        if current > value:
            best, value = proposal.copy(), current
    history.append({"selection_by_actual_development_net": check, "linear_differential_fit": fit.tolist()})
    print(json.dumps(history[-1]), flush=True)
    return best, history


def summarize(record):
    totals = {"net": record["net"], "harvest": record["rewards"].sum(1), "paid": record["paid"].sum(1),
              "bought": record["probe"].sum(1), "actual_taste_receipts": record["actual_receipts"].sum(1)}
    report = {
        **{key: float(values.mean()) for key, values in totals.items()},
        "arrived_at_intended_plates": int(((record["probe"] == record["intentions"]) & record["landed"] & record["active"]).sum()),
        "plate_opportunities": int(record["active"].sum()),
        "continuous_neural_ticks_per_batched_rollout": record["continuous_neural_ticks"],
        "mean_addresses": float(record["addresses"].mean()),
        "rows": {key: values.tolist() for key, values in totals.items()}, "per_arena_paid_counts": record["counts"].tolist(),
    }
    if record["old"] is not None:
        old = record["old"]
        report["earlier_same_life_conflict"] = {k: float(old[k][old["conflict"]].mean()) if old["conflict"].any() else None
                                                for k in ("return_A", "B_after_48", "both")}
        report["earlier_conflict_opportunities"] = int(old["conflict"].sum())
    return report


def randomized_nonfinal(journey, counts, salt):
    decisions = np.zeros_like(journey.arena, dtype=bool)
    time = np.arange(journey.ticks)
    for row, seed in enumerate(journey.seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 378557, salt]))
        for place in (0, 1):
            present = journey.arena[row] == place
            allowed = np.flatnonzero(present & (time < journey.horizon[row] - 1))
            requested = counts[row, place]
            if len(allowed) < requested:
                allowed = np.flatnonzero(present & (time < journey.horizon[row]))
            decisions[row, rng.choice(allowed, requested, replace=False)] = True
    return decisions


def compare_random(chosen, config, journey, *, old_prefix, nonfinal):
    randomized = []
    for seed in range(3):
        schedule = (randomized_nonfinal if nonfinal else active_symmetry_train.matched_schedule)(journey, chosen["counts"], 779 + seed)
        record = lifetime(config, journey, method="matched", quantum=schedule, old_prefix=old_prefix)
        np.testing.assert_array_equal(record["counts"], chosen["counts"])
        np.testing.assert_allclose(record["paid"].sum(1), chosen["paid"].sum(1), atol=1e-14, rtol=0)
        randomized.append(record)
    mean_random = np.stack([r["net"] for r in randomized]).mean(0)
    return {"profiles": [summarize(r) for r in randomized], "mean_net": float(mean_random.mean()),
            "learned_minus_this_net": difference(chosen["net"], mean_random), "same_actual_per_life_per_arena_counts_and_fees": True}


def confirm(config, journey, theta, *, old_prefix=False, extra_audit=False):
    chosen = lifetime(config, journey, method="learned", parameters=theta, old_prefix=old_prefix)
    comparison = {"learned": summarize(chosen)}
    for rule in ("none", "once", "twice"):
        record = lifetime(config, journey, method=rule, old_prefix=old_prefix)
        comparison[rule] = {**summarize(record), "learned_minus_this_net": difference(chosen["net"], record["net"])}
    comparison["matched_random_timing"] = compare_random(chosen, config, journey, old_prefix=old_prefix, nonfinal=False)
    if extra_audit:
        comparison["matched_random_nonfinal_where_feasible"] = compare_random(chosen, config, journey, old_prefix=old_prefix, nonfinal=True)
        for feature in ("state_hidden", "gain_cov_hidden", "familiarity_hidden"):
            blinded = lifetime(config, journey, method=feature, parameters=theta, old_prefix=old_prefix)
            comparison[f"head_{feature}"] = {**summarize(blinded), "learned_minus_this_net": difference(chosen["net"], blinded["net"])}
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=108)
    parser.add_argument("--train-worlds", type=int, default=192)
    parser.add_argument("--development-worlds", type=int, default=80)
    parser.add_argument("--confirmation-worlds", type=int, default=256)
    parser.add_argument("--confirmation-start", type=int)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--paired-life", action="store_true", help="run the old conflict game then paid choices without resetting the brain")
    optimizer = parser.add_mutually_exclusive_group()
    optimizer.add_argument("--search", action="store_true")
    optimizer.add_argument("--fork-fit", action="store_true")
    parser.add_argument("--fork-rounds", type=int, default=12)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--extra-audit", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.paired_life and args.checkpoint is None:
        parser.error("the paired-life transfer uses a previously frozen head")
    if args.search and (args.population < 8 or args.generations < 1):
        parser.error("use at least eight search candidates and a positive generation count")
    torch.set_num_threads(2)
    config, inputs = frozen_brain()
    started = perf_counter()
    if args.pilot:
        pan = game.build(range(3810000, 3810024))
        output = {mode: summarize(lifetime(config, pan, method=mode)) for mode in ("none", "once", "twice")}
    else:
        count = {"train": args.train_worlds, "development": args.development_worlds, "confirmation": args.confirmation_worlds}
        confirmation = args.confirmation_start or (3890000 if args.paired_life else SPLIT["confirmation"])
        starts = {**SPLIT, "confirmation": confirmation}
        settings = {key: [s, s + count[key] - 1] for key, s in starts.items()}
        cards = lambda key: game.build(range(starts[key], starts[key] + count[key]))
        if args.checkpoint:
            previous = json.loads(args.checkpoint.read_text(encoding="utf-8"))
            if previous["frozen_inputs"] != inputs:
                raise ValueError("frozen agent changed")
            theta = np.asarray(previous["parameters"])
            traces = previous["training"]
            settings.update({key: previous["seeds"][key] for key in ("train", "development")})
            training = previous["configuration"]
            known = set()
            for key in ("train", "development"):
                known.update(range(settings[key][0], settings[key][1] + 1))
            if known.intersection(range(settings["confirmation"][0], settings["confirmation"][1] + 1)):
                parser.error("confirmation seeds overlap head training or development")
        else:
            if args.fork_fit:
                if args.fork_rounds < 1:
                    parser.error("use at least one physical paired continuation per life")
                theta, traces = fit_paid_forks(config, cards("train"), cards("development"), rounds=args.fork_rounds)
                training = {"optimizer": "actual_paired_continuations", "rounds": args.fork_rounds, "body_birth": BIRTH}
            elif args.search:
                theta, traces = search_head(config, cards("train"), cards("development"), generations=args.generations, population=args.population, batch=args.batch)
                training = {"optimizer": "deterministic_return_search", "generations": args.generations, "population": args.population, "train_batch": args.batch, "body_birth": BIRTH}
            else:
                theta, traces = fit(config, cards("train"), cards("development"), iterations=args.iterations, batch=args.batch)
                training = {"optimizer": "actor_critic", "iterations": args.iterations, "train_batch": args.batch, "body_birth": BIRTH}
        output = {"parameters": theta.tolist(), "training": traces, "seeds": settings, "frozen_inputs": inputs,
                  "outputs": confirm(config, cards("confirmation"), theta, old_prefix=args.paired_life, extra_audit=args.extra_audit), "running_seconds": perf_counter() - started,
                  "configuration": training, "old_preexposure_without_reset": args.paired_life}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    view = output.get("outputs", output)
    print({k: v.get("net", v.get("mean_net")) for k, v in view.items()}, flush=True)


if __name__ == "__main__":
    main()
