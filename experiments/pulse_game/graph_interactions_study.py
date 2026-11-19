"""Sealed paired evaluation of nonlinear chemical preference learning through physical ports."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from . import graph_interactions as model, mixture_world as food, world
from .unified_live_graph import Graph, Recipe
from .unified_live_graph_study import Body, audit_edits, digest, read, write


OUT = Path(__file__).resolve().parents[2] / "docs/evidence/open-graph-learning-2026-11-19/interactions"
DEV_BIRTHS = tuple(range(864120, 864124))
DEV_WORLDS = tuple(range(864150, 864158))
TEST_BIRTHS = tuple(range(864200, 864208))
TEST_WORLDS = tuple(range(864250, 864266))
TRIALS = 960
TAIL = 128
MODES = ("stable", "reverse")
SOURCES = (
    "graph_interactions.py", "graph_interactions_study.py", "unified_live_graph.py",
    "unified_live_graph_study.py", "mixture_world.py", "world.py",
)
SETTINGS = (
    (Recipe(.18, .31, .998), 1.),
    (Recipe(.32, .28, .999), 1.),
    (Recipe(.32, .28, .999), 1.75),
    (Recipe(.48, .33, .9995), 1.75),
)
SCALES = (0., .45, .85)
CONTROLS = ("random", "linear", "polynomial", "oracle")
PRIMARY = (("square", "matched_tanh"), ("square", "square_sham"), ("tanh", "tanh_sham"))


def candidates():
    return [dict(recipe=asdict(recipe), weight_gain=gain, even=scale)
            for recipe, gain in SETTINGS for scale in SCALES]


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def register(out):
    out.mkdir(parents=True, exist_ok=True)
    write(out / "registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hashes(),
        world_roots=dict(development=DEV_WORLDS, confirm=TEST_WORLDS),
        birth_roots=dict(development=DEV_BIRTHS, confirm=TEST_BIRTHS),
        trials=TRIALS, tail=TAIL, conditions=MODES, candidates=candidates(),
        world="Four independent continuous Uniform(-1,1) self-symmetric odor coordinates for each of two novel foods. Their pairwise-products across the {0,1}/{2,3} split, multiplied by fresh per-world unit-norm private coefficients, determine logistic taste probabilities with gain 10. Changing either coordinate block's signs reverses the logit. Each single feature is therefore marginally uninformative for expected taste. Sign inversion is hidden, occurs near the midpoint, affects only taste, and leaves the same observations, geometry and exploratory drive in paired regimes.",
        architecture="Original tanh nodes and a no-added-parameter even residual share all 16 nodes, one directed graph and the same raw observations, initial support, rates, biases, individual weight updates and birth/retirement schedules. Even arm only replaces the activation at 8 odor and 2 interior nodes with (tanh(x) + signed_scale*tanh(x)^2)/(1+abs(scale)), alternating square sign. Motor, contact, taste and actual body stay identical. Base circuit update and uncapped strong support edits run after signed physical receipts. No explicit pair products reach either graph.",
        selection="Rank candidates by average stable and reversed silent tail actual-contact best-food accuracy, separately for original tanh and square nodes. First in registered order breaks ties. Matched tanh at the square winner's learning rate, exploration, decay and initial-weight gain isolates the selected activation change. Sham for each selected graph keeps its own registered configuration, replacing real contact taste by private independent fair signs. All other bodily interactions and scoring unchanged.",
        controls="Random emitted left/right pulses from one private fair choice per request. Linear and polynomial ridge learn from actual contacted chemistry and actual delivered taste on their own matched bodies. Polynomial uses all six cross-products with no knowledge of private nonzero coefficients or switch. Oracle only for scoring visibility uses hidden expected best side plus the same body. Every probe meal is newly sampled; feedback/exploratory drive at the graph and data learners turn off for the last 128 requests.",
        endpoints="Primary across six contrasts (each condition times square-matched tanh, square-square sham, tanh-tanh sham): silent tail best-side contact accuracy with timeout incorrect, simultaneous 95% percentile intervals from 20,000 crossed paired birth/world draws. Diagnostic accuracy conditional on contact, realized/expected utility, graph edits, post-switch/late windows, and visibility controls.",
        bootstrap_seed=304099224,
    ), exclusive=True)


def require(out):
    registration = read(out / "registered.json")
    if source_hashes() != registration["source_sha256"]:
        raise RuntimeError("Source changed after registering")
    return registration


def run(births, roots, mode, arm, setup=None, *, trials=TRIALS, tail=TAIL):
    birth_seeds = np.repeat(births, len(roots))
    world_seeds = np.tile(roots, len(births))
    menu, changes, noise, sham = model.make_menu(world_seeds, mode, trials)
    graph = arm not in CONTROLS
    body = Body(menu, sham if arm.endswith("_sham") else None, tail)
    n = len(birth_seeds)
    if graph:
        recipe = Recipe(**setup["recipe"])
        if setup["even"] == 0:
            agent = Graph(birth_seeds, recipe)
        else:
            agent = model.SquareGraph(birth_seeds, recipe, scale=setup["even"])
        agent.weight *= setup["weight_gain"]
        agent.initial_weight = agent.weight.copy()
        edits = np.full((n, trials, 2), -1, dtype=np.int16)
        writes = np.zeros((n, trials, 3), dtype=np.int16)
    elif arm in ("linear", "polynomial"):
        agent = model.PublicRidge(n, quadratic=arm == "polynomial")
    probes = np.arange(trials) >= trials - tail
    sensor = body.observe()
    all_best = menu.probability.argmax(axis=-1)
    for trial in range(trials):
        private = noise[:, trial]
        exploring = not probes[trial]
        if graph:
            agent.start(recipe.exploration * private if exploring else np.zeros(n, dtype=np.float32))
        else:
            if arm == "oracle":
                right = all_best[:, trial] == 1
            elif arm == "random":
                right = private > 0
            else:
                right = agent.choose(sensor, private, explore=exploring)
            pulse = np.column_stack((~right, right))
        for _ in range(world.TICKS):
            if graph:
                pulse = agent.advance(sensor)
            elif arm in ("linear", "polynomial"):
                agent.learn(sensor)
            sensor = body.step(pulse)
        if graph:
            writes[:, trial, 0], writes[:, trial, 1], edits[:, trial], writes[:, trial, 2] = agent.update(trial)
    result = body.result()
    contacted = result.contact_side
    hit = contacted >= 0
    shown = result.observed_reward.sum(axis=-1, dtype=np.int8)
    expected = np.take_along_axis(menu.probability, contacted.clip(min=0)[..., None], axis=-1)[..., 0] * hit
    if shown[:, -tail:].any():
        raise RuntimeError("Silent probes delivered a receipt")
    payload = dict(
        births=birth_seeds, worlds=world_seeds, change=changes,
        correct=(contacted == all_best) & hit, taste=result.reward, expected=expected, contact=contacted,
        contact_tick=result.contact_tick, shown=shown, private_best=all_best,
        private_weights=menu.weights[:len(roots)], raw_odors=menu.chemicals[:len(roots)],
        probability=menu.probability[:len(roots)], realized=menu.realized[:len(roots)],
    )
    if graph:
        audit_edits(agent.initial_mask, agent.initial_weight, edits, agent.mask, agent.weight)
        if np.any(writes[shown == 0]) or np.any(edits[shown == 0] >= 0):
            raise RuntimeError("A model wrote without a signed physical receipt")
        payload.update(edits=edits, writes=writes, initial_mask=agent.initial_mask,
                       initial_weight=agent.initial_weight, final_mask=agent.mask, final_weight=agent.weight,
                       final_generation_writes=agent.generation_writes)
    return payload


def develop(out):
    require(out)
    development = out / "development.json"
    if development.exists():
        raise RuntimeError("Already developed")

    def evaluate(setup):
        stats = {}
        for mode in MODES:
            raw = run(DEV_BIRTHS, DEV_WORLDS, mode, "candidate", setup)
            stats[mode] = float(raw["correct"][:, -TAIL:].mean())
        result = dict(setup=setup, accuracy=stats, objective=float(np.mean(list(stats.values()))))
        print("develop", result, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(evaluate, candidates()))
    selected = dict(
        tanh=max((row for row in rows if row["setup"]["even"] == 0), key=lambda row: row["objective"])["setup"],
        square=max((row for row in rows if row["setup"]["even"] > 0), key=lambda row: row["objective"])["setup"],
    )
    selected["matched_tanh"] = selected["square"] | {"even": 0.}
    write(development, dict(selection=selected, candidates=rows, registered_sha256=digest(out / "registered.json")), exclusive=True)


def measure(payload):
    seen = payload["contact"] >= 0
    correct = payload["correct"]
    offsets = np.arange(64)[None, :]
    at = np.arange(len(correct))[:, None]
    shift = payload["change"][:, None]
    stats = dict(
        first64=correct[:, :64].mean(axis=-1),
        before_switch64=correct[at, shift - 64 + offsets].mean(axis=-1),
        after_switch64=correct[at, shift + offsets].mean(axis=-1),
        silent_tail=correct[:, -TAIL:].mean(axis=-1), whole=correct.mean(axis=-1),
        silent_tail_contact=seen[:, -TAIL:].mean(axis=-1),
        silent_tail_given_contact=correct[:, -TAIL:].sum(axis=-1) / np.maximum(1, seen[:, -TAIL:].sum(axis=-1)),
        silent_tail_expected=payload["expected"][:, -TAIL:].mean(axis=-1),
        silent_tail_positive_taste=(payload["taste"][:, -TAIL:] > 0).mean(axis=-1),
    )
    if "edits" in payload:
        stats |= dict(
            births=(payload["edits"][..., 0] >= 0).sum(axis=-1),
            exits=(payload["edits"][..., 1] >= 0).sum(axis=-1),
            actual_weight_updates=payload["writes"][..., 0].sum(axis=-1),
            newborn_weight_updates=payload["writes"][..., 1].sum(axis=-1),
            newborn_generations_trained=(payload["final_generation_writes"] > 0).sum(axis=(1, 2))
                                        + (payload["writes"][..., 2] > 0).sum(axis=-1),
        )
    return stats


def contrast(delta, rng):
    groups, roots = len(TEST_BIRTHS), len(TEST_WORLDS)
    birth = rng.integers(groups, size=(20_000, groups, 1))
    world = rng.integers(roots, size=(20_000, 1, roots))
    sampled = delta.reshape(groups, roots)[birth, world].mean(axis=(1, 2))
    return [float(delta.mean()), *np.quantile(sampled, [.025 / 6, 1. - .025 / 6]).tolist()]


def confirm(out):
    registration = require(out)
    development = read(out / "development.json")
    if development["registered_sha256"] != digest(out / "registered.json") or (out / "summary.json").exists():
        raise RuntimeError("Already confirmed or registration changed")
    setup = development["selection"]
    arms = (*setup, "tanh_sham", "square_sham", *CONTROLS)
    world_keys = ("births", "worlds", "change", "private_best", "private_weights", "raw_odors", "probability", "realized")

    def condition(mode):
        archive = {}
        stats = {}
        for arm in arms:
            start = perf_counter()
            raw = run(TEST_BIRTHS, TEST_WORLDS, mode, arm, setup.get(arm.removesuffix("_sham")))
            for key in world_keys:
                values = raw.pop(key)
                if key in archive:
                    np.testing.assert_array_equal(archive[key], values)
                else:
                    archive[key] = values
            stats[arm] = measure(raw | {"change": archive["change"]})
            archive.update({f"{arm}/{key}": value for key, value in raw.items()})
            print("confirm", mode, arm, stats[arm]["silent_tail"].mean(), round(perf_counter() - start, 2), flush=True)
        path = out / f"{mode}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **archive)
        zipped = out / f"{mode}-raw.zip"
        with ZipFile(zipped, "x", compression=ZIP_STORED) as handle:
            handle.write(path, path.name)
        with ZipFile(zipped) as handle:
            assert handle.testzip() is None
        return mode, stats, digest(path), digest(zipped)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(condition, MODES))
    rng = np.random.default_rng(registration["bootstrap_seed"])
    outcomes = {}
    for mode, stats, sha, zip_sha in results:
        outcomes[mode] = dict(
            means={arm: {key: float(values.mean()) for key, values in measures.items()} for arm, measures in stats.items()},
            paired={f"{a}_minus_{b}": contrast(stats[a]["silent_tail"] - stats[b]["silent_tail"], rng) for a, b in PRIMARY},
            npz_sha256=sha, archive_sha256=zip_sha,
        )
    write(out / "summary.json", dict(outcomes=outcomes, configurations=setup,
                                    registered_sha256=digest(out / "registered.json"),
                                    development_sha256=digest(out / "development.json")), exclusive=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
