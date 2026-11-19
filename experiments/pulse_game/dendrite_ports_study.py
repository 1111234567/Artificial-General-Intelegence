"""Compare multiplicative and equal-slot additive dendrites in one physical graph."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from . import graph_interactions as task, world
from .dendrite_ports_graph import DendriteGraph, FIRST, SECOND, widen_birth
from .unified_live_graph import Graph, Recipe
from .unified_live_graph_study import Body, audit_edits, digest, read, write


BIRTH_DEV = tuple(range(851440, 851443))
WORLD_DEV = tuple(range(851470, 851476))
BIRTH_TEST = tuple(range(851550, 851558))
WORLD_TEST = tuple(range(851580, 851592))
TRIALS, TAIL = 1280, 112
MODES = ("stable", "reverse")
ARMS = ("product", "additive", "tanh", "product_sham", "polynomial")
SETUPS = (
    dict(recipe=asdict(Recipe(.045, .23, .991)), pair_rate=.65, pair_decay=.993),
    dict(recipe=asdict(Recipe(.070, .21, .995)), pair_rate=1.2, pair_decay=.996),
    dict(recipe=asdict(Recipe(.11, .24, .995)), pair_rate=1.0, pair_decay=.997),
)
SOURCES = ("dendrite_ports_graph.py", "dendrite_ports_study.py", "graph_interactions.py",
           "unified_live_graph.py", "unified_live_graph_study.py", "mixture_world.py", "world.py")


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def register(out):
    out.mkdir(parents=True, exist_ok=True)
    write(out / "registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hashes(),
        birth_roots=dict(development=BIRTH_DEV, confirmation=BIRTH_TEST),
        world_roots=dict(development=WORLD_DEV, confirmation=WORLD_TEST),
        scenarios=MODES, arms=ARMS, trials=TRIALS, silent_tail=TAIL, setups=SETUPS,
        hypothesis="A graph with pairs of incoming activity taps per receiver can acquire a novel pure interaction rule from stochastic contact tastes. Compare an equal-slot additive graph, no pair parameter tanh graph, and independent fair-taste graph. Include public polynomial preference regression as a visibility calibration.",
        matching="Common densified random graph initialization, each originally absent nonself edge added independently with probability .52; all arms share W/rate/bias, all pair coefficients start at zero. Each pair arm allocates one coefficient per receiver and unordered pair of active incoming arcs. Product uses s_i*s_j, additive uses .36*(s_i+s_j). Same receipt-gated truncated motor objective for W and Q. Same birth/removal rules in both pair arms. All observed foods pass as raw concentration only.",
        selection="Maximize mean of product graph's stable and reversal silent accuracy on development worlds and births. First candidate breaks ties. One recipe shared by all confirmation graph arms; control polynomial is fixed.",
        endpoint="Physical first-contact expected best-food accuracy on fresh final 112 trials, without public taste or motor exploration. Crossed birth and world paired percentile intervals, 95% simultaneous family across stable/reverse product-minus-tanh, additive and fair-taste; add product-minus-.5 as a secondary point with separate interval. Report contact, additions, removals, pure actual pair-gradient writes, newborn pair writes, total pair count.",
    ), exclusive=True)


def check(out):
    registration = read(out / "registered.json")
    if source_hashes() != registration["source_sha256"]:
        raise RuntimeError("Registered source changed")
    return registration


def run(births, roots, mode, arm, setup, *, trials=TRIALS, tail=TAIL):
    birth_roots = np.repeat(births, len(roots))
    world_roots = np.tile(roots, len(births))
    menu, changes, motor_noise, random_tastes = task.make_menu(world_roots, mode, trials)
    body = Body(menu, random_tastes if arm == "product_sham" else None, tail)
    n = len(world_roots)
    recipe = Recipe(**setup["recipe"])
    pairs = arm in ("product", "additive", "product_sham")
    if pairs:
        graph = DendriteGraph(birth_roots, recipe, product=arm != "additive", pair_rate=setup["pair_rate"], pair_decay=setup["pair_decay"])
    elif arm == "tanh":
        graph = Graph(birth_roots, recipe)
        widen_birth(graph, birth_roots)
    else:
        graph = None
        learner = task.PublicRidge(n, quadratic=True)
    edits = np.full((n, trials, 2), -1, dtype=np.int16)
    writes = np.zeros((n, trials, 5), dtype=np.int16)
    observed = body.observe()
    for trial in range(trials):
        explore = trial < trials - tail
        if graph is not None:
            graph.start(recipe.exploration * motor_noise[:, trial] if explore else np.zeros(n, dtype=np.float32))
        else:
            right = learner.choose(observed, motor_noise[:, trial], explore=explore)
            pulses = np.column_stack((~right, right))
        for _ in range(world.TICKS):
            if graph is not None:
                pulses = graph.advance(observed)
            else:
                learner.learn(observed)
            observed = body.step(pulses)
        if graph is not None:
            writes[:, trial, 0], writes[:, trial, 1], edits[:, trial], writes[:, trial, 2] = graph.update(trial)
            if pairs:
                writes[:, trial, 3] = graph.pair_writes
                writes[:, trial, 4] = graph.newborn_pair_writes
    result = body.result()
    shown = result.observed_reward.sum(axis=-1, dtype=np.int8)
    contacted = result.contact_side >= 0
    correct = (result.contact_side == menu.probability.argmax(axis=-1)) & contacted
    assert not shown[:, -tail:].any()
    assert not writes[shown == 0].any() and not (edits[shown == 0] >= 0).any()
    payload = dict(birth=birth_roots, world=world_roots, change=changes, correct=correct, contact=result.contact_side,
                   contact_tick=result.contact_tick, reward=result.reward, shown=shown, good_side=menu.probability.argmax(axis=-1),
                   edits=edits, writes=writes)
    if graph is not None:
        audit_edits(graph.initial_mask, graph.initial_weight, edits, graph.mask, graph.weight)
        payload.update(initial_mask=graph.initial_mask, initial_weight=graph.initial_weight, final_mask=graph.mask, final_weight=graph.weight,
                       final_generation_writes=graph.generation_writes)
    if pairs:
        assert not graph.pair[~(graph.mask[:, :, FIRST] & graph.mask[:, :, SECOND])].any()
        payload["final_pair"] = graph.pair
    return payload


def develop(out):
    check(out)
    results = []
    for i, setup in enumerate(SETUPS):
        rates = {mode: float(run(BIRTH_DEV, WORLD_DEV, mode, "product", setup)["correct"][:, -TAIL:].mean()) for mode in MODES}
        row = dict(index=i, setup=setup, rates=rates, score=float(np.mean(list(rates.values()))))
        print(row, flush=True)
        results.append(row)
    write(out / "development.json", {"results": results, "winner": max(results, key=lambda item: item["score"]),
                                     "registered_sha256": digest(out / "registered.json")}, exclusive=True)


def measure(data):
    score = data["correct"]
    births, retires = (data["edits"] >= 0).sum(axis=1).T
    shift = data["change"][:, None]
    index = np.arange(len(score))[:, None]
    offsets = np.arange(64)[None, :]
    result = dict(
        first=score[:, :64].mean(axis=1), pre_change=score[index, shift - 64 + offsets].mean(axis=1),
        early_change=score[index, shift + offsets].mean(axis=1), tail=score[:, -TAIL:].mean(axis=1),
        tail_contact=(data["contact"][:, -TAIL:] >= 0).mean(axis=1), positive=(data["reward"][:, -TAIL:] > 0).mean(axis=1),
        growth=births, retirement=retires, pair_gradient_writes=data["writes"][:, :, 3].sum(axis=1),
        newborn_pair_gradient_writes=data["writes"][:, :, 4].sum(axis=1),
    )
    if "final_mask" in data:
        arcs = data["final_mask"]
        result["final_unary"] = arcs.sum(axis=(1, 2))
        if "final_pair" in data:
            result["birth_pairs"] = (data["initial_mask"][:, :, FIRST] & data["initial_mask"][:, :, SECOND]).sum(axis=(1, 2))
            result["final_pairs"] = (arcs[:, :, FIRST] & arcs[:, :, SECOND]).sum(axis=(1, 2))
            result["final_nonzero_pairs"] = (data["final_pair"] != 0).sum(axis=(1, 2))
    return result


def interval(diff, tail, draws):
    sampled = diff.reshape(len(BIRTH_TEST), len(WORLD_TEST))[draws[0], draws[1]].mean(axis=(1, 2))
    return [float(diff.mean()), *np.quantile(sampled, [tail, 1 - tail]).tolist()]


def confirm(out):
    check(out)
    dev = read(out / "development.json")
    if dev["registered_sha256"] != digest(out / "registered.json"):
        raise RuntimeError("Registered manifest changed")
    setup = dev["winner"]["setup"]
    rng = np.random.default_rng(811414031)
    groups, worlds = len(BIRTH_TEST), len(WORLD_TEST)
    draws = rng.integers(groups, size=(20000, groups, 1)), rng.integers(worlds, size=(20000, 1, worlds))
    results = {}
    for mode in MODES:
        archive, metrics = {}, {}
        for arm in ARMS:
            start = perf_counter()
            data = run(BIRTH_TEST, WORLD_TEST, mode, arm, setup)
            metrics[arm] = measure(data)
            print(mode, arm, metrics[arm]["tail"].mean(), round(perf_counter() - start, 2), flush=True)
            for key, value in data.items():
                if key in ("birth", "world", "change", "good_side", "initial_mask", "initial_weight"):
                    if key in archive:
                        np.testing.assert_array_equal(archive[key], value)
                    else:
                        archive[key] = value
                else:
                    archive[f"{arm}/{key}"] = value
        path = out / f"{mode}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, **archive)
        score = metrics["product"]["tail"]
        results[mode] = dict(
            means={a: {k: float(v.mean()) for k, v in rows.items()} for a, rows in metrics.items()},
            primary={control: interval(score - metrics[control]["tail"], .025 / 6, draws)
                     for control in ("additive", "tanh", "product_sham")},
            chance=interval(score - .5, .025, draws), raw_sha256=digest(path),
        )
    write(out / "summary.json", dict(results=results, recipe=setup, development_sha256=digest(out / "development.json")), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"register": register, "develop": develop, "confirm": confirm}[args.stage](args.out)


if __name__ == "__main__":
    main()
