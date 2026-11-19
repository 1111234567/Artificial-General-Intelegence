"""Measure an online corroborating pair rule on matched stable and reversing worlds."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from . import dendrite_ports_check as prior_check, graph_interactions as task, world
from .dendrite_ports_graph import DendriteGraph, widen_birth
from .sparse_alt_model import CorroboratingGraph, PAIR_COUNTS, ProbePolicy, _key
from .unified_live_graph import Graph, Recipe
from .unified_live_graph_study import Body, audit_edits, digest, read, write


BIRTH_DEV = tuple(range(872110, 872113))
WORLD_DEV = tuple(range(872140, 872146))
BIRTH_TEST = tuple(range(872210, 872218))
WORLD_TEST = tuple(range(872240, 872252))
TRIALS, TAIL = 1280, 112
MODES = ("stable", "reverse")
ARMS = ("corroborating", "sham", "dense", "tanh")
SETUP = dict(recipe=asdict(Recipe(.045, .23, .991)), pair_rate=.65, pair_decay=.993)
POLICIES = (
    ProbePolicy(6, 6, 2.0, .001, .003),
    ProbePolicy(6, 6, 1.65, .0004, .0006),
    ProbePolicy(8, 5, 1.45, .0003, .0004),
    ProbePolicy(8, 7, 1.45, .0002, .00015),
    ProbePolicy(12, 4, 1.25, .0003, .0003),
)
SOURCES = (
    "sparse_alt_study.py", "sparse_alt_model.py", "dendrite_ports_graph.py", "dendrite_ports_check.py",
    "graph_interactions.py", "unified_live_graph.py", "unified_live_graph_study.py", "mixture_world.py", "world.py",
)


class DenseCommonEdits(DendriteGraph):
    grow = Graph.grow
    retire = Graph.retire


def hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def register(out):
    out.mkdir(parents=True, exist_ok=True)
    write(out / "registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), sources=hashes(),
        birth_roots=dict(development=BIRTH_DEV, confirmation=BIRTH_TEST),
        world_roots=dict(development=WORLD_DEV, confirmation=WORLD_TEST),
        modes=MODES, arms=ARMS, trials=TRIALS, silent_tail=TAIL, setup=SETUP,
        policies=[asdict(policy) for policy in POLICIES],
        rule="Every receiver can keep a bounded number of transient probes. At a real receipt, sample a missing probe uniformly over current distinct incoming source pairs; reject live and already probed keys. Observe signed truncated receipt gradients over two consecutive disjoint blocks. Two sums must have the same nonzero sign, and pooled evidence must clear both the signed RMS threshold and the minimum mean. Admission initializes a live coefficient from pooled evidence. Live coefficients update through receipt-gated L1 soft thresholding, and expire at exact zero or if an incoming arc is removed. Q starts empty. There is no learned-pair quota or full dormant-pair table.",
        development="Choose the candidate with highest average stable/reversal silent accuracy among candidates remaining below a quarter of contemporaneous legal Q counts in every development lane and trial. Break ties by first candidate. If no candidate meets the constraint, select the one with smallest worst ratio and report failure.",
        confirmation="96 new crossed birth-world lives per scenario and arm, same first-order birth and editing rule. Compare stable and reversal silent physical accuracy with matched sham, native tanh and all-legal-pair dense arms. Also compare frozen learned pair-intact and pair-cut graphs on 128 fresh trials. Report 95% simultaneous crossed bootstrap intervals over the six performance and two deletion contrasts. Report live and probe counts, fraction of Q opportunity evaluated, and strict fraction threshold for every lane/trial. No test outcomes select policies.",
    ), exclusive=True)


def check(out):
    manifest = read(out / "registered.json")
    if manifest["sources"] != hashes():
        raise RuntimeError("Registered source changed")
    return manifest


def brain_for(arm, births, roots, policy):
    recipe = Recipe(**SETUP["recipe"])
    pair_parameters = dict(pair_rate=SETUP["pair_rate"], pair_decay=SETUP["pair_decay"])
    if arm in ("corroborating", "sham"):
        return CorroboratingGraph(births, recipe, world_seeds=roots, policy=policy, **pair_parameters)
    if arm == "dense":
        return DenseCommonEdits(births, recipe, product=True, **pair_parameters)
    graph = Graph(births, recipe)
    widen_birth(graph, births)
    return graph


def legal_count(mask):
    indegree = mask.sum(axis=-1)
    return (indegree * (indegree - 1) // 2).sum(axis=1)


def run(births, roots, mode, arm, policy, *, trials=TRIALS, tail=TAIL):
    births, roots = np.repeat(births, len(roots)), np.tile(roots, len(births))
    n = len(roots)
    menu, change, noise, signs = task.make_menu(roots, mode, trials)
    body = Body(menu, signs if arm == "sham" else None, tail)
    brain = brain_for(arm, births, roots, policy)
    edits = np.full((n, trials, 2), -1, dtype=np.int16)
    writes = np.zeros((n, trials, 3), dtype=np.int16)
    events = np.zeros((n, trials, len(PAIR_COUNTS)), dtype=np.int16)
    zero = np.zeros(n, dtype=np.float32)
    observe = body.observe()
    sparse = isinstance(brain, CorroboratingGraph)
    for trial in range(trials):
        brain.start(brain.recipe.exploration * noise[:, trial] if trial < trials - tail else zero)
        for _ in range(world.TICKS):
            observe = body.step(brain.advance(observe))
        writes[:, trial, 0], writes[:, trial, 1], edits[:, trial], writes[:, trial, 2] = brain.update(trial)
        if sparse:
            events[:, trial] = brain.pair_events
        else:
            events[:, trial, 1] = legal_count(brain.mask)
            if arm == "dense":
                events[:, trial, 0] = events[:, trial, 1]
                events[:, trial, 7] = brain.pair_writes
                events[:, trial, 8] = brain.newborn_pair_writes
    result = body.result()
    truth = menu.probability.argmax(axis=-1)
    payload = dict(
        birth=births, world=roots, change=change, truth=truth, correct=(result.contact_side == truth),
        contact=result.contact_side, tick=result.contact_tick, reward=result.reward,
        shown=result.observed_reward.sum(axis=-1, dtype=np.int8), edits=edits, writes=writes, events=events,
        initial_mask=brain.initial_mask, initial_weight=brain.initial_weight, final_mask=brain.mask, final_weight=brain.weight,
    )
    if sparse:
        payload.update(final_live=brain.live, final_values=brain.values)
    if arm == "dense":
        payload.update(final_dense=brain.pair)
    audit(payload, arm, tail)
    return payload


def audit(data, arm, tail=TAIL):
    events = data["events"]
    shown, contact = data["shown"], data["contact"]
    np.testing.assert_array_equal(data["correct"], contact == data["truth"])
    np.testing.assert_array_equal(contact >= 0, (data["tick"] >= 0) & (data["reward"] != 0))
    np.testing.assert_array_equal(shown[:, :-tail] != 0, contact[:, :-tail] >= 0)
    if arm != "sham":
        np.testing.assert_array_equal(shown[:, :-tail], data["reward"][:, :-tail])
    assert not shown[:, -tail:].any()
    assert not data["writes"][shown == 0].any() and not (data["edits"][shown == 0] >= 0).any()
    assert not events[shown == 0, 3:].any()
    audit_edits(data["initial_mask"], data["initial_weight"], data["edits"], data["final_mask"], data["final_weight"])
    if arm in ("corroborating", "sham"):
        lane, receiver, first, second = data["final_live"]
        keys = _key(*data["final_live"])
        assert len(np.unique(keys)) == len(keys)
        assert np.all(first < second) and np.all(data["final_values"] != 0)
        assert data["final_mask"][lane, receiver, first].all() and data["final_mask"][lane, receiver, second].all()
        np.testing.assert_array_equal(events[:, -1, 0], np.bincount(lane, minlength=len(events)))
        np.testing.assert_array_equal(events[:, :, 0], (events[:, :, 4] - events[:, :, 5] - events[:, :, 6]).cumsum(axis=1))


def metrics(data):
    events = data["events"]
    allocated, legal, probes, evaluated = events[:, :, :4].transpose(2, 0, 1)
    return dict(
        tail=data["correct"][:, -TAIL:].mean(axis=1),
        tail_contact=(data["contact"][:, -TAIL:] >= 0).mean(axis=1),
        pre_change=data["correct"][np.arange(len(events))[:, None], data["change"][:, None] + np.arange(-64, 0)].mean(axis=1),
        post_change=data["correct"][np.arange(len(events))[:, None], data["change"][:, None] + np.arange(64)].mean(axis=1),
        arc_growth=(data["edits"][:, :, 0] >= 0).sum(axis=1), arc_retirement=(data["edits"][:, :, 1] >= 0).sum(axis=1),
        active_average=allocated.mean(axis=1), active_final=allocated[:, -1], active_max=allocated.max(axis=1),
        legal_average=legal.mean(axis=1), legal_final=legal[:, -1],
        active_share_final=allocated[:, -1] / legal[:, -1], active_share_max=(allocated / legal).max(axis=1),
        probe_and_active_share_max=((allocated + probes) / legal).max(axis=1),
        probe_average=probes.mean(axis=1), probe_scores=evaluated.sum(axis=1),
        probe_score_share=evaluated.sum(axis=1) / (legal * (data["shown"] != 0)).sum(axis=1).clip(min=1),
        **{name: events[:, :, index].sum(axis=1) for index, name in enumerate(PAIR_COUNTS) if index >= 4},
    )


def describe(rows):
    return {key: dict(mean=float(values.mean()), min=float(values.min()), max=float(values.max())) for key, values in rows.items()}


def develop(out):
    check(out)
    ranked = []
    for index, policy in enumerate(POLICIES):
        start = perf_counter()
        results = {mode: describe(metrics(run(BIRTH_DEV, WORLD_DEV, mode, "corroborating", policy))) for mode in MODES}
        score = float(np.mean([row["tail"]["mean"] for row in results.values()]))
        worst = max(row["active_share_max"]["max"] for row in results.values())
        print(index, score, worst, round(perf_counter() - start, 2), flush=True)
        ranked.append(dict(index=index, score=score, worst_share=worst, policy=asdict(policy), results=results))
    winner = max(ranked, key=lambda r: (r["worst_share"] < .25, r["score"] if r["worst_share"] < .25 else -r["worst_share"]))
    write(out / "development.json", dict(winner=winner["index"], runs=ranked, registered_sha256=digest(out / "registered.json")), exclusive=True)


def frozen_probe(data, mode, policy, *, cut):
    menu = prior_check.fresh_menu({"world": data["world"], "good_side": data["truth"]}, mode)
    brain = brain_for("corroborating", data["birth"], data["world"], policy)
    brain.weight = data["final_weight"].copy()
    brain.mask = data["final_mask"].copy()
    if not cut:
        brain.live = data["final_live"].copy()
        brain.values = data["final_values"].copy()
    body = Body(menu, None, prior_check.EXTRA)
    observation = body.observe()
    zero = np.zeros(len(brain.state), dtype=np.float32)
    for _ in range(prior_check.EXTRA):
        brain.start(zero)
        for _ in range(world.TICKS):
            observation = body.step(brain.advance(observation))
    result = body.result()
    assert not result.observed_reward.any()
    truth = menu.probability.argmax(axis=-1)
    return dict(contact=result.contact_side, tick=result.contact_tick, truth=truth, correct=result.contact_side == truth)


def confirm(out):
    check(out)
    development = read(out / "development.json")
    if development["registered_sha256"] != digest(out / "registered.json"):
        raise RuntimeError("Registered manifest changed")
    policy = POLICIES[development["winner"]]
    rng = np.random.default_rng(160235359)
    n, m = len(BIRTH_TEST), len(WORLD_TEST)
    draws = rng.integers(n, size=(20000, n, 1)), rng.integers(m, size=(20000, 1, m))

    def interval(vec):
        sampled = vec.reshape(n, m)[draws[0], draws[1]].mean(axis=(1, 2))
        return [float(vec.mean()), *np.quantile(sampled, [.025 / 8, 1 - .025 / 8]).tolist()]

    outcomes, ordered = {}, {}
    for mode in MODES:
        results, archive = {}, {}
        for arm in ARMS:
            start = perf_counter()
            data = run(BIRTH_TEST, WORLD_TEST, mode, arm, policy)
            results[arm] = metrics(data)
            print(mode, arm, results[arm]["tail"].mean(), round(perf_counter() - start, 2), flush=True)
            for field, values in data.items():
                if field in ("birth", "world", "change", "initial_mask", "initial_weight", "truth"):
                    if field in archive:
                        np.testing.assert_array_equal(archive[field], values)
                    else:
                        archive[field] = values
                else:
                    archive[f"{arm}/{field}"] = values
            if arm == "corroborating":
                intact = frozen_probe(data, mode, policy, cut=False)
                cut = frozen_probe(data, mode, policy, cut=True)
                for name, probe in (("intact", intact), ("cut", cut)):
                    archive.update({f"probe/{name}/{key}": value for key, value in probe.items()})
        raw = out / f"{mode}.npz"
        with raw.open("xb") as stream:
            np.savez_compressed(stream, **archive)
        outcomes[mode] = dict(
            means={name: describe(value) for name, value in results.items()},
            paired={arm: interval(results["corroborating"]["tail"] - results[arm]["tail"]) for arm in ARMS[1:]},
            quarter_all_lanes_all_trials=bool(results["corroborating"]["active_share_max"].max() < .25),
            probe=dict(intact=float(intact["correct"].mean()), cut=float(cut["correct"].mean()),
                       intact_contact=float((intact["contact"] >= 0).mean()), cut_contact=float((cut["contact"] >= 0).mean()),
                       paired=interval((intact["correct"].astype(np.int8) - cut["correct"]).mean(axis=1))),
            raw_sha256=digest(raw),
        )
        ordered[mode] = archive
    before = np.arange(TRIALS)[None, :] < ordered["stable"]["change"][:, None]
    for arm in ARMS:
        for field in ("contact", "tick", "shown", "edits", "writes", "events"):
            a, b = (ordered[mode][f"{arm}/{field}"] for mode in MODES)
            np.testing.assert_array_equal(a if arm == "sham" else a[before], b if arm == "sham" else b[before])
    write(out / "summary.json", dict(results=outcomes, setup=SETUP, policy=asdict(policy),
                                    development_sha256=digest(out / "development.json")), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"register": register, "develop": develop, "confirm": confirm}[args.stage](args.out)


if __name__ == "__main__":
    main()
