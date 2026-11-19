"""Audit the recorded pilot and intervene on trained pairs in fresh physical probes."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import graph_interactions as task, world
from .dendrite_ports_graph import DendriteGraph, FIRST, SECOND
from .dendrite_ports_study import ARMS, BIRTH_TEST, MODES, TAIL, TRIALS, WORLD_TEST, interval
from .unified_live_graph import Recipe
from .unified_live_graph_study import Body, audit_edits, digest, read, write


EXTRA = 128


def load(folder, summary, mode):
    path = folder / f"{mode}.npz"
    assert digest(path) == summary["results"][mode]["raw_sha256"]
    with np.load(path, allow_pickle=False) as raw:
        return dict(raw)


def audit(datasets):
    stable, reverse = (datasets[mode] for mode in MODES)
    before = np.arange(TRIALS)[None, :] < stable["change"][:, None]
    for arm in ARMS:
        for mode, data in datasets.items():
            prefix = f"{arm}/"
            contact, tick, correct, reward, shown, writes, edits = (data[prefix + field] for field in (
                "contact", "contact_tick", "correct", "reward", "shown", "writes", "edits"))
            np.testing.assert_array_equal(correct, (contact >= 0) & (contact == data["good_side"]))
            np.testing.assert_array_equal(contact >= 0, (tick >= 0) & (reward != 0))
            np.testing.assert_array_equal(shown[:, :-TAIL] != 0, contact[:, :-TAIL] >= 0)
            assert not shown[:, -TAIL:].any()
            assert not writes[shown == 0].any() and not (edits[shown == 0] >= 0).any()
            if arm != "product_sham":
                np.testing.assert_array_equal(shown[:, :-TAIL], reward[:, :-TAIL])
            else:
                assert (shown[:, :-TAIL] != reward[:, :-TAIL]).any()
            if arm != "polynomial":
                audit_edits(data["initial_mask"], data["initial_weight"], edits,
                            data[prefix + "final_mask"], data[prefix + "final_weight"])
            if prefix + "final_pair" in data:
                arcs = data[prefix + "final_mask"]
                assert not data[prefix + "final_pair"][~(arcs[:, :, FIRST] & arcs[:, :, SECOND])].any()
        for field in ("contact", "contact_tick", "shown", "writes", "edits"):
            a, b = stable[f"{arm}/{field}"], reverse[f"{arm}/{field}"]
            np.testing.assert_array_equal(a if arm == "product_sham" else a[before], b if arm == "product_sham" else b[before])


def fresh_menu(data, mode):
    menu, _, _, _ = task.make_menu(data["world"], mode, TRIALS + EXTRA)
    np.testing.assert_array_equal(menu.probability[:, :32].argmax(axis=-1), data["good_side"][:, :32])
    rest = {field: getattr(menu.body, field)[:, TRIALS:] for field in ("ctx", "cue", "left_id", "good_id", "distances", "wind")}
    return replace(menu, body=replace(menu.body, **rest), chemicals=menu.chemicals[:, TRIALS:],
                   probability=menu.probability[:, TRIALS:], realized=menu.realized[:, TRIALS:])


def probe(data, menu, setup, *, cut):
    brain = DendriteGraph(data["birth"], Recipe(**setup["recipe"]), product=True,
                          pair_rate=setup["pair_rate"], pair_decay=setup["pair_decay"])
    brain.weight = data["product/final_weight"].copy()
    brain.mask = data["product/final_mask"].copy()
    brain.pair = (np.zeros_like(brain.pair) if cut else data["product/final_pair"].copy())
    brain.refresh_pair_support()
    body = Body(menu, None, EXTRA)
    observation = body.observe()
    exploration = np.zeros(len(brain.state), dtype=np.float32)
    for _ in range(EXTRA):
        brain.start(exploration)
        for _ in range(world.TICKS):
            observation = body.step(brain.advance(observation))
    result = body.result()
    assert not result.observed_reward.any()
    truth = menu.probability.argmax(axis=-1)
    score = (result.contact_side >= 0) & (result.contact_side == truth)
    return dict(correct=score, contact=result.contact_side, tick=result.contact_tick, truth=truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    folder = args.out
    summary = read(folder / "summary.json")
    datasets = {mode: load(folder, summary, mode) for mode in MODES}
    audit(datasets)
    write(folder / "pair-ablation-registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(Path(__file__)),
        pilot_summary_sha256=digest(folder / "summary.json"), probes=EXTRA,
        hypothesis="In fresh unexposed physical trials, remove only the learned product pair coefficients from a mature graph. Carry identical learned ordinary weights and support and start both branches with zero activity. Both branches are silent and frozen. Compare crossed paired birth/world accuracy intervals with a two-contrast Bonferroni correction. This mechanism assay follows primary confirmation.",
    ), exclusive=True)
    rng = np.random.default_rng(85761845)
    n, m = len(BIRTH_TEST), len(WORLD_TEST)
    draws = rng.integers(n, size=(20000, n, 1)), rng.integers(m, size=(20000, 1, m))
    output = {}
    for mode, data in datasets.items():
        menu = fresh_menu(data, mode)
        restored = probe(data, menu, summary["recipe"], cut=False)
        removed = probe(data, menu, summary["recipe"], cut=True)
        pair = restored["correct"].mean(axis=1) - removed["correct"].mean(axis=1)
        raw = folder / f"{mode}-pair-ablation.npz"
        with raw.open("xb") as f:
            np.savez_compressed(f, **{f"{arm}/{k}": v for arm, row in (("intact", restored), ("removed", removed)) for k, v in row.items()})
        output[mode] = dict(intact=float(restored["correct"].mean()), removed=float(removed["correct"].mean()),
                            paired=interval(pair, .025 / 2, draws), raw_sha256=digest(raw),
                            contact_intact=float((restored["contact"] >= 0).mean()), contact_removed=float((removed["contact"] >= 0).mean()))
    write(folder / "checked.json", dict(raw_opportunities=int(len(BIRTH_TEST)*len(WORLD_TEST)*TRIALS*len(ARMS)*len(MODES)),
                                       strong_graphs_and_receipt_gating=True, same_trajectory_until_change=True,
                                       fair_taste_counterfactual_invariant=True, inference_pair_ablation=output), exclusive=True)
    print(output)


if __name__ == "__main__":
    main()
