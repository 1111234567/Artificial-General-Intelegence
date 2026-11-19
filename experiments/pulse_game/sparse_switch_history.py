"""Test preserving and reusing historical interaction addresses during a physical rule switch."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import sparse_pair_switch as switch
from .sparse_pairs_model import BIRTH, EVIDENCE, EVENT, PER_LANE, SUPPORT, PairPolicy, SparseDendriteGraph, legal
from .unified_live_graph import Recipe, strongly_connected
from .unified_live_graph_study import digest, read, write


BIRTHS = tuple(switch.REG["births"])
WORLDS = tuple(range(964620, 964628))
TRAIN = switch.REG["training_trials_per_stage"]
TAIL = switch.REG["silent_trials"]
ARMS = ("full", "preserve_a_end", "reuse_a_ever")
CONTRASTS = (("full", "preserve_a_end"), ("full", "reuse_a_ever"), ("reuse_a_ever", "preserve_a_end"))
SEED, DRAWS = 675798, 20000
SOURCES = ("sparse_switch_history.py", "sparse_pair_switch.py", "sparse_pairs_model.py", "ring_logic_study.py",
           "dendrite_ports_graph.py", "unified_live_graph.py", "unified_live_graph_study.py",
           "graph_interactions.py", "mixture_world.py", "world.py")


class HistoryGraph(SparseDendriteGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protect_from_evidence_exit = False
        self.allowed_pair_births = None
        self.events = []

    def _sample(self, trial, fresh, active, batch=0):
        keys, priority = super()._sample(trial, fresh, active, batch=batch)
        if self.allowed_pair_births is not None:
            keep = np.isin(keys, self.allowed_pair_births, kind="sort")
            keys, priority = keys[keep], priority[keep]
        return keys, priority

    def _evidence_retire(self, records, support, fresh, trial, match):
        if self.protect_from_evidence_exit:
            return np.zeros(len(records), dtype=bool)
        return super()._evidence_retire(records, support, fresh, trial, match)

    def update(self, trial, *, match=None):
        ordinary = super().update(trial, match=match)
        if len(self.pair_events):
            self.events.append(self.pair_events.copy())
        return ordinary


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def register(out):
    out.mkdir(parents=True, exist_ok=True)
    if set(WORLDS).intersection(switch.REG["roots"]):
        raise ValueError("Supplement must have distinct world seeds")
    write(out / "history-registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hashes(),
        primary_switch_sha256=digest(switch.EVIDENCE / "protocol.json"),
        recipe_source_sha256=digest(switch.MAIN / "development.json"), setup=switch.setup(),
        births=BIRTHS, worlds=WORLDS, arms=ARMS, contrasts=CONTRASTS, sign_twins=("B", "minus_B"),
        a_training=TRAIN, b_training=TRAIN, b_silent_probe=TAIL, bootstrap=dict(seed=SEED, draws=DRAWS),
        design="Use the primary physical switch tape generator, same birth seed groups, disjoint fresh world roots and architecture recipe selected before this switch. Train A once with the primary guided actor; clone the same state into all B arms and sign twins. At B640 use a fresh silent probe, no extra endpoints or A return.",
        full="Ordinary guided pair proposals and exits continue.",
        preserve_a_end="Disable all B pair births and voluntary evidence-based pair exit, so A-end pairs remain reusable while both incoming W supports remain. W edits and W/Q weight learning continue. Support loss still deletes the relevant Q and that address cannot regrow.",
        reuse_a_ever="Run ordinary B learning, exits and finite random proposals, with the same draw stream and proposal budget. Filter its sampled pair addresses through the complete set born during A in that life. An A address that was lost can be installed again if eligible and responsive. Do not add extra draws or tune the threshold.",
        inference="Physical expected-best first-contact accuracy including timeout as incorrect. Average sign twins first, preserve paired arms, crossed bootstrap over birth groups and independent new world roots. Three two-sided 95% simultaneous percentile contrasts. Describe final counts, birth events inside/outside A ever-set, support/evidence exits, verified address policy, receipt counts and contact rates. No decision based on observed outcomes.",
        limits="B pair stock and birth opportunity can differ across interventions. Effects of prior support retention, recycling and expansion are policy comparisons conditional on their own coadapted W and receipt histories. The full-minus-reuse contrast does not isolate address novelty at a common stock. No decomposition against the historical no-new arm on different worlds.",
    ), exclusive=True)


def concatenated_events(actor):
    return np.concatenate(actor.events) if actor.events else np.empty(0, dtype=EVENT)


def simulate(birth_ids):
    roots = WORLDS * len(birth_ids)
    births = np.repeat(birth_ids, len(WORLDS))
    config = switch.setup()
    actor = HistoryGraph(births, Recipe(**config["recipe"]), world_seeds=roots,
                         policy=PairPolicy(**config["policy"]), pair_rate=config["pair_rate"], pair_decay=config["pair_decay"])
    _, _, used = switch.play(actor, *switch.meal(roots, 0), 0)
    a_events = concatenated_events(actor)
    ever = np.unique(a_events["key"][a_events["type"] == BIRTH])
    a_live = actor.active["key"].copy()
    if not np.isin(a_live, ever).all():
        raise RuntimeError("A-end addresses must be present in the A birth log")
    actor.events.clear()
    result = {"birth": births, "world": np.asarray(roots), "A/stock": used, "A/events": a_events,
              "A/ever": ever, "A/final": actor.active.copy()}

    for twin in (0, 1):
        training = switch.meal(roots, 1, negative=bool(twin))
        probe = switch.meal(roots, 1, negative=bool(twin), probe=True)
        for arm in ARMS:
            current = deepcopy(actor)
            if arm == "preserve_a_end":
                current.proposals_enabled = False
                current.protect_from_evidence_exit = True
            if arm == "reuse_a_ever":
                current.allowed_pair_births = ever
            train_codes, _, stock = switch.play(current, *training, TRAIN)
            events = concatenated_events(current)
            installed = current.active["key"]
            enters = events["key"][events["type"] == BIRTH]
            if arm == "preserve_a_end" and (len(enters) or (events["type"] == EVIDENCE).any()
                                             or not np.isin(installed, a_live).all()):
                raise RuntimeError("A-end preservation did not respect its address policy")
            if arm == "reuse_a_ever" and (not np.isin(enters, ever).all() or not np.isin(installed, ever).all()):
                raise RuntimeError("Recycling installed an address outside A history")
            if not legal(current.mask, installed).all() or not all(
                    strongly_connected(mask & (weight != 0)) for mask, weight in zip(current.mask, current.weight, strict=True)):
                raise RuntimeError("Illegal mature physical support")
            codes, _, _ = switch.play(current, *probe, 2 * TRAIN, silent=True)
            prefix = f"{arm}/{twin}"
            result.update({f"{prefix}/B_probe": codes, f"{prefix}/B_stock": stock,
                           f"{prefix}/B_final": current.active.copy(), f"{prefix}/B_events": events,
                           f"{prefix}/B_shown": train_codes[..., 4]})
    for arm in ARMS:
        one, other = (result[f"{arm}/{twin}/B_probe"] for twin in (0, 1))
        np.testing.assert_array_equal(one[..., 3], 1 - other[..., 3])
    return result


def estimate(chunks, draw_b, draw_w):
    means, scores = {}, {}
    stock_a = np.concatenate([chunk["A/stock"][:, -1] for chunk in chunks])
    ever_a = np.concatenate([np.bincount(chunk["A/ever"] // PER_LANE, minlength=len(chunk["birth"]))
                             for chunk in chunks])
    for arm in ARMS:
        stats = np.zeros(7, np.int64)
        twins, ends, peaks = [], [], []
        for twin in (0, 1):
            prefix = f"{arm}/{twin}"
            probes, stocks = [], []
            for chunk in chunks:
                probes.append(chunk[f"{prefix}/B_probe"])
                stocks.append(chunk[f"{prefix}/B_stock"])
                event = chunk[f"{prefix}/B_events"]
                born = event["key"][event["type"] == BIRTH]
                known = np.isin(born, chunk["A/ever"])
                final = chunk[f"{prefix}/B_final"]["key"]
                stats += (np.count_nonzero(known), np.count_nonzero(~known),
                          np.count_nonzero(event["type"] == SUPPORT), np.count_nonzero(event["type"] == EVIDENCE),
                          np.count_nonzero(np.isin(final, chunk["A/ever"])),
                          np.count_nonzero(~np.isin(final, chunk["A/ever"])),
                          np.count_nonzero(chunk[f"{prefix}/B_shown"]))
            twins.append(np.concatenate(probes))
            stock = np.concatenate(stocks)
            ends.append(stock[:, -1])
            peaks.append(stock.max(axis=1))
        probe = np.stack(twins)
        final_stock = np.stack(ends)
        covered = probe[..., 0] >= 0
        scores[arm] = switch.accuracy(probe).mean(axis=0)
        metrics = ("births_at_A_history", "births_outside_A_history", "support_exits", "evidence_exits",
                   "end_addresses_in_A_history", "end_addresses_outside_A_history", "B_receipts_shown")
        means[arm] = dict(accuracy=float(scores[arm].mean()), trials=covered.size, contacted=int(covered.sum()),
                          sign_accuracy=switch.accuracy(probe).mean(axis=1).tolist(),
                          mean_end_stock=float(final_stock.mean()), min_end_stock=int(final_stock.min()),
                          max_end_stock=int(final_stock.max()), zero_end_stock=int((final_stock == 0).sum()),
                          mean_peak_stock=float(np.mean(peaks)), **dict(zip(metrics, map(int, stats), strict=True)))

    def interval(diff):
        crossed = diff.reshape(len(BIRTHS), len(WORLDS))
        sampled = crossed[draw_b, draw_w].mean(axis=(1, 2))
        tail = .025 / len(CONTRASTS)
        return [float(diff.mean()), *np.quantile(sampled, [tail, 1 - tail]).tolist()]

    return dict(a_mean_end_stock=float(stock_a.mean()), a_mean_ever_installed=float(ever_a.mean()), means=means,
                primary={f"{a}_minus_{b}": interval(scores[a] - scores[b]) for a, b in CONTRASTS})


def confirm(out, workers):
    protocol = read(out / "history-registered.json")
    if (protocol["source_sha256"] != source_hashes() or
            protocol["recipe_source_sha256"] != digest(switch.MAIN / "development.json") or
            protocol["primary_switch_sha256"] != digest(switch.EVIDENCE / "protocol.json")):
        raise RuntimeError("Registered source or frozen switches changed")
    groups = [BIRTHS[i:i+2] for i in range(0, len(BIRTHS), 2)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(simulate, groups))
    rng = np.random.default_rng(SEED)
    draw_b = rng.integers(len(BIRTHS), size=(DRAWS, len(BIRTHS), 1))
    draw_w = rng.integers(len(WORLDS), size=(DRAWS, 1, len(WORLDS)))
    summary = estimate(chunks, draw_b, draw_w)
    path = out / "history-raw.zip"
    with ZipFile(path, "x") as archive:
        hashes = {}
        for group, chunk in zip(groups, chunks, strict=True):
            buf = BytesIO()
            np.savez_compressed(buf, **chunk)
            payload = buf.getvalue()
            name = f"{group[0]}.npz"
            archive.writestr(name, payload)
            hashes[name] = sha256(payload).hexdigest()
    write(out / "history-summary.json", dict(**summary, raw_archive_sha256=digest(path), entry_sha256=hashes,
                                             registered_sha256=digest(out / "history-registered.json")), exclusive=True)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "confirm"))
    parser.add_argument("--out", type=Path, default=switch.EVIDENCE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "register":
        register(args.out)
    else:
        confirm(args.out, args.workers)
