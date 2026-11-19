"""Preregistered paired lifetimes for receipt-guided sparse interaction birth."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from . import graph_interactions as task, world
from .dendrite_ports_graph import DendriteGraph, FIRST, SECOND
from .sparse_pairs_model import BIRTH, EVIDENCE, EVENT, PER_LANE, SUPPORT, DenseWGraph, PairPolicy, SparseDendriteGraph
from .unified_live_graph import SIZE, Recipe
from .unified_live_graph_study import Body, audit_edits, digest, read, write


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/evidence/sparse-native-2026-11-19/main"
DATA = REPO / "data/pulse-game/sparse-native-20261119"
BIRTH_DEV = tuple(range(872110, 872113))
WORLD_DEV = tuple(range(872140, 872146))
BIRTH_TEST = tuple(range(872210, 872218))
WORLD_TEST = tuple(range(872240, 872252))
TRIALS, TAIL = 1280, 112
MODES = ("stable", "reverse")
ARMS = ("guided", "dense", "sealed_dense", "random", "sham")
BASES = ((Recipe(.045, .23, .991), .65, .993), (Recipe(.070, .21, .995), 1.2, .996))
POLICIES = (PairPolicy(24, .035, .075, 88), PairPolicy(48, .022, .075, 96),
            PairPolicy(64, .016, .10, 112), PairPolicy(80, .010, .105, 104),
            PairPolicy(96, .010, .065, 124))
SOURCES = ("sparse_pairs_model.py", "sparse_pairs_study.py", "dendrite_ports_graph.py", "dendrite_ports_study.py",
           "graph_interactions.py", "unified_live_graph.py", "unified_live_graph_study.py", "mixture_world.py", "world.py")
SEED = 29820475
MARGIN = -.03
BUDGET = .25
DEV_BUDGET = .23


def setup(base, policy):
    recipe, pair_rate, pair_decay = base
    return dict(recipe=asdict(recipe), pair_rate=pair_rate, pair_decay=pair_decay, policy=asdict(policy))


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def register(out):
    out.mkdir(parents=True, exist_ok=True)
    write(out / "registered.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hashes(),
        roots=dict(development=dict(birth=BIRTH_DEV, world=WORLD_DEV), confirmation=dict(birth=BIRTH_TEST, world=WORLD_TEST)),
        scenarios=MODES, arms=ARMS, physical_trials_per_life=TRIALS, silent_tail=TAIL,
        candidates=[setup(base, policy) for base in BASES for policy in POLICIES],
        baseline="dense uses the sealed DendriteGraph multiplication, dense pair-gradient update, and support refresh with ordinary Graph.grow and Graph.retire, as used in all sparse arms. sealed_dense uses original sealed base at its historical winner, including its pair-influenced ordinary edits. W is uncapped and updates independently in all arms. Each shares the same birth graph, raw chemical ports, physical body, private motor current, and scenario tape.",
        sparse="Start with zero installed Q and no dormant traces. Allocate proposal samples only when the native taste and contact ports report a signed receipt. Private stateless draws sample unordered physical addresses among all receiver/source combinations, reject duplicates, installed addresses and illegal incoming arcs. A surviving sample with sufficient absolute native receipt gradient receives the signed first update at birth. Later active gradients and decay match dense Q. Remove unsupported Q immediately, and independently retire weak Q once they reach the stated receipt age. No fixed budget, loss replacement, quotas, rank cutoff, or nominated chemical/receiver locations in guided arms.",
        controls="Random gets only the guided arm's current trial's per-life birth and total retirement counts, chooses random legal addresses from the sampled pool with private extra draws to fill the target if needed, uses its own signed gradient at birth and its own later receipts. Support removals take precedence over randomly matching the remainder of the target; a missed receipt, excess independent support losses or insufficient viable samples can give count differences, which are measured. Sham has the whole guided rule with independent fair taste bits replacing shown reward at the same physical contact.",
        selection=f"Use only development roots. For each candidate use the paired dense with equal W/Q hyperparameters. Calculate per mode each life/time guided/dense installed slots after every trial and from shared birth. Restrict to candidates whose maximum over life/time across both modes is at most {DEV_BUDGET:.2f}. In that set maximize the unweighted mean of the two silent accuracies. If none reach the buffer, find the smallest peak ratio, permit candidates within .01 absolute of it, then choose highest equal-mode accuracy. Registered order breaks equal scores. Compare confirm only after locking development selection.",
        endpoint=f"Best expected taste side on the first physical contact, timeout incorrect; final {TAIL} trials have neither public taste nor motor exploration. In both stable/reverse, guided must retain accuracy relative to both dense references, defined by all four lower interval endpoints above {MARGIN:.2f}; 95% crossed paired birth/world bootstrap percentile intervals simultaneously Bonferroni-adjusted across four. The per-life/per-time guided/matched-dense installed Q ratio must never exceed {BUDGET:.2f}. Report aggregate throughput, life peak/mean/final, and actual nonzero as separate metrics. Secondary guided-minus-random and guided-minus-sham in both scenarios form a separate four-contrast family; verify realized near-count matching and report limits.",
        archive="Each trial records actual contact/tick, taste, expected best, public receipt, ordinary edit addresses and W/Q updates, sampled counts, installed/nonzero counts and pair storage bytes. Each sparse pair birth and exit records address, time, receipt age origin, later write count, signed starting weight and gradient at birth, and exit reason/weight. Final records preserve ages and writes. Static bytes are authoritative+address caches+private random arrays; any RSS is whole-process and reported separately.",
        bootstrap=dict(draws=20000, seed=SEED),
    ), exclusive=True)


def check(out):
    registered = read(out / "registered.json")
    if registered["source_sha256"] != source_hashes():
        raise RuntimeError("Registered source changed")
    return registered


def make_graph(births, worlds, arm, configuration):
    if arm not in ARMS:
        raise ValueError(f"Unknown condition {arm}")
    if arm == "sealed_dense":
        configuration = setup(BASES[0], POLICIES[0])
    recipe = Recipe(**configuration["recipe"])
    pairs = dict(pair_rate=configuration["pair_rate"], pair_decay=configuration["pair_decay"])
    if arm in ("dense", "sealed_dense"):
        cls = DenseWGraph if arm == "dense" else DendriteGraph
        return cls(births, recipe, product=True, **pairs)
    return SparseDendriteGraph(births, recipe, world_seeds=worlds, policy=PairPolicy(**configuration["policy"]),
                               control="random" if arm == "random" else "guided", **pairs)


def dense_bytes(graph):
    return sum(array.nbytes for array in (graph.pair, graph.pair_credit, graph.symmetric, graph.valid_pairs))


def run(births, roots, mode, arm, setup, *, trials=TRIALS, tail=TAIL, match=None, mature_hook=None,
        menu_bundle=None, capture_events=True, audit=True):
    """Run complete physical lives, or supply matched custom menu/noise and maturity callback."""
    birth = np.repeat(births, len(roots))
    worlds = np.tile(roots, len(births))
    menu, changes, noise, sham = task.make_menu(worlds, mode, trials) if menu_bundle is None else menu_bundle
    graph = make_graph(birth, worlds, arm, setup)
    sparse = isinstance(graph, SparseDendriteGraph)
    body = Body(menu, sham if arm == "sham" else None, tail)
    n = len(birth)
    edits = np.full((n, trials, 2), -1, dtype=np.int16)
    writes = np.zeros((n, trials, 5), dtype=np.int16)
    counts = np.zeros((n, trials + 1), dtype=np.int16)
    nonzero = np.zeros_like(counts)
    births_count = np.zeros((n, trials), dtype=np.int16)
    retire_w = np.zeros_like(births_count)
    retire_q = np.zeros_like(births_count)
    sampled = np.zeros_like(births_count)
    memory = np.zeros(trials + 1, dtype=np.int64)
    events = []

    def track(index):
        if sparse:
            counts[:, index] = graph.active_counts()
            nonzero[:, index] = np.bincount(graph.lane[graph.active["weight"] != 0], minlength=n)
            memory[index] = graph.pair_storage_bytes()
        else:
            counts[:, index] = graph.valid_pairs.sum(axis=(1, 2))
            nonzero[:, index] = (graph.pair != 0).sum(axis=(1, 2))
            memory[index] = dense_bytes(graph)

    track(0)
    observed = body.observe()
    quiet = np.zeros(len(birth), dtype=np.float32)
    for trial in range(trials):
        if trial == trials - tail and mature_hook is not None:
            mature_hook(graph, body, observed)
        graph.start(graph.recipe.exploration * noise[:, trial] if trial < trials - tail else quiet)
        for _ in range(world.TICKS):
            observed = body.step(graph.advance(observed))
        if sparse:
            target = None if match is None else (match[0][:, trial], match[1][:, trial])
            updated = graph.update(trial, match=target)
            event = graph.pair_events
            sampled[:, trial] = graph.proposal_count
            if len(event):
                if capture_events:
                    events.append(event)
                for key, target in ((BIRTH, births_count), (SUPPORT, retire_w), (EVIDENCE, retire_q)):
                    target[:, trial] = np.bincount(event["key"][event["type"] == key] // PER_LANE, minlength=n)
        else:
            updated = graph.update(trial)
        writes[:, trial, 0], writes[:, trial, 1], edits[:, trial], writes[:, trial, 2] = updated
        writes[:, trial, 3], writes[:, trial, 4] = graph.pair_writes, graph.newborn_pair_writes
        track(trial + 1)

    result = body.result()
    shown = result.observed_reward.sum(axis=-1, dtype=np.int8)
    contact = result.contact_side
    best = menu.probability.argmax(axis=-1)
    correct = (contact == best) & (contact >= 0)
    assert not shown[:, -tail:].any() and not writes[shown == 0].any() and not (edits[shown == 0] >= 0).any()
    if audit:
        audit_edits(graph.initial_mask, graph.initial_weight, edits, graph.mask, graph.weight)
    payload = dict(birth=birth, world=worlds, change=changes, correct=correct, contact=contact, contact_tick=result.contact_tick,
                   taste=result.reward, shown=shown, good_side=best, edits=edits, writes=writes, active=counts, nonzero=nonzero,
                   pair_storage_bytes=memory, pair_births=births_count, pair_support_exits=retire_w, pair_evidence_exits=retire_q,
                   samples=sampled, initial_mask=graph.initial_mask, initial_weight=graph.initial_weight,
                   final_mask=graph.mask, final_weight=graph.weight, final_rate=graph.rate, final_bias=graph.bias, final_state=graph.state,
                   final_receipts=graph.receipts, final_generation=graph.generation, final_generation_writes=graph.generation_writes)
    if sparse:
        assert not counts[:, 0].any() and not ((births_count + retire_w + retire_q)[shown == 0]).any()
        np.testing.assert_array_equal(np.diff(counts, axis=1), births_count - retire_w - retire_q)
        payload["final_active"] = graph.active
        if capture_events:
            payload["pair_events"] = np.concatenate(events) if events else np.empty(0, dtype=EVENT)
    else:
        assert not graph.pair[~graph.valid_pairs].any()
        payload["final_pair"] = graph.pair
    return payload


def _dev_dense(job):
    idx, base = job
    configuration = setup(base, POLICIES[0])
    print("calibrating dense", idx, flush=True)
    return idx, {mode: run(BIRTH_DEV, WORLD_DEV, mode, "dense", configuration, capture_events=False, audit=False)["active"]
                 for mode in MODES}


def _dev_guided(job):
    idx, configuration, dense = job
    stats = {}
    for mode in MODES:
        data = run(BIRTH_DEV, WORLD_DEV, mode, "guided", configuration, capture_events=False, audit=False)
        fraction = data["active"] / dense[mode]
        stats[mode] = dict(accuracy=float(data["correct"][:, -TAIL:].mean()), peak_ratio=float(fraction.max()),
                           mean_ratio=float(fraction.mean()), final_ratio=float(fraction[:, -1].mean()))
    result = dict(index=idx, setup=configuration, scenarios=stats, accuracy=float(np.mean([v["accuracy"] for v in stats.values()])),
                  peak_ratio=max(v["peak_ratio"] for v in stats.values()))
    print("development", result, flush=True)
    return result


def develop(out, workers):
    check(out)
    path = out / "development.json"
    if path.exists():
        raise RuntimeError("Already developed")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        dense = dict(pool.map(_dev_dense, enumerate(BASES)))
        work = [(i * len(POLICIES) + j, setup(base, policy), dense[i]) for i, base in enumerate(BASES)
                for j, policy in enumerate(POLICIES)]
        candidates = list(pool.map(_dev_guided, work))
    eligible = [row for row in candidates if row["peak_ratio"] <= DEV_BUDGET]
    if not eligible:
        peak = min(c["peak_ratio"] for c in candidates)
        eligible = [row for row in candidates if row["peak_ratio"] <= peak + .01]
    selected = max(eligible, key=lambda row: row["accuracy"])
    write(path, dict(registered_sha256=digest(out / "registered.json"), selected=selected,
                     eligible=[row["index"] for row in eligible], candidates=candidates), exclusive=True)


def sample_interval(difference, draws, tail):
    grouped = difference.reshape(len(BIRTH_TEST), len(WORLD_TEST))
    boot = grouped[draws].mean(axis=(1, 2))
    return [float(grouped.mean()), *np.quantile(boot, [tail, 1 - tail]).tolist()]


def metrics(data):
    life = data["active"]
    counts = lambda matrix: float(matrix.sum(axis=1).mean())
    n = len(life)
    frames = np.arange(n)[:, None], data["change"][:, None] + np.arange(64)
    before = frames[0], frames[1] - 64
    active_age = data["final_active"]["born"] if "final_active" in data else np.empty(0)
    observations = dict(tail_accuracy=float(data["correct"][:, -TAIL:].mean()), tail_contact=float((data["contact"][:, -TAIL:] >= 0).mean()),
                        before_change_accuracy=float(data["correct"][before].mean()), early_after_change_accuracy=float(data["correct"][frames].mean()),
                        all_accuracy=float(data["correct"].mean()), physical_trials=int(data["correct"].size),
                        shown_receipts_per_life=counts(data["shown"] != 0), birth_pairs=float(life[:, 0].mean()),
                        mean_pairs=float(life.mean()), mean_life_peak_pairs=float(life.max(axis=1).mean()), max_life_peak_pairs=int(life.max()),
                        final_pairs=float(life[:, -1].mean()), final_nonzero_pairs=float(data["nonzero"][:, -1].mean()),
                        ordinary_births=counts(data["edits"][..., 0] >= 0), ordinary_retires=counts(data["edits"][..., 1] >= 0),
                        ordinary_changed=counts(data["writes"][..., 0]), ordinary_newborn_gradient_writes=counts(data["writes"][..., 1]),
                        pair_births=counts(data["pair_births"]), pair_support_exits=counts(data["pair_support_exits"]),
                        pair_evidence_exits=counts(data["pair_evidence_exits"]), active_pair_gradient_writes=counts(data["writes"][..., 3]),
                        paired_newborn_arc_pair_gradient_writes=counts(data["writes"][..., 4]), sampled_legal_dormant=counts(data["samples"]),
                        persistent_pair_bytes_per_life=float(data["pair_storage_bytes"].mean() / n),
                        peak_persistent_pair_bytes_per_life=float(data["pair_storage_bytes"].max() / n),
                        final_persistent_pair_bytes_per_life=float(data["pair_storage_bytes"][-1] / n))
    if "pair_events" in data:
        events = data["pair_events"]
        exits = events[events["type"] < 0]
        born = events[events["type"] == BIRTH]
        observations |= dict(exit_age_median=float(np.median(exits["trial"] - exits["born"])) if len(exits) else None,
                             final_age_median=float(np.median(TRIALS - 1 - active_age)) if len(active_age) else None,
                             fraction_born_positive=float((born["weight"] > 0).mean()) if len(born) else None,
                             fraction_final_with_later_writes=float((data["final_active"]["writes"] > 0).mean()) if len(active_age) else None)
    return observations


def ratios(numerator, denominator):
    life = numerator / denominator
    aggregate = numerator.sum(axis=0) / denominator.sum(axis=0)
    return dict(peak_all_lives=float(life.max()), mean_life_peaks=float(life.max(axis=1).mean()),
                mean_all_lives=float(life.mean()), final_mean_life=float(life[:, -1].mean()),
                peak_cohort=float(aggregate.max()), mean_cohort=float(aggregate.mean()), final_cohort=float(aggregate[-1]),
                worst_peak_life=int(life.max(axis=1).argmax()), worst_peak_after_trial=int(life.max(axis=0).argmax() - 1),
                life_peaks=life.max(axis=1).tolist(), life_means=life.mean(axis=1).tolist(), life_final=life[:, -1].tolist())


def _confirm_mode(job):
    mode, configuration, destination = job
    archive, scores, means, match = {}, {}, {}, None
    for arm in ARMS:
        start = perf_counter()
        data = run(BIRTH_TEST, WORLD_TEST, mode, arm, configuration, match=match if arm == "random" else None)
        if arm == "guided":
            match = data["pair_births"], data["pair_support_exits"] + data["pair_evidence_exits"]
        means[arm] = metrics(data)
        scores[arm] = data["correct"][:, -TAIL:].mean(axis=1)
        for key, item in data.items():
            if key in ("birth", "world", "change", "good_side", "initial_mask", "initial_weight"):
                if key in archive:
                    np.testing.assert_array_equal(archive[key], item)
                else:
                    archive[key] = item
            else:
                archive[f"{arm}/{key}"] = item
        print("confirmation", mode, arm, means[arm]["tail_accuracy"], round(perf_counter() - start, 2), flush=True)
    storage = ratios(archive["guided/active"], archive["dense/active"])
    storage_sealed = ratios(archive["guided/active"], archive["sealed_dense/active"])
    real_births, real_exits = match
    rand_births, rand_exits = archive["random/pair_births"], archive["random/pair_support_exits"] + archive["random/pair_evidence_exits"]
    matching = dict(exact_birth_trial_fraction=float((rand_births == real_births).mean()),
                    exact_exit_trial_fraction=float((rand_exits == real_exits).mean()),
                    missing_births=int((real_births - rand_births).clip(min=0).sum()),
                    excess_births=int((rand_births - real_births).clip(min=0).sum()),
                    mean_absolute_stock_difference=float(np.abs(archive["random/active"] - archive["guided/active"]).mean()),
                    exact_stock_fraction=float((archive["random/active"] == archive["guided/active"]).mean()))
    path = destination / f"{mode}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **archive)
    return mode, scores, dict(arms=means, installed_vs_dense=storage, installed_vs_sealed=storage_sealed,
                              random_matching=matching, raw_sha256=digest(path))


def confirm(out, destination, workers):
    registration = check(out)
    selection = read(out / "development.json")
    if (out / "summary.json").exists() or selection["registered_sha256"] != digest(out / "registered.json"):
        raise RuntimeError("Already confirmed or registration changed")
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / f"{mode}.npz").exists() for mode in MODES):
        raise RuntimeError("Confirmation archive already exists")
    configuration = selection["selected"]["setup"]
    rng = np.random.default_rng(registration["bootstrap"]["seed"])
    b, w = len(BIRTH_TEST), len(WORLD_TEST)
    draws = rng.integers(b, size=(20000, b, 1)), rng.integers(w, size=(20000, 1, w))
    summaries = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(MODES))) as pool:
        for mode, scores, results in pool.map(_confirm_mode, [(mode, configuration, destination) for mode in MODES]):
            results["paired"] = {arm: sample_interval(scores["guided"] - scores[arm], draws, .025 / 4) for arm in ARMS[1:]}
            results["retention_pass"] = all(results["paired"][arm][1] > MARGIN for arm in ("dense", "sealed_dense"))
            results["sparsity_pass"] = results["installed_vs_dense"]["peak_all_lives"] <= BUDGET
            summaries[mode] = results
    passed = all(value["retention_pass"] and value["sparsity_pass"] for value in summaries.values())
    write(out / "summary.json", dict(registered_sha256=digest(out / "registered.json"), development_sha256=digest(out / "development.json"),
                                     setup=configuration, joint_endpoint_pass=passed, scenarios=summaries), exclusive=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "register":
        register(args.out)
    elif args.stage == "develop":
        develop(args.out, args.workers)
    else:
        confirm(args.out, args.data, args.workers)


if __name__ == "__main__":
    main()
