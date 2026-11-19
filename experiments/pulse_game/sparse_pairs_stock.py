"""Confirm online random placement at the same installed Q count per receiver."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import graph_interactions as task, world
from .dendrite_ports_graph import FIRST, SECOND
from .sparse_pairs_model import BIRTH, RECORD, WIDTH, SparseDendriteGraph, PairPolicy, absent, mix
from .sparse_pairs_study import DATA, MODES, OUT, TAIL, TRIALS, check, run
from .unified_live_graph import Recipe, SIZE
from .unified_live_graph_study import Body, audit_edits, digest, read, write


BIRTHS = tuple(range(872310, 872318))
WORLDS = tuple(range(872340, 872352))
BOOTSTRAP = 29820511


def by_destination(keys, priorities, counts):
    destination = keys // WIDTH
    order = np.lexsort((priorities, destination))
    ordered = destination[order]
    rank = np.arange(len(ordered)) - np.searchsorted(ordered, ordered)
    return order[rank < counts.ravel()[ordered]]


class ReceiverMatchedGraph(SparseDendriteGraph):
    """Quota is current matched receiver stock and never controls which address wins."""

    def _sample(self, trial, fresh, active, batch=0):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint64)

    def _evidence_retire(self, records, support, fresh, trial, match):
        planned, target = match
        keys = records["key"]
        destinations = keys // WIDTH
        current = np.bincount(destinations, minlength=self.state.size).reshape(self.state.shape)
        # Existing active Q cannot depend on an arc that has just been born.
        degree = (self.mask & (self.generation != trial)).sum(axis=2)
        available = degree * (degree - 1) // 2
        feasible = np.minimum(available, target)
        self.quota = np.minimum(np.maximum(planned, feasible - current), np.minimum(feasible, available - current))
        self.quota[~fresh] = 0
        remove = np.where(fresh[:, None], current + self.quota - feasible, 0)
        priority = mix(self.private_keys[destinations // SIZE] ^ keys.astype(np.uint64) ^ np.uint64(trial))
        selected = by_destination(keys, priority, remove)
        retire = np.zeros(len(records), dtype=bool)
        retire[selected] = True
        return retire

    def _post_edit_samples(self, candidates, priority, old, fresh, edits, trial, match):
        # Enumerate eligibility transiently only for receivers requesting random placements.
        lane, receiver = np.nonzero(self.quota > 0)
        live = self.mask[lane, receiver] & (self.generation[lane, receiver] != trial)
        destination, pair = np.nonzero(live[:, FIRST] & live[:, SECOND])
        found = (lane[destination] * SIZE + receiver[destination]) * WIDTH + pair
        found = found[absent(found, old)].astype(np.int32)
        key = np.uint64((trial * 0xA0761D6478BD642F) & ((1 << 64) - 1))
        priority = mix(self.private_keys[found // (SIZE * WIDTH)] ^ found.astype(np.uint64) ^ key)
        selected = by_destination(found, priority, self.quota)
        chosen = found[selected]
        self.proposal_count[:] = np.bincount(chosen // (SIZE * WIDTH), minlength=len(fresh))
        return chosen, priority[selected]

    def _births(self, candidates, priority, gradient, trial, match):
        initial = (self.pair_rate * gradient).clip(-1.25, 1.25)
        chosen = initial != 0
        keys = candidates[chosen]
        new = np.empty(len(keys), dtype=RECORD)
        new["key"], new["weight"], new["born"], new["writes"] = keys, initial[chosen], trial, 0
        new["born_receipt"] = self.receipts[keys // (SIZE * WIDTH)]
        return new, gradient[chosen]


def receiver_stocks(guided):
    events = guided["pair_events"]
    destinations = events["key"] // WIDTH
    lanes, receivers = np.divmod(destinations, SIZE)
    shape = (len(guided["birth"]), TRIALS, SIZE)
    at = lanes, events["trial"], receivers
    births = np.zeros(shape, dtype=np.int16)
    np.add.at(births, at, events["type"] == BIRTH)
    exits = np.zeros_like(births)
    np.add.at(exits, at, events["type"] < 0)
    stock = (births - exits).cumsum(axis=1, dtype=np.int16)
    np.testing.assert_array_equal(stock.sum(axis=-1), guided["active"][:, 1:])
    return births, stock


def run_stock(mode, config, guided, planned, target):
    births, roots = guided["birth"], guided["world"]
    recipe = Recipe(**config["recipe"])
    graph = ReceiverMatchedGraph(births, recipe, world_seeds=roots, pair_rate=config["pair_rate"], pair_decay=config["pair_decay"],
                                 policy=PairPolicy(**config["policy"]), control="random")
    menu, changes, noise, _ = task.make_menu(roots, mode, TRIALS)
    body = Body(menu, None, TAIL)
    stock = np.zeros_like(target)
    birth_count = np.zeros_like(stock)
    capacity = np.zeros_like(stock)
    events = []
    edits = np.full((len(roots), TRIALS, 2), -1, dtype=np.int16)
    writes = np.zeros((len(roots), TRIALS, 4), dtype=np.int16)
    quiet = np.zeros(len(births), dtype=np.float32)
    observed = body.observe()
    for trial in range(TRIALS):
        graph.start(recipe.exploration * noise[:, trial] if trial < TRIALS - TAIL else quiet)
        for _ in range(world.TICKS):
            observed = body.step(graph.advance(observed))
        writes[:, trial, 0], writes[:, trial, 1], edits[:, trial], _ = graph.update(
            trial, match=(planned[:, trial], target[:, trial]))
        writes[:, trial, 2], writes[:, trial, 3] = graph.pair_writes, graph.newborn_pair_writes
        stock[:, trial] = np.bincount(graph.destination, minlength=graph.state.size).reshape(graph.state.shape)
        degree = (graph.mask & (graph.generation != trial)).sum(axis=2)
        capacity[:, trial] = degree * (degree - 1) // 2
        event = graph.pair_events
        if len(event):
            events.append(event)
            born = event["key"][event["type"] == BIRTH] // WIDTH
            birth_count[:, trial] = np.bincount(born, minlength=graph.state.size).reshape(graph.state.shape)
    result = body.result()
    shown = result.observed_reward.sum(axis=-1, dtype=np.int8)
    assert not writes[shown == 0].any() and not (edits[shown == 0] >= 0).any()
    assert not shown[:, -TAIL:].any()
    np.testing.assert_array_equal(guided["change"], changes)
    np.testing.assert_array_equal(guided["initial_mask"], graph.initial_mask)
    np.testing.assert_array_equal(guided["initial_weight"], graph.initial_weight)
    truth = menu.probability.argmax(axis=-1)
    np.testing.assert_array_equal(guided["good_side"], truth)
    audit_edits(graph.initial_mask, graph.initial_weight, edits, graph.mask, graph.weight)
    return dict(active=stock, correct=result.contact_side == truth, contact=result.contact_side, contact_tick=result.contact_tick,
                taste=result.reward, shown=shown, edits=edits, writes=writes, born=birth_count, capacity=capacity,
                final_mask=graph.mask, final_weight=graph.weight, final_active=graph.active, events=np.concatenate(events))


def _evaluate(item):
    mode, config, data_root = item
    guided = run(BIRTHS, WORLDS, mode, "guided", config)
    planned, target = receiver_stocks(guided)
    random = run_stock(mode, config, guided, planned, target)
    penalty = random["active"] - target
    score_a, score_b = [matrix["correct"][:, -TAIL:].mean(axis=1) for matrix in (guided, random)]
    archive = data_root / f"stock-{mode}.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **guided, target=target, planned=planned,
                            **{f"random/{key}": value for key, value in random.items()})
    metrics = dict(accuracy=dict(guided=float(score_a.mean()), random=float(score_b.mean())),
                   receiver_slots_exact=float((penalty == 0).mean()), cohorts_exact=float((penalty == 0).all(axis=2).mean()),
                   mean_absolute_pair_difference_per_life_trial=float(np.abs(penalty).sum(axis=2).mean()),
                   quiet_receiver_slots_exact=float((penalty[:, -TAIL:] == 0).mean()),
                   final_receiver_slots_exact=float((penalty[:, -1] == 0).mean()), excess=int(penalty.clip(min=0).sum()),
                   deficit=int((-penalty).clip(min=0).sum()), capacity_deficit=int((target - random["capacity"]).clip(min=0).sum()),
                   birth_count_exact=float((planned == random["born"]).mean()), born_guided=int(planned.sum()),
                   born_random=int(random["born"].sum()), physical_trials=int(guided["contact"].size), raw_sha256=digest(archive))
    print(mode, metrics, flush=True)
    return mode, score_a - score_b, metrics


def register(out):
    check(out)
    write(out / "stock-match-registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(Path(__file__)),
        base_registered_sha256=digest(out / "registered.json"), base_development_sha256=digest(out / "development.json"),
        roots=dict(birth=BIRTHS, world=WORLDS), modes=MODES, trials=TRIALS, silent_tail=TAIL,
        control="Use each new guided life to record current after-update receiver stock and births by receiver. Random keeps its own physical tastes, ordinary changes, and active pair learning. Remove pairs on ordinary arc retirement first. On its own signed receipt, randomly remove or create enough legal pairs at each receiver to target matched stock, retaining the planned birth count when feasible. Newly added arcs are excluded from births for that trial. Random births sample uniformly without replacement from eligible addresses at that receiver and start with their own signed gradient. No hidden world input or future counts drive decisions. Low legal capacity, zero gradient or missing own receipt can prevent matching; report exact per-receiver rate, joint per-life/trial match rate and size of mismatch.",
        endpoints="Compare final silent physical accuracy on each scenario for guidance versus randomized support with receiver-wise matched stock. Crossed birth/world bootstrap, 95% simultaneous intervals across the two scenarios. Interpret as a successful stock control only if at least 99% of life/trial/receiver cells match in both regimes, including a separately reported quiet tail. Compare independent guides anew without post-hoc changes.",
        bootstrap_seed=BOOTSTRAP, draws=20000), exclusive=True)


def confirm(out, data_root):
    check(out)
    registered = out / "stock-match-registered.json"
    reg = read(registered)
    if (reg["source_sha256"], reg["base_registered_sha256"], reg["base_development_sha256"]) != (
            digest(Path(__file__)), digest(out / "registered.json"), digest(out / "development.json")):
        raise RuntimeError("Supplemental stock control source or base selection changed")
    config = read(out / "development.json")["selected"]["setup"]
    rng = np.random.default_rng(reg["bootstrap_seed"])
    draw_b = rng.integers(len(BIRTHS), size=(reg["draws"], len(BIRTHS), 1))
    draw_w = rng.integers(len(WORLDS), size=(reg["draws"], 1, len(WORLDS)))
    results = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for mode, delta, stats in pool.map(_evaluate, [(mode, config, data_root) for mode in MODES]):
            sample = delta.reshape(len(BIRTHS), len(WORLDS))[draw_b, draw_w].mean(axis=(1, 2))
            stats["guided_minus_random"] = [float(delta.mean()), *np.quantile(sample, [.0125, .9875]).tolist()]
            stats["stock_matching_pass"] = stats["receiver_slots_exact"] >= .99
            results[mode] = stats
    write(out / "stock-match-summary.json", dict(registered_sha256=digest(registered), results=results), exclusive=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.stage == "register":
        register(args.out)
    else:
        confirm(args.out, args.data)


if __name__ == "__main__":
    main()
