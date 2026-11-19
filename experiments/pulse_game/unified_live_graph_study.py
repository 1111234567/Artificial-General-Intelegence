"""Matched lifetime experiments for a directly embodied adaptive graph."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import mixture_world as food, world
from .unified_live_graph import Graph, MOTOR, NO_CUT, Recipe, SIZE, TASTE, strongly_connected


DEV_BIRTHS = tuple(range(831240, 831244))
DEV_WORLDS = tuple(range(831260, 831266))
TEST_BIRTHS = tuple(range(831300, 831308))
TEST_WORLDS = tuple(range(831320, 831332))
TRIALS = 640
TAIL = 48
WINDOW = 64
MODES = ("stable", "reverse", "restore")
ARMS = ("joint", "weights", "frozen", "no_receipt", "independent")
RECIPES = (
    Recipe(.12, .37), Recipe(.22, .28), Recipe(.32, .22),
    Recipe(.16, .25, .999), Recipe(.24, .34, .999),
)
SOURCES = ("unified_live_graph.py", "unified_live_graph_study.py", "mixture_world.py", "world.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, *, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def register(folder):
    folder.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    protocol = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: digest(here / name) for name in SOURCES},
        "birth_roots": {"development": DEV_BIRTHS, "confirmation": TEST_BIRTHS},
        "world_roots": {"development": DEV_WORLDS, "confirmation": TEST_WORLDS},
        "trials": TRIALS, "mode": MODES, "arms": ARMS, "probe_tail": TAIL, "window": WINDOW,
        "recipe_candidates": [asdict(item) for item in RECIPES],
        "development_objective": "Maximize mean of stable silent tail accuracy and reversal silent tail accuracy; ties choose first recipe.",
        "primary_confirmation": "Joint-minus-frozen and joint-minus-independent in stable silent and reversed silent tails. Crossed birth/world paired bootstrap, 95% simultaneous percentile intervals across four contrasts.",
        "secondary_confirmation": "Initial, pre-change, initial-reversal and restored windows, weight-only, receipt-off; physical realized tastes, contact and expected-best side.",
        "causal_audit": "Clone identical states/weights/physical observations and silence only source columns of outgoing motor or taste edges. Compare internal nodes, next motor states, and emitted pulses in fixed public replay.",
    }
    write(folder / "registered.json", protocol, exclusive=True)


def make_menu(seeds, mode, *, trials=TRIALS):
    """One private continuous preference per life; a reversal never alters the sensory tape."""
    base = food.make(seeds, trials, rule="linear")
    taste = base.weights[:, 0, :food.DIM]
    utilities = np.einsum("btij,bj->bti", base.chemicals * 2 - 1, taste)
    change = np.empty(len(seeds), dtype=np.int16)
    recovery = np.empty_like(change)
    uniform = np.empty(utilities.shape, dtype=np.float32)
    noise = np.empty((len(seeds), trials), dtype=np.float32)
    signs = np.empty((len(seeds), trials), dtype=np.int8)
    for i, root in enumerate(seeds):
        schedule, taste_rng, noise_rng, sign_rng = (
            np.random.default_rng(branch) for branch in np.random.SeedSequence([int(root), 9813427]).spawn(4)
        )
        change[i] = trials // 3 + schedule.integers(-trials // 32, trials // 32 + 1)
        recovery[i] = (2 * trials) // 3 + schedule.integers(-trials // 32, trials // 32 + 1)
        if mode != "stable":
            stop = recovery[i] if mode == "restore" else trials
            utilities[i, change[i]:stop] *= -1
        uniform[i] = taste_rng.random((trials, 2))
        sample = noise_rng.random(trials).clip(1e-4, 1 - 1e-4)
        noise[i] = np.log(sample / (1 - sample))
        signs[i] = sign_rng.choice([-1, 1], size=trials)
    probability = (1 / (1 + np.exp(-4.8 * utilities))).astype(np.float32)
    realized = np.where(uniform < probability, 1, -1).astype(np.int8)
    return replace(base, probability=probability, realized=realized), change, recovery, noise, signs


class Body(food.Body):
    """Keep physical movement and actual tastes, with controlled public receipt presentation."""

    def __init__(self, menu, signs, probe_tail):
        self.signs = signs
        super().__init__(menu, hidden_after=menu.body.trials - probe_tail)

    def _shown_reward(self, receipt):
        shown = super()._shown_reward(receipt)
        if self.signs is not None:
            shown = np.where(shown != 0, self.signs[:, self.trial_index], np.int8(0))
        return shown


def audit_edits(initial, initial_weight, events, final, final_weight):
    """Independently replay every address and check strong connectivity after each removal."""
    masks = initial.copy()
    assert all(strongly_connected(mask & (weight != 0)) for mask, weight in zip(masks, initial_weight, strict=True))
    for lane, trial, operation in zip(*np.nonzero(events >= 0), strict=True):
        address = events[lane, trial, operation]
        target, source = divmod(int(address), SIZE)
        assert target != source
        assert masks[lane, target, source] == bool(operation)
        masks[lane, target, source] = not bool(operation)
        if operation:
            assert strongly_connected(masks[lane])
    assert np.array_equal(masks, final)
    assert all(strongly_connected(mask & (weight != 0)) for mask, weight in zip(final, final_weight, strict=True))


def paths(graph, obs, pulses, noise):
    """Matched open-loop interventions use the exact same initial state and subsequent observations."""
    lanes = min(12, len(graph.state))
    chosen = np.linspace(0, len(graph.state) - 1, lanes, dtype=int)
    obs, pulses, noise = obs[chosen], pulses[chosen], noise[chosen]
    birth_states = graph.state[chosen]

    def clone(state):
        replica = Graph(range(lanes), learning=False, editing=False)
        replica.weight = graph.weight[chosen]
        replica.state = state.copy()
        replica.rate = graph.rate[chosen]
        replica.bias = graph.bias[chosen]
        replica.start(noise)
        return replica

    subsets = {"motor": MOTOR, "taste": TASTE}
    counterparts = {name: clone(birth_states) for name in subsets}
    cuts = {name: NO_CUT.copy() for name in subsets}
    for name, port in subsets.items():
        cuts[name][port] = 0
    interior = np.ones(SIZE, dtype=bool)
    interior[MOTOR] = interior[TASTE] = False
    downstream = np.ones(SIZE, dtype=bool)
    downstream[TASTE] = False
    reference = clone(birth_states)
    rows = {name: dict(max_other=0., max_motor=0., pulses_changed=0, observed_ticks=0)
            for name in subsets}
    rows["taste"]["max_receipt_one_hop"] = 0.
    rows["taste"]["max_receipt_one_hop_cut"] = 0.
    rows["taste"]["isolated_receipts"] = 0
    activity = np.zeros_like(reference.state)
    for tick in range(obs.shape[1]):
        public = obs[:, tick]
        receipt_channels = slice(world.REWARD_POS, world.REWARD_NEG + 1)
        received = public[:, receipt_channels].any(axis=1)
        if received.any() and tick + 1 < obs.shape[1]:
            silent_public = public.copy()
            silent_public[:, receipt_channels] = 0
            full_receipt, no_receipt = clone(reference.state), clone(reference.state)
            full_receipt.advance(public)
            no_receipt.advance(silent_public)
            rows["taste"]["isolated_receipts"] += int(received.sum())
            for kind, cut in (("max_receipt_one_hop", NO_CUT), ("max_receipt_one_hop_cut", cuts["taste"])):
                held_receipt, held_silent = clone(full_receipt.state), clone(no_receipt.state)
                held_receipt.advance(obs[:, tick + 1], cut=cut)
                held_silent.advance(obs[:, tick + 1], cut=cut)
                delta = np.abs(held_receipt.state[received][:, downstream] - held_silent.state[received][:, downstream])
                rows["taste"][kind] = max(rows["taste"][kind], float(delta.max()))
        expected = reference.advance(obs[:, tick])
        assert np.array_equal(expected, pulses[:, tick])
        activity = np.maximum(activity, np.abs(reference.tick_delta))
        for name, counterpart in counterparts.items():
            actual = counterpart.advance(obs[:, tick], cut=cuts[name])
            delta = np.abs(counterpart.state - reference.state)
            rows[name]["max_other"] = max(rows[name]["max_other"], float(delta[:, interior].max()))
            rows[name]["max_motor"] = max(rows[name]["max_motor"], float(delta[:, MOTOR].max()))
            active = obs[:, tick, world.ACTIVE] > 0
            rows[name]["pulses_changed"] += int((active & (actual != expected).any(axis=1)).sum())
            rows[name]["observed_ticks"] += int(active.sum())
    return rows, int((activity > 1e-6).all(axis=0).sum())


def run(births, seeds, mode, arm, recipe, *, trials=TRIALS, probe_tail=TAIL, causal=False):
    all_births = np.repeat(births, len(seeds))
    all_seeds = np.tile(seeds, len(births))
    menu, change, recovery, noise, independent = make_menu(all_seeds, mode, trials=trials)
    body = Body(menu, independent if arm == "independent" else None, probe_tail)
    if arm == "no_receipt":
        body.feedback = "none"
    population = Graph(all_births, recipe, learning=arm != "frozen", editing=arm not in ("weights", "frozen"))
    shape = (len(all_seeds), trials)
    edits = np.full((*shape, 2), -1, dtype=np.int16)
    counts = np.zeros((*shape, 3), dtype=np.int16)
    sens = body.observe()
    replay_rows = []
    for trial in range(trials):
        # All meals are freshly sampled; the last window removes receipt and exploratory drive.
        exploration = recipe.exploration * noise[:, trial] if trial < trials - probe_tail else np.zeros(len(all_seeds), dtype=np.float32)
        population.start(exploration)
        inspect = causal and trial in (62, trials // 2, trials - probe_tail - 1)
        if inspect:
            prior = Graph(all_births, learning=False, editing=False)
            prior.weight, prior.rate, prior.bias = population.weight.copy(), population.rate.copy(), population.bias.copy()
            prior.state = population.state.copy()
            observed, actions = [], []
        for _ in range(world.TICKS):
            pulses = population.advance(sens)
            if inspect:
                observed.append(sens)
                actions.append(pulses)
            sens = body.step(pulses)
        if inspect:
            replay_rows.append(paths(prior, np.stack(observed, axis=1), np.stack(actions, axis=1), exploration))
        changed, newborn, edits[:, trial], retired = population.update(trial)
        counts[:, trial, 0] = changed
        counts[:, trial, 1] = newborn
        counts[:, trial, 2] = retired
    result = body.result()
    audit_edits(population.initial_mask, population.initial_weight, edits, population.mask, population.weight)
    if np.any(counts[result.observed_reward.sum(axis=-1) == 0]) or np.any(edits[result.observed_reward.sum(axis=-1) == 0] >= 0):
        raise RuntimeError("A silent trial edited the graph")
    hit = result.contact_side >= 0
    best_side = menu.probability.argmax(axis=2)
    expected = np.take_along_axis(menu.probability, result.contact_side.clip(min=0)[..., None], axis=2)[..., 0] * hit
    score = (result.contact_side == best_side) & hit
    return {
        "birth": all_births, "world": all_seeds, "good_side": best_side, "expected_utility": expected,
        "correct": score, "contact_side": result.contact_side, "contact_tick": result.contact_tick,
        "reward": result.reward, "shown": result.observed_reward.sum(axis=2, dtype=np.int8),
        "change": change, "recovery": recovery, "initial_active": population.initial_mask,
        "initial_weight": population.initial_weight, "final_active": population.mask, "final_weight": population.weight,
        "edits": edits, "writes": counts, "final_generation_writes": population.generation_writes,
    }, replay_rows


def save(folder, mode, arm, data):
    path = folder / f"{mode}-{arm}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **data)
    return digest(path)


def develop(folder):
    assert not (folder / "development.json").exists()
    rows = []
    for index, recipe in enumerate(RECIPES):
        tails, hashes = {}, {}
        for mode in MODES[:2]:
            result, _ = run(DEV_BIRTHS, DEV_WORLDS, mode, "joint", recipe)
            tails[mode] = float(result["correct"][:, -TAIL:].mean())
            hashes[mode] = save(folder, mode, f"development-{index}", result)
        row = dict(candidate=index, recipe=asdict(recipe), tails=tails, objective=sum(tails.values()) / 2, files=hashes)
        rows.append(row)
        print(row, flush=True)
    selected = max(rows, key=lambda row: row["objective"])
    write(folder / "development.json", {"candidates": rows, "selected": selected}, exclusive=True)


def measurements(data):
    score = data["correct"]
    relatives = np.arange(WINDOW)[None, :]
    index = np.arange(len(score))[:, None]
    change = data["change"][:, None]
    return {
        "initial": score[:, :WINDOW].mean(axis=1),
        "pre_change": score[index, change - WINDOW + relatives].mean(axis=1),
        "initial_post_change": score[index, change + relatives].mean(axis=1),
        "pre_restore": score[index, data["recovery"][:, None] - WINDOW + relatives].mean(axis=1),
        "tail": score[:, -TAIL:].mean(axis=1),
        "tail_expected_utility": data["expected_utility"][:, -TAIL:].mean(axis=1),
        "contact": (data["contact_side"] >= 0).mean(axis=1),
        "tail_contact": (data["contact_side"][:, -TAIL:] >= 0).mean(axis=1),
        "positive_reward": (data["reward"] > 0).mean(axis=1),
        "growth": (data["edits"][:, :, 0] >= 0).sum(axis=1),
        "retirement": (data["edits"][:, :, 1] >= 0).sum(axis=1),
        "newborn_receipt_writes": data["writes"][:, :, 1].sum(axis=1),
        "newborn_generations_written": (data["final_generation_writes"] > 0).sum(axis=(1, 2)) + (data["writes"][:, :, 2] > 0).sum(axis=1),
        "effective_arc_delta": np.abs(data["final_weight"] - data["initial_weight"]).sum(axis=(1, 2)),
    }


def bootstrap(difference, *, alpha):
    n, m = len(TEST_BIRTHS), len(TEST_WORLDS)
    samples = 25000
    rng = np.random.default_rng(9836741)
    births = rng.integers(n, size=(samples, n, 1))
    worlds = rng.integers(m, size=(samples, 1, m))
    alternatives = difference.reshape(n, m)[births, worlds].mean(axis=(1, 2))
    return [float(difference.mean()), *np.quantile(alternatives, [alpha / 2, 1 - alpha / 2]).tolist()]


def confirm(folder):
    assert not (folder / "summary.json").exists()
    recipe = Recipe(**read(folder / "development.json")["selected"]["recipe"])
    results, hashes, causal = {}, {}, {}
    for mode in MODES:
        results[mode] = {}
        hashes[mode] = {}
        for arm in ARMS:
            start = perf_counter()
            population, replay = run(TEST_BIRTHS, TEST_WORLDS, mode, arm, recipe, causal=arm == "joint")
            if arm == "joint":
                causal[mode] = replay
            results[mode][arm] = measurements(population)
            hashes[mode][arm] = save(folder, mode, arm, population)
            print(mode, arm, results[mode][arm]["tail"].mean(), round(perf_counter() - start, 2), flush=True)
    comparisons = {f"{mode}_joint_minus_{arm}": bootstrap(results[mode]["joint"]["tail"] - results[mode][arm]["tail"], alpha=.05 / 4)
                   for mode in MODES[:2] for arm in ("frozen", "independent")}
    means = {mode: {arm: {name: float(value.mean()) for name, value in row.items()}
                    for arm, row in arms.items()} for mode, arms in results.items()}
    write(folder / "summary.json", dict(recipe=asdict(recipe), means=means, primary=comparisons, source_cuts=causal,
                                       npz_sha256=hashes), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "register":
        register(args.out)
    else:
        here = Path(__file__).resolve().parent
        registered = read(args.out / "registered.json")
        if {name: digest(here / name) for name in SOURCES} != registered["source_sha256"]:
            raise RuntimeError("Registered source changed")
        (develop if args.stage == "develop" else confirm)(args.out)


if __name__ == "__main__":
    main()
