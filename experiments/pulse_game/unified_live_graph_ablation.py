"""Fresh matched change-address and physically closed port-cut controls."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import world
from .unified_live_graph import Graph, MOTOR, NO_CUT, SIZE, TASTE, Recipe, strongly_connected
from .unified_live_graph_study import Body, TAIL, TRIALS, digest, make_menu, read, run, save, write


BIRTHS = tuple(range(831500, 831508))
WORLDS = tuple(range(831520, 831532))
MODES = ("stable", "reverse")
ARMS = ("matched_random", "cut_motor", "cut_taste")


def register(folder, primary):
    folder.mkdir(parents=True, exist_ok=True)
    registered = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(Path(__file__)),
        "source_primary_registration_sha256": digest(primary / "registered.json"),
        "development_sha256": digest(primary / "development.json"),
        "recipe": read(primary / "development.json")["selected"]["recipe"],
        "births": BIRTHS, "worlds": WORLDS, "modes": MODES, "arms": ARMS,
        "protocol": "Joint donor and matched control have identical birth and physical world. Random edit events follow donor times/types, use independent uniformly drawn valid addresses, initialize from the recipient gradient's sign, and prefer retiring weak edges. Motor or taste cut arms train identically through the last receipt then disable only the named source columns while physically scoring fresh trials.",
        "endpoint": "Silent tail physical accuracy. Simultaneous crossed percentile intervals across six contrasts (each cut or random arm relative to joint in both modes).",
    }
    write(folder / "registered.json", registered, exclusive=True)


def random_edits(graph, donor_events, rng, fallback):
    gradient = graph.gradient() if (donor_events[:, 0] >= 0).any() else None
    added_removed = np.full_like(donor_events, -1)
    for lane, (grow, retire) in enumerate(donor_events):
        active = graph.mask[lane]
        weight = graph.weight[lane]
        if grow >= 0:
            eligible = ~active.copy()
            np.fill_diagonal(eligible, False)
            address = rng[lane].choice(np.flatnonzero(eligible))
            sign = np.sign(gradient[lane].flat[address])
            weight.flat[address] = .035 * (sign if sign else rng[lane].choice([-1, 1]))
            active.flat[address] = True
            added_removed[lane, 0] = address
        if retire >= 0:
            weak = np.abs(weight) <= .085
            preference = (active & weak, active & ~weak)
            stopped = False
            for level, eligible in enumerate(preference):
                for address in rng[lane].permutation(np.flatnonzero(eligible)):
                    active.flat[address] = False
                    if strongly_connected(active & (weight != 0)):
                        weight.flat[address] = 0
                        added_removed[lane, 1] = address
                        fallback[lane] += level
                        stopped = True
                        break
                    active.flat[address] = True
                if stopped:
                    break
            assert stopped
    return added_removed


def trial_run(mode, arm, recipe, donor):
    seeds = np.tile(WORLDS, len(BIRTHS))
    births = np.repeat(BIRTHS, len(WORLDS))
    menu, _, _, noise, _ = make_menu(seeds, mode)
    sim = Body(menu, None, TAIL)
    matched = arm == "matched_random"
    graph = Graph(births, recipe, editing=not matched)
    rng = [np.random.default_rng(np.random.SeedSequence([int(b), int(s), 4756447])) for b, s in zip(births, seeds, strict=True)]
    mask = NO_CUT.copy()
    if not matched:
        mask[MOTOR if arm == "cut_motor" else TASTE] = 0
    supplied = sim.observe()
    ed = np.full_like(donor["edits"], -1)
    fallback = np.zeros(len(seeds), dtype=np.int32)
    for trial in range(TRIALS):
        tail = trial >= TRIALS - TAIL
        cut = mask if tail else NO_CUT
        exploration = np.zeros(len(seeds), dtype=np.float32) if tail else recipe.exploration * noise[:, trial]
        graph.start(exploration)
        for _ in range(world.TICKS):
            supplied = sim.step(graph.advance(supplied, cut=cut))
        _, _, ed[:, trial], _ = graph.update(trial)
        if matched:
            ed[:, trial] = random_edits(graph, donor["edits"][:, trial], rng, fallback)
    res = sim.result()
    if matched:
        assert np.array_equal(ed >= 0, donor["edits"] >= 0)
    else:
        cutoff = TRIALS - TAIL
        assert np.array_equal(res.contact_side[:, :cutoff], donor["contact_side"][:, :cutoff])
        assert np.array_equal(res.contact_tick[:, :cutoff], donor["contact_tick"][:, :cutoff])
        assert np.array_equal(ed, donor["edits"])
        assert np.array_equal(graph.weight, donor["final_weight"])
    assert all(strongly_connected(a & (w != 0)) for a, w in zip(graph.mask, graph.weight, strict=True))
    return dict(correct=res.contact_side == menu.probability.argmax(axis=2), contact_side=res.contact_side,
                contact_tick=res.contact_tick, reward=res.reward, edits=ed, retirement_fallback=fallback,
                final_weight=graph.weight, final_active=graph.mask)


def interval(diff, draws):
    births, worlds = draws
    alternatives = diff.reshape(len(BIRTHS), len(WORLDS))[births, worlds].mean(axis=(1, 2))
    alpha = .05 / (2 * len(MODES) * len(ARMS))
    return [float(diff.mean()), *np.quantile(alternatives, [alpha, 1 - alpha]).tolist()]


def evaluate(folder, primary):
    registered = read(folder / "registered.json")
    primary_registration = read(primary / "registered.json")
    source_ok = all(digest(Path(__file__).parent / name) == value for name, value in primary_registration["source_sha256"].items())
    if not source_ok or registered["source_sha256"] != digest(Path(__file__)):
        raise RuntimeError("Registered source changed")
    if (registered["source_primary_registration_sha256"] != digest(primary / "registered.json")
            or registered["development_sha256"] != digest(primary / "development.json")):
        raise RuntimeError("Primary run provenance changed")
    recipe = Recipe(**registered["recipe"])
    rng = np.random.default_rng(9304358)
    n, m = len(BIRTHS), len(WORLDS)
    draws = rng.integers(n, size=(25000, n, 1)), rng.integers(m, size=(25000, 1, m))
    rows, hashes = {}, {}
    for mode in MODES:
        donor, _ = run(BIRTHS, WORLDS, mode, "joint", recipe)
        hashes[mode] = dict(joint=save(folder, mode, "joint", donor))
        score = donor["correct"][:, -TAIL:].mean(axis=1)
        rows[mode] = dict(joint=float(score.mean()))
        for arm in ARMS:
            value = trial_run(mode, arm, recipe, donor)
            hashes[mode][arm] = save(folder, mode, arm, value)
            control = value["correct"][:, -TAIL:].mean(axis=1)
            rows[mode][arm] = dict(tail=float(control.mean()), joint_minus=interval(score - control, draws),
                                   retirement_fallback=int(value["retirement_fallback"].sum()),
                                   matching_growth=int(((value["edits"][:, :, 0] == donor["edits"][:, :, 0]) & (donor["edits"][:, :, 0] >= 0)).sum()),
                                   matching_retirement=int(((value["edits"][:, :, 1] == donor["edits"][:, :, 1]) & (donor["edits"][:, :, 1] >= 0)).sum()))
        print(mode, rows[mode], flush=True)
    write(folder / "summary.json", dict(recipe=registered["recipe"], rows=rows, npz_sha256=hashes), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "run"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--primary", required=True, type=Path)
    args = parser.parse_args()
    (register if args.stage == "register" else evaluate)(args.out, args.primary)


if __name__ == "__main__":
    main()
