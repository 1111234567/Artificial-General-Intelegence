"""Matched mature-life interventions on the addresses of sparse dendrites."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from . import sparse_pairs_study, world
from .unified_live_graph import SIZE
from .unified_live_graph_study import digest, read, write


BIRTHS = tuple(range(872310, 872318))
WORLDS = tuple(range(872340, 872352))
MODES = ("stable", "reverse")
TRIALS, TAIL, DRAWS = 1280, 112, 4
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED, PERMUTATION_SEED = 70931602, 255362911
FIRST, SECOND = np.triu_indices(SIZE, k=1)
PAIR_COUNT = len(FIRST)
ARMS = ("intact", "selected_shuffle", "eligible_shuffle", "zero")
CONTRASTS = (("intact", "eligible_shuffle"), ("selected_shuffle", "eligible_shuffle"),
             ("intact", "selected_shuffle"), ("intact", "zero"))
MOBILITY = ("selected", "nonzero", "all_eligible", "eligible_at_selected_receivers",
            "relocatable", "reassignable", "selected_locations_lost", "selected_weights_changed",
            "nonzero_at_new_locations")
SOURCES = ("sparse_location_study.py", "sparse_pairs_study.py", "sparse_pairs_model.py",
           "dendrite_ports_graph.py", "graph_interactions.py", "unified_live_graph.py", "unified_live_graph_study.py",
           "mixture_world.py", "world.py")


def reassign(active, mask, births, roots, draw):
    """Place the same permuted signed weights at old or uniformly drawn legal sites."""
    selected = active.copy()
    eligible = active.copy()
    legal = mask[:, :, FIRST] & mask[:, :, SECOND]
    counts = np.zeros((len(mask), len(MOBILITY)), dtype=np.int32)
    counts[:, 2] = legal.sum(axis=(1, 2))
    if len(active) == 0:
        return selected, eligible, counts

    order = np.argsort(active["key"], kind="stable")
    groups = active["key"][order] // PAIR_COUNT
    if groups[0] < 0 or groups[-1] >= len(mask) * SIZE:
        raise ValueError("An interaction receiver falls outside the population")
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    for rows in np.split(order, boundaries):
        group, site = np.divmod(active["key"][rows], PAIR_COUNT)
        lane, receiver = divmod(int(group[0]), SIZE)
        available = np.flatnonzero(legal[lane, receiver])
        if len(np.unique(site)) != len(site) or not np.isin(site, available).all():
            raise ValueError("Active coefficients must occupy distinct legal incoming pairs")
        rng = np.random.default_rng(np.random.SeedSequence(
            [PERMUTATION_SEED, int(births[lane]), int(roots[lane]), draw, receiver]))
        q = active["weight"][rows]
        permutation = rng.permutation(len(rows))
        random_site = np.sort(rng.choice(available, size=len(rows), replace=False))
        selected["weight"][rows] = q[permutation]
        eligible["weight"][rows] = q[permutation]
        eligible["key"][rows] = int(group[0]) * PAIR_COUNT + random_site
        counts[lane] += (len(site), np.count_nonzero(q), 0, len(available),
                         len(site) if len(available) > len(site) else 0,
                         len(site) if len(np.unique(q)) > 1 else 0,
                         np.count_nonzero(~np.isin(site, random_site)),
                         np.count_nonzero(q != q[permutation]),
                         np.count_nonzero((q[permutation] != 0) & ~np.isin(random_site, site)))
    return selected, eligible, counts


def score_fork(graph, body, observed, records, tail):
    """Use the actor's own dynamics and its normal body on a taste-silent identical tape."""
    actor, branch = deepcopy(graph), deepcopy(body)
    obs = observed.copy()
    actor.set_active(records.copy())
    actor.learning = actor.editing = False
    for attr in ("mask", "weight", "state", "rate", "bias"):
        np.testing.assert_array_equal(getattr(actor, attr), getattr(graph, attr))

    for _ in range(tail):
        actor.start(np.zeros(len(actor.state), dtype=np.float32))
        for _ in range(world.TICKS):
            obs = branch.step(actor.advance(obs))
    for attr in ("mask", "weight", "rate", "bias"):
        np.testing.assert_array_equal(getattr(actor, attr), getattr(graph, attr))
    np.testing.assert_array_equal(actor.active, records[np.argsort(records["key"])])
    result = branch.result()
    if result.observed_reward[:, -tail:].any():
        raise RuntimeError("A silent branch displayed a taste")
    return dict(contact=result.contact_side[:, -tail:].copy(),
                contact_tick=result.contact_tick[:, -tail:].copy(),
                reward=result.reward[:, -tail:].copy())


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def register(out, primary):
    sparse_pairs_study.check(primary)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hashes(),
        primary_registration_sha256=digest(primary / "registered.json"),
        recipe_selection="The guided recipe selected by the frozen primary development protocol; no selection on location seeds.",
        births=BIRTHS, worlds=WORLDS, modes=MODES, trials=TRIALS, silent_tail=TAIL, arms=ARMS,
        draws=DRAWS, permutation_seed=PERMUTATION_SEED, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_draws=BOOTSTRAP_DRAWS, contrasts=CONTRASTS,
        boundary="Same mature life at the first tick of the silent tail. All arms start with the same learned ordinary weights, topology, bias, rates, recurrent and body state. Freeze training, hide public taste, disable motor exploration. Each branch makes its own physical contacts.",
        placement="Select a uniformly random subset of eligible pairs of the same size at each receiving vertex, including original addresses in the candidate pool. Independently permute the identical signed Q values to fill either this subset or the original subset. Couple the Q ordering across the two shuffle arms for each fixed draw. Preserve empty receivers and include all lives, even if no move is possible.",
        primary="Expected best-food side among all attempted fresh trials, using physical first contact, with a no-contact scoring zero. Use argmax in a tie. Average the four draws within each life; average matched lifetime differences with a crossed birth/world percentile bootstrap. Two-sided 95% simultaneous family across the four contrasts and two scenarios.",
        secondary="Raw contacts, contact ticks, rewards, no-contact/accuracy among contacts, exact ties, legal and realized intervention denominators, Q intact above chance at nominal 95%, and bit-exact contact replay on the intact training branch. The selected-shuffle minus eligible-shuffle contrast measures placement at fixed receiver and coefficient multiset under randomized assignment.",
    ), exclusive=True)


def run_forks(births, roots, mode, setup, *, trials=TRIALS, tail=TAIL):
    birth = np.repeat(births, len(roots))
    root = np.tile(roots, len(births))
    data = dict(birth=birth, world=root)

    def mature(graph, body, observed):
        if body.trial_index != trials - tail or body.tick_index != 0 or body.hidden_after != body.trial_index:
            raise RuntimeError("Fork requires the first tick of the silent tail")
        np.testing.assert_array_equal(observed, body.observe())
        active = graph.active.copy()
        data.update(mature_mask=graph.mask.copy(), mature_weight=graph.weight.copy(),
                    mature_state=graph.state.copy(), mature_rate=graph.rate.copy(), mature_bias=graph.bias.copy(),
                    mature_receipts=graph.receipts.copy(), pair_intact=active)

        def save(arm, records, draw=None):
            name = arm if draw is None else f"{arm}/{draw}"
            result = score_fork(graph, body, observed, records, tail)
            for key, value in result.items():
                data[f"{name}/{key}"] = value
            if draw is not None:
                data[f"{name}/pair"] = records

        save("intact", active)
        zero = active.copy()
        zero["weight"] = 0
        save("zero", zero)
        for draw in range(DRAWS):
            selected, eligible, counts = reassign(active, graph.mask, birth, root, draw)
            save("selected_shuffle", selected, draw)
            save("eligible_shuffle", eligible, draw)
            data[f"mobility/{draw}"] = counts
        data["good_side"] = body.menu.probability[:, -tail:].argmax(axis=-1)
        data["exact_tie"] = body.menu.probability[:, -tail:, 0] == body.menu.probability[:, -tail:, 1]

    factual = sparse_pairs_study.run(births, roots, mode, "guided", setup,
                                     trials=trials, tail=tail, mature_hook=mature, capture_events=False)
    np.testing.assert_array_equal(data["intact/contact"], factual["contact"][:, -tail:])
    np.testing.assert_array_equal(data["intact/contact_tick"], factual["contact_tick"][:, -tail:])
    np.testing.assert_array_equal(data["intact/reward"], factual["taste"][:, -tail:])
    np.testing.assert_array_equal(data["mature_mask"], factual["final_mask"])
    np.testing.assert_array_equal(data["mature_weight"], factual["final_weight"])
    np.testing.assert_array_equal(data["pair_intact"], factual["final_active"])
    return data


def estimate(data, bootstrap_birth, bootstrap_world):
    scores, means = {}, {}
    good = data["good_side"]
    for arm in ARMS:
        names = [arm] if arm in ("intact", "zero") else [f"{arm}/{draw}" for draw in range(DRAWS)]
        contact = np.stack([data[f"{name}/contact"] for name in names])
        reward = np.stack([data[f"{name}/reward"] for name in names])
        reached = contact >= 0
        correct = (contact == good) & reached
        scores[arm] = correct.mean(axis=(0, 2))
        n_contact, n_correct = int(reached.sum()), int(correct.sum())
        means[arm] = dict(accuracy=float(correct.mean()), contact=float(reached.mean()),
                          accuracy_among_contacts=n_correct / n_contact if n_contact else None,
                          trials=contact.size, contacted=n_contact, no_contact=contact.size - n_contact,
                          correct=n_correct, positive=int(np.count_nonzero(reward > 0)))

    def ci(values, tail):
        crossed = values.reshape(len(BIRTHS), len(WORLDS))
        samples = crossed[bootstrap_birth, bootstrap_world].mean(axis=(1, 2))
        return [float(values.mean()), *np.quantile(samples, [tail, 1 - tail]).tolist()]

    family_tail = .025 / (len(MODES) * len(CONTRASTS))
    counts = np.stack([data[f"mobility/{draw}"] for draw in range(DRAWS)])
    return dict(means=means,
                primary={f"{a}_minus_{b}": ci(scores[a] - scores[b], family_tail) for a, b in CONTRASTS},
                above_chance=ci(scores["intact"] - .5, .025),
                exact_ties=int(data["exact_tie"].sum()),
                lives_without_pairs=int((counts[0, :, 0] == 0).sum()),
                lives_with_relocatable_pairs=int((counts[0, :, 4] > 0).sum()),
                lives_with_reassignable_weights=int((counts[0, :, 5] > 0).sum()),
                mobility={key: count.tolist() for key, count in zip(MOBILITY, counts.sum(axis=1).T, strict=True)})


def confirm(out, primary):
    protocol = read(out / "registered.json")
    if source_hashes() != protocol["source_sha256"] or digest(primary / "registered.json") != protocol["primary_registration_sha256"]:
        raise RuntimeError("Registered source or registered primary protocol changed")
    dev_path = primary / "development.json"
    dev = read(dev_path)
    if dev["registered_sha256"] != protocol["primary_registration_sha256"]:
        raise RuntimeError("Primary development recipe does not match its registered protocol")
    setup = dev["selected"]["setup"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_birth = rng.integers(len(BIRTHS), size=(BOOTSTRAP_DRAWS, len(BIRTHS), 1))
    bootstrap_world = rng.integers(len(WORLDS), size=(BOOTSTRAP_DRAWS, 1, len(WORLDS)))
    by_mode = {}
    path = out / "forks.zip"
    with ZipFile(path, "x") as archive:
        for mode in MODES:
            start = perf_counter()
            data = run_forks(BIRTHS, WORLDS, mode, setup)
            raw = BytesIO()
            np.savez_compressed(raw, **data)
            payload = raw.getvalue()
            archive.writestr(f"{mode}.npz", payload)
            by_mode[mode] = estimate(data, bootstrap_birth, bootstrap_world)
            by_mode[mode]["raw_sha256"] = sha256(payload).hexdigest()
            print(mode, by_mode[mode]["primary"], round(perf_counter() - start, 2), flush=True)
    write(out / "summary.json", dict(registered_sha256=digest(out / "registered.json"),
                                     primary_development_sha256=digest(dev_path), raw_archive_sha256=digest(path),
                                     setup=setup, results=by_mode), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "confirm"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    args = parser.parse_args()
    {"register": register, "confirm": confirm}[args.stage](args.out, args.primary)


if __name__ == "__main__":
    main()
