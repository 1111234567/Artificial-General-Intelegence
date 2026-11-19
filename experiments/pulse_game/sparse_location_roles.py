"""Localize the effect of mature sparse dendrites on fresh physical food probes."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import graph_interactions as task, sparse_location_study as location, sparse_pairs_study
from .sparse_pairs_model import coordinates
from .unified_live_graph import MOTOR, SIZE
from .unified_live_graph_study import Body, digest, read, write


PROBE_OFFSET, PROBE_TRIALS = 2048, 128
BOOTSTRAP_SEED = 9234662
SENSORY_STOP = 8
RECEIVERS = ("left odor 0", "left odor 1", "left odor 2", "left odor 3",
             "right odor 0", "right odor 1", "right odor 2", "right odor 3",
             "left motor", "right motor", "positive taste", "negative taste",
             "left contact", "right contact", "internal 0", "internal 1")
ARMS = ("intact", "drop_motor", "motor_only", "drop_sensory_motor", "sensory_motor_only", "zero")
CONTRASTS = (("intact", "zero"), ("intact", "drop_motor"), ("motor_only", "zero"),
             ("intact", "motor_only"), ("drop_motor", "zero"), ("intact", "drop_sensory_motor"),
             ("sensory_motor_only", "zero"), ("intact", "sensory_motor_only"),
             ("drop_sensory_motor", "zero"), ("motor_only", "sensory_motor_only"))


def role_masks(records):
    _, receiver, first, second = coordinates(records["key"])
    motor = (receiver >= MOTOR.start) & (receiver < MOTOR.stop)
    sensory_motor = motor & (first < SENSORY_STOP) & (second < SENSORY_STOP)
    return receiver, motor, sensory_motor


def fresh_menu(roots, mode, *, offset=PROBE_OFFSET, count=PROBE_TRIALS):
    original, _, _, _ = task.make_menu(roots, mode, location.TRIALS)
    extended, change, _, _ = task.make_menu(roots, mode, offset + count)
    seen, fresh = slice(None, location.TRIALS), slice(offset, offset + count)
    if offset < location.TRIALS or offset <= change.max():
        raise ValueError("Probe offset must follow the historical tape and the entire reversal window")
    np.testing.assert_array_equal(extended.chemicals[:, seen], original.chemicals)
    np.testing.assert_array_equal(extended.weights, original.weights)
    # Check the uniqueness of the 4-component float32 foods for each distinct world root.
    first_rows = np.unique(roots, return_index=True)[1]
    seen_food = np.ascontiguousarray(extended.chemicals[first_rows, seen]).view("V16").ravel()
    new_food = np.ascontiguousarray(extended.chemicals[first_rows, fresh]).view("V16").ravel()
    if np.intersect1d(seen_food, new_food).size or len(np.unique(new_food)) != len(new_food):
        raise RuntimeError("Probe foods must be disjoint from the historical tape and each other")
    time_fields = ("ctx", "cue", "left_id", "good_id", "distances", "wind")
    tape = replace(extended.body, **{name: getattr(extended.body, name)[:, fresh] for name in time_fields})
    return replace(extended, body=tape, chemicals=extended.chemicals[:, fresh],
                   probability=extended.probability[:, fresh], realized=extended.realized[:, fresh], counter=0), len(new_food)


def score(data, mode, setup):
    birth, roots = data["birth"], data["world"]
    graph = sparse_pairs_study.make_graph(birth, roots, "guided", setup)
    for name in ("mask", "weight", "state", "rate", "bias"):
        setattr(graph, name, data[f"mature_{name}"].copy())
    active = data["pair_intact"]
    graph.set_active(active)
    receiver, motor, sensory_motor = role_masks(active)
    menu, distinct = fresh_menu(roots, mode)
    body = Body(menu, None, PROBE_TRIALS)
    obs = body.observe()
    keep = dict(intact=np.ones(len(active), dtype=bool), drop_motor=~motor, motor_only=motor,
                drop_sensory_motor=~sensory_motor, sensory_motor_only=sensory_motor,
                zero=np.zeros(len(active), dtype=bool))
    raw = dict(birth=birth, world=roots, good_side=menu.probability.argmax(axis=-1),
               chemical=menu.chemicals, tape_distances=menu.body.distances, tape_wind=menu.body.wind,
               pair=active, pair_receiver=receiver, pair_receiver_motor=motor, pair_sensory_motor=sensory_motor)
    for name, mask in keep.items():
        intervened = active.copy()
        intervened["weight"][~mask] = 0
        result = location.score_fork(graph, body, obs, intervened, PROBE_TRIALS)
        for key, value in result.items():
            raw[f"{name}/{key}"] = value
    return raw, distinct


def source_hashes():
    return {**location.source_hashes(), Path(__file__).name: digest(Path(__file__))}


def register(out):
    origin = read(out / "summary.json")
    if digest(out / "forks.zip") != origin["raw_archive_sha256"]:
        raise RuntimeError("Historical snapshot archive does not match its summary")
    write(out / "roles-registered.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hashes(),
        snapshot_archive_sha256=digest(out / "forks.zip"), snapshot_summary_sha256=digest(out / "summary.json"),
        modes=location.MODES, arms=ARMS, contrasts=CONTRASTS,
        probes=dict(offset_zero_based=PROBE_OFFSET, trials=PROBE_TRIALS, stable_or_final_reverse_phase=True),
        receivers=RECEIVERS, sensory_vertices=list(range(SENSORY_STOP)), motor_vertices=list(range(MOTOR.start, MOTOR.stop)),
        snapshot="Restore the packed Q and original W, topology, node rates, bias, and recurrent state saved at maturity. Present the continuous foods and physical tape at a suffix disjoint from the historical 1,280 trials, with the same private taste rule at its final sign. All arms see the same candidate foods and tape without exploration or publicly delivered tastes. Freeze updates. Zero coefficients without changing allocated locations or W.",
        isolates="A motor coefficient has receiver 8 or 9. A sensory-to-motor coefficient also has both distinct source vertices among the eight odor input ports. Disable only each set, and in complementary arms keep only that set. Compare with intact and all-Q-zero.",
        primary="Physical best-expected-food first-contact accuracy. Timeout counts as incorrect. Match arms within life and cross resample its birth and world roots for the 20 familywise 95% two-sided contrasts. No selection or early stopping.",
        secondary="For each arm report raw left/right/timeout totals, fraction choosing a different first-contact side from intact, correct totals, and accuracy among contacts. For each receiver and disjoint sender/receiver partition report installed/nonzero counts, sign and absolute weight mass.",
        bootstrap=dict(seed=BOOTSTRAP_SEED, draws=location.BOOTSTRAP_DRAWS),
    ), exclusive=True)


def mass(weights):
    return dict(installed=len(weights), nonzero=int(np.count_nonzero(weights)),
                positive=int(np.count_nonzero(weights > 0)), negative=int(np.count_nonzero(weights < 0)),
                signed_sum=float(weights.sum(dtype=np.float64)), absolute_sum=float(np.abs(weights).sum(dtype=np.float64)),
                mean_absolute=float(np.abs(weights).mean(dtype=np.float64)) if len(weights) else None)


def estimate(data, draw_b, draw_w):
    means, per_life = {}, {}
    intact = data["intact/contact"]
    for arm in ARMS:
        action = data[f"{arm}/contact"]
        contacted = action >= 0
        correct = (action == data["good_side"]) & contacted
        per_life[arm] = correct.mean(axis=1)
        n_contact = int(contacted.sum())
        means[arm] = dict(accuracy=float(correct.mean()), trials=action.size, correct=int(correct.sum()),
                          contacted=n_contact, left=int((action == 0).sum()), right=int((action == 1).sum()),
                          timeout=int((~contacted).sum()), changed_side_or_timeout=int((action != intact).sum()),
                          accuracy_among_contacts=int(correct.sum()) / n_contact if n_contact else None)
    tail = .025 / (len(CONTRASTS) * len(location.MODES))
    intervals = {}
    for first, second in CONTRASTS:
        diff = (per_life[first] - per_life[second]).reshape(len(location.BIRTHS), len(location.WORLDS))
        samples = diff[draw_b, draw_w].mean(axis=(1, 2))
        intervals[f"{first}_minus_{second}"] = [float(diff.mean()), *np.quantile(samples, [tail, 1 - tail]).tolist()]
    weights = data["pair"]["weight"]
    motor, direct = data["pair_receiver_motor"], data["pair_sensory_motor"]
    receiver = data["pair_receiver"]
    return dict(means=means, contrasts=intervals, structure=dict(
        total=mass(weights), all_motor=mass(weights[motor]), sensory_motor=mass(weights[direct]),
        other_motor=mass(weights[motor & ~direct]), nonmotor=mass(weights[~motor]),
        by_receiver=[dict(index=i, role=RECEIVERS[i], **mass(weights[receiver == i])) for i in range(SIZE)]))


def confirm(out):
    protocol = read(out / "roles-registered.json")
    historical = read(out / "summary.json")
    archive_path = out / "forks.zip"
    if (protocol["source_sha256"] != source_hashes() or
            protocol["snapshot_archive_sha256"] != digest(archive_path) or
            protocol["snapshot_summary_sha256"] != digest(out / "summary.json")):
        raise RuntimeError("Registered source or mature snapshot changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    b = len(location.BIRTHS)
    w = len(location.WORLDS)
    draw_b = rng.integers(b, size=(location.BOOTSTRAP_DRAWS, b, 1))
    draw_w = rng.integers(w, size=(location.BOOTSTRAP_DRAWS, 1, w))
    results = {}
    raw_path = out / "roles-probes.zip"
    with ZipFile(archive_path) as snapshots, ZipFile(raw_path, "x") as probes:
        for mode in location.MODES:
            raw = snapshots.read(f"{mode}.npz")
            if sha256(raw).hexdigest() != historical["results"][mode]["raw_sha256"]:
                raise RuntimeError("Historical snapshot entry changed")
            with np.load(BytesIO(raw), allow_pickle=False) as saved:
                data, unique_foods = score(saved, mode, historical["setup"])
            results[mode] = estimate(data, draw_b, draw_w)
            results[mode]["distinct_probe_foods"] = unique_foods
            target = BytesIO()
            np.savez_compressed(target, **data)
            payload = target.getvalue()
            probes.writestr(f"{mode}.npz", payload)
            results[mode]["raw_sha256"] = sha256(payload).hexdigest()
            print(mode, results[mode]["contrasts"], flush=True)
    write(out / "roles-summary.json", dict(registered_sha256=digest(out / "roles-registered.json"),
                                           raw_archive_sha256=digest(raw_path), results=results), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "confirm"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"register": register, "confirm": confirm}[args.stage](args.out)


if __name__ == "__main__":
    main()
