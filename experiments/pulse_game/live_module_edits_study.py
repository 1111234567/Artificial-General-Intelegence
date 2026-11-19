"""Prospective paired single-life support editing pilot with modular birth checkpoints."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from . import live_module_edits as lab, module_graph_study as birth, module_graphs
from . import ring_logic_study as io, ring_logic_world as sensory, world


OUT = io.REPO / "data/pulse-game/live-module-edits-open-20261119"
TRIALS = 120
DEV_GROUPS = (0, 1, 2, 3)
DEV_ROOTS = tuple(range(68810000, 68810006))
TEST_GROUPS = (4, 5, 6, 7, 12, 13, 14, 15)
TEST_ROOTS = tuple(range(68830000, 68830016))
OPTIONS = (lab.Schedule(5, 7, .75), lab.Schedule(7, 5, .75), lab.Schedule(9, 6, .50))
ARMS = ("frozen", "feedback", "random_matched", "independent_sign")
CONDITIONS = ("stable", "reversal")
WINDOWS = ("before", "first20", "last20")
BOOTSTRAP = 20000


def code_hash():
    return {file: io.sha(Path(__file__).with_name(file)) for file in
            ("live_module_edits_study.py", "live_module_edits.py", "module_graphs.py", "free_graph_model.py",
             "module_graph_study.py", "ring_logic_study.py", "ring_logic_world.py", "world.py")}


def heads(groups):
    checkpoint_root = birth.BASE / "checkpoints"
    rows, supports, manifests = [], [], []
    arrays = {k: [] for k in birth.KEYS}
    arm = birth.ARMS.index("modules64")
    for group in groups:
        group_info = checkpoint_root / f"group-{group:02}.json"
        info = io.read(group_info)
        record = next(r for r in info["records"] if r["updates"] == 1200)
        path = checkpoint_root / record["file"]
        if io.sha(path) != record["sha256"]:
            raise RuntimeError("Changed birth weights")
        manifests.append({"group": group, "record_sha256": io.sha(group_info),
                          "weights": path.name, "weight_sha256": record["sha256"]})
        with np.load(path, allow_pickle=False) as fitted:
            for rule in range(len(sensory.RULES)):
                restart = record["selected"][rule][arm]
                lane = np.ravel_multi_index((restart, rule, arm), (2, len(sensory.RULES), len(birth.ARMS)))
                row = info["rows"][lane]
                if row[:2] != [group, restart] or row[3:] != [rule, "modules64"]:
                    raise RuntimeError("Unexpected fitted model order")
                source, target = module_graphs.graph_for(row[2], 64).T
                mask = np.zeros((lab.WIDTH, lab.WIDTH), dtype=bool)
                mask[target, source] = True
                if not lab.strongly_connected(lab.WIDTH, zip(source, target, strict=True)):
                    raise RuntimeError("Online protocol requires a strongly connected birth")
                supports.append(mask)
                rows.append(row)
                for key in arrays:
                    arrays[key].append(fitted[key][lane])
    return {k: np.stack(v) for k, v in arrays.items()}, np.stack(supports), rows, manifests


def tapes(roots, rows, *, reversal):
    """Build all teacher/scorer data externally. The circuit only sees one tick at a time."""
    base = world.make_tape(roots, trials=TRIALS, distance_jitter=.19, wind_scale=.10)
    shape = (len(roots), TRIALS)
    clock = sensory.TIMING["standard"][0] + world.TICKS + sensory.RECOVERY
    observations = np.empty((*shape, clock, sensory.SENSES), dtype=np.float32)
    blocks = np.empty(shape, dtype=np.int16)
    flips = np.empty(len(roots), dtype=np.int16)
    placebo = np.empty(shape, dtype=np.int8)
    for i, root in enumerate(roots):
        draw = np.random.SeedSequence([root, 80359012]).spawn(4)
        rng, order, when, signs = (np.random.default_rng(r) for r in draw)
        flips[i] = when.choice((40, 60))
        placebo[i] = signs.choice((-1, 1), size=TRIALS)
        cells = np.concatenate([order.permutation(len(sensory.PAIRS)) for _ in range(TRIALS // len(sensory.PAIRS))])
        blocks[i] = cells
        for t, cell in enumerate(cells):
            baseline, ticks, codes = sensory._request(rng, "standard")
            observations[i, t] = sensory._place(baseline, ticks, codes[sensory.PAIRS[cell]])
    lane = np.tile(np.arange(len(roots)), len(rows))
    rule = np.repeat([row[3] for row in rows], len(roots))
    right = sensory.truth()[rule[:, None], blocks[lane]]
    if reversal:
        right ^= np.arange(TRIALS)[None] >= flips[lane, None]
    left = base.left_id[lane]
    tape = replace(base, seeds=tuple(base.seeds[i] for i in lane), ctx=base.ctx[lane],
                   cue=base.cue[lane], left_id=left, good_id=np.where(right, 1 - left, left),
                   good_map=base.good_map[lane], distances=base.distances[lane], wind=base.wind[lane])
    return tape, observations, lane, blocks, flips, placebo


def receipt(observation):
    contacts = observation[:, world.CONTACT_ID]
    hit = contacts.sum(axis=1) > 0
    left = (contacts * observation[:, world.LEFT_ID]).sum(axis=1) > 0
    direction = np.where(hit, (~left).astype(np.int8), -1)
    taste = observation[:, world.REWARD_POS] - observation[:, world.REWARD_NEG]
    return direction, taste.astype(np.int8)


def run(head, roots, runs, schedule, arm, donor=None):
    parameters, masks, rows, _ = head
    tape, observations, lanes, _, _, placebo = runs
    n = len(roots)
    start = {name: np.repeat(values, n, axis=0) for name, values in parameters.items()}
    seeds = [np.random.SeedSequence([3882608, int(root), row[0], row[3]]) for row in rows for root in roots]
    brain = lab.Population(start, np.repeat(masks, n, axis=0), schedule, seeds)
    body = world.PulseWorld(tape, feedback="true")
    plastic = arm in ("feedback", "independent_sign")
    edits = np.full((tape.n, TRIALS, 2), -1, dtype=np.int16)
    sensed = np.zeros((tape.n, TRIALS), dtype=np.int8)
    reached = np.full_like(sensed, -1)
    sensory_ticks = sensory.TIMING["standard"][0]
    for trial in range(TRIALS):
        obs = observations[:, trial].transpose(1, 0, 2)
        for tick in range(sensory_ticks):
            brain.act(obs[tick, lanes])
        if plastic:
            brain.begin_request()
        peripheral = body.observe()
        for t in range(world.TICKS):
            side, taste = receipt(peripheral)
            got = side >= 0
            reached[got, trial] = side[got]
            sensed[got, trial] = taste[got]
            active = peripheral[:, world.ACTIVE] if plastic else None
            move = brain.act(obs[sensory_ticks + t, lanes], active)
            peripheral = body.step(np.column_stack((~move, move)))
        for tick in range(sensory_ticks + world.TICKS, len(obs)):
            brain.act(obs[tick, lanes])
        if plastic:
            shown = sensed[:, trial] if arm == "feedback" else placebo[lanes, trial]
            edits[:, trial] = brain.finish_request(reached[:, trial], shown)
        elif arm == "random_matched":
            edits[:, trial] = brain.random_edit(donor[:, trial])
    outcome = body.result()
    observed = outcome.observed_reward.sum(axis=-1, dtype=np.int8)
    if not np.array_equal(outcome.contact_side, reached) or not np.array_equal(observed, sensed):
        raise RuntimeError("Contact instrumentation differs from emitted public receipt")
    end = brain.mask
    shape = (len(rows) // len(sensory.RULES), len(sensory.RULES), n)
    return {"physical": np.stack((outcome.contact_side, outcome.contact_tick, sensed), axis=-1).reshape(*shape, TRIALS, 3),
            "edits": edits.reshape(*shape, TRIALS, 2), "final_mask": end.reshape(*shape, lab.WIDTH, lab.WIDTH)}


def windows(rewards, flips):
    correct = rewards[..., 2] > 0
    return {"before": np.stack([correct[..., i, flip - 20:flip].mean(axis=-1) for i, flip in enumerate(flips)], -1),
            "first20": np.stack([correct[..., i, flip:flip + 20].mean(axis=-1) for i, flip in enumerate(flips)], -1),
            "last20": correct[..., -20:].mean(axis=-1)}


def register(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "registered.json"
    if path.exists():
        raise RuntimeError("Already registered")
    io.write(path, dict(source_sha256=code_hash(), dev_groups=DEV_GROUPS, dev_roots=DEV_ROOTS,
                       test_groups=TEST_GROUPS, test_roots=TEST_ROOTS, trials=TRIALS,
                       worlds="All 20 cells appear in random order in each block; fresh sensory geometry and noise. Stable labels or unannounced all-side inversion at a world-seed draw of trial 40 or 60 (zero-based).",
                       heads="449-parameter 1200-step modules64 checkpoints, with the parent study's physical calibration choice within each starting group and rule. Runtime weights stay fixed.",
                       rule="Trace derivative of frozen motor logit w.r.t. an active neuron's local candidate input, times presynaptic activity and analogue change in the square-root degree normalization. Average over movement ticks before the own actual contact. Multiply approximate structural effect by the direction of physically reached side, the delivered signed taste and a logistic error factor. Separate EWMA scores for adding/removing an arc.",
                       edit="Independent growth and retirement intervals in signed receipt count. Positive estimated improvement required. Eligible growth candidates are absent, non-self arcs. Retire only if all 32 nodes remain strongly connected, starting the next request for newly grown arcs. No artificial count bounds, per-node degree targets, protected rings, or tied growth/deletion.",
                       controls="Frozen; matched-random attempts at exactly the realized feedback-arm change times with its counts, using randomized endpoints and the same connectivity guard; independent fair random signs instead of true taste at own actual contact. All four clones drive their own bodies and hold their own recurrent state throughout life.",
                       schedules=[asdict(s) for s in OPTIONS],
                       selection="Maximize reversal last-20 gain over frozen, with penalty 2*max(0, frozen minus adaptive last-20 accuracy in stable world), on development groups and worlds only. Compare with doing nothing, whose score is 0; ties favor doing nothing. Retain the highest-scoring active schedule as a separate mechanistic arm, with active ties settled by order. Confirm all arms on fresh matched worlds whose roots also differ from the earlier bounded pilot, with distinct fixed heads from development.",
                       confirmation="Primary pooled-rule contrasts adaptive minus each control in four prespecified windows: stable last 20, reversal before/first/last 20. Paired crossed bootstrap over fitted groups and episode roots with simultaneous 95% Bonferroni intervals over 12 cells. Each 20-trial block is balanced across all ordered odor pairs."))


def require_registered(out):
    if io.read(out / "registered.json")["source_sha256"] != code_hash():
        raise RuntimeError("Registered code changed")


def develop(out):
    require_registered(out)
    if (out / "recipe.json").exists():
        raise RuntimeError("Development is already sealed")
    head = heads(DEV_GROUPS)
    sets = {condition: tapes(DEV_ROOTS, head[2], reversal=(condition == "reversal"))
            for condition in CONDITIONS}
    baseline = {condition: windows(run(head, DEV_ROOTS, data, OPTIONS[0], "frozen")["physical"], data[-2])
                for condition, data in sets.items()}
    results = []
    for option in OPTIONS:
        reported = {}
        for condition, data in sets.items():
            result = run(head, DEV_ROOTS, data, option, "feedback")
            summary = windows(result["physical"], data[-2])
            reported[condition] = {key: {"adaptive": float(row.mean()), "frozen": float(baseline[condition][key].mean()),
                                        "difference": float((row - baseline[condition][key]).mean())}
                                   for key, row in summary.items()}
        stable = reported["stable"]["last20"]["difference"]
        reversal = reported["reversal"]["last20"]["difference"]
        results.append(dict(schedule=asdict(option), score=reversal - 2 * max(0, -stable), results=reported))
        print("development", results[-1], flush=True)
    io.write(out / "development.json", dict(heads=head[2], checkpoints=head[3], options=results))
    active = max(results, key=lambda row: row["score"])
    selected = "active" if active["score"] > 0 else "frozen"
    io.write(out / "recipe.json", dict(schedule=active["schedule"], deployment_choice=selected,
                                      frozen_score=0., active_score=active["score"],
                                      registered_sha256=io.sha(out / "registered.json"),
                                      development_sha256=io.sha(out / "development.json")))


def confirm(out):
    require_registered(out)
    recipe = io.read(out / "recipe.json")
    if (recipe["registered_sha256"] != io.sha(out / "registered.json") or
            recipe["development_sha256"] != io.sha(out / "development.json")):
        raise RuntimeError("Sealed selection was modified")
    head = heads(TEST_GROUPS)
    io.write(out / "confirmed-heads.json", dict(rows=head[2], checkpoints=head[3],
                                              recipe_sha256=io.sha(out / "recipe.json")))
    schedule = lab.Schedule(**recipe["schedule"])
    for condition in CONDITIONS:
        data = tapes(TEST_ROOTS, head[2], reversal=(condition == "reversal"))
        archive = {"pairs": data[3], "flip_trial": data[4], "initial_mask": head[1], "placebo_sign": data[5]}
        donor = None
        for arm in ARMS:
            begin = time.perf_counter()
            result = run(head, TEST_ROOTS, data, schedule, arm, donor)
            if arm == "feedback":
                donor = result["edits"].reshape(-1, TRIALS, 2) >= 0
            archive.update({f"{arm}/{key}": value for key, value in result.items()})
            print("confirm", condition, arm, round(time.perf_counter() - begin, 2), flush=True)
        temp = out / f"{condition}.writing.npz"
        np.savez_compressed(temp, **archive)
        temp.replace(out / f"{condition}.npz")


def validate_replay(base, outcome, record):
    if not np.array_equal(base["pairs"], outcome["pairs"]) or not np.array_equal(base["flip_trial"], outcome["flip_trial"]):
        raise RuntimeError("Stable and reversal arms must share stimulus and change-point draws")
    if not np.array_equal(base["frozen/physical"][..., :2], outcome["frozen/physical"][..., :2]):
        raise RuntimeError("Hidden labels affected frozen actions")
    if (not np.array_equal(base["independent_sign/physical"][..., :2], outcome["independent_sign/physical"][..., :2]) or
            not np.array_equal(base["independent_sign/edits"], outcome["independent_sign/edits"]) or
            not np.array_equal(base["independent_sign/final_mask"], outcome["independent_sign/final_mask"])):
        raise RuntimeError("Hidden labels affected an independent sign rollout")
    for root, flip in enumerate(base["flip_trial"]):
        for arm in ARMS:
            for key in ("physical", "edits"):
                if not np.array_equal(base[f"{arm}/{key}"][:, :, root, :flip],
                                      outcome[f"{arm}/{key}"][:, :, root, :flip]):
                    raise RuntimeError(f"Pre-change physical/structural histories differ in {arm}")
    return dict(frozen_contacts_equal_under_stable_and_reversal=True,
                independent_sign_contacts_and_edits_equal_under_stable_and_reversal=True,
                all_pre_change_traces_equal=True, global_birth_manifest_verified=verify_global_heads(record))


def verify_global_heads(record):
    path = birth.BASE / "frozen.json"
    local = [entry for split in ("dev", "heldout") for entry in record[split]]
    if not path.exists():
        return None
    global_groups = io.read(path)["groups"]
    for entry in local:
        group = entry["group"]
        record = birth.BASE / "checkpoints" / f"group-{group:02}.json"
        if entry["record_sha256"] != io.sha(record) or io.read(record) != global_groups[group]:
            raise RuntimeError("Pilot checkpoint choice differs from global modular freeze")
    return True


def report(out):
    require_registered(out)
    heads_info = io.read(out / "confirmed-heads.json")
    if heads_info["recipe_sha256"] != io.sha(out / "recipe.json"):
        raise RuntimeError("Scored heads are tied to a different development seal")
    rng = np.random.default_rng(3928881)
    group, n = len(TEST_GROUPS), len(TEST_ROOTS)
    bg = rng.multinomial(group, np.full(group, 1 / group), size=BOOTSTRAP) / group
    br = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP) / n
    tail = .05 / (2 * 12)
    report = {}
    threshold_keys = {"stable": ("last20",), "reversal": WINDOWS}
    for condition in CONDITIONS:
        with np.load(out / f"{condition}.npz", allow_pickle=False) as data:
            arms = {}
            values = {arm: windows(data[f"{arm}/physical"], data["flip_trial"]) for arm in ARMS}
            adaptive_events = data["feedback/edits"] >= 0
            random_events = data["random_matched/edits"] >= 0
            for arm in ARMS:
                events = data[f"{arm}/edits"] >= 0
                final = data[f"{arm}/final_mask"]
                for mask in final.reshape(-1, lab.WIDTH, lab.WIDTH):
                    if mask.trace() or not lab.strongly_connected(lab.WIDTH, zip(*np.nonzero(mask.T), strict=True)):
                        raise RuntimeError("Final graph violated online support constraints")
                counts = final.sum(axis=(-1, -2))
                initial = data["initial_mask"].reshape(group, len(sensory.RULES), 1, lab.WIDTH, lab.WIDTH)
                retained = (initial & final).sum(axis=(-1, -2))
                arms[arm] = dict(accuracy={key: float(value.mean()) for key, value in values[arm].items()},
                                 accuracy_by_rule={name: {key: float(value[:, j].mean()) for key, value in values[arm].items()}
                                                   for j, name in enumerate(sensory.RULES)},
                                 mean_grown=float(events[..., 0].sum(axis=-1).mean()),
                                 mean_retired=float(events[..., 1].sum(axis=-1).mean()),
                                 final_arcs=[int(counts.min()), float(counts.mean()), int(counts.max())],
                                 retained_birth_arcs=float(retained.mean()),
                                 timeout=float((data[f"{arm}/physical"][..., 0] < 0).mean()))
            contrasts = {}
            for control in ("frozen", "random_matched", "independent_sign"):
                result = {}
                for key in threshold_keys[condition]:
                    pooled = (values["feedback"][key] - values[control][key]).mean(axis=1)
                    distribution = ((bg @ pooled) * br).sum(axis=-1)
                    result[key] = {"delta": float(pooled.mean()), "simultaneous_95pct": np.quantile(distribution, (tail, 1-tail)).tolist(),
                                   "delta_by_group": pooled.mean(axis=1).tolist()}
                contrasts[f"feedback-{control}"] = result
            report[condition] = dict(arms=arms, contrasts=contrasts,
                                     matched_edits=int(np.count_nonzero(adaptive_events & random_events)),
                                     missed_match=int(np.count_nonzero(adaptive_events & ~random_events)),
                                     unexpected_random_edits=int(np.count_nonzero(~adaptive_events & random_events)))
    verification = {"dev": io.read(out / "development.json")["checkpoints"], "heldout": heads_info["checkpoints"]}
    with np.load(out / "stable.npz", allow_pickle=False) as base, np.load(out / "reversal.npz", allow_pickle=False) as outcome:
        replay = validate_replay(base, outcome, verification)
    io.write(out / "summary.json", dict(results=report, worlds_per_head=n, groups=TEST_GROUPS, validation=replay,
                                       schedule=io.read(out / "recipe.json")["schedule"],
                                       scores_sha256={case: io.sha(out / f"{case}.npz") for case in CONDITIONS}))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "develop", "confirm", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    action = {"register": register, "develop": develop, "confirm": confirm, "report": report}
    action[args.phase](args.out)


if __name__ == "__main__":
    main()
