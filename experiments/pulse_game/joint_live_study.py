"""Prospective factorial of receipt-trained weights and uncapped edits in continuous lives."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from . import joint_live_model as lab, live_module_edits_study as original
from . import ring_logic_study as io, ring_logic_world as sensory, world


OUT = io.REPO / "data/pulse-game/joint-live-20261119"
EVIDENCE = io.REPO / "docs/evidence/free-graph-2026-11-19/joint"
DEV_GROUPS = original.DEV_GROUPS
TEST_GROUPS = original.TEST_GROUPS
DEV_ROOTS = tuple(range(69021000, 69021006))
TEST_ROOTS = tuple(range(69023000, 69023016))
TRIALS = original.TRIALS
SCHEDULE = original.lab.Schedule(9, 6, .50)
RATES = (.015, .06, .24, .96)
ARMS = ("neither", "weight_only", "edge_only", "joint", "joint_pin_newborn", "joint_random", "joint_independent_sign")
FACTORIAL = {"weights_alone": ("weight_only", "neither"), "edits_alone": ("edge_only", "neither"),
             "edits_given_weights": ("joint", "weight_only"), "weights_given_edits": ("joint", "edge_only"),
             "joint_minus_neither": ("joint", "neither")}
CONTROLS = {"newborn_weight_writes": ("joint", "joint_pin_newborn"), "selected_vs_random_edges": ("joint", "joint_random"),
            "actual_vs_placebo_taste": ("joint", "joint_independent_sign")}
WINDOWS = {"stable": ("last20",), "reversal": ("before", "first20", "last20"), "return": ("after_return",)}


def code_hash():
    return original.code_hash() | {Path(__file__).name: io.sha(Path(__file__)),
                                   "joint_live_model.py": io.sha(Path(lab.__file__))}


def register(out):
    path = out / "registered.json"
    if path.exists():
        raise RuntimeError("Already registered")
    legacy_recipe = io.REPO / "docs/evidence/free-graph-2026-11-19/live/recipe.json"
    if asdict(SCHEDULE) != io.read(legacy_recipe)["schedule"]:
        raise RuntimeError("The structural recipe differs from the sealed prior comparator")
    if set(DEV_GROUPS) & set(TEST_GROUPS) or set(DEV_ROOTS + TEST_ROOTS) & set(original.DEV_ROOTS + original.TEST_ROOTS):
        raise RuntimeError("Overlapping experiment partitions")
    io.write(path, dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_sha256=code_hash(),
        dev_groups=DEV_GROUPS, dev_roots=DEV_ROOTS, test_groups=TEST_GROUPS, test_roots=TEST_ROOTS,
        trials=TRIALS, schedule=asdict(SCHEDULE), legacy_recipe_sha256=io.sha(legacy_recipe), arms=ARMS,
        conditions=dict(stable="A throughout", reversal="A to B at index 40 or 60, no exposed change point",
                        return_to_A="A to B at the same index, then to A after 40 requests in B"),
        weight_rule="Signed logistic ascent of mean pre-contact motor logit, assigned by cue-independent rule: sign equals actual contact side times delivered taste. Exact one-request BPTT through cue and movement with fixed carried prior. Shared gradient global norm clip at 4, SGD. Updates all live raw recurrent entries independently, all candidate/gate input weights and biases, and motor readout/offset. No writes on missed contact. Apply to existing active entries before new edits; growth init is the checkpoint's original signed incoming gain at the receiving node and first learns on subsequent receipts.",
        structure="Original uncapped receipt-clock schedule, preference EWMA and strongly-connected retirements, generalized by current effective drive and independent raw coefficient being considered for removal. Remove/grow independently. Every effective coefficient divides by the current receiving indegree square root. No tied weights remaining after birth initialization.",
        rate_candidates=RATES, selection="Select active rate on the weight-only dev arm using pooled last20 reversal gain minus twice any stable last20 loss, compared to frozen. Report no-write alternative score 0 separately. Reuse exact selected weight rule and rate in every weight-plastic arm. No selection on confirmation.",
        controls="Whole-life newborn-raw pin still changes all other active coefficients, and recomputes indegree normally. Matched random arm gets just the contemporaneous donor joint arm's accepted growth/retirement opportunities; it uses own true receipt for weights. Independent-sign arm uses independently drawn fair sign on own actual contact in place of taste for both update mechanisms.",
        analysis="Physical first-contact accuracy with timeout as incorrect; 20,000 crossed paired bootstrap replicates over independent fitted groups and world roots, pooled fixed rules. Factorial contrasts plus difference of simple effects and placebo contrasts each use their own simultaneous 95% Bonferroni family over chosen windows. Mark return window single new third-condition endpoint.",
        windows=WINDOWS, statistics=dict(factorial=FACTORIAL, control=CONTROLS, interaction="joint - weight_only - edge_only + neither"),
        newborn_fields=lab.BIRTH_FIELDS))


def require_registered(out):
    if io.read(out / "registered.json")["source_sha256"] != code_hash():
        raise RuntimeError("Registered code changed")


def tapes(roots, rows, condition):
    data = original.tapes(roots, rows, reversal=condition != "stable")
    if condition == "return":
        tape, obs, lanes, blocks, flip, placebo = data
        restored = np.arange(TRIALS)[None, :] >= flip[lanes, None] + 40
        return replace(tape, good_id=np.where(restored, 1 - tape.good_id, tape.good_id)), obs, lanes, blocks, flip, placebo
    return data


def run(head, roots, data, schedule, arm, learning_rate, donor=None, *, population=lab.Population):
    parameters, masks, rows, _ = head
    tape, observations, lanes, _, _, placebo = data
    n = len(roots)
    start = {name: np.repeat(values, n, axis=0) for name, values in parameters.items()}
    seeds = [np.random.SeedSequence([3882608, int(root), row[0], row[3]]) for row in rows for root in roots]
    weight = arm in ("weight_only", "joint", "joint_pin_newborn", "joint_random", "joint_independent_sign")
    edges = arm in ("edge_only", "joint", "joint_pin_newborn", "joint_independent_sign")
    brain = population(start, np.repeat(masks, n, axis=0), schedule, seeds,
                       learning_rate=learning_rate if weight else 0., pin_newborn=arm == "joint_pin_newborn")
    body = world.PulseWorld(tape, feedback="true")
    edits = np.full((tape.n, TRIALS, 2), -1, dtype=np.int16)
    records = np.full((tape.n, TRIALS, 2, len(lab.BIRTH_FIELDS)), np.nan, dtype=np.float32)
    steps = np.zeros((tape.n, TRIALS, len(lab.WEIGHT_NAMES)), dtype=np.float32)
    input_steps = np.zeros((tape.n, TRIALS, sensory.SENSES), dtype=np.float32)
    signed = np.zeros((tape.n, TRIALS), dtype=np.int8)
    reached = np.full_like(signed, -1)
    sensory_ticks = sensory.TIMING["standard"][0]
    for trial in range(TRIALS):
        brain.begin_request(learn_edges=edges)
        obs = observations[:, trial].transpose(1, 0, 2)
        for tick in range(sensory_ticks):
            brain.act(obs[tick, lanes], record=weight)
        peripheral = body.observe()
        for t in range(world.TICKS):
            side, taste = original.receipt(peripheral)
            hit = side >= 0
            reached[hit, trial] = side[hit]
            signed[hit, trial] = taste[hit]
            motor_active = peripheral[:, world.ACTIVE] if weight or edges else None
            move = brain.act(obs[sensory_ticks + t, lanes], motor_active, record=weight)
            peripheral = body.step(np.column_stack((~move, move)))
        for tick in range(sensory_ticks + world.TICKS, len(obs)):
            brain.act(obs[tick, lanes])
        if weight or edges:
            seen = placebo[lanes, trial] if arm == "joint_independent_sign" else signed[:, trial]
            events = donor[:, trial] if donor is not None else None
            edits[:, trial], records[:, trial] = brain.finish_request(reached[:, trial], seen, donor=events)
            steps[:, trial] = brain.weight_step
            input_steps[:, trial] = brain.input_step
    outcome = body.result()
    if not np.array_equal(outcome.contact_side, reached) or not np.array_equal(outcome.observed_reward.sum(axis=-1), signed):
        raise RuntimeError("Scoring differs from actual public receipts")
    end_names = ("raw", "state", "mask", "newborn_trial", "newborn_raw")
    final = {f"final_{name}": getattr(brain, name) for name in end_names}
    final |= {f"final_{name}": brain.parameters[name] for name in lab.WEIGHT_NAMES[1:]}
    active = np.nonzero(brain.newborn_trial >= 0)
    terminal_newborn = np.full((*brain.mask.shape, len(lab.BIRTH_FIELDS)), np.nan, dtype=np.float32)
    terminal_newborn[active] = brain.newborn_records(*active, brain.degree[active[0], active[1]])
    final["final_newborn_records"] = terminal_newborn
    payload = dict(physical=np.stack((outcome.contact_side, outcome.contact_tick, signed), axis=-1), edits=edits,
                   event_records=records, weight_step=steps, input_step=input_steps, **final)
    shape = (len(rows) // len(sensory.RULES), len(sensory.RULES), n)
    return {key: array.reshape(*shape, *array.shape[1:]) for key, array in payload.items()}


def windows(physical, flips):
    return original.windows(physical, flips) | {"after_return": (physical[..., -20:, 2] > 0).mean(axis=-1)}


def develop(out):
    require_registered(out)
    if (out / "recipe.json").exists():
        raise RuntimeError("Development already sealed")
    head = original.heads(DEV_GROUPS)
    sets = {case: tapes(DEV_ROOTS, head[2], case) for case in ("stable", "reversal")}

    def accuracy(case, rate):
        data = sets[case]
        physical = run(head, DEV_ROOTS, data, SCHEDULE, "weight_only" if rate else "neither", rate)["physical"]
        return float(windows(physical, data[-2])["last20"].mean())

    baseline = {case: accuracy(case, 0.) for case in sets}

    def evaluate(rate):
        fitted = {case: accuracy(case, rate) for case in sets}
        delta = {case: fitted[case] - baseline[case] for case in sets}
        result = dict(rate=rate, accuracy=fitted, delta=delta, score=delta["reversal"] - 2 * max(0, -delta["stable"]))
        print("development", result, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        candidates = list(pool.map(evaluate, RATES))
    best = max(candidates, key=lambda row: row["score"])
    io.write(out / "development.json", dict(checkpoints=head[3], rows=head[2], baseline=baseline, candidates=candidates))
    io.write(out / "recipe.json", dict(rate=best["rate"], score=best["score"], deploy="weight_rule" if best["score"] > 0 else "no_writes",
        registered_sha256=io.sha(out / "registered.json"), development_sha256=io.sha(out / "development.json")))


def sealed_recipe(out):
    require_registered(out)
    recipe = io.read(out / "recipe.json")
    for stage in ("registered", "development"):
        if recipe[f"{stage}_sha256"] != io.sha(out / f"{stage}.json"):
            raise RuntimeError("Development seal mismatch")
    return recipe


def confirm(out):
    recipe = sealed_recipe(out)
    head = original.heads(TEST_GROUPS)
    io.write(out / "confirmed-heads.json", dict(rows=head[2], checkpoints=head[3], recipe_sha256=io.sha(out / "recipe.json")))

    def condition(case):
        path = out / f"{case}.npz"
        if path.exists():
            raise RuntimeError(f"Cannot overwrite scored {case}")
        data = tapes(TEST_ROOTS, head[2], case)
        archive = dict(pairs=data[3], flip_trial=data[4], initial_mask=head[1],
                       initial_raw=np.where(head[1], head[0]["gain"][..., None], np.float32(0)), placebo_sign=data[5])
        donor = None
        for arm in ARMS:
            start = time.perf_counter()
            matched = donor if arm == "joint_random" else None
            values = run(head, TEST_ROOTS, data, SCHEDULE, arm, recipe["rate"], donor=matched)
            if arm == "joint":
                donor = values["edits"].reshape(-1, TRIALS, 2) >= 0
            archive.update({f"{arm}/{key}": value for key, value in values.items()})
            print("confirm", case, arm, f"{time.perf_counter()-start:.1f}s", flush=True)
        for arm, reference in (("neither", "frozen"), ("edge_only", "feedback")):
            prior = original.run(head, TEST_ROOTS, data, SCHEDULE, reference)
            for key, expected in prior.items():
                np.testing.assert_array_equal(archive[f"{arm}/{key}"], expected)
        temp = out / f"{case}.partial.npz"
        np.savez_compressed(temp, **archive)
        temp.replace(path)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(condition, WINDOWS))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out)


if __name__ == "__main__":
    main()
