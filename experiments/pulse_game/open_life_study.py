"""Register and evaluate receipt-trained contextual adaptation in an open graph."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from . import joint_live_integrated as integrated, joint_live_study as joint
from . import live_module_edits_study as original, open_life_model as model
from . import ring_logic_study as io, ring_logic_world as sensory


OUT = io.REPO / "data/pulse-game/open-life-20261119"
EVIDENCE = io.REPO / "docs/evidence/open-graph-learning-2026-11-19/learner"
DEV_GROUPS = (0, 1, 2, 3)
VAL_GROUPS = (8, 9, 10, 11)
TEST_GROUPS = original.TEST_GROUPS
DEV_ROOTS = tuple(range(69541000, 69541008))
VAL_ROOTS = tuple(range(69542000, 69542008))
TEST_ROOTS = tuple(range(69543000, 69543020))
CONDITIONS = ("stable", "reversal", "return", "partial")
ARMS = ("adaptive", "legacy", "effective", "head_only", "placebo")
LEGACY_RATE = .06
BOOTSTRAP = 20_000
FIXED = dict(birth_scale=4., memory_prior=.055)
CANDIDATES = (
    # slow-rate, head-rate, context-gain, memory decay, kernel strength, retirement cost, temperature
    (.008, .24, .95, .90, 24., .035, 1.4),
    (.008, .42, .95, .90, 24., .035, 1.4),
    (.008, .42, 1.45, .90, 24., .035, 1.4),
    (.008, .24, 1.45, .86, 18., .035, 1.4),
    (.008, .42, 1.45, .86, 18., .035, 1.4),
    (.008, .28, 1.75, .94, 24., .065, 1.4),
    (.008, .28, 2.25, .92, 32., .065, 1.4),
    (.020, .32, 1.30, .90, 24., .065, 1.4),
    (.008, .32, 1.30, .90, 24., .085, 1.4),
    (.008, .42, 1.75, .94, 18., .085, 2.2),
    (.025, .42, 1.75, .94, 18., .085, 2.2),
    (.008, .32, 1.75, .87, 16., .085, 2.2),
)
TUNED = ("rate", "head_rate", "context_gain", "memory_decay", "kernel_strength", "retirement_cost", "temperature")


def code_hash():
    return joint.code_hash() | {name: io.sha(Path(__file__).with_name(name)) for name in (
        Path(__file__).name, "open_life_model.py", "coupled_birth.py", "joint_live_integrated.py")}


def register(out):
    path = out / "registered.json"
    if path.exists():
        raise RuntimeError("Registration already exists")
    check = io.read(integrated.OUT / "recipe.json")
    if check["rate"] != LEGACY_RATE:
        raise RuntimeError("The selected prior joint rate does not match")
    groups = (DEV_GROUPS, VAL_GROUPS, TEST_GROUPS)
    roots = (DEV_ROOTS, VAL_ROOTS, TEST_ROOTS)
    for family in (groups, roots):
        if sum(map(len, family)) != len(set.union(*map(set, family))):
            raise RuntimeError("Experiment partitions overlap")
    old = original.DEV_ROOTS + original.TEST_ROOTS + joint.DEV_ROOTS + joint.TEST_ROOTS + integrated.TEST_ROOTS
    if set.union(*map(set, roots)) & set(old):
        raise RuntimeError("Registered worlds must be new")
    io.write(path, dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_sha256=code_hash(), legacy_recipe_sha256=io.sha(integrated.OUT / "recipe.json"),
        groups=groups, world_roots=roots, arms=ARMS, conditions=CONDITIONS, trials=original.TRIALS,
        schedule=asdict(joint.SCHEDULE), candidates=[config(values) for values in CANDIDATES],
        learner="A continuous strongly connected modules64 graph. Start from exactly the same normalized incumbent. Speed of motor-prediction regression and local residual kernel fitted on mixed development lives. Novel connections take a signed sensitivity velocity step on birth, are individually trainable on every subsequent actual receipt, and do not rescale incumbent arcs. Retire any previously existing arc whose receipt-averaged predicted improvement offsets a weight-squared cost and preserves strong connectivity.",
        receipt="Correct-side labels are reconstructed only at first contact from own physical side times delivered taste. Placebo swaps only the delivered sign for an independent externally drawn fair sign at that clone's actual contact; timeout keeps the absence of learning in every arm. Every model drives its own body.",
        context="Regress own bounded motor prediction against the reconstructed label. Learn the residual with a finite evolving kernel over the recurrent innovations at both public marked pulses. Own earlier receipts update kernel targets and decay weights. None of rule, truth table, cell, trial index, or switch phase enters the learner. Marker threshold and ordered encoding are architectural priors.",
        arms_notes={"adaptive": "All mechanisms active.", "legacy": "Exact previously integrated active joint arm with rate .06 and original structural implementation.",
                    "head_only": "All graph mechanisms unchanged, contextual term inactive; scalar regression still active.",
                    "effective": "Same independent effective update graph and retirement cost; fast terms inactive.",
                    "placebo": "Same graph and fast algorithm, fresh fair sign at each own physical contact."},
        worlds="120 requests. Stable A throughout. Full reversal A to inverted A at a root-chosen request 40 or 60. Return to A another 40 requests later. Partial: root/rule-selected equal subsets of previously right and left cases reverse at 40 or 60 in fitting; 10/20 changed cases in development, 12/20 validation, 8/20 confirmation, with held-out confirmation timing 47, 55 or 63. All roots and starting fitted groups for the three stages are disjoint.",
        selection="Maximize equally weighted stable late, full reversal late, return late, and partial late. Partial criterion itself equally weights the changed and unchanged cases. Ties use candidate order. Seal winning recipe on development jointly across varied lives. Validation runs sealed recipe once; it is descriptive and does not change any selection or define exclusion. Confirm with same recipe in held-out groups and roots.",
        primary="Evaluate stable last20, reversal last20, return last20, and partial last40 (balanced in changed and unchanged strata). Stable retention requires a positive simultaneous lower bound against legacy. Learning reversal requires a positive simultaneous lower bound versus .5 chance. Successful return requires a positive simultaneous lower bound versus legacy. Conditional adaptation requires positive simultaneous lower bound versus head-only on changed cells, with unchanged also reported. Also report within-world paired primary improvement against legacy and placebo. 20,000 paired crossed bootstrap samples by starting group and root; fixed rules pooled equally. All 25 prespecified contrasts (four arm contrasts and chance for each of five snapshots) use one simultaneous family across declared comparisons; no post hoc exclusions.",
        fast_fields=model.FAST_FIELDS))


def config(values):
    return dict(zip(TUNED, values, strict=True)) | FIXED


def require_registered(out):
    manifest = io.read(out / "registered.json")
    if manifest["source_sha256"] != code_hash():
        raise RuntimeError("Registered source changed")
    if manifest["legacy_recipe_sha256"] != io.sha(integrated.OUT / "recipe.json"):
        raise RuntimeError("The legacy recipe changed")


def tapes(roots, rows, category, stage):
    if category != "partial":
        data = joint.tapes(roots, rows, category)
        return data, np.zeros((len(sensory.RULES), len(roots), original.TRIALS), dtype=bool)
    base, observations, lanes, blocks, _, placebo = original.tapes(roots, rows, reversal=False)
    size, times = {
        "development": (10, (40, 60)),
        "validation": (12, (40, 60)),
        "confirmation": (8, (47, 55, 63)),
    }[stage]
    changes = np.zeros((len(sensory.RULES), len(roots), original.TRIALS), dtype=bool)
    flips = np.empty(len(roots), dtype=np.int16)
    for root_i, root in enumerate(roots):
        rng = np.random.default_rng(np.random.SeedSequence([root, 4093304]))
        flips[root_i] = rng.choice(times)
        for rule, pattern in enumerate(sensory.truth()):
            switched = np.concatenate([rng.choice(np.flatnonzero(pattern == side), size // 2, replace=False)
                                       for side in (False, True)])
            changes[rule, root_i] = np.isin(blocks[root_i], switched) & (np.arange(original.TRIALS) >= flips[root_i])
    rules = np.repeat([row[3] for row in rows], len(roots))
    target = np.where(changes[rules, lanes], 1 - base.good_id, base.good_id)
    return (replace(base, good_id=target), observations, lanes, blocks, flips, placebo), changes


def run(head, roots, data, arm, config):
    """Accept an external physical tape and return all native outcome and edit traces."""
    if arm == "legacy":
        return joint.run(head, roots, data, joint.SCHEDULE, "joint", LEGACY_RATE)
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}")
    instances = []

    def make(*args, **kwargs):
        selected = config
        if arm == "effective":
            selected = config | {"head_rate": 0., "context_gain": 0.}
        brain = model.Population(*args, **kwargs, config=selected, contextual=arm not in ("head_only", "effective"))
        instances.append(brain)
        return brain

    fast = []

    def track(*args, **kwargs):
        brain = make(*args, **kwargs)
        finish = brain.finish_request

        def finish_recorded(*receipt, **options):
            result = finish(*receipt, **options)
            fast.append(brain.fast_step.copy())
            return result

        brain.finish_request = finish_recorded
        return brain

    kind = "joint_independent_sign" if arm == "placebo" else "joint"
    result = joint.run(head, roots, data, joint.SCHEDULE, kind, config["rate"], population=track)
    shape = result["physical"].shape[:-1]
    result["fast_step"] = np.stack(fast, axis=1).reshape(*shape, len(model.FAST_FIELDS))
    result["final_head"] = instances[0].head.reshape(*shape[:-1])
    return result


def views(physical, condition, changes):
    correct = physical[..., 2] > 0
    if condition != "partial":
        return {"late": correct[..., -20:].mean(axis=-1)}
    changed = changes[..., -40:]
    result = {}
    for name, mask in (("changed", changed), ("unchanged", ~changed)):
        result[name] = (correct[..., -40:] * mask).sum(axis=-1) / mask.sum(axis=-1)
    return result


def data_sets(roots, head, stage):
    return {case: tapes(roots, head[2], case, stage) for case in CONDITIONS}


def evaluate(head, roots, sets, recipe):
    accuracy = {}
    for case, (data, change) in sets.items():
        result = run(head, roots, data, "adaptive", recipe)
        accuracy[case] = {key: float(cell.mean()) for key, cell in views(result["physical"], case, change).items()}
    score = float(np.mean([np.mean(list(value.values())) for value in accuracy.values()]))
    return dict(config=recipe, accuracy=accuracy, score=score)


def develop(out):
    require_registered(out)
    if (out / "recipe.json").exists() or (out / "development.json").exists():
        raise RuntimeError("Development already sealed")
    head = original.heads(DEV_GROUPS)
    sets = data_sets(DEV_ROOTS, head, "development")

    def simulate(values):
        result = evaluate(head, DEV_ROOTS, sets, config(values))
        print("development", result, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(simulate, CANDIDATES))
    io.write(out / "development.json", dict(heads=head[2], checkpoints=head[3], results=results))
    winner = max(results, key=lambda result: result["score"])
    io.write(out / "recipe.json", winner | dict(registered_sha256=io.sha(out / "registered.json"),
                                                development_sha256=io.sha(out / "development.json")))


def sealed_recipe(out):
    require_registered(out)
    recipe = io.read(out / "recipe.json")
    for stage in ("registered", "development"):
        if recipe[f"{stage}_sha256"] != io.sha(out / f"{stage}.json"):
            raise RuntimeError("An upstream stage was modified")
    return recipe["config"]


def validate(out):
    recipe = sealed_recipe(out)
    if (out / "validation.json").exists():
        raise RuntimeError("Validation already sealed")
    head = original.heads(VAL_GROUPS)
    sets = data_sets(VAL_ROOTS, head, "validation")
    result = evaluate(head, VAL_ROOTS, sets, recipe)
    io.write(out / "validation.json", result | dict(heads=head[2], checkpoints=head[3], recipe_sha256=io.sha(out / "recipe.json")))
    print("validation", result, flush=True)


def confirm(out):
    recipe = sealed_recipe(out)
    if io.read(out / "validation.json")["recipe_sha256"] != io.sha(out / "recipe.json"):
        raise RuntimeError("Validation ran a different recipe")
    head = original.heads(TEST_GROUPS)
    heads_path = out / "confirmed-heads.json"
    heads = dict(rows=head[2], checkpoints=head[3], recipe_sha256=io.sha(out / "recipe.json"),
                 validation_sha256=io.sha(out / "validation.json"))
    if heads_path.exists() and io.read(heads_path) != heads:
        raise RuntimeError("Confirmation was initialized with different heads")
    if not heads_path.exists():
        io.write(heads_path, heads)

    def condition(case):
        path = out / f"{case}.npz"
        if path.exists():
            return
        data, change = tapes(TEST_ROOTS, head[2], case, "confirmation")
        archive = dict(pairs=data[3], flip_trial=data[4], initial_mask=head[1], changed=change,
                       initial_raw=np.where(head[1], head[0]["gain"][..., None], np.float32(0)), placebo_sign=data[5])
        for arm in ARMS:
            start = time.perf_counter()
            results = run(head, TEST_ROOTS, data, arm, recipe)
            archive.update({f"{arm}/{key}": value for key, value in results.items()})
            summary = {key: float(value.mean()) for key, value in views(results["physical"], case, change).items()}
            print("confirmation", case, arm, summary, f"{time.perf_counter() - start:.1f}s", flush=True)
        temporary = out / f"{case}.partial.npz"
        np.savez_compressed(temporary, **archive)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(condition, CONDITIONS))
    files = {case: io.sha(out / f"{case}.npz") for case in CONDITIONS}
    seal = out / "confirmation.json"
    final = dict(files=files, heads_sha256=io.sha(heads_path))
    if seal.exists() and io.read(seal) != final:
        raise RuntimeError("Confirmed outcomes were modified")
    if not seal.exists():
        io.write(seal, final)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "develop", "validate", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out)


if __name__ == "__main__":
    main()
