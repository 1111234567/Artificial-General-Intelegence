"""Run a paired life experiment of signed and sign-placebo joint edge birth."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
from pathlib import Path
import time

import numpy as np

from . import coupled_birth as model, joint_live_study as shared
from . import live_module_edits_study as prior, ring_logic_study as io


OUT = io.REPO / "data/pulse-game/coupled-birth-20261119"
DEV_ROOTS = tuple(range(69025000, 69025006))
TEST_ROOTS = tuple(range(69027000, 69027016))
CONDITIONS = ("stable", "reversal")
ARMS = ("signed", "independent_birth_sign")
CANDIDATES = ((.015, 4.), (.06, 1.), (.06, 4.), (.24, 4.))
SAVED = ("physical", "edits", "event_records", "final_mask", "final_raw", "final_newborn_records", "weight_step")


def code_hash():
    source = prior.code_hash() | {path.name: io.sha(path) for path in (
        Path(__file__), Path(model.__file__), Path(__file__).with_name("joint_live_model.py"))}
    # Only the shared rollout functions are used. Upstream experimental phases are independent.
    for func in (shared.tapes, shared.run, shared.windows):
        source[f"joint_live_study.py::{func.__name__}"] = hashlib.sha256(inspect.getsource(func).encode()).hexdigest()
    return source


def register(out):
    path = out / "registered.json"
    if path.exists():
        raise RuntimeError("Already registered")
    if set(DEV_ROOTS + TEST_ROOTS) & set(shared.DEV_ROOTS + shared.TEST_ROOTS + prior.DEV_ROOTS + prior.TEST_ROOTS):
        raise RuntimeError("World roots overlap an earlier experiment")
    io.write(path, dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_sha256=code_hash(),
        dev_groups=shared.DEV_GROUPS, test_groups=shared.TEST_GROUPS, dev_roots=DEV_ROOTS, test_roots=TEST_ROOTS,
        trials=shared.TRIALS, conditions=CONDITIONS, arms=ARMS, schedule=asdict(shared.SCHEDULE),
        protocol="Shared causal physical lifecycle. Continuous state, 120 trials, balanced ordered odors, actual first contact only. Stable or hidden reversal at rooted trial 40 or 60. Each arm has all active effective weights, all candidate/gate input weights and biases, motor readout and offset trainable on every actual nonzero receipt; no replay writes. Births can occur every ninth actual receipt and retirements every sixth. Every remaining node stays in one strongly connected graph. No edge count, degree, or backbone preservation constraint.",
        mechanism="Exact dense one-request BPTT before any update. Pick the legal dormant non-self edge maximizing squared current loss sensitivity. At zero, include this edge's signed descent in the same jointly clipped SGD update, scaled by the birth multiplier. In the sign placebo, fresh fair sign at equal magnitude changes only the newborn coordinate on that receipt. The loss for all existing parameters and all subsequent updates uses true contact taste in both arms. Edges retire by positive EWMA of Taylor predicted removal improvement, with two receipts of age for newborns. Surgery does not rescale any other edge. Both arms store independent effective weights after exact initial normalization.",
        audit="Per birth, replay the recorded inputs with unchanged movement mask, pre-update carried state and other pre-update parameters. Estimate signed reduction of the same receipt's surrogate loss from the single added value; this does not feed selection or learning. First common birth should match in opportunity, location and magnitude because the initial lives are identical.",
        selection="Choose the signed arm's learning rate and birth multiplier on dev only by highest average of stable last20 and reversal last20 actual-contact accuracy, equally weighted. Ties use registration order. Confirm both arms with the single chosen pair.",
        candidates=[dict(rate=rate, birth_scale=scale) for rate, scale in CANDIDATES],
        primary="Paired signed minus sign placebo physical accuracy in the final 20 requests of each condition. Also evaluate the next request after the first common birth and the actual contact side differences for causal evidence of the first birth perturbation. Crossed 20000 paired bootstrap draws by development-independent birth group and world root, always retaining all three rules. Primary stable/reversal and next-request stable/reversal contrasts each occupy a simultaneous 95% family of two.",
        audit_fields=model.AUDIT_FIELDS, bootstrap_seed=7647514))


def require_registered(out):
    if io.read(out / "registered.json")["source_sha256"] != code_hash():
        raise RuntimeError("Registered source changed")


def run(head, roots, data, arm, config, *, audit=False):
    instances = []

    def make(*args, **kwargs):
        brain = model.Population(*args, **kwargs, birth_scale=config["birth_scale"],
                                 independent_sign=(arm == "independent_birth_sign"), audit=audit)
        instances.append(brain)
        return brain

    results = shared.run(head, roots, data, shared.SCHEDULE, "joint", config["rate"], population=make)
    traces = {name: results[name] for name in SAVED}
    if audit:
        shape = traces["physical"].shape[:-1]
        traces["birth_checks"] = np.stack(instances[0].birth_checks, axis=1).reshape(*shape, len(model.AUDIT_FIELDS))
    return traces


def develop(out):
    require_registered(out)
    if (out / "recipe.json").exists():
        raise RuntimeError("Development already sealed")
    head = prior.heads(shared.DEV_GROUPS)
    sets = {case: shared.tapes(DEV_ROOTS, head[2], case) for case in CONDITIONS}

    def evaluate(pair):
        config = dict(rate=pair[0], birth_scale=pair[1])
        accuracy = {}
        for case, data in sets.items():
            values = run(head, DEV_ROOTS, data, "signed", config)
            accuracy[case] = float(shared.windows(values["physical"], data[-2])["last20"].mean())
        result = config | dict(accuracy=accuracy, score=float(np.mean(list(accuracy.values()))))
        print("development", result, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        candidates = list(pool.map(evaluate, CANDIDATES))
    best = max(candidates, key=lambda row: row["score"])
    io.write(out / "development.json", dict(checkpoints=head[3], rows=head[2], candidates=candidates))
    io.write(out / "recipe.json", best | dict(registered_sha256=io.sha(out / "registered.json"),
                                              development_sha256=io.sha(out / "development.json")))


def sealed_recipe(out):
    require_registered(out)
    recipe = io.read(out / "recipe.json")
    for stage in ("registered", "development"):
        if recipe[f"{stage}_sha256"] != io.sha(out / f"{stage}.json"):
            raise RuntimeError("Development seal mismatch")
    return recipe


def confirm(out):
    recipe = sealed_recipe(out)
    head = prior.heads(shared.TEST_GROUPS)
    io.write(out / "confirmed-heads.json", dict(rows=head[2], checkpoints=head[3], recipe_sha256=io.sha(out / "recipe.json")))

    def condition(case):
        path = out / f"{case}.npz"
        if path.exists():
            raise RuntimeError(f"Cannot overwrite scored {case}")
        data = shared.tapes(TEST_ROOTS, head[2], case)
        archive = dict(pairs=data[3], flip_trial=data[4], initial_mask=head[1])
        for arm in ARMS:
            start = time.perf_counter()
            values = run(head, TEST_ROOTS, data, arm, recipe, audit=True)
            archive.update({f"{arm}/{key}": value for key, value in values.items()})
            print("confirmation", case, arm, f"{time.perf_counter() - start:.1f}s", flush=True)
        temporary = out / f"{case}.partial.npz"
        np.savez_compressed(temporary, **archive)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(condition, CONDITIONS))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out)


if __name__ == "__main__":
    main()
