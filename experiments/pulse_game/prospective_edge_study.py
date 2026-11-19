"""Compare prospective edge-retention feedback with instant and shuffled controls."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from . import coupled_birth_study as prior, joint_live_study as shared
from . import prospective_edges as model, ring_logic_study as io


OUT = io.REPO / "docs/evidence/open-graph-learning-2026-11-19/edges"
PRIOR_RECIPE = io.REPO / "docs/evidence/free-graph-2026-11-19/coupled/recipe.json"
ROOTS = tuple(range(69075000, 69075016))
CONDITIONS = ("stable", "reversal", "return")
ARMS = model.ARMS
SAVED = (
    "physical", "edits", "event_records", "weight_step", "input_step",
    "final_mask", "final_raw", "final_state", "final_drive", "final_bias", "final_readout", "final_offset",
    "final_newborn_records",
)


def code_hash():
    files = ("prospective_edges.py", "prospective_edge_study.py", "prospective_edge_analysis.py")
    return prior.code_hash() | {file: io.sha(Path(__file__).with_name(file)) for file in files}


def register(out):
    """Seal fixed settings and the confirmation population before running lives."""
    path = out / "registered.json"
    if path.exists():
        raise RuntimeError("Registration already exists")
    borrowed = io.read(PRIOR_RECIPE)
    config = {key: borrowed[key] for key in ("rate", "birth_scale")}
    if set(ROOTS) & set(prior.DEV_ROOTS + prior.TEST_ROOTS + shared.DEV_ROOTS + shared.TEST_ROOTS):
        raise RuntimeError("Confirmation worlds overlap the base comparison")
    head = prior.prior.heads(shared.TEST_GROUPS)
    io.write(path, dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_sha256=code_hash(), earlier_recipe_sha256=io.sha(PRIOR_RECIPE),
        fixed_config=config, groups=shared.TEST_GROUPS, roots=ROOTS, trials=shared.TRIALS,
        conditions=CONDITIONS, arms=ARMS, schedule=asdict(shared.SCHEDULE), fitted_births=head[3], lanes=head[2],
        receipt="Continuous state and shared physical rollout. Each arm acts in its own copy of the same world. A timeout scores incorrect and supplies no signed training or retention sample. Opposite truthful sides are inferred only from actual contact side and delivered taste. Neither world truth, hidden intervention markers, nor reward from another arm is provided to the model.",
        shared_rules="All effective recurrent coefficients, input gate/candidate coefficients and biases, and readout/offset receive exact same clipped receipt SGD rule in all arms. Every ninth actual receipt may grow one dormant legal non-self arc chosen by dense effective gradient magnitude and initialized by the signed clipped gradient times the prior birth multiplier. Edge coefficients are independent and edits do not rescale other active weights. Every sixth actual receipt offers one possible retirement; removal must preserve strong connectivity. No fixed edge count, budgets, or compulsory removal. Schedule and SGD come from the earlier independently developed coupled birth experiment.",
        retirement=dict(
            instant="Original coupled-birth Taylor gradient score including current receipt. Existing newborn age rule requires at least two subsequent receipts.",
            later="On fourth and fifth actual receipts of each six-receipt block, record intact minus one-edge-silenced loss on fixed observed inputs, initial carry, realized movement mask and pre-update coefficients. Use EWMA with decay .5. The birth receipt cannot contribute to that generation's score. Every sixth receipt consider positive scores using only preceding samples and at least two samples for the current arc generation. Remove the highest ranked legally removable arc, or decline.",
            shuffled="Use identical samples and maturity condition. Before each retirement, randomly permute score identities within the life and separately within checkpoint and later-born cohorts using a private seeded RNG. Preserve each cohort's signed score distribution without associating it with its own arcs.",
        ),
        holdout="For all arms, at each sixth actual receipt, diagnose the past-only score association with the removal effect on that sixth receipt evaluated before any writes or edits. Save candidate-level effects, applied past scores and current generation IDs. The exact sixth receipt replay has no effect on any arm decision. The instantaneous comparator itself can use the ordinary Taylor score of the sixth receipt. Average within-life-slot Pearson correlations. Report initial and later-born cohorts separately.",
        physical="Primary simultaneous 95% intervals for six paired differences, later minus instant and later minus shuffled in stable/reversal/return last20. Crossed group/world bootstrap with 20000 draws, retain all three fitted tasks. Report early- and full-life windows descriptively. Evaluate both pairs' first shared-history retirement divergence and immediate next physical request. Stable/reversal and reversal/return share pre-change history and will not be added as independent samples.",
        bootstrap_seed=433587655, audit_index_fields=model.AUDIT_INDEX_FIELDS,
        audit_value_fields=model.AUDIT_VALUE_FIELDS, write_groups=model.WRITE_GROUPS, write_fields=model.WRITE_FIELDS,
    ))


def require(out):
    registration = io.read(out / "registered.json")
    if registration["source_sha256"] != code_hash() or registration["earlier_recipe_sha256"] != io.sha(PRIOR_RECIPE):
        raise RuntimeError("Registered source or borrowed recipe changed")
    return registration


def run(head, roots, data, arm, config):
    instances = []

    def make(*args, **kwargs):
        life = model.Population(*args, **kwargs, arm=arm, birth_scale=config["birth_scale"], audit=True)
        instances.append(life)
        return life

    full = shared.run(head, roots, data, shared.SCHEDULE, "joint", config["rate"], population=make)
    values = {key: full[key] for key in SAVED}
    life = instances[0]
    trajectory = values["physical"].shape[:-1]
    return values | dict(
        edge_count=np.stack(life.edge_counts, axis=1).reshape(trajectory),
        recurrent_write=np.stack(life.writes, axis=1).reshape(*trajectory, len(model.WRITE_GROUPS), len(model.WRITE_FIELDS)),
        birth_checks=np.stack(life.birth_checks, axis=1).reshape(*trajectory, len(prior.model.AUDIT_FIELDS)),
        audit_index=np.concatenate(life.evidence_index), audit_value=np.concatenate(life.evidence_values),
        audit_later_samples=np.concatenate(life.evidence_active),
    )


def confirm(out):
    registration = require(out)
    head = prior.prior.heads(shared.TEST_GROUPS)
    if head[3] != registration["fitted_births"] or head[2] != registration["lanes"]:
        raise RuntimeError("Registered checkpoint manifest changed")

    def condition(case):
        path = out / f"{case}.npz"
        if path.exists():
            raise RuntimeError(f"Cannot overwrite scored {case}")
        data = shared.tapes(ROOTS, head[2], case)
        archive = dict(pairs=data[3], flip_trial=data[4], initial_mask=head[1])
        for arm in ARMS:
            start = time.perf_counter()
            values = run(head, ROOTS, data, arm, registration["fixed_config"])
            archive.update({f"{arm}/{key}": value for key, value in values.items()})
            print(case, arm, f"{time.perf_counter() - start:.1f}s", flush=True)
        temporary = out / f"{case}.partial.npz"
        np.savez_compressed(temporary, **archive)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=len(CONDITIONS)) as pool:
        list(pool.map(condition, CONDITIONS))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out)


if __name__ == "__main__":
    main()
