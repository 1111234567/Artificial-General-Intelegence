"""Score unchanged and continued models on paired held-out transfer worlds."""

from argparse import ArgumentParser
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity as capacity, interleave_capacity_report as stats
from . import interleave_cross_skill_eval as six, interleave_geometry_eval as geometry
from . import interleave_one_life as life, interleave_one_life_eval as paid
from . import interleave_one_life_swaps as swap, interleave_revisit_train as io
from . import interleave_stream as stream, interleave_triangle_train as triangle
from . import local_trace_graph as native, local_trace_train as checkpoint
from . import noon_transfer_archive as archive
from .interleave_interference import ROOT


DATA = ROOT / "data/active-review-20261120"
OUT = ROOT / "docs/evidence/active-review-20261120"
PLAN = OUT / "transfer-plan.json"
OBJECT_COUNTS = (1, 2, 4, 8)
REVIEW_MODES = ("no_review", "idle", "sham", "external_uniform")


def append_probes(raw, key, rule, state, tape, count):
    raw[key + "/side"] = torch.stack([capacity._probe(rule, state, tape, j).side for j in range(count)], 1).numpy()


def object_trials(graph, rule, tape, seed):
    """Fork native normal probes and three physical controls from the same state."""
    state = native.start(graph, tape.lanes)
    rule = rule.expand(tape.lanes)
    raw = {"objects/truth": tape.right_value[:, :, -1].numpy()}
    b_alone, b_log = capacity._study(rule, state, tape, 1, write=True)
    b_snap = stream.wait(rule, b_alone, 4)
    raw["objects/B_alone/side"] = capacity._probe(rule, b_snap, tape, 1).side.numpy()
    raw["objects/B_alone/contact"] = b_log.numpy()
    support = []
    for item in range(tape.max_k + 1):
        state, entry = capacity._study(rule, state, tape, item, write=True)
        support.append(entry.numpy())
        count = item + 1
        if count not in OBJECT_COUNTS:
            continue
        snap = stream.wait(rule, state, 4)
        key = f"objects/{count}"
        append_probes(raw, f"{key}/no_review", rule, snap, tape, count)

        # These invariants make the budget and sensory controls independent of private law.
        rng = np.random.default_rng([2057786111, seed, count])
        selected = torch.from_numpy(np.repeat(rng.integers(count, size=tape.worlds), 4))
        orientation = torch.from_numpy(np.repeat(rng.choice((-1., 1.), tape.worlds).astype(np.float32), 4))
        draws = torch.from_numpy(np.repeat(rng.random((tape.worlds, 6), dtype=np.float32), 4, axis=0))
        rows = torch.arange(tape.lanes)
        odor = tape.directions.repeat_interleave(4, 0)[rows, selected]
        right = -odor * orientation[:, None]
        factor = tape.law[rows, selected] * orientation
        raw[f"{key}/external_selected"] = selected.numpy()
        idle = stream.wait(rule, snap, capacity.CONTACT_TICKS + 4)
        append_probes(raw, f"{key}/idle", rule, idle, tape, count)
        contact = None
        for mode, intervention in (("sham", "sham"), ("external_uniform", "active")):
            visited, side, taste = stream.contact(rule, snap, -right, right, factor,
                                                  torch.zeros_like(factor), draws, intervention)
            if mode == "sham" and not capacity._same_graph(snap.graph, visited.graph):
                raise AssertionError("A sham review changed a graph parameter")
            next_contact = torch.stack((side, taste), 1).numpy()
            if contact is not None:
                np.testing.assert_array_equal(contact, next_contact)
            contact = next_contact
            after = stream.wait(rule, visited, 4)
            append_probes(raw, f"{key}/{mode}", rule, after, tape, count)
        raw[f"{key}/external_contact"] = contact
    raw["objects/support"] = np.stack(support)
    return raw


def object_metrics(raw):
    values = {}
    target = raw["objects/truth"]
    b = raw["objects/B_alone/side"] == target[:, 1]
    values["objects/B_alone/B"] = b.reshape(-1, 4).mean(1)
    for count in OBJECT_COUNTS:
        for mode in REVIEW_MODES:
            key = f"objects/{count}/{mode}"
            ok = raw[key + "/side"] == target[:, :count]
            values[f"{key}/A"] = ok[:, 0].reshape(-1, 4).mean(1)
            values[f"{key}/all"] = ok.all(1).reshape(-1, 4).mean(1)
            values[f"{key}/mean"] = ok.mean(1).reshape(-1, 4).mean(1)
            if count > 1:
                values[f"{key}/B"] = ok[:, 1].reshape(-1, 4).mean(1)
                values[f"{key}/recent"] = ok[:, -1].reshape(-1, 4).mean(1)
    return values


def score(raw):
    values = object_metrics(raw)
    for condition in paid.SUBSETS:
        mean_snapshot = raw[f"snapshot/per_world/{condition}"]
        for i, key in enumerate(swap.SPEC["new"]):
            values[f"snapshot/{condition}/{key}"] = mean_snapshot[:, i]
        mean_paid = np.mean([raw[f"serial/{order}/active/{condition}"] for order in life.ARM], axis=0)
        for i, key in enumerate(paid.PAID):
            values[f"serial/{condition}/{key}"] = mean_paid[:, i]
    for i, key in enumerate(geometry.MEASURES):
        values[f"triangle/{key}"] = raw["triangle/per_world"][:, i]
    return values


@torch.no_grad()
def evaluate(graph, rule, seeds, worlds):
    odor = capacity.make_tape("odor", seeds["objects"], worlds["objects"], max(OBJECT_COUNTS) - 1)
    raw = object_trials(graph, rule, odor, seeds["objects"])
    joint_tape = six.make(seeds["snapshot"], worlds["snapshot"])
    raw.update({f"snapshot/{k}": v for k, v in swap.new_trial(graph, rule, joint_tape).items()})
    raw.update({f"serial/{k}": v for k, v in paid.consecutive(graph, rule, seeds["serial"], worlds["serial"]).items()})
    tape = triangle.make_tape(seeds["triangle"], 0, worlds["triangle"], -.5)
    scores, sides, margins = geometry.triad(graph, rule, tape)
    raw.update({"triangle/per_world": scores, "triangle/side": sides, "triangle/margin_pass": margins,
                "triangle/truth": tape["law"].numpy().reshape(-1, 8, 3)})
    return raw


def code_fingerprints():
    modules = (capacity, stats, six, geometry, life, paid, swap, stream, triangle, native, checkpoint, native.native)
    files = {Path(__file__), *(Path(m.__file__) for m in modules)}
    return {p.name: archive.digest(p) for p in sorted(files)}


def fixed_source(manifest, cohort, name, birth):
    if name in ("uniform", "triangle_expert"):
        return manifest[cohort]["experts"][f"{name}/{birth}"]
    return manifest[cohort]["confirmation"]["checkpoints"][f"{name}/{birth}"]


def row(cohort, birth, key, specs, signature, seeds, counts):
    source_hashes = []
    for params, expected in specs:
        folder, arm, step = params
        path = checkpoint.checkpoint_path(folder, "odor", arm, birth, step)
        identity = archive.digest(path) if expected is None else archive.verify(path, expected)
        source_hashes.append({"path": path.relative_to(ROOT).as_posix(), "sha256": identity})
    source_key = json.dumps(source_hashes, sort_keys=True)
    path = DATA / "transfer" / cohort / f"{key}-{birth}.npz"
    if not path.exists():
        models = [checkpoint.load_checkpoint(root, "odor", arm, birth, at) for (root, arm, at), _ in specs]
        raw = evaluate(models[0][0], models[-1][1], seeds, counts)
        for source in source_hashes:
            archive.verify(ROOT / source["path"], source["sha256"])
        io._write_raw(path, signature=signature, sources=source_key, **raw)
    with np.load(path) as raw:
        if str(raw["signature"]) != signature or str(raw["sources"]) != source_key:
            raise ValueError(f"Cached trace belongs to a different protocol or checkpoint: {path}")
        values = score(raw)
    index = {"trace": path.relative_to(ROOT).as_posix(), "sha256": archive.digest(path), "sources": source_hashes}
    return values, index


def run(cohort, stage, *, swaps):
    torch.set_num_threads(1)
    plan = archive.read(PLAN)
    protocol = plan["cohorts"][cohort]
    archived = archive.read(archive.REPORT)["cohorts"]
    births, seeds, counts = protocol["births"], protocol["seeds"], plan["worlds"]
    sources = {name: archive.model(cohort, name) for name in plan["baseline_models"]}
    if stage:
        sources.update({f"{arm}_{stage}": (DATA / "core" / cohort, arm, stage) for arm in plan["training_arms"]})
    pairs = {model: (model,) for model in sources}
    if swaps:
        pairs[f"native_body_uniform_rule_{stage}"] = (f"native_{stage}", "uniform")
        pairs[f"uniform_body_native_rule_{stage}"] = ("uniform", f"native_{stage}")
    plan_hash = archive.digest(PLAN)
    fingerprints = code_fingerprints()
    signature = json.dumps({"plan": plan_hash, "code": fingerprints, "numpy": np.__version__,
                            "torch": str(torch.__version__)}, sort_keys=True)
    for name, (folder, tag, step) in sources.items():
        for birth in births:
            path = checkpoint.checkpoint_path(folder, "odor", tag, birth, step)
            if not path.is_file():
                raise FileNotFoundError(f"Complete paired stage required for {name}: {path}")
    (DATA / "transfer" / cohort).mkdir(parents=True, exist_ok=True)
    summaries, arrays, hashes = {}, {}, {}
    intervals = {assay: stats.Crossed(len(births), count, seeds[assay] + 367, 4000) for assay, count in counts.items()}
    for name, component_models in pairs.items():
        rows = []
        hashes[name] = {}
        for birth in births:
            specs = []
            for model in component_models:
                expected = None
                if model in plan["baseline_models"]:
                    expected = fixed_source(archived, cohort, model, birth)["sha256"]
                specs.append((sources[model], expected))
            values, hashes[name][str(birth)] = row(cohort, birth, name, specs, signature, seeds, counts)
            rows.append(values)
            print(cohort, name, birth, flush=True)
        arrays[name] = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        summaries[name] = {key: intervals[key.split("/", 1)[0]](v) for key, v in arrays[name].items()}

    comparisons = [("mixed", "long_only"), ("mixed", "parent"), ("long_only", "parent")]
    if stage:
        arms = [f"{arm}_{stage}" for arm in plan["training_arms"]]
        comparisons.extend((model, "uniform") for model in arms)
        comparisons.extend(combinations(arms, 2))
    if swaps:
        comparisons.extend((name, model) for name, parts in pairs.items() if len(parts) == 2 for model in parts)
    between = {f"{a} minus {b}": {key: intervals[key.split("/", 1)[0]](v - arrays[b][key])
                                 for key, v in arrays[a].items()} for a, b in comparisons}
    opportunity = {}
    for name, values in arrays.items():
        opportunity[name] = {}
        for count in OBJECT_COUNTS:
            for a, b in (("external_uniform", "idle"), ("external_uniform", "sham"), ("idle", "no_review")):
                key = f"objects/{count}"
                head = f"{key}/{a}/"
                opportunity[name][f"{count}/{a} minus {b}"] = {
                    metric: intervals["objects"](value - values[f"{key}/{b}/{metric}"])
                    for full_key, value in values.items() if full_key.startswith(head)
                    for metric in (full_key.removeprefix(head),)}
    result = {"cohort": cohort, "stage": stage, "includes_swaps": swaps, "births": births, "seeds": seeds,
              "worlds": counts, "plan_sha256": plan_hash, "source_sha256": fingerprints,
              "numpy": np.__version__, "torch": str(torch.__version__), "hashes": hashes,
              "models": summaries, "paired": between, "external_opportunity": opportunity}
    name = str(stage) + ("-swaps" if swaps else "")
    io._write_json(OUT / f"transfer-{cohort}-{name}.json", result)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=tuple(swap.COHORTS))
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--swaps", action="store_true")
    args = parser.parse_args()
    if args.swaps and not args.stage:
        parser.error("Component swaps require a completed nonzero stage")
    run(args.cohort, args.stage, swaps=args.swaps)
