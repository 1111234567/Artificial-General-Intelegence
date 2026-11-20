"""Cross inherited graph and local write rule in paired frozen one-life assays."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import interleave_capacity as capacity, interleave_capacity_report as stats
from . import interleave_cross_skill_eval as six, interleave_geometry_eval as geometry
from . import interleave_homeostasis as home, interleave_one_life as life
from . import interleave_revisit_train as io, local_trace_train as checkpoint
from .interleave_interference import DATA, EVIDENCE, ROOT


OUT = DATA / "one-life-swaps"
NAME = "one-life-body-rule-swap"
COHORTS = {"original": tuple(range(731, 737)), "independent": tuple(range(811, 817))}
GENERATIONS = ("parent", "interspersed")
COMBINATIONS = tuple(product(GENERATIONS, repeat=2))
SPEC = {"old": ("A", "B", "A_and_B"), "new": (
    "all_six", "triangle_three", "context_both_mark", "context_both", "mark", "number_correct",
    "context_A", "context_B")}
CONDITIONS = ("all_maps", "context_conflict", "hard_and_context_conflict")


def _read(path):
    return json.loads(path.read_text(encoding="utf8"))


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _index(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _hash(path)}


def _export(cohort, mark, birth):
    if mark == "parent":
        source, arm = geometry.sources(cohort, "hard", 64000)
        stage = 64000
    else:
        source, arm, stage = life.DEST / cohort, "interspersed", 8000
    return source, arm, stage, checkpoint.checkpoint_path(source, "odor", arm, birth, stage)


def _seal(cohort):
    label = "original" if cohort == "original" else "confirmation"
    card = EVIDENCE / f"model-cards-{label}.json"
    evidence = {assay: EVIDENCE / f"one-life-{cohort}-{split}-8000.json"
                for assay, split in (("old", "shared"), ("new", "disjoint"))}
    card_value = _read(card)
    metadata = {"old": card_value, "new": _read(evidence["new"])}
    if list(COHORTS[cohort]) != metadata["old"]["births"] or list(COHORTS[cohort]) != metadata["new"]["births"]:
        raise ValueError("Archived paired births disagree")
    for assay, n in (("old", 128), ("new", 96)):
        if metadata[assay]["worlds"] != n:
            raise ValueError("Archive world count differs")
    return card, evidence, metadata


def plan(out):
    protocols = {}
    for cohort, births in COHORTS.items():
        card, evidence, meta = _seal(cohort)
        digests = {assay: _read(file)["hashes"] for assay, file in evidence.items()}
        models = {}
        for generation in GENERATIONS:
            models[generation] = {}
            for birth in births:
                _, _, stage, file = _export(cohort, generation, birth)
                identity = _index(file)
                archives = {}
                for assay, split in (("old", "shared"), ("new", "disjoint")):
                    archive = DATA / "one-life-eval" / cohort / split / f"{generation}-{birth}-{stage}.npz"
                    recorded = digests[assay]
                    if (recorded[file.relative_to(DATA).as_posix()] != identity["sha256"]
                            or recorded[archive.relative_to(DATA).as_posix()] != _hash(archive)):
                        raise ValueError(f"Production artifact changed at {archive}")
                    archives[assay] = _index(archive)
                models[generation][str(birth)] = {"checkpoint": identity, "paired_production": archives}
        protocols[cohort] = dict(births=births, card=_index(card), reports={k:_index(v) for k,v in evidence.items()},
                                 seeds={k:v["seed"] for k,v in meta.items()},
                                 worlds={k:v["worlds"] for k,v in meta.items()}, models=models)
    immutable = dict(cohorts=protocols, cells=[label(*pair) for pair in COMBINATIONS],
                     measures=SPEC, regimes=CONDITIONS, support_orders=six.ORDERS, bootstrap=4000,
                     source_sha256={p.name:_hash(p) for p in map(Path, (__file__, six.__file__, geometry.__file__,
                                                              capacity.__file__, checkpoint.__file__, stats.__file__))},
                     protocol="Cross the inherited graph weights, bias and support as a whole with the frozen local "
                              "write rule from either generation of the same birth. Original models were 64k direct "
                              "triangle parents and one-life interspersed 8k continuations. The old card uses exactly the "
                              "same 128 ordinary odor geometries, crossed private laws and actual A/B terminal probes. "
                              "The new assay uses original one-life disjoint 96 geometries and 64 crossed laws, all "
                              "three published support orders equally, active only and native timing. Six no-taste "
                              "terminal probes branch from one common state. Retain the measured interaction of the "
                              "two components because swaps change their inherited pairing.")
    return home._lock(out / "run.json", immutable), immutable


def label(body, rule):
    return f"{body}_body-{rule}_rule"


def old_trial(graph, rule, tape):
    raw = capacity.evaluate(graph, rule, tape, (0, 1), snapshot_worlds=0)
    truth = raw["right_value"][:, :, -1].T
    side = raw["side"][0, 1]
    ok = side == truth
    scores = np.stack((ok[0], ok[1], ok.all(0))).T.reshape(tape.worlds, 4, 3).mean(1)
    return dict(side=side, truth=truth, contact=raw["contact"][0],
                logit=raw["logit"][0, 1], per_world=scores)


def new_trial(graph, rule, tape):
    worlds = len(tape["left"])
    truth = six.LAWS[None, :, :] * tape["query_orientation"][:, None, :]
    rows = {"truth": truth.astype(np.int8)}
    means = {cond: [] for cond in CONDITIONS}
    for i, order in enumerate(six.ORDERS):
        parts = []
        for j in range(0, worlds, 12):
            parts.append(six.simulate(graph, rule, {k: v[j:j+12] for k,v in tape.items()}, i, "active"))
        side, contact = (np.concatenate(v) for v in zip(*parts, strict=True))
        rows[f"{order}/active/side"] = side
        rows[f"{order}/active/contact"] = contact
        for condition in CONDITIONS:
            joint = six.metrics(side, truth, condition)
            mask = np.ones(len(six.LAWS), dtype=bool)
            if condition != "all_maps":
                mask &= six.LAWS[:, 3] != six.LAWS[:, 4]
            if condition == "hard_and_context_conflict":
                mask &= (six.LAWS[:, :3] == six.LAWS[:, :1]).all(1)
            individual = (side[:, mask, 3:5] == truth[:, mask, 3:5]).mean(1)
            means[condition].append(np.concatenate((joint, individual), axis=1))
    for condition, values in means.items():
        rows[f"per_world/{condition}"] = np.mean(values, axis=0)
    return rows


def _verify_coadapted(assay, raw, source):
    with np.load(source, allow_pickle=False) as archived:
        if assay == "old":
            np.testing.assert_array_equal(raw["per_world"][:, -1], archived["card"][:, geometry.CARD_KEYS.index("all_2")])
            return
        for key, values in raw.items():
            if key == "truth" or "/active/" in key:
                np.testing.assert_array_equal(values, archived[key])


@torch.no_grad()
def replay(out, cohort):
    signature, immutable = plan(out)
    declaration = immutable["cohorts"][cohort]
    tapes = {"old": capacity.make_tape("odor", declaration["seeds"]["old"], 128, 1),
             "new": six.make(declaration["seeds"]["new"], 96)}
    old_public = {k: v.numpy() for k,v in vars(tapes["old"]).items() if isinstance(v, torch.Tensor)}
    for assay, evaluate in (("old", old_trial), ("new", new_trial)):
        bank = (old_public if assay == "old" else tapes[assay])
        io._write_raw(out / cohort / assay / "worlds.npz", **bank)
        for birth in COHORTS[cohort]:
            models = {}
            for parent in GENERATIONS:
                path, arm, step, _ = _export(cohort, parent, birth)
                models[parent] = checkpoint.load_checkpoint(path, "odor", arm, birth, step)
            for body, rule in COMBINATIONS:
                name = label(body, rule)
                dest = out / cohort / assay / f"birth-{birth}" / f"{name}.npz"
                if dest.exists():
                    with np.load(dest, allow_pickle=False) as saved:
                        if saved["protocol_sha256"].item() != signature:
                            raise ValueError(f"Previous replay differs at {dest}")
                    continue
                start = perf_counter()
                raw = evaluate(models[body][0], models[rule][1], tapes[assay])
                if body == rule:
                    archive = declaration["models"][body][str(birth)]["paired_production"][assay]
                    _verify_coadapted(assay, raw, ROOT / archive["path"])
                    raw["paired_production_sha256"] = archive["sha256"]
                io._write_raw(dest, **raw, protocol_sha256=signature)
                print(cohort, assay, birth, name, f"{perf_counter()-start:.2f}s", flush=True)


def summarize(out, assays):
    signature, immutable = plan(out)
    output = dict(protocol_sha256=signature, protocol=immutable, validated_assays=assays, cohorts={}, raw_sha256={})
    pp, pi, ip, ii = (label(*k) for k in COMBINATIONS)
    for cohort, s in immutable["cohorts"].items():
        outputs = {}
        for assay in assays:
            directory = out / cohort / assay
            public = directory / "worlds.npz"
            output["raw_sha256"][public.relative_to(out).as_posix()] = _hash(public)
            raw_values = {}
            alternatives = CONDITIONS if assay == "new" else ("ordinary",)
            compiler = stats.Crossed(len(s["births"]), s["worlds"][assay], s["seeds"][assay] + (1 if assay=="new" else 19), 4000)
            for body, rule in COMBINATIONS:
                name = label(body, rule)
                per_birth = {k: [] for k in alternatives}
                for birth in s["births"]:
                    file = directory / f"birth-{birth}" / f"{name}.npz"
                    with np.load(file, allow_pickle=False) as checkpoint_raw:
                        if checkpoint_raw["protocol_sha256"].item() != signature:
                            raise ValueError(f"Stored simulation uses a different protocol at {file}")
                        if body == rule:
                            archive = s["models"][body][str(birth)]["paired_production"][assay]
                            if checkpoint_raw["paired_production_sha256"].item() != archive["sha256"]:
                                raise ValueError(f"Native archive check missing for {file}")
                            _verify_coadapted(assay, checkpoint_raw, ROOT / archive["path"])
                        for k in alternatives:
                            v = checkpoint_raw["per_world" if assay=="old" else f"per_world/{k}"]
                            per_birth[k].append(v)
                    output["raw_sha256"][file.relative_to(out).as_posix()] = _hash(file)
                raw_values[name] = {k:np.stack(v) for k,v in per_birth.items()}
            results = {}
            for scope in alternatives:
                values = {name: scopes[scope] for name,scopes in raw_values.items()}
                replacements = {"interspersed_minus_parent": values[ii]-values[pp],
                                "rule_on_parent_body": values[pi]-values[pp],
                                "rule_on_interspersed_body": values[ii]-values[ip],
                                "body_with_parent_rule": values[ip]-values[pp],
                                "body_with_interspersed_rule": values[ii]-values[pi],
                                "interaction_difference_of_differences": values[ii]-values[ip]-values[pi]+values[pp]}
                def tidy(cells):
                    return {name:{metric:compiler(matrix[:,:,i]) for i,metric in enumerate(SPEC[assay])}
                            for name,matrix in cells.items()}
                results[scope] = dict(cells=tidy(values), paired_differences=tidy(replacements))
            outputs[assay] = results
        output["cohorts"][cohort] = outputs
    result_path = EVIDENCE / (f"{NAME}.json" if set(assays) == set(SPEC) else f"{NAME}-ordinary.json")
    io._write_json(result_path, output)
    print(result_path, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "replay", "report"))
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--assays", nargs="+", choices=("old","new"), default=("old","new"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase=="replay":
        if args.cohort is None:
            parser.error("One replay process needs one cohort")
        replay(args.out,args.cohort)
    elif args.phase=="report":
        summarize(args.out,args.assays)
    else:
        print(plan(args.out)[0],flush=True)


if __name__=="__main__":
    main()
