"""Measure whether a finite-difference plastic direction generalizes to new worlds."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import random
from statistics import fmean

import torch

from experiments.seed_evolution.large_runner import _lock, _write
from .evolved_plasticity_run import load_or_write, read
from .local_final_comparison import _aggregate
from .local_rule_evaluator import Evaluation
from .local_rule_policy import PLASTIC_FIELDS
from .local_rule_run import OUT as TRAINING_RUN


OUT = TRAINING_RUN.with_name("2026-11-19-local-direction-audit-v1")
ARMS = ("local", "modulated")
SCHEMA = 1
ACTORS, VISITS, ISLANDS = 8, 16, 4
DISCOVERY_OFFSET, HOLDOUT_OFFSET = 400_000_019, 500_000_047
OUTCOMES = ("selection_score", "strict", "pair_strict_rate")


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _champions(training):
    by_arm, sources = {}, {}
    for arm in ARMS:
        folder = training / "arms" / arm / "generations" / "0001"
        checkpoint, summary = folder / "elites.pt", folder / "summary.json"
        for path in (checkpoint, summary):
            if not path.is_file():
                raise ValueError(f"Generation-one champions are not sealed at {path}")
            sources[path.relative_to(training).as_posix()] = _sha(path)
        bank = torch.load(checkpoint, map_location="cpu", weights_only=True)
        record = read(summary)
        winners = bank["champions"]
        expected = [record["islands"][str(i)]["champion"]["id"] for i in range(ISLANDS)]
        if ((bank["arm"], bank["generation"]) != (arm, 1)
                or [item["row"]["id"] for item in winners] != expected):
            raise ValueError(f"Checkpoint does not match the sealed champion IDs in {folder}")
        by_arm[arm] = winners
    return by_arm, sources


def _unit_directions(count, seed, *labels):
    chooser = random.Random(":".join(map(str, (seed, *labels))))
    gen = torch.Generator(device="cpu").manual_seed(chooser.randrange(2**63))
    rows = torch.randn((count, len(PLASTIC_FIELDS)), generator=gen, dtype=torch.float64)
    return rows / rows.norm(dim=1, keepdim=True)


def _variants(winner, directions, epsilon):
    genome = winner["genome"]
    center = genome["plastic"]
    if (center.shape != (len(PLASTIC_FIELDS),) or center.dtype != torch.float32 or not torch.isfinite(center).all()
            or epsilon > 1 - center.abs().max().item()):
        raise ValueError(f"Invalid plastic center or insufficient headroom for {winner['row']['id']}")
    operations = [(name, center + float(sign) * (direction * epsilon).to(center.dtype))
                  for name, direction in directions for sign in (1, -1)]
    return [(f"{name}{'+' if pos % 2 == 0 else '-'}", {**genome, "plastic": coefficients})
            for pos, (name, coefficients) in enumerate(operations)]


def _scores(evaluator, genomes, arm, world, width):
    rows = []
    for start in range(0, len(genomes), width):
        chunk = genomes[start:start + width]
        scores = evaluator(chunk, arm=arm, seed=world, actors=ACTORS, visits=VISITS)
        if len(scores) != len(chunk):
            raise ValueError(f"Incomplete evaluation for {arm}, world {world}")
        for row in scores:
            if any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) for key in OUTCOMES):
                raise ValueError(f"Invalid directional outcome in {arm}, world {world}")
        rows.extend({key: row[key] for key in OUTCOMES} for row in scores)
    return rows


def _panel(out, config, winners, variants, arm, stage):
    evaluator = Evaluation(config["device"], "audit")
    ids = [item["row"]["id"] for item in winners]
    labels = [label for label, _ in variants[0]]
    if any([label for label, _ in group] != labels for group in variants):
        raise ValueError("All island champions must have an identical evaluation budget")
    genomes = [genome for group in variants for _, genome in group]
    results = []
    try:
        for seed in config[f"{stage}_seeds"]:
            path = out / stage / arm / f"{seed}.json"
            meta = {"schema": SCHEMA, "seed": seed, "arm": arm, "stage": stage,
                    "champions": ids, "variants": labels}
            if path.is_file():
                record = read(path)
                if any(record.get(key) != value for key, value in meta.items()) or len(record["rows"]) != len(genomes):
                    raise ValueError(f"Directional journal differs from this request at {path}")
            else:
                rows = _scores(evaluator, genomes, arm, seed, config["eval_batch_size"])
                record = {**meta, "rows": rows}
                _write(path, record)
            grouped = {identifier: dict(zip(labels, record["rows"][start:start+len(labels)], strict=True))
                       for identifier, start in zip(ids, range(0, len(genomes), len(labels)), strict=True)}
            results.append({"seed": seed, "champions": grouped})
    finally:
        evaluator.clear_cache()
    print(f"{stage} {arm}: {len(results)} worlds × {len(genomes)} candidates", flush=True)
    return results


def _estimate(directions, discovery, identifier, epsilon):
    deltas = [[world["champions"][identifier][f"probe-{index}+"]["selection_score"]
               - world["champions"][identifier][f"probe-{index}-"]["selection_score"]
               for index in range(len(directions))] for world in discovery]
    slopes = torch.tensor(deltas, dtype=torch.float64).mean(0) / (2 * epsilon)
    estimate = (slopes[:, None] * directions).mean(0) * len(PLASTIC_FIELDS)
    length = estimate.norm().item()
    unit = estimate / length if length > 1.e-12 else torch.zeros_like(estimate)
    return unit, {"world_differences": deltas, "mean_directional_slopes": slopes.tolist(),
                  "raw_gradient_norm": length, "unit_direction": unit.tolist(), "nonzero": length > 1.e-12}


def _paired(world, ids, random_count):
    effects = {}
    for identifier in ids:
        variants = world["champions"][identifier]
        panel = {}
        for key in OUTCOMES:
            base = variants["parent"][key]
            plus, minus = variants["guided+"][key], variants["guided-"][key]
            random_mean = fmean(variants[f"random-{index}{sign}"][key]
                                for index in range(random_count) for sign in ("+", "-"))
            panel[key] = {"guided_plus_minus_parent": plus - base, "guided_plus_minus_guided_minus": plus - minus,
                          "guided_plus_minus_random_mean": plus - random_mean, "random_mean_minus_parent": random_mean - base}
        effects[identifier] = panel
    return effects


def _summarize(worlds):
    keys = tuple(next(iter(worlds[0]["paired"].values()))[OUTCOMES[0]])
    return {metric: {effect: _aggregate([fmean(parent[metric][effect] for parent in world["paired"].values())
                                          for world in worlds]) for effect in keys} for metric in OUTCOMES}


def _verdict(arm_summary):
    metrics = arm_summary["selection_score"]
    required = ("guided_plus_minus_parent", "guided_plus_minus_random_mean")
    passed = all(metrics[name]["ci95"] is not None and metrics[name]["ci95"][0] > 0. for name in required)
    return {"supports_guided_operator": passed,
            "criterion": "Both lower bounds of the 95% paired-world intervals for selection-score gain must exceed zero "
                         "against the parent and the average of the signed, equal-radius random directions."}


def run(out, config, parents):
    target = out / "direction_audit.json"
    if target.is_file():
        result = read(target)
        if result.get("protocol") != config:
            raise ValueError(f"Directional report does not match the sealed protocol at {target}")
        return result

    seed, epsilon = config["direction_seed"], config["radius_l2"]
    probe_count, random_count = config["discovery_directions"], config["random_directions"]
    result = {"protocol": config, "arms": {}}
    for arm in ARMS:
        winners = parents[arm]
        ids = [item["row"]["id"] for item in winners]
        probes = {ident: _unit_directions(probe_count, seed, arm, ident, "discovery") for ident in ids}
        detection = [_variants(item, ((f"probe-{i}", direction) for i, direction in enumerate(probes[ident])), epsilon)
                     for item, ident in zip(winners, ids, strict=True)]
        discovery = _panel(out, config, winners, detection, arm, "discovery")
        confirmation, directions = [], {}
        for item, ident in zip(winners, ids, strict=True):
            unit, estimate = _estimate(probes[ident], discovery, ident, epsilon)
            independent = _unit_directions(random_count, seed, arm, ident, "holdout")
            variants = [("guided", unit), *((f"random-{i}", d) for i, d in enumerate(independent))]
            confirmation.append([("parent", item["genome"]), *_variants(item, variants, epsilon)])
            directions[ident] = {**estimate, "discovery_unit_directions": probes[ident].tolist(),
                                 "holdout_random_unit_directions": independent.tolist()}
        holdout = _panel(out, config, winners, confirmation, arm, "holdout")
        for world in holdout:
            world["paired"] = _paired(world, ids, random_count)
        summary = _summarize(holdout)
        result["arms"][arm] = {"champions": ids, "directions": directions, "discovery": discovery,
                               "holdout": holdout, "paired_summary": summary, "verdict": _verdict(summary)}
    result["uncertainty"] = "95% percentile intervals bootstrap seeds, averaging the four fixed island champions " \
                            "within each world. They quantify independent-world variation for this bank of champions."
    _write(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=TRAINING_RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--direction-seed", type=int, default=61_119_301)
    parser.add_argument("--radius", type=float, default=.20)
    parser.add_argument("--discovery-directions", type=int, default=8)
    parser.add_argument("--random-directions", type=int, default=4)
    parser.add_argument("--discovery-worlds", type=int, default=4)
    parser.add_argument("--holdout-worlds", type=int, default=8)
    args = parser.parse_args(argv)
    if (min(args.discovery_worlds, args.holdout_worlds, args.discovery_directions,
            args.random_directions, args.eval_batch_size) < 1 or args.direction_seed < 0 or not 0 < args.radius <= 1):
        parser.error("Use nonempty world and direction counts, a positive batch size, and radius in (0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    training, out = args.training.resolve(), args.out.resolve()
    if out == training or training in out.parents:
        parser.error("Use a separate output directory for the direction audit")
    training_protocol = read(training / "protocol.json")
    seed = training_protocol["seed"]
    if (training_protocol["actors"], training_protocol["visits"], training_protocol["islands"]) != (ACTORS, VISITS, ISLANDS):
        raise ValueError("Direction audit requires the formal world and four island champions")
    parents, sources = _champions(training)
    discovery_seeds = list(range(seed + DISCOVERY_OFFSET, seed + DISCOVERY_OFFSET + args.discovery_worlds))
    holdout_seeds = list(range(seed + HOLDOUT_OFFSET, seed + HOLDOUT_OFFSET + args.holdout_worlds))
    discovery_budget = len(ARMS) * ISLANDS * len(discovery_seeds) * 2 * args.discovery_directions
    holdout_budget = len(ARMS) * ISLANDS * len(holdout_seeds) * (3 + 2 * args.random_directions)
    config = {"schema": SCHEMA, "training_run": str(training), "source_generation": 1, "sources": sources,
              "analysis_sha256": _sha(Path(__file__)), "training_protocol_sha256": _sha(training / "protocol.json"),
              "actors": ACTORS, "visits": VISITS, "requests_per_life": VISITS * 4, "device": args.device,
              "eval_batch_size": args.eval_batch_size, "plastic_fields": list(PLASTIC_FIELDS),
              "direction_seed": args.direction_seed, "radius_l2": args.radius, "discovery_seeds": discovery_seeds,
              "holdout_seeds": holdout_seeds, "discovery_directions": args.discovery_directions,
              "random_directions": args.random_directions, "discovery_candidate_world_evaluations": discovery_budget,
              "holdout_candidate_world_evaluations": holdout_budget,
              "score": "same selection score as training, including both-context strict-revisit bonus",
              "method": "central finite differences on discovery worlds, normalize the linear combination of "
                        "Gaussian unit directions, then test the signed estimate, parent and signed random directions "
                        "on common independent holdout worlds. Only the inherited plastic tensor is perturbed."}
    formal_seeds = {seed + training_protocol["development_seed_offset"] + g
                    for g in range(1, training_protocol["generations"] + 1)}
    for g in range(1, training_protocol["generations"] + 1):
        final = g == training_protocol["generations"]
        first = seed + (training_protocol["final_audit_seed_offset"] if final else training_protocol["audit_seed_offset"] + 1024 * g)
        formal_seeds.update(range(first, first + training_protocol["final_audit_seeds" if final else "audit_seeds"]))
    if set(discovery_seeds).intersection(holdout_seeds) or formal_seeds.intersection(discovery_seeds + holdout_seeds):
        raise ValueError("Direction-discovery and holdout worlds must be disjoint from each other and the training protocol")
    with _lock(out / ".lock"):
        load_or_write(out / "protocol.json", config)
        result = run(out, config, parents)
        for arm in ARMS:
            section = result["arms"][arm]
            metric = section["paired_summary"]["selection_score"]["guided_plus_minus_random_mean"]
            print(f"{arm}: guided minus random mean {metric['mean']:+.6f}, 95% CI {metric['ci95']}, "
                  f"supports guided operator={section['verdict']['supports_guided_operator']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
