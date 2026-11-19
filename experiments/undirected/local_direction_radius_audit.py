"""Test two smaller steps along the sealed plastic directions on unused worlds."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _lock, _write
from .evolved_plasticity_run import load_or_write, read
from . import local_direction_audit as previous


SCHEMA = 1
RADII = (0.0125, 0.05)
WORLD_COUNT, SEED_OFFSET = 16, 600_000_079
PRIMARY = ("guided_plus_minus_parent", "guided_plus_minus_random_mean")
FAMILY_SIZE = len(previous.ARMS) * len(RADII) * len(PRIMARY)
BOOTSTRAP_REPLICATES = 32_768
OUT = previous.OUT.with_name("2026-11-19-local-direction-radius-audit-v1")


def _prefix(radius):
    return f"radius-{radius:g}/"


def _candidate_groups(winners, assignments):
    groups = []
    for winner in winners:
        identifier = winner["row"]["id"]
        assignment = assignments[identifier]
        guiding = torch.tensor(assignment["unit_direction"], dtype=torch.float64)
        null_guidance = not assignment["nonzero"]
        if abs(guiding.norm().item() - (0. if null_guidance else 1.)) > 1.e-9:
            raise ValueError(f"Invalid sealed direction for {identifier}")
        randoms = torch.tensor(assignment["holdout_random_unit_directions"], dtype=torch.float64)
        if (randoms.ndim != 2 or randoms.shape[1] != len(previous.PLASTIC_FIELDS)
                or not torch.allclose(randoms.norm(dim=1), torch.ones_like(randoms[:, 0]), atol=1.e-9, rtol=0.)):
            raise ValueError(f"Invalid sealed random directions for {identifier}")
        directions = [("guided", guiding), *((f"random-{i}", d) for i, d in enumerate(randoms))]
        variants = [("parent", winner["genome"])]
        for radius in RADII:
            variants.extend((f"{_prefix(radius)}{name}", genome)
                            for name, genome in previous._variants(winner, directions, radius))
        groups.append(variants)
    return groups


def _radius_worlds(worlds, radius, ids, random_count):
    prefix = _prefix(radius)
    result = []
    for world in worlds:
        selected = {ident: {"parent": variants["parent"], **{key[len(prefix):]: value for key, value in variants.items()
                                                             if key.startswith(prefix)}}
                    for ident, variants in world["champions"].items()}
        row = {"seed": world["seed"], "champions": selected}
        row["paired"] = previous._paired(row, ids, random_count)
        result.append(row)
    return result


def _radius_summary(worlds, draws):
    summary = previous._summarize(worlds)
    for effect in PRIMARY:
        values = np.array([fmean(scores["selection_score"][effect] for scores in world["paired"].values())
                           for world in worlds])
        adjusted_tail = 0.05 / (2 * FAMILY_SIZE)
        summary["selection_score"][effect]["bonferroni_interval"] = np.quantile(
            values[draws].mean(axis=1), [adjusted_tail, 1 - adjusted_tail]).tolist()

    meets = all(summary["selection_score"][effect]["bonferroni_interval"][0] > 0 for effect in PRIMARY)
    return {"worlds": worlds, "paired_summary": summary, "supports_guided_operator": meets}


def run(out, config, prior, parents):
    target = out / "direction_radius_audit.json"
    if target.is_file():
        result = read(target)
        if result.get("protocol") != config:
            raise ValueError(f"Radius report differs from the sealed run at {target}")
        return result

    draws = np.random.default_rng(0).integers(WORLD_COUNT, size=(BOOTSTRAP_REPLICATES, WORLD_COUNT))
    result = {"protocol": config, "arms": {}, "interpretation":
              "Exploratory radius ladder chosen after the first test. Discovery estimates and random directions were "
              "reused without refitting, with unused worlds shared across all radii, arms, and their baselines. "
              "Paired intervals resample worlds and average the four fixed island champions within each world."}
    for arm in previous.ARMS:
        bank = parents[arm]
        ids = [winner["row"]["id"] for winner in bank]
        arm_report = prior["arms"][arm]
        if ids != arm_report["champions"]:
            raise ValueError(f"The previous direction report used different parents for {arm}")
        variants = _candidate_groups(bank, arm_report["directions"])
        scores = previous._panel(out, config, bank, variants, arm, "radius")
        result["arms"][arm] = {"parents": ids, "radius": {f"{radius:g}": _radius_summary(
            _radius_worlds(scores, radius, ids, config["random_directions"]), draws) for radius in RADII}}
    _write(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-audit", type=Path, default=previous.OUT / "direction_audit.json")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out, prior_path = args.out.resolve(), args.first_audit.resolve()
    prior = read(prior_path)
    original = prior["protocol"]
    training = Path(original["training_run"]).resolve()
    if out in (training, prior_path.parent) or training in out.parents or prior_path.parent in out.parents:
        parser.error("Use a separate output folder for the radius audit")
    if original["device"] == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)

    training_protocol = read(training / "protocol.json")
    training_hash = previous._sha(training / "protocol.json")
    parents, sources = previous._champions(training)
    if sources != original["sources"] or training_hash != original["training_protocol_sha256"]:
        raise ValueError("The source champions or sealed training protocol changed between audits")

    seed = training_protocol["seed"]
    worlds = list(range(seed + SEED_OFFSET, seed + SEED_OFFSET + WORLD_COUNT))
    previous_worlds = set(original["discovery_seeds"] + original["holdout_seeds"])
    if previous_worlds.intersection(worlds):
        raise ValueError("The radius audit must use new holdout worlds")
    random_count = original["random_directions"]
    config = {"schema": SCHEMA, "training_run": str(training), "first_direction_audit": str(prior_path),
              "first_direction_audit_sha256": previous._sha(prior_path), "training_protocol_sha256": training_hash,
              "analysis_sha256": previous._sha(Path(__file__)), "device": original["device"],
              "eval_batch_size": original["eval_batch_size"], "radius_seeds": worlds, "radii_l2": list(RADII),
              "random_directions": random_count, "actors": previous.ACTORS, "visits": previous.VISITS,
              "candidate_world_evaluations": len(previous.ARMS) * previous.ISLANDS * WORLD_COUNT *
                  (1 + len(RADII) * (2 + 2 * random_count)), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
              "primary_outcome": "selection_score with the strict both-context revisit bonus",
              "multiplicity": {"method": "Bonferroni over the two primary contrasts, two radii and two arms",
                               "family_size": FAMILY_SIZE, "per_interval_level": 1 - .05 / FAMILY_SIZE,
                               "secondary_outcomes": "Strict and double-key strict comparisons use descriptive 95% intervals."}}
    with _lock(out / ".lock"):
        load_or_write(out / "protocol.json", config)
        result = run(out, config, prior, parents)
        for arm in previous.ARMS:
            for radius, section in result["arms"][arm]["radius"].items():
                metrics = section["paired_summary"]["selection_score"]
                print(f"{arm} radius={radius} parent={metrics[PRIMARY[0]]['mean']:+.6f} "
                      f"random={metrics[PRIMARY[1]]['mean']:+.6f} "
                      f"supports guided operator={section['supports_guided_operator']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
