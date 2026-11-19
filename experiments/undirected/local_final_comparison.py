"""Pair endpoint and receipt contrasts on independently held-out worlds."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from experiments.seed_evolution.large_runner import _write


SCHEMA = 1
BOOTSTRAP_REPLICATES = 10_000
OUTCOME_FIELDS = ("strict", "pair_strict_rate")
RECEIPT_KEYS = ("same", "other_context", "other_raw")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _aggregate(values):
    sample = np.array([value for value in values if value is not None], dtype=float)
    count = len(sample)
    interval = None
    if count > 1:
        draws = np.random.default_rng(0).integers(count, size=(BOOTSTRAP_REPLICATES, count))
        interval = np.quantile(sample[draws].mean(axis=1), [.025, .975]).tolist()
    return {"paired_worlds": count, "mean": float(sample.mean()) if count else None, "ci95": interval,
            "wins": int((sample > 0).sum()), "ties": int((sample == 0).sum()), "losses": int((sample < 0).sum())}


def _difference(minuend, subtrahend, keys):
    return {key: (minuend[key] - subtrahend[key] if minuend[key] is not None and subtrahend[key] is not None else None)
            for key in keys}


def _contrasts(worlds, keys, arms):
    pairs = tuple((arm, baseline) for position, arm in enumerate(arms) for baseline in arms[:position])
    for world in worlds:
        for endpoint in ("first", "final"):
            world[f"{endpoint}_arm_differences"] = {
                f"{arm}_minus_{baseline}": _difference(world[endpoint][arm], world[endpoint][baseline], keys)
                for arm, baseline in pairs}
        world["final_minus_first"] = {arm: _difference(world["final"][arm], world["first"][arm], keys) for arm in arms}

    def summarize(field):
        return {name: {key: _aggregate([world[field][name][key] for world in worlds]) for key in keys}
                for name in worlds[0][field]}

    return {"per_world": worlds, "paired_summary": {field: summarize(field) for field in
            ("first_arm_differences", "final_arm_differences", "final_minus_first")}}


def _endpoint_metrics(scores):
    pairs = sum(row["eligible_pairs"] for row in scores)
    successes = sum(row["pairs_both_strict"] for row in scores)
    if any(not 0 <= row["pairs_both_strict"] <= row["eligible_pairs"] for row in scores):
        raise ValueError("Strict pair count exceeds available pairs")
    return {"strict": fmean(row["strict"] for row in scores),
            "pair_strict_rate": successes / pairs if pairs else None,
            "eligible_pairs": pairs, "pairs_both_strict": successes}


def _sealed_scores(holdout, arms, islands):
    result = {}
    for arm in arms:
        by_seed = {seed: [] for seed in holdout["seeds"]}
        for island in islands:
            group = holdout["arms"][arm]["islands"][island]
            if (group["candidate"] != holdout["champions"][arm][island]
                    or [row["seed"] for row in group["paired_scores"]] != holdout["seeds"]):
                raise ValueError(f"Holdout does not match its seed list or champion for {arm} island {island}")
            for row in group["paired_scores"]:
                by_seed[row["seed"]].append(row)
        result[arm] = by_seed
    return result


def _first_on_final_worlds(out, config, first, final, evaluator_for, progress):
    arms = config["arms"]
    islands = tuple(str(i) for i in range(config["islands"]))
    if final["generation"] == 1:
        return _sealed_scores(final, arms, islands)

    root = out / "audits" / "final-comparison" / "generation-0001"
    seeds = final["seeds"]
    result = {}
    completed = 0
    for arm in arms:
        by_seed = {}
        result[arm] = by_seed
        checkpoint = out / "arms" / arm / "generations" / "0001" / "elites.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        winners = saved["champions"]
        ids = [first["champions"][arm][island] for island in islands]
        if (saved["arm"], saved["generation"], [item["row"]["id"] for item in winners]) != (arm, 1, ids):
            raise ValueError(f"Generation-one champion bank differs from its sealed holdout at {checkpoint}")
        for seed in seeds:
            journal = root / arm / f"{seed}.json"
            request = {"schema": SCHEMA, "generation": 1, "arm": arm, "seed": seed,
                       "actors": config["actors"], "visits": config["visits"], "champions": ids}
            progress(phase="final_comparison", current_arm=arm, comparison_completed=completed,
                     comparison_total=len(arms) * len(seeds), comparison_training_generation=1)
            if journal.is_file():
                record = _read(journal)
                if any(record.get(key) != value for key, value in request.items()):
                    raise ValueError(f"Comparison score does not match request at {journal}")
                scores = record["scores"]
            else:
                evaluator = evaluator_for("audit")
                scores = evaluator([item["genome"] for item in winners], arm=arm, seed=seed,
                                   actors=config["actors"], visits=config["visits"])
                if len(scores) != len(winners):
                    raise ValueError(f"Incomplete generation-one scores for {arm} on world {seed}")
                _endpoint_metrics(scores)
                _write(journal, {**request, "scores": scores})
            if len(scores) != len(winners):
                raise ValueError(f"Incomplete generation-one score checkpoint at {journal}")
            by_seed[seed] = scores
            completed += 1
    progress(phase="final_comparison", current_arm=None, comparison_completed=completed,
             comparison_total=completed, comparison_training_generation=1)
    return result


def _receipt_metrics(rows, opportunities):
    eligible = sum(row["eligible"] for row in rows)

    def rate(numerator, denominator=None):
        if not eligible:
            return None
        numerator_total = sum(row[numerator] for row in rows)
        if denominator is not None:
            numerator_total -= sum(row[denominator] for row in rows)
        return numerator_total / eligible

    metrics = {"eligible_events": eligible, "eligibility_rate": eligible / opportunities,
               "joint_correct_truthful": rate("joint_true"),
               "joint_truthful_minus_inverted": rate("joint_true", "joint_inverted")}
    for key in RECEIPT_KEYS:
        metrics[key + "_ack_minus_rej_repeat"] = rate(key + "_ack_repeat", key + "_rej_repeat")
        metrics[key + "_truthful_correct_gain"] = rate(key + "_truthful_correct", key + "_inverted_correct")
    same, other = metrics["same_ack_minus_rej_repeat"], metrics["other_context_ack_minus_rej_repeat"]
    metrics["repeat_specificity_context"] = same - other if eligible else None
    return metrics


def _receipt_comparison(reports, arms, generations, champion_count, seed_count):
    all_lists = [set(report["seeds"]) for report in reports.values()]
    seeds = sorted(set.intersection(*all_lists))[:seed_count]
    if len(seeds) != seed_count:
        raise ValueError(f"Receipt comparison requires {seed_count} shared seeds, found {len(seeds)}")

    by_panel = {}
    panels = ("first", "final")
    delays = sorted({row["delay"] for report in reports.values() for row in report["rows"]})
    for panel, generation in zip(panels, generations, strict=True):
        for arm in arms:
            report = reports[generation, arm]
            opportunities, remainder = divmod(report["opportunities"], len(report["seeds"]))
            if remainder or opportunities < 1:
                raise ValueError(f"Invalid opportunity denominator in receipt audit {generation}/{arm}")
            rows = defaultdict(list)
            for row in report["rows"]:
                rows[row["delay"], row["seed"]].append(row)
            for delay in delays:
                for seed in seeds:
                    group = rows[delay, seed]
                    if sorted(row["champion_index"] for row in group) != list(range(champion_count)):
                        raise ValueError(f"Incomplete receipt contrast for {generation}/{arm}/{seed}/{delay}")
                    by_panel[panel, arm, delay, seed] = _receipt_metrics(group, opportunities)

    metrics = tuple(by_panel["first", arms[0], delays[0], seeds[0]])
    comparisons = {}
    for delay in delays:
        worlds = [{"seed": seed, **{panel: {arm: by_panel[panel, arm, delay, seed] for arm in arms} for panel in panels}}
                  for seed in seeds]
        comparisons[str(delay)] = _contrasts(worlds, metrics, arms)
    return {"seeds": seeds, "delays": comparisons,
            "estimand": "Per-world change in each receipt effect, conditioning on eligible events separately for each "
                        "arm and generation. A conditional pair contributes only if both sides have eligible events. "
                        "Eligibility itself uses all opportunities and is reported alongside the effects."}


def compare_final(out, config, evaluator_for, progress):
    """Seal one final paired report after all normal training and receipt audits finish."""
    out = Path(out)
    first, final = 1, config["generations"]
    arms = config["arms"]
    audit = out / "audits"
    target = out / "final_comparison.json"
    normal_paths = {i: audit / f"{i:04d}.json" for i in sorted({first, final})}
    receipt_paths = {(g, arm): audit / f"receipt-{g:04d}-{arm}.json" for g in normal_paths for arm in arms}
    final_holdout = _read(normal_paths[final])
    first_holdout = _read(normal_paths[first]) if first != final else final_holdout
    seeds = final_holdout["seeds"]
    if (not final_holdout["final"] or final_holdout["generation"] != final or first_holdout["generation"] != first
            or len(seeds) != config["final_audit_seeds"] or len(seeds) != len(set(seeds))
            or final_holdout["actors"] != config["actors"] or final_holdout["visits"] != config["visits"]):
        raise ValueError("Final comparison requires the sealed first and final holdout audits")
    header = {"schema": SCHEMA, "arms": arms, "first_generation": first, "final_generation": final,
              "holdout_seeds": seeds, "actors": config["actors"], "visits": config["visits"],
              "analysis_sha256": _sha(Path(__file__)),
              "source_sha256": {path.relative_to(out).as_posix(): _sha(path)
                                for path in (*normal_paths.values(), *receipt_paths.values())}}
    if target.is_file():
        result = _read(target)
        if any(result.get(key) != value for key, value in header.items()):
            raise ValueError(f"Final comparison differs from its sealed inputs at {target}")
        return result

    receipts = {}
    for (generation, arm), path in receipt_paths.items():
        report = _read(path)
        expected_ids = list((first_holdout if generation == first else final_holdout)["champions"][arm].values())
        if (report["generation"], report["arm"], report["champions"]) != (generation, arm, expected_ids):
            raise ValueError(f"Receipt comparison mismatches selected champions at {path}")
        receipts[generation, arm] = report
    receipt_report = _receipt_comparison(receipts, arms, (first, final), config["islands"],
                                         config["receipt_seed_counts"]["first"])
    first_scores = _first_on_final_worlds(out, config, first_holdout, final_holdout, evaluator_for, progress)
    final_scores = _sealed_scores(final_holdout, arms, tuple(str(i) for i in range(config["islands"])))
    worlds = [{"seed": seed, "first": {arm: _endpoint_metrics(first_scores[arm][seed]) for arm in arms},
               "final": {arm: _endpoint_metrics(final_scores[arm][seed]) for arm in arms}} for seed in seeds]
    result = {**header, "holdout": {**_contrasts(worlds, OUTCOME_FIELDS, arms),
                                    "estimand": "Each world averages strict request rates across the island champions. "
                                                "Double-key rate pools both-strict pairs and all eligible pairs across islands."},
              "receipt": receipt_report,
              "uncertainty": "95% percentile intervals from 10000 resamples of the paired world deltas. "
                             "The intervals describe world-to-world variation for these checkpoints from one evolutionary run."}
    _write(target, result)
    return result
