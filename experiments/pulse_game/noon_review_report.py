"""Combine preregistered birth-paired review replay results and their frozen parents."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import noon_review_world as world


def load_report(evidence, split, stage, cohort, worlds, counts):
    name = f"noon-review-{split}-{stage:06d}-{cohort}-n{worlds}-{'-'.join(map(str, counts))}.json"
    path = evidence / name
    return json.loads(path.read_text(encoding="utf8")), path


def _vector(report, name, key):
    return np.array([report[name][key]["birth_values"][str(root)] for root in sorted(
        int(root) for root in report[name][key]["birth_values"])])


def _read(report, baseline, measure):
    rates = {f"{arm}_trained_{branch}": _vector(report["rates"][arm], branch, measure)
             for arm, branch in (("native", "native"), ("random", "random"), ("uncertain", "uncertain"),
                                 ("native", "ordinary"), ("random", "ordinary"), ("uncertain", "ordinary"))}
    differences = {name: _vector(report["differences"], name, measure) for name in report["differences"]}
    if baseline is not None:
        parent_natural = _vector(baseline["rates"]["native"], "native", measure)
        parent_ordinary = _vector(baseline["rates"]["native"], "ordinary", measure)
        rates.update(parent_natural=parent_natural, parent_ordinary=parent_ordinary)
        for arm, branch in (("native", "native"), ("random", "random"), ("uncertain", "uncertain")):
            parent_execution = _vector(baseline["rates"]["native"], branch, measure)
            rates[f"parent_{branch}_execution"] = parent_execution
            differences[f"{arm}_trained_natural_minus_parent_same_execution"] = (
                rates[f"{arm}_trained_{branch}"] - parent_execution)
            differences[f"{arm}_trained_ordinary_minus_parent_ordinary"] = (
                rates[f"{arm}_trained_ordinary"] - parent_ordinary)
        differences["native_choice_gain_change_from_parent"] = (
            differences["same_body_native_minus_random"]
            - _vector(baseline["differences"], "same_body_native_minus_random", measure))
    return dict(rates=rates, differences=differences)


def report(path, stage, split, worlds, counts, parents=True):
    metrics = ("accuracy", "all_correct", "sampled_target_accuracy", "both_laws_accuracy", "expected_accuracy")
    reports, parents_by_cohort, deps = {}, {}, {}
    for cohort in world.COHORTS:
        current, filename = load_report(path, split, stage, cohort, worlds, counts)
        reports[cohort] = current["loads"]
        deps[filename.name] = prior.digest(filename)
        if parents:
            start, file = load_report(path, split, 0, cohort, worlds, counts)
            parents_by_cohort[cohort] = start["loads"]
            deps[file.name] = prior.digest(file)

    rng = np.random.default_rng([world.EVAL_DOMAIN, 41, world.EVAL_SEEDS[split]["original"]])
    indices = {cohort: rng.integers(0, len(births), (20_000, len(births))) for cohort, births in world.COHORTS.items()}
    results = {}
    for count in map(str, counts):
        result = {}
        for measure in metrics:
            by_cohort = {cohort: _read(current[count], parents_by_cohort.get(cohort, {}).get(count), measure)
                         for cohort, current in reports.items()}
            result[measure] = {}
            for group in ("rates", "differences"):
                result[measure][group] = {}
                for name in by_cohort["original"][group]:
                    distributions, points, individual = {}, {}, {}
                    for cohort, results_by_group in by_cohort.items():
                        values = results_by_group[group][name]
                        points[cohort] = float(values.mean())
                        distributions[cohort] = values[indices[cohort]].mean(1)
                        individual.update(dict(zip(map(str, world.COHORTS[cohort]), values.tolist(), strict=True)))
                    cohort_draws = tuple(distributions.values())
                    ci = np.quantile(np.mean(cohort_draws, axis=0), [.025, .975]).tolist()
                    result[measure][group][name] = dict(mean=float(np.mean(tuple(points.values()))), ci95=ci,
                                                       cohort_means=points, per_birth=individual)
        results[count] = result
    return dict(stage=stage, split=split, worlds=worlds, counts=counts, source_sha256=deps,
                interval="20,000 paired birth bootstrap draws, sampled separately in the two cohorts",
                results=results)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    p.add_argument("--stage", type=int, choices=world.CHECKPOINTS[1:], required=True)
    p.add_argument("--split", choices=world.EVAL_SEEDS, required=True)
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--counts", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--no-parents", action="store_true")
    a = p.parse_args()
    result = report(a.evidence, a.stage, a.split, a.worlds, a.counts, parents=not a.no_parents)
    file = a.evidence / f"noon-review-{a.split}-{a.stage:06d}-summary-n{a.worlds}-{'-'.join(map(str, a.counts))}.json"
    archive._write_json(file, result)
    if "4" in result["results"]:
        measures = result["results"]["4"]["accuracy"]
        for kind in ("rates", "differences"):
            for name, row in measures[kind].items():
                print(f"{name:50} {100 * row['mean']:+6.2f} "
                      f"[{100 * row['ci95'][0]:+6.2f}, {100 * row['ci95'][1]:+6.2f}]")
    print(file)


if __name__ == "__main__":
    main()
