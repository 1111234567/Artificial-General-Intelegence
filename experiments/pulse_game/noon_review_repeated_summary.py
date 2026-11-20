"""Resample complete matched lives by birth and by shared world."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import noon_review_repeated as life
from . import noon_review_uncertainty as intervals
from . import noon_review_world as world


def summarize(split, stage, worlds, draws):
    comparisons = [("learned", name) for name in life.POLICIES if name != "learned"]
    if stage:
        comparisons.extend((arm, reference) for arm in life.FIT_ARMS for reference in ("learned", "random"))
        comparisons.append(("total", "later"))
    policies = tuple(dict.fromkeys(name for pair in comparisons for name in pair))
    report_sha256, distributions, round_points = {}, {}, {}
    checked = 0
    for cohort, births in world.COHORTS.items():
        filename = world.EVIDENCE / f"noon-review-repeated-{split}-{stage:04d}-{cohort}-n{worlds}.json"
        recorded = json.loads(filename.read_text(encoding="utf8"))
        report_sha256[filename.name] = prior.digest(filename)
        if recorded["plan_sha256"] != prior.digest(life.PLAN) or recorded["source_sha256"] != prior.digest(Path(life.__file__)):
            raise ValueError(f"The physical protocol is different from the one used in {filename}")
        for relative, expected in recorded["raw_sha256"].items():
            if prior.digest(life.DATA / relative) != expected:
                raise ValueError(f"Unexpected change in raw file {relative}")
            checked += 1
        seed = recorded["seed"]
        rng = np.random.default_rng([seed, life.EVAL_DOMAIN, 2])
        b = rng.multinomial(len(births), np.full(len(births), 1 / len(births)), size=draws)
        w = rng.multinomial(worlds, np.full(worlds, 1 / worlds), size=draws)

        def bootstrap(values):
            estimate = np.einsum("iw,iw->i", b @ values, w) / (len(births) * worlds)
            return values.mean(), estimate

        model_values, timeline = {}, {}
        root = life.DATA / "evaluation" / split / f"{stage:04d}" / cohort
        for policy in policies:
            samples, by_round = [], []
            for birth in births:
                with np.load(root / f"{policy}-{birth}-n{worlds}.npz", allow_pickle=False) as r:
                    samples.append({name: vector.reshape(worlds, 2).mean(1)
                                    for name, vector in life.resources(r).items()})
                    by_round.append(r["earned"].mean(0))
            model_values[policy] = {phase: np.array([row[phase] for row in samples]) for phase in samples[0]}
            timeline[policy] = np.array(by_round).mean(0)

        rates = {name: {phase: bootstrap(values) for phase, values in samples.items()}
                 for name, samples in model_values.items()}
        contrasts = {f"{left}_minus_{right}": {phase: bootstrap(value - model_values[right][phase])
                                               for phase, value in model_values[left].items()}
                     for left, right in comparisons}
        distributions[cohort] = dict(rates=rates, contrasts=contrasts)
        round_points[cohort] = {f"{left}_minus_{right}": (timeline[left] - timeline[right]).tolist()
                                for left, right in comparisons}
    pooled = {family: {name: {phase: intervals.combine({cohort: scores[family][name][phase]
                                                       for cohort, scores in distributions.items()})
                               for phase in phases} for name, phases in outcomes.items()}
              for family, outcomes in distributions["original"].items()}
    output = dict(split=split, stage=stage, worlds_per_birth=worlds, world_bootstrap_draws=draws,
                  note="Same-world and same-birth comparisons of actual life receipts. Independent birth weights are "
                       "drawn within each of two cohorts. World weights are shared across arms and births within each cohort; "
                       "both private-law twins remain together. Event units are -1 and +1.",
                  raw_files_verified=checked, reports_sha256=report_sha256, per_round_phases=life.EVENTS,
                  per_round_differences=round_points, **pooled)
    name = world.EVIDENCE / f"noon-review-repeated-{split}-{stage:04d}-summary-n{worlds}.json"
    archive._write_json(name, output)
    print(split, stage, {key: {p: row[p]["pooled"] for p in ("review", "later", "total")}
                         for key, row in pooled["contrasts"].items()}, flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=life.SEEDS, required=True)
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--draws", type=int, default=20_000)
    a = p.parse_args()
    summarize(a.split, a.stage, a.worlds, a.draws)


if __name__ == "__main__":
    main()
