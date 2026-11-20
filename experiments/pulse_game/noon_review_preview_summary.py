"""Uncertainty for the preregistered, paired native preview comparisons."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import noon_review_preview as previewing
from . import noon_review_uncertainty as intervals
from . import noon_review_world as world


READS = ("mean_accuracy", "all_correct")


def rates(values):
    return np.stack((values.mean(axis=-1), values.all(axis=-1)), axis=-1)


def choices(raw, state, behavior):
    options = raw[f"{state}/all_options"]
    if behavior == "random":
        return rates(options).mean(1)
    sides = (raw[f"{behavior}/emitted_side"] > 0).astype(int)
    return rates(options[np.arange(len(options)), sides])


def contrasts(informative, distractor):
    living = choices(informative, "informative", "informative")
    return {
        "native_minus_random_on_same_informative_state": living - choices(informative, "informative", "random"),
        "native_minus_distractor_policy_on_same_informative_state": (
            living - choices(informative, "informative", "distractor")),
        "informative_minus_distractor_training_on_same_informative_test": (
            living - choices(distractor, "informative", "informative")),
        "informative_minus_distractor_preview_as_experienced": living - choices(informative, "distractor", "distractor"),
    }


def analyze(path, end, worlds, draws):
    statistics = {}
    verified = {}
    for cohort, births in world.COHORTS.items():
        seed = previewing.FIT_SEEDS[cohort]
        rng = np.random.default_rng([world.EVAL_DOMAIN, seed, 716])
        birth_weights = rng.multinomial(len(births), np.full(len(births), 1 / len(births)), size=draws)
        world_weights = rng.multinomial(worlds, np.full(worlds, 1 / worlds), size=draws)
        cohort_rows = {}
        for stage in (0, end):
            rows = {}
            actual = {}
            directory = path / "evaluation" / "preview" / f"{stage:06d}" / cohort
            for birth in births:
                models = []
                for arm in previewing.FIT_ARMS:
                    dest = directory / f"{arm}-{birth}-n{worlds}.npz"
                    with np.load(dest, allow_pickle=False) as report:
                        models.append({name: report[name] for name in report.files})
                    verified[dest.relative_to(path).as_posix()] = prior.digest(dest)
                if stage == 0 and (models[0].keys() != models[1].keys() or
                                   any(not np.array_equal(models[0][key], models[1][key]) for key in models[0])):
                    raise ValueError(f"The matched preview training arms differ at birth {birth} before continuation")
                stats = contrasts(*models)
                for name, data in stats.items():
                    rows.setdefault(name, []).append(data.reshape(worlds, 2, len(READS)).mean(1))
                for arm, report in zip(previewing.FIT_ARMS, models, strict=True):
                    attained = choices(report, "informative", "informative")
                    actual.setdefault(arm, []).append(attained.reshape(worlds, 2, len(READS)).mean(1))
            cohort_rows[str(stage)] = {**{arm + "_training_native_on_informative_test": np.stack(values)
                                         for arm, values in actual.items()},
                                       **{name: np.stack(values) for name, values in rows.items()}}
        same_body_gain = "native_minus_random_on_same_informative_state"
        cohort_rows["learned_gain"] = {same_body_gain: cohort_rows[str(end)][same_body_gain] -
                                                       cohort_rows["0"][same_body_gain]}
        statistics[cohort] = {}
        for window, data in cohort_rows.items():
            columns = {}
            for name, matrix in data.items():
                estimates = [np.einsum("iw,iw->i", birth_weights @ matrix[:, :, index], world_weights)
                             / (len(births) * worlds) for index in range(len(READS))]
                columns[name] = dict(zip(READS, zip(matrix.mean((0, 1)), estimates, strict=True), strict=True))
            statistics[cohort][window] = columns
    original = statistics["original"]
    pooled = {stage: {name: {rate: intervals.combine({cohort: scored[stage][name][rate]
                                                    for cohort, scored in statistics.items()})
                            for rate in READS} for name in columns} for stage, columns in original.items()}
    return dict(stage=end, worlds=worlds, draws=draws, results=pooled, raw_sha256=verified,
                interval="Paired resampling of independent births and shared worlds, preserving opposite-law twins.",
                primary_condition="Both comparison arms see true candidates in the reported environment.")


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=previewing.FIT_PATH)
    p.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--draws", type=int, default=20_000)
    a = p.parse_args()
    report = analyze(a.path, a.stage, a.worlds, a.draws)
    file = a.evidence / f"noon-review-preview-paired-{a.stage:06d}-n{a.worlds}.json"
    archive._write_json(file, report)
    for name, scores in report["results"][str(a.stage)].items():
        value = scores["mean_accuracy"]["pooled"]
        print(f"{name:57} {100 * value['mean']:+.3f} "
              f"[{100 * value['ci95'][0]:+.3f}, {100 * value['ci95'][1]:+.3f}]")
    print(file)


if __name__ == "__main__":
    main()
