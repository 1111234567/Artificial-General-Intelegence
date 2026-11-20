"""Check what the frozen native menu prefers on the primary review tape."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import noon_review_world as world


def analyze(data, step, split, worlds):
    output, hashes = {}, {}
    for cohort, births in world.COHORTS.items():
        base = data / "evaluation" / split / f"{step:06d}" / cohort / f"n{worlds}"
        source = base / "tape-4.npz"
        hashes[source.relative_to(data).as_posix()] = prior.digest(source)
        with np.load(source, allow_pickle=False) as file:
            slots, laws, signs = (file[key] for key in ("candidates", "law", "menu_orientation"))
        rows = np.arange(len(laws))
        proxy = -laws[rows[:, None], slots] * signs
        opposed = proxy[:, 0] != proxy[:, 1]
        intervals = prior.BirthIntervals(list(births), 20_000, world.EVAL_SEEDS[split][cohort])
        metrics = []
        for birth in births:
            input_raw = base / f"native-{birth}-4.npz"
            hashes[input_raw.relative_to(data).as_posix()] = prior.digest(input_raw)
            with np.load(input_raw, allow_pickle=False) as raw:
                truth = raw["truth"]
                pre = raw["ordinary/final_side"] == truth
                active = raw["native/final_side"] == truth
                sham = raw["native_sham/final_side"] == truth
                other = raw["flip_action/final_side"] == truth
                menu_side = raw["menu_motor_side"] > 0
                confidence = raw["confidence"]
                native_item, flag_item = raw["native/object"], raw["uncertain/object"]
                logits = raw["menu_motor_logit"]
                actual = {name: raw[f"{name}/final_side"] == truth for name in ("native", "random", "uncertain")}
                review_reward = {name: raw[f"{name}/taste"] for name in actual}
            contains_error = pre[rows[:, None], slots].sum(1) == 1
            probability = 1 / (1 + np.exp(-logits))
            lift = (active.astype(float) - sham).mean(1)
            less_confident = raw_slot(confidence)
            row = dict(
                native_nominal_positive_given_opposed=(proxy[rows, menu_side.astype(int)][opposed] > 0).mean(),
                sole_candidate_error_prevalence=contains_error.mean(),
                native_picked_pre_error_given_sole_error=(~pre[rows, native_item])[contains_error].mean(),
                uncertainty_picked_pre_error_given_sole_error=(~pre[rows, flag_item])[contains_error].mean(),
                candidate_lowest_confidence_agreement=(menu_side[less_confident >= 0] ==
                                                       less_confident[less_confident >= 0]).mean(),
                absolute_menu_logit=np.abs(logits).mean(),
                menu_logistic_local_gradient=(probability * (1 - probability)).mean(),
                option_difference_not_zero=(active.mean(1) != other.mean(1)).mean(),
                mean_absolute_option_difference=np.abs(active.mean(1) - other.mean(1)).mean(),
                native_review_write_gain_given_sole_error=lift[contains_error].mean(),
            )
            terminal_reward = {name: 2 * hits.mean(1) - 1 for name, hits in actual.items()}
            total_reward = {name: review_reward[name] + reward for name, reward in terminal_reward.items()}
            for name in actual:
                row[f"{name}_immediate_review_payoff"] = review_reward[name].mean()
                row[f"{name}_one_uniform_terminal_payoff"] = terminal_reward[name].mean()
                row[f"{name}_review_plus_terminal_payoff"] = total_reward[name].mean()
            for name in ("random", "uncertain"):
                row[f"native_minus_{name}_review_plus_terminal_payoff"] = (
                    total_reward["native"] - total_reward[name]).mean()
            metrics.append(row)
        output[cohort] = {key: intervals.report([row[key] for row in metrics]) for key in metrics[0]}
    return dict(step=step, split=split, worlds=worlds, note="All four ordinary, sham and active choices are physical "
                "read-only probes from their matched snapshots. The pre-error stratum uses the sampled ordinary fork; "
                "the native policy has no access to that fork, its outcome, or the private projection signs.",
                source_sha256=hashes, results=output)


def raw_slot(confidence):
    return np.where(confidence[:, 0] == confidence[:, 1], -1, confidence.argmin(1))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=world.DATA / "core")
    p.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    p.add_argument("--stage", type=int, choices=world.CHECKPOINTS, required=True)
    p.add_argument("--split", choices=world.EVAL_SEEDS, default="heldout")
    p.add_argument("--worlds", type=int, default=384)
    a = p.parse_args()
    output = analyze(a.path, a.stage, a.split, a.worlds)
    name = f"noon-review-{a.split}-{a.stage:06d}-choice-usage-n{a.worlds}.json"
    archive._write_json(a.evidence / name, output)
    print(a.evidence / name)


if __name__ == "__main__":
    main()
