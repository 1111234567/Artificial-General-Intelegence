"""Check paired review effects with both birth and independent world variation."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import noon_review_world as world


COMPARISONS = {
    "native_minus_random_same_body": (("native", "native"), ("native", "random")),
    "native_minus_uncertain_same_body": (("native", "native"), ("native", "uncertain")),
    "native_minus_flipped_action": (("native", "native"), ("native", "flip_action")),
    "native_minus_swapped_content": (("native", "native"), ("native", "swap_content")),
    "native_minus_sham": (("native", "native"), ("native", "native_sham")),
    "native_minus_quiet": (("native", "native"), ("native", "quiet_after_menu")),
    "native_vs_random_training": (("native", "native"), ("random", "random")),
    "native_vs_uncertain_training": (("native", "native"), ("uncertain", "uncertain")),
}


def per_world(raw, branch, worlds, measure):
    correct = raw[f"{branch}/final_side"] == raw["truth"]
    if measure == "accuracy":
        values = correct.mean(1)
    elif measure == "all_correct":
        values = correct.all(1)
    else:
        values = correct[np.arange(len(correct)), raw["sampled_target"]]
    return values.reshape(worlds, 2).mean(1)


def analyze(path, stage, split, worlds, loads, draws):
    result = {}
    for load in loads:
        distributions = {}
        for name, births in world.COHORTS.items():
            seed = world.EVAL_SEEDS[split][name]
            rng = np.random.default_rng([world.EVAL_DOMAIN, seed, 315, load])
            b = rng.multinomial(len(births), np.full(len(births), 1 / len(births)), size=draws)
            w = rng.multinomial(worlds, np.full(worlds, 1 / worlds), size=draws)
            root = path / "evaluation" / split / f"{stage:06d}" / name / f"n{worlds}"
            cohort = {}
            for measure in ("accuracy", "all_correct", "sampled_target_accuracy"):
                comparison = {key: [] for key in COMPARISONS}
                for identifiers in births:
                    raw = {}
                    for arm in world.ARMS:
                        with np.load(root / f"{arm}-{identifiers}-{load}.npz", allow_pickle=False) as record:
                            raw[arm] = {key: value for key, value in record.items()
                                        if key.endswith("/final_side") or key in ("truth", "sampled_target")}
                    for key, ((arm_a, branch_a), (arm_b, branch_b)) in COMPARISONS.items():
                        left = per_world(raw[arm_a], branch_a, worlds, measure)
                        right = per_world(raw[arm_b], branch_b, worlds, measure)
                        comparison[key].append(left - right)
                cohort[measure] = {}
                for key, data in comparison.items():
                    values = np.stack(data)
                    estimates = np.einsum("iw,iw->i", b @ values, w) / (len(births) * worlds)
                    cohort[measure][key] = (values.mean(), estimates)
            distributions[name] = cohort
        result[str(load)] = {
            measure: {key: combine({name: data[measure][key] for name, data in distributions.items()})
                      for key in COMPARISONS} for measure in distributions["original"]}
    return dict(stage=stage, split=split, worlds=worlds, loads=loads, draws=draws,
                unit="For each paired draw and cohort, sample independent births and worlds separately with "
                     "replacement. Share each world's multiplicities across all selected births and arms within a cohort. "
                     "Keep both law twins together within each world.", results=result,
                source_sha256=prior.digest(Path(__file__)))


def combine(in_cohorts):
    def statistics(center, values):
        return dict(mean=float(center), ci95=np.quantile(values, [.025, .975]).tolist())

    point, draws = zip(*in_cohorts.values(), strict=True)
    return dict(pooled=statistics(np.mean(point), np.mean(draws, axis=0)),
                cohorts={name: statistics(*values) for name, values in in_cohorts.items()})


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=world.DATA / "core")
    p.add_argument("--evidence", type=Path, default=world.EVIDENCE)
    p.add_argument("--stage", type=int, choices=world.CHECKPOINTS[1:], required=True)
    p.add_argument("--split", choices=world.EVAL_SEEDS, default="heldout")
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--loads", nargs="+", type=int, default=[4])
    p.add_argument("--draws", type=int, default=20_000)
    a = p.parse_args()
    result = analyze(a.path, a.stage, a.split, a.worlds, a.loads, a.draws)
    name = f"noon-review-{a.split}-{a.stage:06d}-crossed-ci.json"
    archive._write_json(a.evidence / name, result)
    print(a.evidence / name)


if __name__ == "__main__":
    main()
