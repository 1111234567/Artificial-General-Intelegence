"""Audit physically observed cue/length pairs and combine independent birth cohorts."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from . import noon_horizon as env
from . import noon_review_cue_association as assay


CHOICE = ("long_minus_short_unknown", "reversed_mapping_long_minus_short_unknown",
          "fixed_mapping_long_minus_short_unknown", "same_cue_route_switched")
INTACT = ("head", "native", "association", "public")
CONTRASTS = ("head_minus_head_donor", "native_minus_native_donor", "head_minus_native", "association_minus_head",
             "association_minus_association_donor", "public_minus_head")


def audit(raw, worlds):
    count = worlds * 16
    line = np.arange(count)
    donor = line ^ 2
    cue = raw["cue"]
    long = raw["long"]
    np.testing.assert_array_equal(cue, cue[donor])
    np.testing.assert_array_equal(long, ~long[donor])
    for condition in assay.CONDITIONS:
        op = raw[f"{condition}/outcomes"]
        assert op.shape == (count, 2, assay.HORIZON + 1, 2)
        assert (np.abs(op[:, :, 0, 1]) == 1).all()
        assert (np.abs(op[long, :, 1:, 1]) == 1).all()
        assert not op[~long, :, 1:].any()
        support = raw[f"{condition}/support_taste"]
        np.testing.assert_array_equal(support, support[donor])
        if condition != "write":
            assert not raw[f"{condition}/graph_change"].any()
        for name in raw.files:
            if name.startswith(condition + "/") and name.endswith("/side"):
                path = name.removesuffix("/side")
                side = raw[name]
                np.testing.assert_array_equal(raw[path + "/events"], op[line, (side > 0).astype(int)])
                if path.endswith("_donor"):
                    original = path.removesuffix("_donor")
                    np.testing.assert_array_equal(side, raw[original + "/side"][donor])
        if condition == "neutral":
            continue
        demos = raw[f"{condition}/calibration"]
        np.testing.assert_array_equal(demos[:, :, 1:], demos[donor, :, 1:])
        np.testing.assert_array_equal(demos[:, :, 0], -demos[donor, :, 0])
        np.testing.assert_array_equal(demos[:, 0, 0], -demos[:, 1, 0])
        opportunities = np.count_nonzero(demos[:, :, 4:], axis=2)
        np.testing.assert_array_equal(np.sort(opportunities, axis=1), np.broadcast_to([0, 4], opportunities.shape))
        selected = (demos[:, 1, 0] == cue).astype(int)
        np.testing.assert_array_equal(opportunities[line, selected] > 0, long)
    return count


def keep(metric):
    if metric.startswith("write_minus_calibration_sham") or metric.endswith("/calibration_weight_change"):
        return True
    _, model, phase = metric.split("/", 2)
    return model in INTACT and phase in (*CHOICE, "total", "fixed_total", "reversed_total", "review", "tail") or (
        model in CONTRASTS and phase in ("total", "fixed_total", "reversed_total"))


def summarize(split, worlds, draws=8_000):
    collections, per_cohort, input_sha256 = {}, {}, {}
    verified = 0
    for cohort, births in env.COHORTS.items():
        report = env.EVIDENCE / f"noon-review-cue-association-{split}-{cohort}-n{worlds}.json"
        metadata = json.loads(report.read_text(encoding="utf8"))
        input_sha256[report.name] = env.digest(report)
        assert metadata["plan_sha256"] == env.digest(assay.PLAN)
        assert metadata["seal_sha256"] == env.digest(assay.SEAL)
        assert metadata["code_sha256"][Path(assay.__file__).name] == env.digest(assay.__file__)
        rng = np.random.default_rng([metadata["seed"], 14764])
        body = rng.multinomial(len(births), np.full(len(births), 1 / len(births)), size=draws) / len(births)
        geometry = rng.multinomial(worlds, np.full(worlds, 1 / worlds), size=draws) / worlds
        weight = (body[:, :, None] * geometry[:, None, :]).reshape(draws, -1)
        aggregate = []
        for filename, expected in metadata["raw_sha256"].items():
            path = assay.OUT / filename
            if env.digest(path) != expected:
                raise ValueError(f"File disagrees with the archived source: {path}")
            with np.load(path, allow_pickle=False) as record:
                verified += audit(record, worlds)
                aggregate.append({name: values for name, values in assay.rates(record, worlds).items() if keep(name)})
        names = tuple(aggregate[0])
        values = np.stack([np.stack([row[name] for name in names], axis=1) for row in aggregate])
        flat = values.reshape(-1, len(names))
        collections[cohort] = (flat.mean(0), weight @ flat)
        per_cohort[cohort] = {name: metadata["results"][name] for name in names}
    point = np.mean([data[0] for data in collections.values()], axis=0)
    hist = np.mean([data[1] for data in collections.values()], axis=0)
    ci = np.quantile(hist, (.025, .975), axis=0)
    pooled = {key: dict(mean=float(point[i]), ci95=ci[:, i].tolist()) for i, key in enumerate(names)}
    model = dict(split=split, worlds_per_birth=worlds, boot_draws=draws, complete_receipt_lanes_checked=verified,
                 source_sha256=input_sha256, pooled=pooled, cohorts=per_cohort)
    destination = env.EVIDENCE / f"noon-review-cue-association-{split}-summary-n{worlds}.json"
    env.storage._write_json(destination, model)
    print(split, f"audited {verified} lanes", {key: pooled[key] for key in (
        "write/head/long_minus_short_unknown", "write/head_minus_head_donor/total",
        "write/association/long_minus_short_unknown", "write/association_minus_head/total",
        "calibration_sham/association_minus_head/total", "calibration_sham/association_minus_association_donor/total")},
          flush=True)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--split", choices=assay.SEED, required=True)
    cli.add_argument("--worlds", type=int, default=256)
    cli.add_argument("--draws", type=int, default=8_000)
    a = cli.parse_args()
    summarize(a.split, a.worlds, a.draws)
