"""Compare fixed repair checkpoints on the same ordinary and joint held-out lives."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity_report as stats, interleave_one_life_repair as repair
from . import interleave_one_life_eval as paid, interleave_one_life_swaps as swap
from .interleave_interference import DATA, EVIDENCE


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def measures(ordinary, joint, earned):
    truth = joint["truth"]
    out = {"ordinary": ordinary}
    for condition in paid.SUBSETS:
        snapshot = []
        for order in paid.six.ORDERS:
            score = paid.six.metrics(joint[f"{order}/active/side"], truth, condition)
            snapshot.append(score)
            out[f"snapshot/{condition}/{order}"] = score
        out[f"snapshot/{condition}"] = np.mean(snapshot, axis=0)
        consecutive = []
        for order in paid.study.ARM:
            score = earned[f"{order}/active/{condition}"]
            consecutive.append(score)
            out[f"paid/{condition}/{order}"] = score
        out[f"paid/{condition}"] = np.mean(consecutive, axis=0)
    return out


@torch.no_grad()
def run(cohort, step):
    torch.set_num_threads(1)
    births, _ = repair.setup(cohort)
    group = "original" if cohort == "original" else "confirmation"
    card = read(EVIDENCE / f"model-cards-{group}.json")
    old_worlds, old_seed = card["worlds"], card["seed"]
    production = read(EVIDENCE / f"one-life-{cohort}-disjoint-8000.json")
    seed, worlds = production["seed"], production["worlds"]
    old_tape = swap.capacity.make_tape("odor", old_seed, old_worlds, 1)
    joint_tape = paid.six.make(seed, worlds)
    out = DATA / "one-life-repair-eval" / cohort / str(step)
    out.mkdir(parents=True, exist_ok=True)
    paths = (Path(__file__), Path(repair.__file__), Path(paid.__file__), Path(swap.__file__),
             Path(paid.six.__file__), Path(swap.capacity.__file__), Path(repair.net.__file__),
             Path(repair.motor.__file__), Path(repair.prior.__file__), Path(repair.net.native.__file__))
    code = {p.name: repair.digest(p) for p in paths}
    cache_key = json.dumps(code, sort_keys=True)
    old = repair.prior.tri.old_fit
    series, hashes = {}, {}
    for arm in ("parent", *repair.ARMS):
        root_rows = []
        for b in births:
            if arm == "parent":
                location, tag, at = repair.prior.DEST / cohort, "interspersed", 8000
            else:
                location, tag, at = repair.DEST / cohort, arm, step
            checkpoint = old.checkpoint_path(location, "odor", tag, b, at)
            identity = repair.digest(checkpoint)
            hashes[checkpoint.relative_to(DATA).as_posix()] = identity
            g, r = old.load_checkpoint(location, "odor", tag, b, at)
            if arm == "parent":
                key = checkpoint.relative_to(DATA).as_posix()
                if production["hashes"][key] != identity:
                    raise ValueError("Parent checkpoint differs from the recorded benchmark")
                source = DATA / "one-life-eval" / cohort / "disjoint" / f"interspersed-{b}-8000.npz"
                key = source.relative_to(DATA).as_posix()
                if repair.digest(source) != production["hashes"][key]:
                    raise ValueError("Parent joint outcomes differ from the recorded benchmark")
                hashes[key] = production["hashes"][key]
                isolated = swap.old_trial(g, r, old_tape)
                original = swap.OUT / cohort / "old" / f"birth-{b}" / "interspersed_body-interspersed_rule.npz"
                hashes[original.relative_to(DATA).as_posix()] = repair.digest(original)
                with np.load(original) as existing:
                    for name in ("side", "truth", "per_world"):
                        np.testing.assert_array_equal(isolated[name], existing[name])
                with np.load(source) as raw:
                    root_rows.append(measures(isolated["per_world"], raw,
                                              {k[7:]: raw[k] for k in raw.files if k.startswith("reward/")}))
            else:
                file = out / f"{arm}-{b}.npz"
                if not file.exists():
                    isolated = swap.old_trial(g, r, old_tape)
                    joint = swap.new_trial(g, r, joint_tape)
                    earned = paid.consecutive(g, r, seed + 71, worlds)
                    raw = {**joint, **{f"ordinary/{k}": v for k, v in isolated.items()},
                           **{f"reward/{k}": v for k, v in earned.items()},
                           "checkpoint_sha256": identity, "evaluator": cache_key}
                    repair.prior.tri.revisit._write_raw(file, **raw)
                hashes[file.relative_to(DATA).as_posix()] = repair.digest(file)
                with np.load(file) as raw:
                    if str(raw["checkpoint_sha256"]) != identity or str(raw["evaluator"]) != cache_key:
                        raise ValueError("Repair cache does not match the checkpoint or evaluator")
                    root_rows.append(measures(raw["ordinary/per_world"], raw,
                                              {k[7:]: raw[k] for k in raw.files if k.startswith("reward/")}))
            print(cohort, step, arm, b, flush=True)
        series[arm] = {k: np.stack([r[k] for r in root_rows]) for k in root_rows[0]}
    for a, b in (("mixed", "long_only"), ("mixed", "parent"), ("long_only", "parent")):
        series[f"{a} minus {b}"] = {k: v - series[b][k] for k, v in series[a].items()}
    old_interval = stats.Crossed(len(births), old_worlds, old_seed + 19, 4000)
    joint_interval = stats.Crossed(len(births), worlds, seed + 1, 4000)
    results = {}
    for name, rows in series.items():
        results[name] = {}
        for key, values in rows.items():
            if key == "ordinary":
                labels, interval = swap.SPEC["old"], old_interval
            else:
                labels = paid.PAID if key.startswith("paid/") else paid.six.METRICS
                interval = joint_interval
            results[name][key] = {m: interval(values[:, :, i]) for i, m in enumerate(labels)}
    summary = dict(cohort=cohort, step=step, births=births, plan_sha256=repair.digest(repair.PLAN),
                   source_sha256=code, hashes=hashes, ordinary_worlds=old_worlds, ordinary_seed=old_seed,
                   joint_worlds=worlds, joint_seed=seed, results=results,
                   denominator="Each ordinary geometry crosses 4 A/B maps. Each joint geometry crosses 64 maps, "
                               "32 for context conflict or 8 for concurrent hard triangle and conflict. Three support "
                               "orders are equally weighted for independent probes and two for paid serial choices. "
                               "Draw 4000 paired crossed samples of births and common geometries. Parent ordinary "
                               "selection and per-world scores were reproduced exactly against the prior archive.")
    repair.prior.tri.revisit._write_json(EVIDENCE / f"one-life-repair-{cohort}-{step}.json", summary)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=repair.COHORTS, required=True)
    parser.add_argument("--step", type=int, choices=repair.STAGES, required=True)
    args = parser.parse_args()
    run(args.cohort, args.step)
