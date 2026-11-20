"""Independently rescore frozen transfer traces and verify recorded identities."""

from argparse import ArgumentParser

import numpy as np
import torch

from . import interleave_capacity_report as stats, interleave_revisit_train as io
from . import noon_transfer as card, noon_transfer_archive as archive
from . import noon_transfer_forgetting as forgetting


def check_objects(z, tape, seed):
    target = tape.right_value[:, :, -1].numpy()
    np.testing.assert_array_equal(target, z["objects/truth"])
    support = z["objects/support"]
    # Each support contact's taste must be its actual motor side times its current physical truth.
    expected = support[:, 0, 0] * tape.right_value[:, :, 0].numpy().T
    np.testing.assert_array_equal(support[:, 1, 0], expected)
    values = {"objects/B_alone/B": (z["objects/B_alone/side"] == target[:, 1]).reshape(-1, 4).mean(1)}
    for count in card.OBJECT_COUNTS:
        key = f"objects/{count}"
        rng = np.random.default_rng([2057786111, seed, count])
        draws = np.repeat(rng.integers(count, size=tape.worlds), 4)
        flip = np.repeat(rng.choice((-1., 1.), tape.worlds), 4)
        np.testing.assert_array_equal(draws, z[f"{key}/external_selected"])
        event = z[f"{key}/external_contact"]
        np.testing.assert_array_equal(event[:, 1], event[:, 0] * tape.law.numpy()[np.arange(tape.lanes), draws] * flip)
        for mode in card.REVIEW_MODES:
            cell = f"{key}/{mode}"
            ok = (z[f"{cell}/side"] == target[:, :count]).reshape(-1, 4, count)
            values.update({f"{cell}/A": ok[:, :, 0].mean(1),
                           f"{cell}/all": (ok.sum(2) == count).mean(1), f"{cell}/mean": ok.mean((1, 2))})
            if count > 1:
                values[f"{cell}/B"] = ok[:, :, 1].mean(1)
                values[f"{cell}/recent"] = ok[:, :, -1].mean(1)
    return values


def check_snapshot(z, tape):
    laws = card.six.LAWS
    true = laws[None, :, :] * tape["query_orientation"][:, None, :]
    np.testing.assert_array_equal(true, z["snapshot/truth"])
    conflict = laws[:, 3] != laws[:, 4]
    conditions = {"all_maps": np.ones(len(laws), dtype=bool), "context_conflict": conflict,
                  "hard_and_context_conflict": conflict & (laws[:, :3] == laws[:, :1]).all(1)}
    result = {}
    for condition, keep in conditions.items():
        instances = []
        for order in card.six.ORDERS:
            a, b, c, x, y, mark = (z[f"snapshot/{order}/active/side"] == true)[:, keep].transpose(2, 0, 1)
            tri, both = a & b & c, x & y
            score = (tri & both & mark, tri, both & mark, both, mark,
                     a.astype(np.int8) + b + c + x + y + mark, x, y)
            instances.append(np.stack([column.mean(1) for column in score], 1))
        metrics = np.mean(instances, 0)
        np.testing.assert_array_equal(metrics, z[f"snapshot/per_world/{condition}"])
        result.update({f"snapshot/{condition}/{key}": v for key, v in zip(card.swap.SPEC["new"], metrics.T, strict=True)})
    return result


def check_triangle(z, law):
    true = law.numpy().reshape(-1, 8, 3)
    np.testing.assert_array_equal(true, z["triangle/truth"])
    side, passed = z["triangle/side"], z["triangle/margin_pass"]
    correct = (side == true[None, :, :, :, None]).sum(3) == 3
    both = passed[:, :, 0] & passed[:, :, 7]
    measures = (both.mean(0), both.all(0), passed.all((0, 2)), correct.mean((0, 2, 3)),
                correct[:, :, [0, 7]].mean((0, 2, 3)), correct[:, :, 1:7].mean((0, 2, 3)))
    np.testing.assert_array_equal(np.stack(measures, 1), z["triangle/per_world"])
    return {f"triangle/{name}": np.asarray(v, dtype=float)
            for name, v in zip(card.geometry.MEASURES, measures, strict=True)}


def verify(cohort, stage, swaps):
    suffix = "-swaps" if swaps else ""
    file = card.OUT / f"transfer-{cohort}-{stage}{suffix}.json"
    y = archive.read(file)
    plan, signature = archive.read(card.PLAN), archive.digest(card.PLAN)
    protocol = plan["cohorts"][cohort]
    births, counts, seeds = protocol["births"], plan["worlds"], protocol["seeds"]
    if (y["births"], y["seeds"], y["worlds"], y["plan_sha256"]) != (births, seeds, counts, signature):
        raise ValueError(f"Unexpected frozen transfer report identity: {file}")
    odor = card.capacity.make_tape("odor", seeds["objects"], counts["objects"], 7)
    scene = card.six.make(seeds["snapshot"], counts["snapshot"])
    triangle = card.triangle.make_tape(seeds["triangle"], 0, counts["triangle"], -.5)
    serial = {o: card.life.ticket(seeds["serial"], 0, 131, counts["serial"], o, domain=1665931, all_laws=True)
              for o in card.life.ARM}
    intervals = {k: stats.Crossed(len(births), n, seeds[k] + 367, 4000) for k, n in counts.items()}
    arrays = {}
    checks = 0

    def compare(summary, cells):
        if set(summary) != set(cells):
            raise ValueError("A recorded score is missing")
        for key, values in cells.items():
            computed = intervals[key.split("/", 1)[0]](values)
            for kind, datum in computed.items():
                np.testing.assert_allclose(datum, summary[key][kind], rtol=0, atol=1e-13)
        return len(cells)

    for model, index in y["hashes"].items():
        rows = []
        for birth in births:
            entry = index[str(birth)]
            for source in entry["sources"]:
                archive.verify(card.ROOT / source["path"], source["sha256"])
            trace = card.ROOT / entry["trace"]
            archive.verify(trace, entry["sha256"])
            with np.load(trace) as z:
                payments = {f"reward/{k.removeprefix('serial/')}": z[k] for k in z.files if k.startswith("serial/")}
                gains = archive.paid_from_raw(payments, serial)
                values = {**check_objects(z, odor, seeds["objects"]), **check_snapshot(z, scene),
                          **check_triangle(z, triangle["law"])}
                for condition in card.paid.SUBSETS:
                    mean = (gains["blocked/active/" + condition] + gains["interspersed/active/" + condition]) / 2
                    values.update({f"serial/{condition}/{metric}": m
                                   for metric, m in zip(card.paid.PAID, mean.T, strict=True)})
            rows.append(values)
        arrays[model] = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        checks += compare(y["models"][model], arrays[model])
        for group, summary in y["external_opportunity"][model].items():
            label, comparison = group.split("/", 1)
            first, second = comparison.split(" minus ")
            contrasted = {f"objects/{metric}": arrays[model][f"objects/{label}/{first}/{metric}"] -
                          arrays[model][f"objects/{label}/{second}/{metric}"] for metric in summary}
            checks += compare({f"objects/{m}": cell for m, cell in summary.items()}, contrasted)
    for pair, summary in y["paired"].items():
        a, b = pair.split(" minus ")
        checks += compare(summary, {k: v - arrays[b][k] for k, v in arrays[a].items()})
    return {"report_sha256": archive.digest(file), "checked_scores": checks, "models": len(arrays),
            "trajectories": len(arrays) * len(births)}


@torch.no_grad()
def verify_retention(stage):
    path = card.OUT / f"transfer-retention-{stage}.json"
    report, plan = map(archive.read, (path, forgetting.PLAN))
    archive.verify(forgetting.PLAN, report["plan"]["sha256"])
    archive.verify(card.ROOT / report["source"]["path"], report["source"]["sha256"])
    counts, worlds = plan["counts"], plan["worlds"]
    result = {}
    for cohort, group in report["cohorts"].items():
        seed = plan["cohorts"][cohort]
        tape = card.capacity.make_tape("odor", seed, worlds, max(counts) - 1)
        historical = archive.model(cohort, "uniform")
        births = group["births"]
        tests = stats.Crossed(len(births), worlds, seed + 41, 4000)
        data = {model: [] for model in group["models"]}
        for birth in births:
            folder, arm, at = historical
            parent = card.checkpoint.load_checkpoint(folder, "odor", arm, birth, at)
            forced = forgetting.select_support(*parent, tape)
            matched_side, matched_taste = (tensor.numpy() for tensor in forced)
            np.testing.assert_array_equal(matched_taste, matched_side * tape.right_value[:, :, 0].numpy().T)
            for model, rows in data.items():
                record = group["hashes"][f"{model}/{birth}"]
                for entry in (*record["sources"], record["trace"]):
                    archive.verify(card.ROOT / entry["path"], entry["sha256"])
                with np.load(card.ROOT / record["trace"]["path"]) as z:
                    np.testing.assert_array_equal(matched_side, z["forced_side"])
                    np.testing.assert_array_equal(matched_taste, z["earned"])
                    np.testing.assert_array_equal(tape.right_value[:, 0, -1].numpy(), z["truth"])
                    np.testing.assert_array_equal(z["write"][0], z["sham"][0])
                    np.testing.assert_array_equal(z["write"][0], z["idle"][0])
                    rows.append(forgetting.cells(z, counts))
        arrays = {model: {key: np.stack([row[key] for row in rows]) for key in rows[0]} for model, rows in data.items()}
        checked = 0

        def compare(summary, matrix):
            for key, values in matrix.items():
                for label, datum in tests(values).items():
                    np.testing.assert_allclose(summary[key][label], datum, atol=1e-13, rtol=0)
            return len(matrix)

        for model, matrix in arrays.items():
            checked += compare(group["models"][model], matrix)
        for pair, cell in group["contrasts"].items():
            a, b = pair.split(" minus ")
            checked += compare(cell, {k: v - arrays[b][k] for k, v in arrays[a].items()})
        checked += compare(group["body_rule_interaction"], {
            k: v - arrays["native_body_uniform_rule"][k] - arrays["uniform_body_native_rule"][k] + arrays["uniform"][k]
            for k, v in arrays["native"].items()
        })
        result[cohort] = {"verified_trajectories": len(data) * len(births), "checked_statistics": checked}
    result["report_sha256"] = archive.digest(path)
    io._write_json(card.OUT / f"transfer-retention-verification-{stage}.json", result)
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--swaps", action="store_true")
    parser.add_argument("--retention-only", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(1)
    if options.retention_only:
        print(verify_retention(options.stage))
    else:
        record = {cohort: verify(cohort, options.stage, options.swaps) for cohort in card.swap.COHORTS}
        name = str(options.stage) + ("-swaps" if options.swaps else "")
        io._write_json(card.OUT / f"transfer-verification-{name}.json", record)
        print(record)
