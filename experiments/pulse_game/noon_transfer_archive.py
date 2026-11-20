"""Verify the repair endpoints and their original behavioral evidence."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_capacity_report as stats
from . import interleave_one_life_eval as paid
from . import interleave_one_life_repair_eval as audit
from . import interleave_one_life_swaps as swap
from . import interleave_revisit_train as io
from . import local_trace_train as checkpoint
from .interleave_interference import DATA, EVIDENCE, ROOT


REPORT = ROOT / "docs/evidence/active-review-20261120/transfer-archived-verification.json"
ARMS = ("parent", "long_only", "mixed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return actual


def model(cohort, name):
    """Return the original loader location, arm and stage for the fixed source."""
    if name == "uniform":
        folder = "retention-objective" if cohort == "original" else "retention-objective-confirmation"
        return DATA / folder, "uniform", 8000
    if name == "triangle_expert":
        return DATA / f"triangle-long-{cohort}", "direct", 64000
    if name == "parent":
        return DATA / "one-life" / cohort, "interspersed", 8000
    if name not in ARMS:
        raise ValueError(f"Unknown archive model: {name}")
    return DATA / "one-life-repair" / cohort, name, 2000


def paid_from_raw(raw, tickets):
    """Recompute both payouts and three outcomes from physical sides and the tape."""
    result = {}
    for order, t in tickets.items():
        query = t["query"].numpy().reshape(-1, 64, 2)
        np.testing.assert_array_equal(query, raw[f"reward/{order}/query"])
        terminal = raw[f"reward/{order}/paid"]
        expected = terminal[:, :, :, 0] * (t["qsign"] * t["law"].gather(1, t["query"])).numpy().reshape(query.shape)
        np.testing.assert_array_equal(expected, terminal[:, :, :, 1])
        good = expected > 0
        triangle = (good & (query < 3)).any(2)
        context = (good & (query >= 3)).any(2)
        for condition, mask in zip(paid.SUBSETS, paid.masks(), strict=True):
            actual = np.stack((triangle & context, triangle, context), axis=-1)[:, mask].mean(1)
            np.testing.assert_array_equal(actual, raw[f"reward/{order}/active/{condition}"])
            result[f"{order}/active/{condition}"] = actual
    return result


def verify_score_table(report_path, cohort, confirmation):
    y = read(report_path)
    births = swap.COHORTS[cohort]
    if y["births"] != list(births) or y["step"] != 2000:
        raise ValueError(f"Unexpected archive cohort/stage in {report_path}")
    plan = (EVIDENCE / "one-life-repair-confirmation-plan.json") if confirmation else audit.repair.PLAN
    plan_hash = verify(plan, y["plan_sha256"])
    for rel, expected in y["hashes"].items():
        verify(DATA / rel, expected)
    n, seed = y["joint_worlds"], y["joint_seed"]
    t = paid.six.make(seed, n)
    truth = paid.six.LAWS[None, :, :] * t["query_orientation"][:, None, :]
    tickets = {order: paid.study.ticket(seed + 71, 0, 131, n, order, domain=1665931, all_laws=True)
               for order in paid.study.ARM}
    cap = swap.capacity.make_tape("odor", y["ordinary_seed"], y["ordinary_worlds"], 1)
    target = cap.right_value[:, :, -1].numpy().T
    models = {}
    trajectories = {}
    for arm in ARMS:
        arrays = []
        for birth in births:
            location, tag, step = model(cohort, arm)
            file = checkpoint.checkpoint_path(location, "odor", tag, birth, step)
            identity = y["hashes"][file.relative_to(DATA).as_posix()]
            models[f"{arm}/{birth}"] = {"path": file.relative_to(ROOT).as_posix(), "sha256": identity}
            if confirmation:
                raw_path = DATA / "one-life-repair-confirmation" / cohort / f"{arm}-{birth}.npz"
            elif arm == "parent":
                raw_path = DATA / "one-life-eval" / cohort / "disjoint" / f"interspersed-{birth}-8000.npz"
            else:
                raw_path = DATA / "one-life-repair-eval" / cohort / "2000" / f"{arm}-{birth}.npz"
            with np.load(raw_path) as raw:
                if str(raw["checkpoint_sha256"]) != identity:
                    raise ValueError(f"Archived trajectory does not match its model: {raw_path}")
                np.testing.assert_array_equal(truth, raw["truth"])
                if not confirmation and arm == "parent":
                    isolated = swap.OUT / cohort / "old" / f"birth-{birth}" / "interspersed_body-interspersed_rule.npz"
                    with np.load(isolated) as old:
                        np.testing.assert_array_equal(target, old["truth"])
                        ok = old["side"] == target
                        recorded = old["per_world"]
                else:
                    np.testing.assert_array_equal(target, raw["ordinary/truth"])
                    ok = raw["ordinary/side"] == target
                    recorded = raw["ordinary/per_world"]
                scores = np.stack((ok[0], ok[1], ok.all(0))).T.reshape(-1, 4, 3).mean(1)
                np.testing.assert_array_equal(scores, recorded)
                arrays.append(audit.measures(scores, raw, paid_from_raw(raw, tickets)))
        trajectories[arm] = {key: np.stack([a[key] for a in arrays]) for key in arrays[0]}
    for a, b in (("mixed", "long_only"), ("mixed", "parent"), ("long_only", "parent")):
        trajectories[f"{a} minus {b}"] = {k: v - trajectories[b][k] for k, v in trajectories[a].items()}

    ordinary = stats.Crossed(len(births), y["ordinary_worlds"], y["ordinary_seed"] + 19, 4000)
    joint = stats.Crossed(len(births), n, seed + 1, 4000)
    checked = 0
    for arm, values in trajectories.items():
        for assay, array in values.items():
            if assay == "ordinary":
                interval, labels = ordinary, swap.SPEC["old"]
            else:
                interval = joint
                labels = paid.PAID if assay.startswith("paid/") else paid.six.METRICS
            for index, label in enumerate(labels):
                rescore = interval(array[:, :, index])
                for stat, value in rescore.items():
                    np.testing.assert_allclose(value, y["results"][arm][assay][label][stat], rtol=0, atol=1e-13)
                checked += 1
    headline = (
        ("ordinary", "A_and_B"),
        ("snapshot/context_conflict", "context_both_mark"),
        ("snapshot/context_conflict", "all_six"),
        ("snapshot/all_maps", "all_six"),
        ("paid/context_conflict", "two_paid"),
    )
    return {
        "index": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": digest(report_path)},
        "plan_sha256": plan_hash,
        "verified_archive_files": len(y["hashes"]),
        "checked_statistics": checked,
        "checkpoints": models,
        "scores": {arm: {f"{a}/{k}": v[a][k] for a, k in headline} for arm, v in y["results"].items()},
    }


def verify_experts(cohort):
    label = "original" if cohort == "original" else "confirmation"
    uniform_index = EVIDENCE / f"retention-objective-{'matched' if label == 'original' else label}-008000.json"
    triangle_index = EVIDENCE / f"triangle-transfer-{label}.json"
    card_index = EVIDENCE / f"model-cards-{label}.json"
    uniform, triangle, card = map(read, (uniform_index, triangle_index, card_index))
    card_folder = DATA / "model-cards" / label
    result = {}
    for birth in swap.COHORTS[cohort]:
        for name in ("uniform", "triangle_expert"):
            folder, arm, step = model(cohort, name)
            file = checkpoint.checkpoint_path(folder, "odor", arm, birth, step)
            if name == "uniform":
                key = file.relative_to(folder).as_posix()
                identity = verify(file, uniform["checkpoint_sha256"][key])
                export = checkpoint.checkpoint_path(card_folder / "learners", "mixed", "uniform-memory", birth, step)
                verify(export, card["hashes"][export.relative_to(card_folder).as_posix()])
                current = checkpoint.load_checkpoint(folder, "odor", arm, birth, step)
                copied = checkpoint.load_checkpoint(card_folder / "learners", "mixed", "uniform-memory", birth, step)
                assert_same_parameters(current, copied)
                verify(card_folder / f"uniform-memory-{birth}.npz", card["hashes"][f"uniform-memory-{birth}.npz"])
            else:
                identity = verify(file, triangle["raw_sha256"][file.relative_to(DATA).as_posix()])
            result[f"{name}/{birth}"] = {"path": file.relative_to(ROOT).as_posix(), "sha256": identity}
    return result


def assert_same_parameters(first, second):
    for field in swap.capacity.GRAPH_FIELDS:
        torch.testing.assert_close(getattr(first[0], field), getattr(second[0], field), rtol=0, atol=0)
    for key, value in first[1].state_dict().items():
        torch.testing.assert_close(value, second[1].state_dict()[key], rtol=0, atol=0)


def verify_repair_resume(cohort):
    """Bind all exported births to the final optimizer at the intended credit count."""
    results = {}
    for arm in ("long_only", "mixed"):
        location, name, step = model(cohort, arm)
        resume = location / arm / "resume.pt"
        manifest = location / arm / "run.json"
        y = torch.load(resume, map_location="cpu", weights_only=True)
        if y["step"] != step or y["manifest"] != digest(manifest):
            raise ValueError(f"Repair resume identity differs: {resume}")
        for lane, birth in enumerate(swap.COHORTS[cohort]):
            graph, rule = checkpoint.load_checkpoint(location, "odor", name, birth, step)
            for key, all_lanes in y["graph"].items():
                torch.testing.assert_close(getattr(graph, key), all_lanes[lane:lane+1], rtol=0, atol=0)
            for key, tensor in rule.state_dict().items():
                torch.testing.assert_close(tensor, y["rule"][key][lane:lane+1], rtol=0, atol=0)
        credit = y["trace"][:, :, 3].sum(0).tolist()
        if credit != [1000 if arm == "mixed" else 0] * len(swap.COHORTS[cohort]):
            raise ValueError(f"Unexpected credit budget: {resume}")
        results[arm] = {"resume_sha256": digest(resume), "manifest_sha256": digest(manifest),
                        "step": step, "short_credited_by_birth": credit}
    return results


def main():
    torch.set_num_threads(1)
    results = {}
    for cohort in swap.COHORTS:
        results[cohort] = {
            "previous_worlds": verify_score_table(EVIDENCE / f"one-life-repair-{cohort}-2000.json", cohort, False),
            "confirmation": verify_score_table(EVIDENCE / f"one-life-repair-confirmation-{cohort}-2000.json", cohort, True),
            "experts": verify_experts(cohort),
            "repair_resume": verify_repair_resume(cohort),
        }
        print(f"verified {cohort}", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    io._write_json(REPORT, {"script_sha256": digest(Path(__file__)), "cohorts": results})


if __name__ == "__main__":
    main()
