"""Score the ticket and coverage endpoints on the sealed old-task benchmark."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import noon_transfer as card, noon_transfer_archive as archive, noon_transfer_custody as custody


STAGE = 8000
MODELS = {
    "ticket_native": ("tickets", "native", 8000),
    "ticket_coin": ("tickets", "coin", 8000),
    "coverage_evidence": ("coverage", "evidence", 16000),
    "coverage_dose": ("coverage", "dose", 16000),
}
PAIRS = (
    ("ticket_native", "uniform"), ("ticket_coin", "uniform"), ("ticket_native", "ticket_coin"),
    ("coverage_evidence", "ticket_native"), ("coverage_dose", "ticket_native"),
    ("coverage_evidence", "coverage_dose"), ("coverage_evidence", "uniform"), ("coverage_dose", "uniform"),
)


def verified_sources(cohort, births, archived, model_specs):
    results = {}
    for name, (family, arm, _) in model_specs.items():
        folder = card.DATA / family / cohort
        run = folder / arm / "run.json"
        optimizer = folder / arm / "resume.pt"
        run_hash = archive.digest(run)
        manifest = archive.read(run)
        saved = torch.load(optimizer, map_location="cpu", weights_only=True)
        if (saved["plan"], saved["step"], manifest["births"]) != (run_hash, STAGE, births):
            raise ValueError(f"Unexpected endpoint: {optimizer}")
        fields = saved.get("graph", saved)
        planned = card.OUT / ("ticket-plan.json" if family == "tickets" else "coverage-plan.json")
        archive.verify(planned, manifest["plan_sha256"] if family == "tickets" else manifest["plan"])
        recorded = {}
        for index, b in enumerate(births):
            if family == "tickets":
                reference = archived["experts"][f"uniform/{b}"]
            else:
                reference = results["ticket_native"]["models"][str(b)]
            if manifest["parents"][str(b)] != reference["sha256"]:
                raise ValueError(f"Training parent differs at {family} {arm} birth {b}")
            archive.verify(card.ROOT / reference["path"], reference["sha256"])
            file = card.checkpoint.checkpoint_path(folder, "odor", arm, b, STAGE)
            graph, rule = card.checkpoint.load_checkpoint(folder, "odor", arm, b, STAGE)
            for key in card.capacity.GRAPH_FIELDS:
                torch.testing.assert_close(getattr(graph, key), fields[key][index:index+1], rtol=0, atol=0)
            for key, value in rule.state_dict().items():
                torch.testing.assert_close(value, saved["rule"][key][index:index+1], rtol=0, atol=0)
            recorded[str(b)] = custody.record(file)
        result = {"models": recorded, "optimizer": custody.record(optimizer),
                  "manifest": custody.record(run), "training_plan": custody.record(planned)}
        if family == "coverage":
            gain = folder / arm / f"gain-{STAGE}.npz"
            with np.load(gain) as z:
                np.testing.assert_array_equal(z["births"], births)
                np.testing.assert_array_equal(z["gain"], saved["gain"].numpy())
            result["paid_menu_gain_only"] = custody.record(gain)
        results[name] = result
    return results


def metrics(raw):
    result = card.object_metrics(raw)
    for condition in card.paid.SUBSETS:
        scores = raw[f"snapshot/per_world/{condition}"]
        result.update({f"snapshot/{condition}/{k}": v
                       for k, v in zip(card.swap.SPEC["new"], scores.T, strict=True)})
    return result


@torch.no_grad()
def run(cohort, ticket_only):
    torch.set_num_threads(1)
    plan, archived = map(archive.read, (card.PLAN, archive.REPORT))
    cohort_plan = plan["cohorts"][cohort]
    births, seeds, counts = cohort_plan["births"], cohort_plan["seeds"], plan["worlds"]
    model_specs = {name: spec for name, spec in MODELS.items() if not ticket_only or spec[0] == "tickets"}
    sources = verified_sources(cohort, births, archived["cohorts"][cohort], model_specs)
    baseline = card.OUT / f"transfer-{cohort}-0.json"
    initial = archive.read(baseline)
    if (initial["births"], initial["seeds"], initial["plan_sha256"]) != (births, seeds, archive.digest(card.PLAN)):
        raise ValueError(f"Unexpected transfer baseline {baseline}")
    world = card.capacity.make_tape("odor", seeds["objects"], counts["objects"], 7)
    scene = card.six.make(seeds["snapshot"], counts["snapshot"])
    code = {**card.code_fingerprints(), Path(__file__).name: archive.digest(Path(__file__))}
    signature = json.dumps({"code": code, "base_plan": archive.digest(card.PLAN)}, sort_keys=True)
    traces = {"uniform": initial["hashes"]["uniform"]}
    groups = {}
    for model in ("uniform", *model_specs):
        rows = []
        if model != "uniform":
            family, arm, _ = model_specs[model]
            folder = card.DATA / family / cohort
            traces[model] = {}
        for birth in births:
            b = str(birth)
            if model == "uniform":
                entry = traces[model][b]
                path = card.ROOT / entry["trace"]
                archive.verify(path, entry["sha256"])
            else:
                source = sources[model]["models"][b]
                path = card.DATA / "transfer" / cohort / f"{model}-{STAGE}-{birth}.npz"
                if not path.is_file():
                    graph, rule = card.checkpoint.load_checkpoint(folder, "odor", arm, birth, STAGE)
                    raw = card.object_trials(graph, rule, world, seeds["objects"])
                    raw.update({f"snapshot/{k}": v for k, v in card.swap.new_trial(graph, rule, scene).items()})
                    archive.verify(card.ROOT / source["path"], source["sha256"])
                    io._write_raw(path, signature=signature, source_sha256=source["sha256"], **raw)
                traces[model][b] = {"trace": path.relative_to(card.ROOT).as_posix(), "sha256": archive.digest(path),
                                    "sources": [source]}
            with np.load(path) as raw:
                if model != "uniform" and (str(raw["signature"]), str(raw["source_sha256"])) != (signature, source["sha256"]):
                    raise ValueError(f"Supplemental cache does not match {model}: {path}")
                rows.append(metrics(raw))
            print(cohort, model, birth, flush=True)
        groups[model] = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    intervals = {key: card.stats.Crossed(len(births), counts[key], seeds[key] + 367, 4000)
                 for key in ("objects", "snapshot")}
    by_model = {model: {k: intervals[k.split("/")[0]](v) for k, v in rows.items()} for model, rows in groups.items()}
    paired = {f"{a} minus {b}": {k: intervals[k.split("/")[0]](v - groups[b][k]) for k, v in groups[a].items()}
              for a, b in PAIRS if a in groups and b in groups}
    artifact = {"cohort": cohort, "births": births, "seeds": seeds, "worlds": counts,
                "updates_after_uniform": {"uniform": 0, **{m: steps for m, (_, _, steps) in model_specs.items()}},
                "deployment": "Standard native graph and write rule with no paid menu or coverage channel. The paid-menu gain is checked only for checkpoint custody.",
                "baseline": custody.record(baseline), "source_sha256": code, "model_custody": sources,
                "models": by_model, "paired": paired, "hashes": traces}
    prefix = "ticket" if ticket_only else "ticket-coverage"
    io._write_json(card.OUT / f"transfer-{prefix}-{cohort}-8000.json", artifact)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=tuple(card.swap.COHORTS))
    parser.add_argument("--ticket-only", action="store_true")
    args = parser.parse_args()
    run(args.cohort, args.ticket_only)
