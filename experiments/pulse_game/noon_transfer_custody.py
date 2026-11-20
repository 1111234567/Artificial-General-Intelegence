"""Check that a complete transfer stage matches the sealed optimizer archives."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as io
from . import local_trace_train as checkpoint
from . import noon_transfer as card
from . import noon_transfer_archive as archive


def record(path):
    return {"path": path.relative_to(card.ROOT).as_posix(), "sha256": archive.digest(path)}


def verify_all(stage, with_report, swaps):
    torch.set_num_threads(1)
    location = card.DATA / "core"
    plan, manifest = map(archive.read, (card.PLAN, location / "run.json"))
    manifest_hash = archive.digest(location / "run.json")
    if stage not in manifest["checkpoints"] or manifest["arms"] != plan["training_arms"]:
        raise ValueError("Requested stage or arm does not match the fixed training contract")
    archived = archive.read(archive.REPORT)["cohorts"]
    results = {}
    for cohort, protocol in plan["cohorts"].items():
        births = protocol["births"]
        if manifest["cohorts"][cohort] != births:
            raise ValueError(f"Unexpected training births: {cohort}")
        results[cohort] = {}
        for arm in manifest["arms"]:
            folder = location / cohort / "training" / arm
            optimizer = folder / f"optimizer-{stage:06d}.pt"
            trace = folder / f"trace-{stage:06d}.npz"
            saved = torch.load(optimizer, map_location="cpu", weights_only=True)
            expected = (manifest_hash, cohort, arm, stage)
            actual = tuple(saved[k] for k in ("manifest_sha256", "cohort", "arm", "step"))
            if actual != expected:
                raise ValueError(f"Unexpected optimizer identity: {optimizer}")
            counters = {int(v["step"]) for v in saved["optimizer"]["state"].values()}
            if counters != ({stage} if stage else set()):
                raise ValueError(f"Unexpected Adam update counters {counters}: {optimizer}")
            with np.load(trace) as z:
                np.testing.assert_array_equal(z["births"], births)
                np.testing.assert_array_equal(z["keys"], manifest["metrics"])
                np.testing.assert_array_equal(z["values"], saved["metrics"].numpy())
            items = {}
            for index, birth in enumerate(births):
                ref = archived[cohort]["experts"][f"uniform/{birth}"]
                if manifest["parent_sha256"][cohort][str(birth)] != ref["sha256"]:
                    raise ValueError(f"Parent protocol changed at birth {birth}")
                path = checkpoint.checkpoint_path(location / cohort, "odor", arm, birth, stage)
                graph, rule = checkpoint.load_checkpoint(location / cohort, "odor", arm, birth, stage)
                for key in ("weights", "bias", "present"):
                    torch.testing.assert_close(getattr(graph, key), saved[key][index:index+1], rtol=0, atol=0)
                for key, value in rule.state_dict().items():
                    torch.testing.assert_close(value, saved["rule"][key][index:index+1], rtol=0, atol=0)
                if stage == 0:
                    folder, tag, step = archive.model(cohort, "uniform")
                    parent = checkpoint.load_checkpoint(folder, "odor", tag, birth, step)
                    archive.assert_same_parameters((graph, rule), parent)
                items[str(birth)] = record(path)
            results[cohort][arm] = {"optimizer": record(optimizer), "trace": record(trace), "models": items}
    reports = {}
    if with_report:
        for cohort, arms in results.items():
            suffix = "-swaps" if swaps else ""
            path = card.OUT / f"transfer-{cohort}-{stage}{suffix}.json"
            scored = archive.read(path)
            if (scored["births"] != plan["cohorts"][cohort]["births"] or
                    scored["plan_sha256"] != archive.digest(card.PLAN)):
                raise ValueError(f"Unexpected transfer report identity: {path}")
            for arm, released in arms.items():
                for birth, entry in scored["hashes"][f"{arm}_{stage}"].items():
                    if entry["sources"] != [released["models"][birth]]:
                        raise ValueError(f"Transfer trace does not use the released optimizer stage: {entry}")
                    archive.verify(card.ROOT / entry["trace"], entry["sha256"])
            if swaps:
                for b in scored["births"]:
                    birth = str(b)
                    start = archived[cohort]["experts"][f"uniform/{birth}"]
                    trained = arms["native"]["models"][birth]
                    for name, expected in (("native_body_uniform_rule", [trained, start]),
                                           ("uniform_body_native_rule", [start, trained])):
                        row = scored["hashes"][f"{name}_{stage}"][birth]
                        if row["sources"] != expected:
                            raise ValueError(f"Swapped checkpoint ancestry does not match the released stage: {row}")
                        archive.verify(card.ROOT / row["trace"], row["sha256"])
            reports[cohort] = record(path)
    output = dict(stage=stage, manifest=record(location / "run.json"), script=record(Path(__file__)),
                  reports=reports, checkpoints=results, updates_per_birth=stage,
                  no_review_updates_per_birth=stage // manifest["config"]["retention_interval"])
    io._write_json(card.OUT / f"transfer-stage-{stage}-custody.json", output)
    print(f"Verified {stage}: {sum(len(arms) for arms in results.values())} optimizers, "
          f"{sum(len(arm['models']) for arms in results.values() for arm in arms.values())} checkpoints", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int)
    parser.add_argument("--with-report", action="store_true")
    parser.add_argument("--swaps", action="store_true")
    options = parser.parse_args()
    if options.swaps and not options.with_report:
        parser.error("--swaps requires --with-report")
    verify_all(options.stage, options.with_report, options.swaps)
