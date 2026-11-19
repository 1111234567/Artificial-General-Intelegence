"""Bind completed native study artifacts without loading large diagnostic tables."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import ijson
from ijson.common import ObjectBuilder


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def report_header(path):
    builder = ObjectBuilder()
    with Path(path).open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == "" and event == "map_key" and value == "inputs":
                builder.event("end_map", None)
                return builder.value
            builder.event(event, value)
    raise ValueError("missing report input bindings")


def finalize(evidence):
    evidence = Path(evidence).resolve()
    aggregate = evidence / "aggregate-v1"
    header = report_header(aggregate / "report.json")
    if header.get("complete") is not True or header.get("audit", {}).get("status") != "passed":
        raise ValueError("aggregate is not complete and passed")
    if header["counts"]["evaluation_forks"] != 1860 or header["counts"]["training_runs"] != 30:
        raise ValueError("incomplete study population")
    artifacts = json.loads((aggregate / "artifacts.json").read_text(encoding="utf-8"))
    for name, expected in artifacts.items():
        if sha(aggregate / name) != expected:
            raise ValueError(f"aggregate artifact changed: {name}")
    suites = ET.parse(evidence / "tests-final.xml").getroot()
    cases = suites.findall(".//testcase")
    if not cases or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("regression tests incomplete or unsuccessful")
    queues = {}
    for study in ("main-v1", "coverage-v1"):
        for kind in ("training", "evaluation", "baseline"):
            path = ROOT / "data/lifelong" / study / f"{kind}-execution.json"
            state = json.loads(path.read_text(encoding="utf-8"))
            expected = 30 if kind == "evaluation" or (study == "main-v1" and kind == "training") else 0
            if not state.get("complete") or len(state["jobs"]) != expected or any(
                row.get("returncode") != 0 or row.get("missing_outputs") for row in state["jobs"].values()
            ):
                raise ValueError(f"incomplete execution: {path}")
            queues[f"{study}/{kind}"] = dict(path=str(path), sha256=sha(path), successful_jobs=expected,
                limits_history=state.get("limits_history", []), resume_history=state.get("resume_history", []))
    figure_data = evidence / "figures/figure-data.json"
    if not figure_data.exists() or len(list((evidence / "figures").glob("*.png"))) < 3:
        raise ValueError("scientific figures missing")
    now = datetime.now(timezone.utc)
    value = dict(format="lifelong-study-completion-v1", complete=True,
        completed_utc=now.isoformat(), completed_asia_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        counts=header["counts"], independent_training_seeds=3, unique_tests=len(cases), audit=header["audit"],
        execution=queues, aggregate_header=header, aggregate_artifacts=artifacts,
        raw_bindings_in=str(aggregate / "report.json"),
        notes=["Protocol date is2026-10-29; completion date records actual finalization.",
               "Regression cases and invariant checks are implementation evidence, not independent experimental seeds."])
    paths = list(evidence.glob("*.json")) + list(evidence.glob("*.xml")) + list(evidence.glob("*.md")) + list(evidence.glob("*.py"))
    paths += list((evidence / "figures").glob("*")) + list((ROOT / "experiments/lifelong").glob("*.py"))
    paths += [ROOT / name for name in ("experiments/lifelong/README.md", "docs/lifelong-experiment-2026-10-29.md",
        "README.md", "agi.md", "CHANGELOG.md", "docs/design.md", "docs/experiments.md",
        "docs/reference.md", "experiments/README.md")]
    value["file_hashes"] = {str(path): sha(path) for path in sorted(set(paths))
        if path.is_file() and path.name != "completion.json"}
    target = evidence / "completion.json"
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(path=str(target), tests=len(cases), counts=header["counts"], audit=header["audit"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/lifelong-2026-10-29")
    finalize(parser.parse_args().evidence)
