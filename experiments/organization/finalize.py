"""Bind the two completed studies, regression evidence, and scientific figures."""
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import ijson


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def meta(path):
    result = {}
    for key in ("format", "complete", "audit", "counts", "protocol_sha256"):
        with Path(path).open("rb") as stream:
            result[key] = next(ijson.items(stream, key, use_float=True))
    if not result["complete"] or result["audit"]["status"] != "passed" or result["audit"]["errors"]:
        raise ValueError("incomplete or failed strict aggregate")
    return result


def finalize(evidence):
    evidence = Path(evidence).resolve()
    summaries, aggregate_bindings = {}, {}
    for name in ("acquisition-aggregate-v1", "aggregate-v1"):
        folder = evidence / name
        summaries[name] = meta(folder / "report.json")
        bindings = json.loads((folder / "artifacts.json").read_text(encoding="utf-8"))
        for filename, expected in bindings.items():
            if sha(folder / filename) != expected:
                raise ValueError(f"aggregate artifact changed: {filename}")
        aggregate_bindings[name] = dict(manifest_sha256=sha(folder / "artifacts.json"), files=bindings)
    acquisition, main = summaries["acquisition-aggregate-v1"]["counts"], summaries["aggregate-v1"]["counts"]
    if (acquisition["new_training_runs"], acquisition["new_evaluation_forks"],
        acquisition["borrowed_old_evaluation_forks"], main["training_runs"], main["frozen_forks"]) != (9, 216, 216, 27, 648):
        raise ValueError("unexpected scientific population")
    suites = ET.parse(evidence / "tests-final.xml").getroot()
    cases = suites.findall(".//testcase")
    if not cases or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("final regression must pass with no skipped cases")
    execution = {}
    for study, number in (("acquisition-v1", 9), ("main-v1", 27)):
        for kind in ("training", "evaluation", "baseline"):
            path = ROOT / "data/organization" / study / f"{kind}-execution.json"
            state = json.loads(path.read_text(encoding="utf-8"))
            expected = 0 if kind == "baseline" else number
            if not state["complete"] or len(state["jobs"]) != expected or any(
                row["returncode"] != 0 or row.get("missing_outputs") for row in state["jobs"].values()):
                raise ValueError(f"incomplete execution: {path}")
            execution[f"{study}/{kind}"] = dict(path=str(path), sha256=sha(path), successful_jobs=expected,
                limits_history=state.get("limits_history", []), resume_history=state.get("resume_history", []))
    figure_folder = evidence / "figures"
    figures = json.loads((figure_folder / "artifacts.json").read_text(encoding="utf-8"))
    for path, expected in figures["inputs"].items():
        if sha(path) != expected:
            raise ValueError("figure input changed")
    for filename, expected in figures["outputs"].items():
        if sha(figure_folder / filename) != expected:
            raise ValueError("figure output changed")
    if len(list(figure_folder.glob("*.png"))) < 3:
        raise ValueError("scientific figure set incomplete")
    journal = json.loads((evidence / "isolated-benchmark-v2-process-journal.json").read_text(encoding="utf-8"))
    if journal["active"] or len(journal["resumed"]) + len(journal.get("exited_before_resume", [])) != len(journal["suspended"]):
        raise ValueError("operational benchmark left unresolved suspended processes")
    now = datetime.now(timezone.utc)
    value = dict(format="organization-study-completion-v1", complete=True,
        completed_utc=now.isoformat(), completed_asia_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        counts=dict(new_training_runs=36, independent_training_seeds=3, new_full_checkpoints=144,
                    new_frozen_forks=864, borrowed_old_forks=216),
        new_unique_regression_tests=len(cases), aggregates=summaries, execution=execution,
        aggregate_artifact_bindings=aggregate_bindings, figure_artifact_sha256=sha(figure_folder / "artifacts.json"),
        notes=["Borrowed old forks are not new training runs or independent seeds.",
            "The acquisition execution retry used the same saved full state, seed, source and budget; failure evidence retained.",
            "A mechanical CPU preparation was never executed and was archived before formal preparation.",
            "Operational isolated benchmark preserved states but its suspension can affect wall-time counters.",
            "Frozen execution source remains separate from analysis and documentation."])
    paths = [p for p in evidence.iterdir() if p.is_file() and p.suffix in (".json", ".xml", ".md", ".py", ".txt")]
    paths += list((evidence / "unexecuted-mechanical-cpu-prepare").glob("*.json"))
    paths += list(figure_folder.glob("*")) + list((ROOT / "experiments/organization").glob("*.py"))
    paths += [ROOT / name for name in ("experiments/organization/README.md", "docs/organization-experiment-2026-10-30.md",
        "README.md", "agi.md", "CHANGELOG.md", "docs/design.md", "docs/experiments.md",
        "docs/reference.md", "experiments/README.md")]
    value["file_hashes"] = {str(path): sha(path) for path in sorted(set(paths))
        if path.is_file() and path.name != "completion.json"}
    target = evidence / "completion.json"
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(path=str(target), counts=value["counts"], tests=len(cases),
        checks={key: row["audit"]["checks"] for key, row in summaries.items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/organization-2026-10-30")
    finalize(parser.parse_args().evidence)
