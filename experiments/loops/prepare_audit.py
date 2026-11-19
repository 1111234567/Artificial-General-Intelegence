"""Freeze the finite existing-model audit before observing its outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from experiments.continuity.run import atomic_json
from experiments.credit.run import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def prepare(output, raw):
    output, raw = Path(output).resolve(), Path(raw).resolve()
    protocol = output / "audit-protocol-v1.json"
    if protocol.exists():
        raise FileExistsError(protocol)
    sources = [ROOT / "experiments" / folder / name for folder, names in {
        "loops": ["__init__.py", "audit_run.py", "structure.py", "interventions.py", "prepare_audit.py"],
        "continuity": ["model.py", "world.py", "run.py", "evaluate.py"],
        "integration": ["run.py", "diagnose.py"],
        "credit": ["run.py", "local.py", "batch.py", "windows_runner.py"],
    }.items() for name in names]
    hashes = {str(path): file_sha256(path) for path in sources}
    checkpoints = []
    for seed in (0, 1, 2):
        for mode in ("A", "bp64", "bp_replay"):
            path = ROOT / (f"data/continuity/main-20261027-v1/seed_{seed:03d}/online128/checkpoint.pt"
                if mode == "A" else f"data/credit/main-v1/seed_{seed:03d}/{mode}/checkpoint.pt")
            checkpoints.append({"id": f"seed_{seed:03d}_{mode}", "seed": seed, "mode": mode,
                                "path": str(path), "sha256": file_sha256(path)})
    plan = {"format": "loops-existing-audit-protocol-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": checkpoints, "sources": hashes, "mask_seeds": [11, 29, 47],
        "conditions": ["iid", "long_sparse", "longer"], "steps": 1024, "dt": .25,
        "state_samples": 16, "device": "cpu", "parameter_updates": 0,
        "hypotheses": [
            "The existing dense recurrent field already has distinct-neuron directed cycles of many lengths.",
            "Removing all distinct-neuron feedback may change autonomous completion; matched random deletion is a damage comparator, not a perfect topology-only intervention.",
            "Instantaneous state-conditioned weighted closed walks describe local connectivity, not the causal share or duration of a simple cycle."],
        "interpretation": {
            "dag": "Keep self recurrence and a random-order directed acyclic off-diagonal graph.",
            "random": "Same per-target retained edge counts as DAG; outgoing degree, spectrum, support and forcing are not matched.",
            "matched": "DAG/random rows shrink to their shared smaller off-diagonal L2 norm, without amplifying weights; self weights remain exact.",
            "quarter": "Interpolate 0.75 original + 0.25 matched endpoint. These gentler interventions retain cycles; endpoint row norm matching does not imply interpolated row norm matching.",
            "continuity": "All forks inherit full h/world/clocks/pending contracts. Only copied recurrent parameters change; these forks are counterfactuals, not resets or training.",
            "statistics": "Average three fixed mask draws within each trained seed; summarize across three trained seeds. Actors and mask draws are not independent training replicates.",
            "no_claim": "No inference that a particular cycle length causes a behavior, or that no output under severe deletion means no memory."}}
    atomic_json(protocol, plan)
    jobs = []
    for item in checkpoints:
        for seed in plan["mask_seeds"]:
            name = f"{item['id']}_mask_{seed}"
            target = raw / f"{name}.json"
            jobs.append({"id": name, "command": [sys.executable, "-m", "experiments.loops.audit_run",
                "--protocol", str(protocol), "--source-id", item["id"], "--mask-seed", str(seed),
                "--output", str(target)], "outputs": [str(target)], "log": str(raw / "logs" / f"{name}.log")})
    atomic_json(output / "audit-commands-v1.json", {"cwd": str(ROOT),
        "file_hashes": {**hashes, str(protocol): file_sha256(protocol)}, "jobs": jobs})
    print(f"Frozen {len(checkpoints)} checkpoints, {len(jobs)} jobs, 540 autonomous/counterfactual forks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.raw)
