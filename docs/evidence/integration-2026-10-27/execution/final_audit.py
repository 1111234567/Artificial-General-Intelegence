"""Read-only final state and input audit after the finite formal queue ends."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.continuity.run import atomic_json, state_digest

FOLDER = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    path = FOLDER / "execution-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = Path(summary["execution"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {"all_36_training_complete": summary["all_36_completed"],
              "all_72_evaluation_jobs_complete": summary["all_evaluations_completed"],
              "raw_forks_288": summary["evaluation_records"] == 288,
              "no_execution_errors": not summary["errors"],
              "manifest_unchanged": sha256(manifest_path) == summary["execution"]["manifest_sha256"],
              "frozen_inputs_unchanged": summary["frozen_inputs_unchanged"],
              "all_origin_groups_paired": all(summary["same_seed_origin_pairing"].values()),
              "all_main10_event_groups_paired": all(summary["noncoverage_event_pairing"].values())}
    rows = summary["completed"]
    per_seed = []
    for seed in manifest["seeds"]:
        group = [item for item in rows if item["seed"] == seed]
        source = manifest["source_checkpoints"][str(seed)]
        source_state = torch.load(source["path"], map_location="cpu", weights_only=False)
        old_readout = {name: value for name, value in source_state["model"].items() if name != "recurrent"}
        pure = next(item for item in group if item["mode"] == "coact_only_mix")
        pure_state = torch.load(pure["checkpoint"], map_location="cpu", weights_only=False)
        new_readout = {name: value for name, value in pure_state["continuation"]["model"].items() if name != "recurrent"}
        frozen = next(item for item in group if item["mode"] == "frozen")
        sampled = [item for item in group if item["mode"] in {
            "replay64", "coact_live", "coact_mix", "coact_mix_shuffled", "coact_only_mix"}]
        values = {"seed": seed, "twelve_final_runs": len(group) == 12,
                  "source_checkpoint_unchanged": sha256(source["path"]) == source["sha256"],
                  "source_report_unchanged": sha256(Path(source["path"]).with_name("report.json")) == source["report_sha256"],
                  "origin_semantic_hashes_unique": len({item["origin"]["semantic_hash"] for item in group}) == 1,
                  "frozen_model_unchanged": frozen["final_state_hashes"]["model"] == frozen["origin"]["model_hash"],
                  "frozen_adam_unchanged": frozen["final_state_hashes"]["optimizer"] == frozen["origin"]["optimizer_hash"],
                  "pure_local_adam_unchanged": pure["final_state_hashes"]["optimizer"] == pure["origin"]["optimizer_hash"],
                  "pure_local_encoder_readout_unchanged": state_digest(old_readout) == state_digest(new_readout),
                  "all_old_banks_and_their_rng_unchanged": all(item["final_state_hashes"]["replay"] == item["origin"]["replay_hash"] for item in group),
                  "all_five_old_samplers_end_identically": len({item["final_state_hashes"]["replay_rng"] for item in sampled}) == 1,
                  "all_final_checkpoint_files_unchanged": all(sha256(item["checkpoint"]) == item["checkpoint_sha256"] for item in group)}
        per_seed.append(values)
        checks[f"seed_{seed:03d}"] = all(value for key, value in values.items() if key != "seed")
    evaluations = summary["evaluations_completed"]
    checks["all_evaluation_files_unchanged"] = all(sha256(item["path"]) == item["sha256"] for item in evaluations)
    checks["all_raw_source_files_unchanged"] = all(sha256(ROOT / name) == digest for name, digest in manifest["source_file_hashes"].items())
    result = {"format": "integration-main-final-state-audit-v1", "manifest": str(manifest_path),
              "manifest_sha256": sha256(manifest_path), "execution_summary_sha256": sha256(path),
              "checks": checks, "per_seed": per_seed, "passed": all(checks.values())}
    destination = FOLDER / "final-audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    atomic_json(destination, result)
    print(json.dumps({"passed": result["passed"], "audit": str(destination), "sha256": sha256(destination)}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
