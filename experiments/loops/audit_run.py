"""Predeclared graph interventions on complete, frozen historical continuations.

Only parameter copies are intervened on. Hidden state, world, clocks and pending
contracts are inherited, and the original own-action observer supplies scoring.
Deleting many edges is a destructive counterfactual, not an autonomous gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.continuity.evaluate import _preserve_process_state
from experiments.continuity.run import atomic_json, state_digest
from experiments.credit.run import file_sha256, load_checkpoint
from experiments.integration.diagnose import diagnose_fork
from experiments.loops.interventions import build_interventions, clone_with_recurrent
from experiments.loops.structure import analyze_states, analyze_weight


def verify_files(files):
    for name, expected in files.items():
        if file_sha256(name) != expected:
            raise ValueError(f"frozen input/source changed: {name}")


def run_job(protocol_path, source_id, mask_seed, output):
    protocol_path, output = Path(protocol_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol_hash = file_sha256(protocol_path)
    plan = json.loads(protocol_path.read_text(encoding="utf-8"))
    if plan["format"] != "loops-existing-audit-protocol-v1" or mask_seed not in plan["mask_seeds"]:
        raise ValueError("unknown diagnostic protocol or mask draw")
    verify_files(plan["sources"])
    item = next(row for row in plan["checkpoints"] if row["id"] == source_id)
    path, expected = item["path"], item["sha256"]
    if file_sha256(path) != expected:
        raise ValueError("historical checkpoint changed")
    with _preserve_process_state():
        trainer = load_checkpoint(path, device="cpu")
        before = state_digest(trainer.semantic_state())
        weights, metadata = build_interventions(trainer.model.recurrent, seed=mask_seed)
        names = list(weights) if mask_seed == plan["mask_seeds"][0] else [
            name for name in weights if name not in ("none", "diagonal_only")]
        records = []
        for condition in plan["conditions"]:
            shared = None
            for name in names:
                model = clone_with_recurrent(trainer.model, weights[name])
                result = diagnose_fork(model, trainer.world, trainer.h,
                    condition=condition, steps=plan["steps"], dt=plan["dt"], device="cpu")
                history = {key: result[key] for key in (
                    "start_world_state_hash", "start_hidden_hash", "start_scheduler_hash",
                    "end_scheduler_hash", "end_event_hash", "primary_eligible_issued")}
                history["primary_fingerprint"] = result["primary"]["request_fingerprint"]
                if shared is not None and history != shared:
                    raise AssertionError("graph interventions changed exogenous requests or initial state")
                shared = history
                result.update(graph_intervention=name, mask_seed=mask_seed,
                    graph_parameter_hash=state_digest(model.state_dict()),
                    graph_action_policy="unmodified_model" if name == "none" else "parameter_intervention_counterfactual")
                records.append(result)
                print(json.dumps({"source": source_id, "mask_seed": mask_seed,
                    "condition": condition, "intervention": name,
                    "strict": result["primary"]["clean_completion_rate"]}), flush=True)
        structure = None
        if mask_seed == plan["mask_seeds"][0]:
            structure = {"weight": analyze_weight(trainer.model.recurrent),
                "state": analyze_states(trainer.model.recurrent, trainer.h[:plan["state_samples"]]),
                "sampling": "First declared actors at the stored continuation boundary; not a trajectory distribution or independent model seeds."}
        after = state_digest(trainer.semantic_state())
        if before != after:
            raise AssertionError("diagnostic changed the live historical continuation")
    verify_files(plan["sources"])
    if file_sha256(protocol_path) != protocol_hash:
        raise AssertionError("diagnostic protocol changed during execution")
    if file_sha256(path) != expected:
        raise AssertionError("historical checkpoint bytes changed")
    result = {"format": "loops-existing-audit-v1", "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_hash, "source": item, "mask_seed": mask_seed,
        "source_semantic_hash_before": before, "source_semantic_hash_after": after,
        "intervention_metadata": metadata, "records": records, "structure": structure,
        "source_files": plan["sources"], "checkpoint_bytes_unchanged": True}
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--mask-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_job(args.protocol, args.source_id, args.mask_seed, args.output)
