"""Seal or replay the ACK/REJ-trained address on the original independent worlds."""

from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from . import research_audit as audit, ring_spikes
from .legacy_grounded import DEFAULT_GRAPH, GroundedAgent, Sense, Setting, base_config, play, read, sense, sha
from .legacy_grounded_outcomes import Address


def seal(graph, original_report, output):
    peer = original_report.parent
    results = read(original_report)
    prior = read(peer / "legacy_grounded-selected.json")
    if sha(graph) != results["graph_sha256"] or sha(graph) != prior["source_graph_sha256"]:
        raise ValueError("The frozen graph has changed")
    f = sense(graph)
    branches = {}
    for name in ("actual", "shuffled", "initial"):
        path = peer / f"legacy_grounded-outcomes-{name}.pt"
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["source_graph_sha256"] != results["graph_sha256"]:
            raise ValueError("An address was trained against a different graph")
        if path.stat().st_mtime_ns >= original_report.stat().st_mtime_ns:
            raise ValueError("Address checkpoint did not precede the closed-loop report")
        branches[name] = {"state": record["model"], "frozen_artifact_sha256": sha(path),
                          "frozen_artifact_mtime_ns": path.stat().st_mtime_ns}
    selection = min(results["learning_curves"]["actual"], key=lambda step: step["actual_outcome_brier_signed"])
    protocol = {key: rows["roots"] for key, rows in results["sets"].items()}
    if set(protocol) != {"original32", "new32"} or any(len(roots) != 32 for roots in protocol.values()):
        raise ValueError("Expected two complete frozen 32-world confirmations")
    training = results["recorded_legal_exploration"]
    namespaces = {
        **protocol, "receipt_training": training["train_seeds"], "receipt_development": training["development_seeds"],
        "actuator_development": prior["development_seeds"], "public_calibration": prior["calibration_seeds"],
    }
    values = list(namespaces.values())
    if any(len(set(x)) != len(x) for x in values) or sum(map(len, values)) != len(set(sum(values, []))):
        raise ValueError("Evaluation or training world roots overlap")
    record = {
        "format": "legacy-grounded-public-receipt-v1",
        "frozen_sense": {"dynamics": torch.from_numpy(f.dynamics.copy()), "afferents": torch.from_numpy(f.afferents.copy()),
                         "odor": torch.from_numpy(f.odor.copy()), "bias": torch.from_numpy(f.bias.copy())},
        "source_graph_sha256": results["graph_sha256"],
        "addresses": branches, "original_rls_and_motor_config": asdict(base_config()),
        "motor_setting": results["setting"], "world_namespaces": namespaces,
        "receipt_selection": {"criterion": "mean squared signed prediction error before the actual receipt",
                              "best_development_step": selection["step"],
                              "best_development_error": selection["actual_outcome_brier_signed"],
                              "train_actual_receipts": training["train_actually_received_labels"],
                              "development_actual_receipts": training["dev_actually_received_labels"],
                              "selection_and_save_precede_both_closed_loop_splits": True},
        "original_closed_loop_report_sha256": sha(original_report),
    }
    torch.save(record, output)
    print("sealed", output, sha(output), flush=True)


def replay(checkpoint, original, destination, arm):
    r = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if r["format"] != "legacy-grounded-public-receipt-v1":
        raise ValueError("Unsupported release format")
    os = read(original)
    if sha(original) != r["original_closed_loop_report_sha256"]:
        raise ValueError("Archived reference report no longer matches")
    model = r["addresses"][arm]["state"]
    address = Address(model["mean"], model["std"], dimension=len(model["second.bias"]))
    address.load_state_dict(model)
    k = address.export()
    f = r["frozen_sense"]
    graph = Sense(*(f[key].numpy() for key in ("dynamics","afferents","odor","bias")))
    cfg = ring_spikes.Config(**r["original_rls_and_motor_config"])
    setting = Setting(**r["motor_setting"])
    measured = {}
    for name in ("original32", "new32"):
        roots = r["world_namespaces"][name]
        build = lambda seeds: GroundedAgent(cfg, graph, None, setting, seeds, learned_key=k)
        rows = [row for start in range(0,len(roots),8) for row in play(roots[start:start+8],build)]
        reference = os["sets"][name]["rows"][arm]
        if rows != reference:
            raise ValueError(f"The released policy is not exactly reproducible in {name}")
        measured[name] = {"matches_original_rows_exactly": True,
                          "strict": float(np.mean([row["strict"] for row in rows])),
                          "pair_strict_rate": float(np.mean([row["pair_strict_rate"] for row in rows])),
                          "roots": roots, "rows": rows}
        print(name, arm, measured[name]["strict"], measured[name]["pair_strict_rate"], flush=True)
    destination.write_text(json.dumps({"checkpoint_sha256":sha(checkpoint),"arm":arm,"sets":measured},
                                       indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("seal","replay"), required=True)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--checkpoint", type=Path, default=audit.OUT/"legacy_grounded-release.pt")
    parser.add_argument("--original", type=Path, default=audit.OUT/"legacy_grounded-outcomes.json")
    parser.add_argument("--output", type=Path, default=audit.OUT/"legacy_grounded-replay.json")
    parser.add_argument("--arm", choices=("actual","shuffled","initial"), default="actual")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.step=="seal":
        seal(args.graph,args.original,args.checkpoint)
    else:
        replay(args.checkpoint,args.original,args.output,args.arm)


if __name__ == "__main__":
    main()
