"""Bounded fixed-input initial-state sensitivity audit; never trains or acts.

Exact and stage-wise diagonal tangents follow the same unchanged Heun primal.
The checkpoint's final weights and recorded observations are held fixed. This
conditional dynamics audit cannot establish task memory or policy robustness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "credit-fixed-input-initial-state-tangent-v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}/{tuple(value.shape)}".encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def unit_directions(actors, directions, width, generator, dtype=torch.float64):
    """Private-generator draws only; no process RNG consumption."""
    values = torch.randint(0, 2, (actors, directions, width), generator=generator, device="cpu")
    return values.to(dtype).mul_(2).sub_(1).div_(math.sqrt(width))


@torch.no_grad()
def advance_tangents(weight, encoder, hidden, observation, exact, diagonal, dt, max_step=.25):
    """One real pulse/flow step; inputs are never mutated.

    Tangents have shape [actor,direction,hidden]. Only derivatives with respect
    to the initial hidden state are propagated; W/U/observations are constants.
    """
    elapsed, limit = float(dt), float(max_step)
    if (not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(limit) or limit <= 0
        or hidden.ndim != 2 or weight.shape != (hidden.shape[1], hidden.shape[1])
        or encoder.ndim != 2 or encoder.shape[0] != hidden.shape[1]
        or observation.shape != (hidden.shape[0], encoder.shape[1])
        or exact.ndim != 3 or exact.shape != diagonal.shape
        or exact.shape[0] != hidden.shape[0] or exact.shape[-1] != hidden.shape[1]):
        raise ValueError("invalid trajectory/tangent dimensions or solver increment")
    for value in (weight, encoder, observation, exact, diagonal):
        if value.device != hidden.device or value.dtype != hidden.dtype:
            raise ValueError("primal and tangents must share dtype/device")
    substeps = max(1, math.ceil(elapsed/limit))
    delta = elapsed/substeps
    h, v, q = hidden, exact, diagonal
    diagonal_weight = weight.diag()
    for _ in range(substeps):
        rate = h.tanh()
        derivative = 1.-rate.square()
        first = -h+F.linear(rate, weight)
        trial = h+delta*first
        trial_rate = trial.tanh()
        trial_derivative = 1.-trial_rate.square()
        first_v = -v+F.linear(derivative[:, None, :]*v, weight)
        trial_v = v+delta*first_v
        second_v = -trial_v+F.linear(trial_derivative[:, None, :]*trial_v, weight)
        a, a_trial = -1.+diagonal_weight*derivative, -1.+diagonal_weight*trial_derivative
        first_q = a[:, None, :]*q
        trial_q = q+delta*first_q
        second_q = a_trial[:, None, :]*trial_q
        v = v+.5*delta*(first_v+second_v)
        q = q+.5*delta*(first_q+second_q)
        second = -trial+F.linear(trial_rate, weight)
        h = h+.5*delta*(first+second)
    # Additive encoder pulse: derivative with respect to initial h is identity.
    return h+F.linear(observation, encoder), v, q


def pair_statistics(exact, diagonal):
    exact_norm = torch.linalg.vector_norm(exact, dim=-1)
    diagonal_norm = torch.linalg.vector_norm(diagonal, dim=-1)
    product = exact_norm*diagonal_norm
    cosine = (exact*diagonal).sum(-1)/product.clamp_min(torch.finfo(exact.dtype).tiny)
    return {"exact_l2": exact_norm, "diagonal_l2": diagonal_norm,
            "cosine": torch.where(product > 0, cosine.clamp(-1., 1.), torch.nan),
            "norm_ratio": torch.where(exact_norm > 0, diagonal_norm/exact_norm.clamp_min(torch.finfo(exact.dtype).tiny), torch.nan)}


def descriptive(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"count": len(values), "mean": float(tensor.mean()), "median": float(tensor.quantile(.5)),
            "q25": float(tensor.quantile(.25)), "q75": float(tensor.quantile(.75)),
            "min": float(tensor.min()), "max": float(tensor.max())}


def aggregate(records, horizons):
    metrics = ("exact_hidden_l2", "diagonal_hidden_l2", "hidden_cosine", "hidden_norm_ratio",
               "exact_tanh_feature_l2", "diagonal_tanh_feature_l2", "tanh_feature_cosine", "tanh_feature_norm_ratio")
    result = []
    for horizon in horizons:
        rows = [row for row in records if row["horizon_ticks"] == horizon]
        result.append({"horizon_ticks": horizon, "measurements": len(rows),
            "metrics": {name: descriptive(row[name] for row in rows) for name in metrics},
            "exact_hidden_gain_above_one_count": sum(row["exact_hidden_l2"] > 1. for row in rows),
            "exact_feature_gain_above_one_count": sum(row["exact_tanh_feature_l2"] > 1. for row in rows),
            "by_block": [{"block_index": block, "metrics": {name: descriptive(row[name] for row in rows if row["block_index"] == block)
                         for name in metrics}} for block in sorted({row["block_index"] for row in rows})]})
    return result


def _selected_fingerprint(state, blocks, actors):
    return {"model": {name: tensor_sha256(value) for name, value in state["model"].items()},
            "bank": {str(index): {"start_h": tensor_sha256(state["replay"]["blocks"][index]["start_h"][actors]),
                                  "observations": tensor_sha256(state["replay"]["blocks"][index]["observations"][:, actors]),
                                  "elapsed": list(state["replay"]["blocks"][index]["elapsed"])} for index in blocks}}


@torch.no_grad()
def run_audit(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    folder = protocol_path.parent
    if (folder/"summary.json").exists():
        raise FileExistsError("audit already has a summary; preserve its predeclared evidence")
    protocol_hash = file_sha256(protocol_path)
    plan = json.loads(protocol_path.read_text(encoding="utf-8"))
    if plan["format"] != "credit-fixed-input-initial-state-tangent-protocol-v1" or plan["dtype"] != "float64" or plan["device"] != "cpu":
        raise ValueError("unsupported tangent audit protocol")
    checkpoint = Path(plan["source_checkpoint"])
    before_checkpoint = file_sha256(checkpoint)
    if before_checkpoint != plan["source_checkpoint_sha256"]:
        raise ValueError("independent seed101 source bytes changed")
    before_sources = {name: file_sha256(name) for name in plan["frozen_byte_hashes_before"]}
    if before_sources != plan["frozen_byte_hashes_before"]:
        raise ValueError("frozen runtime/protocol changed since predeclaration")
    own_paths = [Path(__file__), Path(__file__).with_name("test_tangent_audit.py")]
    own_sources = {str(path.resolve()): file_sha256(path) for path in own_paths}
    process_rng = torch.get_rng_state().clone()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (state["format"] != "continuity-trainer-v1" or state["config"]["seed"] != plan["source_required_seed"]
        or state["stage"] != plan["source_required_stage"] or state["variant"]["kind"] != "learned"):
        raise ValueError("requires the independent learned Aseed101 checkpoint")
    blocks, actors, horizons = plan["block_indices"], plan["actor_indices"], plan["horizons_ticks"]
    if (horizons != sorted(set(horizons)) or not horizons or min(horizons) < 1
        or len(set(blocks)) != len(blocks) or len(set(actors)) != len(actors)
        or min(blocks) < 0 or min(actors) < 0):
        raise ValueError("distinct blocks/actors and positive sorted horizons required")
    before_tensors = _selected_fingerprint(state, blocks, actors)
    weight = state["model"]["recurrent"].double().clone()
    encoder = state["model"]["encoder.weight"].double().clone()
    generator = torch.Generator(device="cpu").manual_seed(plan["direction_seed"])
    records, block_metadata = [], []
    for index in blocks:
        block = state["replay"]["blocks"][index]
        if max(horizons) > len(block["elapsed"]) or any(block["resets"]):
            raise ValueError("audit requires enough continuous recorded observations")
        observations = block["observations"][:, actors].double().clone()
        hidden = block["start_h"][actors].double().clone()
        initial = unit_directions(len(actors), plan["directions_per_actor"], weight.shape[0], generator)
        exact, diagonal = initial.clone(), initial.clone()
        elapsed = 0.
        block_metadata.append({"block_index": index, "start_tick": block["start_tick"],
            "direction_sha256": tensor_sha256(initial), "initial_h_sha256": tensor_sha256(hidden),
            "max_unit_norm_error": float((torch.linalg.vector_norm(initial, dim=-1)-1.).abs().max())})
        for tick in range(max(horizons)):
            dt = float(block["elapsed"][tick])
            hidden, exact, diagonal = advance_tangents(weight, encoder, hidden, observations[tick], exact, diagonal,
                                                       dt, state["config"]["max_step"])
            elapsed += dt
            if tick+1 not in horizons:
                continue
            if not all(bool(torch.isfinite(value).all()) for value in (hidden, exact, diagonal)):
                raise FloatingPointError("nonfinite conditional primal or tangent")
            hidden_stats = pair_statistics(exact, diagonal)
            derivative = (1.-hidden.tanh().square())[:, None, :]
            feature_stats = pair_statistics(derivative*exact, derivative*diagonal)
            for actor_offset, actor in enumerate(actors):
                for direction in range(plan["directions_per_actor"]):
                    row = {"block_index": index, "actor_index": actor, "direction": direction,
                           "horizon_ticks": tick+1, "elapsed_time": elapsed}
                    for prefix, stats in (("hidden", hidden_stats), ("tanh_feature", feature_stats)):
                        for name, values in stats.items():
                            value = float(values[actor_offset, direction])
                            key = f"{name[:-3]}_{prefix}_l2" if name in ("exact_l2", "diagonal_l2") else f"{prefix}_{name}"
                            row[key] = value if math.isfinite(value) else None
                    records.append(row)
    unchanged = {"checkpoint_bytes": file_sha256(checkpoint) == before_checkpoint,
                 "protocol_bytes": file_sha256(protocol_path) == protocol_hash,
                 "frozen_runtime_and_protocol_bytes": all(file_sha256(name) == expected for name, expected in before_sources.items()),
                 "own_source_bytes": all(file_sha256(name) == expected for name, expected in own_sources.items()),
                 "loaded_model_and_selected_inputs": _selected_fingerprint(state, blocks, actors) == before_tensors,
                 "process_torch_cpu_rng": torch.equal(torch.get_rng_state(), process_rng)}
    if not all(unchanged.values()):
        raise AssertionError(f"read-only audit changed state/source: {unchanged}")
    measurements_path = folder/"measurements.json"
    write_json(measurements_path, {"format": FORMAT, "protocol_sha256": protocol_hash, "records": records})
    summary = {"format": FORMAT, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path), "protocol_sha256": protocol_hash,
        "checkpoint": str(checkpoint), "checkpoint_sha256_before": before_checkpoint,
        "checkpoint_sha256_after": file_sha256(checkpoint), "seed": state["config"]["seed"],
        "dtype": "float64", "device": "cpu", "frozen_byte_hashes": before_sources,
        "audit_source_byte_hashes": own_sources, "unchanged": unchanged,
        "selected_input_fingerprints": before_tensors, "block_metadata": block_metadata,
        "measurements": len(records), "measurements_file": str(measurements_path.resolve()),
        "measurements_sha256": file_sha256(measurements_path), "by_horizon": aggregate(records, horizons),
        "interpretation": plan["interpretation"],
        "norm_convention": "L2 norm per unit initial direction. Cosine is null if either sensitivity is zero; ratio is null if exact norm is zero. Tanh feature tangent is (1-tanh(h)^2)*v after the encoder pulse.",
        "sampling_note": "128 selected block-actor-direction samples per horizon share one independent Aseed101 model. Descriptive summaries are not independent-seed statistical replicates."}
    write_json(folder/"summary.json", summary)
    evidence = {name: value for name, value in summary.items() if name not in {"selected_input_fingerprints", "block_metadata"}}
    write_json(ROOT/"docs/evidence/credit-2026-10-27/tangent-audit-v1.json", evidence)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"data/credit/tangent-audit-v1/protocol.json")
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text(encoding="utf-8"))
    torch.set_num_threads(plan["threads"])
    result = run_audit(args.protocol)
    print(json.dumps({"summary": str(args.protocol.parent/"summary.json"), "measurements": result["measurements"], "unchanged": result["unchanged"]}))
