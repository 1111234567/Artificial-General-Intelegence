"""Read-only node activity diagnostic for the completed uncapped A/B experiment."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evolved_plasticity import ACTIONS, INPUTS, JOBS, make_genome
from .evolved_plasticity_world import FIRST_FINISH, N_SYMBOLS, NEW, _receipt_table
from .plasticity_continuous_world import make_continuous_schedule
from .uncapped_bf16_backend import _BF16Policy, _rollout
from .uncapped_cuda_backend import _stack_genomes


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/undirected/2026-11-16-plasticity-uncapped-v1"
OUT = ROOT / "data/undirected/2026-11-16-uncapped-modularity/activity.json"
FIRST_OUTPUT = INPUTS
FIRST_FREE = INPUTS + JOBS * ACTIONS
ACTIVITY_THRESHOLD = 0.5
SATURATION_THRESHOLD = 0.95
OCCUPANCY_THRESHOLD = 0.95


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidates(run, nodes):
    final_path = run / "champion.pt"
    previous_path = run / "generations/0896/elites.pt"
    founders_path = run / "generations/0001/plan.json"
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    previous = torch.load(previous_path, map_location="cpu", weights_only=True)
    previous_id = read(run / "audits/0896.json")["candidate"]
    founders = [entry for entry in read(founders_path) if entry["kind"] == "founder"]
    one_per_island = {}
    for entry in founders:
        one_per_island.setdefault(entry["island"], entry)
    founders = list(one_per_island.values())
    entries = [
        {"id": final["row"]["id"], "role": "final"},
        {"id": previous_id, "role": "generation_896"},
        *({"id": entry["id"], "role": "random_founder", "seed": entry["seed"]}
          for entry in founders),
    ]
    genomes = [final["genome"], previous["genomes"][previous_id]]
    genomes.extend(make_genome(entry["seed"], nodes) for entry in founders)
    fingerprints = {path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (final_path, previous_path, founders_path)}
    return entries, genomes, fingerprints


class _RecordingPolicy(_BF16Policy):
    def __init__(self, birth, actors, device):
        super().__init__(birth, actors, device)
        self.recorded_h = []
        self.recorded_activity = []

    def act(self, obs):
        action = super().act(obs)
        self.recorded_h.append(self.h.float().clone())
        self.recorded_activity.append(self.activity.float())
        return action


def record_world(genomes, seed, actors, device, visits):
    nodes, birth = _stack_genomes(genomes)
    schedule = make_continuous_schedule(seed=seed, actors=actors, device=device, visits=visits)
    schedule = replace(schedule, public=schedule.public.bfloat16())
    requests, ticks = schedule.stage.shape[:2]
    policy = _RecordingPolicy({key: value.to(device) for key, value in birth.items()}, actors, device)
    with torch.no_grad():
        metrics = _rollout(policy, schedule, _receipt_table(device).bfloat16())

    h = torch.stack(policy.recorded_h).cpu().numpy().reshape(requests, ticks, -1, actors, nodes)
    activity = torch.stack(policy.recorded_activity).cpu().numpy()
    activity_stats = {
        "full_mean_abs": np.abs(activity).mean(axis=(0, 2)),
        "full_active_fraction": (np.abs(activity) >= ACTIVITY_THRESHOLD).mean(axis=(0, 2)),
        "full_saturated_fraction": (np.abs(activity) >= SATURATION_THRESHOLD).mean(axis=(0, 2)),
        "full_mean_abs_step_change": np.abs(np.diff(activity, axis=0)).mean(axis=(0, 2)),
    }
    issue = schedule.public[..., 0, NEW].bool().cpu().numpy().argmax(axis=1)
    ready = (schedule.stage.cpu().numpy() == 2).argmax(axis=1)
    coordinates = (np.arange(requests)[:, None], np.arange(actors)[None, :])
    view = h.transpose(0, 1, 3, 2, 4)
    activity_view = activity.reshape(requests, ticks, -1, actors, nodes).transpose(0, 1, 3, 2, 4)
    samples = {
        "pre_new": view[coordinates[0], issue - 1, coordinates[1]],
        "new": view[coordinates[0], issue, coordinates[1]],
        "first_ready": view[coordinates[0], ready, coordinates[1]],
        "first_ready_activity": activity_view[coordinates[0], ready, coordinates[1]],
    }
    # Issue offset zero uses the real preceding request's final state. The
    # first request starts from zero and cannot wrap to the end of the tape.
    prior_tail = np.zeros_like(samples["pre_new"])
    prior_tail[1:] = view[:-1, -1]
    samples["pre_new"] = np.where(issue[..., None, None] == 0, prior_tail, samples["pre_new"])
    target = schedule.target.cpu().numpy() - FIRST_FINISH
    labels = {
        "raw": schedule.raw.cpu().numpy(),
        "version": target // N_SYMBOLS,
        "context": schedule.context.cpu().numpy(),
        "private_target_symbol": target % N_SYMBOLS,
        "wait": ready - issue,
        "issue": issue,
        "visit": np.broadcast_to(coordinates[0] // N_SYMBOLS, issue.shape),
        "request_in_visit": np.broadcast_to(coordinates[0] % N_SYMBOLS, issue.shape),
        "actor": np.broadcast_to(coordinates[1], issue.shape),
    }
    scores = {key: metrics[key].cpu().tolist() for key in (
        "completion", "strict", "ready_coverage", "version_accuracy", "symbol_accuracy",
        "mean_edges", "final_edges", "births", "deaths",
    )}
    return samples, labels, activity_stats, scores


def adjust_within_actor(values, actor):
    result = values.copy()
    for actor_id in np.unique(actor):
        select = actor == actor_id
        result[select] -= result[select].mean(axis=0)
    return result


def fit_contrasts(samples, labels, select):
    """Fit partial effects at the same phase, with actor fixed effects."""
    factors = {key: np.unique(value[select], return_inverse=True)[1] for key, value in labels.items()
               if key != "actor"}
    encodings = {key: np.eye(value.max() + 1)[value] for key, value in factors.items()}
    design = np.concatenate(list(encodings.values()), axis=-1)
    actor = labels["actor"][select]
    shape = samples.shape[1:]
    centered_x = adjust_within_actor(design, actor)
    centered_h = adjust_within_actor(samples[select].reshape(-1, np.prod(shape)).astype(float), actor)
    coefficients = np.linalg.lstsq(centered_x, centered_h, rcond=1.e-9)[0]
    contrasts = {}
    offset = 0
    for factor, coding in encodings.items():
        levels = coding.shape[1]
        if levels > 1:
            block = coefficients[offset:offset + levels].reshape(levels, *shape)
            contrasts[factor] = (block - block.mean(axis=0)) * levels / (levels - 1)
        offset += levels
    return contrasts


def top(values, start, stop, k):
    return np.argsort(-values[start:stop], kind="stable")[:k] + start


def cosine(first, second):
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first.ravel(), second.ravel()) / norm) if norm > 0 else None


def occupancy_stats(snapshots, labels, split, stats, index):
    activity = snapshots["first_ready_activity"][:, index]
    nodes = activity.shape[-1]
    absolute = np.abs(activity)
    ready = {"ab": absolute.mean(axis=0)}
    ready.update({name: absolute[labels["context"] == context].mean(axis=0)
                  for context, name in enumerate(("a", "b"))})
    ready.update({"train": absolute[split].mean(axis=0), "test": absolute[~split].mean(axis=0)})
    full = {key: val[index] for key, val in stats.items()}
    result = {
        "per_node": {key: values.round(7).tolist() for key, values in {**full, **ready}.items()},
        "groups": {},
    }
    for name, start, stop, k in (
        ("all", 0, nodes, 8), ("inputs", 0, INPUTS, 8),
        ("outputs", FIRST_OUTPUT, FIRST_FREE, 4), ("free", FIRST_FREE, nodes, nodes - FIRST_FREE),
    ):
        positions = slice(start, stop)
        ranked = {key: top(values, start, stop, k) for key, values in ready.items()}
        result["groups"][name] = {
            "mean_abs_activity": float(ready["ab"][positions].mean()),
            "full_mean_abs_step_change": float(full["full_mean_abs_step_change"][positions].mean()),
            "always_active_nodes": (start + np.flatnonzero(full["full_active_fraction"][positions]
                                                           >= OCCUPANCY_THRESHOLD)).tolist(),
            "saturated_nodes": (start + np.flatnonzero(full["full_saturated_fraction"][positions]
                                                       >= OCCUPANCY_THRESHOLD)).tolist(),
            "top_first_ready_nodes": {key: ids.tolist() for key, ids in ranked.items()},
            "top_overlap_a_b": len(set(ranked["a"]) & set(ranked["b"])),
            "top_overlap_train_test": len(set(ranked["train"]) & set(ranked["test"])),
        }
    return result


def summarize_response(fits, index, factor):
    train, test, a, b = [fit[factor][:, index] for fit in fits]
    sizes = np.sqrt(np.mean(train ** 2, axis=0))
    test_sizes = np.sqrt(np.mean(test ** 2, axis=0))
    noninput = slice(FIRST_OUTPUT, None)
    nodes = sizes.shape[-1]
    leading = top(sizes, FIRST_OUTPUT, nodes, 4)
    a_top = top(np.sqrt(np.mean(a ** 2, axis=0)), FIRST_OUTPUT, nodes, 4)
    b_top = top(np.sqrt(np.mean(b ** 2, axis=0)), FIRST_OUTPUT, nodes, 4)
    energy = sizes[noninput] ** 2
    test_top = top(test_sizes, FIRST_OUTPUT, nodes, 4)
    return {
        "train_noninput_effect_rms": float(np.sqrt(energy.mean())),
        "test_noninput_effect_rms": float(np.sqrt(np.mean(test_sizes[noninput] ** 2))),
        "train_test_noninput_effect_cosine": cosine(train[:, noninput], test[:, noninput]),
        "a_b_noninput_effect_cosine": cosine(a[:, noninput], b[:, noninput]),
        "train_effective_noninput_nodes": float(energy.sum() ** 2 / (energy ** 2).sum()),
        "train_top4_effect_energy_share": float((sizes[leading] ** 2).sum() / energy.sum()),
        "top4_train": leading.tolist(),
        "top4_test": test_top.tolist(),
        "top4_a": a_top.tolist(),
        "top4_b": b_top.tolist(),
        "top4_overlap_a_b": len(set(a_top) & set(b_top)),
        "top4_overlap_train_test": len(set(leading) & set(test_top)),
        "test_effect_by_top4_train": test[:, leading].round(7).tolist(),
        "test_effect_by_free_node": test[:, FIRST_FREE:].round(7).tolist(),
        "train_effect_rms_per_node": sizes.round(7).tolist(),
        "test_effect_rms_per_node": test_sizes.round(7).tolist(),
    }


def finish_version_readout(samples, labels, index):
    """Which version wins the actual FINISH port comparison at first READY?"""
    version = labels["version"]
    result = {}
    for phase in ("pre_new", "first_ready"):
        start = FIRST_OUTPUT + FIRST_FINISH
        finish_ports = samples[phase][:, index, start:start + 2 * N_SYMBOLS].reshape(-1, 2, N_SYMBOLS)
        predicted_version = finish_ports.max(axis=-1).argmax(axis=-1)
        result[phase] = {"balanced_accuracy": float(np.mean([
            (predicted_version[version == number] == number).mean() for number in range(2)
        ]))}
    return result


def audit(run, device):
    protocol = read(run / "protocol.json")
    holdout = read(run / "holdout.json")
    entries, genomes, fingerprints = load_candidates(run, protocol["nodes"])
    assert entries[0]["id"] == holdout["candidate"]
    seeds = holdout["seeds"]
    actors = holdout["actors"]
    visits = protocol["visits"]
    records = [record_world(genomes, seed, actors, device, visits) for seed in seeds]
    samples = {key: np.concatenate([record[0][key].reshape(-1, len(genomes), protocol["nodes"])
                                    for record in records]) for key in records[0][0]}
    samples["first_ready_minus_pre_new"] = samples["first_ready"] - samples["pre_new"]
    labels = {key: np.concatenate([record[1][key].flatten() for record in records])
              for key in records[0][1]}
    per_seed = labels["actor"].size // len(seeds)
    seed_index = np.repeat(np.arange(len(seeds)), per_seed)
    labels["actor"] += seed_index * actors
    stats = {key: np.mean([record[2][key] for record in records], axis=0) for key in records[0][2]}
    train = seed_index < len(seeds) // 2
    by_a = labels["context"] == 0
    fits = {}
    for phase in ("pre_new", "new", "first_ready", "first_ready_minus_pre_new"):
        fits[phase] = [fit_contrasts(samples[phase], labels, select)
                       for select in (train, ~train, by_a, ~by_a)]

    result = {
        "source_run": str(run), "source_sha256": fingerprints,
        "device": str(device), "torch": torch.__version__, "precision": "sealed BF16 policy, eager",
        "seeds": seeds, "train_seeds": seeds[:len(seeds) // 2], "actors": actors, "visits": visits,
        "node_groups": {"inputs": list(range(INPUTS)), "outputs": list(range(FIRST_OUTPUT, FIRST_FREE)),
                        "free": list(range(FIRST_FREE, protocol["nodes"]))},
        "definitions": {
            "activation": "absolute tanh(h); activity/saturation thresholds 0.50/0.95; always or saturated requires at least 95% of all trajectory ticks and actors",
            "response": "partial single-factor contrasts in h with seed-actor fixed effects and simultaneous categorical controls for the other cues, wait, issue offset, visit and request ordinal",
            "response_windows": "one matched state per request: immediately before NEW, after NEW, and first valid READY before feedback from the corresponding action; also within-request first READY minus pre-NEW",
            "controls": "first actual birth-only founder from each island in generation 1 plan, drawn before fitness selection, same fixed ports, node count, degree and full local rule family",
            "estimates": "descriptive disjoint seed splits and A/B comparisons; factor labels enter the diagnostic only",
        },
        "official_champion_per_seed": holdout["paired_scores"]["full"],
        "candidates": [],
    }
    for index, entry in enumerate(entries):
        scores = [{key: values[index] for key, values in record[3].items()} for record in records]
        context_factors = ("version", "raw", "private_target_symbol")
        response = {phase: {factor: summarize_response(fits[phase], index, factor)
                            for factor in context_factors} for phase in fits}
        # The context contrast itself can only be estimated when both A/B are present.
        for phase in fits:
            first, second = fits[phase][:2]
            first_effect = first["context"][:, index, FIRST_OUTPUT:]
            second_effect = second["context"][:, index, FIRST_OUTPUT:]
            response[phase]["context"] = {
                "train_noninput_effect_rms": float(np.sqrt(np.mean(first_effect ** 2))),
                "test_noninput_effect_rms": float(np.sqrt(np.mean(second_effect ** 2))),
                "train_test_noninput_effect_cosine": cosine(first_effect, second_effect),
            }
        result["candidates"].append({
            **entry, "mean_behavior": {key: float(np.mean([score[key] for score in scores]))
                                      for key in scores[0]},
            "per_seed_behavior": scores, "activation": occupancy_stats(samples, labels, train, stats, index),
            "response": response, "finish_version_readout": finish_version_readout(samples, labels, index),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = audit(args.run, torch.device(args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
                        encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
