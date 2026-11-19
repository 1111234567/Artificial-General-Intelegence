"""Replay edit journals and estimate paired physical success differences."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

from . import plastic_topology_study as study
from .ring_logic_world import PAIRS, RULES
from .ring_logic_study import read, sha, write


DRAWS = 20_000
CONTRASTS = (("guided", "frozen"), ("guided", "random"))


def entries(array):
    return {"min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())}


def physical_metrics(scores):
    result = {}
    correct = scores[..., 2] > 0
    timeout = scores[..., 0] < 0
    for k, rule in enumerate(RULES):
        result[rule] = {}
        for a, arm in enumerate(study.ARMS):
            values = correct[:, :, k, a]
            result[rule][arm] = {"correct": float(values.mean()),
                                 "all_20_correct": float(values.all(axis=-1).mean()),
                                 "per_cell_min": float(values.mean(axis=(0, 1, 2)).min()),
                                 "timeout": float(timeout[:, :, k, a].mean()),
                                 "per_birth_restart": values.mean(axis=(2, 3)).tolist()}
    return result


def contrast_intervals(scores):
    # Sum over all 20 cells before resampling; their odor and motor realization is shared.
    accuracy = (scores[..., 2] > 0).mean(axis=(1, 5))
    group_count, _, _, root_count = accuracy.shape
    rng = np.random.default_rng(3978721)
    group_weights = rng.multinomial(group_count, np.full(group_count, 1 / group_count), DRAWS)
    root_weights = rng.multinomial(root_count, np.full(root_count, 1 / root_count), DRAWS)
    family = len(RULES) * len(CONTRASTS)
    tail = .05 / (2 * family)
    results = []
    for k, rule in enumerate(RULES):
        for a, b in CONTRASTS:
            diff = accuracy[:, k, study.ARMS.index(a)] - accuracy[:, k, study.ARMS.index(b)]
            draws = ((group_weights @ diff) * root_weights).sum(axis=-1) / (group_count * root_count)
            results.append({"rule": rule, "contrast": [a, b], "delta": float(diff.mean()),
                            "simultaneous_95": np.quantile(draws, [tail, 1 - tail]).tolist(),
                            "pointwise_95": np.quantile(draws, [.025, .975]).tolist()})
    return results


def candidate_mask(prior, record, signed_count):
    if record is None:
        return None
    changed = prior.copy()
    for key, value in (("growth", True), ("retired", False)):
        for source, target in record[key]:
            if source == target or changed[target, source] == value:
                raise RuntimeError(f"Illegal candidate {key} operation")
            changed[target, source] = value
    if (len(record["growth"]) != max(0, signed_count)
            or len(record["retired"]) != max(0, -signed_count)
            or not np.isfinite(record["loss"]) or not study.model.strong(changed)
            or study.mask_hash(changed) != record["sha256"]):
        raise RuntimeError("Invalid candidate arcs, loss, connectivity, or fingerprint")
    return changed


def replay_shard(path, artifact, birth_for=study.model.birth_mask):
    journal = read(path / artifact["edits"])
    rows = artifact["rows"]
    if rows != journal["rows"] or tuple(r["update"] for r in journal["rounds"]) != study.EDITS:
        raise RuntimeError("Unexpected source rows or edit schedule")
    row_ids = {(group, restart, seed, RULES[k], arm): r
               for r, (group, restart, seed, k, arm) in enumerate(rows)}
    with np.load(path / artifact["checkpoint"], allow_pickle=False) as saved:
        masks, final = saved["birth"], saved["final"]
    starting = masks.copy()
    for i, (_, _, seed, _, _) in enumerate(rows):
        if (not np.array_equal(masks[i], birth_for(seed))
                or study.topology(masks[i]) != journal["birth"][i]):
            raise RuntimeError("Unexpected birth topology")

    results = []
    for round_data in journal["rounds"]:
        if len(round_data["paired"]) != len(rows) // len(study.ARMS):
            raise RuntimeError("Missing edit pair")
        visited = set()
        for pair in round_data["paired"]:
            identity = tuple(pair[key] for key in ("group", "restart", "seed", "rule"))
            if identity in visited:
                raise RuntimeError("Duplicate edit pair")
            visited.add(identity)
            control = []
            losses = []
            first_record = {}
            for arm in ("guided", "random"):
                row = row_ids[(*identity, arm)]
                record = pair[arm]
                prior = masks[row]
                if study.mask_hash(prior) != record["prior_hash"] or prior.sum() != record["prior_count"]:
                    raise RuntimeError("Broken incumbent chain")
                checked = [candidate_mask(prior, candidate, amount) for candidate, amount in
                           zip(record["candidates"], study.PROPOSALS, strict=True)]
                picked = record["chosen_candidate"]
                result = checked[picked]
                if (result is None or study.PROPOSALS[picked] != pair["signed_edit"]
                        or study.mask_hash(result) != record["result_hash"]
                        or result.sum() != record["result_count"] or not record["strong"]):
                    raise RuntimeError("Invalid selected edit")
                if arm == "guided":
                    losses = [c["loss"] if c is not None else np.inf for c in record["candidates"]]
                else:
                    control = checked
                first_record[arm] = record["candidates"][picked]["loss"] - record["candidates"][0]["loss"]
                masks[row] = result
            rng = np.random.default_rng([identity[2], RULES.index(identity[3]), round_data["update"], 2334187])
            expected = study.choice(losses, control, rng)
            if expected != (pair["guided"]["chosen_candidate"], pair["random"]["chosen_candidate"]):
                raise RuntimeError("Edit choice differed from registered rule")
            if pair["guided"]["result_count"] != pair["random"]["result_count"]:
                raise RuntimeError("Unmatched edge count")
            results.append({"rule": pair["rule"], "delta_count": pair["signed_edit"],
                            "edge_count": pair["guided"]["result_count"], **first_record})

    if not np.array_equal(masks, final):
        raise RuntimeError("Replayed final masks differ from frozen checkpoint")
    if any(study.topology(mask) != record for mask, record in zip(final, journal["final"], strict=True)):
        raise RuntimeError("Final topology record differs from frozen checkpoint")
    timings = {key: sum(r[key] for r in journal["rounds"]) for key in
               ("proposal_seconds", "score_seconds", "selection_and_install_seconds", "evaluated_candidates")}
    return results, final, starting, timings


def structural_metrics(edits, final, birth):
    by_rule = {}
    edit_types = Counter(r["delta_count"] for r in edits)
    changes = [r for r in edits if r["delta_count"] != 0]
    for rule in RULES:
        rule_attempts = [r for r in edits if r["rule"] == rule]
        rule_changes = [r for r in rule_attempts if r["delta_count"] != 0]
        by_rule[rule] = {"rounds": len(rule_attempts), "accepted": len(rule_changes),
                         "guided_loss_delta_when_edited": entries(np.array([r["guided"] for r in rule_changes])),
                         "random_loss_delta_when_edited": entries(np.array([r["random"] for r in rule_changes])),
                         "random_improved": sum(r["random"] < 0 for r in rule_changes)}
    topology = {"strong_final_circuits": sum(study.model.strong(mask) for mask in final),
                "self_arcs": int(final.diagonal(axis1=1, axis2=2).sum()),
                "total_circuits": len(final), "all_intermediate_edge_count": entries(np.array([r['edge_count'] for r in edits]))}
    shape = (study.GROUPS * study.RESTARTS * len(RULES), len(study.ARMS), study.model.WIDTH, study.model.WIDTH)
    paired_final = final.reshape(shape)
    paired_birth = birth.reshape(shape)
    for i, arm in enumerate(study.ARMS):
        mask, origin = paired_final[:, i], paired_birth[:, i]
        topology[arm] = {"final_edge_count": entries(mask.sum(axis=(1, 2))),
                         "in_degree": entries(mask.sum(axis=-1)), "out_degree": entries(mask.sum(axis=-2)),
                         "retained_birth_edges": entries((mask & origin).sum(axis=(1, 2))),
                         "final_differs_from_birth": int(np.any(mask != origin, axis=(1, 2)).sum())}
    guide, random = paired_final[:, 1], paired_final[:, 2]
    topology["guided_random_final_jaccard"] = entries((guide & random).sum(axis=(1, 2)) / (guide | random).sum(axis=(1, 2)))
    return {"rounds": len(edits), "accepted": len(changes), "edit_counts_signed": dict(sorted(edit_types.items())),
            "rules": by_rule, "topology": topology}


def collect(out, registration, artifacts, birth_for=study.model.birth_mask):
    all_edits, all_final, all_birth, timing, raw, score_hashes = [], [], [], [], [], {}
    for artifact in artifacts:
        edits, final, birth, timings = replay_shard(out, artifact, birth_for=birth_for)
        all_edits.extend(edits)
        all_final.append(final)
        all_birth.append(birth)
        timing.append({"first_group": artifact["first_group"], "training_and_edit_seconds":
                       artifact["seconds_through_edit_install"], **timings})
        score_path = out / "scores" / f"birth-{artifact['first_group']:02d}.npz"
        score_hashes[score_path.relative_to(out).as_posix()] = sha(score_path)
        with np.load(score_path, allow_pickle=False) as saved:
            raw.append(saved["standard"])
    scores = np.concatenate(raw).reshape(study.GROUPS, study.RESTARTS, len(RULES), len(study.ARMS),
                                         len(study.FINAL_ROOTS), len(PAIRS), 3)
    if (not np.isin(scores[..., 2], [-1, 0, 1]).all()
            or not np.array_equal(scores[..., 0] < 0, scores[..., 2] == 0)):
        raise RuntimeError("Physical score archive contains inconsistent contacts")
    result = {"physical": physical_metrics(scores), "paired_intervals": contrast_intervals(scores),
              "structure": structural_metrics(all_edits, np.concatenate(all_final), np.concatenate(all_birth)),
              "compute": timing, "bootstrap_draws": DRAWS,
              "registration_sha256": sha(out / "registered.json"), "seal_sha256": sha(out / "frozen.json"),
              "analysis_sha256": sha(Path(__file__)), "score_sha256": score_hashes,
              "runtime": registration["runtime"]}
    return result


def summarize(out):
    write(out / "summary.json", collect(out, study.registered(out), study.frozen(out)))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.DEFAULT)
    summarize(parser.parse_args().out)
