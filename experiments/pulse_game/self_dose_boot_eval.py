"""Sealed NumPy-only evaluation of matched self-write bootstrap checkpoints."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import self_write_core as core
from .self_dose_boot import ARMS, EDIT_METRICS, EVIDENCE, digest_arrays, edit_metrics, physical_contact, split_rng, world
from .self_write_transfer import flatten_body_edits, validate_transplant, writer_delta


SCENARIOS = ("stable_clean", "stable_noise", "flip_noise")
METRICS = ("correct", "rest_probe", "writer_distance", "writer_support", *EDIT_METRICS)
NAMES = tuple(f"{bank}_{credit}" for bank, credit in ARMS)
COEFFICIENTS = {
    "diversity": dict(zip(NAMES, (-.5, -.5, .5, .5), strict=True)),
    "credit_horizon": dict(zip(NAMES, (-.5, .5, -.5, .5), strict=True)),
    "interaction": dict(zip(NAMES, (1, -1, -1, 1), strict=True)),
    "horizon_at_low": {"low_short": -1, "low_long": 1},
    "horizon_at_high": {"high_short": -1, "high_long": 1},
    "diversity_at_short": {"low_short": -1, "high_short": 1},
    "diversity_at_long": {"low_long": -1, "high_long": 1},
}


def outcome_files(evidence, plan):
    for root in plan["birth_roots"]:
        record = json.loads((evidence / f"train-{root}.json").read_bytes())
        if record["protocol_sha256"] != sha256((evidence / "protocol.json").read_bytes()).hexdigest():
            raise ValueError(f"Birth root {root} was trained under a different protocol.")
        for name, expected in record["sources"].items():
            source = Path(__file__).with_name(name)
            if sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source mismatch for {root}: {name}.")
        yield root, record


def test_worlds(plan, split, root):
    recipient = split == "recipient"
    count = plan["development_lives"] if split == "development" else plan["held_out_lives"]
    contacts = plan["recipient_contacts"] if recipient else plan["evaluation_contacts"]
    flip_after = plan["recipient_flip_after"] if recipient else plan["eval_flip_after"]
    tape = world(split_rng(plan, split, root), count, contacts, plan["evaluation_pairs"],
                 plan["eval_noise"], 0, None)
    cues, targets, order, orient, noise = tape
    flipped = targets.copy()
    flipped[:, flip_after:] *= -1
    keys = split_rng(plan, split, root, 1).integers(2**32, size=contacts + 1)
    return {
        "stable_clean": (cues, targets, order, orient, np.ones_like(noise)),
        "stable_noise": tape,
        "flip_noise": (cues, flipped, order, orient, noise),
    }, keys


def deploy(initial, tape, keys, arm, limit=None):
    lives, contacts = tape[2].shape
    graph = initial if len(initial.w) == lives else core.repeated_graph(initial, lives)
    starting = graph
    h = core.rest(graph)
    still_initial = graph.mask.copy()
    result = {field: [] for field in (*METRICS, "side", "taste")}
    for t, key in enumerate(keys[:contacts if limit is None else limit]):
        before = graph
        left, target, taste_by_side = physical_contact(tape, t)
        graph, h, side, taste, edit, receipt = core.step(
            before, h, left, -left, taste_by_side, int(key), arm=arm, view="own",
        )
        result["correct"].append(side == target)
        result["side"].append(side.astype(np.int8))
        result["taste"].append(taste.astype(np.int8))
        both_scores = []
        for sign in (-1, 1):
            probe = core.advance(graph, core.rest(graph), core.sense(sign * left, -sign * left), core.MOTOR_TICKS)
            both_scores.append(core.motor(probe) * sign * target > 0)
        result["rest_probe"].append(np.mean(both_scores, axis=0).astype(np.float32))
        distance, support = writer_delta(starting, graph)
        result["writer_distance"].append(distance)
        result["writer_support"].append(support.astype(np.int8))
        changes = edit_metrics(before.w, before.b, before.mask, graph.w, graph.b, graph.mask, edit, receipt, still_initial)
        for name, values in zip(EDIT_METRICS, changes.T, strict=True):
            result[name].append(values.astype(np.int16))
        still_initial &= graph.mask
    return graph, {field: np.asarray(values).T for field, values in result.items()}


def graft_writer(initial, donor):
    graft = core.transplant_writer(initial, donor)
    validate_transplant(initial, graft, donor)
    connected = np.array([core.connected(mask) for mask in graft.mask])
    # A structurally invalid graft retains the starting writer for its entire assay.
    for field in ("w", "b", "mask"):
        getattr(graft, field)[~connected] = getattr(initial, field)[~connected]
    return graft, connected


def common_first_edit(initial, graft, tape, key):
    left, _, both_tastes = physical_contact(tape, 0)
    lanes = np.arange(len(left))
    forced_side = lanes % 2 * 2 - 1
    taste = both_tastes[lanes, (forced_side > 0).astype(int)]
    h = core.advance(initial, core.rest(initial), core.sense(left, -left), core.MOTOR_TICKS)
    moved = []
    for graph in (initial, graft):
        pre, post = core.contact_states(graph, h, left, -left, forced_side, taste)
        edit = core.proposal(graph, pre, post, int(key), view="own")
        changed, _ = core.apply(graph, edit)
        moved.append((flatten_body_edits(graph, changed), post[:, core.WRITER],
                      changed.mask[:, :core.WRITER.start] != graph.mask[:, :core.WRITER.start]))
    return {"body_edit_l2": np.linalg.norm(moved[1][0] - moved[0][0], axis=1),
            "writer_activity_l2": np.linalg.norm(moved[1][1] - moved[0][1], axis=1),
            "different_body_structural_edits": np.any(moved[1][2] != moved[0][2], axis=(1, 2))}


def unique_realized_examples(data):
    schedule = data["history_schedule"].ravel()
    shape = (len(schedule), -1)
    side = data["sides"].transpose(0, 2, 1).reshape(shape)
    taste = data["tastes"].transpose(0, 2, 1).reshape(shape)
    structure = data["accepted_structure"].transpose(0, 2, 1, 3).reshape(len(schedule), -1, 2)
    seen_outcomes = set()
    seen_commits = set()
    for history, actions, receipts, commits in zip(schedule, side, taste, structure, strict=True):
        for count in range(1, len(actions) + 1):
            prefix = (int(history), actions[:count].tobytes(), receipts[:count].tobytes())
            seen_outcomes.add(prefix)
            seen_commits.add((*prefix, commits[:count].tobytes()))
    return len(seen_outcomes), len(seen_commits)


def train_summary(data, plan):
    realized, committed = unique_realized_examples(data)
    return {
        "unique_environment_action_taste_prefixes": realized,
        "unique_environment_action_taste_structure_prefixes": committed,
        "edits": {k: int(v) for k, v in zip(EDIT_METRICS, data["edits"].sum(axis=(0, 1)), strict=True)},
        "final_30_step_mean_loss": float(data["loss"][-30:].mean()),
        "final_30_step_actual_choice": float(data["actual_correct"][-30:].mean() / plan["batch"]),
        "final_30_step_last_probe": float(data["probe_accuracy"][-30:, -1].mean()),
        "forward_s": float(data["forward_s"].sum()),
        "backward_and_optimizer_s": float(data["backward_and_optimizer_s"].sum()),
    }


def draw_pairs(plan):
    n_births = len(plan["birth_roots"])
    n_lives = plan["held_out_lives"]
    samples = plan["resampling_draws"]
    rng = split_rng(plan, "resampling", 0)
    roots = rng.integers(n_births, size=(samples, n_births, 1))
    lives = rng.integers(n_lives, size=(samples, n_births, n_lives))
    return roots, lives


def paired_stats(values, draws, windows):
    results = {}
    for lo, hi in windows:
        both = values[:, :, lo - 1:hi].mean(axis=2)
        replicates = both[draws].mean(axis=(1, 2))
        results[f"{lo}-{hi}"] = {
            "mean": float(both.mean()), "ci95": np.quantile(replicates, [0.025, 0.975]).tolist(),
            "by_root": both.mean(axis=1).tolist(),
        }
    return results


def test_statistics(records, plan):
    draws = draw_pairs(plan)
    roots = plan["birth_roots"]
    windows = plan["primary_windows"]
    result = {}
    for scenario in SCENARIOS:
        conditions = {}
        for checkpoint in (*NAMES, "spawn"):
            for policy in ("active", "primitive", "none"):
                conditions[f"{checkpoint}|{policy}"] = np.stack([
                    records[f"{root}|held_out|{scenario}|{checkpoint}|{policy}|correct"] for root in roots
                ]).astype(float)
        result[scenario] = {
            "conditions": {name: paired_stats(values, draws, windows) for name, values in conditions.items()},
            "factorial_active": {
                contrast: paired_stats(sum(scale * conditions[f"{cell}|active"] for cell, scale in formula.items()),
                                       draws, windows)
                for contrast, formula in COEFFICIENTS.items()
            },
            "ablate_writes": {
                f"{cell}|active_minus_{arm}": paired_stats(
                    conditions[f"{cell}|active"] - conditions[f"{cell}|{arm}"], draws, windows)
                for cell in NAMES for arm in ("none", "primitive")
            },
        }
    return result


def transfer_statistics(records, plan):
    draws = draw_pairs(plan)
    roots = plan["birth_roots"]
    windows = [[1, 12], [13, 24], [13, 15], [22, 24]]
    result = {}
    for scenario in ("stable_noise", "flip_noise"):
        result[scenario] = {}
        for cell in NAMES:
            values = {group: np.stack([
                records[f"{root}|recipient|{scenario}|{cell}|{group}|correct"] for root in roots
            ]).astype(float) for group in ("base-active", "base-none", "graft-active", "graft-none")}
            graft = values["graft-active"] - values["base-active"]
            adjusted = graft - (values["graft-none"] - values["base-none"])
            witnesses = {}
            for name in ("body_edit_l2", "writer_activity_l2", "different_body_structural_edits",
                         "writer_distance", "writer_support", "connected"):
                samples = np.stack([records[f"{root}|recipient|{scenario}|{cell}|witness|{name}"] for root in roots])
                witnesses[name] = float(samples.mean())
            result[scenario][cell] = {
                "conditions": {arm: paired_stats(scores, draws, windows) for arm, scores in values.items()},
                "while_writable": paired_stats(graft, draws, windows),
                "writable_minus_frozen_interaction": paired_stats(adjusted, draws, windows),
                "witnesses": witnesses,
            }
    return result


def organ_edit_evidence(records, plan):
    totals = {}
    for scenario in SCENARIOS:
        totals[scenario] = {}
        for cell in NAMES:
            key = lambda root, metric: f"{root}|held_out|{scenario}|{cell}|active|{metric}"
            totals[scenario][cell] = {
                name: sum(int(records[key(root, name)].sum()) for root in plan["birth_roots"])
                for name in EDIT_METRICS
            }
            totals[scenario][cell]["mean_final_writer_distance"] = float(np.mean([
                records[key(root, "writer_distance")][:, -1] for root in plan["birth_roots"]]))
    return totals


def run(evidence):
    protocol = (evidence / "protocol.json").read_bytes()
    plan = json.loads(protocol)
    records, training, tapes = {}, {}, {}
    for root, record in outcome_files(evidence, plan):
        root_key = str(root)
        training[root_key] = {"provenance": record, "runs": {}}
        for split in ("development", "held_out", "recipient"):
            scenarios, keys = test_worlds(plan, split, root)
            tapes[root_key + "|" + split] = {"keys": digest_arrays(keys), "tapes": {}}
            records[root_key + "|" + split + "|receipt_keys"] = keys
            for scenario, tape in scenarios.items():
                tapes[root_key + "|" + split]["tapes"][scenario] = digest_arrays(*tape)
                for label, values in zip(("cues", "targets", "order", "orientation", "noise"), tape, strict=True):
                    records[f"{root}|{split}|{scenario}|world|{label}"] = values
        births = {"spawn": core.spawn_graph(root)}
        for bank, credit in ARMS:
            cell = f"{bank}_{credit}"
            path = evidence / f"train-{root}-{bank}-{credit}.npz"
            if sha256(path.read_bytes()).hexdigest() != record["checkpoints"][path.name]:
                raise ValueError(f"Checkpoint provenance mismatch: {path}")
            with np.load(path, allow_pickle=False) as data:
                training[root_key]["runs"][cell] = train_summary(data, plan)
                births[cell] = core.Graph(data["w"], data["b"], data["mask"])
        def collect(prefix, values):
            records.update({f"{prefix}|{name}": array for name, array in values.items()})
        for cell, birth in births.items():
            for split in ("development", "held_out"):
                scenarios, keys = test_worlds(plan, split, root)
                for scenario, tape in scenarios.items():
                    for arm in ("active", "primitive", "none") if split == "held_out" else ("active",):
                        _, results = deploy(birth, tape, keys, arm)
                        collect(f"{root}|{split}|{scenario}|{cell}|{arm}", results)
            if cell == "spawn":
                continue
            donor_worlds, donor_keys = test_worlds(plan, "held_out", root)
            receiver_worlds, receiver_keys = test_worlds(plan, "recipient", root)
            starting = core.repeated_graph(birth, plan["held_out_lives"])
            for scenario in ("stable_noise", "flip_noise"):
                donor, _ = deploy(starting, donor_worlds[scenario], donor_keys, "active", limit=plan["donor_contacts"])
                transplanted, connected = graft_writer(starting, donor)
                tape = receiver_worlds[scenario]
                prefix = f"{root}|recipient|{scenario}|{cell}"
                for group, initial in (("base", starting), ("graft", transplanted)):
                    for arm in ("active", "none"):
                        _, results = deploy(initial, tape, receiver_keys, arm)
                        collect(f"{prefix}|{group}-{arm}", results)
                distance, support = writer_delta(starting, transplanted)
                witness = common_first_edit(starting, transplanted, tape, receiver_keys[-1])
                collect(f"{prefix}|witness",
                        {**witness, "writer_distance": distance, "writer_support": support, "connected": connected})
        print(f"Evaluated {root} using sealed worlds and frozen optimizer.", flush=True)
    trace = evidence / "evaluation_traces.npz"
    np.savez_compressed(trace, **records)
    dense_forward_flops = 2 * core.N**2 * (
        core.MOTOR_TICKS + core.CONTACT_TICKS + core.TASTE_TICKS + 2 * plan["train_pairs"] * core.MOTOR_TICKS)
    report = {
        "provenance": {"protocol_sha256": sha256(protocol).hexdigest(),
                       "evaluator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "transfer_helpers_sha256": sha256(Path(__file__).with_name("self_write_transfer.py").read_bytes()).hexdigest(),
                       "traces_sha256": sha256(trace.read_bytes()).hexdigest(), "tape_and_receipt_keys": tapes},
        "train": training,
        "forward_dense_multiply_add_flops_per_contact": dense_forward_flops,
        "forward_dense_multiply_add_flops_per_arm": dense_forward_flops * plan["train_steps"] * plan["batch"] * plan["train_contacts"],
        "held_out": test_statistics(records, plan),
        "held_out_edits": organ_edit_evidence(records, plan),
        "transfer": transfer_statistics(records, plan),
    }
    (evidence / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    run(args.evidence)


if __name__ == "__main__":
    main()
