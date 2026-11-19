"""Repeat the ecological dose assay on frozen full-permission editor births."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import self_dose_life as life
from . import self_dose_life_analysis as analysis
from . import self_param_writer as full


PLAN = life.EVIDENCE / "expanded-preregistered.json"
FIELDS = ("weights", "bias", "present")
REGIONS = (slice(0, full.BODY), slice(full.BODY, None))


def map_graph(graph, fn):
    return full.Graph(*(fn(getattr(graph, name)) for name in FIELDS))


def repeat(graph, copies):
    return map_graph(graph, lambda values: values.repeat_interleave(copies, dim=0))


def graph_arrays(prefix, graph):
    return {f"{prefix}_{name}": getattr(graph, name).numpy() for name in FIELDS}


def connected(graph):
    return np.array([full.strongly_connected(row) for row in graph.present.numpy()])


def validate(graph):
    if graph.present.dtype != torch.bool:
        raise AssertionError("Graph support must be decoded as Boolean")
    if torch.any(graph.weights[~graph.present]):
        raise ValueError("Use retrained checkpoints from the masked-readback protocol")
    for field in (graph.weights, graph.bias):
        assert torch.isfinite(field).all() and (field.abs() <= full.LIMIT).all()


def load_births(plan, writer_width):
    paths = [Path(plan["checkpoint_template"].format(width=writer_width, root=root)) for root in plan["birth_roots"]]
    births = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            support = saved["present"]
            if not np.isin(support, (0, 1)).all():
                raise ValueError(f"Unexpected support bits in {path}")
            graph = full.Graph(torch.from_numpy(saved["weights"]), torch.from_numpy(saved["bias"]),
                               torch.from_numpy(support.astype(bool)))
        validate(graph)
        if graph.bias.shape != (1, full.BODY + writer_width) or not connected(graph).all():
            raise ValueError(f"Unexpected birth graph in {path}")
        births.append(graph)
    return full.Graph(*(torch.cat([getattr(graph, name) for graph in births]) for name in FIELDS)), {
        path.as_posix(): life.fingerprint(path) for path in paths
    }


def graft(base, donor):
    result = full.Graph(*(torch.cat((getattr(base, f)[:, REGIONS[0]],
                                    getattr(donor, f)[:, REGIONS[1]]), dim=1) for f in FIELDS))
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(result, name)[:, REGIONS[0]], getattr(base, name)[:, REGIONS[0]])
        np.testing.assert_array_equal(getattr(result, name)[:, REGIONS[1]], getattr(donor, name)[:, REGIONS[1]])
    validate(result)
    return result


def scramble(base, intact, namespace):
    before = life.core.Graph(*(getattr(base, name).numpy() for name in FIELDS))
    after = life.core.Graph(*(getattr(intact, name).numpy().copy() for name in FIELDS))
    rng = np.random.default_rng(namespace)
    for lane in range(len(before.w)):
        for row in range(full.BODY, base.width):
            for active_before in (False, True):
                addresses = np.flatnonzero(after.mask[lane, row] & (before.mask[lane, row] == active_before))
                after.w[lane, row, addresses] = life.transfer.bounded_shuffle(
                    before.w[lane, row, addresses], after.w[lane, row, addresses], rng,
                )
        after.b[lane, full.BODY:] = life.transfer.bounded_shuffle(
            before.b[lane, full.BODY:], after.b[lane, full.BODY:], rng,
        )
    result = full.Graph(*(torch.from_numpy(getattr(after, name)) for name in life.FIELDS))
    np.testing.assert_allclose(distance(base, result)[:, 1], distance(base, intact)[:, 1], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.present, intact.present)
    validate(result)
    return result


def numeric_distance(weight, bias, support):
    return np.stack([
        np.stack((np.linalg.norm(weight[:, rows].reshape(len(weight), -1), axis=1),
                  np.linalg.norm(bias[:, rows], axis=1), np.count_nonzero(support[:, rows], axis=(1, 2))), axis=-1)
        for rows in REGIONS
    ], axis=1)


def effect(signed, blank):
    return ((signed.weights - blank.weights).numpy(), (signed.bias - blank.bias).numpy(),
            (signed.present.to(torch.int8) - blank.present.to(torch.int8)).numpy())


def distance(before, after):
    return numeric_distance(*effect(after, before))


def assess(graph, tape, key, *, intervention="active", snapshots=None):
    lanes = np.tile(np.arange(tape.good.shape[1]), len(graph.bias) // tape.good.shape[1])
    events = {name: [] for name in ("side", "taste", "correct", "motor", "command", "payload", "receipt")}
    draws = torch.from_numpy(np.random.default_rng(key).random((len(tape.good), len(lanes), 2), dtype=np.float32))
    hidden = torch.zeros_like(graph.bias)
    states = {0: (graph, hidden)}
    for time in range(len(tape.good)):
        left, right, truth, table = tape.present(time, lanes)
        left, right = torch.from_numpy(left), torch.from_numpy(right)
        hidden = full.advance(graph, hidden, full.sensory(graph, left, right), 5)
        motor = 2.5 * (hidden[:, 11] - hidden[:, 10])
        side = torch.where(motor > 0, 1, -1)
        events["side"].append(side.numpy().astype(np.int8))
        events["correct"].append(side.numpy() == truth)
        events["motor"].append(motor.numpy())
        if snapshots is None and time + 1 == len(tape.good):
            break
        taste = torch.from_numpy(table[np.arange(len(lanes)), (side > 0).numpy().astype(int)])
        edit = full.propose(graph, hidden, left, right, side, taste, draws[time], view="own",
                            intervention=intervention)
        contact = full.advance(graph, hidden, full.sensory(graph, left, right, side), 3)
        hidden = full.advance(graph, contact, full.sensory(graph, left, right, side, taste), 3)
        graph = edit.next_graph
        events["taste"].append(taste.numpy().astype(np.int8))
        events["command"].append(np.stack((edit.command.address.numpy(), edit.command.op.numpy()), axis=-1).astype(
            np.int16
        ))
        events["payload"].append(edit.command.payload.numpy())
        events["receipt"].append(edit.receipt.status.numpy().astype(np.int8))
        if snapshots is not None and time + 1 in snapshots:
            states[snapshots[time + 1]] = (graph, hidden)
    return states, {name: np.stack(values) for name, values in events.items()}


def matched_replay(initials, tape, key):
    base = initials["birth_null"]
    lanes = np.tile(np.arange(tape.good.shape[1]), len(base.bias) // tape.good.shape[1])
    left, right, _, table = tape.present(0, lanes)
    left, right = torch.from_numpy(left), torch.from_numpy(right)
    hidden, logit = full.motor(base, left, right)
    side = torch.where(logit > 0, 1, -1)
    true_taste = torch.from_numpy(table[np.arange(len(lanes)), (side > 0).numpy().astype(int)])
    draws = torch.from_numpy(np.random.default_rng(key).random((len(lanes), 2), dtype=np.float32))
    raw = {"matched_action": side.numpy().astype(np.int8), "matched_taste": true_taste.numpy().astype(np.int8)}
    effects = {}
    for arm, graph in initials.items():
        signed, blank = [full.propose(graph, hidden, left, right, side, taste, draws, view="own")
                         for taste in (true_taste, torch.zeros_like(true_taste))]
        effects[arm] = effect(signed.next_graph, blank.next_graph)
        for name, values in zip(FIELDS, effects[arm]):
            raw[f"{arm}_taste_effect_{name}"] = values
    for control in life.ARMS[1:]:
        raw[f"writer_vs_{control}"] = numeric_distance(*(
            a - b for a, b in zip(effects["writer"], effects[control])
        ))
    return raw


def donor_execution(log, width, end):
    addresses, ops = np.moveaxis(log["command"][:end], -1, 0)
    accepted = log["receipt"][:end] == full.Status.APPLIED
    is_bias = addresses >= width ** 2
    receiving = np.where(is_bias, addresses - width ** 2, addresses // width)
    return {
        "accuracy": float(log["correct"][:end].mean()) if end else None,
        "body_and_writer_accepted_edge_adjustments": [
            int((region & ~is_bias & (ops == full.Op.ADJUST) & accepted).sum())
            for region in (receiving < full.BODY, receiving >= full.BODY)
        ],
        "body_and_writer_accepted_bias_adjustments": [
            int((region & is_bias & (ops == full.Op.ADJUST) & accepted).sum())
            for region in (receiving < full.BODY, receiving >= full.BODY)
        ],
        "body_and_writer_accepted_births": [
            int((region & (ops == full.Op.BIRTH) & accepted).sum())
            for region in (receiving < full.BODY, receiving >= full.BODY)
        ],
        "body_and_writer_accepted_retirements": [
            int((region & (ops == full.Op.RETIRE) & accepted).sum())
            for region in (receiving < full.BODY, receiving >= full.BODY)
        ],
    }


def run_width(plan, donors, receivers, writer_width, raw_directory, evidence):
    births, manifest = load_births(plan, writer_width)
    sizes = (len(plan["birth_roots"]), len(plan["donor_world_roots"]), len(plan["receiver_world_roots"]))
    base = repeat(births, sizes[1])
    keys = plan["editor_key_roots"]
    times = {dose * plan["contacts_per_donor_problem"]: dose for dose in plan["doses"]}
    boot = analysis.CrossedBootstrap(sizes, plan["bootstrap"])
    snapshots = {}
    ecology_count = len(plan["ecology"])
    windows = plan["receiver_receipts_before_action"]
    raw = {**donors.arrays("donor"), **receivers.arrays("receiver")}
    donor_rows, dose_summaries, gains, action_cells = {}, {}, {}, {}
    for arm in ("active", "protect_writer"):
        states, rows = assess(base, donors, keys["donor"], intervention=arm, snapshots=times)
        donor_rows[arm] = rows
        raw.update({f"donor_{arm}_{name}": value for name, value in rows.items()})
        analysis.verify_action_log(raw, f"donor_{arm}", "donor")
        snapshots[arm] = states
        print(f"width {writer_width} {arm} donor paths complete", flush=True)
    for dose in plan["doses"]:
        body = {"actions": {}, "grafts": {}, "contrasts": {}, "donors": {}, "replay": {}}
        for arm, states in snapshots.items():
            graph, hidden = states[dose]
            validate(graph)
            assert connected(graph).all()
            raw.update(graph_arrays(f"d{dose}_{arm}_donor", graph))
            raw[f"d{dose}_{arm}_hidden"] = hidden.numpy()
            body["donors"][arm] = donor_execution(donor_rows[arm], graph.width, dose * plan["contacts_per_donor_problem"])
        writer = graft(base, snapshots["active"][dose][0])
        null = graft(base, snapshots["protect_writer"][dose][0])
        for field in FIELDS:
            np.testing.assert_array_equal(getattr(base, field), getattr(null, field))
        sham = scramble(base, writer, [keys["scramble"], dose])
        body["grafts"]["scramble_vs_writer"] = analysis.count_summary(distance(writer, sham))
        initials = dict(zip(life.ARMS, (writer, null, sham)))
        repeat_initials = {arm: repeat(graph, receivers.good.shape[1]) for arm, graph in initials.items()}
        rows = {}
        for arm, graph in repeat_initials.items():
            _, logs = assess(graph, receivers, keys["receiver"])
            rows[arm] = logs
            raw.update({f"d{dose}_{arm}_{name}": value for name, value in logs.items()})
            raw.update(graph_arrays(f"d{dose}_{arm}_initial", initials[arm]))
            analysis.verify_action_log(raw, f"d{dose}_{arm}", "receiver")
        arrays = {}
        for arm, logs in rows.items():
            correct = logs["correct"][windows].reshape(len(windows), *sizes, ecology_count, 2)
            curve = np.moveaxis(correct.mean(axis=-1), 0, -1)
            np.testing.assert_array_equal(curve[..., 0], 0.5)
            if dose == 0:
                np.testing.assert_array_equal(logs["side"], rows["birth_null"]["side"])
            arrays[arm] = action_cells[f"d{dose}_{arm}"] = curve
            body["actions"][arm] = analysis.action_summary(boot.interval, curve)
            body["grafts"][arm] = {
                "disconnected_lives": int((~connected(initials[arm])).sum()),
                "distance_from_birth": analysis.count_summary(distance(base, initials[arm])),
                "zero_receipt_action_disagreement_with_birth": float(
                    (logs["side"][0] != rows["birth_null"]["side"][0]).mean()),
            }
        witnesses = matched_replay(repeat_initials, receivers, keys["next_edit"])
        raw.update({f"d{dose}_replay_{name}": value for name, value in witnesses.items()})
        gains[dose] = {}
        for control in life.ARMS[1:]:
            name = f"writer_vs_{control}"
            curve = arrays["writer"] - arrays[control]
            gains[dose][control] = analysis.average_acquisition(curve).mean(axis=-1)
            body["contrasts"][name] = analysis.action_summary(boot.interval, curve)
            body["contrasts"][name]["connectivity_sensitivity"] = analysis.connectivity_description(curve, connected(writer))
            body["replay"][name] = analysis.count_summary(witnesses[name])
        dose_summaries[str(dose)] = body
        print(f"width {writer_width} dose {dose} complete", flush=True)
    path = raw_directory / f"expanded-writer-{writer_width}-raw.npz"
    np.savez_compressed(path, **raw)
    cells = evidence / f"expanded-writer-{writer_width}-cells.npz"
    np.savez_compressed(cells, **action_cells)
    primary = {
        "late_vs_null": boot.interval(gains[512]["birth_null"]),
        "late_minus_early_vs_null": boot.interval(gains[512]["birth_null"] - gains[8]["birth_null"]),
        "late_vs_scramble": boot.interval(gains[512]["numeric_scramble"]),
    }
    return {
        "raw": {"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": life.fingerprint(path)},
        "cells_sha256": life.fingerprint(cells),
        "checkpoint_sha256": manifest,
        "primary": primary,
        "positive_rule_met": all(item["ci95"][0] > 0 for item in primary.values()),
        "doses": dose_summaries,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", default=life.RAW, type=Path)
    parser.add_argument("--evidence", default=life.EVIDENCE, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    replication = json.loads(PLAN.read_text(encoding="utf-8"))
    plan = json.loads(life.PLAN.read_text(encoding="utf-8"))
    plan.update(replication)
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    donors, receivers = life.donor_tape(plan), life.receiver_tape(plan)
    life.verify_private_preferences(donors, receivers, plan["contacts_per_donor_problem"])
    report = {
        "base_protocol_sha256": life.fingerprint(life.PLAN),
        "expanded_protocol_sha256": life.fingerprint(PLAN),
        "source_sha256": {Path(p).name: life.fingerprint(Path(p)) for p in (__file__, full.__file__, life.__file__,
                                                                        analysis.__file__)},
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "axes": {"regions": ("body", "writer"), "metrics": ("weight_l2", "bias_l2", "support_hamming"),
                 "receipts": plan["receiver_receipts_before_action"], "ecology": list(plan["ecology"])},
        "writer_widths": {},
    }
    for width in plan["writer_widths"]:
        result = run_width(plan, donors, receivers, width, args.raw_directory, args.evidence)
        report["writer_widths"][str(width)] = result
        print(json.dumps({str(width): result["primary"]}, indent=2), flush=True)
    if report["source_sha256"][Path(full.__file__).name] != life.fingerprint(Path(full.__file__)):
        raise RuntimeError("Expanded source changed while this replication was running")
    (args.evidence / "expanded-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
