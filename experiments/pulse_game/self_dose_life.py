"""Run one uninterrupted donor lifetime and assay only the inherited writer edits."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from . import self_write_core as core
from . import self_write_transfer as transfer


EVIDENCE = Path("docs/evidence/self-edit-dose-2026-11-19/life")
RAW = Path("data/pulse-game/self-edit-dose-2026-11-19/life")
PLAN = EVIDENCE / "preregistered.json"
ARMS = ("writer", "birth_null", "numeric_scramble")
REGIONS = (slice(0, core.WRITER.start), core.WRITER)
FIELDS = ("w", "b", "mask")
EDIT_FIELDS = ("weight_tuned", "weight_changed", "bias_changed", "support_born", "support_retired")


def fingerprint(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def checked_graph(w, b, mask):
    # Old archives and new editor ABIs may store support as integer flags.
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("This reference runtime expects binary support flags")
    graph = core.Graph(w, b, mask.astype(bool))
    validate_graph(graph)
    return graph


def validate_graph(graph):
    if graph.mask.dtype != np.bool_:
        raise AssertionError("Raw graph flags must be decoded before treating them as a mask")
    if np.any(graph.mask & core.DIAGONAL) or np.any(graph.w[~graph.mask]):
        raise AssertionError("Found a self edge or an inactive weight")
    for values in (graph.w, graph.b):
        if not np.isfinite(values).all() or np.any(np.abs(values) > core.LIMIT):
            raise AssertionError("Parameter is invalid or exceeds the runtime bound")


def load_births(plan):
    checkpoints = [Path(plan["birth_checkpoint_template"].format(root=root)) for root in plan["birth_roots"]]
    loaded = []
    for path in checkpoints:
        with np.load(path, allow_pickle=False) as saved:
            graph = checked_graph(saved["w"], saved["b"], saved["mask"])
        if len(graph.w) != 1 or not core.connected(graph.mask[0]):
            raise ValueError(f"Expected one strongly connected archival birth in {path}")
        loaded.append(graph)
    return core.Graph(*(np.concatenate([getattr(graph, f) for graph in loaded]) for f in FIELDS)), {
        path.as_posix(): fingerprint(path) for path in checkpoints
    }


@dataclass(frozen=True)
class Tape:
    odor: np.ndarray
    good: np.ndarray
    ecology: np.ndarray
    orientation: np.ndarray
    flipped: np.ndarray

    def present(self, trial, lanes):
        """The environment returns stimuli and a taste table; only the chosen taste is consumed."""
        left = self.odor[trial, lanes] * self.orientation[trial, lanes, None]
        truth = self.good[trial, lanes] * self.orientation[trial, lanes]
        reported_good = np.where(self.flipped[trial, lanes], -truth, truth)
        return left, -left, truth, np.stack((-reported_good, reported_good), axis=1).astype(np.float32)

    def arrays(self, prefix):
        return {f"{prefix}_{name}": values for name, values in vars(self).items()}


def editor_keys(namespace, roots, length):
    return np.array([
        np.random.default_rng([namespace, int(root)]).integers(0, 2**32, length, dtype=np.uint64)
        for root in roots
    ])


def ecological_arrays(plan):
    regimes = tuple(plan["ecology"].values())
    return np.array([r["radius"] for r in regimes]), np.array([r["noise"] for r in regimes])


def scaled_odors(rng, regime, radii):
    random_directions = rng.uniform(-1, 1, (*regime.shape, 3))
    return (random_directions * (radii[regime] / np.linalg.norm(random_directions, axis=-1))[..., None]).astype(
        np.float32
    )


def donor_tape(plan):
    problems = max(plan["doses"])
    contacts = plan["contacts_per_donor_problem"]
    radii, noise = ecological_arrays(plan)
    rows = []
    for root in plan["donor_world_roots"]:
        rng = np.random.default_rng(root)
        # Each prefix dose is balanced between the two reliability cues.
        ecology = np.concatenate([rng.permutation(np.repeat((0, 1), 4)) for _ in range(problems // 8)])
        odor = scaled_odors(rng, ecology, radii)
        good = rng.choice((-1, 1), problems).astype(np.int8)
        orientation = rng.choice((-1, 1), (problems, contacts)).astype(np.int8)
        flipped = rng.random((problems, contacts)) < noise[ecology, None]
        rows.append(Tape(np.repeat(odor, contacts, axis=0), np.repeat(good, contacts),
                         np.repeat(ecology, contacts), orientation.flatten(), flipped.flatten()))
    return Tape(*(np.stack([getattr(row, name) for row in rows], axis=1) for name in vars(rows[0])))


def receiver_tape(plan):
    contacts = max(plan["receiver_receipts_before_action"]) + 1
    radii, noise = ecological_arrays(plan)
    rows = []
    for root in plan["receiver_world_roots"]:
        rng = np.random.default_rng(root)
        regime = np.arange(len(radii))
        odor = scaled_odors(rng, regime, radii)
        first_good = rng.choice((-1, 1), len(radii)).astype(np.int8)
        good = np.stack((first_good, -first_good), axis=-1).flatten()
        orientation = rng.choice((-1, 1), (contacts, len(radii))).astype(np.int8)
        flipped = rng.random((contacts, len(radii))) < noise[regime]
        rows.append(Tape(
            np.broadcast_to(np.repeat(odor, 2, axis=0), (contacts, good.size, 3)),
            np.broadcast_to(good, (contacts, good.size)),
            np.broadcast_to(np.repeat(regime, 2), (contacts, good.size)),
            np.repeat(orientation, 2, axis=1),
            np.repeat(flipped, 2, axis=1),
        ))
    return Tape(*(np.concatenate([getattr(row, name) for row in rows], axis=1) for name in vars(rows[0])))


def verify_private_preferences(donor, receiver, contact_count):
    def direction_keys(odors):
        return {tuple(odor) for odor in np.concatenate((odors, -odors))}

    donor_odors = donor.odor[::contact_count].reshape(-1, 3)
    receiver_odors = receiver.odor[0, ::2]
    donor_keys, receiver_keys = direction_keys(donor_odors), direction_keys(receiver_odors)
    if len(donor_keys) != 2 * len(donor_odors) or len(receiver_keys) != 2 * len(receiver_odors):
        raise AssertionError("A nominally new odor pair was reused within a tape family")
    if not donor_keys.isdisjoint(receiver_keys):
        raise AssertionError("A donor odor appeared in the cold recipient")
    for name in ("odor", "ecology", "orientation", "flipped"):
        values = getattr(receiver, name)
        np.testing.assert_array_equal(values[:, ::2], values[:, 1::2])
    np.testing.assert_array_equal(receiver.good[:, ::2], -receiver.good[:, 1::2])


def repeat(graph, copies):
    return core.Graph(*(np.repeat(getattr(graph, f), copies, axis=0) for f in FIELDS))


def edit_statistics(before, after, receipt):
    return np.stack([
        np.stack((
            np.count_nonzero(receipt.edge_delta[:, rows], axis=(1, 2)),
            np.count_nonzero(after.w[:, rows] != before.w[:, rows], axis=(1, 2)),
            np.count_nonzero(after.b[:, rows] != before.b[:, rows], axis=1),
            np.count_nonzero(after.mask[:, rows] & ~before.mask[:, rows], axis=(1, 2)),
            np.count_nonzero(before.mask[:, rows] & ~after.mask[:, rows], axis=(1, 2)),
        ), axis=-1)
        for rows in REGIONS
    ], axis=1).astype(np.uint8)


def train(plan, base, tape, arm):
    world_count = len(plan["donor_world_roots"])
    lanes = np.tile(np.arange(world_count), len(plan["birth_roots"]))
    contacts = plan["contacts_per_donor_problem"]
    checkpoints = {dose * contacts: dose for dose in plan["doses"]}
    keys = editor_keys(plan["editor_key_roots"]["donor"], (0,), len(tape.good))[0]
    graph = base.clone()
    hidden = core.rest(graph)
    snapshots = {0: (graph, hidden)}
    events = {name: [] for name in ("side", "taste", "correct", "address", "status", "edits")}
    for event, key in enumerate(keys):
        left, right, truth, tastes = tape.present(event, lanes)
        before = graph
        graph, hidden, side, taste, proposal, receipt = core.step(
            graph, hidden, left, right, tastes, int(key), arm=arm, view="own",
        )
        events["side"].append(side.astype(np.int8))
        events["taste"].append(taste.astype(np.int8))
        events["correct"].append(side == truth)
        events["address"].append(np.stack((proposal.birth, proposal.retire), axis=-1).astype(np.int16))
        events["status"].append(np.stack((receipt.birth_status, receipt.retire_status), axis=-1))
        events["edits"].append(edit_statistics(before, graph, receipt))
        if event + 1 in checkpoints:
            snapshots[checkpoints[event + 1]] = (graph, hidden)
    return snapshots, {name: np.stack(values) for name, values in events.items()}


def assert_same_region(actual, expected, region):
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(actual, field)[:, region], getattr(expected, field)[:, region])


def writer_payload(base, donor):
    if base.w.shape != donor.w.shape:
        raise ValueError("Writer payload requires one bitwise identical-origin birth per donor")
    # Absolute assignment equals a delta transplant from the identical birth and
    # avoids the floating-point round trip at +/- saturation.
    graft = core.transplant_writer(base, donor)
    assert_same_region(graft, base, REGIONS[0])
    assert_same_region(graft, donor, REGIONS[1])
    validate_graph(graft)
    return graft


def structural_audit(graph):
    return {
        "connected": np.array([core.connected(row) for row in graph.mask]),
        "weights_at_limit": np.stack([(np.abs(graph.w[:, rows]) == core.LIMIT).sum(axis=(1, 2))
                                      for rows in REGIONS], axis=-1),
        "biases_at_limit": np.stack([(np.abs(graph.b[:, rows]) == core.LIMIT).sum(axis=1)
                                     for rows in REGIONS], axis=-1),
    }


def parameter_distance(before, after):
    return np.stack([
        np.stack((
            np.linalg.norm((after.w[:, rows] - before.w[:, rows]).reshape(len(before.w), -1), axis=1),
            np.linalg.norm(after.b[:, rows] - before.b[:, rows], axis=1),
            np.count_nonzero(after.mask[:, rows] != before.mask[:, rows], axis=(1, 2)),
        ), axis=-1)
        for rows in REGIONS
    ], axis=1)


def action_state(graph, hidden, left, right):
    next_state = core.advance(graph, hidden, core.sense(left, right), ticks=core.MOTOR_TICKS)
    motor = core.motor(next_state)
    return next_state, motor, np.where(motor > 0, 1, -1)


def follow(initial, tape, key_root):
    lanes = np.tile(np.arange(tape.good.shape[1]), len(initial.w) // tape.good.shape[1])
    hidden = core.rest(initial)
    graph = initial
    events = {name: [] for name in ("side", "taste", "correct", "motor", "edits", "address", "status")}
    keys = editor_keys(key_root, (0,), len(tape.good))[0]
    for trial, key in enumerate(keys):
        left, right, truth, tastes = tape.present(trial, lanes)
        _, motor, side = action_state(graph, hidden, left, right)
        events["side"].append(side.astype(np.int8))
        events["correct"].append(side == truth)
        events["motor"].append(motor)
        if trial + 1 < len(keys):
            before = graph
            graph, hidden, made_side, taste, proposal, receipt = core.step(
                graph, hidden, left, right, tastes, int(key), arm="active", view="own",
            )
            np.testing.assert_array_equal(made_side, side)
            events["taste"].append(taste.astype(np.int8))
            events["edits"].append(edit_statistics(before, graph, receipt))
            events["address"].append(np.stack((proposal.birth, proposal.retire), axis=-1).astype(np.int16))
            events["status"].append(np.stack((receipt.birth_status, receipt.retire_status), axis=-1))
    return {name: np.stack(values) for name, values in events.items()}


def receipt_replay(graph, preact, left, right, action, taste, key):
    def apply_feedback(feedback):
        pre, post = core.contact_states(graph, preact, left, right, action, feedback)
        proposal = core.proposal(graph, pre, post, key, view="own")
        committed, receipt = core.apply(graph, proposal, arm="active")
        return committed, np.stack((proposal.birth, proposal.retire), axis=-1).astype(np.int16), np.stack(
            (receipt.birth_status, receipt.retire_status), axis=-1
        )

    signed, signed_address, signed_status = apply_feedback(taste)
    blank, blank_address, blank_status = apply_feedback(np.zeros_like(taste))
    effect = tuple(
        getattr(signed, f).astype(np.int8 if f == "mask" else np.float32)
        - getattr(blank, f).astype(np.int8 if f == "mask" else np.float32)
        for f in FIELDS
    )
    return effect, np.stack((signed_address, blank_address)), np.stack((signed_status, blank_status))


def replay_effect_difference(main, control):
    weight, bias, support = (after - before for after, before in zip(main, control))
    return np.stack([
        np.stack((np.linalg.norm(weight[:, rows].reshape(len(weight), -1), axis=1),
                  np.linalg.norm(bias[:, rows], axis=1),
                  np.count_nonzero(support[:, rows], axis=(1, 2))), axis=-1)
        for rows in REGIONS
    ], axis=1)


def assay(plan, base, initials, tape):
    copies = tape.good.shape[1]
    start = repeat(base, copies)
    lanes = np.tile(np.arange(copies), len(base.w))
    left, right, _, tastes = tape.present(0, lanes)
    preact, _, action = action_state(start, core.rest(start), left, right)
    signed = tastes[np.arange(len(tastes)), (action > 0).astype(int)]
    key = int(editor_keys(plan["editor_key_roots"]["next_edit"], (0,), 1)[0, 0])
    histories = {}
    replay = {"matched_receipt": signed.astype(np.int8), "matched_action": action.astype(np.int8)}
    effects = {}
    for name, initial in initials.items():
        graph = repeat(initial, copies)
        histories[name] = follow(graph, tape, plan["editor_key_roots"]["receiver"])
        effect, address, status = receipt_replay(graph, preact, left, right, action, signed, key)
        effects[name] = effect
        for field, values in zip(FIELDS, effect):
            replay[f"{name}_taste_effect_{field}"] = values
        replay[f"{name}_replay_address"] = address
        replay[f"{name}_replay_status"] = status
    for control in ("birth_null", "numeric_scramble"):
        replay[f"writer_vs_{control}_replay_difference"] = replay_effect_difference(effects["writer"], effects[control])
    return histories, replay


def collect(plan, save_to):
    births, source_manifest = load_births(plan)
    base = repeat(births, len(plan["donor_world_roots"]))
    donor, receiver = donor_tape(plan), receiver_tape(plan)
    verify_private_preferences(donor, receiver, plan["contacts_per_donor_problem"])
    active, active_rows = train(plan, base, donor, "active")
    print("Uninterrupted active donor paths complete", flush=True)
    protected, protected_rows = train(plan, base, donor, "primitive")
    print("Paired writer-protected donor paths complete", flush=True)
    arrays = {**donor.arrays("donor"), **receiver.arrays("receiver")}
    for arm, logs in (("active", active_rows), ("protected", protected_rows)):
        arrays.update({f"donor_{arm}_{name}": values for name, values in logs.items()})
    for dose in plan["doses"]:
        live, live_hidden = active[dose]
        frozen, frozen_hidden = protected[dose]
        for arm, state, hidden in (("active", live, live_hidden), ("protected", frozen, frozen_hidden)):
            prefix = f"d{dose}_donor_{arm}"
            validate_graph(state)
            arrays.update({f"{prefix}_{field}": getattr(state, field) for field in FIELDS})
            arrays[f"{prefix}_hidden"] = hidden
            arrays.update({f"{prefix}_{field}": values for field, values in structural_audit(state).items()})
        graft = writer_payload(base, live)
        null = writer_payload(base, frozen)
        assert_same_region(null, base, slice(None))
        scramble = transfer.scramble_writer(base, graft, [plan["editor_key_roots"]["scramble"], dose])
        validate_graph(scramble)
        assert_same_region(scramble, base, REGIONS[0])
        np.testing.assert_array_equal(scramble.mask, graft.mask)
        initials = dict(zip(ARMS, (graft, null, scramble)))
        for name, graph in initials.items():
            prefix = f"d{dose}_initial_{name}"
            arrays.update({f"{prefix}_{field}": getattr(graph, field) for field in FIELDS})
            arrays.update({f"{prefix}_{field}": values for field, values in structural_audit(graph).items()})
            arrays[f"{prefix}_distance_from_birth"] = parameter_distance(base, graph)
        arrays[f"d{dose}_scramble_distance_from_graft"] = parameter_distance(graft, scramble)
        histories, replay = assay(plan, base, initials, receiver)
        for name, logs in histories.items():
            arrays.update({f"d{dose}_{name}_{field}": values for field, values in logs.items()})
            if dose == 0:
                for field, values in histories["birth_null"].items():
                    np.testing.assert_array_equal(logs[field], values)
        arrays.update({f"d{dose}_{field}": values for field, values in replay.items()})
        print(f"Dose {dose} complete", flush=True)
    path = save_to / "raw.npz"
    save_to.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path, source_manifest


def main():
    from . import self_dose_life_analysis as analysis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--raw-directory", type=Path, default=RAW)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    path, source_manifest = collect(plan, args.raw_directory)
    analysis.report(path, plan, args.evidence, {
        "preregistered_plan_sha256": fingerprint(args.plan),
        "checkpoint_sha256": source_manifest,
        "source_sha256": {Path(p).name: fingerprint(Path(p)) for p in (__file__, core.__file__, transfer.__file__,
                                                                     analysis.__file__)},
        "numpy_version": np.__version__,
    })


if __name__ == "__main__":
    main()
