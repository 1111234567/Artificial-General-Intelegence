"""Verify forward equality, receipt parity, and long-path credit before interpreting the dose."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from . import self_write_core as core
from .self_dose_boot import EVIDENCE, digest_arrays, make_training, physical_contact, split_rng, tensor, training_loss, world
from .self_dose_boot_eval import test_worlds


def chain(initial, tape, keys, credit):
    w = tensor(initial.w[0].copy()).requires_grad_()
    b = tensor(initial.b[0].copy()).requires_grad_()
    trace = []
    training_loss(w, b, initial.mask[0], tape, keys, credit, trace)
    last = nn.functional.softplus(-5.0 * trace[-1]["score"]).mean()
    return trace, torch.autograd.grad(last, (trace[0]["w"], trace[0]["b"], w, b), allow_unused=True)


def grad_norm(gradient):
    return None if gradient is None else float(gradient.norm().item())


def audit_paths(plan, root):
    count = 128
    contacts = plan["train_contacts"]
    tape = world(split_rng(plan, "audit", root), count, contacts, plan["train_pairs"], 0, 0, None)
    keys = split_rng(plan, "audit", root, 1).integers(2**32, size=contacts)
    initial = core.spawn_graph(root)
    short_trace, short_gradients = chain(initial, tape, keys, "short")
    long_trace, long_gradients = chain(initial, tape, keys, "long")
    for short, long in zip(short_trace, long_trace, strict=True):
        for key in short:
            np.testing.assert_array_equal(np.asarray(short[key].detach() if torch.is_tensor(short[key]) else short[key]),
                                          np.asarray(long[key].detach() if torch.is_tensor(long[key]) else long[key]))
    first, last = long_trace[0], long_trace[-1]
    lanes = np.flatnonzero(first["birth_status"] == 1)
    row, col = np.divmod(first["birth"][lanes], core.N)
    still_live = np.logical_and.reduce([step["mask"][lanes, row, col] for step in long_trace])
    birth_grads = long_gradients[0].detach().numpy()[lanes, row, col]
    births = sum(int((record["birth_status"] == 1).sum()) for record in long_trace)
    retirements = sum(int((record["retire_status"] == 1).sum()) for record in long_trace)
    assert births and retirements
    assert short_gradients[0] is None and short_gradients[1] is None
    assert all(grad_norm(gradient) > 0 for gradient in (*short_gradients[2:], *long_gradients))
    assert np.any(still_live & (birth_grads != 0))

    graph = core.repeated_graph(initial, count)
    h = core.rest(graph)
    discrepancies = {label: 0.0 for label in ("w", "b", "h")}
    for step, (key, record) in enumerate(zip(keys, long_trace, strict=True)):
        left, _, taste_sides = physical_contact(tape, step)
        graph, h, side, taste, edit, receipt = core.step(graph, h, left, -left, taste_sides, int(key), view="own")
        for label, native, torch_value in (
                ("w", graph.w, record["w"]), ("b", graph.b, record["b"]), ("h", h, record["h"])):
            error = np.max(np.abs(native - torch_value.detach().numpy()))
            discrepancies[label] = max(discrepancies[label], float(error))
        for label, native in zip(("side", "taste", "birth", "retire", "birth_status", "retire_status", "mask"),
                                 (side, taste, edit.birth, edit.retire, receipt.birth_status, receipt.retire_status, graph.mask),
                                 strict=True):
            np.testing.assert_array_equal(native, record[label])
    if max(discrepancies.values()) > 1e-4:
        raise AssertionError(f"Training/deploy value mismatch: {discrepancies}")
    return {"root": root, "tape_sha256": digest_arrays(*tape), "keys_sha256": digest_arrays(keys),
            "matched_bitwise": len(keys) * count, "committed_births": births, "committed_retirements": retirements,
            "first_write_born_and_surviving": int(still_live.sum()),
            "surviving_born_slots_with_nonzero_gradient": int(np.sum(still_live & (birth_grads != 0))),
            "long_gradient_max_to_surviving_born_weight": float(np.max(np.abs(birth_grads[still_live]))),
            "gradient_targets": ["first_committed_w", "first_committed_b", "initial_w", "initial_b"],
            "short_gradient_norms": list(map(grad_norm, short_gradients)),
            "long_gradient_norms": list(map(grad_norm, long_gradients)),
            "native_discrete_parity": True, "native_max_abs_errors": discrepancies}


def flops_profile(plan, root):
    bank, schedules, keys = make_training(plan, root)
    tape = tuple(array[schedules["high"][0]] for array in bank)
    initial = core.spawn_graph(root)
    result = {}
    for credit in ("short", "long"):
        w = tensor(initial.w[0].copy()).requires_grad_()
        b = tensor(initial.b[0].copy()).requires_grad_()
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as recorder:
            loss, _ = training_loss(w, b, initial.mask[0], tape, keys[0], credit)
            loss.backward()
        result[credit] = {event.key: event.flops for event in recorder.key_averages() if event.flops}
    return result


def split_audit(plan):
    actual_cues = {name: set() for name in ("train", "development", "held_out", "recipient")}
    actual_keys = {name: set() for name in actual_cues}
    for root in plan["birth_roots"]:
        bank, _, train_keys = make_training(plan, root)
        actual_cues["train"].update(cue.tobytes() for cue in bank[0].reshape(-1, 3))
        actual_keys["train"].update(train_keys.ravel().tolist())
        for split in ("development", "held_out", "recipient"):
            scenarios, keys = test_worlds(plan, split, root)
            actual_cues[split].update(cue.tobytes() for cue in scenarios["stable_clean"][0].reshape(-1, 3))
            actual_keys[split].update(keys.tolist())
    results = {}
    for i, a in enumerate(actual_cues):
        for b in list(actual_cues)[i + 1:]:
            a_cues, b_cues = actual_cues[a], actual_cues[b]
            results[f"{a}|{b}"] = {"cue_overlap": len(a_cues & b_cues),
                                  "receipt_key_overlap": len(actual_keys[a] & actual_keys[b])}
            assert not (a_cues & b_cues)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = (args.evidence / "protocol.json").read_bytes()
    plan = json.loads(protocol)
    audit = {"protocol_sha256": sha256(protocol).hexdigest(),
             "audit_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
             "torch_version": torch.__version__,
             "split_overlap": split_audit(plan),
             "paths": [audit_paths(plan, root) for root in plan["birth_roots"]],
             "cpu_profile_partial_flops_by_op": flops_profile(plan, plan["birth_roots"][0])}
    (args.evidence / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Same forwards and real receipts; only long credit reaches early commits.", flush=True)


if __name__ == "__main__":
    main()
