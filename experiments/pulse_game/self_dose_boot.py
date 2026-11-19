"""Matched gradient horizon and world-history diversity in the native self-writing graph."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from . import self_write_core as core
from .self_write_core_bootstrap import calculate, tensor


EVIDENCE = Path("docs/evidence/self-edit-dose-2026-11-19/bootstrap")
ROOTED_FILES = ("self_write_core.py", "self_write_core_bootstrap.py", "self_dose_boot.py")
ARMS = tuple((bank, credit) for bank in ("low", "high") for credit in ("short", "long"))


def digest_arrays(*arrays):
    digest = sha256()
    for array in arrays:
        digest.update(f"{array.dtype}:{array.shape}:".encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def split_rng(plan, split, root, scenario=0):
    return np.random.default_rng(np.random.SeedSequence([plan["world_roots"][split], root, scenario]))


def world(rng, histories, contacts, pairs, noise, flip_fraction, flip_after):
    cues = rng.normal(size=(histories, pairs, 3))
    cues = (1.35 * cues / np.linalg.norm(cues, axis=2, keepdims=True)).astype(np.float32)
    good = rng.choice(np.array([-1, 1], dtype=np.int8), (histories, pairs))
    flip = rng.random(histories) < flip_fraction
    direction = np.repeat(good[:, None], contacts, axis=1)
    if flip_after is not None:
        direction[flip, flip_after:] *= -1
    # Every pair gets an equal number of opportunities when contacts is divisible by pairs.
    order = rng.random((histories, (contacts + pairs - 1) // pairs, pairs)).argsort(axis=-1)
    order = order.reshape(histories, -1)[:, :contacts].astype(np.int8)
    orientation = rng.choice(np.array([-1, 1], dtype=np.int8), (histories, contacts))
    delivery = np.where(rng.random((histories, contacts)) < noise, -1, 1).astype(np.int8)
    return cues, direction, order, orientation, delivery


def make_training(plan, root):
    steps = plan["train_steps"]
    batch = plan["batch"]
    high_count = steps * batch
    bank = world(split_rng(plan, "train", root), high_count, plan["train_contacts"],
                 plan["train_pairs"], plan["train_noise"], plan["train_flip_fraction"],
                 plan["train_flip_after"])
    draw = split_rng(plan, "schedule", root)
    schedules = {"low": draw.integers(plan["low_histories"], size=(steps, batch)),
                 "high": draw.permutation(high_count).reshape(steps, batch)}
    keys = split_rng(plan, "train_receipt", root).integers(2**32, size=(steps, plan["train_contacts"]))
    return bank, schedules, keys


def physical_contact(tape, step):
    cues, direction, order, orientation, delivery = tape
    rows = np.arange(len(cues))
    cue = cues[rows, order[:, step]] * orientation[:, step, None]
    target = direction[rows, step, order[:, step]] * orientation[:, step]
    delivered = delivery[:, step] * target
    taste_sides = np.stack((-delivered, delivered), axis=1).astype(np.float32)
    return cue, target, taste_sides


def split_report(bank, schedule):
    used, frequencies = np.unique(schedule, return_counts=True)
    unique_tape = tuple(values[used] for values in bank)
    histories = len({digest_arrays(*(array[i] for array in unique_tape)) for i in range(len(used))})
    contacts = bank[2].shape[1]
    # A prefix carries cues, private assignment, ordered presentations and delivered taste rule.
    prefixes = {digest_arrays(unique_tape[0][i], unique_tape[1][i, :t], unique_tape[2][i, :t],
                              unique_tape[3][i, :t], unique_tape[4][i, :t])
                for i in range(len(used)) for t in range(1, contacts + 1)}
    return {"used_histories": histories, "unique_environmental_prefixes": len(prefixes),
            "history_visits": int(schedule.size), "repeat_min_max": [int(frequencies.min()), int(frequencies.max())],
            "tape_sha256": digest_arrays(*unique_tape), "schedule_sha256": digest_arrays(schedule)}


def edit_step(w, b, mask, pre, post, key):
    graph = core.Graph(w.detach().numpy(), b.detach().numpy(), mask)
    support = tensor(mask)
    local, gate, edge, bias = core.write_fields(pre, post, w, b, support, torch.tanh)
    addresses = core.structural_addresses(
        graph, post.detach().numpy(), local.detach().numpy(), gate.detach().numpy(),
        *core.view_arrays(graph, "own", None), key,
    )
    edit = core.Proposal(edge.detach().numpy(), bias.detach().numpy(), *addresses)
    actual, receipt = core.apply(graph, edit)
    edge_value = edge * (support & (edge.abs() >= core.EPSILON))
    bias_value = bias * (bias.abs() >= core.EPSILON)
    step_w = w + edge_value
    step_b = b + bias_value
    intended_w = step_w.clamp(-core.LIMIT, core.LIMIT)
    intended_b = step_b.clamp(-core.LIMIT, core.LIMIT)
    lanes = np.flatnonzero(receipt.birth_status == 1)
    if lanes.size:
        receiver, source = np.divmod(edit.birth[lanes], core.N)
        intended_w[lanes, receiver, source] = core.BIRTH_STEP * torch.tanh(
            2.0 * (local[lanes, receiver, source] + gate[lanes]),
        )
    intended_w = intended_w * tensor(actual.mask)
    # The NumPy execution receipt sets the exact forward values in both horizon arms.
    next_w = tensor(actual.w) + (intended_w - intended_w.detach())
    next_b = tensor(actual.b) + (intended_b - intended_b.detach())
    pass_w = (step_w.detach().abs().numpy() <= core.LIMIT) & actual.mask
    pass_b = step_b.detach().abs().numpy() <= core.LIMIT
    if lanes.size:
        pass_w[lanes, receiver, source] = False
    return next_w, next_b, actual.mask, edit, receipt, pass_w, pass_b


def probes(w, b, stimuli):
    h = torch.zeros_like(stimuli)
    drive = b[:, :, None] + stimuli
    for _ in range(core.MOTOR_TICKS):
        h = (1.0 - core.LEAK) * h + core.LEAK * torch.tanh(w @ h + drive)
    return h[:, 11] - h[:, 10]


def edit_metrics(before_w, before_b, before_mask, after_w, after_b, after_mask, edit, receipt,
                 still_initial):
    writer = core.WRITER
    born = receipt.birth_status == 1
    retired = receipt.retire_status == 1
    surviving = before_mask & after_mask
    changed_w = (after_w != before_w) & surviving
    changed_b = after_b != before_b
    return np.stack((
        changed_w.sum(axis=(1, 2)), changed_b.sum(axis=1), born, retired,
        changed_w[:, writer].sum(axis=(1, 2)), changed_b[:, writer].sum(axis=1),
        born & (edit.birth // core.N >= writer.start),
        retired & (edit.retire // core.N >= writer.start),
        np.sum((np.abs(after_w) >= core.LIMIT) & after_mask, axis=(1, 2)),
        np.sum(changed_w & ~still_initial, axis=(1, 2)),
    ), axis=1)


EDIT_METRICS = ("continuous_edge_changes", "continuous_bias_changes", "births", "retirements",
                "writer_edge_changes", "writer_bias_changes", "writer_births", "writer_retirements",
                "saturated_edge_contacts", "post_birth_continuous_edits")


def training_loss(w0, b0, start_mask, tape, keys, credit, audit=None):
    cues, direction, order, _, _ = tape
    count, contacts = order.shape
    h = torch.zeros((count, core.N))
    mask = np.broadcast_to(start_mask, (count, core.N, core.N)).copy()
    native_w = mask.copy()
    still_initial = mask.copy()
    native_b = np.ones((count, core.N), dtype=bool)
    base_w = w0.expand(mask.shape)
    base_b = b0.expand_as(h)
    w = base_w * tensor(mask)
    b = base_b
    both_cues = np.concatenate((cues, -cues), axis=1)
    inputs = tensor(core.sense(both_cues.reshape(-1, 3), -both_cues.reshape(-1, 3))
                    .reshape(count, both_cues.shape[1], core.N).transpose(0, 2, 1).copy())
    scores, actions, realized_sides, realized_tastes, accepted_structure, edits = [], [], [], [], [], []
    seen = np.zeros_like(direction[:, 0], dtype=bool)
    for step, key in enumerate(keys):
        left, target, tastes = physical_contact(tape, step)
        h = calculate(w, b, h, tensor(core.sense(left, -left)), core.MOTOR_TICKS)
        side = np.where(core.motor(h.detach().numpy()) > 0, 1, -1)
        taste = tastes[np.arange(count), (side > 0).astype(int)]
        pre = calculate(w, b, h, tensor(core.sense(left, -left, side)), core.CONTACT_TICKS)
        post = calculate(w, b, pre, tensor(core.sense(left, -left, side, taste)), core.TASTE_TICKS)
        next_w, next_b, new_mask, edit, receipt, pass_w, pass_b = edit_step(w, b, mask, pre, post, int(key))
        counts = edit_metrics(w.detach().numpy(), b.detach().numpy(), mask, next_w.detach().numpy(),
                              next_b.detach().numpy(), new_mask, edit, receipt, still_initial)
        still_initial &= new_mask
        edits.append(counts.sum(axis=0))
        actions.append(side == target)
        realized_sides.append(side.astype(np.int8))
        realized_tastes.append(taste.astype(np.int8))
        accepted_structure.append(np.stack((np.where(receipt.birth_status == 1, edit.birth, -1),
                                             np.where(receipt.retire_status == 1, edit.retire, -1)),
                                            axis=1).astype(np.int16))
        directions = tensor(np.concatenate((direction[:, step], -direction[:, step]), axis=1))
        score = directions * probes(next_w, next_b, inputs)
        seen[np.arange(count), order[:, step]] = True
        included = tensor(np.concatenate((seen, seen), axis=1))
        scores.append(score[included])
        if audit is not None:
            audit.append({"w": next_w, "b": next_b, "mask": new_mask, "h": post, "score": score,
                          "side": side, "taste": taste, "birth": edit.birth.copy(),
                          "birth_status": receipt.birth_status.copy(), "retire": edit.retire.copy(),
                          "retire_status": receipt.retire_status.copy()})
        h = post.detach()
        if credit == "short":
            native_w &= pass_w
            native_b &= pass_b
            # Keep direct sensitivity to surviving birth values; drop all earlier edit Jacobians.
            w = next_w.detach() + (base_w - base_w.detach()) * tensor(native_w)
            b = next_b.detach() + (base_b - base_b.detach()) * tensor(native_b)
        else:
            w, b = next_w, next_b
        mask = new_mask
    losses = torch.stack([nn.functional.softplus(-5.0 * score).mean() for score in scores])
    return losses.mean(), {
        "actual_correct": np.asarray(actions).sum(axis=1),
        "probe_accuracy": [(score.detach() > 0).float().mean().item() for score in scores],
        "edits": np.asarray(edits),
        "sides": np.asarray(realized_sides),
        "tastes": np.asarray(realized_tastes),
        "accepted_structure": np.asarray(accepted_structure),
    }


def train(plan, root, bank_name, credit, bank, schedules, keys):
    initial = core.spawn_graph(root)
    mask = initial.mask[0]
    w = nn.Parameter(tensor(initial.w[0].copy()))
    b = nn.Parameter(tensor(initial.b[0].copy()))
    optimizer = torch.optim.Adam([w, b], lr=plan["learning_rate"])
    losses, forward, backward, gradients, metrics = [], [], [], [], []
    schedule = schedules[bank_name]
    for step, indices in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        tape = tuple(array[indices] for array in bank)
        begin = perf_counter()
        task_loss, record = training_loss(w, b, mask, tape, keys[step], credit)
        stop_forward = perf_counter()
        penalty = plan["regularizer"] * (w.square().mean() + b.square().mean())
        (task_loss + penalty).backward()
        norm = nn.utils.clip_grad_norm_([w, b], plan["clip_grad"])
        optimizer.step()
        with torch.no_grad():
            w.clamp_(-core.LIMIT, core.LIMIT)
            b.clamp_(-core.LIMIT, core.LIMIT)
        end = perf_counter()
        losses.append(task_loss.item())
        gradients.append(norm.item())
        forward.append(stop_forward - begin)
        backward.append(end - stop_forward)
        metrics.append(record)
        if (step + 1) % 60 == 0 or step + 1 == len(schedule):
            print(f"{root} {bank_name}/{credit} step {step + 1}, "
                  f"loss {np.mean(losses[-20:]):.3f}, "
                  f"seconds {(np.sum(forward) + np.sum(backward)):.1f}", flush=True)
    return {
        "w": (w.detach().numpy() * mask)[None], "b": b.detach().numpy()[None],
        "mask": initial.mask, "history_schedule": schedule, "loss": np.asarray(losses),
        "forward_s": np.asarray(forward),
        "backward_and_optimizer_s": np.asarray(backward), "gradient_norm_before_clip": np.asarray(gradients),
        **{field: np.stack([record[field] for record in metrics]) for field in metrics[0]},
    }


def run(evidence, root, steps=None):
    torch.set_num_threads(1)
    protocol = (evidence / "protocol.json").read_bytes()
    plan = json.loads(protocol)
    if root not in plan["birth_roots"]:
        raise ValueError("The birth root must belong to the preregistered bank.")
    if steps is not None:
        plan["train_steps"] = steps
        plan["low_histories"] = min(plan["low_histories"], steps * plan["batch"])
    bank, schedules, keys = make_training(plan, root)
    prefix = "smoke" if steps is not None else "train"
    sources = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ROOTED_FILES}
    report = {"root": root, "protocol_sha256": sha256(protocol).hexdigest(), "sources": sources,
              "root_spawn_sha256": digest_arrays(*vars(core.spawn_graph(root)).values()),
              "train_receipt_schedule_sha256": digest_arrays(keys), "train_steps": plan["train_steps"],
              "train_contacts_per_arm": plan["train_steps"] * plan["batch"] * plan["train_contacts"],
              "banks": {k: split_report(bank, schedule) for k, schedule in schedules.items()}, "checkpoints": {}}
    for bank_name, credit in ARMS:
        path = evidence / f"{prefix}-{root}-{bank_name}-{credit}.npz"
        result = train(plan, root, bank_name, credit, bank, schedules, keys)
        np.savez_compressed(path, **result)
        report["checkpoints"][path.name] = sha256(path.read_bytes()).hexdigest()
    (evidence / f"{prefix}-{root}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    run(args.evidence, args.root, args.smoke_steps)


if __name__ == "__main__":
    main()
