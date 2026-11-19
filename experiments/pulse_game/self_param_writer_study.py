"""Reward-only bootstrap and paired gradient-free tests of full graph writing."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import self_param_writer as full
from .self_write_transfer import build_tape


EVIDENCE = Path("docs/evidence/self-edit-dose-2026-11-19/permissions")


def rollout(initial, tape, uniforms, *, stochastic_motor, view="own", intervention="active",
            collect=False):
    graph = initial.repeat(tape.lives)
    scores, rewards, correct = [], [], []
    address_scores, op_scores = [], []
    receipts = Counter()
    for event in range(tape.contacts):
        left, right = map(torch.from_numpy, tape.present(event))
        hidden, logit = full.motor(graph, left, right)
        go_right = logit > 0
        if stochastic_motor:
            go_right = uniforms[:, event, 2] < logit.detach().sigmoid()
        side = torch.where(go_right, 1, -1)
        taste = torch.from_numpy(tape.taste(event, side.numpy()))
        edit = full.propose(graph, hidden, left, right, side, taste, uniforms[:, event, :2],
                            view=view, intervention=intervention)
        scores.append(F.logsigmoid(side * logit))
        address_scores.append(edit.address_log_prob)
        op_scores.append(edit.op_log_prob)
        rewards.append(taste)
        correct.append(side.numpy() == tape.target(event))
        if collect:
            _, after_logit = full.motor(edit.next_graph, left, right)
            goal = torch.from_numpy(tape.target(event))
            gain = (goal * after_logit).sigmoid() - (goal * logit).sigmoid()
            count_receipts(receipts, edit, graph.width, gain, goal * after_logit, goal * logit)
        graph = edit.next_graph
    return tuple(map(lambda fields: torch.stack(fields, dim=1),
                     (scores, address_scores, op_scores, rewards))) + (
                         np.stack(correct, axis=1), graph, dict(receipts))


def count_receipts(counts, edit, width, gain, after_margin, before_margin):
    command, receipt = edit.command, edit.receipt
    edge, receiver, sender = full.decode(command.address, width)
    edge, receiver, sender = edge.numpy(), receiver.numpy(), sender.numpy()
    gain, after_margin, before_margin = gain.numpy(), after_margin.numpy(), before_margin.numpy()
    operations, statuses = command.op.numpy(), receipt.status.numpy()
    op_names = [op.name.lower() for op in full.Op]
    status_names = [status.name.lower() for status in full.Status]
    for lane in range(len(edge)):
        dst = "W" if receiver[lane] >= full.BODY else "B"
        src = "W" if sender[lane] >= full.BODY else "B"
        address_type = src + ">" + dst if edge[lane] else "bias_" + dst
        kinds = [address_type]
        if edge[lane] and receiver[lane] == sender[lane]:
            kinds.append("loop_" + dst)
        op = op_names[operations[lane]]
        status = status_names[statuses[lane]]
        for kind in kinds:
            name = f"{kind}/{op}/{status}"
            counts[name] += 1
            if status == "applied":
                counts[f"{kind}/{op}/mean_same_odor_probability_delta_numerator"] += gain[lane].item()
                counts[f"{kind}/{op}/wrong_to_right"] += int(
                    before_margin[lane] <= 0 and after_margin[lane] > 0)
                counts[f"{kind}/{op}/right_to_wrong"] += int(
                    before_margin[lane] > 0 and after_margin[lane] <= 0)


def gradient_audit(initial, logs, baseline_motor, baseline_writer, write_horizon):
    motor_scores, addr_scores, op_scores, reward = logs
    zero = torch.zeros_like(motor_scores)
    pieces = {
        "motor": (motor_scores, zero, zero),
        "address": (zero, addr_scores, zero),
        "operation": (zero, zero, op_scores),
    }
    audits = {}
    for key, inputs in pieces.items():
        loss = full.policy_loss(*inputs, reward, baseline_motor, baseline_writer,
                                write_horizon=write_horizon)
        w, b = torch.autograd.grad(loss, (initial.weights, initial.bias), retain_graph=True)
        audits[key] = {
            "weight_norm": w.norm().item(),
            "bias_norm": b.norm().item(),
            "writer_incoming_weight_norm": w[:, full.BODY:].norm().item(),
            "writer_bias_norm": b[:, full.BODY:].norm().item(),
            "absent_weight_norm": w[~initial.present].norm().item(),
        }
    return audits


def train(seed, writers, plan):
    contacts = plan["train_contacts"]
    batch = plan["train_batch"]
    write_horizon = plan.get("write_horizon")
    initial = full.spawn(seed, writers)
    initial = full.Graph(nn.Parameter(initial.weights), nn.Parameter(initial.bias), initial.present)
    optimizer = torch.optim.Adam((initial.weights, initial.bias), lr=plan["lr"])
    rng = np.random.default_rng(861_000 + seed)
    baseline_motor = torch.zeros(contacts)
    baseline_writer = torch.zeros(contacts)
    audits = {}
    history = []
    for iteration in range(plan["iterations"]):
        tape = build_tape(int(rng.integers(2**32)), batch, 1, contacts, contacts, 0.0,
                          complementary=True)
        uniforms = torch.from_numpy(rng.random((batch, contacts, 3), dtype=np.float32))
        *logs, _, _, _ = rollout(initial, tape, uniforms, stochastic_motor=True)
        if iteration in (0, plan["iterations"] - 1):
            audits[str(iteration + 1)] = gradient_audit(
                initial, logs, baseline_motor, baseline_writer, write_horizon)
        optimizer.zero_grad(set_to_none=True)
        loss = full.policy_loss(*logs, baseline_motor, baseline_writer, write_horizon=write_horizon)
        loss.backward()
        nn.utils.clip_grad_norm_((initial.weights, initial.bias), 2.0)
        optimizer.step()
        with torch.no_grad():
            initial.weights.clamp_(-full.LIMIT, full.LIMIT)
            initial.bias.clamp_(-full.LIMIT, full.LIMIT)
            baseline_motor.lerp_(full.reward_to_go(logs[-1]).mean(dim=0), 0.10)
            baseline_writer.lerp_(full.reward_to_go(logs[-1], write_horizon).mean(dim=0), 0.10)
        history.append(logs[-1].mean(dim=0).tolist())
        if iteration in (0, plan["iterations"] - 1) or (iteration + 1) % 100 == 0:
            recent = (np.array(history[-50:]).mean(axis=0) + 1) / 2
            print(f"root={seed} writer={writers} iteration={iteration+1} accuracy={recent.round(3)}",
                  flush=True)
    return initial.detach(), {"gradient_audit": audits, "taste_history": history}


def graph_counts(graph):
    width = graph.width
    return {
        "writable_bias_addresses": width,
        "writable_edge_addresses": width * width,
        "live_weights": graph.present.sum().item(),
        "bootstrap_coefficients": width + graph.present.sum().item(),
        "bootstrap_writer_incoming_weights": graph.present[:, full.BODY:].sum().item(),
        "bootstrap_writer_incident_weights": (
            graph.present.sum() - graph.present[:, :full.BODY, :full.BODY].sum()).item(),
        "bootstrap_writer_biases": width - full.BODY,
    }


@torch.inference_mode()
def holdout(initial, plan, *, collect):
    reports = {}
    for condition in plan["holdout"]:
        tape = build_tape(
            condition["seed"], plan["holdout_lives"], condition["odors"],
            condition["rounds"], condition["phase_length"], condition["noise"],
            complementary=True)
        rng = np.random.default_rng(condition["seed"] + 10_000)
        uniforms = torch.from_numpy(rng.random((tape.lives, tape.contacts, 3), dtype=np.float32))
        arms = {}
        for name in condition["arms"]:
            view = "blind" if name == "blind" else "own"
            intervention = "active" if name == "blind" else name
            *_, correct, final, counts = rollout(
                initial, tape, uniforms, stochastic_motor=False, view=view,
                intervention=intervention, collect=collect)
            arms[name] = {
                "curve": correct.mean(axis=0).tolist(),
                "life_accuracy": correct.mean(axis=1).tolist(),
                "first_eight_by_life": correct[:, :8].mean(axis=1).tolist(),
                "last_eight_by_life": correct[:, -8:].mean(axis=1).tolist(),
                "receipts": counts,
                "changed_writer_bias": (final.bias[:, full.BODY:] !=
                                        initial.bias[:, full.BODY:]).sum().item(),
                "changed_writer_incident_weight": ((final.weights != initial.weights).sum() -
                    (final.weights[:, :full.BODY, :full.BODY]
                     != initial.weights[:, :full.BODY, :full.BODY]).sum()).item(),
                "changed_writer_incident_support": ((final.present != initial.present).sum() -
                    (final.present[:, :full.BODY, :full.BODY]
                     != initial.present[:, :full.BODY, :full.BODY]).sum()).item(),
                "saturated_weights": (final.weights.abs() >= full.LIMIT).sum().item(),
            }
        reports[condition["name"]] = arms
    return reports


def write_checkpoint(path, graph):
    np.savez_compressed(path, weights=graph.weights.numpy(), bias=graph.bias.numpy(),
                        present=graph.present.numpy())


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--plan", type=Path, default=EVIDENCE / "protocol.json")
    args.add_argument("--out", type=Path, default=EVIDENCE)
    args.add_argument("--writers", type=int, choices=full.WRITER_WIDTHS, required=True)
    args.add_argument("--roots", type=int, nargs="+")
    options = args.parse_args()
    torch.set_num_threads(1)
    options.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(options.plan.read_text(encoding="utf-8"))
    for root in plan["roots"] if options.roots is None else options.roots:
        path = options.out / f"writer-{options.writers}-root-{root}"
        before = full.spawn(root, options.writers)
        after, training = train(root, options.writers, plan)
        record = {
            "root": root, "writer_width": options.writers, "counts": graph_counts(before),
            "bootstrap": training,
            "before": holdout(before, plan, collect=False),
            "after": holdout(after, plan, collect=True),
        }
        write_checkpoint(path.with_suffix(".npz"), after)
        path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
