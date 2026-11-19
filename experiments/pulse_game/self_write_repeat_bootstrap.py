"""Train a birth graph through six rounds of the same simple two-odor contact."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import self_write_core as core
from .self_write_core_bootstrap import calculate, pair_examples, tensor


EVIDENCE = Path("docs/evidence/self-writing-2026-11-19/repeat")


def differentiable_write(w, b, mask, pre, post, key):
    """Use native discrete addresses and acceptance, with gradients on continuous edits."""

    visible_w = w.detach().numpy()
    visible_b = b.detach().numpy()
    graph = core.Graph(visible_w, visible_b, mask)
    support = tensor(mask)
    local, gate, edge, bias = core.write_fields(pre, post, w, b, support, torch.tanh)
    address = core.structural_addresses(
        graph, post.detach().numpy(), local.detach().numpy(), gate.detach().numpy(),
        *core.view_arrays(graph, "own", None), key,
    )
    edit = core.Proposal(edge.detach().numpy(), bias.detach().numpy(), *address)
    actual, receipt = core.apply(graph, edit)
    accepted_edge = edge * (support & (edge.abs() >= core.EPSILON))
    accepted_bias = bias * (bias.abs() >= core.EPSILON)
    next_w = (w + accepted_edge).clamp(-core.LIMIT, core.LIMIT)
    next_b = (b + accepted_bias).clamp(-core.LIMIT, core.LIMIT)
    lanes = np.flatnonzero(receipt.birth_status == 1)
    if lanes.size:
        receiver, source = np.divmod(edit.birth[lanes], core.N)
        next_w[lanes, receiver, source] = core.BIRTH_STEP * torch.tanh(
            2.0 * (local[lanes, receiver, source] + gate[lanes]),
        )
    return next_w * tensor(actual.mask), next_b, actual.mask


def repeated_loss(w, b, starting_mask, examples, signs, key):
    left_base, _, direction, _ = examples
    count = len(left_base)
    rest = torch.zeros((count, core.N))
    h = rest
    mask = np.broadcast_to(starting_mask, (count, *starting_mask.shape)).copy()
    w = w.expand(mask.shape) * tensor(mask)
    b = b.expand_as(rest)

    both_stimuli = tensor(np.concatenate((core.sense(left_base, -left_base), core.sense(-left_base, left_base))))
    both_directions = tensor(np.concatenate((direction, -direction)))
    both_rest = torch.cat((rest, rest))
    scores = []
    for step, sign in enumerate(signs):
        left = left_base * sign[:, None]
        right = -left
        h = calculate(w, b, h, tensor(core.sense(left, right)), core.MOTOR_TICKS)
        side = np.where(core.motor(h.detach().numpy()) > 0, 1, -1)
        taste = np.where(side == direction * sign, 1.0, -1.0)
        pre = calculate(w, b, h, tensor(core.sense(left, right, side)), core.CONTACT_TICKS)
        h = calculate(w, b, pre, tensor(core.sense(left, right, side, taste)), core.TASTE_TICKS)
        w, b, mask = differentiable_write(w, b, mask, pre, h, key + step)
        probed = calculate(torch.cat((w, w)), torch.cat((b, b)), both_rest, both_stimuli, core.MOTOR_TICKS)
        scores.append(both_directions * core.motor(probed))
    margins = torch.stack(scores)
    return nn.functional.softplus(-5.0 * margins).mean(), (margins.detach() > 0).float().mean(axis=1)


def train(root, plan):
    torch.set_num_threads(1)
    graph = core.spawn_graph(root)
    mask = graph.mask[0]
    rng = np.random.default_rng(root + plan["training_rng_offset"])
    w = nn.Parameter(tensor(graph.w[0]))
    b = nn.Parameter(tensor(graph.b[0]))
    optimizer = torch.optim.Adam([w, b], lr=0.007)
    trace = []
    for iteration in range(1, plan["train_iterations"] + 1):
        pairs = pair_examples(rng, plan["batch"])
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), (plan["train_contacts"], plan["batch"]))
        key = int(rng.integers(2**32))
        optimizer.zero_grad(set_to_none=True)
        loss, correct = repeated_loss(w, b, mask, pairs, signs, key)
        (loss + 0.0012 * (w.square().mean() + b.square().mean())).backward()
        nn.utils.clip_grad_norm_([w, b], 5.0)
        optimizer.step()
        with torch.no_grad():
            w.clamp_(-core.LIMIT, core.LIMIT)
            b.clamp_(-core.LIMIT, core.LIMIT)
        trace.append([loss.item(), *correct.tolist()])
        if iteration % 100 == 0 or iteration == plan["train_iterations"]:
            print(f"root {root}, iteration {iteration}, loss {loss.item():.4f}, correct {correct.tolist()}", flush=True)
    trained_w = (w.detach().numpy() * mask)[None]
    return core.Graph(trained_w.copy(), b.detach().numpy()[None].copy(), mask[None].copy()), np.array(trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    plan = json.loads((args.evidence / "protocol.json").read_bytes())
    if args.root not in plan["roots"]:
        raise ValueError("This birth seed is outside the fixed experiment")
    graph, trace = train(args.root, plan)
    np.savez_compressed(args.evidence / f"bootstrap-repeat-{args.root}.npz", w=graph.w, b=graph.b, mask=graph.mask,
                        root=args.root, train_trace=trace)


if __name__ == "__main__":
    main()
