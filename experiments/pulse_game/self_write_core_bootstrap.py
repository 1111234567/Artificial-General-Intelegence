"""Outer-loop bootstrap on one-contact, two-odor discrimination only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import self_write_core as core


EVIDENCE = Path("docs/evidence/self-writing-2026-11-19/core")


def pair_examples(rng, batch):
    vector = rng.normal(size=(batch, 3))
    vector = (1.35 * vector / np.linalg.norm(vector, axis=1, keepdims=True)).astype(np.float32)
    good = rng.choice(np.array([-1, 1]), size=batch)
    orientation = rng.choice(np.array([-1, 1]), size=batch)
    left = vector * orientation[:, None].astype(np.float32)
    right = -left
    target = (-good * orientation).astype(np.float32)
    taste_by_side = np.stack((-target, target), axis=1)
    return left, right, target, taste_by_side


def calculate(w, b, h, stimulus, ticks):
    drive = b + stimulus
    for _ in range(ticks):
        h = (1.0 - core.LEAK) * h + core.LEAK * torch.tanh(
            (w @ h.unsqueeze(-1)).squeeze(-1) + drive)
    return h


def tensor(array):
    return torch.from_numpy(array)


def training_trial(w, b, mask, examples, seed, view, sparse=False, field_fn=core.write_fields):
    left, right, target, taste_by_side = examples
    h = torch.zeros((len(left), core.N))
    w = w * tensor(mask)
    query = calculate(w, b, h, tensor(core.sense(left, right)), core.MOTOR_TICKS)
    side = np.where(core.motor(query.detach().numpy()) > 0, 1, -1)
    tastes = taste_by_side[np.arange(len(left)), (side > 0).astype(int)]
    pre = calculate(w, b, query, tensor(core.sense(left, right, side)), core.CONTACT_TICKS)
    post = calculate(w, b, pre, tensor(core.sense(left, right, side, tastes)), core.TASTE_TICKS)
    support = np.broadcast_to(mask, (len(left), *mask.shape)).copy()
    graph = core.Graph(np.broadcast_to(w.detach().numpy(), support.shape),
                       np.broadcast_to(b.detach().numpy(), h.shape), support)
    before_w = w.expand(*support.shape)
    before_b = b.expand_as(h)
    readable_mask = tensor(support) if view == "own" else torch.zeros_like(before_w)
    readable_weight = before_w if view == "own" else torch.zeros_like(before_w)
    readable_bias = before_b if view == "own" else torch.zeros_like(before_b)
    local, gate, edge, bias = field_fn(pre, post, readable_weight, readable_bias,
                                     readable_mask, torch.tanh)
    if sparse:
        edge_slot, bias_slot = core.sparse_slots(edge.detach().numpy(), bias.detach().numpy())
        edge = edge * tensor(edge_slot)
        bias = bias * tensor(bias_slot)
    visible = core.view_arrays(graph, view, None)
    structural = core.structural_addresses(graph, post.detach().numpy(), local.detach().numpy(),
                                            gate.detach().numpy(), *visible, seed)
    edit = core.Proposal(edge.detach().numpy(), bias.detach().numpy(), *structural)
    actual, receipt = core.apply(graph, edit)
    edge = edge * (tensor(support) & (edge.abs() >= core.EPSILON))
    bias = bias * (bias.abs() >= core.EPSILON)
    next_w = (before_w + edge).clamp(-core.LIMIT, core.LIMIT)
    next_b = (before_b + bias).clamp(-core.LIMIT, core.LIMIT)
    new_rows = tensor(np.flatnonzero(receipt.birth_status == 1))
    if len(new_rows):
        receiver, source = np.divmod(edit.birth[new_rows.numpy()], core.N)
        next_w = next_w.clone()
        next_w[new_rows, receiver, source] = core.BIRTH_STEP * torch.tanh(
            2.0 * (local[new_rows, receiver, source] + gate[new_rows]))
    next_w = next_w * tensor(actual.mask)
    target = tensor(target)
    results = []
    for sign in (1, -1):
        odor_pair = (left, right) if sign == 1 else (right, left)
        output = calculate(next_w, next_b, h, tensor(core.sense(*odor_pair)), core.MOTOR_TICKS)
        results.append(sign * target * core.motor(output))
    margins = torch.cat(results)
    loss = nn.functional.softplus(-5.0 * margins).mean()
    return loss, (margins.detach() > 0).float().mean().item()


def evaluate(graph, data_seed, count=512, view="own", sparse=False, step_fn=core.step):
    rng = np.random.default_rng(data_seed)
    left, right, target, tastes = pair_examples(rng, count)
    graph = core.repeated_graph(graph, count)
    scores = {}
    for arm in ("active", "primitive", "none"):
        used, _, _, _, _, _ = step_fn(graph, core.rest(graph), left, right, tastes,
                                      data_seed + 10, arm=arm, view=view, sparse=sparse)
        correct = []
        for sign in (1, -1):
            odor_pair = (left, right) if sign == 1 else (right, left)
            out = core.advance(used, core.rest(used), core.sense(*odor_pair), core.MOTOR_TICKS)
            correct.append(sign * target * core.motor(out) > 0)
        scores[arm] = float(np.mean(correct))
    return scores


def train(root, iterations, batch=96, view="own", eval_seed=1901, sparse=False, *,
          field_fn=core.write_fields, step_fn=core.step):
    torch.set_num_threads(2)
    graph = core.spawn_graph(root)
    mask = graph.mask[0]
    rng = np.random.default_rng(root + 800000)
    w = nn.Parameter(tensor(graph.w[0]))
    b = nn.Parameter(tensor(graph.b[0]))
    optimizer = torch.optim.Adam([w, b], lr=0.007)
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        examples = pair_examples(rng, batch)
        loss, accuracy = training_trial(w, b, mask, examples, int(rng.integers(2**32)), view,
                                        sparse, field_fn=field_fn)
        total = loss + 0.0012 * (w.square().mean() + b.square().mean())
        total.backward()
        nn.utils.clip_grad_norm_([w, b], 5.0)
        optimizer.step()
        with torch.no_grad():
            w.clamp_(-core.LIMIT, core.LIMIT)
            b.clamp_(-core.LIMIT, core.LIMIT)
        if iteration % 100 == 0 or iteration == iterations:
            print(f"root={root} trial={iteration} loss={loss.item():.4f} batch={accuracy:.4f}",
                  flush=True)
    trained = core.Graph((w.detach().numpy() * mask)[None].copy(),
                          b.detach().numpy()[None].copy(), mask[None].copy())
    results = evaluate(trained, eval_seed, view=view, sparse=sparse, step_fn=step_fn)
    print(f"root={root} {json.dumps(results)}", flush=True)
    return trained, results


def load_checkpoint(path):
    with np.load(path) as data:
        return core.Graph(data["w"], data["b"], data["mask"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, nargs="+", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--view", choices=["blind", "own"], default="own")
    parser.add_argument("--eval-seed", type=int, default=1901)
    parser.add_argument("--prefix", default="bootstrap")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    for root in args.roots:
        graph, scores = train(root, args.iterations, args.batch, args.view, args.eval_seed, args.sparse)
        filename = EVIDENCE / f"{args.prefix}-{args.view}-{root}.npz"
        np.savez_compressed(filename, w=graph.w, b=graph.b, mask=graph.mask, root=root,
                            iterations=args.iterations, view=args.view,
                            bootstrap_scores=json.dumps(scores), eval_seed=args.eval_seed, sparse=args.sparse)


if __name__ == "__main__":
    main()
