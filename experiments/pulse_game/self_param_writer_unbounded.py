"""Probe actual actor-emitted commands with numeric and topology guards removed."""

import json

import numpy as np
import torch

from . import self_param_writer as full
from .self_param_writer_effects import load
from .self_param_writer_study import EVIDENCE
from .self_write_transfer import build_tape


def near_limit_cycle(initial, lives, seed):
    """Use a fresh disposable ring with finite parameters just below the normal cap."""
    graph = initial.repeat(lives)
    width = graph.width
    ids = torch.arange(width)
    support = torch.zeros_like(graph.present)
    support[:, (ids + 1) % width, ids] = True
    rng = np.random.default_rng(seed)
    weight_sign = torch.from_numpy(rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                                              size=graph.weights.shape))
    bias_sign = torch.from_numpy(rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                                            size=graph.bias.shape))
    bias_exposed = torch.from_numpy(rng.random(graph.bias.shape) < 0.20)
    return full.Graph(2.45 * weight_sign * support,
                      torch.where(bias_exposed, 2.45 * bias_sign, graph.bias), support)


@torch.inference_mode()
def run(initial, seed):
    lives, rounds = 128, 18
    tape = build_tape(seed, lives, 1, rounds, rounds, 0.0, complementary=True)
    rng = np.random.default_rng(seed + 5_000)
    draws = torch.from_numpy(rng.random((lives, rounds, 3), dtype=np.float32))
    graph = near_limit_cycle(initial, lives, seed + 20_000)
    original = graph
    distinct_cases = {
        "first_crossing": None,
        "first_guard_cut": None,
        "first_clipped_different": None,
    }
    for event in range(rounds):
        left, right = map(torch.from_numpy, tape.present(event))
        hidden, logit = full.motor(graph, left, right)
        side = torch.where(logit > 0, 1, -1)
        taste = torch.from_numpy(tape.taste(event, side.numpy()))
        emitted = full.propose(graph, hidden, left, right, side, taste, draws[:, event, :2],
                               keep_connected=False, value_limit=None)
        _, bounded = full.commit(graph, emitted.command)
        actual = emitted.receipt
        applied = actual.status == full.Status.APPLIED
        events = {
            "first_crossing": (applied & (actual.value_before.abs() <= full.LIMIT)
                               & (actual.value_after.abs() > full.LIMIT)),
            "first_guard_cut": applied & (bounded.status == full.Status.CONNECTIVITY),
            "first_clipped_different": (applied & (actual.value_after.abs() > full.LIMIT)
                                        & (actual.value_after != bounded.value_after)),
        }
        for label, occurrences in events.items():
            if distinct_cases[label] is None and occurrences.any():
                lane = occurrences.nonzero().flatten()[0].item()
                distinct_cases[label] = {
                    "lane": lane, "contact": event + 1,
                    "address": emitted.command.address[lane].item(),
                    "operation": full.Op(emitted.command.op[lane].item()).name.lower(),
                    "emitted_payload": emitted.command.payload[lane].item(),
                    "value_before": actual.value_before[lane].item(),
                    "unbounded_value_after": actual.value_after[lane].item(),
                    "bounded_value_after": bounded.value_after[lane].item(),
                    "bounded_status": full.Status(bounded.status[lane].item()).name.lower(),
                }
        graph = emitted.next_graph
    return {
        "lives": lives, "contacts_per_life": rounds,
        "initial_support_strongly_connected": all(full.strongly_connected(x)
                                                for x in original.present.numpy()),
        "last_out_of_bounds_weights": (graph.weights.abs() > full.LIMIT).sum().item(),
        "last_out_of_bounds_biases": (graph.bias.abs() > full.LIMIT).sum().item(),
        "last_non_strongly_connected_graphs": sum(not full.strongly_connected(x)
                                                 for x in graph.present.numpy()),
        "emitted_witnesses": distinct_cases,
    }


def main():
    torch.set_num_threads(1)
    reports = {}
    for width in full.WRITER_WIDTHS:
        graph = load(EVIDENCE / f"writer-{width}-root-421.npz")
        reports[str(width)] = run(graph, 290_371)
    path = EVIDENCE / "unbounded-actor-replay.json"
    path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
