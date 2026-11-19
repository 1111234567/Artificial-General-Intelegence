"""Pair a real writer-incident edit with an unchanged graph and replay its next proposal."""

from collections import Counter
import json

import numpy as np
import torch

from . import self_param_writer as full
from .self_param_writer_study import EVIDENCE
from .self_write_transfer import build_tape


def load(path):
    with np.load(path) as stored:
        return full.Graph(*(torch.from_numpy(stored[name])
                            for name in ("weights", "bias", "present")))


@torch.inference_mode()
def causal_trace(initial, condition, batch):
    tape = build_tape(
        condition["seed"], batch, condition["odors"], condition["rounds"],
        condition["phase_length"], condition["noise"], complementary=True)
    rng = np.random.default_rng(condition["seed"] + 10_000)
    draws = torch.from_numpy(rng.random((tape.lives, tape.contacts, 3), dtype=np.float32))
    stimuli = [tuple(map(torch.from_numpy, tape.present(event))) for event in range(tape.contacts)]
    graph = initial.repeat(tape.lives)
    counts = Counter()
    for event, inputs in enumerate(stimuli):
        hidden, logit = full.motor(graph, *inputs)
        side = torch.where(logit > 0, 1, -1)
        taste = torch.from_numpy(tape.taste(event, side.numpy()))
        edit = full.propose(graph, hidden, *inputs, side, taste, draws[:, event, :2])
        after = edit.next_graph
        edge, receiver, sender = full.decode(edit.command.address, graph.width)
        relevant = (edit.receipt.status == full.Status.APPLIED) & (
            (receiver >= full.BODY) | (edge & (sender >= full.BODY)))
        if event + 1 < tape.contacts and relevant.any():
            step = event + 1
            next_inputs = stimuli[step]
            common_hidden, _ = full.motor(graph, *next_inputs)
            common_side = torch.from_numpy(tape.scripted_arms[:, step])
            common_taste = torch.from_numpy(tape.taste(step, common_side.numpy()))
            old = full.propose(graph, common_hidden, *next_inputs, common_side, common_taste,
                               draws[:, step, :2]).command
            new = full.propose(after, common_hidden, *next_inputs, common_side, common_taste,
                               draws[:, step, :2]).command
            same_address = old.address == new.address
            same_op = old.op == new.op
            for lane in relevant.nonzero().flatten().tolist():
                dst = "W" if receiver[lane] >= full.BODY else "B"
                src = "W" if sender[lane] >= full.BODY else "B"
                groups = [f"{src}>{dst}" if edge[lane] else "bias_W"]
                if edge[lane] and receiver[lane] == sender[lane]:
                    groups.append("loop_W")
                op = full.Op(edit.command.op[lane].item()).name.lower()
                for group in groups:
                    key = f"{group}/{op}"
                    counts[f"{key}/paired"] += 1
                    counts[f"{key}/changed_address"] += int(not same_address[lane])
                    counts[f"{key}/changed_op_at_same_address"] += int(
                        same_address[lane] and not same_op[lane])
                    counts[f"{key}/changed_decline_decision"] += int(
                        (old.op[lane] == full.Op.NONE) != (new.op[lane] == full.Op.NONE))
                    if same_address[lane] and same_op[lane] and old.op[lane] in (
                            full.Op.ADJUST, full.Op.BIRTH):
                        counts[f"{key}/same_numeric_command"] += 1
                        change = abs(old.payload[lane].item() - new.payload[lane].item())
                        counts[f"{key}/changed_payload_by_more_than_1e-5"] += int(change > 1e-5)
        graph = after
    return dict(counts)


def main():
    torch.set_num_threads(1)
    plan = json.loads((EVIDENCE / "protocol.json").read_text(encoding="utf-8"))
    result = {}
    for width in full.WRITER_WIDTHS:
        out = {}
        for seed in plan["roots"]:
            graph = load(EVIDENCE / f"writer-{width}-root-{seed}.npz")
            out[str(seed)] = {condition["name"]: causal_trace(graph, condition, plan["holdout_lives"])
                              for condition in plan["holdout"]}
        result[str(width)] = out
    path = EVIDENCE / "real-edit-causal-replay.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
