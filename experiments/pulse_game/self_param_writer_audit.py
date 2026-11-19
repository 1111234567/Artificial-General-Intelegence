"""Exhaustive address and interpreter audit for the full pulse writer."""

import json

import torch

from . import self_param_writer as full
from .self_param_writer_study import EVIDENCE


def matched_states(graph, address, operation):
    """Prepare a strongly connected graph where the requested operation is valid."""
    width = graph.width
    base = graph.repeat(len(address))
    body = full.Graph(torch.full_like(base.weights, 0.25), base.bias,
                      torch.ones_like(base.present))
    edge, receiver, sender = full.decode(address, width)
    rows = torch.arange(len(address))
    if operation == full.Op.BIRTH:
        body.present[rows[edge], receiver[edge], sender[edge]] = False
        body.weights[rows[edge], receiver[edge], sender[edge]] = 0
    return body


def assert_commit(before, after, receipt, address, operation, payload):
    width = before.width
    edge, receiver, sender = full.decode(address, width)
    rows = torch.arange(len(address))
    slot = rows[edge], receiver[edge], sender[edge]
    bias_slot = rows[~edge], receiver[~edge]
    expected_weights = before.weights.clone()
    expected_bias = before.bias.clone()
    expected_support = before.present.clone()
    old_presence, old_value, old_bias = full.read_address(before, address)
    if operation == full.Op.ADJUST:
        value = old_value + payload
    elif operation == full.Op.BIRTH:
        value = payload
        expected_support[slot] = True
    elif operation == full.Op.RETIRE:
        value = torch.zeros_like(payload)
        expected_support[slot] = False
    else:
        value = old_value
    expected_weights[slot] = value[edge]
    expected_bias[bias_slot] = value[~edge]
    torch.testing.assert_close(after.weights, expected_weights, atol=0, rtol=0)
    torch.testing.assert_close(after.bias, expected_bias, atol=0, rtol=0)
    torch.testing.assert_close(after.present, expected_support, atol=0, rtol=0)
    torch.testing.assert_close(receipt.value_before, old_value, atol=0, rtol=0)
    torch.testing.assert_close(receipt.present_before, old_presence, atol=0, rtol=0)
    torch.testing.assert_close(old_bias, before.bias[rows, receiver], atol=0, rtol=0)
    presence, new, _ = full.read_address(after, address)
    torch.testing.assert_close(new, value, atol=0, rtol=0)
    torch.testing.assert_close(receipt.value_after, value, atol=0, rtol=0)
    torch.testing.assert_close(receipt.present_after, presence, atol=0, rtol=0)
    status = full.Status.DECLINED if operation == full.Op.NONE else full.Status.APPLIED
    assert (receipt.status == status).all()
    assert all(full.strongly_connected(present) for present in after.present.numpy())


def bridge_witness(graph):
    """Request the same critical edge retirement with and without the SCC guard."""
    width = graph.width
    sender = full.BODY
    receiver = sender + 1
    ids = torch.arange(width)
    support = torch.zeros_like(graph.present)
    support[:, (ids + 1) % width, ids] = True
    before = full.Graph(support.to(torch.float32) * 0.25, graph.bias, support)
    address = torch.tensor([receiver * width + sender])
    op = torch.tensor([full.Op.RETIRE])
    command = full.Command(address, op, torch.tensor([0.0]))
    guarded, guard_status = full.commit(before, command)
    cut, cut_status = full.commit(before, command, keep_connected=False)
    birth = full.Command(address, torch.tensor([full.Op.BIRTH]), torch.tensor([0.25]))
    restored, birth_status = full.commit(cut, birth)
    assert guard_status.status.item() == full.Status.CONNECTIVITY
    assert cut_status.status.item() == full.Status.APPLIED
    assert birth_status.status.item() == full.Status.APPLIED
    assert torch.equal(guarded.weights, before.weights)
    assert torch.equal(guarded.present, before.present)
    assert not full.strongly_connected(cut.present[0].numpy())
    assert cut.weights[0, receiver, sender] == 0
    assert torch.equal(restored.weights, before.weights)
    assert torch.equal(restored.present, before.present)
    return {"sender": sender, "receiver": receiver,
            "guarded_status": full.Status(guard_status.status.item()).name.lower(),
            "unguarded_status": full.Status(cut_status.status.item()).name.lower(),
            "unguarded_strongly_connected": False, "rebuilt_exact": True}


def audit_width(writers):
    graph = full.spawn(8, writers)
    width = graph.width
    operations = {}
    for operation in full.Op:
        slots = width * width + (width if operation in (full.Op.NONE, full.Op.ADJUST) else 0)
        address = torch.arange(slots)
        op = torch.full_like(address, operation)
        before = matched_states(graph, address, operation)
        payload = torch.where(address % 2 == 0, 0.125, -0.125)
        after, receipt = full.commit(before, full.Command(address, op, payload))
        assert_commit(before, after, receipt, address, operation, payload)
        edge, receiver, sender = full.decode(address, width)
        operations[operation.name.lower()] = {
            "tested": slots,
            "edges_body_to_body": (edge & (sender < full.BODY) & (receiver < full.BODY)).sum().item(),
            "edges_body_to_writer": (edge & (sender < full.BODY) & (receiver >= full.BODY)).sum().item(),
            "edges_writer_to_body": (edge & (sender >= full.BODY) & (receiver < full.BODY)).sum().item(),
            "edges_writer_to_writer": (edge & (sender >= full.BODY) & (receiver >= full.BODY)).sum().item(),
            "body_self_loops": (edge & (sender == receiver) & (sender < full.BODY)).sum().item(),
            "writer_self_loops": (edge & (sender == receiver) & (sender >= full.BODY)).sum().item(),
            "body_biases": (~edge & (receiver < full.BODY)).sum().item(),
            "writer_biases": (~edge & (receiver >= full.BODY)).sum().item(),
            "all_exact_one_address": True,
        }
    return {"writer_width": writers, "operations": operations, "bridge": bridge_witness(graph)}


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    report = [audit_width(width) for width in full.WRITER_WIDTHS]
    path = EVIDENCE / "exhaustive-permission-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
