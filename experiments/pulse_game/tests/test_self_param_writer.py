import numpy as np
import pytest
import torch

from experiments.pulse_game import self_param_writer as full
from experiments.pulse_game import self_param_writer_audit as audit


@pytest.mark.parametrize("writers", full.WRITER_WIDTHS)
def test_every_parameter_and_operation_including_self_loops(writers):
    report = audit.audit_width(writers)
    operations = report["operations"]
    width = full.BODY + writers
    assert operations["birth"]["tested"] == operations["retire"]["tested"] == width ** 2
    assert operations["adjust"]["tested"] == operations["none"]["tested"] == width ** 2 + width
    assert operations["adjust"]["writer_biases"] == writers
    assert operations["retire"]["edges_writer_to_writer"] == writers ** 2


def test_blind_content_is_zero_and_does_not_mask_actual_support():
    graph = full.spawn(1, 4)
    width = graph.width
    absent, present = (torch.where(graph.present.flatten() == value)[0][0].item()
                       for value in (False, True))
    body = graph.repeat(3)
    address = torch.tensor([present, absent, width * width + full.BODY])
    exists, value, bias = full.read_address(body, address)
    assert exists.tolist() == [True, False, True]
    assert value[1] == 0
    assert value[2] == bias[2]
    assert all(not port.any() for port in full.read_address(body, address, "blind"))
    left = torch.tensor([[0.6, -0.2, 0.4]]).expand(3, -1)
    right = -left
    post, _ = full.motor(body, left, right)
    assert full.address_logits(body, post, post).isfinite().all()
    own, _ = full.operation_logits(body, post, address, view="own")
    blind, _ = full.operation_logits(body, post, address, view="blind")
    assert own[:2].isfinite().all() and blind[:2].isfinite().all()
    assert not torch.allclose(own[:2], blind[:2])
    assert not own[2, [full.Op.BIRTH, full.Op.RETIRE]].isfinite().any()
    op = torch.full_like(address, full.Op.BIRTH)
    _, receipt = full.commit(body, full.Command(address, op, torch.zeros_like(value)))
    assert receipt.status.tolist() == [full.Status.SUPPORT_MISMATCH, full.Status.APPLIED,
                                       full.Status.SUPPORT_MISMATCH]
    assert receipt.present_after[1] and receipt.value_after[1] == 0


def test_readback_never_trains_a_coefficient_for_an_absent_edge():
    original = full.spawn(4, 4)
    raw = torch.where(original.present, original.weights, 1.0).requires_grad_()
    body = full.Graph(raw, original.bias, original.present)
    present = torch.where(body.present.flatten())[0][0]
    absent = torch.where(~body.present.flatten())[0][0]
    addresses = torch.stack((present, absent))
    _, value, _ = full.read_address(body.repeat(2), addresses)
    value.sum().backward()
    assert value[1] == 0
    assert raw.grad.flatten()[present] == 1
    assert raw.grad.flatten()[absent] == 0
    command = full.Command(absent.reshape(1), torch.tensor([full.Op.NONE]), torch.tensor([0.1]))
    after, receipt = full.commit(body, command)
    assert receipt.status.item() == full.Status.DECLINED
    assert torch.equal(after.weights, body.weights)


@pytest.mark.parametrize("operation", [full.Op.ADJUST, full.Op.BIRTH])
def test_numeric_payload_reaches_actual_later_motor_logit_by_bptt(operation):
    graph = full.spawn(4, 4)
    address = torch.tensor([11 * graph.width])
    before = audit.matched_states(graph, address, operation)
    left = torch.tensor([[0.8, -0.2, 0.4]])
    value = torch.tensor([0.125], requires_grad=True)
    op = torch.tensor([operation])

    def measured(payload):
        next_graph, receipt = full.commit(before, full.Command(address, op, payload))
        assert receipt.status.item() == full.Status.APPLIED
        return full.motor(next_graph, left, -left)[1]

    logit = measured(value)
    grad = torch.autograd.grad(logit, value)[0]
    epsilon = 0.0001
    difference = (measured(value.detach() + epsilon) - measured(value.detach() - epsilon)) / (
        2 * epsilon)
    assert grad.abs().item() > 0.001
    np.testing.assert_allclose(grad.numpy(), difference.numpy(), rtol=0.025, atol=0.001)


def test_writer_protection_covers_incoming_outgoing_and_bias_ports():
    graph = full.spawn(5, 4)
    width = graph.width
    addresses = torch.tensor([0, full.BODY * width, full.BODY, width * width + full.BODY])
    op = torch.full_like(addresses, full.Op.ADJUST)
    before = audit.matched_states(graph, addresses, full.Op.ADJUST)
    command = full.Command(addresses, op, torch.full_like(addresses, 0.1, dtype=torch.float32))
    next_graph, receipt = full.commit(before, command, intervention="protect_writer")
    assert (receipt.status == torch.tensor([full.Status.APPLIED] + [full.Status.WRITER_PROTECTED]
                                          * 3)).all()
    torch.testing.assert_close(next_graph.weights[1:], before.weights[1:], atol=0, rtol=0)
    torch.testing.assert_close(next_graph.bias, before.bias, atol=0, rtol=0)


@pytest.mark.parametrize("writers", full.WRITER_WIDTHS)
@pytest.mark.parametrize("operation", [full.Op.ADJUST, full.Op.BIRTH])
def test_unbounded_updates_and_births_at_every_applicable_address(writers, operation):
    graph = full.spawn(4, writers)
    width = graph.width
    address = torch.arange(width * width + (width if operation == full.Op.ADJUST else 0))
    op = torch.full_like(address, operation)
    before = audit.matched_states(graph, address, operation)
    payload = torch.where(address % 2 == 0, 10.0, -10.0)
    command = full.Command(address, op, payload)
    after, receipt = full.commit(before, command, value_limit=None, keep_connected=False)
    audit.assert_commit(before, after, receipt, address, operation, payload)
    assert (receipt.value_after.abs() > full.LIMIT).all()
    _, bounded = full.commit(before, command)
    assert (bounded.value_after.abs() == full.LIMIT).all()


def test_unbounded_mode_refuses_fp_overflow_without_affecting_graph():
    graph = full.spawn(1, 4)
    limit = torch.finfo(graph.bias.dtype).max
    before = full.Graph(graph.weights, torch.full_like(graph.bias, limit), graph.present)
    command = full.Command(torch.tensor([graph.width ** 2]), torch.tensor([full.Op.ADJUST]),
                           torch.tensor([limit]))
    after, receipt = full.commit(before, command, value_limit=None, keep_connected=False)
    assert receipt.status.item() == full.Status.NUMERIC_RANGE
    assert torch.equal(after.bias, before.bias)
