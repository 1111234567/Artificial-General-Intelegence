"""Endogenous destination-value writes for the same pulse graph and body."""
from functools import partial
from pathlib import Path

import numpy as np

from . import self_write_core as core


EDGE_TARGET = 1.35
BIAS_TARGET = 0.50
EDGE_BLEND = 0.20
BIAS_BLEND = 0.08
EVIDENCE = Path("docs/evidence/self-writing-2026-11-19/target")
ROOTS = tuple(range(301, 309))
PAIR_SEED = 92110420


def write_fields(pre, post, visible_w, visible_b, visible_mask, tanh=np.tanh):
    """Decode local destination and strength solely from circuit activity."""
    source = tanh(2.0 * pre)
    receiver = tanh(2.0 * (post - 0.35 * pre))
    local = receiver[:, :, None] * source[:, None, :]
    edge_gate = tanh(2.5 * post[:, 12])
    bias_gate = tanh(2.5 * post[:, 13])
    edge_command = edge_gate[:, None, None] * local
    bias_command = bias_gate[:, None] * receiver
    edge_delta = EDGE_BLEND * abs(edge_command) * (
        EDGE_TARGET * edge_command - visible_w * visible_mask)
    bias_delta = BIAS_BLEND * abs(bias_command) * (BIAS_TARGET * bias_command - visible_b)
    return local, edge_gate, edge_delta, bias_delta


def proposal(graph, pre, post, key, view="blind", view_graph=None, sparse=False):
    visible = core.view_arrays(graph, view, view_graph)
    local, edge_gate, edge_delta, bias_delta = write_fields(pre, post, *visible)
    structural = core.structural_addresses(graph, post, local, edge_gate, *visible, key)
    if sparse:
        edge_slot, bias_slot = core.sparse_slots(edge_delta, bias_delta)
        edge_delta = np.where(edge_slot, edge_delta, 0.0)
        bias_delta = np.where(bias_slot, bias_delta, 0.0)
    return core.Proposal(edge_delta, bias_delta, *structural)


step = partial(core.step, proposal_fn=proposal)
