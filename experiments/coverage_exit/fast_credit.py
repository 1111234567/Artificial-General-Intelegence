"""Opt-in, full-precision block RTRL Jacobians for the archived Heun graph.

Block J/K execution and its work bookkeeping change. The inherited engine
still computes the live gradient, cross-boundary correction, full S recurrence,
validation, and persistence. This is an algebraic execution optimization, not
an approximation: every state coupling and every potential edge is included.

The native jacrev method repeatedly differentiates expansion of the same W.
Here W is expanded once, full output-by-state adjoints are propagated through
each Heun substep, and their W outer products are summed across time by bmm.
Floating point reduction order changes; numerical equivalence is expected,
not bitwise identity. The archived trainer/model/credit sources are untouched.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from copy import deepcopy

import torch
from torch.nn import functional as F

from experiments.undirected.credit import TemporalCreditEngine
from experiments.undirected.model import UndirectedModel, PortGraphModel, _INV_SQRT_TWO


BACKEND = "analytic-full-heun-block-adjoint-fp64-v1"


def backend_manifest():
    source = Path(__file__).resolve()
    return {
        "backend": BACKEND,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "precision": "float64; no AMP, TF32, reduced dtype, sparsification or truncation",
        "equivalence": "same real-arithmetic derivatives; floating reduction order differs",
        "scope": "block J/K only; inherited live loss, full S, clipping and Adam",
    }


def _supported(engine):
    # Exact types protect against silently applying the formula to an override
    # of flow/pulse. Other models use the original engine explicitly.
    return (type(engine.model) in (UndirectedModel, PortGraphModel)
            and engine.dtype == torch.float64
            and engine.dynamic_names == tuple(engine.model.dynamic_parameter_names()))


@torch.no_grad()
def analytic_block_jacobians(engine, record):
    """Return exact dense J[B,H,H], K[B,H,P] at one recorded block.

    Observation pulses occur AFTER flow, including dt=0. Core pulses contribute
    encoder derivatives at that boundary. No derivative crosses a recorded
    optimizer/action/world decision. Input/output ports remain evolving nodes.
    """
    if not _supported(engine):
        raise ValueError("analytic block adjoint requires an unchanged FP64 graph model")
    model, device = engine.model, engine.device
    fixed = {name: value.to(device) for name, value in record["parameters"].items()}
    buffers = {name: value.to(device) for name, value in record["buffers"].items()}
    h = record["start_h"].to(device)
    observations = record["observations"].to(device, dtype=engine.dtype)
    batch, width = h.shape
    u, v = buffers["pair_index"].unbind(0)
    shared = fixed["edge_shared"]
    upper = lower = shared * _INV_SQRT_TWO
    if model.mode == "untied":
        upper = (shared + fixed["edge_skew"]) * _INV_SQRT_TWO
        lower = (shared - fixed["edge_skew"]) * _INV_SQRT_TWO
    indices = torch.cat((u * width + v, v * width + u))
    matrix = shared.new_zeros(width * width).scatter(
        0, indices, torch.cat((upper, lower))).reshape(width, width)
    core = type(model) is UndirectedModel
    # Keep only tanh activations; all state and adjoint axes are dense.
    stages, pulse_inputs = [], []
    for observation, elapsed in zip(observations, record["elapsed"]):
        count = max(1, math.ceil(elapsed / model.max_step))
        delta = elapsed / count
        tick_stages = []
        for _ in range(count):
            first_tanh = h.tanh()
            first = -h + F.linear(first_tanh, matrix)
            predictor = h + delta * first
            second_tanh = predictor.tanh()
            second = -predictor + F.linear(second_tanh, matrix)
            h = h + .5 * delta * (first + second)
            tick_stages.append((first_tanh, second_tanh, delta))
        normalized = observation * buffers["input_scale"].to(dtype=engine.dtype)
        if core:
            h = h + F.linear(normalized, fixed["encoder.weight"])
            pulse_inputs.append(normalized)
        else:
            values = normalized * model.pulse_gain
            pulse = values.new_zeros(batch, width).scatter(
                -1, buffers["input_ports"].expand(batch, model.input_size), values)
            h = h + pulse
        stages.append(tick_stages)
    engine._same_hidden(h, record["end_h"], "analytic block end_h")

    adjoint = torch.eye(width, device=device, dtype=engine.dtype).expand(batch, width, width)
    coefficients, activations, pulse_adjoints = [], [], []
    for tick_stages in reversed(stages):
        if core:
            # This is d(final state)/d(state immediately after this pulse).
            pulse_adjoints.append(adjoint)
        for first_tanh, second_tanh, delta in reversed(tick_stages):
            second_bar = .5 * delta * adjoint
            predictor_bar = ((second_bar @ matrix)
                             * (1. - second_tanh.square()).unsqueeze(1) - second_bar)
            first_bar = .5 * delta * adjoint + delta * predictor_bar
            adjoint = (adjoint + predictor_bar
                       + (first_bar @ matrix) * (1. - first_tanh.square()).unsqueeze(1)
                       - first_bar)
            coefficients.extend((second_bar, first_bar))
            activations.extend((second_tanh, first_tanh))

    def accumulated_outer(left, right):
        # [B,O*I,L] @ [B,L,D] sums all time uses of each matrix coordinate.
        left = torch.stack(left, dim=-1).reshape(batch, width * width, -1)
        right = torch.stack(right, dim=1)
        return torch.bmm(left, right).reshape(batch, width, width, right.shape[-1])

    matrix_derivative = accumulated_outer(coefficients, activations)
    upper_gradient = matrix_derivative[..., u, v]
    lower_gradient = matrix_derivative[..., v, u]
    pieces = [(upper_gradient + lower_gradient) * _INV_SQRT_TWO]
    if model.mode == "untied":
        pieces.append((upper_gradient - lower_gradient) * _INV_SQRT_TWO)
    if core:
        encoder_derivative = accumulated_outer(pulse_adjoints, list(reversed(pulse_inputs)))
        pieces.append(encoder_derivative.flatten(-2))
    return adjoint.contiguous(), torch.cat(pieces, dim=-1)


class FastTemporalCreditEngine(TemporalCreditEngine):
    """Native engine with an explicitly scoped analytic J/K execution path.

    Unknown models/dtypes fall back to native jacrev, preserving compatibility.
    The state format stays native-compatible; execution provenance is separate.
    """
    def _block_jacobians(self, record):
        if not _supported(self):
            return super()._block_jacobians(record)
        return analytic_block_jacobians(self, record)

    def gradients(self, chunk, parameters=None):
        gradients, metrics = super().gradients(chunk, parameters)
        if self.config.method.startswith("rtrl") and _supported(self):
            metrics["execution_backend"] = BACKEND
            # The parent counts jacrev chunks by ceil(H/chunk_size). No jacrev
            # chunk is executed here; keep the ordinary model/S counts intact.
            metrics["work"]["jacobian_reverse_chunks"] = 0
            metrics["work"]["analytic_block_reverse_substeps"] = sum(
                max(1, math.ceil(dt / self.model.max_step)) for dt in self.tape[-1]["elapsed"])
            metrics["work"]["note"] += (
                " Block Jacobians use a dense analytic Heun adjoint; reverse_rows "
                "counts output coordinates, reverse_chunks is zero, and analytic "
                "substeps count actual reverse solver steps. No jacrev is run.")
            self.last_metrics = deepcopy(metrics)
        return gradients, metrics


def install_fast_credit(trainer):
    """Opt in on a new research trainer or a resumed native trainer.

    Call after construction/loading, before train_update. The complete credit
    state is loaded into the replacement and validated; no sensitivity reset.
    Callers writing research results must save backend_manifest with results.
    """
    previous = trainer.credit
    if type(previous) is FastTemporalCreditEngine:
        return trainer
    if type(previous) is not TemporalCreditEngine:
        raise ValueError("refusing to replace an unrecognized credit engine")
    replacement = FastTemporalCreditEngine(trainer.model, previous.config)
    if not _supported(replacement):
        raise ValueError("fast trainer opt-in requires an unchanged FP64 graph model")
    replacement.load_state_dict(previous.state_dict())
    trainer.credit = replacement
    trainer.fast_credit_manifest = backend_manifest()
    return trainer
