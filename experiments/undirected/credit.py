"""Temporal derivatives on the controller's *recorded* parameter trajectory.

The runner owns live state, observations, actions, optimizer and structure.  This
module only reads a completed live block and returns a gradient.  Historical
parameters are constants plus a shared current-parameter perturbation; thus a
reconstruction has the actual primal values even after intervening updates.
The derivative stops through the optimizer, structure decisions, actions and
world.  It is not the derivative of a counterfactual constant-parameter history.

Block RTRL carries the full state-by-dynamic-parameter sensitivity.  A 128-tick
block is a computational batching choice, not a temporal truncation for RTRL.
The reset control computes exactly the same Jacobians and matrix product, but
uses zero previous sensitivity.  No diagonal or neuron-local approximation is
used.  Dormant zero-valued edges remain differentiable potential connections.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral
from typing import Any

import torch
from torch.func import functional_call, jacrev, vmap
from torch.nn import functional as F


FORMAT_VERSION = "undirected-recorded-temporal-credit-v2"
CREDIT_METHODS = ("tbptt128", "recorded512", "recorded1024", "rtrl", "rtrl_reset")
DERIVATIVE_CONVENTION = (
    "Common additive perturbation of every recorded parameter use; realized "
    "optimizer/structure changes, observations, targets, actions and world are constants."
)


def _integer(value, name, lower=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return int(value)


@dataclass(frozen=True)
class CreditConfig:
    method: str = "tbptt128"
    update_ticks: int = 128
    wait_weight: float = .1
    jacrev_chunk_size: int = 8
    primal_atol: float = 1e-6
    primal_rtol: float = 1e-5

    def __post_init__(self):
        if self.method not in CREDIT_METHODS:
            raise ValueError("unknown temporal credit method")
        _integer(self.update_ticks, "update_ticks", 1)
        _integer(self.jacrev_chunk_size, "jacrev_chunk_size", 1)
        for name in ("wait_weight", "primal_atol", "primal_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if self.wait_weight <= 0:
            raise ValueError("positive WAIT weight required")

    @property
    def history_blocks(self):
        # Names describe the formal update_ticks=128 protocol.  Smaller blocks
        # are useful for numerical tests and preserve the 1/4/8-block contrast.
        return {"tbptt128": 1, "recorded512": 4, "recorded1024": 8,
                "rtrl": 1, "rtrl_reset": 1}[self.method]

    @property
    def credit_ticks(self):
        return None if self.method == "rtrl" else self.history_blocks * self.update_ticks


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return deepcopy(value)


def state_digest(value):
    """Content hash including tensor dtype/shape and container structure."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor/{tensor.dtype}/{tuple(tensor.shape)}/".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping[")
            for key in sorted(item):
                add(key)
                add(item[key])
            digest.update(b"]")
        elif isinstance(item, (list, tuple)):
            digest.update(b"tuple[" if isinstance(item, tuple) else b"list[")
            for child in item:
                add(child)
            digest.update(b"]")
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    add(value)
    return digest.hexdigest()


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def snapshot_parameters(model):
    return {name: _cpu(parameter) for name, parameter in model.named_parameters()}


def weighted_loss(logits, targets, wait_weight=.1):
    """One weighted CE denominator over the complete block, never per tick."""
    classes = logits.shape[-1]
    flat_targets = targets.reshape(-1)
    weights = logits.new_ones(classes)
    weights[0] = wait_weight
    numerator = F.cross_entropy(logits.reshape(-1, classes), flat_targets,
                                weight=weights, reduction="sum")
    denominator = weights[flat_targets].sum()
    return numerator / denominator, numerator, denominator


def stable_l2(value):
    """Float64 L2 without overflowing/underflowing the intermediate squares.

    Scaling is confined to a temporary diagnostic expression; the learner's
    state, sensitivity and gradient tensors are never changed.
    """
    if not value.numel():
        return 0.
    values = value.detach().double()
    scale = float(values.abs().max())
    if not math.isfinite(scale):
        raise FloatingPointError("nonfinite diagnostic norm input")
    if scale == 0.:
        return 0.
    result = scale * float(torch.linalg.vector_norm(values / scale))
    if not math.isfinite(result):
        raise FloatingPointError("true diagnostic L2 norm exceeds float64 range")
    return result


def _combined_l2(values):
    result = math.hypot(*values)
    if not math.isfinite(result):
        raise FloatingPointError("true combined gradient L2 norm exceeds float64 range")
    return result


def diagnostic_inner_product(first, second):
    """Return (value, overflow); an unavailable diagnostic never writes Inf."""
    value = float((first.detach().double().reshape(-1) * second.detach().double().reshape(-1)).sum())
    return (value, False) if math.isfinite(value) else (None, True)


class TemporalCreditEngine:
    """Stateful gradient provider; call once, before each runner optimizer step.

    ``gradients(chunk, parameters=None)`` accepts observations[T,B,D],
    targets[T,B,J], elapsed[T], start_h[B,H], end_h[B,H], and optional aligned
    start_tick/end_tick.  ``parameters`` must be this block's pre-update snapshot
    and equal the model's current values.  The returned tensors are detached and
    do not populate Parameter.grad.  Model buffers are recorded as well.  The
    model declares all parameters influencing persistent state through
    ``dynamic_parameter_names()``; remaining parameters receive local gradients
    only, and that remaining set may be empty for a graph with fixed ports.
    """
    def __init__(self, model, config: CreditConfig | None = None):
        self.model = model
        self.config = CreditConfig() if config is None else config
        self.parameters = dict(model.named_parameters())
        if not self.parameters or any(not parameter.requires_grad for parameter in self.parameters.values()):
            raise ValueError("temporal credit requires trainable model parameters")
        declare_dynamic = getattr(model, "dynamic_parameter_names", None)
        if not callable(declare_dynamic):
            raise ValueError("controller must declare dynamic_parameter_names()")
        declared = declare_dynamic()
        if (not isinstance(declared, (tuple, list)) or not declared
                or any(not isinstance(name, str) for name in declared)
                or len(set(declared)) != len(declared) or not set(declared).issubset(self.parameters)):
            raise ValueError("invalid declared dynamic parameter layout")
        self.dynamic_names = tuple(declared)
        self.local_names = tuple(name for name in self.parameters if name not in self.dynamic_names)
        first = next(iter(self.parameters.values()))
        self.device, self.dtype = first.device, first.dtype
        if any(value.device != self.device or value.dtype != self.dtype for value in self.parameters.values()):
            raise ValueError("controller parameters must share device/dtype")
        self.parameter_layout = tuple((name, tuple(value.shape), str(value.dtype))
                                      for name, value in self.parameters.items())
        self.dynamic_sizes = tuple(self.parameters[name].numel() for name in self.dynamic_names)
        self.dynamic_parameter_count = sum(self.dynamic_sizes)
        self.buffer_layout = tuple((name, tuple(value.shape), str(value.dtype)) for name, value in model.named_buffers())
        self.model_metadata = {"class": f"{type(model).__module__}.{type(model).__qualname__}",
                               "config": deepcopy(model.config()) if callable(getattr(model, "config", None)) else None}
        self.tape: list[dict[str, Any]] = []
        self.sensitivity: torch.Tensor | None = None
        self.batch_size: int | None = None
        self.width: int | None = None
        self.update_age = 0
        self.trajectory_hash = state_digest({"format": FORMAT_VERSION, "config": asdict(self.config),
                                             "layout": self.parameter_layout, "dynamic_names": self.dynamic_names,
                                             "local_names": self.local_names, "model": self.model_metadata})
        self.last_metrics: dict[str, Any] | None = None

    @property
    def updates(self):
        return self.update_age

    def _validate_parameters(self, values, *, current=False):
        if not isinstance(values, Mapping) or set(values) != set(self.parameters):
            raise ValueError("incomplete recorded parameters")
        for name, parameter in self.parameters.items():
            value = values[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != self.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"invalid recorded parameter {name}")
            if current and not torch.equal(value.detach().cpu(), parameter.detach().cpu()):
                raise ValueError(f"snapshot is not the current pre-update parameter: {name}")

    def _validate_buffers(self, values):
        if not isinstance(values, Mapping) or set(values) != {name for name, _, _ in self.buffer_layout}:
            raise ValueError("incomplete recorded buffers")
        for name, shape, dtype in self.buffer_layout:
            value = values[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(f"invalid recorded buffer {name}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite recorded buffer {name}")

    def _record(self, chunk, parameters):
        if not isinstance(chunk, Mapping):
            raise ValueError("live chunk must be a mapping")
        required = {"observations", "targets", "elapsed", "start_h", "end_h"}
        if not required.issubset(chunk):
            raise ValueError("incomplete live chunk")
        start = _integer(chunk.get("start_tick", self.update_age * self.config.update_ticks), "start_tick")
        end = _integer(chunk.get("end_tick", start + self.config.update_ticks), "end_tick")
        if start != self.update_age * self.config.update_ticks or end != start + self.config.update_ticks:
            raise ValueError("live chunk is not the next aligned optimizer block")
        values = snapshot_parameters(self.model) if parameters is None else parameters
        self._validate_parameters(values, current=True)
        record = {key: _cpu(chunk[key]) for key in required}
        record.update(start_tick=start, end_tick=end, parameters=_cpu(values),
                      buffers={name: _cpu(value) for name, value in self.model.named_buffers()})
        self._validate_record(record)
        if self.batch_size is not None and record["start_h"].shape != (self.batch_size, self.width):
            raise ValueError("live batch/state dimensions changed")
        if self.tape and not torch.equal(record["start_h"], self.tape[-1]["end_h"]):
            raise ValueError("live hidden state was replaced between optimizer blocks")
        return record

    def _validate_record(self, record):
        if not isinstance(record, Mapping) or set(record) != {"observations", "targets", "elapsed", "start_h", "end_h",
                                                               "start_tick", "end_tick", "parameters", "buffers"}:
            raise ValueError("incomplete recorded credit block")
        start = _integer(record["start_tick"], "record start_tick")
        end = _integer(record["end_tick"], "record end_tick")
        if start % self.config.update_ticks or end - start != self.config.update_ticks:
            raise ValueError("unaligned recorded credit block")
        obs, targets, h0, h1 = (record[name] for name in ("observations", "targets", "start_h", "end_h"))
        if any(not isinstance(value, torch.Tensor) for value in (obs, targets, h0, h1)):
            raise ValueError("recorded block values must be tensors")
        if (obs.ndim != 3 or h0.ndim != 2 or h1.shape != h0.shape or obs.shape[:2] != (self.config.update_ticks, len(h0))
                or obs.shape[-1] != self.model.input_size
                or h0.shape[-1] != self.model.width
                or targets.shape != (self.config.update_ticks, len(h0), self.model.n_jobs)
                or targets.dtype != torch.long or bool((targets < 0).any()) or bool((targets >= self.model.n_actions).any())):
            raise ValueError("invalid recorded block dimensions/targets")
        if not obs.is_floating_point() or not bool(torch.isfinite(obs).all()):
            raise ValueError("nonfinite or nonfloating recorded observation")
        for value in (h0, h1):
            if value.dtype != self.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite or wrong-dtype recorded activity")
        elapsed = record["elapsed"]
        if not isinstance(elapsed, (tuple, list)) or len(elapsed) != self.config.update_ticks:
            raise ValueError("invalid recorded elapsed times")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in elapsed):
            raise ValueError("invalid recorded elapsed time")
        self._validate_parameters(record["parameters"])
        self._validate_buffers(record["buffers"])

    def _same_hidden(self, actual, expected, label):
        expected = expected.to(device=actual.device, dtype=actual.dtype)
        error = float((actual.detach() - expected).abs().max())
        if not torch.allclose(actual.detach(), expected, atol=self.config.primal_atol, rtol=self.config.primal_rtol):
            raise ValueError(f"recorded trajectory does not reproduce {label}; max abs error {error}")
        return error

    def _shifted_parameters(self, record):
        return {name: record["parameters"][name].to(self.device) + (parameter - parameter.detach())
                for name, parameter in self.parameters.items()}

    def _replay(self, records, *, start_h=None):
        h = records[0]["start_h"].to(self.device) if start_h is None else start_h
        outputs = []
        max_error = 0.
        for block_index, record in enumerate(records):
            max_error = max(max_error, self._same_hidden(h, record["start_h"], "block start_h"))
            parameters = self._shifted_parameters(record)
            buffers = {name: value.to(self.device) for name, value in record["buffers"].items()}
            observations = record["observations"].to(self.device)
            for observation, dt in zip(observations, record["elapsed"]):
                output, h = functional_call(self.model, (parameters, buffers), (observation, h, dt), strict=True)
                if block_index == len(records) - 1:
                    outputs.append(output)
            max_error = max(max_error, self._same_hidden(h, record["end_h"], "block end_h"))
        return torch.stack(outputs), h, max_error

    def _block_jacobians(self, record):
        fixed = {name: value.to(self.device) for name, value in record["parameters"].items()}
        buffers = {name: value.to(self.device) for name, value in record["buffers"].items()}
        flat = torch.cat([fixed[name].reshape(-1) for name in self.dynamic_names])
        observations = record["observations"].to(self.device)

        def transition(h, dynamic, inputs):
            parameters = dict(fixed)
            for name, value in zip(self.dynamic_names, dynamic.split(self.dynamic_sizes)):
                parameters[name] = value.reshape(self.parameters[name].shape)
            for observation, dt in zip(inputs, record["elapsed"]):
                _, h = functional_call(self.model, (parameters, buffers), (observation, h, dt), strict=True)
            return h

        derivatives = jacrev(transition, argnums=(0, 1), chunk_size=self.config.jacrev_chunk_size)
        with torch.enable_grad():
            j, k = vmap(derivatives, in_dims=(0, None, 1))(record["start_h"].to(self.device), flat, observations)
        return j.detach(), k.detach()

    def gradients(self, chunk, parameters=None):
        """Return ``(named_gradients, metrics)`` and commit credit state only."""
        record = self._record(chunk, parameters)
        batch, width = record["start_h"].shape
        is_rtrl = self.config.method.startswith("rtrl")
        records = [record] if is_rtrl else (self.tape + [record])[-self.config.history_blocks:]
        h0 = record["start_h"].to(self.device).detach().requires_grad_(True) if is_rtrl else None
        with torch.enable_grad():
            outputs, reconstructed_h, primal_error = self._replay(records, start_h=h0)
            loss, numerator, denominator = weighted_loss(outputs, record["targets"].to(self.device), self.config.wait_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite temporal learning loss")
            leaves = tuple(self.parameters.values()) + ((h0,) if is_rtrl else ())
            derivatives = torch.autograd.grad(loss, leaves)
        gradients = {name: derivative.detach().clone()
                     for name, derivative in zip(self.parameters, derivatives)}
        diagnostic_norm_elements = 0

        def norm(value):
            nonlocal diagnostic_norm_elements
            diagnostic_norm_elements += value.numel()
            return stable_l2(value)

        # These are reductions of already-computed derivatives, never extra
        # model probes.  Longer recorded windows do not have a local-reference
        # gradient available without another reverse sweep, so do not invent it.
        local_norms = ({name: norm(value) for name, value in gradients.items()}
                       if is_rtrl or len(records) == 1 else None)
        cross_norms = {name: 0. for name in self.parameters} if local_norms is not None else None
        local_cross_inner_product = 0. if local_norms is not None else None
        diagnostic_overflow = False
        previous_sensitivity_norm = next_sensitivity_norm = None
        next_sensitivity = None
        jacobian_rows = jacobian_chunks = sensitivity_mac = 0
        if is_rtrl:
            previous = (torch.zeros(batch, width, self.dynamic_parameter_count, device=self.device, dtype=self.dtype)
                        if self.sensitivity is None or self.config.method == "rtrl_reset" else self.sensitivity)
            correction = torch.einsum("bh,bhp->p", derivatives[-1].detach(), previous)
            for name, value in zip(self.dynamic_names, correction.split(self.dynamic_sizes)):
                cross_norms[name] = norm(value)
                product, overflow = diagnostic_inner_product(gradients[name], value)
                diagnostic_overflow |= overflow
                if not diagnostic_overflow:
                    local_cross_inner_product += product
                    diagnostic_overflow |= not math.isfinite(local_cross_inner_product)
                if diagnostic_overflow:
                    local_cross_inner_product = None
                gradients[name].add_(value.reshape_as(gradients[name]))
            j, k = self._block_jacobians(record)
            next_sensitivity = torch.bmm(j, previous) + k
            previous_sensitivity_norm, next_sensitivity_norm = norm(previous), norm(next_sensitivity)
            jacobian_rows = batch * width
            jacobian_chunks = math.ceil(width / self.config.jacrev_chunk_size)
            sensitivity_mac = batch * width * width * self.dynamic_parameter_count
        if not all(bool(torch.isfinite(value).all()) for value in gradients.values()):
            raise FloatingPointError("nonfinite temporal gradient")
        if next_sensitivity is not None and not bool(torch.isfinite(next_sensitivity).all()):
            raise FloatingPointError("nonfinite full block-RTRL sensitivity")
        full_norms = ({name: norm(value) for name, value in gradients.items()}
                      if is_rtrl or local_norms is None else dict(local_norms))
        gradient_diagnostics = {
            "full_gradient_l2": _combined_l2(full_norms.values()),
            "full_by_parameter_l2": full_norms,
            "local_gradient_l2": None if local_norms is None else _combined_l2(local_norms.values()),
            "local_by_parameter_l2": local_norms,
            "cross_boundary_gradient_l2": None if cross_norms is None else _combined_l2(cross_norms.values()),
            "cross_boundary_by_parameter_l2": cross_norms,
            "local_cross_inner_product": local_cross_inner_product,
            "diagnostic_overflow": diagnostic_overflow,
            "previous_sensitivity_l2": previous_sensitivity_norm,
            "next_sensitivity_l2": next_sensitivity_norm,
            "split_status": ("exact_block_rtrl_decomposition" if is_rtrl else
                             "single_block_no_cross_boundary_paths" if len(records) == 1 else
                             "not_computed_local_reference_requires_additional_reverse_sweep"),
            "note": "Live-loss gradients before auxiliary combination, clipping or Adam. "
                    "L2 norms describe derivative components, not causal shares of behavioral performance. "
                    "No extra model forward or reverse sweep is used for these diagnostics.",
        }
        tape = (self.tape + [record])[-self.config.history_blocks:]
        path_ticks = ((self.update_age + 1) * self.config.update_ticks
                      if self.config.method == "rtrl" else len(records) * self.config.update_ticks)
        forward_ticks = len(records) * self.config.update_ticks
        metrics = {
            "method": self.config.method, "update_age": self.update_age + 1,
            "derivative_convention": DERIVATIVE_CONVENTION,
            "start_tick": record["start_tick"], "end_tick": record["end_tick"],
            "loss": float(loss.detach()), "weighted_loss_numerator": float(numerator.detach()),
            "target_weight_denominator": float(denominator.detach()),
            "loss_ticks": self.config.update_ticks, "loss_positions": int(record["targets"].numel()),
            "credit_path_ticks": path_ticks, "credit_hard_cap_ticks": self.config.credit_ticks,
            "primal_max_abs_error": primal_error, "end_h_hash": state_digest(reconstructed_h),
            "record_hash": state_digest(record), "snapshot_hash": state_digest(record["parameters"]),
            "gradient_hash": state_digest(gradients),
            "gradient_diagnostics": gradient_diagnostics,
            "work": {
                "loss_forward_calls": forward_ticks, "loss_forward_actor_ticks": forward_ticks * batch,
                "jacobian_forward_calls": self.config.update_ticks if is_rtrl else 0,
                "jacobian_forward_actor_ticks": self.config.update_ticks * batch if is_rtrl else 0,
                "loss_reverse_sweeps": 1, "jacobian_reverse_rows": jacobian_rows,
                "jacobian_reverse_chunks": jacobian_chunks,
                "sensitivity_matmul_mac": sensitivity_mac,
                "trace_scalars": 0 if next_sensitivity is None else next_sensitivity.numel(),
                "trace_bytes": tensor_bytes(next_sensitivity), "tape_tensor_bytes": tensor_bytes(tape),
                "dynamic_parameters": self.dynamic_parameter_count,
                "diagnostic_forward_calls": 0, "diagnostic_reverse_sweeps": 0,
                "diagnostic_norm_elements": diagnostic_norm_elements,
                "diagnostic_inner_product_elements": self.dynamic_parameter_count if is_rtrl else 0,
                "optimizer_steps": 0, "world_steps": 0,
                "reset_uses_full_jacobian_and_matmul": self.config.method == "rtrl_reset",
                "note": "Counts describe this gradient provider; runner live forwards and optimizer work are separate. "
                        "MAC counts cover J@S only, not Jacobian construction or the model's forward/backward operations.",
            },
        }
        self.tape, self.sensitivity = tape, next_sensitivity
        self.batch_size, self.width = batch, width
        self.update_age += 1
        self.trajectory_hash = state_digest((self.trajectory_hash, metrics["record_hash"]))
        metrics["trajectory_hash"] = self.trajectory_hash
        self.last_metrics = deepcopy(metrics)
        return gradients, metrics

    def state_dict(self):
        state = {"format": FORMAT_VERSION, "config": asdict(self.config), "model_metadata": deepcopy(self.model_metadata),
                 "parameter_layout": self.parameter_layout, "dynamic_names": self.dynamic_names, "local_names": self.local_names,
                 "buffer_layout": self.buffer_layout, "update_age": self.update_age,
                 "ticks": self.update_age * self.config.update_ticks, "batch_size": self.batch_size, "width": self.width,
                 "sensitivity": _cpu(self.sensitivity), "tape": _cpu(self.tape),
                 "trajectory_hash": self.trajectory_hash, "last_metrics": deepcopy(self.last_metrics)}
        state["state_hash"] = state_digest(state)
        return state

    def load_state_dict(self, state):
        expected_keys = set(self.state_dict())
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise ValueError("incomplete temporal credit state")
        if state["state_hash"] != state_digest({key: value for key, value in state.items() if key != "state_hash"}):
            raise ValueError("temporal credit state hash mismatch")
        for name, expected in (("format", FORMAT_VERSION), ("config", asdict(self.config)),
                               ("model_metadata", self.model_metadata), ("parameter_layout", self.parameter_layout),
                               ("dynamic_names", self.dynamic_names), ("local_names", self.local_names),
                               ("buffer_layout", self.buffer_layout)):
            if state[name] != expected:
                raise ValueError(f"temporal credit checkpoint mismatch: {name}")
        age = _integer(state["update_age"], "update_age")
        if state["ticks"] != age * self.config.update_ticks:
            raise ValueError("credit clock mismatch")
        tape = state["tape"]
        if not isinstance(tape, list) or len(tape) != min(age, self.config.history_blocks):
            raise ValueError("incomplete credit history tape")
        for index, record in enumerate(tape):
            self._validate_record(record)
            if record["end_tick"] != (age - len(tape) + index + 1) * self.config.update_ticks:
                raise ValueError("credit tape is not the latest contiguous history")
            if index and not torch.equal(record["start_h"], tape[index - 1]["end_h"]):
                raise ValueError("credit tape hidden discontinuity")
        sensitivity = state["sensitivity"]
        if age:
            batch = _integer(state["batch_size"], "batch_size", 1)
            width = _integer(state["width"], "width", 1)
            if any(record["start_h"].shape != (batch, width) for record in tape):
                raise ValueError("credit tape/state dimensions differ")
            if self.config.method.startswith("rtrl"):
                if (not isinstance(sensitivity, torch.Tensor) or sensitivity.shape != (batch, width, self.dynamic_parameter_count)
                        or sensitivity.dtype != self.dtype or not bool(torch.isfinite(sensitivity).all())):
                    raise ValueError("invalid full block-RTRL sensitivity")
            elif sensitivity is not None:
                raise ValueError("recorded-window credit cannot contain RTRL sensitivity")
            if not isinstance(state["last_metrics"], Mapping) or state["last_metrics"].get("update_age") != age:
                raise ValueError("incomplete last credit metrics")
        elif any(state[key] is not None for key in ("batch_size", "width", "sensitivity", "last_metrics")):
            raise ValueError("newborn credit state contains learning history")
        chain = state["trajectory_hash"]
        if not isinstance(chain, str) or len(chain) != 64 or any(char not in "0123456789abcdef" for char in chain):
            raise ValueError("invalid trajectory hash")
        self.tape = _cpu(tape)
        self.sensitivity = None if sensitivity is None else sensitivity.detach().to(self.device).clone()
        self.batch_size, self.width = state["batch_size"], state["width"]
        self.update_age, self.trajectory_hash = age, chain
        self.last_metrics = deepcopy(state["last_metrics"])


def _native_process_memory():
    """Native process memory, including Torch allocations; lifetime peak scope."""
    import os
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return {"working_set_bytes": counters.WorkingSetSize,
                "lifetime_peak_working_set_bytes": counters.PeakWorkingSetSize,
                "private_commit_bytes": counters.PagefileUsage,
                "lifetime_peak_private_commit_bytes": counters.PeakPagefileUsage}
    import resource
    import sys
    scale = 1 if sys.platform == "darwin" else 1024
    return {"lifetime_peak_working_set_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale}


def benchmark_credit(*, width=64, batch_size=8, device="cpu", dtype="float64", blocks=1, method="rtrl",
                     interface="core", n_jobs=4, port_matched=False):
    """Synthetic derivative-only benchmark: no optimizer, world or behavior score.

    Run each case in a fresh process to interpret the OS lifetime peak.  CUDA
    allocator peaks are reset immediately before each timed gradient call.
    """
    import platform
    import time
    from experiments.undirected.model import UndirectedModel, PortGraphModel

    _integer(blocks, "blocks", 1)
    if dtype not in ("float32", "float64"):
        raise ValueError("benchmark dtype must be float32 or float64")
    if interface not in ("core", "ports"):
        raise ValueError("benchmark interface must be core or ports")
    _integer(n_jobs, "n_jobs", 1)
    torch.set_num_threads(1)
    torch.manual_seed(117)
    selected_device, selected_dtype = torch.device(device), getattr(torch, dtype)
    model_type = UndirectedModel if interface == "core" else PortGraphModel
    model = model_type(width=width, n_jobs=n_jobs, input_size=28 * n_jobs).to(device=selected_device, dtype=selected_dtype)
    if interface == "core" and port_matched:
        model.initialize_port_matched_interfaces_(pulse_gain=.25)
    degree = min(16, width - 1)
    if width * degree % 2:
        degree -= 1
    model.initialize_birth(mask_seed=119, weight_seed=121, degree=degree)
    engine = TemporalCreditEngine(model, CreditConfig(method))
    h = model.initial_state(batch_size)
    generator = torch.Generator().manual_seed(123)
    results = []
    for index in range(blocks):
        observations = (torch.randn(128, batch_size, model.input_size, generator=generator) * .02).to(selected_device)
        targets = torch.randint(model.n_actions, (128, batch_size, model.n_jobs), generator=generator).to(selected_device)
        start_h = h.detach().clone()
        parameters = snapshot_parameters(model)
        with torch.no_grad():
            for observation in observations:
                _, h = model(observation, h, .25)
        chunk = {"observations": observations, "targets": targets, "elapsed": (.25,) * 128,
                 "start_h": start_h, "end_h": h.detach().clone(),
                 "start_tick": index * 128, "end_tick": (index + 1) * 128}
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
            torch.cuda.reset_peak_memory_stats(selected_device)
            baseline_cuda = torch.cuda.memory_allocated(selected_device)
        before = _native_process_memory()
        started = time.perf_counter()
        gradients, metrics = engine.gradients(chunk, parameters)
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        seconds = time.perf_counter() - started
        result = {"block": index + 1, "seconds": seconds, "cold_first_derivative_call": index == 0,
                  "native_memory_before": before, "native_memory_after": _native_process_memory(),
                  "work": metrics["work"], "primal_max_abs_error": metrics["primal_max_abs_error"],
                  "gradient_finite": all(bool(torch.isfinite(value).all()) for value in gradients.values()),
                  "gradient_hash": metrics["gradient_hash"], "trajectory_hash": metrics["trajectory_hash"]}
        if selected_device.type == "cuda":
            result.update(cuda_allocated_baseline_bytes=baseline_cuda,
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected_device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(selected_device))
        results.append(result)
    return {"format": "undirected-credit-benchmark-v1", "credit_format": FORMAT_VERSION, "python": platform.python_version(),
            "torch": torch.__version__, "device": str(selected_device), "dtype": dtype,
            "cuda_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
            "cpu_threads": torch.get_num_threads(), "width": width, "batch_size": batch_size,
            "update_ticks": 128, "method": method, "seed": 117,
            "interface": interface, "port_matched": port_matched if interface == "core" else True,
            "model_config": model.config(), "dynamic_parameters": engine.dynamic_parameter_count,
            "dynamic_parameter_names": engine.dynamic_names, "local_parameter_names": engine.local_names,
            "jacrev_chunk_size": engine.config.jacrev_chunk_size, "results": results,
            "scope": "Synthetic gradient-provider work only. No optimization, task evaluation or behavior score. "
                     "OS peaks include interpreter/import/fixture memory over this process lifetime; "
                     "CUDA allocator peaks cover each timed call including baseline tensors. "
                     "Forward Jacobian construction costs are additional to the reported J@S MAC count."}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Derivative-only temporal credit benchmark")
    parser.add_argument("--benchmark", action="store_true", required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--method", choices=CREDIT_METHODS, default="rtrl")
    parser.add_argument("--interface", choices=("core", "ports"), default="core")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--port-matched", action="store_true")
    arguments = vars(parser.parse_args())
    arguments.pop("benchmark")
    print(json.dumps(benchmark_credit(**arguments), indent=2), flush=True)

