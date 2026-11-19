"""Supervised forward eligibility for the unchanged continuous Heun controller.

The primal remains dh/dt=-h+W*tanh(h), followed by one encoder pulse and a
tanh readout. At EACH Heun stage the tangent calculation drops cross-neuron
state sensitivities. This is a biased diagonal approximation, not RFLO/e-prop
replication or exact RTRL. It retains synaptic derivative traces rather than
the previous activity-only EMA. Carrying them after parameter updates also
introduces the usual stale-sensitivity approximation.

The observer reconstructs autonomous primal intermediates because the frozen
model.step API does not expose them. This costs an additional autonomous flow
per observation, with no extra encoder/head/world call, and is counted below.
No optimizer, autograd backward, hidden-state replacement, homeostasis or Oja
update occurs in this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral
from typing import Any

import torch
from torch.nn import functional as F


FORMAT_VERSION = "credit-diagonal-heun-v1"
PARAMETER_NAMES = ("recurrent", "encoder.weight", "head.weight", "head.bias")
GAIN_TENSORS = ("last_gain", "gain_log_abs_sum", "gain_zero_count", "gain_min", "gain_max")


@dataclass(frozen=True)
class LocalConfig:
    feedback: str = "symmetric"
    shuffle: bool = False
    seed: int = 0
    n_jobs: int = 4
    n_actions: int = 11
    wait_weight: float = .1

    def __post_init__(self):
        if self.feedback not in {"symmetric", "random"}:
            raise ValueError("feedback must be symmetric or random")
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
               for value in (self.n_jobs, self.n_actions)):
            raise ValueError("positive job/action dimensions required")
        if not math.isfinite(self.wait_weight) or self.wait_weight <= 0:
            raise ValueError("wait_weight must be finite and positive")


def _seed(seed: int, namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{FORMAT_VERSION}/{seed}/{namespace}".encode()).digest()[:8], "little") % (2**63-1)


@torch.no_grad()
def analytic_head_gradients(logits: torch.Tensor, post_h: torch.Tensor, targets: torch.Tensor,
                            wait_weight: float = .1) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Unnormalized weighted-CE head gradients, with no eligibility allocation.

    Numerators must be divided by the SUM of target weights over the complete
    learning chunk, matching weighted F.cross_entropy(reduction='mean'). The
    returned weighted residual is also used as the hidden learning signal.
    """
    if (logits.ndim != 3 or post_h.ndim != 2 or targets.shape != logits.shape[:2]
            or post_h.shape[0] != logits.shape[0] or targets.dtype != torch.long):
        raise ValueError("invalid head/logit/target shapes or target dtype")
    if not math.isfinite(wait_weight) or wait_weight <= 0:
        raise ValueError("wait_weight must be finite and positive")
    if post_h.device != logits.device or post_h.dtype != logits.dtype or targets.device != logits.device:
        raise ValueError("head values/targets must share device and activity dtype")
    labels = targets.detach()
    ones = logits.new_ones(labels.shape)
    weights = torch.where(labels == 0, ones*wait_weight, ones)
    log_probability = logits.detach().log_softmax(-1)
    residual = (log_probability.exp() - F.one_hot(labels, logits.shape[-1]).to(logits.dtype)) * weights[..., None]
    flat = residual.flatten(1)
    features = post_h.detach().tanh()
    gradients = {"head.weight": flat.T @ features, "head.bias": flat.sum(0)}
    loss = -log_probability.gather(-1, labels[..., None]).squeeze(-1) * weights
    return gradients, {"weight_mass": weights.double().sum(), "weighted_loss_sum": loss.double().sum(), "residual": residual}


class ForwardEligibility:
    """Persistent E[B,H,H], F[B,H,D], feedback and gradient numerators."""
    def __init__(self, batch_size: int, width: int, input_size: int,
                 config: LocalConfig | None = None, device="cpu", dtype=torch.float32):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
               for value in (batch_size, width, input_size)):
            raise ValueError("positive integer helper dimensions required")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("floating-point eligibility dtype required")
        self.batch_size, self.width, self.input_size = int(batch_size), int(width), int(input_size)
        self.config = LocalConfig() if config is None else config
        if self.config.shuffle and self.batch_size < 2:
            raise ValueError("shuffle requires at least two independent actors")
        self.E = torch.zeros(self.batch_size, width, width, device=device, dtype=dtype)
        self.F = torch.zeros(self.batch_size, width, input_size, device=self.E.device, dtype=dtype)
        self.device, self.dtype = self.E.device, dtype
        output_size = self.config.n_jobs * self.config.n_actions
        shapes = ((width, width), (width, input_size), (output_size, width), (output_size,))
        self.gradients = {name: torch.zeros(shape, device=self.device, dtype=dtype) for name, shape in zip(PARAMETER_NAMES, shapes)}
        self.weight_mass = torch.zeros((), device=self.device, dtype=torch.float64)
        self.weighted_loss_sum = self.weight_mass.clone()
        # Observer-only statistics: no normalization or feedback into E/F.
        self.last_gain = torch.ones(self.batch_size, width, device=self.device, dtype=dtype)
        self.gain_log_abs_sum = self.weight_mass.clone()
        self.gain_zero_count = torch.zeros((), device=self.device, dtype=torch.int64)
        self.gain_min = torch.ones((), device=self.device, dtype=dtype)
        self.gain_max = self.gain_min.clone()
        self.fixed_feedback: torch.Tensor | None = None
        self.initial_feedback_row_norms: torch.Tensor | None = None
        self.feedback_initialized = False
        self.feedback_rng = torch.Generator(device="cpu").manual_seed(_seed(self.config.seed, "feedback"))
        self.shuffle_rng = torch.Generator(device="cpu").manual_seed(_seed(self.config.seed, "shuffle"))
        self.observed_ticks = self.loss_ticks = self.pending_loss_ticks = self.solver_substeps = self.eligibility_resets = 0
        self.observed_elapsed = 0.
        self.last_shift = 0

    @torch.no_grad()
    def initialize_feedback_(self, head_weight: torch.Tensor):
        shape = (self.config.n_jobs*self.config.n_actions, self.width)
        if head_weight.shape != shape or head_weight.device != self.device or head_weight.dtype != self.dtype:
            raise ValueError("readout weight does not match helper dimensions/device/dtype")
        if self.feedback_initialized:
            return
        transposed = head_weight.detach().T
        self.initial_feedback_row_norms = torch.linalg.vector_norm(transposed, dim=1).clone()
        if self.config.feedback == "random":
            random = torch.randn(transposed.shape, generator=self.feedback_rng, device="cpu", dtype=torch.float64)
            norms = torch.linalg.vector_norm(random, dim=1, keepdim=True)
            if bool((norms == 0).any()):
                raise FloatingPointError("zero random-feedback row")
            self.fixed_feedback = (random / norms * self.initial_feedback_row_norms.cpu().double()[:, None]).to(self.device, self.dtype)
        self.feedback_initialized = True

    @torch.no_grad()
    def reset_eligibility_(self):
        """Explicit trace control/disconnected history; never modifies live h."""
        self.E.zero_()
        self.F.zero_()
        self.eligibility_resets += 1

    @torch.no_grad()
    def clear_gradients(self):
        """Consume only numerator accumulators, preserving E/F/feedback/RNG."""
        for gradient in self.gradients.values():
            gradient.zero_()
        self.weight_mass.zero_()
        self.weighted_loss_sum.zero_()
        self.pending_loss_ticks = 0

    @torch.no_grad()
    def observe(self, model, observation, pre_h, post_h, dt, logits, target, *, accumulate=True):
        shape = (self.batch_size, self.width)
        if pre_h.shape != shape or post_h.shape != shape or observation.shape != (self.batch_size, self.input_size):
            raise ValueError("observation/hidden shape differs from eligibility")
        for value in (observation, pre_h, post_h, logits, model.recurrent):
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError("helper/model/activity device or dtype mismatch")
        if model.recurrent.shape != (self.width, self.width) or getattr(model, "kind", "learned") == "gru":
            raise ValueError("helper requires the unchanged continuous recurrent core")
        if logits.shape != (self.batch_size, self.config.n_jobs, self.config.n_actions):
            raise ValueError("logit shape differs from helper output dimensions")
        elapsed, limit = float(dt), float(model.max_step)
        if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(limit) or limit <= 0:
            raise ValueError("invalid solver elapsed time or maximum step")
        if not isinstance(accumulate, bool):
            raise ValueError("accumulate must be boolean")
        self.initialize_feedback_(model.head.weight)
        n = max(1, math.ceil(elapsed/limit))
        delta = elapsed/n
        h, weight = pre_h.detach(), model.recurrent.detach()
        diagonal = weight.diag()
        gain = torch.ones_like(self.last_gain)
        for _ in range(n):
            rate = h.tanh()
            first = -h + F.linear(rate, weight)
            trial = h + delta*first
            trial_rate = trial.tanh()
            a = -1. + diagonal * (1. - rate.square())
            a_trial = -1. + diagonal * (1. - trial_rate.square())
            gain.mul_(1. + .5*delta*(a + a_trial*(1. + delta*a)))
            first_E = a[..., None]*self.E + rate[:, None, :]
            trial_E = self.E + delta*first_E
            self.E = self.E + .5*delta*(first_E + a_trial[..., None]*trial_E + trial_rate[:, None, :])
            first_F = a[..., None]*self.F
            trial_F = self.F + delta*first_F
            self.F = self.F + .5*delta*(first_F + a_trial[..., None]*trial_F)
            second = -trial + F.linear(trial_rate, weight)
            h = h + .5*delta*(first + second)
        # The encoder pulse occurs AFTER all autonomous substeps.
        self.F.add_(observation.detach()[:, None, :])
        self.last_gain.copy_(gain)
        absolute_gain = gain.abs().double()
        self.gain_log_abs_sum.add_(torch.where(absolute_gain > 0, absolute_gain.log(), 0.).sum())
        self.gain_zero_count.add_((absolute_gain == 0).sum())
        if self.observed_ticks == 0:
            self.gain_min.copy_(gain.min())
            self.gain_max.copy_(gain.max())
        else:
            self.gain_min.copy_(torch.minimum(self.gain_min, gain.min()))
            self.gain_max.copy_(torch.maximum(self.gain_max, gain.max()))
        self.observed_ticks += 1
        self.observed_elapsed += elapsed
        self.solver_substeps += n
        if not accumulate:
            return
        head, metrics = analytic_head_gradients(logits, post_h, target, self.config.wait_weight)
        flat = metrics["residual"].flatten(1)
        feedback = model.head.weight.detach().T if self.config.feedback == "symmetric" else self.fixed_feedback
        signal = (flat @ feedback.T) * (1. - post_h.detach().tanh().square())
        self.last_shift = int(torch.randint(1, self.batch_size, (), generator=self.shuffle_rng)) if self.config.shuffle else 0
        if self.last_shift:
            signal = torch.roll(signal, self.last_shift, dims=0)
        self.gradients["recurrent"].add_((signal[..., None]*self.E).sum(0))
        self.gradients["encoder.weight"].add_((signal[..., None]*self.F).sum(0))
        for name, gradient in head.items():
            self.gradients[name].add_(gradient)
        self.weight_mass.add_(metrics["weight_mass"])
        self.weighted_loss_sum.add_(metrics["weighted_loss_sum"])
        self.pending_loss_ticks += 1
        self.loss_ticks += 1

    @torch.no_grad()
    def normalized_gradients(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if not float(self.weight_mass) > 0:
            raise ValueError("no positive-weight loss samples accumulated")
        denominator = self.weight_mass.to(dtype=self.dtype)
        gradients = {name: value/denominator for name, value in self.gradients.items()}
        if not all(bool(torch.isfinite(value).all()) for value in (*gradients.values(), self.E, self.F)):
            raise FloatingPointError("nonfinite forward eligibility or gradient")
        return gradients, {"weight_mass": float(self.weight_mass), "weighted_loss_sum": float(self.weighted_loss_sum),
                           "weighted_loss": float(self.weighted_loss_sum/self.weight_mass),
                           "pending_loss_ticks": self.pending_loss_ticks}

    def state_dict(self):
        clone = lambda value: None if value is None else value.detach().cpu().clone()
        return {"format": FORMAT_VERSION, "config": asdict(self.config), "batch_size": self.batch_size,
                "width": self.width, "input_size": self.input_size, "dtype": str(self.dtype),
                "E": clone(self.E), "F": clone(self.F), "gradients": {name: clone(value) for name, value in self.gradients.items()},
                "weight_mass": clone(self.weight_mass), "weighted_loss_sum": clone(self.weighted_loss_sum),
                **{name: clone(getattr(self, name)) for name in GAIN_TENSORS},
                "fixed_feedback": clone(self.fixed_feedback), "initial_feedback_row_norms": clone(self.initial_feedback_row_norms),
                "feedback_initialized": self.feedback_initialized, "feedback_rng": self.feedback_rng.get_state().clone(),
                "shuffle_rng": self.shuffle_rng.get_state().clone(),
                **{name: deepcopy(getattr(self, name)) for name in ("observed_ticks", "observed_elapsed", "loss_ticks",
                   "pending_loss_ticks", "solver_substeps", "eligibility_resets", "last_shift")}}

    @torch.no_grad()
    def load_state_dict(self, state):
        for name, expected in (("format", FORMAT_VERSION), ("config", asdict(self.config)), ("batch_size", self.batch_size),
                               ("width", self.width), ("input_size", self.input_size), ("dtype", str(self.dtype))):
            if state.get(name) != expected:
                raise ValueError(f"eligibility checkpoint mismatch: {name}")
        for name in ("E", "F", "weight_mass", "weighted_loss_sum", *GAIN_TENSORS):
            destination, value = getattr(self, name), state[name]
            if value.shape != destination.shape or value.dtype != destination.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid eligibility tensor: {name}")
        for name, destination in self.gradients.items():
            value = state["gradients"][name]
            if value.shape != destination.shape or value.dtype != destination.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid gradient numerator: {name}")
        for name in ("observed_ticks", "loss_ticks", "pending_loss_ticks", "solver_substeps", "eligibility_resets", "last_shift"):
            if isinstance(state[name], bool) or not isinstance(state[name], Integral) or state[name] < 0:
                raise ValueError(f"invalid eligibility counter: {name}")
        if not (0 <= state["pending_loss_ticks"] <= state["loss_ticks"] <= state["observed_ticks"]
                and 0 <= state["last_shift"] < self.batch_size and math.isfinite(state["observed_elapsed"]) and state["observed_elapsed"] >= 0):
            raise ValueError("inconsistent eligibility checkpoint counters")
        if (float(state["weight_mass"]) < 0 or float(state["weighted_loss_sum"]) < 0
                or bool(state["pending_loss_ticks"]) != bool(float(state["weight_mass"]) > 0)):
            raise ValueError("inconsistent loss numerator/mass")
        if (not 0 <= int(state["gain_zero_count"]) <= state["observed_ticks"]*self.batch_size*self.width
                or float(state["gain_min"]) > float(state["gain_max"])):
            raise ValueError("inconsistent diagonal-gain observer")
        initialized = state["feedback_initialized"]
        rows, feedback = state["initial_feedback_row_norms"], state["fixed_feedback"]
        if not isinstance(initialized, bool) or (rows is None) != (not initialized):
            raise ValueError("invalid feedback initialization state")
        if initialized and (rows.shape != (self.width,) or rows.dtype != self.dtype or not bool(torch.isfinite(rows).all()) or bool((rows < 0).any())):
            raise ValueError("invalid initial feedback scales")
        if (feedback is not None) != (initialized and self.config.feedback == "random"):
            raise ValueError("invalid fixed feedback presence")
        if feedback is not None and (feedback.shape != (self.width, self.config.n_jobs*self.config.n_actions)
                or feedback.dtype != self.dtype or not bool(torch.isfinite(feedback).all())):
            raise ValueError("invalid fixed feedback matrix")
        feedback_rng, shuffle_rng = torch.Generator(device="cpu"), torch.Generator(device="cpu")
        feedback_rng.set_state(state["feedback_rng"].cpu())
        shuffle_rng.set_state(state["shuffle_rng"].cpu())
        for name in ("E", "F", "weight_mass", "weighted_loss_sum", *GAIN_TENSORS):
            getattr(self, name).copy_(state[name].to(self.device))
        for name in self.gradients:
            self.gradients[name].copy_(state["gradients"][name].to(self.device))
        self.fixed_feedback = None if feedback is None else feedback.to(self.device).clone()
        self.initial_feedback_row_norms = None if rows is None else rows.to(self.device).clone()
        self.feedback_initialized = initialized
        self.feedback_rng.set_state(feedback_rng.get_state())
        self.shuffle_rng.set_state(shuffle_rng.get_state())
        for name in ("observed_ticks", "observed_elapsed", "loss_ticks", "pending_loss_ticks", "solver_substeps", "eligibility_resets", "last_shift"):
            setattr(self, name, deepcopy(state[name]))

    @torch.no_grad()
    def summary(self, model=None):
        tensors = [self.E, self.F, self.weight_mass, self.weighted_loss_sum, *self.gradients.values()]
        tensors += [getattr(self, name) for name in GAIN_TENSORS]
        tensors += [value for value in (self.fixed_feedback, self.initial_feedback_row_norms) if value is not None]
        feedback = None
        if self.initial_feedback_row_norms is not None:
            scales = self.initial_feedback_row_norms.double()
            feedback = {"initial_row_norm_min": float(scales.min()), "initial_row_norm_max": float(scales.max()),
                        "initial_row_norm_mean": float(scales.mean()), "initial_row_norms": scales.tolist(),
                        "fixed_random_parameters": 0 if self.fixed_feedback is None else self.fixed_feedback.numel()}
            if model is not None:
                current = model.head.weight.detach().T.double()
                matrix = current if self.fixed_feedback is None else self.fixed_feedback.double()
                denominator = torch.linalg.vector_norm(current)*torch.linalg.vector_norm(matrix)
                feedback["current_head_global_cosine"] = float((current*matrix).sum()/denominator) if float(denominator) else None
                feedback["current_feedback_frob"] = float(torch.linalg.vector_norm(matrix))
        samples = self.observed_ticks*self.batch_size*self.width
        zero_count = int(self.gain_zero_count)
        nonzero_log_mean = float(self.gain_log_abs_sum)/(samples-zero_count) if samples > zero_count else None
        log_mean = nonzero_log_mean if samples and zero_count == 0 else None
        geometric = lambda exponent: math.exp(exponent) if exponent < math.log(float.fromhex("0x1.fffffffffffffp+1023")) else None
        last_gain = self.last_gain.double()
        gain_summary = {"last_min": float(last_gain.min()), "last_median": float(last_gain.median()),
                        "last_max": float(last_gain.max()), "last_mean": float(last_gain.mean()),
                        "observed_min": float(self.gain_min) if samples else None,
                        "observed_max": float(self.gain_max) if samples else None,
                        "neuron_actor_observations": samples, "zero_gain_count": zero_count,
                        "mean_log_abs_gain_per_observation": log_mean,
                        "mean_log_abs_gain_nonzero_only": nonzero_log_mean,
                        "typical_log_abs_gain_over_64_observations": None if log_mean is None else 64*log_mean,
                        "geometric_mean_abs_gain_per_observation": (0. if zero_count else geometric(log_mean)) if samples else None,
                        "geometric_mean_abs_gain_over_64_observations": (0. if zero_count else geometric(64*log_mean)) if samples else None,
                        "note": "Observer only: product of homogeneous diagonal Heun factors per observation; 64-observation values extrapolate the mean log absolute gain. This omits cross-neuron paths and source injection and is not whole-network memory or a gradient horizon."}
        return {"format": FORMAT_VERSION, "config": asdict(self.config), "E_shape": list(self.E.shape), "F_shape": list(self.F.shape),
                "eligibility_bytes": (self.E.numel()+self.F.numel())*self.E.element_size(),
                "all_persistent_tensor_bytes": sum(value.numel()*value.element_size() for value in tensors),
                "private_rng_bytes": self.feedback_rng.get_state().numel()+self.shuffle_rng.get_state().numel(),
                "observed_ticks": self.observed_ticks, "loss_ticks": self.loss_ticks, "pending_loss_ticks": self.pending_loss_ticks,
                "solver_substeps": self.solver_substeps, "reconstructed_flow_ticks": self.observed_ticks,
                "reconstructed_flow_matvecs": 2*self.solver_substeps, "eligibility_resets": self.eligibility_resets,
                "E_frob": float(torch.linalg.vector_norm(self.E.double())), "F_frob": float(torch.linalg.vector_norm(self.F.double())),
                "E_rms": float(self.E.double().square().mean().sqrt()), "F_rms": float(self.F.double().square().mean().sqrt()),
                "E_absmax": float(self.E.abs().max()), "F_absmax": float(self.F.abs().max()),
                "diagonal_homogeneous_gain": gain_summary,
                "feedback": feedback, "tangent_note": "Stage-wise diagonal approximation; stale after parameter updates; no guarantee of long-delay credit.",
                "budget_note": "Persistent tensor counts exclude temporary Heun/tangent arrays and optimizer/autograd storage. Observer reconstructs one autonomous flow per tick."}
