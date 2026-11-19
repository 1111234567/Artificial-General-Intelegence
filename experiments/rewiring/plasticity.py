"""Exact-zero structural bookkeeping around an unchanged recurrent W forward.

Dormant coordinates remain numerical zeros in W itself. Their ordinary BPTT
derivatives are used only by a private, lifetime-aged shadow Adam proposal.
The live Adam receives active gradients, then an explicit Euclidean shrink
step is applied. This is an optimizer heuristic, not an exact Adam L1 solver.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import math

import torch


ARMS = ("dense", "retire", "learned", "random")
FORMAT_VERSION = "rewire-operator-v1"


def _number(value, name, *, lower=0., upper=None, strict_lower=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if (not math.isfinite(result) or (result <= lower if strict_lower else result < lower)
            or (upper is not None and result > upper)):
        raise ValueError(f"invalid {name}")
    return result


def _integer(value, name, *, lower=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"invalid {name}")
    return value


def tensor_hash(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class RewireConfig:
    arm: str = "dense"
    shrink_step: float = 0.
    window_updates: int = 8
    structural_seed: int = 0

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError("unknown rewiring arm")
        _number(self.shrink_step, "shrink_step")
        _integer(self.window_updates, "window_updates", lower=1)
        _integer(self.structural_seed, "structural_seed")
        if self.structural_seed >= 2 ** 64:
            raise ValueError("structural_seed exceeds the torch generator range")


class RewireOperator:
    """No model/optimizer/h/global-RNG mutation occurs during attachment."""

    _FLOAT_BUFFERS = ("shadow_m", "shadow_v", "proposal_sum")
    _COUNT_BUFFERS = ("shadow_count", "proposal_count")
    _MASK_BUFFERS = ("mask", "initial_mask", "survival_mask", "ever_born")
    _ROW_BUFFERS = ("births_per_row", "deaths_per_row", "peak_per_row", "min_per_row")

    def __init__(self, weight, config: RewireConfig):
        self._validate_weight(weight)
        if not isinstance(config, RewireConfig):
            raise ValueError("config must be RewireConfig")
        self.config = config
        self.mask = weight.detach().ne(0).clone()
        self.initial_mask = self.mask.clone()
        self.survival_mask = self.mask.clone()
        self.ever_born = torch.zeros_like(self.mask)
        for name in self._FLOAT_BUFFERS:
            setattr(self, name, torch.zeros_like(weight, requires_grad=False))
        for name in self._COUNT_BUFFERS:
            setattr(self, name, torch.zeros_like(weight, dtype=torch.long, requires_grad=False))
        rows = self.mask.sum(-1)
        self.births_per_row = torch.zeros_like(rows)
        self.deaths_per_row = torch.zeros_like(rows)
        self.peak_per_row = rows.clone()
        self.min_per_row = rows.clone()
        self.peak_active = self.min_active = int(rows.sum())
        self.cumulative_births = self.cumulative_deaths = 0
        self.updates = self.updates_in_window = 0
        self.generator = torch.Generator(device="cpu").manual_seed(config.structural_seed)

    @staticmethod
    def _validate_weight(weight):
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.numel()
                or weight.shape[0] != weight.shape[1] or weight.dtype not in (torch.float32, torch.float64)):
            raise ValueError("weight must be a nonempty square float32/float64 tensor")
        if not bool(torch.isfinite(weight).all()):
            raise FloatingPointError("nonfinite recurrent weight")

    def _validate_gradient(self, gradient):
        if (not isinstance(gradient, torch.Tensor) or gradient.shape != self.mask.shape
                or gradient.dtype != self.shadow_m.dtype or gradient.device != self.mask.device):
            raise ValueError("recurrent gradient shape/dtype/device mismatch")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite recurrent gradient")

    def before_gradient(self, raw_gradient):
        """Mask before the trainer computes the live joint clipping norm."""
        self._validate_gradient(raw_gradient)
        if self.config.arm == "dense":
            return raw_gradient
        return raw_gradient * self.mask

    def _validate_runtime(self, weight, adam_state, raw_gradient, clip_scale, lr, betas, eps):
        self._validate_weight(weight)
        self._validate_gradient(raw_gradient)
        if weight.shape != self.mask.shape or weight.dtype != self.shadow_m.dtype or weight.device != self.mask.device:
            raise ValueError("operator and recurrent weight mismatch")
        clip_scale = _number(clip_scale, "clip_scale", upper=1.)
        lr = _number(lr, "lr")
        eps = _number(eps, "eps", strict_lower=True)
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must contain two values")
        betas = tuple(_number(value, "Adam beta", upper=1.) for value in betas)
        if any(value == 1. for value in betas):
            raise ValueError("Adam betas must be less than one")
        if not isinstance(adam_state, Mapping):
            raise ValueError("Adam state must be a mapping")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name not in adam_state:
                continue
            value = adam_state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != weight.shape
                    or value.dtype != weight.dtype or value.device != weight.device):
                raise ValueError(f"Adam {name} shape/dtype/device mismatch")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"nonfinite Adam {name}")
            if name != "exp_avg" and bool((value < 0).any()):
                raise ValueError(f"negative Adam {name}")
        if "step" in adam_state:
            step = torch.as_tensor(adam_state["step"])
            if step.numel() != 1 or not bool(torch.isfinite(step).all()) or float(step) < 0:
                raise ValueError("invalid Adam step")
        return clip_scale, lr, betas, eps

    def _reset_shadow(self, selected):
        for name in (*self._FLOAT_BUFFERS, *self._COUNT_BUFFERS):
            getattr(self, name).masked_fill_(selected, 0)

    def _shadow_observe(self, dormant, gradient, clip_scale, lr, betas, eps):
        if not bool(dormant.any()):
            return
        beta1, beta2 = betas
        g = gradient[dormant] * clip_scale
        m = beta1 * self.shadow_m[dormant] + (1. - beta1) * g
        v = beta2 * self.shadow_v[dormant] + (1. - beta2) * g.square()
        count = self.shadow_count[dormant] + 1
        # Float64 bias corrections mirror scalar Adam ages without losing the
        # first few corrections to float32 cancellation.
        age = count.to(torch.float64)
        correction1 = (1. - beta1 ** age).to(m.dtype)
        correction2 = (1. - beta2 ** age).to(v.dtype)
        delta = -lr * (m / correction1) / ((v / correction2).sqrt() + eps)
        accumulated = self.proposal_sum[dormant] + delta
        if not all(bool(torch.isfinite(value).all()) for value in (m, v, delta, accumulated)):
            raise FloatingPointError("nonfinite dormant shadow Adam proposal")
        self.shadow_m[dormant], self.shadow_v[dormant] = m, v
        self.shadow_count[dormant] = count
        self.proposal_sum[dormant] = accumulated
        self.proposal_count[dormant] += 1

    @staticmethod
    def _clear_live_moments(adam_state, selected):
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in adam_state:
                adam_state[name].masked_fill_(selected, 0)

    def _observe_counts(self):
        rows = self.mask.sum(-1)
        self.peak_per_row = torch.maximum(self.peak_per_row, rows)
        self.min_per_row = torch.minimum(self.min_per_row, rows)
        self.peak_active = max(self.peak_active, int(rows.sum()))
        self.min_active = min(self.min_active, int(rows.sum()))

    @torch.no_grad()
    def apply_after_optimizer(self, weight, adam_state, raw_gradient, clip_scale, lr, betas, eps):
        """Apply structural work at the complete live optimizer boundary.

        A newly retired edge cannot contribute evidence or be reborn on this
        update. Random births relocate this branch's own informed candidates;
        signed candidate values and birth counts match per recipient, while
        different arms can subsequently follow different event schedules.
        """
        clip_scale, lr, betas, eps = self._validate_runtime(
            weight, adam_state, raw_gradient, clip_scale, lr, betas, eps)
        before = self.mask.clone()
        dormant_at_start = ~before
        window = self.updates_in_window + 1 == self.config.window_updates
        candidates_count = 0
        born = torch.zeros_like(before)
        retired = torch.zeros_like(before)
        if self.config.arm == "dense":
            # Bookkeeping can observe ordinary exact-support changes; neither
            # W, the gradient, nor any live Adam field is written here.
            self.mask.copy_(weight.ne(0))
            born, retired = ~before & self.mask, before & ~self.mask
        else:
            self._shadow_observe(dormant_at_start, raw_gradient, clip_scale, lr, betas, eps)
            shrunk = weight.sign() * (weight.abs() - self.config.shrink_step).clamp_min(0)
            weight.copy_(torch.where(before, shrunk, torch.zeros_like(weight)))
            retired = before & weight.eq(0)
            self.mask.logical_and_(~retired)
            self._reset_shadow(retired)
            self._clear_live_moments(adam_state, dormant_at_start | retired)
            self._observe_counts()
            if window:
                threshold = self.config.shrink_step * self.proposal_count.to(self.proposal_sum.dtype)
                proposal = self.proposal_sum.sign() * (self.proposal_sum.abs() - threshold).clamp_min(0)
                candidates = dormant_at_start & ~self.mask & proposal.ne(0)
                candidates_count = int(candidates.sum())
                if self.config.arm == "learned":
                    born = candidates
                    weight[born] = proposal[born]
                elif self.config.arm == "random":
                    for row in range(weight.shape[0]):
                        values = proposal[row][candidates[row]]
                        if not values.numel():
                            continue
                        eligible = dormant_at_start[row].nonzero().flatten().cpu()
                        order = torch.randperm(eligible.numel(), generator=self.generator)[:values.numel()]
                        selected = eligible[order].to(weight.device)
                        born[row, selected] = True
                        weight[row, selected] = values
                self.mask.logical_or_(born)
                self._reset_shadow(born)
            if bool(weight[~self.mask].ne(0).any()) or not torch.equal(weight.ne(0), self.mask):
                raise AssertionError("dormant weights must remain exactly zero")
        self.survival_mask.logical_and_(~retired)
        self.ever_born.logical_or_(born)
        self.births_per_row.add_(born.sum(-1))
        self.deaths_per_row.add_(retired.sum(-1))
        births, deaths = int(born.sum()), int(retired.sum())
        self.cumulative_births += births
        self.cumulative_deaths += deaths
        self.updates += 1
        self.updates_in_window = 0 if window else self.updates_in_window + 1
        if window:
            self.proposal_sum.zero_()
            self.proposal_count.zero_()
        self._observe_counts()
        return {**self.summary(), "update": self.updates, "window": window,
                "births": births, "deaths": deaths,
                "births_per_row": born.sum(-1).cpu().tolist(),
                "deaths_per_row": retired.sum(-1).cpu().tolist(),
                "proposal_candidates": candidates_count,
                "eligible_dormant": int(dormant_at_start.sum()),
                "nonzero_hard_cuts": 0}

    def summary(self):
        return {"arm": self.config.arm, "shrink_step": self.config.shrink_step,
                "updates": self.updates, "updates_in_window": self.updates_in_window,
                "active": int(self.mask.sum()), "active_per_row": self.mask.sum(-1).cpu().tolist(),
                "cumulative_births": self.cumulative_births, "cumulative_deaths": self.cumulative_deaths,
                "cumulative_births_per_row": self.births_per_row.cpu().tolist(),
                "cumulative_deaths_per_row": self.deaths_per_row.cpu().tolist(),
                "initial_survivors": int(self.survival_mask.sum()),
                "ever_born": int(self.ever_born.sum()),
                "peak_active": self.peak_active, "min_active": self.min_active,
                "peak_per_row": self.peak_per_row.cpu().tolist(), "min_per_row": self.min_per_row.cpu().tolist(),
                "mask_sha256": tensor_hash(self.mask), "survival_sha256": tensor_hash(self.survival_mask),
                "initial_mask_sha256": tensor_hash(self.initial_mask),
                "shadow_lifetime_max": int(self.shadow_count.max()),
                "proposal_observations": int(self.proposal_count.sum())}

    def state_dict(self):
        buffers = (*self._FLOAT_BUFFERS, *self._COUNT_BUFFERS, *self._MASK_BUFFERS, *self._ROW_BUFFERS)
        return {"format": FORMAT_VERSION, "config": asdict(self.config),
                **{name: getattr(self, name).detach().cpu().clone() for name in buffers},
                "updates": self.updates, "updates_in_window": self.updates_in_window,
                "cumulative_births": self.cumulative_births, "cumulative_deaths": self.cumulative_deaths,
                "peak_active": self.peak_active, "min_active": self.min_active,
                "generator_state": self.generator.get_state().clone()}

    def semantic_state(self):
        return self.state_dict()

    @classmethod
    def from_state(cls, weight, state):
        if not isinstance(state, Mapping) or state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported rewiring operator state")
        try:
            expected_fields = {"format", "config", "updates", "updates_in_window", "cumulative_births",
                               "cumulative_deaths", "peak_active", "min_active", "generator_state",
                               *cls._FLOAT_BUFFERS, *cls._COUNT_BUFFERS, *cls._MASK_BUFFERS, *cls._ROW_BUFFERS}
            if set(state) != expected_fields:
                raise ValueError("incomplete or unsupported rewiring operator fields")
            if not isinstance(state["config"], Mapping) or set(state["config"]) != set(RewireConfig.__dataclass_fields__):
                raise ValueError("incomplete or unsupported rewiring configuration fields")
            config = RewireConfig(**state["config"])
            result = cls(weight, config)
            groups = ((cls._FLOAT_BUFFERS, weight.shape, weight.dtype),
                      (cls._COUNT_BUFFERS, weight.shape, torch.long),
                      (cls._MASK_BUFFERS, weight.shape, torch.bool),
                      (cls._ROW_BUFFERS, (weight.shape[0],), torch.long))
            for names, shape, dtype in groups:
                for name in names:
                    value = state[name]
                    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
                        raise ValueError(f"invalid operator buffer {name}")
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"nonfinite operator buffer {name}")
                    if dtype == torch.long and bool((value < 0).any()):
                        raise ValueError(f"negative operator count {name}")
                    setattr(result, name, value.detach().to(weight.device).clone())
            for name in ("updates", "updates_in_window", "cumulative_births", "cumulative_deaths", "peak_active", "min_active"):
                setattr(result, name, _integer(state[name], name))
            if result.updates_in_window != result.updates % config.window_updates:
                raise ValueError("invalid structural window position")
            if not torch.equal(result.mask, weight.detach().ne(0)):
                raise ValueError("saved mask and exact recurrent support differ")
            if bool((result.survival_mask & ~(result.initial_mask & result.mask)).any()):
                raise ValueError("invalid continuous initial-edge survival")
            if bool((result.shadow_v < 0).any()):
                raise ValueError("negative shadow Adam second moment")
            for name in (*cls._FLOAT_BUFFERS, *cls._COUNT_BUFFERS):
                if bool(getattr(result, name)[result.mask].ne(0).any()):
                    raise ValueError("active edge contains dormant proposal state")
            # A dormant edge is observed once on every update of its current
            # lifetime. Only the proposal accumulator resets at a window end.
            expected_observations = result.shadow_count.clamp_max(result.updates_in_window)
            if (not torch.equal(result.proposal_count, expected_observations)
                    or bool((result.shadow_count > result.updates).any())):
                raise ValueError("inconsistent dormant observation counts")
            if bool((result.proposal_sum.ne(0) & result.proposal_count.eq(0)).any()):
                raise ValueError("proposal without observations")
            if bool(((result.shadow_m.ne(0) | result.shadow_v.ne(0)) & result.shadow_count.eq(0)).any()):
                raise ValueError("shadow Adam moments without lifetime observations")
            current, initial = result.mask.sum(-1), result.initial_mask.sum(-1)
            if (int(result.births_per_row.sum()) != result.cumulative_births
                    or int(result.deaths_per_row.sum()) != result.cumulative_deaths
                    or not torch.equal(initial + result.births_per_row - result.deaths_per_row, current)):
                raise ValueError("inconsistent cumulative structural counts")
            if (bool((result.ever_born & result.survival_mask).any())
                    or bool((result.mask & ~result.survival_mask & ~result.ever_born).any())
                    or bool((result.ever_born.sum(-1) > result.births_per_row).any())
                    or bool(((result.initial_mask & ~result.survival_mask).sum(-1) > result.deaths_per_row).any())
                    or bool((result.births_per_row + result.deaths_per_row > weight.shape[1] * result.updates).any())):
                raise ValueError("inconsistent birth/death lifetime history")
            if config.arm == "retire" and (result.cumulative_births or bool(result.ever_born.any())):
                raise ValueError("retire arm cannot contain births")
            if config.arm == "dense" and any(bool(getattr(result, name).ne(0).any())
                                             for name in (*cls._FLOAT_BUFFERS, *cls._COUNT_BUFFERS)):
                raise ValueError("dense arm cannot contain dormant shadow observations")
            if (bool((result.peak_per_row < torch.maximum(current, initial)).any())
                    or bool((result.min_per_row > torch.minimum(current, initial)).any())
                    or bool((result.peak_per_row > weight.shape[1]).any())
                    or not 0 <= result.min_active <= min(int(current.sum()), int(initial.sum()))
                    or not max(int(current.sum()), int(initial.sum())) <= result.peak_active <= weight.numel()
                    or result.min_active < int(result.min_per_row.sum())
                    or result.peak_active > int(result.peak_per_row.sum())
                    or bool((result.peak_per_row > initial + result.births_per_row).any())
                    or bool((result.min_per_row < initial - result.deaths_per_row).any())
                    or result.peak_active > int(initial.sum()) + result.cumulative_births
                    or result.min_active < int(initial.sum()) - result.cumulative_deaths):
                raise ValueError("inconsistent peak/minimum capacity")
            generator_state = state["generator_state"]
            if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8:
                raise ValueError("invalid structural generator state")
            generator_state = generator_state.detach().cpu().clone()
            if config.arm != "random" and not torch.equal(generator_state, result.generator.get_state()):
                raise ValueError("unused structural generator has changed")
            result.generator.set_state(generator_state)
            return result
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("invalid rewiring operator state") from error
