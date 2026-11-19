"""Factorized structural controls attached to an inherited rewiring state.

Eligibility and exact numerical support are different objects. Dense controls
allow every coordinate; fixed controls allow the source checkpoint's support
even after an eligible weight reaches zero. Only retire/grow use the original
dynamic operator. Its dormant Adam lifetime and global Adam age are unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import torch

from experiments.rewiring.plasticity import (
    RewireConfig, RewireOperator, _integer, _number, tensor_hash,
)


MODES = ("dense", "dense_shrink", "fixed", "fixed_shrink", "retire", "grow")
FORMAT_VERSION = "factor-operator-v1"
_DYNAMIC = ("retire", "grow")
_SHRINK = ("dense_shrink", "fixed_shrink", "retire", "grow")


@dataclass(frozen=True)
class FactorConfig:
    mode: str = "dense"
    idle_lambda: float = 0.
    shrink_step: float = 2e-4

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown factor mode")
        _number(self.idle_lambda, "idle_lambda")
        _number(self.shrink_step, "shrink_step")

    @property
    def effective_shrink_step(self):
        return self.shrink_step if self.mode in _SHRINK else 0.


class FactorOperator:
    """Preserve source state at attachment; count actual support events in all modes.

    ``mask`` aliases eligibility, not exact support. ``support_mask`` records
    numerical nonzeros. The wrapped old operator owns absolute event history;
    ``factor_updates`` counts only updates since this attachment.
    """

    def __init__(self, weight, config: FactorConfig, *, inherited_operator=None):
        RewireOperator._validate_weight(weight)
        if not isinstance(config, FactorConfig):
            raise ValueError("config must be FactorConfig")
        if inherited_operator is None:
            inherited_operator = RewireOperator(weight, RewireConfig(
                arm="learned" if config.mode == "grow" else "retire",
                shrink_step=config.effective_shrink_step))
        if not isinstance(inherited_operator, RewireOperator):
            raise ValueError("inherited_operator must be RewireOperator")
        self.config = config
        self._source_state = inherited_operator.state_dict()
        self.rewire = RewireOperator.from_state(weight, self._source_state)
        self.rewire.config = self._runtime_config(self.rewire.config, config)
        self.factor_start_updates = self.rewire.updates
        self._allowed_mask = (torch.ones_like(self.rewire.mask) if config.mode.startswith("dense")
                              else self.rewire.mask.clone())
        self.factor_survival_mask = self.rewire.mask.clone()
        self.factor_ever_born = torch.zeros_like(self.rewire.mask)

    @staticmethod
    def _runtime_config(inherited, config):
        # Only retire/grow execute the old operator. Passive controls keep its
        # source arm as provenance while implementing their explicit eligibility.
        arm = {"retire": "retire", "grow": "learned"}.get(config.mode, inherited.arm)
        return replace(inherited, arm=arm, shrink_step=config.effective_shrink_step)

    @property
    def mask(self):
        return self.allowed_mask

    @property
    def allowed(self):
        return self.allowed_mask

    @property
    def allowed_mask(self):
        return self.rewire.mask if self.config.mode in _DYNAMIC else self._allowed_mask

    @property
    def support_mask(self):
        return self.rewire.mask

    @property
    def updates(self):
        return self.rewire.updates

    @property
    def updates_in_window(self):
        return self.rewire.updates_in_window

    @property
    def factor_start_update(self):
        return self.factor_start_updates

    @property
    def factor_updates(self):
        return self.updates - self.factor_start_updates

    def before_gradient(self, raw_gradient):
        if self.config.mode in _DYNAMIC:
            return self.rewire.before_gradient(raw_gradient)
        self.rewire._validate_gradient(raw_gradient)
        if self.config.mode.startswith("dense"):
            return raw_gradient
        return raw_gradient * self.allowed_mask

    @torch.no_grad()
    def apply_after_optimizer(self, weight, adam_state, raw_gradient, clip_scale, lr, betas, eps):
        before = self.support_mask.clone()
        if self.config.mode in _DYNAMIC:
            event = self.rewire.apply_after_optimizer(
                weight, adam_state, raw_gradient, clip_scale, lr, betas, eps)
        else:
            event = self._apply_passive(weight, adam_state, raw_gradient, clip_scale, lr, betas, eps)
        born, retired = ~before & self.support_mask, before & ~self.support_mask
        self.factor_survival_mask.logical_and_(~retired)
        self.factor_ever_born.logical_or_(born)
        return {**event, **self.summary()}

    def _apply_passive(self, weight, adam_state, raw_gradient, clip_scale, lr, betas, eps):
        op = self.rewire
        op._validate_runtime(weight, adam_state, raw_gradient, clip_scale, lr, betas, eps)
        before = op.mask.clone()
        never_allowed = ~self.allowed_mask
        shrink = self.config.effective_shrink_step
        if shrink:
            weight.copy_(weight.sign() * (weight.abs() - shrink).clamp_min(0))
        if bool(never_allowed.any()):
            weight.masked_fill_(never_allowed, 0)
            op._clear_live_moments(adam_state, never_allowed)
        # In particular, eligible exact zeros retain their live Adam moments.
        # Old dormant evidence stays frozen; no proposal is used in D/F modes.
        op.mask.copy_(weight.ne(0))
        born, retired = ~before & op.mask, before & ~op.mask
        op.survival_mask.logical_and_(~retired)
        op.ever_born.logical_or_(born)
        births_per_row, deaths_per_row = born.sum(-1), retired.sum(-1)
        op.births_per_row.add_(births_per_row)
        op.deaths_per_row.add_(deaths_per_row)
        births, deaths = int(born.sum()), int(retired.sum())
        op.cumulative_births += births
        op.cumulative_deaths += deaths
        op.updates += 1
        op.updates_in_window = op.updates % op.config.window_updates
        op._observe_counts()
        return {"update": op.updates, "window": op.updates_in_window == 0,
                "births": births, "deaths": deaths,
                "births_per_row": births_per_row.cpu().tolist(),
                "deaths_per_row": deaths_per_row.cpu().tolist(),
                "proposal_candidates": 0, "eligible_dormant": int((self.allowed_mask & ~before).sum()),
                "nonzero_hard_cuts": 0}

    def summary(self):
        summary = self.rewire.summary()
        summary.update(mode=self.config.mode, idle_lambda=self.config.idle_lambda,
            shrink_step=self.config.effective_shrink_step,
            factor_start_updates=self.factor_start_updates, factor_updates=self.factor_updates,
            allowed=int(self.allowed_mask.sum()), allowed_per_row=self.allowed_mask.sum(-1).cpu().tolist(),
            allowed_mask_sha256=tensor_hash(self.allowed_mask),
            support_sha256=tensor_hash(self.support_mask),
            factor_initial_active=int(self._source_state["mask"].sum()),
            factor_initial_survivors=int(self.factor_survival_mask.sum()),
            factor_ever_born=int(self.factor_ever_born.sum()),
            factor_cumulative_births=self.rewire.cumulative_births - self._source_state["cumulative_births"],
            factor_cumulative_deaths=self.rewire.cumulative_deaths - self._source_state["cumulative_deaths"])
        return summary

    def state_dict(self):
        return {"format": FORMAT_VERSION, "config": asdict(self.config),
                "factor_start_updates": self.factor_start_updates,
                "allowed_mask": self.allowed_mask.detach().cpu().clone(),
                "factor_survival_mask": self.factor_survival_mask.detach().cpu().clone(),
                "factor_ever_born": self.factor_ever_born.detach().cpu().clone(),
                "inherited_operator": deepcopy(self._source_state),
                "operator": self.rewire.state_dict()}

    def semantic_state(self):
        return self.state_dict()

    @classmethod
    def _restore_passive(cls, weight, state, source, config):
        if not isinstance(state, Mapping):
            raise ValueError("invalid factor support history")
        expected_config = cls._runtime_config(RewireConfig(**source["config"]), config)
        if state.get("config") != asdict(expected_config):
            raise ValueError("factor structural configuration mismatch")
        for name in (*RewireOperator._FLOAT_BUFFERS, *RewireOperator._COUNT_BUFFERS, "generator_state"):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), source[name].cpu()):
                raise ValueError("inactive inherited shadow/RNG state changed")
        # Reuse the old strict event-history validator. Inactive shadow fields
        # are zeroed only on this validation copy, since their preserved source
        # lifetimes intentionally stop advancing in passive factor controls.
        checked = deepcopy(state)
        checked["config"] = {**checked["config"], "arm": "random"}
        for name in (*RewireOperator._FLOAT_BUFFERS, *RewireOperator._COUNT_BUFFERS):
            checked[name] = torch.zeros_like(checked[name])
        result = RewireOperator.from_state(weight, checked)
        result.config = expected_config
        for name in (*RewireOperator._FLOAT_BUFFERS, *RewireOperator._COUNT_BUFFERS):
            setattr(result, name, state[name].detach().to(weight.device).clone())
        return result

    @classmethod
    def from_state(cls, weight, state):
        RewireOperator._validate_weight(weight)
        fields = {"format", "config", "factor_start_updates", "allowed_mask",
                  "factor_survival_mask", "factor_ever_born", "inherited_operator", "operator"}
        if not isinstance(state, Mapping) or state.get("format") != FORMAT_VERSION or set(state) != fields:
            raise ValueError("unsupported or incomplete factor operator state")
        try:
            if not isinstance(state["config"], Mapping) or set(state["config"]) != set(FactorConfig.__dataclass_fields__):
                raise ValueError("incomplete factor configuration")
            config = FactorConfig(**state["config"])
            source = state["inherited_operator"]
            if not isinstance(source, Mapping):
                raise ValueError("invalid inherited operator")
            source_mask = source["mask"]
            if not isinstance(source_mask, torch.Tensor) or source_mask.dtype != torch.bool or source_mask.shape != weight.shape:
                raise ValueError("invalid inherited support")
            # The old validator depends on exact support, not source magnitudes.
            source_weight = source_mask.to(device=weight.device, dtype=weight.dtype)
            inherited = RewireOperator.from_state(source_weight, source)
            result = cls.__new__(cls)
            result.config, result._source_state = config, inherited.state_dict()
            start = _integer(state["factor_start_updates"], "factor_start_updates")
            if start != inherited.updates:
                raise ValueError("factor start differs from inherited absolute updates")
            result.factor_start_updates = start
            if config.mode in _DYNAMIC:
                result.rewire = RewireOperator.from_state(weight, state["operator"])
                if result.rewire.config != cls._runtime_config(inherited.config, config):
                    raise ValueError("factor structural configuration mismatch")
            else:
                result.rewire = cls._restore_passive(weight, state["operator"], result._source_state, config)
            for name in ("allowed_mask", "factor_survival_mask", "factor_ever_born"):
                value = state[name]
                if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != weight.shape:
                    raise ValueError(f"invalid factor buffer {name}")
                setattr(result, "_allowed_mask" if name == "allowed_mask" else name,
                        value.detach().to(weight.device).clone())
            expected_allowed = (result.support_mask if config.mode in _DYNAMIC else
                torch.ones_like(source_weight, dtype=torch.bool) if config.mode.startswith("dense") else inherited.mask)
            if not torch.equal(result._allowed_mask, expected_allowed):
                raise ValueError("factor source eligibility mismatch")
            if result.factor_updates < 0 or bool((result.support_mask & ~result.allowed_mask).any()):
                raise ValueError("invalid factor update/support history")
            source_support = inherited.mask
            births_per_row = result.rewire.births_per_row - inherited.births_per_row
            deaths_per_row = result.rewire.deaths_per_row - inherited.deaths_per_row
            if (bool((births_per_row < 0).any()) or bool((deaths_per_row < 0).any())
                    or bool((births_per_row + deaths_per_row > weight.shape[1] * result.factor_updates).any())
                    or bool((result.factor_survival_mask & ~(source_support & result.support_mask)).any())
                    or bool((result.factor_ever_born & result.factor_survival_mask).any())
                    or bool((result.support_mask & ~result.factor_survival_mask & ~result.factor_ever_born).any())
                    or bool((result.factor_ever_born.sum(-1) > births_per_row).any())
                    or bool(((source_support & ~result.factor_survival_mask).sum(-1) > deaths_per_row).any())):
                raise ValueError("inconsistent factor lifetime history")
            if result.factor_updates == 0:
                current = result.rewire.state_dict()
                expected = deepcopy(result._source_state)
                expected["config"] = asdict(cls._runtime_config(inherited.config, config))
                for name in expected:
                    same = (torch.equal(current[name], expected[name]) if isinstance(expected[name], torch.Tensor)
                            else current[name] == expected[name])
                    if not same:
                        raise ValueError("factor attachment changed inherited state")
            return result
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("invalid factor operator state") from error
