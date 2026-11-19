"""Free-capacity connection trials on the next real learning block.

All coordinates receive ordinary Adam gradients and moments, including dormant
coordinates.  The operator restores dormant weights to zero after Adam without
clearing those moments.  Window proposals are tested only on the next block;
these are training-trajectory probes, not independent held-out evaluation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch

from experiments.rewiring.plasticity import RewireOperator, _integer, _number, tensor_hash


MODES = ("functional_fixed", "functional_task", "functional_magnitude", "functional_random")
FORMAT_VERSION = "organization-functional-operator-v1"


@dataclass(frozen=True)
class FunctionalConfig:
    mode: str = "functional_task"
    window_updates: int = 8
    candidates_per_kind: int = 4
    structural_seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown functional structural mode")
        _integer(self.window_updates, "window_updates", lower=1)
        _integer(self.candidates_per_kind, "candidates_per_kind", lower=1)
        _integer(self.structural_seed, "structural_seed")
        if self.structural_seed >= 2 ** 64:
            raise ValueError("structural_seed exceeds the torch generator range")


class FunctionalOperator:
    """Preserve the initial eligible graph; independently test births and fading.

    ``mask`` denotes eligibility, while ``support_mask`` denotes numerical
    nonzeros.  A runner injects ``loss_fn`` immediately before an update with a
    pending trial.  The callable reads the current weights and returns live,
    history and mixed scalar losses; it is always cleared at the boundary.
    """

    _BOOL = ("allowed_mask", "initial_mask", "survival_mask", "ever_born", "_support_mask")
    _ROWS = ("births_per_row", "deaths_per_row", "peak_per_row", "min_per_row")
    _TRAITS = ("last_birth_update", "last_retire_update", "activation_count")
    _SCALARS = ("updates", "cumulative_births", "cumulative_deaths", "probe_calls",
                "numeric_support_births", "numeric_support_deaths", "peak_active", "min_active",
                "peak_allowed", "min_allowed")

    def __init__(self, weight, config: FunctionalConfig):
        RewireOperator._validate_weight(weight)
        if not isinstance(config, FunctionalConfig):
            raise ValueError("config must be FunctionalConfig")
        self.config = config
        self._dtype = weight.dtype
        self.allowed_mask = weight.detach().ne(0).clone()
        self.initial_mask = self.allowed_mask.clone()
        self.survival_mask = self.allowed_mask.clone()
        self.ever_born = torch.zeros_like(self.allowed_mask)
        self._support_mask = self.allowed_mask.clone()
        self.window_gradient_sum = torch.zeros_like(weight, dtype=torch.float64)
        rows = self.allowed_mask.sum(-1)
        self.births_per_row = torch.zeros_like(rows)
        self.deaths_per_row = torch.zeros_like(rows)
        self.peak_per_row, self.min_per_row = rows.clone(), rows.clone()
        self.last_birth_update = torch.where(self.allowed_mask,
            torch.zeros_like(weight, dtype=torch.long), torch.full_like(weight, -1, dtype=torch.long))
        self.last_retire_update = torch.full_like(weight, -1, dtype=torch.long)
        self.activation_count = self.allowed_mask.long()
        self.updates = self.cumulative_births = self.cumulative_deaths = self.probe_calls = 0
        self.numeric_support_births = self.numeric_support_deaths = 0
        self.peak_active = self.min_active = int(rows.sum())
        self.peak_allowed = self.min_allowed = int(rows.sum())
        self.pending = None
        self.event_history = []
        self.loss_fn = None
        self.generator = torch.Generator(device="cpu").manual_seed(config.structural_seed)

    @property
    def mask(self):
        return self.allowed_mask

    @property
    def support_mask(self):
        return self._support_mask

    @property
    def updates_in_window(self):
        return self.updates % self.config.window_updates

    @property
    def factor_start_updates(self):
        return 0

    @property
    def factor_updates(self):
        return self.updates

    @property
    def rewire(self):
        return self

    def _validate_gradient(self, gradient):
        if (not isinstance(gradient, torch.Tensor) or gradient.shape != self.allowed_mask.shape
                or gradient.dtype != self._weight_dtype or gradient.device != self.allowed_mask.device):
            raise ValueError("functional gradient shape/dtype/device mismatch")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite functional gradient")

    @property
    def _weight_dtype(self):
        # Captured separately in state, because the gradient accumulator is f64.
        return self._dtype

    def before_gradient(self, raw_gradient):
        self._validate_gradient(raw_gradient)
        return raw_gradient

    def _runtime(self, weight, adam_state, raw_gradient, clip_scale, lr, betas, eps):
        RewireOperator._validate_weight(weight)
        self._validate_gradient(raw_gradient)
        if weight.shape != self.allowed_mask.shape or weight.device != self.allowed_mask.device or weight.dtype != self._dtype:
            raise ValueError("functional operator/weight mismatch")
        _number(clip_scale, "clip_scale", upper=1.)
        lr = _number(lr, "lr")
        eps = _number(eps, "eps", strict_lower=True)
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("Adam betas must contain two values")
        betas = tuple(_number(value, "Adam beta", upper=1.) for value in betas)
        if any(value == 1. for value in betas):
            raise ValueError("Adam betas must be less than one")
        if not isinstance(adam_state, Mapping) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(adam_state):
            raise ValueError("functional operator requires complete Adam state")
        step = torch.as_tensor(adam_state["step"])
        if step.numel() != 1 or not bool(torch.isfinite(step).all()) or float(step) != self.updates + 1:
            raise ValueError("functional Adam age must equal the completed next update")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name not in adam_state:
                continue
            value = adam_state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != weight.shape or value.dtype != weight.dtype
                    or value.device != weight.device or not bool(torch.isfinite(value).all())
                    or (name != "exp_avg" and bool((value < 0).any()))):
                raise ValueError(f"invalid functional Adam {name}")
        return lr, betas, eps, int(float(step))

    def _select(self, eligible, scores, *, largest):
        flat = eligible.flatten().nonzero().flatten()
        count = min(self.config.candidates_per_kind, flat.numel())
        if not count:
            return torch.empty((0, 2), dtype=torch.long)
        if self.config.mode == "functional_random":
            order = torch.randperm(flat.numel(), generator=self.generator)[:count].to(flat.device)
        else:
            order = torch.argsort(scores.flatten()[flat], descending=largest, stable=True)[:count]
        indices = flat[order]
        return torch.stack((indices // eligible.shape[1], indices % eligible.shape[1]), dim=-1).cpu()

    @staticmethod
    def _values(value, coordinates):
        coordinates = coordinates.to(value.device)
        return value[coordinates[:, 0], coordinates[:, 1]]

    @staticmethod
    def _put(value, coordinates, replacement):
        coordinates = coordinates.to(value.device)
        value[coordinates[:, 0], coordinates[:, 1]] = replacement

    def _generate_pending(self, weight, adam_state, lr, betas, eps, step):
        mean = self.window_gradient_sum / self.config.window_updates
        beta1, beta2 = betas
        m = adam_state["exp_avg"].double() / (1. - beta1 ** step)
        variance = adam_state.get("max_exp_avg_sq", adam_state["exp_avg_sq"])
        v = variance.double() / (1. - beta2 ** step)
        delta = (-lr * m / (v.sqrt() + eps)).to(weight.dtype)
        birth_score = -mean * delta.double()
        exit_score = (-weight.double() * mean if self.config.mode != "functional_magnitude"
                      else weight.double().abs())
        births = self._select(~self.allowed_mask & delta.ne(0), birth_score, largest=True)
        exits = self._select(self.allowed_mask, exit_score, largest=False)
        self.pending = {"generated_update": self.updates, "execute_update": self.updates + 1,
            "adam_step": step, "proposal_lr": lr,
            "birth_coordinates": births, "birth_delta": self._values(delta, births).detach().cpu().clone(),
            "birth_scores": self._values(birth_score, births).detach().cpu().clone(),
            "birth_weights_at_proposal": self._values(weight, births).detach().cpu().clone(),
            "exit_coordinates": exits, "exit_scores": self._values(exit_score, exits).detach().cpu().clone(),
            "exit_weights_at_proposal": self._values(weight, exits).detach().cpu().clone()}
        self.window_gradient_sum.zero_()
        return self._pending_record(self.pending)

    @staticmethod
    def _pending_record(pending):
        return {name: value.tolist() if isinstance(value, torch.Tensor) else value
                for name, value in pending.items()}

    def _loss(self):
        value = self.loss_fn()
        if not isinstance(value, Mapping) or set(value) != {"live", "history", "mixed"}:
            raise ValueError("functional loss_fn must return live/history/mixed")
        result = {}
        for name in ("live", "history", "mixed"):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError("functional probe loss must be a finite scalar")
            result[name] = float(item)
        if not math.isclose(result["mixed"], .5 * (result["live"] + result["history"]), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("functional mixed loss must average live and history")
        return result

    @staticmethod
    def _difference(candidate, baseline):
        return {name: candidate[name] - baseline[name] for name in baseline}

    def _probe_pending(self, weight, lr):
        pending = self.pending
        if pending["execute_update"] != self.updates + 1:
            raise ValueError("functional pending proposal is not for this update")
        if not callable(self.loss_fn):
            raise RuntimeError("pending functional trial requires a temporary loss_fn")
        birth, exit = pending["birth_coordinates"], pending["exit_coordinates"]
        original = weight.clone()
        born = torch.zeros_like(self.allowed_mask)
        retired = torch.zeros_like(self.allowed_mask)
        try:
            baseline = self._loss()
            birth_old = self._values(weight, birth).clone()
            birth_new = pending["birth_delta"].to(weight.device)
            self._put(weight, birth, birth_new)
            birth_loss = self._loss()
            birth_improves = birth_loss["mixed"] < baseline["mixed"]
            birth_accept = birth_improves and self.config.mode != "functional_fixed" and bool(len(birth))
            if birth_accept:
                self._put(born, birth, True)
                exit_baseline = birth_loss
            else:
                self._put(weight, birth, birth_old)
                exit_baseline = baseline
            exit_old = self._values(weight, exit).clone()
            self._put(weight, exit, 0.)
            ablation_loss = self._loss()
            self._put(weight, exit, exit_old)
            exit_new = torch.where(exit_old.abs() <= lr, torch.zeros_like(exit_old), .5 * exit_old)
            self._put(weight, exit, exit_new)
            half_loss = self._loss()
            exit_improves = half_loss["mixed"] < exit_baseline["mixed"]
            exit_accept = exit_improves and self.config.mode != "functional_fixed" and bool(len(exit))
            if exit_accept:
                self._put(retired, exit, exit_new.eq(0))
            else:
                self._put(weight, exit, exit_old)
            event = {**self._pending_record(pending), "executed_update": self.updates + 1,
                "probe_calls": 4, "half_exit_lr": lr, "baseline": baseline, "birth_loss": birth_loss,
                "birth_damage": self._difference(birth_loss, baseline),
                "birth_improves": birth_improves, "birth_accepted": birth_accept,
                "exit_baseline": exit_baseline, "full_ablation_loss": ablation_loss,
                "full_ablation_damage": self._difference(ablation_loss, exit_baseline),
                "half_exit_loss": half_loss, "half_exit_damage": self._difference(half_loss, exit_baseline),
                "exit_improves": exit_improves, "exit_accepted": exit_accept,
                "birth_old_weights": birth_old.cpu().tolist(), "birth_trial_weights": birth_new.cpu().tolist(),
                "birth_final_weights": self._values(weight, birth).cpu().tolist(),
                "exit_old_weights": exit_old.cpu().tolist(), "exit_trial_weights": exit_new.cpu().tolist(),
                "exit_final_weights": self._values(weight, exit).cpu().tolist(),
                "actual_birth_coordinates": born.nonzero().cpu().tolist(),
                "actual_retire_coordinates": retired.nonzero().cpu().tolist(),
                "births": int(born.sum()), "deaths": int(retired.sum())}
            return event, born, retired
        except Exception:
            weight.copy_(original)
            raise

    @torch.no_grad()
    def apply_after_optimizer(self, weight, adam_state, raw_gradient, clip_scale, lr, betas, eps):
        try:
            lr, betas, eps, step = self._runtime(weight, adam_state, raw_gradient, clip_scale, lr, betas, eps)
            before_support = self._support_mask.clone()
            # Adam has updated *all* coordinates and moments. Only forward
            # eligibility is enforced here; no moment or global-age writes.
            weight.masked_fill_(~self.allowed_mask, 0.)
            born = torch.zeros_like(self.allowed_mask)
            retired = torch.zeros_like(self.allowed_mask)
            probe = None
            if self.pending is not None:
                probe, born, retired = self._probe_pending(weight, lr)
                self.pending = None
                self.probe_calls += 4
                self.event_history.append(deepcopy(probe))
            self.allowed_mask.logical_or_(born).logical_and_(~retired)
            self.survival_mask.logical_and_(~retired)
            self.ever_born.logical_or_(born)
            self.last_birth_update.masked_fill_(born, self.updates + 1)
            self.last_retire_update.masked_fill_(retired, self.updates + 1)
            self.activation_count.add_(born.long())
            self.births_per_row.add_(born.sum(-1))
            self.deaths_per_row.add_(retired.sum(-1))
            self.cumulative_births += int(born.sum())
            self.cumulative_deaths += int(retired.sum())
            self._support_mask.copy_(weight.ne(0))
            self.numeric_support_births += int((~before_support & self._support_mask).sum())
            self.numeric_support_deaths += int((before_support & ~self._support_mask).sum())
            self.window_gradient_sum.add_(raw_gradient.double())
            if not bool(torch.isfinite(self.window_gradient_sum).all()):
                raise FloatingPointError("nonfinite functional window gradient accumulator")
            self.updates += 1
            generated = None
            if self.updates_in_window == 0:
                generated = self._generate_pending(weight, adam_state, lr, betas, eps, step)
            rows = self.support_mask.sum(-1)
            self.peak_per_row = torch.maximum(self.peak_per_row, rows)
            self.min_per_row = torch.minimum(self.min_per_row, rows)
            self.peak_active = max(self.peak_active, int(rows.sum()))
            self.min_active = min(self.min_active, int(rows.sum()))
            self.peak_allowed = max(self.peak_allowed, int(self.allowed_mask.sum()))
            self.min_allowed = min(self.min_allowed, int(self.allowed_mask.sum()))
            if bool(weight[~self.allowed_mask].ne(0).any()):
                raise AssertionError("dormant functional weights must remain zero")
            return {**self.summary(), "update": self.updates, "window": self.updates_in_window == 0,
                "births": int(born.sum()), "deaths": int(retired.sum()),
                "births_per_row": born.sum(-1).cpu().tolist(), "deaths_per_row": retired.sum(-1).cpu().tolist(),
                "functional_probe": probe, "pending_generated": generated,
                "nonzero_hard_cuts": 0}
        finally:
            self.loss_fn = None

    def summary(self):
        return {"arm": self.config.mode, "mode": self.config.mode,
            "updates": self.updates, "updates_in_window": self.updates_in_window,
            "factor_start_updates": 0, "factor_updates": self.updates,
            "active": int(self.support_mask.sum()), "allowed": int(self.allowed_mask.sum()),
            "active_per_row": self.support_mask.sum(-1).cpu().tolist(),
            "allowed_per_row": self.allowed_mask.sum(-1).cpu().tolist(),
            "initial_active": int(self.initial_mask.sum()), "initial_survivors": int(self.survival_mask.sum()),
            "initial_eligibility_survivors": int(self.survival_mask.sum()),
            "ever_born": int(self.ever_born.sum()), "peak_active": self.peak_active, "min_active": self.min_active,
            "peak_allowed": self.peak_allowed, "min_allowed": self.min_allowed,
            "peak_per_row": self.peak_per_row.cpu().tolist(), "min_per_row": self.min_per_row.cpu().tolist(),
            "cumulative_births": self.cumulative_births, "cumulative_deaths": self.cumulative_deaths,
            "cumulative_births_per_row": self.births_per_row.cpu().tolist(),
            "cumulative_deaths_per_row": self.deaths_per_row.cpu().tolist(),
            "numeric_support_births": self.numeric_support_births, "numeric_support_deaths": self.numeric_support_deaths,
            "probe_calls": self.probe_calls, "pending": self.pending is not None,
            "mask_sha256": tensor_hash(self.allowed_mask), "allowed_mask_sha256": tensor_hash(self.allowed_mask),
            "support_sha256": tensor_hash(self.support_mask), "initial_mask_sha256": tensor_hash(self.initial_mask),
            "survival_sha256": tensor_hash(self.survival_mask),
            "event_semantics": "birth/death count eligibility changes; numerical support crossings are separate",
            "optimizer_semantics": "full-coordinate global-age Adam; never reset moments at structural events",
            "probe_semantics": "next live block and potentially reused history; not independent held-out evaluation"}

    def state_dict(self):
        if self.loss_fn is not None:
            raise RuntimeError("cannot checkpoint while a temporary functional loss_fn is attached")
        return {"format": FORMAT_VERSION, "config": asdict(self.config), "weight_dtype": str(self._dtype),
            **{name: getattr(self, name).detach().cpu().clone() for name in (*self._BOOL, *self._ROWS)},
            "window_gradient_sum": self.window_gradient_sum.detach().cpu().clone(),
            "traits": {name: getattr(self, name).detach().cpu().clone() for name in self._TRAITS},
            **{name: getattr(self, name) for name in self._SCALARS},
            "pending": deepcopy(self.pending), "event_history": deepcopy(self.event_history),
            "generator_state": self.generator.get_state().clone()}

    def semantic_state(self):
        return self.state_dict()

    @classmethod
    def from_state(cls, weight, state):
        RewireOperator._validate_weight(weight)
        expected = {"format", "config", "weight_dtype", *cls._BOOL, *cls._ROWS, "window_gradient_sum", "traits",
                    *cls._SCALARS, "pending", "event_history", "generator_state"}
        if not isinstance(state, Mapping) or set(state) != expected or state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported or incomplete functional operator state")
        if not isinstance(state["config"], Mapping) or set(state["config"]) != set(FunctionalConfig.__dataclass_fields__):
            raise ValueError("incomplete functional configuration")
        result = cls(weight, FunctionalConfig(**state["config"]))
        if state["weight_dtype"] != str(weight.dtype):
            raise ValueError("functional checkpoint weight dtype mismatch")
        for names, shape, dtype in ((cls._BOOL, weight.shape, torch.bool), (cls._ROWS, (weight.shape[0],), torch.long)):
            for name in names:
                value = state[name]
                if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
                    raise ValueError(f"invalid functional {name}")
                setattr(result, name, value.detach().to(weight.device).clone())
        gradient = state["window_gradient_sum"]
        if (not isinstance(gradient, torch.Tensor) or gradient.shape != weight.shape or gradient.dtype != torch.float64
                or not bool(torch.isfinite(gradient).all())):
            raise ValueError("invalid functional gradient accumulator")
        result.window_gradient_sum = gradient.detach().to(weight.device).clone()
        if not isinstance(state["traits"], Mapping) or set(state["traits"]) != set(cls._TRAITS):
            raise ValueError("incomplete functional connection traits")
        for name in cls._TRAITS:
            value = state["traits"][name]
            if not isinstance(value, torch.Tensor) or value.shape != weight.shape or value.dtype != torch.long:
                raise ValueError(f"invalid functional connection trait {name}")
            setattr(result, name, value.detach().to(weight.device).clone())
        for name in cls._SCALARS:
            setattr(result, name, _integer(state[name], name))
        result.pending = deepcopy(state["pending"])
        result.event_history = deepcopy(state["event_history"])
        try:
            result.generator.set_state(state["generator_state"].cpu())
            result._validate_restored(weight)
        except (TypeError, RuntimeError, KeyError, AttributeError) as error:
            raise ValueError("invalid functional operator state") from error
        return result

    def _validate_restored(self, weight):
        if not torch.equal(self.support_mask, weight.ne(0)) or bool((self.support_mask & ~self.allowed_mask).any()):
            raise ValueError("functional eligibility/support mismatch")
        if bool((self.survival_mask & ~(self.initial_mask & self.allowed_mask)).any()):
            raise ValueError("invalid functional initial survival")
        if bool((self.allowed_mask & ~self.survival_mask & ~self.ever_born).any()):
            raise ValueError("functional eligibility lacks birth history")
        if (int(self.births_per_row.sum()) != self.cumulative_births or int(self.deaths_per_row.sum()) != self.cumulative_deaths
                or bool((self.births_per_row < 0).any()) or bool((self.deaths_per_row < 0).any())
                or not torch.equal(self.initial_mask.sum(-1) + self.births_per_row - self.deaths_per_row,
                                   self.allowed_mask.sum(-1))):
            raise ValueError("invalid functional event totals")
        if (bool((self.activation_count < self.initial_mask.long()).any())
                or int((self.activation_count - self.initial_mask.long()).sum()) != self.cumulative_births
                or bool((self.last_birth_update < -1).any()) or bool((self.last_birth_update > self.updates).any())
                or bool((self.last_retire_update < -1).any()) or bool((self.last_retire_update > self.updates).any())):
            raise ValueError("invalid functional connection lifetime traits")
        if self.updates_in_window == 0 and bool(self.window_gradient_sum.ne(0).any()):
            raise ValueError("functional complete window has residual gradients")
        expected_trials = (self.updates - 1) // self.config.window_updates if self.updates else 0
        if (not isinstance(self.event_history, list) or len(self.event_history) != expected_trials
                or self.probe_calls != 4 * expected_trials):
            raise ValueError("invalid functional probe history/budget")
        if sum(event["births"] for event in self.event_history) != self.cumulative_births or sum(
                event["deaths"] for event in self.event_history) != self.cumulative_deaths:
            raise ValueError("functional probe history and event totals disagree")
        self._validate_event_history()
        initial = int(self.initial_mask.sum())
        active, allowed = int(self.support_mask.sum()), int(self.allowed_mask.sum())
        if (not 0 <= self.min_active <= min(initial, active) <= max(initial, active) <= self.peak_active <= weight.numel()
                or not 0 <= self.min_allowed <= min(initial, allowed) <= max(initial, allowed) <= self.peak_allowed <= weight.numel()
                or bool((self.min_per_row < 0).any())
                or bool((self.min_per_row > self.support_mask.sum(-1)).any())
                or bool((self.min_per_row > self.initial_mask.sum(-1)).any())
                or bool((self.peak_per_row < self.support_mask.sum(-1)).any())
                or bool((self.peak_per_row < self.initial_mask.sum(-1)).any())
                or bool((self.peak_per_row > weight.shape[1]).any())):
            raise ValueError("invalid functional capacity extrema")
        needs_pending = self.updates > 0 and self.updates_in_window == 0
        if (self.pending is not None) != needs_pending:
            raise ValueError("functional pending proposal/window mismatch")
        if self.pending is not None:
            p = self.pending
            fields = {"generated_update", "execute_update", "adam_step", "proposal_lr", "birth_coordinates", "birth_delta",
                      "birth_scores", "birth_weights_at_proposal", "exit_coordinates", "exit_scores", "exit_weights_at_proposal"}
            if not isinstance(p, Mapping) or set(p) != fields or p["generated_update"] != self.updates or p["execute_update"] != self.updates + 1 or p["adam_step"] != self.updates:
                raise ValueError("invalid functional pending clock/fields")
            _number(p["proposal_lr"], "proposal_lr")
            for kind in ("birth", "exit"):
                coordinates = p[f"{kind}_coordinates"]
                if (not isinstance(coordinates, torch.Tensor) or coordinates.dtype != torch.long or coordinates.ndim != 2
                        or coordinates.shape[1] != 2 or len(coordinates) > self.config.candidates_per_kind
                        or bool((coordinates < 0).any()) or bool((coordinates >= weight.shape[0]).any())
                        or len(torch.unique(coordinates, dim=0)) != len(coordinates)):
                    raise ValueError("invalid functional pending coordinates")
                selected = self._values(self.allowed_mask, coordinates)
                if bool((selected if kind == "birth" else ~selected).any()):
                    raise ValueError("functional pending eligibility mismatch")
                for suffix in ("scores", "weights_at_proposal", *( ("delta",) if kind == "birth" else ())):
                    value = p[f"{kind}_{suffix}"]
                    dtype = torch.float64 if suffix == "scores" else weight.dtype
                    if not isinstance(value, torch.Tensor) or value.shape != (len(coordinates),) or value.dtype != dtype or not bool(torch.isfinite(value).all()):
                        raise ValueError("invalid functional pending value")

    def _validate_event_history(self):
        allowed = self.initial_mask.clone()
        survived = self.initial_mask.clone()
        ever = torch.zeros_like(allowed)
        last_birth = torch.where(allowed, torch.zeros_like(self.last_birth_update),
                                 torch.full_like(self.last_birth_update, -1))
        last_retire = torch.full_like(self.last_retire_update, -1)
        activations = allowed.long()
        births_per_row = torch.zeros_like(self.births_per_row)
        deaths_per_row = torch.zeros_like(self.deaths_per_row)
        for index, event in enumerate(self.event_history, start=1):
            expected_update = index * self.config.window_updates + 1
            if (not isinstance(event, Mapping) or event.get("generated_update") != expected_update - 1
                    or event.get("execute_update") != expected_update or event.get("executed_update") != expected_update
                    or event.get("probe_calls") != 4):
                raise ValueError("invalid functional probe event clock")
            for field in ("baseline", "birth_loss", "exit_baseline", "full_ablation_loss", "half_exit_loss"):
                losses = event.get(field)
                if (not isinstance(losses, Mapping) or set(losses) != {"live", "history", "mixed"}
                        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                               for value in losses.values())
                        or not math.isclose(losses["mixed"], .5 * (losses["live"] + losses["history"]), rel_tol=1e-6, abs_tol=1e-8)):
                    raise ValueError("invalid saved functional probe loss")
            for prefix, candidate, baseline in (
                    ("birth", "birth_loss", "baseline"),
                    ("full_ablation", "full_ablation_loss", "exit_baseline"),
                    ("half_exit", "half_exit_loss", "exit_baseline")):
                if event.get(f"{prefix}_damage") != self._difference(event[candidate], event[baseline]):
                    raise ValueError("invalid saved functional probe loss difference")
            birth_improves = event["birth_loss"]["mixed"] < event["baseline"]["mixed"]
            exit_improves = event["half_exit_loss"]["mixed"] < event["exit_baseline"]["mixed"]
            if event.get("birth_improves") != birth_improves or event.get("exit_improves") != exit_improves:
                raise ValueError("invalid saved functional probe comparison")
            born = torch.zeros_like(allowed)
            retired = torch.zeros_like(allowed)
            for kind, mask, count in (("birth", born, "births"), ("retire", retired, "deaths")):
                coordinates = event.get(f"actual_{kind}_coordinates")
                if not isinstance(coordinates, list):
                    raise ValueError("invalid saved functional event coordinates")
                if coordinates:
                    tensor = torch.tensor(coordinates, dtype=torch.long, device=mask.device)
                    if (tensor.ndim != 2 or tensor.shape[1] != 2 or len(tensor) > self.config.candidates_per_kind
                            or bool((tensor < 0).any()) or bool((tensor >= mask.shape[0]).any())
                            or len(tensor.unique(dim=0)) != len(tensor)):
                        raise ValueError("invalid saved functional event coordinates")
                    self._put(mask, tensor, True)
                if event.get(count) != int(mask.sum()):
                    raise ValueError("functional event coordinate/count mismatch")
            if bool((born & allowed).any()) or bool((retired & ~allowed).any()) or bool((born & retired).any()):
                raise ValueError("functional event eligibility mismatch")
            if (self.config.mode == "functional_fixed" and (bool(born.any()) or bool(retired.any())
                    or event.get("birth_accepted") or event.get("exit_accepted"))):
                raise ValueError("fixed functional strategy changed structure")
            if (event.get("birth_accepted") and not birth_improves) or (event.get("exit_accepted") and not exit_improves):
                raise ValueError("functional event accepted a non-improving trial")
            if bool(born.any()) and not event.get("birth_accepted") or bool(retired.any()) and not event.get("exit_accepted"):
                raise ValueError("functional events lack an accepted trial")
            expected_base = event["birth_loss"] if event.get("birth_accepted") else event["baseline"]
            if event["exit_baseline"] != expected_base:
                raise ValueError("functional exit trial used the wrong baseline")
            allowed.logical_or_(born).logical_and_(~retired)
            survived.logical_and_(~retired)
            ever.logical_or_(born)
            last_birth.masked_fill_(born, expected_update)
            last_retire.masked_fill_(retired, expected_update)
            activations.add_(born.long())
            births_per_row.add_(born.sum(-1))
            deaths_per_row.add_(retired.sum(-1))
        for name, expected in (("allowed_mask", allowed), ("survival_mask", survived), ("ever_born", ever),
                               ("last_birth_update", last_birth), ("last_retire_update", last_retire),
                               ("activation_count", activations), ("births_per_row", births_per_row),
                               ("deaths_per_row", deaths_per_row)):
            if not torch.equal(getattr(self, name), expected):
                raise ValueError(f"functional saved {name} differs from coordinate event history")

