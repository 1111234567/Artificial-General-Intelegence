"""Atomic pair growth and gradual retirement on stored-trajectory probes.

All potential coordinates retain their ordinary Adam gradient/moment history.
Only physical values at real forward boundaries are projected to eligibility;
neither optimizer moments nor temporal-credit traces are edited here.  Four
probe policies use the same ranking and four-call budget.  Their acceptance
flags independently enable births and finite fading, with no edge-count cap.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math

import torch


FORMAT_VERSION = "undirected-pair-operator-v1"
MODES = ("fixed", "fixed_probe", "grow", "retire", "both")


def _integer(value, name, lower=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return value


def _number(value, name, *, positive=False, upper=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0) or (upper is not None and value > upper)):
        raise ValueError(f"invalid {name}")
    return float(value)


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    return deepcopy(value)


def _digest(value):
    digest = hashlib.sha256()

    def put(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"dict:")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                put(key)
                put(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                put(child)
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    put(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class PairConfig:
    mode: str = "fixed"
    window_updates: int = 8
    candidates: int = 4
    seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown pair structural policy")
        _integer(self.window_updates, "window_updates", 1)
        _integer(self.candidates, "candidates", 1)
        _integer(self.seed, "seed")


class PairOperator:
    """One eligibility per pair; task proposals do not imply safe deletion."""

    _BOOL = ("mask", "initial_mask", "initial_survival", "ever_born", "support_mask", "initial_support")
    _TRAITS = ("last_birth", "last_retire", "activation_counts")
    _COUNTERS = ("updates", "cumulative_births", "cumulative_deaths", "probe_calls",
                 "numeric_support_births", "numeric_support_deaths", "window_count")

    def __init__(self, model, initial_mask, config: PairConfig):
        if not isinstance(config, PairConfig):
            raise ValueError("PairOperator requires PairConfig")
        if model.mode != "undirected" and config.mode != "fixed":
            raise ValueError("pair probes/adaptation require shared undirected weights; untied supports fixed only")
        if (not isinstance(initial_mask, torch.Tensor) or initial_mask.dtype != torch.bool
                or initial_mask.shape != model.edge_shared.shape):
            raise ValueError("initial_mask must contain one boolean per unordered pair")
        self.model, self.config = model, deepcopy(config)
        self.model_layout = {"config": model.config(), "parameters": [
            (name, tuple(value.shape), str(value.dtype)) for name, value in model.named_parameters()]}
        self.mask = _copy(initial_mask)
        self.initial_mask = self.mask.clone()
        self.initial_survival = self.mask.clone()
        self.ever_born = torch.zeros_like(self.mask)
        model.validate_eligibility(self.mask)
        self.support_mask = self._support()
        self.initial_support = self.support_mask.clone()
        self.last_birth = torch.where(self.mask, torch.zeros_like(self.mask, dtype=torch.long),
                                       torch.full_like(self.mask, -1, dtype=torch.long))
        self.last_retire = torch.full_like(self.last_birth, -1)
        self.activation_counts = self.mask.long()
        self.window_gradient_sum = torch.zeros(len(self.mask), dtype=torch.float64)
        for name in self._COUNTERS:
            setattr(self, name, 0)
        self.pending = None
        self.event_history, self.update_history = [], []
        self._probe = None

    @property
    def allowed(self):
        return self.mask

    @property
    def allowed_mask(self):
        return self.mask

    def _support(self):
        support = self.model.edge_shared.detach().cpu().ne(0)
        if self.model.mode == "untied":
            support |= self.model.edge_skew.detach().cpu().ne(0)
        return support

    def project_and_validate(self):
        self.model.project_inactive_(self.mask)
        self.model.validate_eligibility(self.mask)

    def _runtime(self, optimizer, raw_gradients, clip_scale):
        _number(clip_scale, "clip_scale", upper=1.)
        if not isinstance(raw_gradients, Mapping) or "edge_shared" not in raw_gradients:
            raise ValueError("pair learning requires raw edge_shared gradients")
        raw = raw_gradients["edge_shared"]
        parameter = self.model.edge_shared
        if (not isinstance(raw, torch.Tensor) or raw.shape != parameter.shape or raw.dtype != parameter.dtype
                or not bool(torch.isfinite(raw).all())):
            raise ValueError("invalid pair gradient dtype/layout/value")
        groups = [group for group in optimizer.param_groups if any(item is parameter for item in group["params"])]
        if len(groups) != 1:
            raise ValueError("shared pair parameter must occur in exactly one optimizer group")
        group = groups[0]
        lr, eps = _number(group["lr"], "Adam lr", positive=True), _number(group["eps"], "Adam eps", positive=True)
        betas = tuple(_number(value, "Adam beta", upper=1.) for value in group["betas"])
        if len(betas) != 2 or any(value >= 1 for value in betas) or group.get("weight_decay", 0.) != 0.:
            raise ValueError("pair proposals require two valid Adam betas and no weight decay")
        state = optimizer.state.get(parameter)
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not isinstance(state, Mapping) or not required.issubset(state):
            raise ValueError("complete ordinary Adam history is required")
        step = state["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != self.updates + 1:
            raise ValueError("Adam/operator update ages disagree")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state:
                value = state[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype
                        or not bool(torch.isfinite(value).all()) or (name != "exp_avg" and bool(value.lt(0).any()))):
                    raise ValueError("invalid Adam pair moments")
        return _copy(raw).double(), lr, eps, betas, state

    def _select(self, mask, scores, *, largest):
        candidates = mask.nonzero().flatten()
        order = torch.argsort(scores[candidates], descending=largest, stable=True)
        return candidates[order[:self.config.candidates]].clone()

    def _generate(self, state, lr, eps, betas, clip_scale):
        mean = self.window_gradient_sum / self.window_count
        first = _copy(state["exp_avg"]).double() / (1. - betas[0] ** self.updates)
        variance = _copy(state.get("max_exp_avg_sq", state["exp_avg_sq"])).double() / (1. - betas[1] ** self.updates)
        delta = (-lr * first / (variance.sqrt() + eps)).to(self.model.edge_shared.dtype)
        weight = _copy(self.model.edge_shared)
        self.pending = {"generated_update": self.updates, "execute_update": self.updates + 1,
            "adam_step": self.updates, "proposal_lr": lr, "clip_scale_at_proposal": float(clip_scale),
            "window_count": self.window_count, "gmean": mean.clone(), "weight_at_proposal": weight,
            "delta_all": delta, "mask_at_proposal": self.mask.clone(),
            "birth_indices": self._select(~self.mask, -mean * delta.double(), largest=True),
            "fade_indices": self._select(self.mask, -weight.double() * mean, largest=False)}
        self.window_gradient_sum.zero_()
        self.window_count = 0
        return self._proposal_summary(self.pending)

    @staticmethod
    def _proposal_summary(pending):
        return {name: value.tolist() if isinstance(value, torch.Tensor) else value
                for name, value in pending.items() if name not in {"gmean", "weight_at_proposal", "delta_all", "mask_at_proposal"}}

    def _loss(self):
        with torch.no_grad():
            value = self._probe()
        return self._validated_loss(value)

    @staticmethod
    def _validated_loss(value):
        if not isinstance(value, Mapping) or set(value) != {"live", "history", "mixed"}:
            raise ValueError("pair probe must return live/history/mixed")
        result = {}
        for name in ("live", "history", "mixed"):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError("pair probe losses must be finite scalars")
            result[name] = float(item)
        if not math.isclose(result["mixed"], .5 * (result["live"] + result["history"]), rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("mixed pair loss must average live and history")
        return result

    @staticmethod
    def _damage(candidate, baseline):
        return {name: candidate[name] - baseline[name] for name in baseline}

    @torch.no_grad()
    def _execute(self, lr):
        pending = self.pending
        if pending["execute_update"] != self.updates + 1 or not torch.equal(pending["mask_at_proposal"], self.mask):
            raise ValueError("pending pair proposal no longer belongs to this graph boundary")
        if not callable(self._probe):
            raise ValueError("pending pair trials require a temporary probe closure")
        weight = self.model.edge_shared
        original = weight.detach().clone()
        birth_cpu, fade_cpu = pending["birth_indices"], pending["fade_indices"]
        birth, fade = birth_cpu.to(weight.device), fade_cpu.to(weight.device)
        born, retired = torch.zeros_like(self.mask), torch.zeros_like(self.mask)
        try:
            baseline = self._loss()
            birth_old = weight[birth].clone()
            birth_new = pending["delta_all"][birth_cpu].to(weight)
            weight[birth] = birth_new
            birth_loss = self._loss()
            birth_improves = birth_loss["mixed"] < baseline["mixed"]
            accept_birth = (self.config.mode in ("grow", "both") and birth_improves
                            and bool(birth_new.ne(0).any()))
            if accept_birth:
                born[birth_cpu] = birth_new.cpu().ne(0)
                fade_baseline = birth_loss
            else:
                weight[birth] = birth_old
                fade_baseline = baseline
            fade_old = weight[fade].clone()
            weight[fade] = 0.
            full_loss = self._loss()
            weight[fade] = fade_old
            fade_new = torch.where(fade_old.abs() <= lr, torch.zeros_like(fade_old), .5 * fade_old)
            weight[fade] = fade_new
            half_loss = self._loss()
            fade_improves = half_loss["mixed"] < fade_baseline["mixed"]
            accept_fade = self.config.mode in ("retire", "both") and fade_improves and bool(len(fade_cpu))
            if accept_fade:
                retired[fade_cpu] = fade_new.cpu().eq(0)
            else:
                weight[fade] = fade_old
            event = {**self._proposal_summary(pending), "executed_update": self.updates + 1,
                "probe_calls": 4, "fade_lr": lr, "baseline": baseline, "birth_loss": birth_loss,
                "birth_damage": self._damage(birth_loss, baseline), "birth_improves": birth_improves,
                "birth_accepted": accept_birth, "fade_baseline": fade_baseline,
                "full_ablation_loss": full_loss, "full_ablation_damage": self._damage(full_loss, fade_baseline),
                "half_fade_loss": half_loss, "half_fade_damage": self._damage(half_loss, fade_baseline),
                "fade_improves": fade_improves, "fade_accepted": accept_fade,
                "birth_old_weights": birth_old.cpu().tolist(), "birth_trial_weights": birth_new.cpu().tolist(),
                "birth_final_weights": weight[birth].cpu().tolist(), "fade_old_weights": fade_old.cpu().tolist(),
                "fade_trial_weights": fade_new.cpu().tolist(), "fade_final_weights": weight[fade].cpu().tolist(),
                "actual_birth_indices": born.nonzero().flatten().tolist(),
                "actual_retire_indices": retired.nonzero().flatten().tolist(),
                "births": int(born.sum()), "deaths": int(retired.sum())}
            return event, born, retired
        except BaseException:
            weight.copy_(original)
            raise

    @torch.no_grad()
    def after_optimizer(self, optimizer, raw_gradients, clip_scale, probe=None):
        self._probe = probe
        try:
            raw, lr, eps, betas, adam = self._runtime(optimizer, raw_gradients, clip_scale)
            new_sum = self.window_gradient_sum + raw if self.config.mode != "fixed" else self.window_gradient_sum
            if not bool(torch.isfinite(new_sum).all()):
                raise ValueError("nonfinite pair gradient-window accumulator")
            self.project_and_validate()
            post_adam_support = self._support()
            born, retired = torch.zeros_like(self.mask), torch.zeros_like(self.mask)
            event = None
            if self.pending is not None:
                event, born, retired = self._execute(lr)
                self.event_history.append({"proposal": _copy(self.pending), "event": deepcopy(event)})
                self.pending = None
                self.probe_calls += 4
            self.mask.logical_or_(born).logical_and_(~retired)
            self.initial_survival.logical_and_(~retired)
            self.ever_born.logical_or_(born)
            self.last_birth[born] = self.updates + 1
            self.last_retire[retired] = self.updates + 1
            self.activation_counts.add_(born.long())
            self.cumulative_births += int(born.sum())
            self.cumulative_deaths += int(retired.sum())
            support = self._support()
            numeric_birth, numeric_death = (~self.support_mask & support), (self.support_mask & ~support)
            self.numeric_support_births += int(numeric_birth.sum())
            self.numeric_support_deaths += int(numeric_death.sum())
            self.support_mask = support
            self.updates += 1
            self.update_history.append({"update": self.updates,
                "birth_indices": born.nonzero().flatten().tolist(), "retire_indices": retired.nonzero().flatten().tolist(),
                "numeric_birth_indices": numeric_birth.nonzero().flatten().tolist(),
                "numeric_death_indices": numeric_death.nonzero().flatten().tolist()})
            generated = None
            if self.config.mode != "fixed":
                self.window_gradient_sum = new_sum
                self.window_count += 1
                if self.window_count == self.config.window_updates:
                    generated = self._generate(adam, lr, eps, betas, clip_scale)
            self.project_and_validate()
            return {**self.summary(), "update": self.updates, "births": int(born.sum()), "deaths": int(retired.sum()),
                "probe": event, "pending_generated": generated,
                "structural_numeric_births": int((~post_adam_support & self.support_mask).sum()),
                "structural_numeric_deaths": int((post_adam_support & ~self.support_mask).sum())}
        finally:
            self._probe = None
            self.project_and_validate()

    def summary(self, model=None):
        return {"structure": self.config.mode, "mode": self.config.mode, "updates": self.updates,
            "eligible_pairs": int(self.mask.sum()), "nonzero_pairs": int(self.support_mask.sum()),
            "potential_pairs": len(self.mask), "initial_surviving_pairs": int(self.initial_survival.sum()),
            "ever_born_pairs": int(self.ever_born.sum()), "cumulative_births": self.cumulative_births,
            "cumulative_deaths": self.cumulative_deaths, "numeric_support_births": self.numeric_support_births,
            "numeric_support_deaths": self.numeric_support_deaths, "probe_calls": self.probe_calls,
            "updates_in_window": self.window_count, "pending": self.pending is not None,
            "tie_break": "stable canonical pair index; no random draws",
            "optimizer_rule": "all potential gradients/moments retained; inactive physical values project to zero; no credit reset"}

    def state_dict(self):
        if self._probe is not None:
            raise ValueError("cannot checkpoint inside a structural probe")
        result = {"format": FORMAT_VERSION, "config": asdict(self.config), "model_layout": _copy(self.model_layout),
            **{name: _copy(getattr(self, name)) for name in (*self._BOOL, *self._TRAITS, *self._COUNTERS)},
            "window_gradient_sum": self.window_gradient_sum.clone(), "pending": _copy(self.pending),
            "event_history": _copy(self.event_history), "update_history": deepcopy(self.update_history)}
        result["state_hash"] = _digest(result)
        return result

    def _validate_proposal(self, proposal):
        names = {"generated_update", "execute_update", "adam_step", "proposal_lr", "clip_scale_at_proposal",
                 "window_count", "gmean", "weight_at_proposal", "delta_all", "mask_at_proposal", "birth_indices", "fade_indices"}
        if not isinstance(proposal, Mapping) or set(proposal) != names:
            raise ValueError("incomplete pair proposal")
        age = _integer(proposal["generated_update"], "proposal age", 1)
        if (age % self.config.window_updates or proposal["execute_update"] != age + 1 or proposal["adam_step"] != age
                or proposal["window_count"] != self.config.window_updates):
            raise ValueError("invalid pair proposal age/window")
        _number(proposal["proposal_lr"], "proposal lr", positive=True)
        _number(proposal["clip_scale_at_proposal"], "proposal clip", upper=1.)
        for name, dtype in (("gmean", torch.float64), ("weight_at_proposal", self.model.edge_shared.dtype),
                            ("delta_all", self.model.edge_shared.dtype), ("mask_at_proposal", torch.bool)):
            value = proposal[name]
            if (not isinstance(value, torch.Tensor) or value.shape != self.mask.shape or value.dtype != dtype
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError("invalid pair proposal ranking tensor")
        if bool(proposal["weight_at_proposal"][~proposal["mask_at_proposal"]].ne(0).any()):
            raise ValueError("proposal has inactive nonzero weights")
        birth = self._select(~proposal["mask_at_proposal"], -proposal["gmean"] * proposal["delta_all"].double(), largest=True)
        fade = self._select(proposal["mask_at_proposal"], -proposal["weight_at_proposal"].double() * proposal["gmean"], largest=False)
        for name, expected in (("birth_indices", birth), ("fade_indices", fade)):
            if not isinstance(proposal[name], torch.Tensor) or proposal[name].dtype != torch.long or not torch.equal(proposal[name], expected):
                raise ValueError("pair proposal does not reproduce its full-vector ranking")

    @classmethod
    def from_state(cls, model, state):
        if not isinstance(state, Mapping) or "config" not in state or not isinstance(state["config"], Mapping):
            raise ValueError("invalid pair operator state")
        if set(state["config"]) != set(PairConfig.__dataclass_fields__):
            raise ValueError("incomplete pair operator configuration")
        config = PairConfig(**state["config"])
        result = cls(model, state.get("mask"), config)
        expected = set(result.state_dict())
        if (set(state) != expected or state["format"] != FORMAT_VERSION
                or state["state_hash"] != _digest({key: value for key, value in state.items() if key != "state_hash"})
                or state["model_layout"] != result.model_layout):
            raise ValueError("pair operator schema/hash/model layout mismatch")
        for name in cls._BOOL:
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != result.mask.shape or value.dtype != torch.bool:
                raise ValueError("invalid pair mask layout")
        for name in cls._TRAITS:
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != result.mask.shape or value.dtype != torch.long:
                raise ValueError("invalid pair trait layout")
        for name in cls._COUNTERS:
            _integer(state[name], name)
        gradient = state["window_gradient_sum"]
        if not isinstance(gradient, torch.Tensor) or gradient.dtype != torch.float64 or gradient.shape != result.mask.shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError("invalid pair window gradients")
        for name in (*cls._BOOL, *cls._TRAITS, *cls._COUNTERS, "window_gradient_sum", "pending", "event_history", "update_history"):
            setattr(result, name, _copy(state[name]))
        result._validate_restored()
        return result

    def _validate_restored(self):
        if not isinstance(self.update_history, list) or len(self.update_history) != self.updates:
            raise ValueError("pair update ledger does not cover its age")
        mask, support = self.initial_mask.clone(), self.initial_support.clone()
        survival, ever = mask.clone(), torch.zeros_like(mask)
        last_birth = torch.where(mask, torch.zeros_like(self.last_birth), torch.full_like(self.last_birth, -1))
        last_retire = torch.full_like(self.last_retire, -1)
        activations = mask.long()
        births = deaths = numeric_births = numeric_deaths = 0
        window_masks, window_support = {}, {}

        def indices(value):
            if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < len(mask) for item in value) or len(set(value)) != len(value):
                raise ValueError("invalid pair event indices")
            return torch.tensor(value, dtype=torch.long)

        for age, item in enumerate(self.update_history, 1):
            if not isinstance(item, Mapping) or set(item) != {"update", "birth_indices", "retire_indices", "numeric_birth_indices", "numeric_death_indices"} or item["update"] != age:
                raise ValueError("invalid pair update ledger")
            born, retired = indices(item["birth_indices"]), indices(item["retire_indices"])
            nb, nd = indices(item["numeric_birth_indices"]), indices(item["numeric_death_indices"])
            if (bool(mask[born].any()) or not bool(mask[retired].all()) or bool(support[nb].any())
                    or not bool(support[nd].all()) or set(born.tolist()) & set(retired.tolist()) or set(nb.tolist()) & set(nd.tolist())):
                raise ValueError("impossible pair lifetime/support transition")
            mask[born], mask[retired] = True, False
            support[nb], support[nd] = True, False
            if bool((support & ~mask).any()):
                raise ValueError("numerical support escaped pair eligibility")
            survival[retired], ever[born] = False, True
            last_birth[born], last_retire[retired] = age, age
            activations[born] += 1
            births += len(born)
            deaths += len(retired)
            numeric_births += len(nb)
            numeric_deaths += len(nd)
            if self.config.mode != "fixed" and age % self.config.window_updates == 0:
                window_masks[age], window_support[age] = mask.clone(), support.clone()
        tensors = {"mask": mask, "support_mask": support, "initial_survival": survival, "ever_born": ever,
                   "last_birth": last_birth, "last_retire": last_retire, "activation_counts": activations}
        if any(not torch.equal(getattr(self, name), expected) for name, expected in tensors.items()):
            raise ValueError("pair lifetime traits do not rebuild from events")
        if (births, deaths, numeric_births, numeric_deaths) != (self.cumulative_births, self.cumulative_deaths, self.numeric_support_births, self.numeric_support_deaths):
            raise ValueError("pair turnover counters do not rebuild from events")
        if bool((self.initial_support & ~self.initial_mask).any()) or not torch.equal(self.support_mask, self._support()):
            raise ValueError("pair support does not match its physical model")
        expected_events = 0 if self.config.mode == "fixed" else max(0, (self.updates - 1) // self.config.window_updates)
        if not isinstance(self.event_history, list) or len(self.event_history) != expected_events or self.probe_calls != 4 * expected_events:
            raise ValueError("pair probe budget/history mismatch")
        for index, record in enumerate(self.event_history, 1):
            if not isinstance(record, Mapping) or set(record) != {"proposal", "event"}:
                raise ValueError("incomplete executed pair proposal")
            self._validate_proposal(record["proposal"])
            expected_age = index * self.config.window_updates + 1
            event = record["event"]
            if record["proposal"]["execute_update"] != expected_age or not isinstance(event, Mapping) or event.get("executed_update") != expected_age or event.get("probe_calls") != 4:
                raise ValueError("invalid executed pair proposal clock")
            ledger = self.update_history[expected_age - 1]
            if event.get("actual_birth_indices") != ledger["birth_indices"] or event.get("actual_retire_indices") != ledger["retire_indices"]:
                raise ValueError("pair event and lifetime ledger disagree")
            proposal = record["proposal"]
            generated = proposal["generated_update"]
            if (not torch.equal(proposal["mask_at_proposal"], window_masks[generated])
                    or not torch.equal(proposal["weight_at_proposal"].ne(0), window_support[generated])):
                raise ValueError("pair proposal graph/support does not match its actual window boundary")
            self._validate_event(proposal, event)
        expected_count = 0 if self.config.mode == "fixed" else self.updates % self.config.window_updates
        if self.window_count != expected_count or (self.window_count == 0 and bool(self.window_gradient_sum.ne(0).any())):
            raise ValueError("pair partial-window age mismatch")
        needs_pending = self.config.mode != "fixed" and self.updates > 0 and self.updates % self.config.window_updates == 0
        if (self.pending is not None) != needs_pending:
            raise ValueError("pair pending proposal missing or premature")
        if self.pending is not None:
            self._validate_proposal(self.pending)
            if self.pending["generated_update"] != self.updates or not torch.equal(self.pending["mask_at_proposal"], self.mask):
                raise ValueError("pair pending graph/age mismatch")
        if self.config.mode in ("fixed", "fixed_probe") and (births or deaths):
            raise ValueError("fixed policy changed pair eligibility")
        if self.config.mode == "grow" and deaths or self.config.mode == "retire" and births:
            raise ValueError("pair acceptance flags were violated")
        self.model.validate_eligibility(self.mask)

    def _validate_event(self, proposal, event):
        summary = self._proposal_summary(proposal)
        extras = {"executed_update", "probe_calls", "fade_lr", "baseline", "birth_loss", "birth_damage",
            "birth_improves", "birth_accepted", "fade_baseline", "full_ablation_loss", "full_ablation_damage",
            "half_fade_loss", "half_fade_damage", "fade_improves", "fade_accepted", "birth_old_weights",
            "birth_trial_weights", "birth_final_weights", "fade_old_weights", "fade_trial_weights", "fade_final_weights",
            "actual_birth_indices", "actual_retire_indices", "births", "deaths"}
        if set(event) != set(summary) | extras or any(event[name] != value for name, value in summary.items()):
            raise ValueError("executed pair proposal schema/summary mismatch")
        lr = _number(event["fade_lr"], "finite fade lr", positive=True)
        for name in ("baseline", "birth_loss", "fade_baseline", "full_ablation_loss", "half_fade_loss"):
            self._validated_loss(event[name])
        for name, candidate, baseline in (("birth_damage", "birth_loss", "baseline"),
                                         ("full_ablation_damage", "full_ablation_loss", "fade_baseline"),
                                         ("half_fade_damage", "half_fade_loss", "fade_baseline")):
            if event[name] != self._damage(event[candidate], event[baseline]):
                raise ValueError("pair probe damage does not match recorded losses")
        birth, fade = proposal["birth_indices"], proposal["fade_indices"]

        def weights(name, count):
            value = event[name]
            if not isinstance(value, list) or len(value) != count or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
                raise ValueError("invalid pair probe weight record")
            return torch.tensor(value, dtype=self.model.edge_shared.dtype)

        birth_old = weights("birth_old_weights", len(birth))
        birth_trial = weights("birth_trial_weights", len(birth))
        birth_final = weights("birth_final_weights", len(birth))
        fade_old = weights("fade_old_weights", len(fade))
        fade_trial = weights("fade_trial_weights", len(fade))
        fade_final = weights("fade_final_weights", len(fade))
        if bool(birth_old.ne(0).any()) or not torch.equal(birth_trial, proposal["delta_all"][birth]):
            raise ValueError("pair birth trial does not match its stored Adam proposal")
        expected_fade = torch.where(fade_old.abs() <= lr, torch.zeros_like(fade_old), .5 * fade_old)
        if not torch.equal(fade_trial, expected_fade):
            raise ValueError("pair finite-fade trial does not match the half/threshold rule")
        birth_improves = event["birth_loss"]["mixed"] < event["baseline"]["mixed"]
        birth_accept = self.config.mode in ("grow", "both") and birth_improves and bool(birth_trial.ne(0).any())
        fade_baseline = event["birth_loss"] if birth_accept else event["baseline"]
        fade_improves = event["half_fade_loss"]["mixed"] < fade_baseline["mixed"]
        fade_accept = self.config.mode in ("retire", "both") and fade_improves and bool(len(fade))
        for name, expected in (("birth_improves", birth_improves), ("birth_accepted", birth_accept),
                               ("fade_improves", fade_improves), ("fade_accepted", fade_accept)):
            if not isinstance(event[name], bool) or event[name] != expected:
                raise ValueError("pair event violates independent strict-loss acceptance")
        if event["fade_baseline"] != fade_baseline:
            raise ValueError("finite-fade baseline does not follow accepted births")
        # Proposals retain score order; committed boolean-mask events retain
        # canonical pair order.  Weight lists still align with proposal order.
        expected_births = sorted(birth[birth_trial.ne(0)].tolist()) if birth_accept else []
        expected_retired = sorted(fade[fade_trial.eq(0)].tolist()) if fade_accept else []
        if (event["actual_birth_indices"] != expected_births or event["actual_retire_indices"] != expected_retired
                or event["births"] != len(expected_births) or event["deaths"] != len(expected_retired)
                or not torch.equal(birth_final, birth_trial if birth_accept else birth_old)
                or not torch.equal(fade_final, fade_trial if fade_accept else fade_old)):
            raise ValueError("pair trial final weights and eligibility events disagree")
