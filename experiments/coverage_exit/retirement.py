"""Retire-only full-ablation switch over the unchanged frozen pair machinery.

The frozen implementation supplies ranking, gradient accumulation, four-probe
scheduling, Adam checks, lifetime updates and projection. Only acceptance and
its explicit state/proposal/event validation differ here.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import torch
from experiments.undirected.plasticity import (
    PairConfig, PairOperator, _copy, _digest, _integer, _number,
)

FORMAT_VERSION = "coverage-exit-retirement-operator-v1"

@dataclass(frozen=True)
class RetirementConfig(PairConfig):
    full_eligible: bool = False
    activation_update: int = 0

    def __post_init__(self):
        super().__post_init__()
        _integer(self.activation_update, "activation_update")
        if not isinstance(self.full_eligible, bool) or self.mode != "retire":
            raise ValueError("retirement study is retire-only with an explicit full switch")

class RetirementOperator(PairOperator):
    """Full removal is eligible only when its actual mixed-loss trial improves."""
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
            full_improves = full_loss["mixed"] < fade_baseline["mixed"]
            full_accepted = self.config.full_eligible and full_improves and bool(len(fade_cpu))
            if full_accepted:
                weight[fade] = 0.
                retired[fade_cpu] = True
                accept_fade = True
            elif accept_fade:
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
                "full_improves": full_improves, "full_accepted": full_accepted,
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
        if ((age - self.config.activation_update) % self.config.window_updates or age <= self.config.activation_update or proposal["execute_update"] != age + 1 or proposal["adam_step"] != age
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
        if set(state["config"]) != set(RetirementConfig.__dataclass_fields__):
            raise ValueError("incomplete pair operator configuration")
        config = RetirementConfig(**state["config"])
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
            if age > self.config.activation_update and (age-self.config.activation_update) % self.config.window_updates == 0:
                window_masks[age], window_support[age] = mask.clone(), support.clone()
        tensors = {"mask": mask, "support_mask": support, "initial_survival": survival, "ever_born": ever,
                   "last_birth": last_birth, "last_retire": last_retire, "activation_counts": activations}
        if any(not torch.equal(getattr(self, name), expected) for name, expected in tensors.items()):
            raise ValueError("pair lifetime traits do not rebuild from events")
        if (births, deaths, numeric_births, numeric_deaths) != (self.cumulative_births, self.cumulative_deaths, self.numeric_support_births, self.numeric_support_deaths):
            raise ValueError("pair turnover counters do not rebuild from events")
        if bool((self.initial_support & ~self.initial_mask).any()) or not torch.equal(self.support_mask, self._support()):
            raise ValueError("pair support does not match its physical model")
        expected_events = 0 if self.config.mode == "fixed" else max(0, (self.updates - self.config.activation_update - 1) // self.config.window_updates)
        if not isinstance(self.event_history, list) or len(self.event_history) != expected_events or self.probe_calls != 4 * expected_events:
            raise ValueError("pair probe budget/history mismatch")
        for index, record in enumerate(self.event_history, 1):
            if not isinstance(record, Mapping) or set(record) != {"proposal", "event"}:
                raise ValueError("incomplete executed pair proposal")
            self._validate_proposal(record["proposal"])
            expected_age = self.config.activation_update + index * self.config.window_updates + 1
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
        expected_count = 0 if self.config.mode == "fixed" else (self.updates-self.config.activation_update) % self.config.window_updates
        if self.window_count != expected_count or (self.window_count == 0 and bool(self.window_gradient_sum.ne(0).any())):
            raise ValueError("pair partial-window age mismatch")
        needs_pending = self.config.mode != "fixed" and self.updates > self.config.activation_update and (self.updates-self.config.activation_update) % self.config.window_updates == 0
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
            "actual_birth_indices", "actual_retire_indices", "births", "deaths", "full_improves", "full_accepted"}
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
        full_improves = event["full_ablation_loss"]["mixed"] < fade_baseline["mixed"]
        full_accept = self.config.full_eligible and full_improves and bool(len(fade))
        fade_accept = (full_accept or fade_improves) and bool(len(fade))
        for name, expected in (("birth_improves", birth_improves), ("birth_accepted", birth_accept),
                               ("fade_improves", fade_improves), ("fade_accepted", fade_accept),
                               ("full_improves", full_improves), ("full_accepted", full_accept)):
            if not isinstance(event[name], bool) or event[name] != expected:
                raise ValueError("pair event violates independent strict-loss acceptance")
        if event["fade_baseline"] != fade_baseline:
            raise ValueError("finite-fade baseline does not follow accepted births")
        # Proposals retain score order; committed boolean-mask events retain
        # canonical pair order.  Weight lists still align with proposal order.
        expected_births = sorted(birth[birth_trial.ne(0)].tolist()) if birth_accept else []
        committed_fade = torch.zeros_like(fade_trial) if full_accept else fade_trial
        expected_retired = sorted(fade[committed_fade.eq(0)].tolist()) if fade_accept else []
        if (event["actual_birth_indices"] != expected_births or event["actual_retire_indices"] != expected_retired
                or event["births"] != len(expected_births) or event["deaths"] != len(expected_retired)
                or not torch.equal(birth_final, birth_trial if birth_accept else birth_old)
                or not torch.equal(fade_final, committed_fade if fade_accept else fade_old)):
            raise ValueError("pair trial final weights and eligibility events disagree")

def adopt_fixed(model, state, *, full_eligible):
    """Validate original fixed state then retain all lifetime and numeric history."""
    from experiments.undirected.plasticity import PairOperator
    old = PairOperator.from_state(model, state)
    if old.config.mode != "fixed" or old.pending is not None or old.window_count or old.event_history:
        raise ValueError("adoption requires an original fixed operator")
    config = RetirementConfig(mode="retire", window_updates=old.config.window_updates,
        candidates=old.config.candidates, seed=old.config.seed, full_eligible=full_eligible,
        activation_update=old.updates)
    result = RetirementOperator(model, old.initial_mask, config)
    for name in (*old._BOOL, *old._TRAITS, *old._COUNTERS, "update_history"):
        setattr(result, name, _copy(getattr(old, name)))
    result.window_gradient_sum.zero_()
    result.window_count, result.pending, result.event_history = 0, None, []
    result._validate_restored()
    return result
