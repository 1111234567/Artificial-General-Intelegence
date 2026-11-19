"""Magnitude-only candidate ranking; all loss acceptance and learning reuse B.

The checkpoint loader repeats the necessary B restore sequence with an explicit
local operator factory. Collection, learning, histories and all continuation
validation remain inherited, without changing any frozen module globals.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import torch
from experiments.coverage_exit import retirement_run as b
from experiments.coverage_exit.retirement import RetirementOperator, _digest
from experiments.coverage_exit.retirement_run import (
    NativeConfig, SEMANTIC_FIELDS, RECORD_FIELDS, HASH_FIELDS, DEPENDENCIES,
    _configure, _model, _layout, _validate_rng, _validate_tensor_map, _validate_optimizer,
    dependency_snapshots, seed_for, continuity, ContextWorld, Schedule,
    HistoryReservoir, TemporalCreditEngine, CreditConfig,
)

RANKING = "eligible smallest abs shared weight; stable canonical pair-index ties"
FORMAT_VERSION = "coverage-exit-retirement-magnitude-trainer-v1"

class MagnitudeOperator(RetirementOperator):
    def _select(self, mask, scores, *, largest):
        if largest:
            return super()._select(mask, scores, largest=True)
        weights = getattr(self, "_ranking_weight", None)
        if weights is None:
            weights = self.model.edge_shared.detach().cpu()
        return super()._select(mask, weights.abs().double(), largest=False)

    def _validate_proposal(self, proposal):
        # Historical proposals must rank their own stored weights, not today's.
        previous = getattr(self, "_ranking_weight", None)
        self._ranking_weight = proposal.get("weight_at_proposal") if isinstance(proposal, Mapping) else None
        try:
            return super()._validate_proposal(proposal)
        finally:
            self._ranking_weight = previous

    def state_dict(self):
        result = super().state_dict()
        result["fade_ranking"] = RANKING
        result["state_hash"] = _digest({k:v for k,v in result.items() if k != "state_hash"})
        return result

    @classmethod
    def from_state(cls, model, state):
        if not isinstance(state, Mapping) or state.get("fade_ranking") != RANKING:
            raise ValueError("magnitude operator ranking marker missing")
        return super().from_state(model, state)

PairOperator = MagnitudeOperator


def source_snapshots():
    return {**b.source_snapshots(), "retirement_magnitude.py": Path(__file__).read_text(encoding="utf-8")}


def source_hashes(sources=None):
    return b.source_hashes(source_snapshots() if sources is None else sources)

RUNTIME_SOURCES = tuple(source_snapshots())

class NativeTrainer(b.NativeTrainer):
    def semantic_state(self):
        result = super().semantic_state()
        result["format"] = FORMAT_VERSION
        return result

    def _validate_continuation(self):
        if self.retirement_origin.get("fade_ranking") != RANKING or not isinstance(self.operator, MagnitudeOperator):
            raise ValueError("magnitude continuation/ranking type mismatch")
        return super()._validate_continuation()

    @classmethod
    def from_state_dict(cls, state, *, device=None, strict_sources=True):
        if not isinstance(state, Mapping) or set(state) != SEMANTIC_FIELDS | RECORD_FIELDS or state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported or incomplete undirected trainer state")
        if not isinstance(state["config"], Mapping) or set(state["config"]) != set(NativeConfig.__dataclass_fields__):
            raise ValueError("incomplete configuration")
        if set(state["state_hashes"]) != set(HASH_FIELDS) | {"semantic"}:
            raise ValueError("incomplete continuation hashes")
        if any(state["state_hashes"][name] != continuity.state_digest(state[name]) for name in HASH_FIELDS):
            raise ValueError("continuation digest mismatch")
        if state["state_hashes"]["semantic"] != continuity.state_digest({name: state[name] for name in SEMANTIC_FIELDS}):
            raise ValueError("complete continuation digest mismatch")
        for key, names in (("sources", RUNTIME_SOURCES), ("dependencies", DEPENDENCIES)):
            if not isinstance(state[key], Mapping) or set(state[key]) != set(names):
                raise ValueError(f"incomplete {key}")
            hashes = "source_hashes" if key == "sources" else "dependency_hashes"
            if state[hashes] != source_hashes(state[key]):
                raise ValueError(f"{key} hash mismatch")
        if strict_sources and (state["source_hashes"] != source_hashes() or state["dependency_hashes"] != source_hashes(dependency_snapshots())):
            raise ValueError("runtime source mismatch; restore using the saved implementation")
        config = NativeConfig(**state["config"])
        if device is not None:
            config = replace(config, device=str(device))
        _validate_rng(state["rng"])
        _configure(config)
        result = cls.__new__(cls)
        result.config, result.device = config, torch.device(config.device)
        result.model = _model(config)  # no graph/weight birth; constructor RNG is isolated
        if result.model.config() != state["model_config"] or _layout(result.model) != state["parameter_layout"]:
            raise ValueError("model configuration or parameter layout mismatch")
        _validate_tensor_map(state["model"], result.model.state_dict(), "model")
        result.model.load_state_dict(state["model"], strict=True)
        result.optimizer = torch.optim.Adam(result.model.parameters(), lr=config.lr, weight_decay=0.)
        _validate_optimizer(state["optimizer"], result.optimizer, result.model, state["updates"])
        result.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        hidden = state["hidden"]
        if not isinstance(hidden, torch.Tensor) or hidden.shape != (config.batch_size, config.width) or hidden.dtype != getattr(torch, config.dtype) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("invalid hidden state dtype/layout")
        result.h = hidden.to(result.device).detach().clone()
        result.world = ContextWorld.from_state_dict(state["world"], device=config.world_device)
        result.schedule = Schedule.from_state_dict(state["schedule"])
        if result.schedule.config != config.schedule_config():
            raise ValueError("schedule configuration mismatch")
        result.replay = HistoryReservoir(config.replay_capacity, 2 * config.chunk_ticks, seed_for(config.seed, "replay"))
        result.replay.load_state_dict(state["replay"])
        result.operator = PairOperator.from_state(result.model, state["operator"])
        if result.operator.config != config.pair_config():
            raise ValueError("pair operator configuration mismatch")
        result.credit = TemporalCreditEngine(result.model, CreditConfig(method=config.credit,
            update_ticks=config.chunk_ticks, wait_weight=config.wait_weight, jacrev_chunk_size=config.jacrev_chunk_size))
        result.credit.load_state_dict(state["credit"])
        result.stats = continuity.ActionStats(11, result.device, config.wait_weight)
        _validate_tensor_map(state["stats"], result.stats.state_dict(), "statistics")
        result.stats.load_state_dict(state["stats"])
        for name in ("hidden_time", "schedule_history", "course_history", "auxiliary_stats", "work", "ticks", "updates", "hidden_resets",
            "birth", "action_hash", "data_hash", "trace", "last_chunk", "last_auxiliary", "last_live_family_tick",
            "last_aux_family_tick", "live_supervised_family_positions", "retirement_origin", "sources", "source_hashes", "dependencies",
            "dependency_hashes", "runtime_history", "seconds"):
            setattr(result, name, continuity.cpu_tree(state[name]))
        result._at_boundary = True
        result._validate_continuation()
        if result.updates and result.trace[-1]["rng_hash"] != continuity.state_digest(state["rng"]):
            raise ValueError("checkpoint RNG differs from last complete update")
        result.runtime = continuity.runtime_record(result.device)
        if not result.runtime_history or result.runtime_history[-1] != result.runtime:
            result.runtime_history.append(deepcopy(result.runtime))
        continuity.restore_rng_state(state["rng"])
        return result


def from_original(path, *, full_eligible=True):
    if full_eligible is not True:
        raise ValueError("magnitude arm retains the B full-eligible acceptance policy")
    trainer = b.from_original(path, full_eligible=True)
    trainer.__class__ = NativeTrainer
    trainer.operator.__class__ = MagnitudeOperator
    trainer.retirement_origin["fade_ranking"] = RANKING
    trainer.sources = source_snapshots()
    trainer.source_hashes = source_hashes(trainer.sources)
    trainer._validate_continuation()
    return trainer
