"""Complete-state retirement adoption using the frozen native training loop.

Only configuration, source binding, restore and adoption validation differ.
All real collection, credit/replay, Adam updates, probes, lifetime trace work,
checkpoint writing and original-history checks remain inherited unchanged.
"""
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import torch
from experiments.undirected import run as native
from experiments.undirected.run import (
    _configure, _model, _layout, _validate_rng, _validate_tensor_map,
    _validate_optimizer, file_sha256, _atomic_json,
    dependency_snapshots, seed_for, HASH_FIELDS, RECORD_FIELDS,
)
from experiments.continuity import run as continuity
from experiments.lifelong.world import ContextWorld
from experiments.lifelong.schedule import Schedule
from experiments.lifelong.replay import HistoryReservoir
from experiments.undirected.credit import TemporalCreditEngine, CreditConfig
from experiments.undirected.plasticity import PairOperator as OriginalPairOperator
from experiments.coverage_exit.retirement import (
    RetirementConfig as PairConfig, RetirementOperator as PairOperator, adopt_fixed,
)

FORMAT_VERSION = "coverage-exit-retirement-trainer-v1"
SEMANTIC_FIELDS = native.SEMANTIC_FIELDS | {"retirement_origin"}
RUNTIME_SOURCES = ("run.py", "model.py", "credit.py", "plasticity.py", "fast_window.py")
DEPENDENCIES = native.DEPENDENCIES

@dataclass(frozen=True)
class NativeConfig(native.NativeConfig):
    structure: str = "retire"
    full_eligible: bool = False
    activation_update: int = 0

    def pair_config(self):
        return PairConfig(mode=self.structure, window_updates=self.structural_window_updates,
            candidates=self.structural_candidates, seed=seed_for(self.seed, "structure"),
            full_eligible=self.full_eligible, activation_update=self.activation_update)


def source_snapshots():
    here = Path(__file__).parent
    return {"run.py": Path(__file__).read_text(encoding="utf-8"),
        "plasticity.py": (here/"retirement.py").read_text(encoding="utf-8"),
        "fast_window.py": (here/"fast_window.py").read_text(encoding="utf-8"),
        **{name: (here.parent/"undirected"/name).read_text(encoding="utf-8") for name in ("model.py", "credit.py")}}


def source_hashes(sources=None):
    return native.source_hashes(source_snapshots() if sources is None else sources)


class NativeTrainer(native.NativeTrainer):
    def __init__(self, config):
        raise ValueError("use from_original or from_checkpoint; no new births in this study")

    def semantic_state(self):
        result = super().semantic_state()
        result["format"] = FORMAT_VERSION
        result["retirement_origin"] = deepcopy(self.retirement_origin)
        return result

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

    def _validate_continuation(self):
        c, origin = self.config, self.retirement_origin
        expected_original = {key: value for key, value in asdict(c).items() if key not in {"full_eligible", "activation_update"}}
        expected_original["structure"] = "fixed"
        if origin["activation_update"] != c.activation_update or origin["original_config"] != expected_original:
            raise ValueError("invalid retirement origin")
        old_operator = origin["original_operator"]
        if (old_operator["updates"] != c.activation_update or old_operator["config"]["mode"] != "fixed"
                or origin["operator_hash"] != continuity.state_digest(old_operator)
                or origin["trace_prefix_hash"] != continuity.state_digest(self.trace[:c.activation_update])
                or self.operator.update_history[:c.activation_update] != old_operator["update_history"]
                or not torch.equal(self.operator.initial_mask, old_operator["initial_mask"])):
            raise ValueError("retirement adoption lost the original operator/trace lifetime prefix")
        if c.activation_update and self.trace[c.activation_update-1]["operator_hash"] != origin["operator_hash"]:
            raise ValueError("adoption operator differs from its original committed boundary")
        if self.updates < c.activation_update or self.operator.config != c.pair_config():
            raise ValueError("retirement clock/config mismatch")
        self.operator._validate_restored()
        if self.updates == c.activation_update:
            # Validate the original committed boundary with its ORIGINAL operator.
            # A shallow validation view leaves both the historical trace and the
            # newly activated live operator untouched. Its other state is shared.
            view = copy(self)
            view.operator = OriginalPairOperator.from_state(self.model, old_operator)
            native.NativeTrainer._validate_continuation(view)
        else:
            native.NativeTrainer._validate_continuation(self)


def from_original(path, *, full_eligible):
    """Adopt at a strict validated original boundary without changing its trace."""
    from experiments.undirected.run import NativeTrainer as OriginalTrainer
    original = OriginalTrainer.from_checkpoint(path, device="cpu")
    if original.config.structure != "fixed":
        raise ValueError("source must use fixed structure")
    before = original.semantic_state()
    original.__class__ = NativeTrainer
    original.config = NativeConfig(**{**asdict(original.config), "structure": "retire",
        "full_eligible": full_eligible, "activation_update": original.updates})
    original.operator = adopt_fixed(original.model, before["operator"], full_eligible=full_eligible)
    original.retirement_origin = dict(activation_update=original.updates,
        source=str(Path(path).resolve()), source_sha256=file_sha256(path),
        semantic_hash=continuity.state_digest(before), original_config=before["config"],
        operator_hash=continuity.state_digest(before["operator"]),
        original_operator=before["operator"], trace_prefix_hash=continuity.state_digest(before["trace"]))
    original.sources, original.dependencies = source_snapshots(), dependency_snapshots()
    original.source_hashes, original.dependency_hashes = source_hashes(original.sources), source_hashes(original.dependencies)
    after = original.semantic_state()
    changed = {key for key in before if continuity.state_digest(before[key]) != continuity.state_digest(after[key])}
    if changed != {"format", "config", "operator"}:
        raise AssertionError(f"adoption changed continuing state: {changed}")
    original._validate_continuation()
    return original
