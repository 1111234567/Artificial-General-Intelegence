"""Finite, seed-private curricula; boundaries only change future request draws.

These are scheduling blocks, not resettable episodes. A returned family set is
an instruction for the world's *future NEW* generator. Existing requests,
pending replacements, sampled gaps, clocks, hidden state, and optimizer state
must continue unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
import random
from typing import Any


FORMAT_VERSION = "lifelong-schedule-v1"
SCHEDULE_MODES = ("blocked", "mixed")
CHECKPOINT_TICKS = (
    0, 1024, 32768, 33792, 65536, 66560, 98304, 99328,
    131072, 163840, 164864, 167936, 172032,
)


def _integer(name: str, value: int, *, lower: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return int(value)


def _private_seed(seed: int) -> int:
    digest = hashlib.sha256(f"{FORMAT_VERSION}/{seed}/family-order".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class ScheduleConfig:
    mode: str = "mixed"
    seed: int = 0
    main_ticks: int = 131072
    withdrawal_ticks: int = 32768
    return_ticks: int = 8192
    macrocycle_ticks: int = 32768
    block_ticks: int = 8192
    chunk_ticks: int = 128

    def __post_init__(self) -> None:
        if self.mode not in SCHEDULE_MODES:
            raise ValueError(f"schedule mode must be one of {SCHEDULE_MODES}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        for name in ("main_ticks", "withdrawal_ticks", "return_ticks", "macrocycle_ticks",
                     "block_ticks", "chunk_ticks"):
            object.__setattr__(self, name, _integer(name, getattr(self, name), lower=1))
        if self.macrocycle_ticks != 4 * self.block_ticks:
            raise ValueError("each macrocycle must have four family blocks")
        if self.main_ticks % self.macrocycle_ticks:
            raise ValueError("main_ticks must contain complete macrocycles")
        if any(getattr(self, name) % self.chunk_ticks for name in
               ("main_ticks", "withdrawal_ticks", "return_ticks", "macrocycle_ticks", "block_ticks")):
            raise ValueError("all schedule boundaries must align with chunk_ticks")

    @property
    def total_ticks(self) -> int:
        return self.main_ticks + self.withdrawal_ticks + self.return_ticks

    @property
    def return_start(self) -> int:
        return self.main_ticks + self.withdrawal_ticks


class Schedule:
    """Stateless tick lookup with a reproducible, private initial permutation.

    Constructor arguments may be supplied directly or as a ScheduleConfig.
    Rotation within each macrocycle balances recency across four main cycles;
    it does not assert equal realized family request counts. Mixed families are
    sampled by the world at NEW, using that world's independent private RNG.
    """

    def __init__(self, mode: str | ScheduleConfig = "mixed", seed: int = 0, **changes: Any):
        if isinstance(mode, ScheduleConfig):
            if changes or seed != 0:
                raise ValueError("supply either ScheduleConfig or constructor arguments")
            self.config = mode
        else:
            self.config = ScheduleConfig(mode=mode, seed=seed, **changes)
        order = list(range(4))
        random.Random(_private_seed(self.config.seed)).shuffle(order)
        self.initial_permutation = tuple(order)

    @property
    def total_ticks(self) -> int:
        return self.config.total_ticks

    @property
    def main_ticks(self) -> int:
        return self.config.main_ticks

    @property
    def return_start(self) -> int:
        return self.config.return_start

    def order_for_cycle(self, cycle: int) -> tuple[int, ...]:
        cycle = _integer("cycle", cycle)
        shift = cycle % 4
        return self.initial_permutation[shift:] + self.initial_permutation[:shift]

    def mode_for_tick(self, tick: int) -> dict[str, Any]:
        """Return the phase and permitted future NEW families at this boundary.

        tick == total_ticks returns a completion marker with family 0 retained;
        values beyond this finite protocol raise rather than silently extending
        training. Lookup never mutates RNG or any controller/world state.
        """
        tick = _integer("tick", tick)
        config = self.config
        if tick > config.total_ticks:
            raise ValueError("tick lies beyond the finite lifelong schedule")
        cycle, block = None, None
        if tick == config.total_ticks:
            phase, name, allowed = "complete", "complete", (0,)
        elif tick >= config.return_start:
            phase, name, allowed = "return", "family0-return", (0,)
        elif tick >= config.main_ticks:
            phase, name, allowed = "withdrawal", "family0-withdrawal", (1, 2, 3)
        else:
            phase = "main"
            cycle = tick // config.macrocycle_ticks
            if config.mode == "mixed":
                name, allowed = "main-mixed", (0, 1, 2, 3)
            else:
                block = (tick % config.macrocycle_ticks) // config.block_ticks
                family = self.order_for_cycle(cycle)[block]
                name, allowed = f"main-blocked-cycle{cycle}-family{family}", (family,)
        return {"phase": phase, "name": name, "future_family_allowed": allowed,
                "macrocycle": cycle, "family_block": block}

    def boundaries(self) -> tuple[int, ...]:
        """All ticks at which mode_for_tick's future-family instruction can change."""
        values = {0, self.config.main_ticks, self.config.return_start, self.total_ticks}
        if self.config.mode == "blocked":
            values.update(range(self.config.block_ticks, self.config.main_ticks, self.config.block_ticks))
        return tuple(sorted(values))

    def state_dict(self) -> dict[str, Any]:
        return {"format": FORMAT_VERSION, "config": asdict(self.config),
                "initial_permutation": self.initial_permutation}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Schedule":
        fields = {"format", "config", "initial_permutation"}
        if not isinstance(state, dict) or set(state) != fields or state.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported or incomplete lifelong schedule format")
        if not isinstance(state["config"], dict) or set(state["config"]) != set(ScheduleConfig.__dataclass_fields__):
            raise ValueError("unsupported or incomplete lifelong schedule configuration")
        result = cls(ScheduleConfig(**state["config"]))
        if tuple(state.get("initial_permutation", ())) != result.initial_permutation:
            raise ValueError("saved family permutation does not match its private seed")
        return result

    def summary(self) -> dict[str, Any]:
        return {**self.state_dict(), "total_ticks": self.total_ticks,
                "return_start": self.return_start,
                "boundaries": self.boundaries(),
                "exposure_note": "Equal nominal live ticks, not equal realized requests per family.",
                "continuity": "Only future NEW family draws change; no state or pending event resets."}

