"""Two physical contacts and a room-only outcome revaluation intervention."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np

from experiments.pulse_game.world import (
    ACTIVE, CONTACT_ID, LEFT_DISTANCE, LEFT_ID, MOVEMENT_TICKS, POSITION, PULSE_ACCELERATION,
    REWARD_NEG, REWARD_POS, RIGHT_DISTANCE, RIGHT_ID, TICKS, TRIAL_START, VELOCITY, VELOCITY_DECAY,
    Policy, _deranged_rows, _validated_int,
)


ROOM_ID = slice(0, 2)
ENTRANCE_STAGE = 2
ROOM_STAGE = 3
VISIT = 4
STAGE_START = 5
ENTRANCE_RECEIPT = slice(20, 22)
AVAILABLE = slice(22, 24)
OBS_SIZE = 24
EPISODE_TICKS = 2 * TICKS
DEFAULT_ACQUISITION = 32
DEFAULT_VISITS = 12

_ONE_HOT = np.eye(2, dtype=np.float32)


@dataclass(frozen=True)
class TwoStepTape:
    seeds: tuple[int, ...]
    acquisition: int
    visits: int
    rare_probability: float
    common_room: np.ndarray
    initial_good_room: np.ndarray
    final_good_room: np.ndarray
    good_fruit: np.ndarray
    entry_left: np.ndarray
    fruit_left: np.ndarray
    rare: np.ndarray
    forced_room: np.ndarray
    offered_fruit: np.ndarray
    distances: np.ndarray
    wind: np.ndarray

    @property
    def n(self) -> int:
        return len(self.seeds)

    @property
    def episodes(self) -> int:
        return self.acquisition + self.visits + 1

    @property
    def probe(self) -> int:
        return self.episodes - 1

    @property
    def initial_entrance(self) -> np.ndarray:
        return (self.common_room[:, 0] != self.initial_good_room).astype(np.int8)

    @property
    def final_entrance(self) -> np.ndarray:
        return (self.common_room[:, 0] != self.final_good_room).astype(np.int8)

    def preferred_fruit(self, rows: np.ndarray, episode: int, room: np.ndarray) -> np.ndarray:
        return self.good_fruit[rows]

    def positive(self, rows: np.ndarray, episode: int, room: np.ndarray, fruit: np.ndarray) -> np.ndarray:
        good_room = self.initial_good_room if episode < self.acquisition else self.final_good_room
        return (fruit == self.preferred_fruit(rows, episode, room)) & (room == good_room[rows])


def make_tape(
    seeds: Sequence[int],
    *,
    acquisition: int = DEFAULT_ACQUISITION,
    visits: int = DEFAULT_VISITS,
    rare_probability: float = 0.0,
    reversal: bool = True,
    distance_jitter: float = 0.0,
    wind_scale: float = 0.0,
) -> TwoStepTape:
    """Generate independent world rows and balanced visits in random order.

    The single rewarded pair moves to the other room on reversal. The reward
    bearing fruit identity is the same before and after the switch, so room-
    blind visits have the same marginal payoff distribution for each fruit.
    """
    seeds = tuple(_validated_int(seed, "seed", 0, 2**64 - 1) for seed in seeds)
    if not seeds:
        raise ValueError("provide at least one world seed")
    acquisition = _validated_int(acquisition, "acquisition", 1, 2**31 - 1)
    visits = _validated_int(visits, "visits", 4, 2**31 - 1)
    if visits % 4:
        raise ValueError("visits must be a multiple of four to cover each pair equally")
    if not 0 <= rare_probability < 0.5:
        raise ValueError("rare_probability must be in [0, 0.5)")
    if not 0 <= distance_jitter < 1:
        raise ValueError("distance_jitter must be in [0, 1)")
    wind_limit = (1 - distance_jitter) * (1 - VELOCITY_DECAY)
    if not 0 <= wind_scale < wind_limit:
        raise ValueError("a wind gust must not move a resting body to an endpoint")

    count = acquisition + visits + 1
    shape = (len(seeds), count)
    common = np.empty((len(seeds), 2), dtype=np.int8)
    good_room = np.empty(len(seeds), dtype=np.int8)
    good_fruit = np.empty(len(seeds), dtype=np.int8)
    entrance_left = np.empty(shape, dtype=np.int8)
    fruit_left = np.empty(shape, dtype=np.int8)
    rare = np.zeros(shape, dtype=bool)
    forced_room = np.full(shape, -1, dtype=np.int8)
    offered_fruit = np.full(shape, -1, dtype=np.int8)
    distances = np.ones((*shape, 2, 2), dtype=np.float32)
    wind = np.zeros((*shape, 2, MOVEMENT_TICKS), dtype=np.float32)
    episode, stage = np.indices((count, 2))
    visit_slice = slice(acquisition, acquisition + visits)
    for row, seed in enumerate(seeds):
        children = np.random.SeedSequence([seed, 0x4598BC21]).spawn(8)
        rng_map, rng_value, rng_layout, rng_rare, rng_visits, rng_distance, rng_tick, rng_gust = (
            np.random.default_rng(child) for child in children
        )
        start_room = rng_map.integers(2)
        common[row] = [start_room, 1 - start_room]
        good_room[row], good_fruit[row] = rng_value.integers(2, size=2)
        entrance_left[row], fruit_left[row] = rng_layout.integers(2, size=(2, count))
        rare[row] = rng_rare.random(count) < rare_probability
        visits_order = rng_visits.permutation(np.tile(np.arange(4), visits // 4))
        forced_room[row, visit_slice] = visits_order // 2
        offered_fruit[row, visit_slice] = visits_order % 2
        if distance_jitter:
            distances[row] = rng_distance.uniform(1 - distance_jitter, 1 + distance_jitter, size=(count, 2, 2))
        if wind_scale:
            tick = rng_tick.integers(MOVEMENT_TICKS, size=(count, 2))
            wind[row, episode, stage, tick] = rng_gust.uniform(-wind_scale, wind_scale, (count, 2))
    final_room = 1 - good_room if reversal else good_room.copy()
    return TwoStepTape(
        seeds, acquisition, visits, rare_probability, common, good_room, final_room, good_fruit,
        entrance_left, fruit_left, rare, forced_room, offered_fruit, distances, wind,
    )


def set_reversal(tape: TwoStepTape, reversal: bool) -> TwoStepTape:
    final_room = 1 - tape.initial_good_room if reversal else tape.initial_good_room.copy()
    return replace(tape, final_good_room=final_room)


@dataclass(frozen=True)
class TwoStepRollout:
    seeds: tuple[int, ...]
    acquisition: int
    visits: int
    entry_id: np.ndarray
    entry_side: np.ndarray
    entry_tick: np.ndarray
    room_id: np.ndarray
    fruit_id: np.ndarray
    fruit_side: np.ndarray
    fruit_tick: np.ndarray
    reward: np.ndarray
    shown: np.ndarray
    visits_confirmed: np.ndarray
    effective_entry: np.ndarray
    obs: np.ndarray | None = None
    pulses: np.ndarray | None = None

    @property
    def probe_entry(self) -> np.ndarray:
        return self.entry_id[:, -1]

    @property
    def probe_accuracy(self) -> float:
        return float(self.effective_entry[:, -1].mean())


@dataclass(frozen=True)
class VisitReplay:
    reward: np.ndarray
    donor_seeds: tuple[int, ...]


def make_visit_replay(donor: TwoStepRollout, *, seed: int = 0) -> VisitReplay:
    """Match the visit sign distribution but assign it to independent lives."""
    rows = _deranged_rows(donor.seeds, seed)
    start = donor.acquisition
    signs = donor.shown[rows, start:start + donor.visits].sum(axis=-1, dtype=np.int8)
    return VisitReplay(signs, tuple(donor.seeds[i] for i in rows))


class TwoStepWorld:
    """Fixed 16 entrance slots and 16 room slots, with one policy state."""

    def __init__(
        self,
        tape: TwoStepTape,
        *,
        hide_rooms: Literal["never", "visits", "always"] = "never",
        visit_replay: VisitReplay | np.ndarray | None = None,
        record: bool = False,
    ) -> None:
        if hide_rooms not in ("never", "visits", "always"):
            raise ValueError("hide_rooms must be 'never', 'visits', or 'always'")
        if isinstance(visit_replay, VisitReplay):
            if any(own == donor for own, donor in zip(tape.seeds, visit_replay.donor_seeds)):
                raise ValueError("a life cannot replay its own visit sequence")
            visit_replay = visit_replay.reward
        self._visit_replay = None
        if visit_replay is not None:
            replay = np.asarray(visit_replay)
            if replay.shape != (tape.n, tape.visits) or not np.isin(replay, (-1, 0, 1)).all():
                raise ValueError("visit replay must provide one sign per life and forced visit")
            self._visit_replay = replay.astype(np.int8, copy=True)
        self.tape = tape
        self.hide_rooms = hide_rooms
        self.episode_index = 0
        self.tick_index = 0
        self.rows = np.arange(tape.n)
        self._position = np.zeros(tape.n, dtype=np.float32)
        self._velocity = np.zeros(tape.n, dtype=np.float32)
        shape = (tape.n, tape.episodes)
        self._entry_id = np.full(shape, -1, dtype=np.int8)
        self._entry_side = np.full(shape, -1, dtype=np.int8)
        self._entry_tick = np.full(shape, -1, dtype=np.int8)
        self._room_id = np.full(shape, -1, dtype=np.int8)
        self._fruit_id = np.full(shape, -1, dtype=np.int8)
        self._fruit_side = np.full(shape, -1, dtype=np.int8)
        self._fruit_tick = np.full(shape, -1, dtype=np.int8)
        self._reward = np.zeros(shape, dtype=np.int8)
        self._shown = np.zeros((*shape, EPISODE_TICKS), dtype=np.int8)
        self._obs = np.empty((*shape, EPISODE_TICKS, OBS_SIZE), dtype=np.float32) if record else None
        self._pulses = np.empty((*shape, EPISODE_TICKS, 2), dtype=bool) if record else None
        self._load_stage()

    @property
    def done(self) -> bool:
        return self.episode_index == self.tape.episodes

    @property
    def stage(self) -> int:
        return self.tick_index // TICKS

    @property
    def local_tick(self) -> int:
        return self.tick_index % TICKS

    @property
    def is_visit(self) -> bool:
        return self.tape.acquisition <= self.episode_index < self.tape.probe

    def _load_stage(self) -> None:
        t, stage = self.episode_index, self.stage
        self._position.fill(0)
        self._velocity.fill(0)
        background = np.zeros((self.tape.n, OBS_SIZE), dtype=np.float32)
        background[:, VISIT] = self.is_visit
        if stage == 0:
            present = np.full(self.tape.n, not self.is_visit, dtype=bool)
            left_id = self.tape.entry_left[:, t]
            background[:, ENTRANCE_STAGE] = present
            available = np.broadcast_to(present[:, None], (self.tape.n, 2))
        else:
            entry = self._entry_id[:, t]
            if self.is_visit:
                room = self.tape.forced_room[:, t]
            else:
                common = self.tape.common_room[self.rows, np.maximum(entry, 0)]
                room = np.where(entry >= 0, common ^ self.tape.rare[:, t], -1)
            self._room_id[:, t] = room
            present = room >= 0
            if self.hide_rooms == "never" or (self.hide_rooms == "visits" and not self.is_visit):
                background[present, ROOM_ID] = _ONE_HOT[room[present]]
            left_id = self.tape.fruit_left[:, t]
            background[:, ROOM_STAGE] = present
            if self.is_visit:
                left_available = self.tape.offered_fruit[:, t] == left_id
                available = np.column_stack((left_available, ~left_available))
            else:
                available = np.broadcast_to(present[:, None], (self.tape.n, 2))
        background[present, LEFT_ID] = _ONE_HOT[left_id[present]]
        background[present, RIGHT_ID] = _ONE_HOT[1 - left_id[present]]
        background[present, LEFT_DISTANCE:RIGHT_DISTANCE + 1] = self.tape.distances[present, t, stage]
        background[:, AVAILABLE] = available
        self._background = background
        self._left_id = left_id
        self._present = present
        self._available = available

    def _active(self) -> np.ndarray:
        contacted = self._entry_id if self.stage == 0 else self._fruit_id
        return (self.local_tick < MOVEMENT_TICKS) & self._present & (contacted[:, self.episode_index] < 0)

    def _receipt(self) -> np.ndarray:
        contact_tick = self._entry_tick if self.stage == 0 else self._fruit_tick
        prev = contact_tick[:, self.episode_index]
        return (prev >= 0) & (prev + 1 == self.local_tick)

    def _feedback(self, receipt: np.ndarray) -> np.ndarray:
        if self.stage == 0 or self.episode_index == self.tape.probe:
            return np.zeros(self.tape.n, dtype=np.int8)
        if self.is_visit and self._visit_replay is not None:
            return self._visit_replay[:, self.episode_index - self.tape.acquisition] * receipt
        return self._reward[:, self.episode_index] * receipt

    def observe(self) -> np.ndarray:
        if self.done:
            raise RuntimeError("the life has finished")
        obs = self._background.copy()
        obs[:, POSITION] = self._position
        obs[:, VELOCITY] = self._velocity
        obs[:, ACTIVE] = self._active()
        obs[:, TRIAL_START] = self.tick_index == 0
        obs[:, STAGE_START] = (self.local_tick == 0) & self._present
        receipt = self._receipt()
        if self.stage == 0:
            obs[receipt, ENTRANCE_RECEIPT] = _ONE_HOT[self._entry_id[receipt, self.episode_index]]
        else:
            obs[receipt, CONTACT_ID] = _ONE_HOT[self._fruit_id[receipt, self.episode_index]]
            sign = self._feedback(receipt)
            obs[:, REWARD_POS] = sign > 0
            obs[:, REWARD_NEG] = sign < 0
        return obs

    def _move(self, pulses: np.ndarray) -> None:
        t, stage, local = self.episode_index, self.stage, self.local_tick
        active = self._active()
        drive = pulses[:, 1].astype(np.float32) - pulses[:, 0].astype(np.float32)
        velocity = VELOCITY_DECAY * self._velocity + PULSE_ACCELERATION * drive + self.tape.wind[:, t, stage, local]
        position = self._position + velocity
        left_edge, right_edge = self.tape.distances[:, t, stage].T
        at_left = active & (position <= -left_edge)
        at_right = active & (position >= right_edge)
        hit_left = at_left & self._available[:, 0]
        hit_right = at_right & self._available[:, 1]
        hit = hit_left | hit_right
        self._position[active] = np.clip(position, -left_edge, right_edge)[active]
        self._velocity[active] = velocity[active]
        self._velocity[at_left | at_right] = 0
        contacted = self._entry_id if stage == 0 else self._fruit_id
        side = self._entry_side if stage == 0 else self._fruit_side
        tick = self._entry_tick if stage == 0 else self._fruit_tick
        contacted[hit_left, t] = self._left_id[hit_left]
        contacted[hit_right, t] = 1 - self._left_id[hit_right]
        side[hit_left, t], side[hit_right, t] = 0, 1
        tick[hit, t] = local
        if stage == 1 and hit.any():
            rows = np.flatnonzero(hit)
            correct = self.tape.positive(rows, t, self._room_id[rows, t], contacted[rows, t])
            self._reward[rows, t] = np.where(correct, 1, -1)

    def step(self, pulses: np.ndarray) -> np.ndarray | None:
        if self.done:
            raise RuntimeError("the life has finished")
        pulses = np.asarray(pulses)
        if pulses.shape != (self.tape.n, 2) or not ((pulses == 0) | (pulses == 1)).all():
            raise ValueError("policy must produce a batch of 0/1 left and right pulses")
        self._shown[:, self.episode_index, self.tick_index] = self._feedback(self._receipt())
        if self._obs is not None:
            self._obs[:, self.episode_index, self.tick_index] = self.observe()
            self._pulses[:, self.episode_index, self.tick_index] = pulses
        if self.local_tick < MOVEMENT_TICKS:
            self._move(pulses)
        self.tick_index += 1
        if self.tick_index == EPISODE_TICKS:
            self.tick_index = 0
            self.episode_index += 1
            if self.done:
                return None
        if self.local_tick == 0:
            self._load_stage()
        return self.observe()

    def result(self) -> TwoStepRollout:
        if not self.done:
            raise RuntimeError("the life is still active")
        t = self.tape
        visited = slice(t.acquisition, t.probe)
        visit_fruit = self._fruit_id[:, visited]
        visit_room = self._room_id[:, visited]
        preferred = np.column_stack([
            t.preferred_fruit(self.rows, episode, self._room_id[:, episode]) for episode in range(t.acquisition, t.probe)
        ])
        saw_old = ((visit_fruit == preferred) & (visit_room == t.initial_good_room[:, None])).any(axis=1)
        saw_alternative = ((visit_fruit == preferred) & (visit_room != t.initial_good_room[:, None])).any(axis=1)
        target = np.broadcast_to(t.initial_entrance[:, None], (t.n, t.episodes)).copy()
        target[:, t.probe] = t.final_entrance
        return TwoStepRollout(
            t.seeds, t.acquisition, t.visits, self._entry_id, self._entry_side, self._entry_tick,
            self._room_id, self._fruit_id, self._fruit_side, self._fruit_tick, self._reward, self._shown,
            saw_old & saw_alternative, self._entry_id == target, self._obs, self._pulses,
        )


def rollout(policy: Policy, tape: TwoStepTape, **world_options) -> TwoStepRollout:
    world = TwoStepWorld(tape, **world_options)
    policy.reset(tape.n)
    obs = world.observe()
    while obs is not None:
        obs = world.step(policy.act(obs))
    return world.result()
