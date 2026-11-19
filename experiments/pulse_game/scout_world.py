"""Foraging with a physical sensor that reveals a hidden trial context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence

import numpy as np

from experiments.pulse_game.world import (
    CONTACT_ID,
    CTX,
    CUE,
    CUES,
    LEFT_ID,
    OBS_SIZE,
    REWARD_NEG,
    REWARD_POS,
    TICKS,
    TRIAL_START,
    TRIALS,
    VELOCITY_DECAY,
    Policy,
    PulseWorld,
    Rollout,
    Tape,
    YokeSchedule,
    _deranged_rows,
    _run_policy,
    _steer,
    make_tape,
)


SCOUT_OBS_SIZE = OBS_SIZE + 4
SCOUT_SIDE = slice(OBS_SIZE, OBS_SIZE + 2)
SCOUT_RANGE = OBS_SIZE + 2
SCOUT_HIT = OBS_SIZE + 3
SCOUT_DISTANCE = 0.35
N_SENSITIVE_CUES = 2

_SIDE = np.eye(2, dtype=np.float32)


@dataclass(frozen=True)
class ScoutTape(Tape):
    context_visible: ClassVar[bool] = False

    sensitive: np.ndarray
    scout_side: np.ndarray

    @property
    def scout_distance(self) -> float:
        return SCOUT_DISTANCE


def make_scout_tape(
    seeds: Sequence[int],
    trials: int = TRIALS,
    *,
    distance_jitter: float = 0.0,
    wind_scale: float = 0.0,
) -> ScoutTape:
    """Draw two sensitive cues per life, independently of the base identities.

    For each cue the A preference is an independent fair bit. B inverts the
    bit for the sampled sensitive cues and shares it for the other cues.
    Each trial draws an IID hidden context and independent beacon side.
    """
    if not 0 <= distance_jitter < 1 - SCOUT_DISTANCE:
        raise ValueError("both far targets must lie beyond the scout distance")
    if not 0 <= wind_scale < SCOUT_DISTANCE * (1 - VELOCITY_DECAY):
        raise ValueError("one gust must be too small to reach the scout from rest")
    tape = make_tape(seeds, trials, distance_jitter=distance_jitter, wind_scale=wind_scale)
    sensitive = np.zeros((tape.n, CUES), dtype=bool)
    scout_side = np.empty(tape.ctx.shape, dtype=np.int8)
    for i, seed in enumerate(tape.seeds):
        structure, beacon = np.random.SeedSequence([seed, 0xE76129F6]).spawn(2)
        cues = np.random.default_rng(structure).choice(CUES, N_SENSITIVE_CUES, replace=False)
        sensitive[i, cues] = True
        scout_side[i] = np.random.default_rng(beacon).integers(2, size=tape.trials)
    good_map = tape.good_map.copy()
    good_map[:, 1] = good_map[:, 0] ^ sensitive
    good_id = good_map[np.arange(tape.n)[:, None], tape.ctx, tape.cue]
    return ScoutTape(
        tape.seeds, tape.ctx, tape.cue, tape.left_id, good_id, good_map,
        tape.distances, tape.wind, sensitive, scout_side,
    )


@dataclass(frozen=True, kw_only=True)
class ScoutRollout(Rollout):
    scout_tick: np.ndarray
    scout_side: np.ndarray
    true_context: np.ndarray
    observed_context: np.ndarray
    sensitive_trial: np.ndarray

    @property
    def probed(self) -> np.ndarray:
        return self.scout_tick >= 0

    @property
    def reversed(self) -> np.ndarray:
        return self.probed & (self.contact_side >= 0) & (self.contact_side != self.scout_side)

    @property
    def probe_rate(self) -> float:
        return float(self.probed.mean())

    @property
    def reverse_rate(self) -> float:
        """Per-trial rate of reaching the endpoint opposite a visited beacon."""
        return float(self.reversed.mean())

    @property
    def latency_ticks(self) -> np.ndarray:
        return np.where(self.contact_tick >= 0, self.contact_tick + 1, -1)

    @property
    def timed_out(self) -> np.ndarray:
        return self.contact_tick < 0


@dataclass(frozen=True)
class ContextReplay:
    context: np.ndarray
    donor_rows: np.ndarray
    donor_seeds: tuple[int, ...]


def make_context_replay(donor: ScoutRollout, *, seed: int = 0) -> ContextReplay:
    """Fix a donor's entire sensor waveform independently of receiver actions."""
    rows = _deranged_rows(donor.seeds, seed)
    return ContextReplay(donor.observed_context[rows].copy(), rows, tuple(donor.seeds[i] for i in rows))


class ScoutWorld(PulseWorld):
    """Receive one context pulse a tick after physically crossing a beacon.

    All conditions keep the visible beacon and physical crossing receipt.
    The replay condition substitutes a prerecorded context stream while the
    receiving body still moves, probes, contacts, and earns its own rewards.
    """

    observation_size = SCOUT_OBS_SIZE

    def __init__(
        self,
        tape: ScoutTape,
        *,
        sensor: str = "true",
        replay: ContextReplay | np.ndarray | None = None,
        feedback: str = "true",
        yoke: YokeSchedule | np.ndarray | None = None,
        feedback_from_trial: int = 0,
        record: bool = False,
    ) -> None:
        if not isinstance(tape, ScoutTape):
            raise ValueError("scout world requires a ScoutTape")
        if sensor not in ("true", "none", "replay"):
            raise ValueError("sensor must be 'true', 'none', or 'replay'")
        if (sensor == "replay") != (replay is not None):
            raise ValueError("supply prerecorded contexts only in the replay condition")
        self.sensor = sensor
        shape = (tape.n, tape.trials)
        if isinstance(replay, ContextReplay):
            if any(donor == recipient for donor, recipient in zip(replay.donor_seeds, tape.seeds)):
                raise ValueError("a receiving life cannot replay its own context row")
            replay = replay.context
        self._replay = None
        if replay is not None:
            replay = np.asarray(replay)
            if replay.shape != (*shape, TICKS) or not np.isin(replay, (-1, 0, 1)).all():
                raise ValueError(f"replay context must contain values -1, 0, 1 with shape {(*shape, TICKS)}")
            if np.any(replay[..., 0] >= 0) or np.any((replay >= 0).sum(axis=-1) > 1):
                raise ValueError("replay context may reveal at most once and after movement starts")
            self._replay = replay.astype(np.int8, copy=True)
        self._scout_tick = np.full(shape, -1, dtype=np.int8)
        self._observed_context = np.full((*shape, TICKS), -1, dtype=np.int8)
        super().__init__(tape, feedback=feedback, yoke=yoke, feedback_from_trial=feedback_from_trial, record=record)

    def _load_trial(self) -> None:
        super()._load_trial()
        self._background[:, SCOUT_SIDE] = _SIDE[self.tape.scout_side[:, self.trial_index]]
        self._background[:, SCOUT_RANGE] = self.tape.scout_distance

    def _probe_receipt(self) -> np.ndarray:
        scout_tick = self._scout_tick[:, self.trial_index]
        return (scout_tick >= 0) & (self.tick_index == scout_tick + 1)

    def _shown_context(self, receipt: np.ndarray) -> np.ndarray:
        trial, tick = self.trial_index, self.tick_index
        if self.sensor == "true":
            return np.where(receipt, self.tape.ctx[:, trial], -1)
        if self._replay is not None:
            return self._replay[:, trial, tick]
        return np.full(self.tape.n, -1, dtype=np.int8)

    def _move(self, pulses: np.ndarray) -> None:
        super()._move(pulses)
        trial = self.trial_index
        scout_tick = self._scout_tick[:, trial]
        direction = 2 * self.tape.scout_side[:, trial] - 1
        reached = self._position * direction >= self.tape.scout_distance
        scout_tick[(scout_tick < 0) & reached] = self.tick_index

    def observe(self) -> np.ndarray:
        obs = super().observe()
        receipt = self._probe_receipt()
        shown = self._shown_context(receipt)
        reveal = shown >= 0
        obs[reveal, CTX] = _SIDE[shown[reveal]]
        obs[:, SCOUT_HIT] = receipt
        return obs

    def step(self, pulses: np.ndarray) -> np.ndarray | None:
        if not self.done:
            self._observed_context[:, self.trial_index, self.tick_index] = self._shown_context(self._probe_receipt())
        return super().step(pulses)

    def result(self) -> ScoutRollout:
        result = super().result()
        return ScoutRollout(
            **vars(result),
            scout_tick=self._scout_tick,
            scout_side=self.tape.scout_side,
            true_context=self.tape.ctx,
            observed_context=self._observed_context,
            sensitive_trial=self.tape.sensitive[np.arange(self.tape.n)[:, None], self.tape.cue],
        )


def rollout_scout(
    policy: Policy,
    tape: ScoutTape,
    *,
    sensor: str = "true",
    replay: ContextReplay | np.ndarray | None = None,
    feedback: str = "true",
    yoke: YokeSchedule | np.ndarray | None = None,
    feedback_from_trial: int = 0,
    record: bool = False,
) -> ScoutRollout:
    world = ScoutWorld(
        tape, sensor=sensor, replay=replay, feedback=feedback,
        yoke=yoke, feedback_from_trial=feedback_from_trial, record=record,
    )
    _run_policy(policy, world)
    return world.result()


def _scout_steer(obs: np.ndarray, target: np.ndarray, probing: np.ndarray) -> np.ndarray:
    beacon_target = obs[:, LEFT_ID].argmax(axis=1) ^ obs[:, SCOUT_SIDE].argmax(axis=1)
    return _steer(obs, np.where(probing, beacon_target, target))


class ScoutOraclePolicy:
    """Known preference map, current context available only through observation."""

    def __init__(self, tape: ScoutTape, *, probe: bool = True) -> None:
        self._map = tape.good_map
        self._sensitive = tape.sensitive
        self._probe = probe

    def reset(self, n: int) -> None:
        if n != len(self._map):
            raise ValueError("oracle batch size must match its map array")
        self._target = np.zeros(n, dtype=np.int8)
        self._probing = np.zeros(n, dtype=bool)

    def act(self, obs: np.ndarray) -> np.ndarray:
        cue = obs[:, CUE].argmax(axis=1)
        starts = np.flatnonzero(obs[:, TRIAL_START])
        self._target[starts] = self._map[starts, 0, cue[starts]]
        self._probing[starts] = self._probe & self._sensitive[starts, cue[starts]]

        if self._probe:
            seen = np.flatnonzero(obs[:, CTX].sum(axis=1) == 1)
            ctx = obs[seen, CTX].argmax(axis=1)
            self._target[seen] = self._map[seen, ctx, cue[seen]]
            self._probing[seen] = False
            self._probing[obs[:, SCOUT_HIT] > 0] = False
        return _scout_steer(obs, self._target, self._probing)


class ScoutHistoryPolicy:
    """Learn visible cue-context pairs and a cue-only fallback from receipts."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def reset(self, n: int) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._memory = np.full((n, 2, CUES), -1, dtype=np.int8)
        self._cue_votes = np.zeros((n, CUES, 2), dtype=np.int32)
        self._target = np.zeros(n, dtype=np.int8)
        self._guess = np.zeros(n, dtype=np.int8)
        self._ctx = np.full(n, -1, dtype=np.int8)
        self._probing = np.zeros(n, dtype=bool)

    def act(self, obs: np.ndarray) -> np.ndarray:
        cue = obs[:, CUE].argmax(axis=1)
        starts = np.flatnonzero(obs[:, TRIAL_START])
        self._ctx[starts] = -1
        votes = self._cue_votes[starts, cue[starts]]
        coin = self._rng.integers(2, size=starts.size)
        self._guess[starts] = np.where(votes[:, 0] == votes[:, 1], coin, votes.argmax(axis=1))
        memory_a = self._memory[starts, 0, cue[starts]]
        memory_b = self._memory[starts, 1, cue[starts]]
        known = (memory_a >= 0) & (memory_a == memory_b)
        self._target[starts] = np.where(known, memory_a, self._guess[starts])
        self._probing[starts] = ~known

        seen = np.flatnonzero(obs[:, CTX].sum(axis=1) == 1)
        self._ctx[seen] = obs[seen, CTX].argmax(axis=1)
        chosen = self._memory[seen, self._ctx[seen], cue[seen]]
        self._target[seen] = np.where(chosen >= 0, chosen, self._guess[seen])
        self._probing[seen] = False
        self._probing[obs[:, SCOUT_HIT] > 0] = False

        receipt = obs[:, CONTACT_ID]
        signed = obs[:, REWARD_POS] - obs[:, REWARD_NEG]
        updates = np.flatnonzero((receipt.sum(axis=1) == 1) & (signed != 0))
        contacted = receipt[updates].argmax(axis=1)
        inferred = np.where(signed[updates] > 0, contacted, 1 - contacted)
        self._cue_votes[updates, cue[updates], inferred] += 1
        with_context = self._ctx[updates] >= 0
        tagged = updates[with_context]
        self._memory[tagged, self._ctx[tagged], cue[tagged]] = inferred[with_context]
        return _scout_steer(obs, self._target, self._probing)
