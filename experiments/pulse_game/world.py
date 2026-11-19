"""Closed-loop pulse-driven foraging with a hidden preference map per life."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import ClassVar, Protocol, Sequence

import numpy as np


TRIALS = 64
CUES = 4
TICKS = 16
MOVEMENT_TICKS = TICKS - 1
OBS_SIZE = 20
VELOCITY_DECAY = 0.65
PULSE_ACCELERATION = 0.15

CTX = slice(0, 2)
CUE = slice(2, 6)
LEFT_ID = slice(6, 8)
RIGHT_ID = slice(8, 10)
POSITION = 10
VELOCITY = 11
ACTIVE = 12
REWARD_POS = 13
REWARD_NEG = 14
CONTACT_ID = slice(15, 17)
TRIAL_START = 17
LEFT_DISTANCE = 18
RIGHT_DISTANCE = 19

_TWO = np.eye(2, dtype=np.float32)
_CUES = np.eye(CUES, dtype=np.float32)


class Policy(Protocol):
    def reset(self, n: int) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Tape:
    context_visible: ClassVar[bool] = True

    seeds: tuple[int, ...]
    ctx: np.ndarray
    cue: np.ndarray
    left_id: np.ndarray
    good_id: np.ndarray
    good_map: np.ndarray
    distances: np.ndarray
    wind: np.ndarray

    @property
    def n(self) -> int:
        return len(self.seeds)

    @property
    def trials(self) -> int:
        return self.ctx.shape[1]

    @property
    def right_id(self) -> np.ndarray:
        return 1 - self.left_id

    @property
    def map_codes(self) -> np.ndarray:
        return self.good_map[:, 0] @ (1 << np.arange(self.good_map.shape[-1]))


def _validated_int(value: int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def make_tape(
    seeds: Sequence[int],
    trials: int = TRIALS,
    cues: int = CUES,
    *,
    distance_jitter: float = 0.0,
    wind_scale: float = 0.0,
    map_codes: Sequence[int] | None = None,
) -> Tape:
    """Generate each life independently of array order and batch size.

    Bits of a map code give the four A preferences. B inverts each bit.
    Separate seeded streams draw preferences, contexts, cues, layout, geometry
    and wind. Short tapes are prefixes of longer versions of the same lives.
    The optional wind is one hidden gust per trial. The bound on its
    magnitude guarantees that residual inertial motion cannot carry a body
    with zero net motor input to either target.
    """
    seeds = tuple(_validated_int(s, "seed", 0, 2**64 - 1) for s in seeds)
    if not seeds:
        raise ValueError("at least one life seed is required")
    trials = _validated_int(trials, "trials", 1, 2**31 - 1)
    cues = _validated_int(cues, "cues", 1, CUES)
    if not 0 <= distance_jitter < 1:
        raise ValueError("distance_jitter must be in [0, 1)")
    wind_limit = (1 - distance_jitter) * (1 - VELOCITY_DECAY)
    if not 0 <= wind_scale < wind_limit:
        raise ValueError(f"wind_scale must be in [0, {wind_limit}) for these endpoints")
    codes = tuple(range(1 << cues)) if map_codes is None else tuple(
        _validated_int(code, "map code", 0, (1 << cues) - 1) for code in map_codes
    )
    if not codes or len(set(codes)) != len(codes):
        raise ValueError("map_codes must contain distinct codes")

    shape = (len(seeds), trials)
    ctx = np.empty(shape, dtype=np.int8)
    cue = np.empty(shape, dtype=np.int8)
    left_id = np.empty(shape, dtype=np.int8)
    good_map = np.empty((len(seeds), 2, cues), dtype=np.int8)
    distances = np.ones((*shape, 2), dtype=np.float32)
    wind = np.zeros((*shape, MOVEMENT_TICKS), dtype=np.float32)
    trials_index = np.arange(trials)
    cue_bits = np.arange(cues)
    for row, seed in enumerate(seeds):
        streams = np.random.SeedSequence([seed, 0x58CBAC24]).spawn(7)
        rng_map, rng_ctx, rng_cue, rng_layout, rng_distance, rng_gust_tick, rng_gust_size = (
            np.random.default_rng(stream) for stream in streams
        )
        code = codes[rng_map.integers(len(codes))]
        good_map[row, 0] = (code >> cue_bits) & 1
        good_map[row, 1] = 1 - good_map[row, 0]
        ctx[row] = rng_ctx.integers(2, size=trials)
        cue[row] = rng_cue.integers(cues, size=trials)
        left_id[row] = rng_layout.integers(2, size=trials)
        if distance_jitter:
            distances[row] = rng_distance.uniform(1 - distance_jitter, 1 + distance_jitter, (trials, 2))
        if wind_scale:
            gust_ticks = rng_gust_tick.integers(MOVEMENT_TICKS, size=trials)
            wind[row, trials_index, gust_ticks] = rng_gust_size.uniform(-wind_scale, wind_scale, trials)
    good_id = good_map[np.arange(len(seeds))[:, None], ctx, cue]
    return Tape(seeds, ctx, cue, left_id, good_id, good_map, distances, wind)


def swap_sides(tape: Tape, *, first_trial: int = 0) -> Tape:
    """Swap the visible target identities at and after the specified trial."""
    first_trial = _validated_int(first_trial, "first_trial", 0, tape.trials)
    left_id = tape.left_id.copy()
    left_id[:, first_trial:] = 1 - left_id[:, first_trial:]
    return replace(tape, left_id=left_id)


def assert_unseen(train: Tape, evaluation: Tape, *, disjoint_maps: bool = False) -> None:
    """Check held-out life seeds and optionally held-out preference map codes."""
    common_seeds = set(train.seeds).intersection(evaluation.seeds)
    if common_seeds:
        raise ValueError(f"train and evaluation share life seeds: {sorted(common_seeds)}")
    if disjoint_maps:
        if train.good_map.shape[-1] != evaluation.good_map.shape[-1]:
            raise ValueError("map code comparison requires the same cue count")
        common_maps = set(train.map_codes).intersection(evaluation.map_codes)
        if common_maps:
            raise ValueError(f"train and evaluation share map codes: {sorted(common_maps)}")


@dataclass(frozen=True)
class Rollout:
    seeds: tuple[int, ...]
    contact_id: np.ndarray
    contact_side: np.ndarray
    contact_tick: np.ndarray
    correct: np.ndarray
    reward: np.ndarray
    observed_reward: np.ndarray
    motor_pulses: np.ndarray
    obs: np.ndarray | None = None
    pulses: np.ndarray | None = None

    @property
    def score(self) -> np.ndarray:
        return self.correct.mean(axis=1)

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean())

    @property
    def contact_rate(self) -> float:
        return float((self.contact_id >= 0).mean())


@dataclass(frozen=True)
class YokeSchedule:
    reward: np.ndarray
    donor_rows: np.ndarray
    donor_seeds: tuple[int, ...]


def _deranged_rows(seeds: tuple[int, ...], seed: int) -> np.ndarray:
    n = len(seeds)
    if n < 2:
        raise ValueError("a deranged replay requires at least two donor lives")
    if len(set(seeds)) != n:
        raise ValueError("a deranged replay requires distinct donor life seeds")
    order = np.random.default_rng(seed).permutation(n)
    rows = np.empty(n, dtype=np.intp)
    rows[order] = np.roll(order, 1)
    return rows


def make_yoke(donor: Rollout, *, seed: int = 0) -> YokeSchedule:
    """Derange donor life rows while preserving the recorded reward waveforms.

    The yoke preserves the multiset of signs and offsets at every trial and
    tick. Its signs do not depend on the receiving body's contact. The
    receiving body still observes its own contact identity and trajectory.
    """
    rows = _deranged_rows(donor.seeds, seed)
    return YokeSchedule(donor.observed_reward[rows].copy(), rows, tuple(donor.seeds[i] for i in rows))


class PulseWorld:
    """Step a batch of bodies through synchronized trials.

    At the start of each trial a stationary body faces a target on each side.
    A left/right pulse adds negative/positive acceleration after velocity
    decay. Both pulses give zero net acceleration. For steps 0 through 14,
    the new position can contact exactly one endpoint. The body parks there
    and emits a one-shot receipt in the following observation. The cue,
    endpoints and target identities stay visible through slot 15, which can
    carry a receipt but has no movement. Each new trial begins at the center.
    """

    observation_size = OBS_SIZE

    def __init__(
        self,
        tape: Tape,
        *,
        feedback: str = "true",
        yoke: YokeSchedule | np.ndarray | None = None,
        feedback_from_trial: int = 0,
        record: bool = False,
    ) -> None:
        if feedback not in ("true", "none", "yoked"):
            raise ValueError("feedback must be 'true', 'none', or 'yoked'")
        if (feedback == "yoked") != (yoke is not None):
            raise ValueError("supply a reward schedule only with 'yoked' feedback")
        if type(self) is PulseWorld and not tape.context_visible:
            raise ValueError("use the scout world for a tape with hidden context")
        self.tape = tape
        self.feedback = feedback
        self.feedback_from_trial = _validated_int(feedback_from_trial, "feedback_from_trial", 0, tape.trials)
        shape = (tape.n, tape.trials)
        if isinstance(yoke, YokeSchedule):
            if any(donor == recipient for donor, recipient in zip(yoke.donor_seeds, tape.seeds)):
                raise ValueError("a receiving life cannot replay its own reward row")
            yoke = yoke.reward
        self._yoke = None
        if yoke is not None:
            yoke = np.asarray(yoke)
            if yoke.shape != (*shape, TICKS) or not np.isin(yoke, (-1, 0, 1)).all():
                raise ValueError(f"yoke must contain signs -1, 0, 1 with shape {(*shape, TICKS)}")
            self._yoke = yoke.astype(np.int8, copy=True)

        self.trial_index = 0
        self.tick_index = 0
        self._position = np.zeros(tape.n, dtype=np.float32)
        self._velocity = np.zeros(tape.n, dtype=np.float32)
        self._contact_id = np.full(shape, -1, dtype=np.int8)
        self._contact_side = np.full(shape, -1, dtype=np.int8)
        self._contact_tick = np.full(shape, -1, dtype=np.int8)
        self._reward = np.zeros(shape, dtype=np.int8)
        self._shown = np.zeros((*shape, TICKS), dtype=np.int8)
        self._motor_pulses = np.zeros((*shape, 2), dtype=np.uint8)
        self._observations = np.empty((*shape, TICKS, self.observation_size), dtype=np.float32) if record else None
        self._pulses = np.empty((*shape, TICKS, 2), dtype=bool) if record else None
        self._load_trial()

    @property
    def done(self) -> bool:
        return self.trial_index == self.tape.trials

    def _load_trial(self) -> None:
        trial = self.trial_index
        self._position.fill(0)
        self._velocity.fill(0)
        left_id = self.tape.left_id[:, trial]
        background = np.zeros((self.tape.n, self.observation_size), dtype=np.float32)
        if self.tape.context_visible:
            background[:, CTX] = _TWO[self.tape.ctx[:, trial]]
        background[:, CUE] = _CUES[self.tape.cue[:, trial]]
        background[:, LEFT_ID] = _TWO[left_id]
        background[:, RIGHT_ID] = _TWO[1 - left_id]
        background[:, LEFT_DISTANCE:RIGHT_DISTANCE + 1] = self.tape.distances[:, trial]
        self._background = background

    def _active(self) -> np.ndarray:
        return (self.tick_index < MOVEMENT_TICKS) & (self._contact_id[:, self.trial_index] < 0)

    def _receipt(self) -> np.ndarray:
        contact_tick = self._contact_tick[:, self.trial_index]
        return (contact_tick >= 0) & (contact_tick + 1 == self.tick_index)

    def _shown_reward(self, receipt: np.ndarray) -> np.ndarray:
        trial = self.trial_index
        if self.feedback == "none" or trial < self.feedback_from_trial:
            return np.zeros(self.tape.n, dtype=np.int8)
        if self._yoke is not None:
            return self._yoke[:, trial, self.tick_index]
        return self._reward[:, trial] * receipt

    def observe(self) -> np.ndarray:
        if self.done:
            raise RuntimeError("the life has finished")
        obs = self._background.copy()
        obs[:, POSITION] = self._position
        obs[:, VELOCITY] = self._velocity
        obs[:, ACTIVE] = self._active()
        receipt = self._receipt()
        shown = self._shown_reward(receipt)
        obs[:, REWARD_POS] = shown > 0
        obs[:, REWARD_NEG] = shown < 0
        obs[receipt, CONTACT_ID] = _TWO[self._contact_id[receipt, self.trial_index]]
        obs[:, TRIAL_START] = self.tick_index == 0
        return obs

    def step(self, pulses: np.ndarray) -> np.ndarray | None:
        if self.done:
            raise RuntimeError("the life has finished")
        pulses = np.asarray(pulses)
        if pulses.shape != (self.tape.n, 2):
            raise ValueError(f"motor pulses must have shape {(self.tape.n, 2)}")
        if pulses.dtype != np.bool_:
            if not np.all((pulses == 0) | (pulses == 1)):
                raise ValueError("each motor channel must return 0 or 1")
            pulses = pulses.astype(bool, copy=False)

        trial, tick = self.trial_index, self.tick_index
        self._shown[:, trial, tick] = self._shown_reward(self._receipt())
        if self._observations is not None:
            self._observations[:, trial, tick] = self.observe()
            self._pulses[:, trial, tick] = pulses
        if tick < MOVEMENT_TICKS:
            self._move(pulses)

        self.tick_index += 1
        if self.tick_index == TICKS:
            self.tick_index = 0
            self.trial_index += 1
            if self.done:
                return None
            self._load_trial()
        return self.observe()

    def _move(self, pulses: np.ndarray) -> None:
        trial, tick = self.trial_index, self.tick_index
        active = self._active()
        self._motor_pulses[:, trial] += pulses & active[:, None]
        drive = pulses[:, 1].astype(np.float32) - pulses[:, 0].astype(np.float32)
        velocity = VELOCITY_DECAY * self._velocity + PULSE_ACCELERATION * drive + self.tape.wind[:, trial, tick]
        position = self._position + velocity
        left_edge, right_edge = self.tape.distances[:, trial].T
        hit_left = active & (position <= -left_edge)
        hit_right = active & (position >= right_edge)
        hit = hit_left | hit_right
        self._position[active] = np.clip(position, -left_edge, right_edge)[active]
        self._velocity[active] = velocity[active]
        self._velocity[hit] = 0

        left_id = self.tape.left_id[:, trial]
        self._contact_id[hit_left, trial] = left_id[hit_left]
        self._contact_id[hit_right, trial] = 1 - left_id[hit_right]
        self._contact_side[hit_left, trial] = 0
        self._contact_side[hit_right, trial] = 1
        self._contact_tick[hit, trial] = tick
        self._reward[hit, trial] = np.where(
            self._contact_id[hit, trial] == self.tape.good_id[hit, trial], 1, -1
        )

    def result(self) -> Rollout:
        if not self.done:
            raise RuntimeError("the life is still active")
        return Rollout(
            self.tape.seeds,
            self._contact_id,
            self._contact_side,
            self._contact_tick,
            self._reward > 0,
            self._reward,
            self._shown,
            self._motor_pulses,
            self._observations,
            self._pulses,
        )


def rollout(
    policy: Policy,
    tape: Tape,
    *,
    feedback: str = "true",
    yoke: YokeSchedule | np.ndarray | None = None,
    feedback_from_trial: int = 0,
    record: bool = False,
) -> Rollout:
    world = PulseWorld(tape, feedback=feedback, yoke=yoke, feedback_from_trial=feedback_from_trial, record=record)
    _run_policy(policy, world)
    return world.result()


def _run_policy(policy: Policy, world: PulseWorld) -> None:
    policy.reset(world.tape.n)
    obs = world.observe()
    while obs is not None:
        obs = world.step(policy.act(obs))


def _steer(obs: np.ndarray, target_id: np.ndarray) -> np.ndarray:
    left = obs[:, LEFT_ID].argmax(axis=1) == target_id
    active = obs[:, ACTIVE].astype(bool)
    return np.column_stack((left & active, ~left & active))


class NoopPolicy:
    def reset(self, n: int) -> None:
        self._silent = np.zeros((n, 2), dtype=bool)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._silent


class OraclePolicy:
    """Evaluator-only motion and scoring control with access to good_id."""

    def __init__(self, tape: Tape) -> None:
        self._tape = tape

    def reset(self, n: int) -> None:
        if n != self._tape.n:
            raise ValueError("oracle batch size must match its tape")
        self._trial = -1

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs[0, TRIAL_START]:
            self._trial += 1
        return _steer(obs, self._tape.good_id[:, self._trial])


class RandomPolicy:
    """Uniform random target identity per trial with consistent steering."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def reset(self, n: int) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._target = np.zeros(n, dtype=np.int8)

    def act(self, obs: np.ndarray) -> np.ndarray:
        starts = obs[:, TRIAL_START].astype(bool)
        self._target[starts] = self._rng.integers(2, size=np.count_nonzero(starts))
        return _steer(obs, self._target)


class HistoryPolicy(RandomPolicy):
    """Infer target preferences only from the visible receipt and its sign."""

    def __init__(self, seed: int = 0, *, share_contexts: bool = False) -> None:
        super().__init__(seed)
        self._share_contexts = share_contexts

    def reset(self, n: int) -> None:
        super().reset(n)
        self._preference = np.full((n, 2, CUES), -1, dtype=np.int8)

    def act(self, obs: np.ndarray) -> np.ndarray:
        ctx = obs[:, CTX].argmax(axis=1)
        cue = obs[:, CUE].argmax(axis=1)
        receipt = obs[:, CONTACT_ID]
        signed_feedback = obs[:, REWARD_POS] - obs[:, REWARD_NEG]
        updates = np.flatnonzero((receipt.sum(axis=1) == 1) & (signed_feedback != 0))
        if updates.size:
            contacted = receipt[updates].argmax(axis=1)
            inferred = np.where(signed_feedback[updates] > 0, contacted, 1 - contacted)
            self._preference[updates, ctx[updates], cue[updates]] = inferred
            if self._share_contexts:
                self._preference[updates, 1 - ctx[updates], cue[updates]] = 1 - inferred

        starts = np.flatnonzero(obs[:, TRIAL_START])
        if starts.size:
            known = self._preference[starts, ctx[starts], cue[starts]]
            guess = self._rng.integers(2, size=starts.size)
            self._target[starts] = np.where(known >= 0, known, guess)
        return _steer(obs, self._target)
