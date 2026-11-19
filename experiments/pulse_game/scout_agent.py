"""Online cue-conditioned value learning and a fixed spiking steering primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.pulse_game.scout_world import SCOUT_HIT, SCOUT_SIDE
from experiments.pulse_game.world import (
    ACTIVE,
    CONTACT_ID,
    CTX,
    CUE,
    CUES,
    LEFT_ID,
    REWARD_NEG,
    REWARD_POS,
    TICKS,
    TRIAL_START,
)


_HYPOTHESES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
_MOTOR_ONE_HOT = np.eye(2, dtype=np.float32)
_STRATEGIES = frozenset(("adaptive", "reward_actor", "all_probe", "never_probe", "passive", "fixed_schedule"))


@dataclass(frozen=True)
class ScoutConfig:
    probe_cost: float = 0.05
    novelty: float = 0.11
    actor_weight: float = 0.05
    value_step: float = 0.27
    actor_ucb: float = 0.16
    decision_noise: float = 0.015
    memory_error: float = 0.05
    memory_retention: float = 0.995
    latency_cost: float = 0.04
    motor_decay: float = 0.4
    context_decay: float = 0.94


class LocalScoutPolicy:
    """A cue-local outcome learner with recurrent context and motor states.

    The four synaptic hypothesis values estimate which object is good in
    A and B. A signed endpoint outcome and the public contacted identity
    update only the current cue's values. A leaky context trace can tag
    that outcome. Untagged outcomes update a mixture over the two contexts.
    A separate two-action critic updates the value of the executed intent.

    Adaptive choice compares predicted context-conditioned and blind success
    plus a bonus for resolving uncertainty. Reward-actor choice instead uses
    the critic and its own opportunity counts. Steering to a chosen beacon
    or endpoint is a supplied motor primitive. Its two leaky integrate-and-
    fire units translate desired direction into pulses. The body handles
    acceleration, coasting, overshoot and physical contact.
    """

    def __init__(
        self,
        config: ScoutConfig = ScoutConfig(),
        *,
        seed: int = 0,
        strategy: str = "adaptive",
        learn: bool = True,
        probe_schedule: np.ndarray | None = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"unknown exploration strategy {strategy!r}")
        if (strategy == "fixed_schedule") != (probe_schedule is not None):
            raise ValueError("only fixed_schedule requires a probe_schedule")
        self._probe_schedule = None
        if probe_schedule is not None:
            schedule = np.asarray(probe_schedule, dtype=np.float32)
            if schedule.ndim != 1 or not len(schedule) or not ((schedule >= 0) & (schedule <= 1)).all():
                raise ValueError("probe_schedule must contain one probability per trial")
            self._probe_schedule = schedule.copy()
        if not 0 < config.memory_error < 0.5:
            raise ValueError("memory_error must be in (0, 0.5)")
        if not 0 <= config.motor_decay < 1 or not 0 < config.context_decay <= 1:
            raise ValueError("invalid recurrent leak")
        self.config = config
        self.seed = seed
        self.strategy = strategy
        self.learn = learn
        error = config.memory_error
        likelihood_a_b = np.where(_HYPOTHESES.T[:, None] == np.arange(2)[None, :, None], 1 - error, error)
        likelihood = np.concatenate([likelihood_a_b, likelihood_a_b.mean(axis=0, keepdims=True)])
        self._log_likelihood = np.log(likelihood).astype(np.float32)

    def reset(self, n: int) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._rows = np.arange(n)
        self._belief = np.zeros((n, CUES, len(_HYPOTHESES)), dtype=np.float32)
        self._quality = np.full((n, CUES, 2), 0.5, dtype=np.float32)
        self._visits = np.zeros((n, CUES, 2), dtype=np.int32)
        self._ctx_voltage = np.zeros((n, 2), dtype=np.float32)
        self._motor_voltage = np.zeros((n, 2), dtype=np.float32)
        self._target = np.zeros(n, dtype=np.int8)
        self._tie = np.zeros(n, dtype=np.int8)
        self._intent = np.zeros(n, dtype=bool)
        self._seeking = np.zeros(n, dtype=bool)
        self._resolved = np.zeros(n, dtype=bool)
        self._trial = -1
        self._tick = 0
        self._intents: list[np.ndarray] = []

    @property
    def intents(self) -> np.ndarray:
        """Per-life routing choice at each trial's first observation."""
        return np.stack(self._intents, axis=1)

    def _probability(self, rows: np.ndarray, cue: np.ndarray) -> np.ndarray:
        weights = np.exp(self._belief[rows, cue])
        return (weights @ _HYPOTHESES) / weights.sum(axis=1, keepdims=True)

    def _select(self, p: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return np.where(p == 0.5, self._tie[rows], p > 0.5)

    def _begin_trial(self, cue: np.ndarray) -> None:
        self._trial += 1
        self._tick = 0
        self._ctx_voltage.fill(0)
        self._resolved.fill(False)
        self._tie[:] = self._rng.integers(2, size=self._rows.size)
        coin = self._rng.random(self._rows.size)
        p = self._probability(self._rows, cue)
        p_blind = p.mean(axis=1)
        self._target[:] = self._select(p_blind, self._rows)

        if self.strategy == "all_probe":
            self._intent.fill(True)
        elif self.strategy in ("never_probe", "passive"):
            self._intent.fill(False)
        elif self.strategy == "fixed_schedule":
            self._intent[:] = coin < self._probe_schedule[self._trial]
        else:
            self._choose_exploration(cue, p, p_blind, coin)
        self._seeking[:] = self._intent
        self._intents.append(self._intent.copy())

    def _choose_exploration(self, cue: np.ndarray, p: np.ndarray, p_blind: np.ndarray, coin: np.ndarray) -> None:
        quality = self._quality[self._rows, cue]
        learned_diff = quality[:, 1] - quality[:, 0]
        if self.strategy == "reward_actor":
            visits = self._visits[self._rows, cue]
            bonuses = self.config.actor_ucb / np.sqrt(visits + 1)
            value = learned_diff + bonuses[:, 1] - bonuses[:, 0]
        else:
            confidence = np.abs(p - 0.5)
            information_value = confidence.mean(axis=1) - np.abs(p_blind - 0.5)
            novelty = self.config.novelty * (1.0 - 2.0 * confidence.min(axis=1))
            value = information_value + novelty + self.config.actor_weight * learned_diff
        noise = self.config.decision_noise * (2 * coin - 1)
        self._intent[:] = value + noise > self.config.probe_cost

    def _learn_outcome(self, obs: np.ndarray, cue: np.ndarray) -> None:
        sign = obs[:, REWARD_POS] - obs[:, REWARD_NEG]
        receipt = obs[:, CONTACT_ID]
        contact = receipt.sum(axis=1) == 1
        self._resolved |= contact
        tagged = np.flatnonzero(contact & (sign != 0))
        if tagged.size:
            actual = receipt[tagged].argmax(axis=1)
            best = np.where(sign[tagged] > 0, actual, 1 - actual)
            context_voltage = self._ctx_voltage[tagged]
            ctx = np.where(context_voltage.max(axis=1) > 0.2, context_voltage.argmax(axis=1), 2)
            values = self.config.memory_retention * self._belief[tagged, cue[tagged]]
            values += self._log_likelihood[ctx, best]
            values -= values.max(axis=1, keepdims=True)
            self._belief[tagged, cue[tagged]] = np.maximum(values, -32)
            utility = (sign[tagged] > 0).astype(np.float32) - self.config.latency_cost * self._tick / (TICKS - 1)
            self._update_critic(tagged, cue, utility)
        if self._tick == TICKS - 1:
            expired = np.flatnonzero(~self._resolved)
            self._update_critic(expired, cue, -self.config.latency_cost)

    def _update_critic(self, rows: np.ndarray, cue: np.ndarray, reward: np.ndarray | float) -> None:
        actions = self._intent[rows].astype(np.int8)
        key = rows, cue[rows], actions
        current = self._quality[key]
        self._quality[key] = current + self.config.value_step * (reward - current)
        self._visits[key] += 1

    def act(self, obs: np.ndarray) -> np.ndarray:
        cue = obs[:, CUE].argmax(axis=1)
        if obs[0, TRIAL_START]:
            self._begin_trial(cue)
        self._ctx_voltage *= self.config.context_decay
        if self.strategy != "never_probe":
            self._ctx_voltage += obs[:, CTX]
            seen = np.flatnonzero(obs[:, CTX].sum(axis=1) == 1)
            if seen.size:
                context = obs[seen, CTX].argmax(axis=1)
                p = self._probability(seen, cue[seen])[np.arange(seen.size), context]
                self._target[seen] = self._select(p, seen)
                self._seeking[seen] = False
        self._seeking[obs[:, SCOUT_HIT] > 0] = False
        if self.learn:
            self._learn_outcome(obs, cue)
        direction = (obs[:, LEFT_ID].argmax(axis=1) != self._target).astype(np.int8)
        beacon_side = obs[:, SCOUT_SIDE].argmax(axis=1)
        direction = np.where(self._seeking, beacon_side, direction)
        active = obs[:, ACTIVE, None]
        self._motor_voltage *= self.config.motor_decay
        self._motor_voltage += 1.05 * active * _MOTOR_ONE_HOT[direction]
        spikes = self._motor_voltage >= 1.0
        self._motor_voltage -= spikes
        self._tick += 1
        return spikes
