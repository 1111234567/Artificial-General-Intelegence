"""Continuous sparse LIF controller with reward-modulated eligibility."""

from dataclasses import dataclass

import numpy as np

from .world import ACTIVE, CONTACT_ID, CTX, CUE, LEFT_ID, OBS_SIZE, REWARD_NEG, REWARD_POS, RIGHT_ID, TRIAL_START


@dataclass(frozen=True)
class SpikeConfig:
    """Fixed birth parameters. Only association synapses learn during life."""

    representation: str = "random"
    feedback: str = "true"
    units: int = 96
    heads: int = 8
    groups: int = 4
    decision_tick: int = 3
    learning_rate: float = 0.080
    exploration: float = 0.12
    cross_density: float = 0.04
    within_density: float = 0.17
    recurrence_gain: float = 1.0
    eligibility_decay: float = 0.97
    feature_leak: float = 0.53
    feature_trace_decay: float = 0.44
    use_residual: bool = False

    def __post_init__(self):
        if self.representation not in {"random", "engineered", "additive"}:
            raise ValueError(f"unsupported representation: {self.representation}")
        if self.feedback not in {"true", "polarity-only", "contact-only", "inverted"}:
            raise ValueError(f"unsupported feedback mode: {self.feedback}")
        if self.units < 16 or self.heads < 1 or not 1 <= self.groups <= self.units:
            raise ValueError("invalid population size")
        if not 1 <= self.decision_tick <= 7:
            raise ValueError("decision tick must leave time for movement")
        for name in ("exploration", "within_density", "cross_density", "eligibility_decay",
                     "feature_leak", "feature_trace_decay"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.learning_rate < 0 or self.recurrence_gain < 0:
            raise ValueError("gains must be nonnegative")


class SpikingPolicy:
    """One life, from birth to the final receipt, with no per-trial brain reset.

    The task-facing adapter knows the public observation and motor formats. It
    routes the chosen object identity to the visible side. The recurrent network
    learns which identity to prefer from outcomes of its own movements. Birth
    weights encode no answer. In the random condition there are no hand-wired
    conjunction cells or cue/context lookup.
    """

    def __init__(self, config=SpikeConfig(), *, network_seed=0, action_seed=None):
        self.config = config
        rng = np.random.default_rng(network_seed)
        if action_seed is None:
            action_seed = np.random.SeedSequence((int(network_seed), 1))
        self.action_rng = np.random.default_rng(action_seed)
        n = config.units
        count = 2 * config.heads
        groups = rng.permutation(n) % config.groups
        within = groups[:, None] == groups
        edge_density = np.where(within, config.within_density, config.cross_density)
        self.recurrent_mask = rng.random((n, n)) < edge_density
        np.fill_diagonal(self.recurrent_mask, False)
        # Inhibitory neurons project negative recurrent currents throughout.
        signs = np.where(rng.random(n) < 0.24, -1.9, 1.0).astype(np.float32)
        self.recurrent = rng.uniform(0.012, 0.075, (n, n)).astype(np.float32)
        self.recurrent *= self.recurrent_mask * signs * config.recurrence_gain
        self.command_feedback = rng.uniform(-0.045, 0.045, (n, 2)).astype(np.float32)
        self.command_feedback *= (rng.random((n, 2)) < 0.24) * config.recurrence_gain
        self.head_feedback = rng.uniform(-0.025, 0.025, (n, count)).astype(np.float32)
        self.head_feedback *= (rng.random((n, count)) < 0.10) * config.recurrence_gain

        # The outcome signal reaches the plastic synapses through their local
        # eligibility. It does not enter the reservoir's direct sensory drive.
        self.projection = rng.normal(0, 0.035, (n, OBS_SIZE)).astype(np.float32)
        self.projection *= rng.random((n, OBS_SIZE)) < 0.25
        self.projection[:, REWARD_POS:CONTACT_ID.stop] = 0
        task_sensors = slice(CTX.start, CUE.stop)
        self.projection[:, task_sensors] = rng.normal(0.25, 0.50, (n, CUE.stop)).astype(np.float32)
        self.projection[:, task_sensors] *= rng.random((n, CUE.stop)) < 0.62
        if config.representation == "engineered":
            # Explicit, fixed coincidence detectors for cue × context in
            # this positive control. No answer is attached to these cells.
            self.coincidence_index = np.arange(n) % 8
            self.projection[:, task_sensors] = 0
        elif config.representation == "additive":
            # Feature units receive only one of the two task signal banks.
            self.projection[::2, CTX] = 0
            self.projection[1::2, CUE] = 0
            self.recurrent[::2, 1::2] = 0
            self.recurrent[1::2, ::2] = 0
            self.head_feedback.fill(0)
            self.command_feedback.fill(0)

        self.weights = np.zeros((count, n), dtype=np.float32)
        self.synapses = rng.random(self.weights.shape) < 0.68
        self.v = np.zeros(n, dtype=np.float32)
        self.spikes = np.zeros(n, dtype=np.float32)
        self.trace = np.zeros(n, dtype=np.float32)
        self.mean_trace = np.full(n, 0.20, dtype=np.float32)
        self.feature_threshold = rng.uniform(0.90, 1.10, n).astype(np.float32)
        self.feature_bias = rng.uniform(0.035, 0.105, n).astype(np.float32)
        self.head_threshold = np.tile(np.linspace(0.69, 1.31, config.heads, dtype=np.float32), 2)
        self.head_v = np.zeros(count, dtype=np.float32)
        self.head_spikes = np.zeros(count, dtype=np.float32)
        self.head_trace = np.zeros(count, dtype=np.float32)
        self.head_prediction = np.zeros(count, dtype=np.float32)
        self.command_v = np.zeros(2, dtype=np.float32)
        self.command_spikes = np.zeros(2, dtype=np.float32)
        self.eligibility = np.zeros_like(self.weights)
        self._clock = -1
        self._step_count = 0
        self._reservoir_spikes = 0
        self._head_spike_count = 0
        self._command_spike_count = 0
        self._motor_pulses = 0
        self._feedback_count = 0
        self._updates = 0
        self._actual_identity_matches = 0
        self._actual_contacts = 0
        self._positive_updates = 0
        self._negative_updates = 0

    def step(self, observation):
        """Read the public 20-channel observation and emit [left, right] pulses."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (OBS_SIZE,):
            raise ValueError("a policy instance accepts one 20-channel observation per tick")
        self._clock = 0 if obs[TRIAL_START] else self._clock + 1
        self._step_count += 1
        self.eligibility *= self.config.eligibility_decay
        self._apply_feedback(obs)

        self.v *= self.config.feature_leak
        self.v += self.feature_bias + self.projection @ obs
        if self.config.representation == "engineered":
            coincidence = (obs[CTX, None] * obs[None, CUE]).ravel()
            self.v += 1.25 * coincidence[self.coincidence_index]
        self.v += self.recurrent @ self.spikes
        self.v += self.command_feedback @ self.command_spikes
        self.v += self.head_feedback @ self.head_spikes
        np.greater_equal(self.v, self.feature_threshold, out=self.spikes)
        self.v[self.spikes != 0] = 0.0
        self.trace *= self.config.feature_trace_decay
        self.trace += (1 - self.config.feature_trace_decay) * self.spikes
        centered = self.trace - self.mean_trace

        self.head_v *= 0.51
        head_current = self.weights @ centered
        self.head_v += 0.47 + head_current
        np.maximum(self.head_v, -1, out=self.head_v)
        np.greater_equal(self.head_v, self.head_threshold, out=self.head_spikes)
        self.head_v[self.head_spikes != 0] = 0.0
        self.head_trace *= 0.59
        self.head_trace += 0.41 * self.head_spikes

        # Command neurons form a two-unit winner-take-all latch. A new-trial
        # pulse and the absence of active movement inhibit them via input.
        self.command_v *= 0.50
        self.command_v += 1.16 * self.command_spikes - 0.31 * self.command_spikes[::-1]
        launch = self._clock == self.config.decision_tick and obs[ACTIVE] != 0
        if launch:
            choice = self._sample_identity()
            self.command_v[choice] += 2.4
        if obs[TRIAL_START] or not obs[ACTIVE]:
            self.command_v -= 2.4
        np.maximum(self.command_v, -1.0, out=self.command_v)
        np.greater_equal(self.command_v, 1.0, out=self.command_spikes)
        self.command_v[self.command_spikes != 0] = 0.0
        if launch:
            # A pulse in the chosen identity command population retroactively
            # tags the attached posts, with each synapse's own pre trace.
            post_gate = self.command_spikes.repeat(self.config.heads)
            self.eligibility *= 1 - post_gate[:, None]
            self.eligibility += post_gate[:, None] * centered * self.synapses
            if self.config.use_residual:
                self.head_prediction = np.tanh(2.5 * head_current)
            self.mean_trace += 0.018 * centered

        motor = obs[LEFT_ID.start:RIGHT_ID.stop].reshape(2, 2) @ self.command_spikes > 0.5
        self._reservoir_spikes += int(self.spikes.sum())
        self._head_spike_count += int(self.head_spikes.sum())
        self._command_spike_count += int(self.command_spikes.sum())
        self._motor_pulses += int(motor.sum())
        return motor

    act = step
    __call__ = step

    def _sample_identity(self):
        rates = self.head_trace.reshape(2, self.config.heads).mean(axis=1)
        explore = self.action_rng.random() < self.config.exploration
        alternate = int(self.action_rng.integers(2))
        if explore or rates[0] == rates[1]:
            return alternate
        return int(rates[1] > rates[0])

    def _apply_feedback(self, obs):
        reward = float(obs[REWARD_POS] - obs[REWARD_NEG])
        contact = obs[CONTACT_ID]
        reached = bool(contact.any())
        if reached:
            self._actual_contacts += 1
            self._actual_identity_matches += int(contact @ self.command_spikes == 1)
        mode = self.config.feedback
        if mode == "contact-only":
            reward = float(reached)
        elif mode == "inverted":
            reward *= -1
        if reward == 0:
            return
        self._feedback_count += 1
        if mode == "polarity-only":
            # Identity credit is already implicit in the command spike.
            credit = self.command_spikes
        else:
            credit = contact * self.command_spikes
        if self.config.learning_rate == 0 or not credit.any():
            return
        post_credit = credit.repeat(self.config.heads)
        error = reward - self.head_prediction if self.config.use_residual else reward
        change = self.config.learning_rate * (post_credit * error)[:, None] * self.eligibility
        self.weights += change
        np.clip(self.weights, -0.65, 0.65, out=self.weights)
        self._updates += 1
        self._positive_updates += int(reward > 0)
        self._negative_updates += int(reward < 0)

    def statistics(self):
        """Aggregate neural activity and effective edge/weight counts."""
        return {
            "ticks": self._step_count,
            "reservoir_units": self.config.units,
            "association_units": 2 * self.config.heads,
            "reservoir_edges": int(np.count_nonzero(self.recurrent)),
            "head_return_edges": int(np.count_nonzero(self.head_feedback)),
            "command_return_edges": int(np.count_nonzero(self.command_feedback)),
            "plastic_edges": int(self.synapses.sum()),
            "reservoir_spikes": self._reservoir_spikes,
            "association_spikes": self._head_spike_count,
            "command_spikes": self._command_spike_count,
            "motor_pulses": self._motor_pulses,
            "contacts_visible": self._actual_contacts,
            "contact_command_matches": self._actual_identity_matches,
            "nonzero_feedback": self._feedback_count,
            "updates": self._updates,
            "positive_updates": self._positive_updates,
            "negative_updates": self._negative_updates,
            "changed_edges": int(np.count_nonzero(self.weights)),
            "weight_l2": float(np.linalg.norm(self.weights)),
            "weight_max_abs": float(np.abs(self.weights).max()),
        }


class BatchSpikingPolicy:
    """Rollout adapter. Same birth topology, independent action noise per body."""

    def __init__(self, config=SpikeConfig(), *, network_seed=0, action_seed=0, action_seeds=None):
        self.config = config
        self.network_seed = network_seed
        self.action_seed = action_seed
        self.action_seeds = None if action_seeds is None else tuple(action_seeds)
        self.agents = []

    def reset(self, n):
        seeds = self.action_seeds
        if seeds is None:
            seeds = np.random.SeedSequence(self.action_seed).spawn(n)
        if len(seeds) != n:
            raise ValueError("one independent action seed is required per body")
        self.agents = [
            SpikingPolicy(self.config, network_seed=self.network_seed, action_seed=seed)
            for seed in seeds
        ]

    def act(self, obs):
        if len(obs) != len(self.agents):
            raise ValueError("one observation is required per body")
        return np.stack([agent.act(row) for agent, row in zip(self.agents, obs, strict=True)])
