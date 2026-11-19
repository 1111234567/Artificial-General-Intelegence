"""Lifetime activity-dependent control of credit in the frozen LIF network."""

from dataclasses import dataclass

import numpy as np

from .spike_policy import BatchSpikingPolicy, SpikeConfig, SpikingPolicy
from .world import ACTIVE, CONTACT_ID, REWARD_NEG, REWARD_POS, TRIAL_START


@dataclass(frozen=True)
class Consolidation:
    kind: str = "ordinary"
    covariance_rate: float = 0.1
    refresh: float = 0.0
    direction: str = "length"
    mixing: float = 1.0
    strength: float = 12.0
    importance_rate: float = 0.05

    def __post_init__(self):
        if self.kind not in {"ordinary", "decorrelated", "diagonal", "permuted", "protect"}:
            raise ValueError(f"unsupported consolidation {self.kind}")
        if self.direction not in {"length", "gain", "raw"}:
            raise ValueError(f"unsupported direction normalization {self.direction}")
        if min(self.covariance_rate, self.strength) < 0:
            raise ValueError("nonnegative gains required")
        for name in ("refresh", "mixing", "importance_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


class ConsolidatingLIF(SpikingPolicy):
    """Credit is filtered by ongoing activity in an added lateral synaptic bank.

    P starts as the identity and is updated by a common gain times an outer
    product of the local endpoint activities z = P x. It carries no cue,
    context, target label, trial register, or future receipt. The original
    readout still reads x; P changes only eligibility for reward-driven writes.
    """

    def __init__(self, config, mechanism, *, network_seed, action_seed):
        super().__init__(config, network_seed=network_seed, action_seed=action_seed)
        self.mechanism = mechanism
        covariant = mechanism.kind in {"decorrelated", "diagonal", "permuted"}
        self.lateral = np.eye(config.units, dtype=np.float32) if covariant else None
        self.permutation = np.random.default_rng(np.random.SeedSequence((network_seed, 48923))).permutation(config.units)
        self.importance = np.zeros_like(self.weights) if mechanism.kind == "protect" else None
        self.covariance_updates = 0
        self.write_norms = []
        self.changed_counts = []
        self.write_budget = None
        self.forced_identity = None

    def act(self, obs):
        capture = self.lateral is not None and self._clock + 1 == self.config.decision_tick
        capture = capture and not obs[TRIAL_START] and obs[ACTIVE] != 0
        center = self.mean_trace.copy() if capture else None
        pulses = super().step(obs)
        if capture:
            self._filter_eligibility(self.trace - center)
        return pulses

    step = act
    __call__ = act

    def _sample_identity(self):
        choice = super()._sample_identity()
        return choice if self.forced_identity is None else self.forced_identity

    def _filter_eligibility(self, feature):
        spec = self.mechanism
        lateral = self.lateral
        coactivity = lateral @ feature
        energy = float(feature @ coactivity)
        stability = 1 + spec.covariance_rate * max(energy, 0.0)
        if spec.kind == "diagonal":
            direction = lateral.diagonal() * feature
        elif spec.kind == "permuted":
            shuffled = np.empty_like(feature)
            shuffled[self.permutation] = feature
            direction = (lateral @ shuffled)[self.permutation]
        else:
            direction = coactivity
        if spec.direction == "length":
            gain = float(np.linalg.norm(feature)) / max(float(np.linalg.norm(direction)), 1e-8)
            direction = direction * gain
        elif spec.direction == "gain":
            direction = direction / stability
        direction = spec.mixing * direction + (1 - spec.mixing) * feature
        post = self.command_spikes.repeat(self.config.heads).astype(bool)
        self.eligibility[post] = direction * self.synapses[post]

        self.lateral -= (spec.covariance_rate / stability) * np.outer(coactivity, coactivity)
        if spec.refresh:
            self.lateral *= 1 - spec.refresh
            self.lateral.flat[::len(feature) + 1] += spec.refresh
        self.covariance_updates += 1

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
        credit = self.command_spikes if mode == "polarity-only" else contact * self.command_spikes
        if self.config.learning_rate == 0 or not credit.any():
            return
        post = credit.repeat(self.config.heads).astype(bool)
        error = reward - self.head_prediction[post] if self.config.use_residual else np.full(post.sum(), reward)
        local_eligibility = self.eligibility[post]
        change = self.config.learning_rate * error[:, None] * local_eligibility
        if self.importance is not None:
            stability = self.importance[post]
            rule = self.mechanism
            change = change / (1 + rule.strength * stability)
            certainty = 1 - np.minimum(np.square(error) / 4, 1)
            self.importance[post] = stability + rule.importance_rate * (
                certainty[:, None] * np.square(local_eligibility) - stability
            )
        before = self.weights[post]
        after = self._within_budget(before, change)
        self.weights[post] = after
        actual = after - before
        self.write_norms.append(float(np.linalg.norm(actual)))
        self.changed_counts.append(int(np.count_nonzero(actual)))
        self._updates += 1
        self._positive_updates += int(reward > 0)
        self._negative_updates += int(reward < 0)

    def _within_budget(self, before, change):
        if self.write_budget is None:
            return np.clip(before + change, -.65, .65)
        norm = float(np.linalg.norm(change))
        target = self.write_budget
        if target == 0 or norm == 0:
            return before.copy()
        gain = target / norm
        response = lambda g: np.clip(before + g * change, -.65, .65)
        after = response(gain)
        if float(np.linalg.norm(after - before)) >= target * (1 - 1e-5):
            return after
        # A clipping edge can spend less than the target. If other eligible
        # edges permit it, increase the common step to match the realized norm.
        low, high = gain, gain * 2
        limit = gain * 2**12
        while float(np.linalg.norm(response(high) - before)) < target and high < limit:
            low, high = high, high * 2
        for _ in range(18):
            middle = (low + high) / 2
            if float(np.linalg.norm(response(middle) - before)) < target:
                low = middle
            else:
                high = middle
        return response(high)

    def statistics(self):
        return {
            **super().statistics(),
            "l2_write_sum": float(sum(self.write_norms)),
            "weight_coordinates_changed": sum(self.changed_counts),
            "covariance_updates": self.covariance_updates,
            "lateral_coordinates_changed": int(np.count_nonzero(self.lateral - np.eye(len(self.lateral)))) if self.lateral is not None else 0,
        }


class BatchConsolidatingLIF(BatchSpikingPolicy):
    def __init__(self, mechanism=Consolidation(), config=SpikeConfig(use_residual=True), **seeds):
        super().__init__(config, **seeds)
        self.mechanism = mechanism

    def reset(self, n):
        seeds = self.action_seeds
        if seeds is None:
            seeds = np.random.SeedSequence(self.action_seed).spawn(n)
        if len(seeds) != n:
            raise ValueError("one independent action seed is required per body")
        self.agents = [
            ConsolidatingLIF(self.config, self.mechanism, network_seed=self.network_seed, action_seed=seed)
            for seed in seeds
        ]
