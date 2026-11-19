"""Measure circuit topology and nonlinear discrimination at birth."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .spike_policy import SpikeConfig, SpikingPolicy
from .spike_run import BALANCED_CODES, build_policy
from .world import CTX, CUE, TRIAL_START, make_tape, rollout


def topology(agent):
    links = agent.recurrent != 0
    reached = links.copy()
    np.fill_diagonal(reached, True)
    for k in range(len(links)):
        reached |= reached[:, k, None] & reached[None, k, :]
    mutual = reached & reached.T
    _, sizes = np.unique(mutual, axis=0, return_counts=True)
    edges = links.astype(np.int32)
    squared = edges @ edges
    return {
        "reservoir_edges": int(edges.sum()),
        "strong_components": sorted(sizes.tolist(), reverse=True),
        "directed_cycles_2": int(np.trace(squared) // 2),
        "directed_cycles_3": int(np.trace(squared @ edges) // 3),
    }


class DecisionProbe:
    """Read the reservoir from outside the policy, without altering its inputs."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, n):
        self.policy.reset(n)
        self.frames = []
        self.keys = []
        self.tick = -1

    def act(self, obs):
        self.tick = 0 if obs[0, TRIAL_START] else self.tick + 1
        pulses = self.policy.act(obs)
        if self.tick == self.policy.config.decision_tick:
            self.frames.append(np.stack([agent.trace for agent in self.policy.agents]))
            self.keys.append(np.column_stack([obs[:, CTX].argmax(axis=1), obs[:, CUE].argmax(axis=1)]))
        return pulses

    def geometry(self):
        samples = np.concatenate(self.frames, axis=0).astype(np.float64)
        keys = np.concatenate(self.keys, axis=0)
        means = np.stack([samples[(keys[:, 0] == a) & (keys[:, 1] == b)].mean(axis=0)
                          for a in range(2) for b in range(4)]).reshape(2, 4, -1)
        grand = means.mean(axis=(0, 1))
        distance = means - grand
        interaction = means - means.mean(axis=0) - means.mean(axis=1, keepdims=True) + grand
        energy = np.sum(distance ** 2, axis=None)
        return {
            "samples": len(samples),
            "interaction_energy_fraction": float(np.sum(interaction ** 2) / energy),
            "decision_pattern_singular_values": np.linalg.svd(distance.reshape(8, -1), compute_uv=False).tolist(),
            "average_unit_variation_within_label": float(
                np.mean((samples - means[keys[:, 0], keys[:, 1]]) ** 2)),
            "average_unit_variation_between_labels": float(np.mean(distance ** 2)),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    tape = make_tape(range(94501, 94509), trials=64, map_codes=BALANCED_CODES)
    untouched = SpikeConfig(learning_rate=0)
    configs = (
        ("random", untouched),
        ("reservoir-recurrence-severed", replace(untouched, recurrence_gain=0)),
        ("engineered", replace(untouched, representation="engineered")),
        ("additive", replace(untouched, representation="additive")),
    )
    entries = []
    for network_seed in (3, 7):
        for name, config in configs:
            policy = build_policy(config, tape, network_seed)
            probe = DecisionProbe(policy)
            rollout(probe, tape, feedback="none")
            agent = SpikingPolicy(config, network_seed=network_seed)
            entries.append({"name": name, "birth_seed": network_seed,
                            **topology(agent), **probe.geometry()})
    args.artifact.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
