"""Discovery-only test of low-dimensional activity decorrelation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .spike_consolidate import BatchConsolidatingLIF, ConsolidatingLIF, Consolidation
from .spike_consolidate_run import DISCOVERY_CODES, STOP, TrialSchedule, curriculum, row_scores
from .spike_policy import SpikeConfig
from .world import rollout


class SketchLIF(ConsolidatingLIF):
    def __init__(self, config, mechanism, *, network_seed, action_seed, channels, bypass=False):
        super().__init__(config, mechanism, network_seed=network_seed, action_seed=action_seed)
        if not 2 <= channels <= config.units:
            raise ValueError("the sketch needs between two and one pool per feature unit")
        self.lateral = np.eye(channels, dtype=np.float32)
        rng = np.random.default_rng(np.random.SeedSequence((network_seed, 7268943)))
        self.bucket = rng.permutation(config.units) % channels
        counts = np.bincount(self.bucket)
        signs = rng.choice(np.array([-1, 1], dtype=np.float32), config.units)
        self.sketch_weight = signs / np.sqrt(counts[self.bucket]).astype(np.float32)
        self.bypass = bypass

    def _filter_eligibility(self, feature):
        sketch = np.bincount(self.bucket, weights=self.sketch_weight * feature,
                             minlength=len(self.lateral)).astype(np.float32)
        lateral = self.lateral
        coactivity = lateral @ sketch
        rate = self.mechanism.covariance_rate
        stability = 1 + rate * max(0.0, float(sketch @ coactivity))
        if self.bypass:
            direction = feature + (coactivity - sketch)[self.bucket] * self.sketch_weight
        else:
            direction = coactivity[self.bucket] * self.sketch_weight
        gain = float(np.linalg.norm(feature)) / max(1e-8, float(np.linalg.norm(direction)))
        post = self.command_spikes.repeat(self.config.heads).astype(bool)
        self.eligibility[post] = gain * direction * self.synapses[post]
        self.lateral -= (rate / stability) * np.outer(coactivity, coactivity)
        self.covariance_updates += 1


class BatchSketchLIF(BatchConsolidatingLIF):
    def __init__(self, mechanism, config, *, network_seed, action_seeds, channels, bypass):
        super().__init__(mechanism, config, network_seed=network_seed, action_seeds=action_seeds)
        self.channels = channels
        self.bypass = bypass

    def reset(self, n):
        if len(self.action_seeds) != n:
            raise ValueError("one independent action seed is required per body")
        self.agents = [SketchLIF(self.config, self.mechanism, network_seed=self.network_seed, action_seed=seed,
                                 channels=self.channels, bypass=self.bypass) for seed in self.action_seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    cfg = SpikeConfig(use_residual=True)
    tape = curriculum(list(range(97101, 97113)), codes=DISCOVERY_CODES)
    action_seeds = [np.random.SeedSequence((s, 0x434F4E53)) for s in tape.seeds]
    spec = [(m, rate, bypass) for m in (12, 24, 48) for rate, bypass in ((.0, False), (.055, False), (.2, False), (.2, True))]
    rows = []
    for birth in (3, 7):
        for channels, rate, bypass in spec:
            mechanism = Consolidation(kind="decorrelated", covariance_rate=rate)
            controller = BatchSketchLIF(mechanism, cfg, network_seed=birth, action_seeds=action_seeds,
                                        channels=channels, bypass=bypass)
            score = row_scores(tape, rollout(TrialSchedule(controller), tape), stop=STOP)
            rows.append({"birth_seed": birth, "mechanism": asdict(mechanism), "channels": channels,
                         "bypass": bypass, "scores": score})
            print(f"{birth} channels {channels:2d}, rate {rate:5.3f}, bypass {bypass}: "
                  f"old/new {np.mean(score['old_revisit']):.3f} / {np.mean(score['new_revisit']):.3f}", flush=True)
    args.artifact.write_text(json.dumps({"life_seeds": tape.seeds, "results": rows}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
