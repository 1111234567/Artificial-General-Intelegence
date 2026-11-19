"""Split learned weights from residual LIF state at a no-feedback checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_policy import SpikeConfig
from .spike_run import BALANCED_CODES, FeedbackMask, build_policy
from .world import TRIAL_START, make_tape, rollout


class SubstrateIntervention(FeedbackMask):
    """Apply one intervention only at the start of the same suffix in each arm."""

    def __init__(self, policy, checkpoint, intervention):
        super().__init__(policy, checkpoint, ())
        self.intervention = intervention

    def act(self, obs):
        if obs[0, TRIAL_START] and self.trial + 1 == self.quiet_from:
            for agent in self.agents:
                if self.intervention in {"weights-only", "weights-and-centering"}:
                    for name in ("v", "spikes", "trace", "head_v", "head_spikes", "head_trace",
                                 "command_v", "command_spikes", "eligibility", "head_prediction"):
                        getattr(agent, name).fill(0)
                    if self.intervention == "weights-only":
                        agent.mean_trace.fill(.20)
                elif self.intervention == "clear-weights":
                    agent.weights.fill(0)
        return super().act(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    tape = make_tape(range(96001, 96025), trials=144, map_codes=BALANCED_CODES)
    cfg = SpikeConfig(use_residual=True)
    checkpoint = 96
    results = {name: [] for name in ("reference", "weights-only", "weights-and-centering", "clear-weights")}
    for seed in (3, 7):
        reference = None
        for name in results:
            behavior = build_policy(cfg, tape, seed)
            scope = SubstrateIntervention(behavior, checkpoint, name)
            result = rollout(scope, tape)
            prefix = result.correct[:, :checkpoint]
            if reference is None:
                reference = prefix.copy()
            else:
                np.testing.assert_array_equal(prefix, reference)
            scores = result.correct[:, checkpoint:]
            results[name].append({"birth_seed": seed, "suffix_correct": scores.tolist(),
                                  "prefix_accuracy": float(prefix.mean())})
            print(f"{name} birth {seed}: prefix {prefix.mean():.3f}, suffix {scores.mean():.3f}", flush=True)
    matrices = {
        name: np.mean([row["suffix_correct"] for row in rows], axis=(0, 2))
        for name, rows in results.items()
    }
    summary = {name: {"suffix_accuracy": estimates(values),
                      "suffix_minus_reference": estimates(values - matrices["reference"])}
               for name, values in matrices.items()}
    artifact = {"life_seeds": tape.seeds, "quiet_and_intervene_from_trial": checkpoint,
                "results": results, "summary": summary}
    args.artifact.write_text(json.dumps(artifact) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
