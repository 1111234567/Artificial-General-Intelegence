"""Checkpoint interventions on lateral activity memory before the new block."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_consolidate import Consolidation
from .spike_consolidate_run import ACQUIRE, CONFIRM_CODES, STOP, TrialSchedule, behavior, curriculum, row_scores
from .spike_policy import SpikeConfig
from .world import TRIAL_START, rollout


class LateralCheckpoint(TrialSchedule):
    def __init__(self, policy, intervention):
        super().__init__(policy, quiet_from=STOP)
        self.intervention = intervention

    def reset(self, n):
        super().reset(n)
        self.matrices = {}

    def act(self, obs):
        if obs[0, TRIAL_START] and self.trial + 1 in (ACQUIRE, STOP):
            matrix = np.stack([agent.lateral.copy() for agent in self.policy.agents])
            self.matrices[self.trial + 1] = np.linalg.eigvalsh(matrix).tolist()
        if obs[0, TRIAL_START] and self.trial + 1 == ACQUIRE:
            agents = self.policy.agents
            for agent in agents:
                if self.intervention == "clear-old-lateral":
                    agent.lateral[:] = np.eye(agent.config.units, dtype=np.float32)
                elif self.intervention == "permute-old-lateral":
                    agent.lateral[:] = agent.lateral[np.ix_(agent.permutation, agent.permutation)]
        return super().act(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    mechanism = Consolidation(**frozen["mechanism"])
    config = SpikeConfig(**frozen["config"])
    tape = curriculum(list(range(100101, 100125)), codes=CONFIRM_CODES)
    results = []
    conditions = ("untouched", "clear-old-lateral", "permute-old-lateral")
    for seed in frozen["birth_seeds"]:
        prefix = None
        for name in conditions:
            policy = LateralCheckpoint(behavior(tape, mechanism, config, seed), name)
            result = rollout(policy, tape, record=True)
            if prefix is None:
                prefix = result.pulses[:, :ACQUIRE].copy()
            else:
                np.testing.assert_array_equal(result.pulses[:, :ACQUIRE], prefix)
            scores = row_scores(tape, result, stop=STOP)
            results.append({"name": name, "birth_seed": seed, "scores": scores,
                            "lateral_eigenvalues": policy.matrices,
                            "correct": result.correct.tolist(), "contact_id": result.contact_id.tolist()})
            print(f"{seed} {name:20s} old/new quiet {np.mean(scores['old_revisit']):.3f} "
                  f"/ {np.mean(scores['new_revisit']):.3f}", flush=True)
    values = {name: {metric: np.mean([row["scores"][metric] for row in results if row["name"] == name], axis=0)
                     for metric in ("old_revisit", "new_revisit", "new_learning")}
              for name in conditions}
    summary = {name: {metric: {"estimate": estimates(value),
                               "minus_untouched": estimates(value - values["untouched"][metric])}
                      for metric, value in scores.items()}
               for name, scores in values.items()}
    report = {"life_seeds": tape.seeds, "mechanism": asdict(mechanism), "config": asdict(config),
              "results": results, "summary": summary}
    args.artifact.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
