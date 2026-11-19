"""Separate post-training retention from unsupervised activity adaptation."""

import argparse
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_consolidate import Consolidation
from .spike_consolidate_run import CONFIRM_CODES, STOP, TrialSchedule, behavior, curriculum, row_scores
from .spike_policy import SpikeConfig
from .world import TRIAL_START, rollout


class AllWritesStopped(TrialSchedule):
    """Freeze associative weights, lateral statistics, and centering after learning."""

    def reset(self, n):
        super().reset(n)
        self.center = None

    def act(self, obs):
        if obs[0, TRIAL_START] and self.trial + 1 == self.quiet_from:
            self.center = [agent.mean_trace.copy() for agent in self.policy.agents]
            for agent in self.policy.agents:
                # The lattice only writes future eligibility. Its value does
                # not contribute to the feed-forward current at any tick.
                agent.lateral = None
        pulses = super().act(obs)
        if self.center is not None:
            for agent, value in zip(self.policy.agents, self.center, strict=True):
                agent.mean_trace[:] = value
        return pulses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    candidate = Consolidation(**frozen["mechanism"])
    config = SpikeConfig(**frozen["config"])
    tape = curriculum(list(range(100401, 100425)), codes=CONFIRM_CODES)
    results = []
    for seed in frozen["birth_seeds"]:
        for name, mechanism in (("candidate", candidate), ("ordinary", Consolidation())):
            prefix = None
            for kind, scope in (("outcomes-muted", TrialSchedule), ("all-writes-stopped", AllWritesStopped)):
                controller = behavior(tape, mechanism, config, seed)
                result = rollout(scope(controller), tape, record=True)
                if prefix is None:
                    prefix = result.pulses[:, :STOP].copy()
                else:
                    np.testing.assert_array_equal(result.pulses[:, :STOP], prefix)
                scores = row_scores(tape, result, stop=STOP)
                results.append({"method": name, "intervention": kind, "network_seed": seed, "scores": scores})
                print(f"{seed:2d} {name:9s} {kind:21s} old/new quiet "
                      f"{np.mean(scores['old_revisit']):.3f} / {np.mean(scores['new_revisit']):.3f}", flush=True)
    summary = {}
    for key in ("old_revisit", "new_revisit"):
        for name in ("candidate", "ordinary"):
            values = {arm: np.mean([row["scores"][key] for row in results
                                    if row["method"] == name and row["intervention"] == arm], axis=0)
                      for arm in ("outcomes-muted", "all-writes-stopped")}
            summary.setdefault(name, {})[key] = {
                **{arm: estimates(x) for arm, x in values.items()},
                "stop_minus_muted": estimates(values["all-writes-stopped"] - values["outcomes-muted"]),
            }
    args.artifact.write_text(json.dumps({"life_seeds": tape.seeds, "summary": summary, "results": results}) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
