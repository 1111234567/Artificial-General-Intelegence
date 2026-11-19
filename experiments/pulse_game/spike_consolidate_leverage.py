"""Discovery-only equalization of current-stimulus response after filtering."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .spike_analysis import estimates
from .spike_consolidate import BatchConsolidatingLIF, ConsolidatingLIF, Consolidation
from .spike_consolidate_partition import partition
from .spike_consolidate_run import STOP, TrialSchedule, behavior, curriculum, row_scores
from .spike_policy import SpikeConfig
from .world import rollout


class LeverageLIF(ConsolidatingLIF):
    """Give novel features an effective step even when projected at a large angle."""

    def __init__(self, config, mechanism, *, network_seed, action_seed, gain_cap):
        super().__init__(config, mechanism, network_seed=network_seed, action_seed=action_seed)
        self.gain_cap = gain_cap

    def _filter_eligibility(self, feature):
        lateral = self.lateral
        response = lateral @ feature
        leverage = max(float(feature @ response), 0.)
        length = float(np.linalg.norm(feature))
        norm_gain = length / max(float(np.linalg.norm(response)), 1e-8)
        leverage_gain = length**2 / max(leverage, 1e-8)
        gain = min(self.gain_cap * norm_gain, leverage_gain)
        post = self.command_spikes.repeat(self.config.heads).astype(bool)
        self.eligibility[post] = gain * response * self.synapses[post]
        rate = self.mechanism.covariance_rate
        self.lateral -= (rate / (1 + rate * leverage)) * np.outer(response, response)
        self.covariance_updates += 1


class LeverageBatch(BatchConsolidatingLIF):
    def __init__(self, mechanism, config, *, gain_cap, **seeds):
        super().__init__(mechanism, config, **seeds)
        self.gain_cap = gain_cap

    def reset(self, n):
        if len(self.action_seeds) != n:
            raise ValueError("one independent action seed is required per body")
        self.agents = [LeverageLIF(self.config, self.mechanism, network_seed=self.network_seed, action_seed=seed,
                                   gain_cap=self.gain_cap) for seed in self.action_seeds]


def discover():
    tape = curriculum(list(range(101101, 101117)), codes=(3, 12))
    cfg = SpikeConfig(use_residual=True)
    seeds = [np.random.SeedSequence((s, 0x434F4E53)) for s in tape.seeds]
    variants = [(f"leverage-{rate}-{cap}", rate, cap) for rate in (.055, .16) for cap in (1.25, 1.65, 2.2)]
    variants.extend((("ordinary", 0, 0), ("frozen-length", .055, 0)))
    rows = []
    for birth in (3, 7):
        for name, rate, cap in variants:
            mechanism = Consolidation(kind="decorrelated", covariance_rate=rate) if rate else Consolidation()
            controller = (LeverageBatch(mechanism, cfg, gain_cap=cap, network_seed=birth, action_seeds=seeds)
                          if cap else behavior(tape, mechanism, cfg, birth))
            result = rollout(TrialSchedule(controller), tape)
            score = row_scores(tape, result, stop=STOP)
            rows.append({"name": name, "birth_seed": birth, "mechanism": asdict(mechanism),
                         "gain_cap": cap, "scores": score})
            print(f"{birth} {name:25s} new train {np.mean(score['new_learning']):.3f}  "
                  f"old/new {np.mean(score['old_revisit']):.3f} / {np.mean(score['new_revisit']):.3f}", flush=True)
    return {"life_seeds": tape.seeds, "results": rows}


def confirm(protocol):
    low, high = protocol["life_range"]
    base = curriculum(list(range(low, high + 1)), codes=tuple(protocol["codes"]))
    cfg = SpikeConfig(use_residual=True)
    rule = Consolidation(kind="decorrelated", covariance_rate=protocol["rate"])
    seeds = [np.random.SeedSequence((s, 0x434F4E53)) for s in base.seeds]
    rows = []
    for order in protocol["orders"]:
        tape = partition(base, order)
        for birth in protocol["birth_seeds"]:
            factories = (
                ("fixed-length", behavior(tape, rule, cfg, birth)),
                ("current-leverage", LeverageBatch(rule, cfg, gain_cap=protocol["gain_cap"],
                                                    network_seed=birth, action_seeds=seeds)),
                ("ordinary", behavior(tape, Consolidation(), cfg, birth)),
            )
            for name, policy in factories:
                scores = row_scores(base, rollout(TrialSchedule(policy), tape), stop=STOP)
                rows.append({"name": name, "birth_seed": birth, "order": order, "scores": scores})
                print(f"{birth} {order} {name:18s} new train {np.mean(scores['new_learning']):.3f}  "
                      f"old/new {np.mean(scores['old_revisit']):.3f} / {np.mean(scores['new_revisit']):.3f}", flush=True)
    metrics = ("new_learning", "old_revisit", "new_revisit", "joint_revisit")
    values = {name: {metric: np.mean([row["scores"][metric] for row in rows if row["name"] == name], axis=0)
                     for metric in metrics} for name in ("ordinary", "fixed-length", "current-leverage")}
    summary = {name: {key: estimates(x) for key, x in arm.items()} for name, arm in values.items()}
    summary["leverage_minus_fixed"] = {metric: estimates(values["current-leverage"][metric] - values["fixed-length"][metric])
                                       for metric in metrics}
    return {"protocol": protocol, "life_seeds": base.seeds, "results": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--confirm", type=Path)
    args = parser.parse_args()
    report = confirm(json.loads(args.confirm.read_text(encoding="utf-8"))) if args.confirm else discover()
    args.artifact.write_text(json.dumps(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
