"""Offline diagnostic probes of frozen Scout states following physical sensing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import world
from .meta_scout_model import ScoutRawAgent, load_scout_model
from .meta_scout_train import make_batch, targets_from_public
from .scout_world import SCOUT_HIT, rollout_scout


class SampleStates(ScoutRawAgent):
    def __init__(self, model, trials, lag):
        super().__init__(model)
        self.trials = trials
        self.lag = lag

    def reset(self, n):
        super().reset(n)
        self.tick = 0
        self.after_beacon = np.full(n, -1, dtype=int)
        self.exists = np.zeros((n, self.trials), dtype=bool)
        self.reads = np.zeros((n, self.trials, self.model.config.values), dtype=np.float32)
        self.hidden = np.zeros((n, self.trials, self.model.config.hidden), dtype=np.float32)
        self.action = np.full((n, self.trials), -1, dtype=np.int8)
        self.endpoint_rank = self.action.copy()

    def act(self, obs):
        trial = self.tick // world.TICKS
        beginning = obs[:, world.TRIAL_START] > 0
        sensed = (obs[:, SCOUT_HIT] > 0) & (obs[:, world.CTX].sum(axis=1) == 1)
        self.after_beacon[beginning] = -1
        self.after_beacon[sensed] = self.tick
        take = (self.after_beacon == self.tick - self.lag) & (self.after_beacon >= 0) & (obs[:, world.ACTIVE] > 0)
        pulses = super().act(obs)
        if np.any(take):
            self.exists[take, trial] = True
            self.reads[take, trial] = self.last_step.read.cpu().numpy()[take]
            self.hidden[take, trial] = self.state.activity.cpu().numpy()[take]
            self.action[take, trial] = (pulses[:, 0] + 2 * pulses[:, 1])[take]
            if self.model.config.endpoint_value:
                with torch.no_grad():
                    values = self.model.endpoint_probe(torch.cat((self.state.activity, self.last_step.read), dim=1))
                self.endpoint_rank[take, trial] = values.argmax(dim=1).cpu().numpy()[take]
            elif self.model.config.indexed_outcomes:
                self.endpoint_rank[take, trial] = self.last_step.read.argmax(dim=1).cpu().numpy()[take]
        self.tick += 1
        return pulses


def read_dataset(model, root, lives, trials, lag, sensitive_count=None):
    tape = make_batch(root, lives, trials, sensitive_count=sensitive_count)
    agent = SampleStates(model, trials, lag)
    result = rollout_scout(agent, tape, record=True)
    labels = targets_from_public(result.obs, "cpu")
    previously_known = labels.known.numpy().all(axis=-1)
    late = np.arange(trials)[None, :] >= trials - min(32, trials)
    chosen = agent.exists & previously_known & result.sensitive_trial & late
    state = agent.hidden[chosen]
    read = agent.reads[chosen]
    rank = agent.endpoint_rank[chosen]
    return {"reads": read, "activity": state, "both": np.column_stack((read, state)),
            "target": tape.good_id[chosen], "context": tape.ctx[chosen],
            "direct_motor_correct": int((agent.action[chosen] == 1 + (tape.left_id[chosen] != tape.good_id[chosen])).sum()),
            "native_endpoint_readout": {"correct": int((rank == tape.good_id[chosen]).sum()), "scored": int((rank >= 0).sum())}}


def ridge_fit(train, target, development, labels):
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 0.01)
    x = (train - mean) / scale
    y = (development - mean) / scale
    x = np.column_stack((x, np.ones(len(x)))).astype(np.float64)
    y = np.column_stack((y, np.ones(len(y)))).astype(np.float64)
    penalty = np.eye(x.shape[-1], dtype=float) * (0.001 * len(x))
    ridge = np.linalg.solve(x.T @ x + penalty, x.T @ (2 * target.astype(float) - 1))
    predicted = y @ ridge > 0
    return {"correct": int((predicted == labels).sum()), "trials": len(labels)}


def audit(checkpoint, root, lives, trials, sensitive_count=None):
    torch.set_num_threads(2)
    model = load_scout_model(checkpoint)
    stages = {}
    for lag in (0, 2):
        first = read_dataset(model, root, lives, trials, lag, sensitive_count)
        second = read_dataset(model, root + lives, lives, trials, lag, sensitive_count)
        result = {"training_rows": len(first["target"]), "test_rows": len(second["target"]),
                  "direct_motor_correct": second["direct_motor_correct"],
                  "native_endpoint_readout": second["native_endpoint_readout"]}
        if len(first["target"]) and len(second["target"]):
            for label in ("target", "context"):
                result[label] = {feature: ridge_fit(first[feature], first[label], second[feature], second[label])
                                 for feature in ("reads", "activity", "both")}
        stages[str(lag)] = result
    return {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "train_seeds": [root, root + lives - 1], "test_seeds": [root + lives, root + 2 * lives - 1],
            "sensitive_count_override": sensitive_count,
            "condition": "sensitive late trials with tagged prior public receipts for both contexts and actual physical probe",
            "offline_probes_only": stages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4_348_800_000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--sensitive-count", type=int, choices=range(5))
    args = parser.parse_args()
    result = audit(args.checkpoint, args.seed, args.lives, args.trials, args.sensitive_count)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["offline_probes_only"]))
