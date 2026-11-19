"""Independent peer signals delivered at the focal body's physical receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pulse_game.scout_neural import ScoutNetPolicy
from experiments.pulse_game.scout_neural_train import load, measure, tape_for, used_world_seeds
from experiments.pulse_game.scout_search import _paired
from experiments.pulse_game.scout_world import ScoutRollout, ScoutTape, ScoutWorld, rollout_scout
from experiments.pulse_game.world import assert_unseen


class PeerReceiptWorld(ScoutWorld):
    """Replay independent values at the receiving body's probe or contact."""

    def __init__(self, tape: ScoutTape, *, peer_context: np.ndarray | None = None, peer_reward: np.ndarray | None = None) -> None:
        self.peer_context = peer_context
        self.peer_reward = peer_reward
        super().__init__(tape)

    def _shown_context(self, receipt: np.ndarray) -> np.ndarray:
        if self.peer_context is None:
            return super()._shown_context(receipt)
        return np.where(receipt, self.peer_context[:, self.trial_index], -1)

    def _shown_reward(self, receipt: np.ndarray) -> np.ndarray:
        if self.peer_reward is None:
            return super()._shown_reward(receipt)
        return self.peer_reward[:, self.trial_index] * receipt


def _run(model, seed: int, world: PeerReceiptWorld) -> ScoutRollout:
    network = ScoutNetPolicy(model, seed=seed)
    network.reset(world.tape.n)
    obs = world.observe()
    while obs is not None:
        obs = world.step(network.act(obs))
    return world.result()


def compare(model, subject: ScoutTape, peer: ScoutTape, *, policy_seed: int, peer_policy_seed: int) -> dict:
    assert_unseen(subject, peer)
    if subject.ctx.shape != peer.ctx.shape:
        raise ValueError("subject and peer must share the rollout shape")
    independent = rollout_scout(ScoutNetPolicy(model, seed=peer_policy_seed), peer)
    cases = {
        "own_sensor_own_reward": PeerReceiptWorld(subject),
        "peer_sensor_own_reward": PeerReceiptWorld(subject, peer_context=peer.ctx),
        "own_sensor_peer_reward": PeerReceiptWorld(subject, peer_reward=independent.reward),
        "peer_sensor_peer_reward": PeerReceiptWorld(subject, peer_context=peer.ctx, peer_reward=independent.reward),
    }
    reports = {}
    control = None
    for label, world in cases.items():
        result = _run(model, policy_seed, world)
        if control is None:
            control = result
        physical_probe = result.probed
        signed = result.observed_reward.sum(axis=-1)
        shown_ctx = result.observed_context.max(axis=-1)
        np.testing.assert_array_equal(shown_ctx >= 0, physical_probe)
        for row, trial in zip(*np.nonzero(signed)):
            assert result.observed_reward[row, trial, result.contact_tick[row, trial] + 1] == signed[row, trial]
        report = measure(result, subject)
        report["shown_context_matches_own"] = float((shown_ctx[physical_probe] == subject.ctx[physical_probe]).mean())
        report["shown_positive_rate"] = float((signed > 0).mean())
        report["shown_negative_rate"] = float((signed < 0).mean())
        report["paired_own_minus_case_accuracy"] = _paired(control.score - result.score)
        reports[label] = report
    return {"peer_endpoint_rate": independent.contact_rate, "peer_positive_rate": independent.accuracy, "conditions": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--subject-start", type=int, required=True)
    parser.add_argument("--peer-start", type=int, required=True)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=615200)
    parser.add_argument("--peer-policy-seed", type=int, default=615201)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, metadata = load(args.model)
    subject = tape_for(range(args.subject_start, args.subject_start + args.lives))
    peer = tape_for(range(args.peer_start, args.peer_start + args.lives))
    used = used_world_seeds(metadata)
    if used.intersection(subject.seeds) or used.intersection(peer.seeds):
        parser.error("subject and peer must be held out from training and development")
    results = compare(model, subject, peer, policy_seed=args.policy_seed, peer_policy_seed=args.peer_policy_seed)
    print(json.dumps({"subject": [subject.seeds[0], subject.seeds[-1]], "peer": [peer.seeds[0], peer.seeds[-1]], **results}, indent=2))


if __name__ == "__main__":
    main()
