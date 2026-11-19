"""Match paired bodies exactly during visits before independent choice probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from experiments.pulse_game.twostep_distinct import make_distinct_tape
from experiments.pulse_game.twostep_eval import _episode_stats
from experiments.pulse_game.twostep_neural import TwoStepNetPolicy
from experiments.pulse_game.twostep_neural_train import load
from experiments.pulse_game.twostep_world import TwoStepTape, TwoStepWorld, make_tape, set_reversal
from experiments.pulse_game.world import REWARD_NEG, REWARD_POS, Policy


def matched(new_policy: Callable[[], Policy], source: TwoStepTape, *, seed: int) -> dict:
    """Each pair gets its own shared visit motor sequence from an assigned twin.

    Which twin physically drives a given pair's visit body is sampled before
    feedback. Both twins retain their own observed outcomes and state. The
    first subsequent entrance and fruit are executed by each twin separately.
    """
    stay_tape, flip_tape = set_reversal(source, False), set_reversal(source, True)
    stay, reverse = TwoStepWorld(stay_tape, record=True), TwoStepWorld(flip_tape, record=True)
    a, b = new_policy(), new_policy()
    a.reset(source.n)
    b.reset(source.n)
    use_a = np.random.default_rng(seed).integers(2, size=source.n).astype(bool)
    oa, ob = stay.observe(), reverse.observe()
    while oa is not None:
        pa, pb = a.act(oa), b.act(ob)
        if stay.is_visit:
            shared = np.where(use_a[:, None], pa, pb)
            pa = pb = shared
        oa, ob = stay.step(pa), reverse.step(pb)

    result_a, result_b = stay.result(), reverse.result()
    visit = slice(source.acquisition, source.probe)
    public = [i for i in range(result_a.obs.shape[-1]) if i not in (REWARD_POS, REWARD_NEG)]
    np.testing.assert_array_equal(result_a.obs[:, visit][..., public], result_b.obs[:, visit][..., public])
    np.testing.assert_array_equal(result_a.pulses[:, visit], result_b.pulses[:, visit])
    np.testing.assert_array_equal(result_a.obs[:, source.probe, 0], result_b.obs[:, source.probe, 0])
    np.testing.assert_array_equal(result_a.obs[:, :source.acquisition], result_b.obs[:, :source.acquisition])
    directionally_correct = result_a.effective_entry[:, source.probe] & result_b.effective_entry[:, source.probe]
    return {
        "same_actual_visit_body_and_public_inputs_barring_sign": True,
        "same_first_probe_observation": True,
        "pairs_driven_by_stay_twin": int(use_a.sum()),
        "revaluation_signs_changed": float((result_a.shown[:, visit].sum(axis=2) != result_b.shown[:, visit].sum(axis=2)).mean()),
        "stay": _episode_stats(stay_tape, result_a),
        "reversed": _episode_stats(flip_tape, result_b),
        "both_first_probes_optimal": float(directionally_correct.mean()),
        "no_probe_reward_observed": bool(not np.any(result_a.shown[:, source.probe]) and not np.any(result_b.shown[:, source.probe])),
    }


def main() -> None:
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--model", type=Path, required=True)
    cmd.add_argument("--seed-start", type=int, required=True)
    cmd.add_argument("--lives", type=int, default=256)
    cmd.add_argument("--policy-seed", type=int, default=571226)
    args = cmd.parse_args()
    torch.set_num_threads(2)
    model, meta = load(args.model)
    known = set(range(*meta["development"]))
    for start, stop in meta["world_ranges"]:
        known.update(range(start, stop))
    seeds = range(args.seed_start, args.seed_start + args.lives)
    if known.intersection(seeds):
        cmd.error("comparison lives must be held out")
    distinct = meta.get("distinct_fruit", False)
    build = make_distinct_tape if distinct else make_tape
    report = {}
    for rare in (0.0, 0.2):
        tape = build(seeds, rare_probability=rare, distance_jitter=0.22, wind_scale=0.12)
        report[f"rare_{rare}"] = matched(
            lambda: TwoStepNetPolicy(model, seed=args.policy_seed, mean_visits=distinct), tape, seed=args.policy_seed + 497,
        )
    print(json.dumps({"world_seeds": [seeds.start, seeds.stop - 1], "policy_seed": args.policy_seed, "conditions": report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
