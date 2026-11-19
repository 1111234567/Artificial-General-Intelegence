"""Scout sensitivity to sensor distance, endpoint displacement and motion noise."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import world
from .meta_scout_env import RangeTape
from .meta_scout_model import ScoutRawAgent, load_scout_model
from .meta_scout_train import metrics
from .scout_world import ScoutOraclePolicy, make_scout_tape, rollout_scout


SETTINGS = (
    (0.35, 0.22, 0.12), (0.35, 0.00, 0.00), (0.35, 0.38, 0.12),
    (0.21, 0.22, 0.06), (0.48, 0.22, 0.12), (0.55, 0.38, 0.12),
)


def geometry_tape(seeds, trials, sensor, distance, wind):
    if 1 - distance <= sensor or wind >= sensor * (1 - world.VELOCITY_DECAY):
        raise ValueError("The beacon must lie before both endpoints and be unreachable from a gust alone")
    base = make_scout_tape(seeds, trials, distance_jitter=distance, wind_scale=wind)
    base = world.swap_sides(base, first_trial=trials // 2)
    return RangeTape(**vars(base), range_override=sensor)


def audit(checkpoint, seed, lives, trials):
    torch.set_num_threads(2)
    model = load_scout_model(checkpoint)
    seeds = range(seed, seed + lives)
    rows = []
    for sensor, distance, wind in SETTINGS:
        tape = geometry_tape(seeds, trials, sensor, distance, wind)
        learned = rollout_scout(ScoutRawAgent(model), tape)
        known_map = rollout_scout(ScoutOraclePolicy(tape), tape)
        rows.append({
            "beacon_distance": sensor, "endpoint_jitter": distance, "wind_scale": wind,
            "learned": metrics(learned, tape), "known_map_physical_reference": metrics(known_map, tape),
        })
    return {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "seeds": list(seeds), "trials": trials, "conditions": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4_348_300_000)
    parser.add_argument("--lives", type=int, default=256)
    parser.add_argument("--trials", type=int, default=64)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.seed, args.lives, args.trials)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["conditions"]:
        print(row["beacon_distance"], row["endpoint_jitter"], row["wind_scale"],
              row["learned"]["accuracy"], row["learned"]["late_accuracy"], row["known_map_physical_reference"]["accuracy"])
