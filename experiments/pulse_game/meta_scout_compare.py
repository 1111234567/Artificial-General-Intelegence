"""Paired native-world comparison of two frozen raw Scout policies."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import world
from .meta_scout_env import scaled_geometry
from .meta_scout_model import ScoutRawAgent, load_scout_model
from .meta_scout_train import action_ids, actual_body_prediction_targets, make_batch, metrics, sequence
from .scout_world import rollout_scout


def comparison(before, after, *, seed, lives, trials, sensitive_count=None, body_scale=None,
               body_beacon=0.22, hide_reward=False, forecast_lives=0):
    torch.set_num_threads(2)
    tape = make_batch(seed, lives, trials, sensitive_count=sensitive_count)
    if body_scale is not None:
        tape = scaled_geometry(tape, body_scale, body_beacon)

    def rollout(path):
        model = load_scout_model(path)
        policy = ScoutRawAgent(model)
        result = rollout_scout(policy, tape, feedback="none" if hide_reward else "true", record=bool(forecast_lives))
        info = torch.load(path, map_location="cpu", weights_only=True)
        timing = np.where(result.timed_out, world.MOVEMENT_TICKS, result.latency_ticks)
        nominal_return = result.reward + 0.1 * ~result.timed_out - 0.012 * timing - 0.03 * result.probed
        values = {
            "accuracy": result.correct.mean(axis=1), "late_accuracy": result.correct[:, -16:].mean(axis=1),
            "mean_ticks": timing.mean(axis=1), "probe_rate": result.probed.mean(axis=1),
            "contact_rate": (~result.timed_out).mean(axis=1), "probe_then_other_end": result.reversed.mean(axis=1),
            "nominal_return": nominal_return.mean(axis=1),
        }
        record = {"checkpoint": str(path), "step": info["step"], "metrics": metrics(result, tape),
                  "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        if forecast_lives:
            if not model.config.body_model:
                raise ValueError("Forecast auditing requires a body predictor")
            record["body_forecast"] = body_forecast(model, result.obs[:forecast_lives], result.pulses[:forecast_lives])
        return values, record

    a, baseline = rollout(before)
    b, changed = rollout(after)
    samples = np.random.default_rng(1173709).integers(lives, size=(3500, lives))
    effects = {}
    for name in a:
        gap = b[name] - a[name]
        effects[name] = {"before": float(a[name].mean()), "after": float(b[name].mean()),
                         "paired_delta": float(gap.mean()),
                         "life_bootstrap_ci95": np.quantile(gap[samples].mean(axis=1), [0.025, 0.975]).tolist()}
    return {"seeds": list(tape.seeds), "before": baseline, "after": changed, "effects": effects,
            "nominal_return_terms": {"contact_bonus": 0.1, "per_tick_cost": 0.012, "per_probe_cost": 0.03},
            "sensitive_count_override": sensitive_count, "body_scale": body_scale,
            "beacon_distance": tape.scout_distance, "hidden_reward_sign": hide_reward}


@torch.no_grad()
def body_forecast(model, observations, pulses):
    x = torch.as_tensor(observations)
    actions = torch.as_tensor(action_ids(pulses))
    *_, states = sequence(model, x)
    prediction = model.body_predictor(torch.cat((states, F.one_hot(actions, 3).to(states.dtype)), dim=-1))
    motion, events = actual_body_prediction_targets(x)
    active = x[..., world.ACTIVE] > 0
    position_now = x[..., [world.POSITION, world.VELOCITY]]
    rmse = ((prediction[active][:, :2] - motion[active]).square().mean(dim=0)).sqrt()
    copied = ((position_now[active] - motion[active]).square().mean(dim=0)).sqrt()
    event_scores = prediction[active][:, 2:].sigmoid()
    brier = ((event_scores - events[active]).square().mean(dim=0))
    return {"lives": len(x), "active_ticks": int(active.sum()), "next_x_v_rmse": rmse.tolist(),
            "copy_current_x_v_rmse": copied.tolist(), "scout_contact_brier": brier.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4_264_000_000)
    parser.add_argument("--lives", type=int, default=512)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--sensitive-count", type=int, choices=range(5))
    parser.add_argument("--body-scale", type=float)
    parser.add_argument("--body-beacon", type=float, default=0.22)
    parser.add_argument("--hide-reward", action="store_true")
    parser.add_argument("--forecast-lives", type=int, default=0)
    args = parser.parse_args()
    report = comparison(args.before, args.after, seed=args.seed, lives=args.lives, trials=args.trials,
                        sensitive_count=args.sensitive_count, body_scale=args.body_scale, body_beacon=args.body_beacon,
                        hide_reward=args.hide_reward, forecast_lives=args.forecast_lives)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["effects"]))
