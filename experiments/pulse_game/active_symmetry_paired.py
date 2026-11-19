"""Actual-return training on paired price and lifetime counterfactuals."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import active_symmetry_agent as p, active_symmetry_world as w
from .active_symmetry_train import subset


def prices(panel, rng):
    n = len(panel.seeds)
    row = np.arange(n)
    changed = rng.integers(2, size=n)
    low, high = (panel.costs.copy(), panel.costs.copy())
    low[row, changed], high[row, changed] = min(w.COSTS), max(w.COSTS)
    lower = rng.choice([4, 8], size=n).astype(np.int16)
    greater = np.where(lower == 8, 14, rng.choice([8, 14], size=n)).astype(np.int16)
    return [replace(panel, costs=c, horizon=h) for c in (low, high) for h in (lower, greater)]


def learn(train, develop, seed, *, steps, count):
    rng = np.random.default_rng(seed)
    # The first six channels are raw inputs. Ordinary linear layout is an
    # engineering choice; every coefficient remains unconstrained in sign.
    k = 6
    d = p.WIDTH
    output = np.zeros(d)
    output[:k] = .03 * rng.normal(size=k)
    critic_gram = np.eye(d)
    critic_signal = np.zeros(d)
    critic = np.zeros(d)
    one = np.zeros(k)
    two = np.zeros(k)
    best_score = -np.inf
    best_model = None
    history = []
    clock = perf_counter()
    for step in range(1, steps + 1):
        personal = subset(train, rng.choice(len(train.seeds), size=count, replace=False))
        active = prices(personal, rng)
        noise = rng.random((count, personal.ticks))
        experiences = [p.lifetime(scene, p.Gate("learned", output, noisy=True), (101, 137)[step % 2], toss=noise)
                       for scene in active]
        x, labels, coeff, value = [], [], [], []
        for actual in experiences:
            seen = actual["active"]
            x.append(actual["traces"][seen])
            labels.append(actual["intentions"][seen])
            coeff.append(actual["probabilities"][seen])
            value.append((actual["rewards"] - actual["paid"])[:, ::-1].cumsum(1)[:, ::-1][seen])
        fb, act, prob, future = tuple(map(np.concatenate, (x, labels, coeff, value)))
        error = future - fb @ critic
        entropy = -prob * (1 - prob) * np.log(np.maximum(prob, 1e-7) / np.maximum(1-prob, 1e-7))
        gradient = fb[:, :k].T @ ((act - prob) * error + .012 * entropy) / len(prob)
        one = .90 * one + .10 * gradient
        two = .99 * two + .01 * gradient**2
        rate = .07 / (1 + step / steps)
        output[:k] += rate * (one / (1 - .90**step)) / (np.sqrt(two / (1 - .99**step)) + 1e-4)
        critic_gram = .92 * critic_gram + fb.T @ fb / len(fb)
        critic_signal = .92 * critic_signal + fb.T @ future / len(fb)
        critic = np.linalg.solve(critic_gram + np.eye(d)*.015, critic_signal)
        if step == 1 or step % 12 == 0 or step == steps:
            states = [p.lifetime(develop, p.Gate("learned", output), b) for b in (101, 137)]
            outcome = float(np.mean([x["net"].mean() for x in states]))
            exchanges = float(np.mean([x["probe"].sum(1).mean() for x in states]))
            if outcome > best_score:
                best_model, best_score = output.copy(), outcome
            history.append({"step": step, "dev_net": outcome, "dev_probes": exchanges, "coefficients": output[:k].tolist()})
            print(f"{seed} step {step:3d} dev {outcome:+.3f} best {best_score:+.3f} price {output[3]:+.3f} "
                  f"remaining {output[5]:+.3f}, {perf_counter()-clock:.1f}s", flush=True)
    return {"selected_params": best_model.tolist(), "development_best": best_score, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--train-worlds", type=int, default=768)
    parser.add_argument("--dev-worlds", type=int, default=144)
    parser.add_argument("--base-batch", type=int, default=64)
    args = parser.parse_args()
    t = range(args.start, args.start + args.train_worlds)
    e = range(args.start + 20_000, args.start + 20_000 + args.dev_worlds)
    analysis = learn(w.build(t), w.build(e), args.seed, count=args.base_batch, steps=args.steps)
    root = Path(__file__).resolve().parent
    analysis.update(optimizer_seed=args.seed, training_worlds=list(t), development_worlds=list(e),
                    method="Same-body forked cheap/costly and short/long policies. Unconstrained six-channel gate, actor-critic from each actually realized payoff.",
                    code={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in (
                        "active_symmetry_paired.py", "active_symmetry_world.py", "active_symmetry_agent.py", "active_symmetry_train.py")})
    args.artifact.write_text(json.dumps(analysis, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
