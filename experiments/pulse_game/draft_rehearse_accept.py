"""Fit a three-coefficient gate that can accept the native B draft unchanged."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import draft_rehearse as draft
from . import draft_rehearse_fit as address_fit
from . import interleave_revisit_train as storage
from . import noon_ticket as parents


DATA = address_fit.DATA
EVIDENCE = address_fit.EVIDENCE
PLAN = EVIDENCE / "accept-plan.md"
SEEDS = (62261039, 62261057)
DOMAIN = 53086339
STEPS = 256
WORLDS = len(address_fit.DESIGN)
POLICIES = ("context", "constant")
EVENTS = ("accepted_native", "a_payout", "b_payout")
REPLAY_READS = ("intercept", "retained_known_side_margin", "b_known_side_margin")
LR = 0.04
L2 = 1e-5
LIMIT = 8.0


def load_address_heads(path, *, require_final=True):
    with np.load(path, allow_pickle=False) as bundle:
        if (tuple(bundle["feature_names"]) != draft.FEATURES
                or tuple(bundle["replica_seeds"]) != address_fit.SEEDS):
            raise ValueError("Address head coefficients do not match the sealed task")
        if require_final and int(bundle["final_step"]) != address_fit.STEPS:
            raise ValueError("Expected final frozen address heads")
        if require_final and str(bundle["protocol_sha256"]) != address_fit.digest(address_fit.PROTOCOL):
            raise ValueError("Address heads do not match the sealed training protocol")
        result = {arm: torch.from_numpy(bundle[arm]) for arm in address_fit.ARMS}
    for arm, weights in result.items():
        if weights.shape != (len(SEEDS), len(draft.FEATURES)) or not torch.isfinite(weights).all():
            raise ValueError(f"Invalid address head array for {arm}")
    return result


def read_rehearsal(rehearsal):
    """Use the temporary motor probability for each physically archived target side."""
    def margin(trial):
        remembered_side = trial.archived.side * trial.archived.taste
        return 2 * (trial.action.motor_logit * remembered_side).sigmoid() - 1

    retention = margin(rehearsal.retained)
    return torch.stack((torch.ones_like(retention), retention, margin(rehearsal.current)), dim=1)


@torch.no_grad()
def one_step(weights, optimizer, prefix, world, first, selection, inputs, draws):
    probability = (inputs @ weights).sigmoid()
    accept = draws < probability
    state = draft.apply(prefix, selection, accept=accept)
    probe = draft.probe(prefix.rule, state, world, wait=address_fit.REVISIT_WAIT, order=first)
    a, b = probe.correct.float().unbind(1)
    reward = (a + b + a * b).reshape(len(address_fit.BIRTHS), -1)
    n = reward.shape[1]
    centered = ((reward - reward.mean(1, keepdim=True)) * n / (n - 1)).flatten()
    optimizer.zero_grad(set_to_none=True)
    weights.grad = -(centered[:, None] * (accept.float() - probability)[:, None] * inputs).mean(0) + L2 * weights
    optimizer.step()
    weights.clamp_(-LIMIT, LIMIT)
    return torch.cat((accept[:, None], probe.payout.to(torch.int8)), dim=1).numpy(), weights.numpy().copy()


def fit(replica, address_path, *, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_heads = load_address_heads(address_path, require_final=not smoke)
    graph, rule = parents.inherited("original")
    seed = SEEDS[replica]
    heads = {arm: weights[replica] for arm, weights in all_heads.items()}
    weights = {(arm, policy): torch.zeros(len(REPLAY_READS)) for arm in address_fit.ARMS for policy in POLICIES}
    optimizers = {key: torch.optim.Adam([values], lr=LR) for key, values in weights.items()}
    steps = 4 if smoke else STEPS
    lanes = len(address_fit.BIRTHS) * WORLDS
    actions = np.empty((len(address_fit.ARMS), len(POLICIES), steps, lanes, len(EVENTS)), dtype=np.int8)
    coefficient = np.empty((len(address_fit.ARMS), len(POLICIES), steps, len(REPLAY_READS)), dtype=np.float32)
    started = perf_counter()
    for step in range(1, steps + 1):
        world, first, address_draws = address_fit.training_world(seed, step, WORLDS)
        prefix = draft.prepare(graph, rule, world, seed=(DOMAIN, seed, step, 0))
        rng = np.random.default_rng([DOMAIN, seed, step, 1])
        draws = torch.from_numpy(rng.random(lanes, dtype=np.float32))
        for i, arm in enumerate(address_fit.ARMS):
            rehearsal = draft.rehearse(prefix, arm)
            selection = draft.choose(prefix, rehearsal, heads[arm], draw=address_draws[:, :prefix.candidates.valid.shape[1]])
            context = read_rehearsal(rehearsal)
            constant = context.clone()
            constant[:, 1:] = 0
            for j, (policy, inputs) in enumerate(zip(POLICIES, (context, constant), strict=True)):
                key = arm, policy
                actions[i, j, step - 1], coefficient[i, j, step - 1] = one_step(
                    weights[key], optimizers[key], prefix, world, first, selection, inputs, draws)
        if step % 64 == 0 or smoke:
            acceptance = actions[:, :, max(0, step - 64):step, :, 0].mean(axis=(2, 3))
            print(replica, step, acceptance.round(3).tolist(), f"{perf_counter() - started:.2f}s", flush=True)
    if not smoke:
        storage._write_raw(DATA / f"accept-fit-{replica}.npz", actions=actions, coefficient=coefficient,
                           step=np.arange(1, steps + 1), arms=address_fit.ARMS, policies=POLICIES, events=EVENTS,
                           feature_names=REPLAY_READS, seed=seed, replica=replica, domain=DOMAIN,
                           address_heads_sha256=address_fit.digest(address_path), plan_sha256=address_fit.digest(PLAN),
                           source_sha256=address_fit.digest(Path(__file__)))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, default=DATA / "fit-heads.npz")
    parser.add_argument("--replica", type=int, choices=range(len(SEEDS)), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    fit(args.replica, args.heads, smoke=args.smoke)


if __name__ == "__main__":
    main()
