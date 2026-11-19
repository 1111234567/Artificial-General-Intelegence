"""Hold graph parameters or carried activity fixed after a single proposed write."""

import argparse
from pathlib import Path

import numpy as np
import torch

from . import edit_receipt_graph as native
from . import self_param_writer as fw


def heldout(birth, pairs, length):
    rng = np.random.default_rng([396175, birth])
    odors = rng.normal(size=(pairs, 2, 3))
    odors *= 1.35 / np.linalg.norm(odors, axis=-1, keepdims=True)
    cues = rng.integers(2, size=(pairs, length))
    flip = rng.choice([-1, 1], size=cues.shape)
    pref = rng.choice([-1, 1], size=(pairs, 2))
    gold = pref[np.arange(pairs)[:, None], cues] * flip
    slip = np.where(rng.random(cues.shape) < .15, -1, 1)
    vector = odors[np.arange(pairs)[:, None], cues] * flip[..., None]
    draws = rng.random((pairs, length, 4), dtype=np.float32)
    def twins(x):
        return torch.from_numpy(np.repeat(x, 2, axis=0)).float()
    gold = twins(gold)
    gold[1::2] *= -1
    return twins(vector), gold, gold * twins(slip), twins(draws)


@torch.no_grad()
def audit(birth_graph, birth, arm, pairs=48, horizon=24):
    """00 and 11 are natural worlds. 10 and 01 exchange only the post-commit h."""
    times = (3, 9, 15)
    x, truth, outcomes, draws = heldout(birth, pairs, times[-1] + horizon + 1)
    count = len(x)
    sensing = "zero" if arm == "baseline" else "own"
    state = native.start(birth_graph, count)
    records = {k: [] for k in ("applied", "op", "address", "accept", "prediction",
                               "correct", "choice", "h_diff", "logit")}
    for t in range(times[-1] + 1):
        prop = native.propose(state, x[:, t], -x[:, t], outcomes[:, t], draws[:, t],
                              sensing=sensing)
        live = native.commit(state, prop, draws[:, t], sensing=sensing)
        if t in times:
            forced = [native.commit(state, prop, draws[:, t], sensing=sensing,
                                    accept_override=torch.full((count,), flag, dtype=torch.bool))
                      for flag in (False, True)]
            s0, s1 = (result.state for result in forced)
            branches = ((s0.graph, s0.hidden), (s0.graph, s1.hidden),
                        (s1.graph, s0.hidden), (s1.graph, s1.hidden))
            fused = native.State(fw.Graph(*(torch.cat([getattr(g, k) for g, _ in branches])
                                            for k in ("weights", "bias", "present"))),
                                 torch.cat([h for _, h in branches]))
            choices, logits, correct = [], [], []
            for j in range(t + 1, t + horizon + 1):
                left = x[:, j].repeat(4, 1)
                shared = draws[:, j].repeat(4, 1)
                p = native.propose(fused, left, -left, outcomes[:, j].repeat(4), shared,
                                   sensing=sensing)
                fused = native.commit(fused, p, shared, sensing=sensing).state
                choice = p.side.reshape(4, count)
                choices.append(choice)
                logits.append(p.motor_logit.reshape(4, count))
                correct.append(choice == truth[:, j])
            applied = forced[1].receipt.status == fw.Status.APPLIED
            path = torch.stack(choices, -1)
            motor = torch.stack(logits, -1)
            # Without a physical change, exchanging only parameters preserves the future.
            if not torch.equal(path[2, ~applied], path[0, ~applied]) or not torch.equal(
                    path[3, ~applied], path[1, ~applied]):
                raise AssertionError("A branch with identical graph and activity diverged")
            values = (applied, prop.command.op, prop.command.address, live.accepted, prop.benefit,
                      torch.stack(correct, -1), path.to(torch.int8),
                      (s1.hidden - s0.hidden).abs().amax(1), motor)
            for key, value in zip(records, values, strict=True):
                records[key].append(value.cpu().numpy())
        state = live.state
    return {k: np.stack(v) for k, v in records.items()}


def main():
    from .edit_receipt_train import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--births", type=int, required=True, nargs="+")
    p.add_argument("--arms", required=True, nargs="+")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--pairs", type=int, default=48)
    a = p.parse_args()
    torch.set_num_threads(1)
    out = a.run / "channels"
    out.mkdir(parents=True, exist_ok=True)
    for birth in a.births:
        for arm in a.arms:
            graph = load_checkpoint(a.run, arm, birth, a.step)
            raw = audit(graph, birth, arm, a.pairs)
            np.savez_compressed(out / f"{birth}-{arm}-{a.step}.npz", **raw)


if __name__ == "__main__":
    main()
