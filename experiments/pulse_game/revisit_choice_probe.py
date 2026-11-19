"""Decode taste and physical side from autonomous, frozen odor-revisit graphs."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import edit_receipt_train as storage
from . import revisit_choice_graph as native
from . import revisit_choice_train as train


CALIBRATION_DOMAIN = 0x4D71D
TEST_DOMAIN = 0x4D71E
SHUFFLE_DOMAIN = 0x4D71F
PHASES = ("pre_taste", "post_taste", "post_commit", "post_wait", "return_cue")
TARGETS = ("taste_sign", "first_good_side", "return_good_side")


@dataclass(frozen=True)
class Worlds:
    left: torch.Tensor
    first_good: torch.Tensor
    swap: torch.Tensor
    draws: torch.Tensor


def worlds(seed, birth, pairs, *, calibration):
    """Cross the private good side and return swap at each fixed odor and draw tape."""
    domain = CALIBRATION_DOMAIN if calibration else TEST_DOMAIN
    rng = np.random.default_rng([seed, birth, domain])
    odor = rng.normal(size=(pairs, 3))
    odor = (1.35 * odor / np.linalg.norm(odor, axis=1, keepdims=True)).astype(np.float32)
    draws = rng.random((pairs, 7), dtype=np.float32)
    first_good = torch.tensor([-1., -1., 1., 1.]).repeat(pairs)
    swap = torch.tensor([False, True, False, True]).repeat(pairs)
    return Worlds(torch.from_numpy(odor).repeat_interleave(4, dim=0), first_good, swap,
                  torch.from_numpy(draws).repeat_interleave(4, dim=0))


@torch.no_grad()
def live(graph, world, delay, *, keep_connected=True):
    """Record node states from a complete live path with physical choices and taste only."""
    left, right = world.left, -world.left
    state = native.start(graph, len(left))
    first = native.act(state, left, right, world.draws[:, 0])
    observed = first.side * world.first_good
    experience = native.taste(first, left, right, observed)
    proposals = native.propose(experience, world.draws[:, 1:5].reshape(-1, 2, 2))
    decision = native.choose(proposals, world.draws[:, 5])
    commit = native.commit(experience, proposals, decision, keep_connected=keep_connected)
    waited = native.wait(commit.state, delay)
    layout = torch.where(world.swap, -1., 1.)
    return_left = left * layout[:, None]
    revisit = native.act(waited, return_left, -return_left, world.draws[:, 6])
    hidden = torch.stack((experience.contact_hidden, experience.state.hidden, commit.state.hidden,
                         waited.hidden, revisit.state.hidden), 1)
    # The activity in Action.state is the motor input state before the physical side is sampled.
    labels = torch.stack((observed, first.side * observed, world.first_good * layout), 1)
    return {
        "hidden": hidden.numpy(), "targets": labels.numpy().astype(np.int8),
        "first_side": first.side.numpy().astype(np.int8), "swap": world.swap.numpy(),
        "first_motor_logit": first.motor_logit.numpy(), "return_motor_logit": revisit.motor_logit.numpy(),
        "return_side": revisit.side.numpy().astype(np.int8), "selection": decision.index.numpy().astype(np.int8),
        "receipt_status": commit.receipt.status.numpy().astype(np.int8),
    }


def balanced_weights(first_side, swap, taste):
    """Assign equal mass to each autonomous first action × swap × observed taste cell."""
    index = (first_side > 0).astype(np.int8) * 4 + swap.astype(np.int8) * 2 + (taste > 0)
    count = np.bincount(index, minlength=8)
    if (count == 0).any():
        raise ValueError("At least one side × swap × taste cell is empty; increase the number of pairs")
    return 1 / (8 * count[index]), count.reshape(2, 2, 2)


def fit_readout(x, y, weight, ridge):
    """Fit one centered ridge classifier per phase and target using calibration rows only."""
    center = np.einsum("n,npw->pw", weight, x)
    centered = x - center
    scale = np.sqrt(np.einsum("n,npw->pw", weight, centered ** 2)).clip(min=1e-6)
    standardized = centered / scale
    label_center = weight @ y
    gram = np.einsum("n,npw,npv->pwv", weight, standardized, standardized)
    covariance = np.einsum("n,npw,nt->pwt", weight, standardized, y - label_center)
    coefficients = np.linalg.solve(gram + ridge * np.eye(x.shape[-1]), covariance) / scale[:, :, None]
    intercept = label_center - np.einsum("pw,pwt->pt", center, coefficients)
    return coefficients, intercept


def score_readout(hidden, head):
    coefficients, intercept = head
    return np.einsum("npw,pwt->npt", hidden, coefficients) + intercept


def accuracy(scores, targets, weight):
    margin = scores * targets[:, None, :]
    correct = (margin > 0) + .5 * (margin == 0)
    return np.einsum("n,npt->pt", weight, correct)


def diagnose(graph, birth, seed, cal_pairs, test_pairs, delay, ridge, *, keep_connected):
    calibration = live(graph, worlds(seed, birth, cal_pairs, calibration=True), delay,
                       keep_connected=keep_connected)
    test = live(graph, worlds(seed, birth, test_pairs, calibration=False), delay,
                keep_connected=keep_connected)
    for data in (calibration, test):
        data["weight"], data["cell_count"] = balanced_weights(data["first_side"], data["swap"], data["targets"][:, 0])

    weight, x, y = (calibration[key] for key in ("weight", "hidden", "targets"))
    head = fit_readout(x, y, weight, ridge)
    rng = np.random.default_rng([seed, birth, SHUFFLE_DOMAIN])
    flip = np.repeat(rng.choice((-1, 1), size=cal_pairs), 4)
    shuffled = fit_readout(x, y * flip[:, None], weight, ridge)
    score = score_readout(test["hidden"], head)
    shuffle_score = score_readout(test["hidden"], shuffled)
    targets, weight = test["targets"], test["weight"]
    return_good = targets[:, 2]
    correct = test["return_side"] == return_good
    expected = .5 * (1 + np.tanh(.5 * test["return_motor_logit"] * return_good))
    metrics = {
        "accuracy": accuracy(score, targets, weight).tolist(),
        "shuffled_accuracy": accuracy(shuffle_score, targets, weight).tolist(),
        "native_return_accuracy": float(weight @ correct),
        "native_expected_return_accuracy": float(weight @ expected),
        "calibration_cell_count": calibration["cell_count"].tolist(),
        "test_cell_count": test["cell_count"].tolist(),
    }
    data = {f"test_{name}": value for name, value in test.items() if name != "hidden"}
    data.update({"score": score, "shuffle_score": shuffle_score,
                 "coef": head[0], "intercept": head[1], "shuffle_coef": shuffled[0], "shuffle_intercept": shuffled[1]})
    return metrics, data


def run(input_path, summary_path, scores_path, *, seed=20261124, cal_pairs=2048, test_pairs=4096,
        delay=4, ridge=.1):
    if min(cal_pairs, test_pairs) < 2 or seed < 0 or delay < 0 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Require at least two pairs per split, nonnegative seed and delay, and positive ridge")
    input_path = Path(input_path)
    plan = json.loads((input_path / "run.json").read_text(encoding="utf8"))
    arm = "ranked"
    step = plan["config"]["updates"]
    if arm not in plan["arms"] or step not in plan["checkpoints"]:
        raise ValueError("Expected the final ranked arm of the main study")
    births = plan["births"]
    checkpoints = [storage.checkpoint_path(input_path, arm, birth, step) for birth in births]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Final ranked checkpoint missing: {checkpoint}")
    torch.set_num_threads(1)
    records = []
    for birth in births:
        graph = train.load_checkpoint(input_path, arm, birth, step)
        records.append(diagnose(graph, birth, seed, cal_pairs, test_pairs, delay, ridge,
                                keep_connected=plan["config"]["keep_connected"]))
    metrics, rows = zip(*records, strict=True)
    summary = {
        "schema_version": 1, "arm": arm, "step": step,
        "phases": PHASES, "targets": TARGETS, "births": births, "seed": seed,
        "calibration_pairs_per_birth": cal_pairs, "test_pairs_per_birth": test_pairs,
        "delay": delay, "ridge": ridge,
        "cell_order": {"autonomous_first_side": [-1, 1], "return_swap": [False, True], "actual_taste": [-1, 1]},
        "checkpoint_sha256": {str(birth): sha256(path.read_bytes()).hexdigest()
                              for birth, path in zip(births, checkpoints, strict=True)},
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                          for path in map(Path, (__file__, native.__file__, native.fw.__file__))},
        "by_birth": dict(zip(births, metrics, strict=True)),
        "mean": {name: np.mean([metric[name] for metric in metrics], axis=0).tolist()
                 for name in ("accuracy", "shuffled_accuracy", "native_return_accuracy",
                              "native_expected_return_accuracy")},
    }
    storage.write_arrays(Path(scores_path), births=births, phases=PHASES, targets=TARGETS,
                         **{name: np.stack([row[name] for row in rows]) for name in rows[0]})
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=train.DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--calibration-pairs", type=int, default=2048)
    parser.add_argument("--test-pairs", type=int, default=4096)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=.1)
    args = parser.parse_args()
    summary = args.summary or args.input / "probe-summary.json"
    scores = args.scores or args.input / "probe-scores.npz"
    outcome = run(args.input, summary, scores, seed=args.seed, cal_pairs=args.calibration_pairs,
                  test_pairs=args.test_pairs, delay=args.delay, ridge=args.ridge)
    for phase, values in zip(PHASES, outcome["mean"]["accuracy"], strict=True):
        print(f"{phase:12s} " + " ".join(f"{target}={value:.4f}" for target, value in zip(TARGETS, values, strict=True)))
    print(f"native_return_accuracy={outcome['mean']['native_return_accuracy']:.4f}")
    print(f"Wrote {summary} and {scores}")


if __name__ == "__main__":
    main()
