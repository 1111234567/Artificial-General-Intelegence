"""Exchange activity and graph parameters between opposite-law feeder-mark twins."""

from argparse import ArgumentParser
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import revisit_choice_value as value


ARMS = ("fixed", "variable")
BRANCHES = ("own", "cross_h", "cross_parameters", "cross_both")
BOOKS = ("pooled", "negative", "positive")
CONTRASTS = ("cross_h_minus_own", "cross_parameters_minus_own", "cross_both_minus_own",
             "h_by_parameters_interaction")
BOOTSTRAP_DOMAIN = 419639
DEFAULT_INPUT = Path("data/revisit-choice-20261119/value")


def subset(scene, indices):
    return value.Scene(*(getattr(scene, field.name)[indices] for field in fields(value.Scene)))


def opposite(scene):
    """Map each tape and all unchanged signs to the sibling with opposite private law."""
    twin = torch.arange(len(scene.book)) ^ 8
    assert torch.equal(scene.book[twin], -scene.book)
    for field in ("first_mark", "right_mark", "other_book", "uniforms"):
        x = getattr(scene, field)
        assert torch.equal(x[twin], x)
    return twin


def prefix(graph, scene):
    """Carry the two calibration contacts, resource receipts and native graph writes."""
    state = value.expand(graph, len(scene.book))
    policy = value.policy
    for t, mark in enumerate((scene.first_mark, -scene.first_mark)):
        icon = value.marks(mark)
        draws = scene.uniforms[:, t]
        action = policy.act(state, icon, icon, draws[:, 0])
        experience = policy.taste(action, icon, icon, mark * scene.book)
        candidates = policy.propose(experience, draws[:, 1:5].reshape(-1, 2, 2))
        decision = policy.choose(candidates, draws[:, 5])
        state = policy.commit(experience, candidates, decision).state
    return state


@torch.no_grad()
def evaluate_chunk(graph, scene):
    policy = value.policy
    twin = opposite(scene)
    own = prefix(graph, scene)
    other = own.subset(twin)
    states = (own, policy.State(own.graph, other.hidden), policy.State(other.graph, own.hidden), other)
    right = value.marks(scene.right_mark)
    final_draw = scene.uniforms[:, -1, 0]
    actions = [policy.act(state, -right, right, final_draw) for state in states]
    side = torch.stack([action.side for action in actions], 1)
    logits = torch.stack([action.motor_logit for action in actions], 1)
    good = scene.right_mark * scene.book
    success = side == good[:, None]

    reference, earned, _ = value.interact(graph, scene, len(scene.book))
    assert torch.equal(actions[0].side, reference.side)
    assert torch.equal(actions[0].motor_logit, reference.motor_logit)
    assert torch.equal(success[:, 0], earned > 0)
    assert torch.equal(side[:, 3], side[twin, 0]) and torch.equal(side[:, 1], side[twin, 2])
    assert torch.equal(logits[:, 3], logits[twin, 0]) and torch.equal(logits[:, 1], logits[twin, 2])
    assert torch.equal(success[:, 3], ~success[twin, 0])
    assert torch.equal(success[:, 1], ~success[twin, 2])

    changes = torch.stack((
        (own.hidden != other.hidden).any(1),
        (own.graph.weights != other.graph.weights).flatten(1).any(1),
        (own.graph.present != other.graph.present).flatten(1).any(1),
        (own.graph.bias != other.graph.bias).any(1),
    ), 1)
    return {
        "book": scene.book.numpy().astype(np.int8), "good": good.numpy().astype(np.int8),
        "side": side.numpy().astype(np.int8), "logits": logits.numpy(), "success": success.numpy(),
        "changes": changes.numpy(), "first_mark": scene.first_mark.numpy().astype(np.int8),
        "right_mark": scene.right_mark.numpy().astype(np.int8),
    }


def evaluate(graph, scene, chunk_pairs):
    chunk_size = 16 * chunk_pairs
    rows = [evaluate_chunk(graph, subset(scene, slice(start, start + chunk_size)))
            for start in range(0, len(scene.book), chunk_size)]
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}


def intervals(values, indices):
    """Root-cluster percentile intervals with the same draw for all paired cells and arms."""
    samples = values[indices].mean(axis=1)
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {"mean": values.mean(0).tolist(), "ci95_low": bounds[0].tolist(), "ci95_high": bounds[1].tolist(),
            "by_birth": values.tolist()}


def summarize(rows, indices):
    accuracy = np.array([[row["success"][mask].mean(0) for mask in (
        np.ones(len(row["book"]), dtype=bool), row["book"] < 0, row["book"] > 0,
    )] for row in rows])
    own, cross_h, cross_p, cross_both = accuracy.transpose(2, 0, 1)
    contrasts = np.stack((cross_h - own, cross_p - own, cross_both - own,
                          cross_both - cross_h - cross_p + own), -1)
    np.testing.assert_allclose(cross_both[:, 0], 1 - own[:, 0], atol=1e-15, rtol=0)
    np.testing.assert_allclose(cross_h[:, 0] + cross_p[:, 0], 1, atol=1e-15, rtol=0)
    return {
        "accuracy": intervals(accuracy, indices), "contrasts": intervals(contrasts, indices),
        "choice_changed_vs_own": intervals(np.array([(r["side"] != r["side"][:, :1]).mean(0) for r in rows]), indices),
        "state_differs_from_twin": intervals(np.array([r["changes"].mean(0) for r in rows]), indices),
    }, accuracy, contrasts


def run(root, *, step=None, pairs=None, chunk_pairs=64, bootstraps=20000, output=None):
    root = Path(root)
    plan = json.loads((root / "protocol.json").read_text(encoding="utf8"))
    stages = plan.get("steps", plan.get("stages"))
    step = max(stages) if step is None else step
    pairs = plan.get("pairs", plan.get("tapes_per_birth")) if pairs is None else pairs
    domain = plan.get("holdout_domain", 795821)
    if step not in stages or min(pairs, chunk_pairs, bootstraps) < 1:
        raise ValueError("Expected a protocol checkpoint and positive tape, chunk and bootstrap counts")
    births = plan["births"]
    paths = {arm: [root / arm / f"{birth}-{step}.npz" for birth in births] for arm in ARMS}
    for path in (*paths["fixed"], *paths["variable"]):
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint missing: {path}")
    torch.set_num_threads(1)
    start = perf_counter()
    rng = np.random.default_rng([BOOTSTRAP_DOMAIN, domain, step])
    indices = rng.integers(len(births), size=(bootstraps, len(births)))
    raw = {}
    summaries, accuracies, contrasts = {}, {}, {}
    for arm, checkpoints in paths.items():
        rows = []
        for birth, path in zip(births, checkpoints, strict=True):
            with np.load(path, allow_pickle=False) as data:
                graph = value.fw.Graph(*(torch.from_numpy(data[key]) for key in ("w", "b", "m")))
            scene = value.scenes([domain, birth], pairs, balanced=True)
            rows.append(evaluate(graph, scene, chunk_pairs))
        summaries[arm], accuracies[arm], contrasts[arm] = summarize(rows, indices)
        raw.update({f"{arm}_{key}": np.stack([row[key] for row in rows]) for key in rows[0]})

    result = {
        "schema_version": 1, "step": step, "births": births, "pairs_per_birth": pairs,
        "holdout_domain": domain, "lanes_per_tape": 16, "max_chunk_lanes": min(pairs, chunk_pairs) * 16,
        "book_axis": BOOKS, "branch_axis": BRANCHES, "contrast_axis": CONTRASTS,
        "state_difference_axis": ("h", "weights", "support", "bias"),
        "targets": "Physical success uses the recipient's true mark-resource law in both training arms.",
        "validated": ("Intact paths match value.interact exactly, including logit, side and earned sign.",
                      "Both swaps replay the sibling's intact choice exactly on the opposite utility.",
                      "Cross h equals sibling cross parameters in logit and side.",
                      "The reciprocal design implies pooled h/parameter interaction is identically zero. "
                      "The law-conditional interaction can vary and is reported."),
        "bootstrap": {"seed_domain": BOOTSTRAP_DOMAIN, "samples": bootstraps,
                      "unit": "paired root", "confidence": .95},
        "arms": summaries,
        "variable_minus_fixed": {
            "accuracy": intervals(accuracies["variable"] - accuracies["fixed"], indices),
            "contrasts": intervals(contrasts["variable"] - contrasts["fixed"], indices),
        },
        "checkpoint_sha256": {arm: [sha256(path.read_bytes()).hexdigest() for path in checkpoints]
                              for arm, checkpoints in paths.items()},
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in map(Path, (
            __file__, value.__file__, value.policy.__file__, value.fw.__file__, value.policy.receipt_graph.__file__,
        ))},
        "runtime_seconds": perf_counter() - start,
    }
    output = Path(output) if output is not None else root / f"exchange-{step}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    value.write_arrays(output.with_suffix(".npz"), births=births, branches=BRANCHES, **raw)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--step", type=int)
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--chunk-pairs", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.input, step=args.step, pairs=args.pairs, chunk_pairs=args.chunk_pairs,
                 bootstraps=args.bootstrap, output=args.output)
    for arm in ARMS:
        mean = result["arms"][arm]["accuracy"]["mean"][0]
        print(arm, ", ".join(f"{branch}={x:.4f}" for branch, x in zip(BRANCHES, mean, strict=True)))
    print(f"Finished {result['step']} in {result['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
