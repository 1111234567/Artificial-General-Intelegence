"""Replay frozen joint fits on fresh optimizer-equivalent worlds and held-out layouts."""

from argparse import ArgumentParser
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_train as train


METRICS = ("joint", "odor", "marks", "a", "b", "c", "both_a_b", "a_b_agree", "a_b_conflict",
           "a_agree", "b_agree", "a_conflict", "b_conflict")
FAMILIES = ("matched", "heldout_layout")
OUT = train.DEFAULT_OUTPUT.parent / "development"


def _layouts(family):
    if family == "matched":
        return train.TRAIN_LAYOUTS
    trained = {tuple(row) for row in train.TRAIN_LAYOUTS}
    return np.array([row for row in train.modality_layouts(5) if tuple(row) not in trained])


def _batch_checkpoint(path, arm, births, step):
    loaded = [train.load_checkpoint(path, arm, birth, step) for birth in births]
    graph = train.fw.Graph(*(torch.cat([getattr(g, part.name) for g, _ in loaded])
                             for part in fields(train.fw.Graph)))
    if arm == "legacy_active":
        return graph, None
    states = [rule.state_dict() for _, rule in loaded]
    rule = train.local.Rule(births)
    rule.load_state_dict({name: torch.cat([state[name] for state in states]) for name in states[0]})
    return graph, rule.eval().requires_grad_(False)


def _shared_tape(births, seed, pairs, family, block, wait):
    config = train.Config(pairs=pairs, world_seed=seed, max_wait=wait)
    tape = train.make_tape((0,), config, block, domain=train.DEVELOPMENT_DOMAIN, layouts=_layouts(family))
    return train.Tape(*(value.expand(len(births), *value.shape[1:])
                        for value in (getattr(tape, field.name) for field in fields(train.Tape))))


def _per_pair(roll, tape):
    probe = tape.probe
    success = (roll.resource > 0).float()
    by_probe = [(success * (probe == index)).sum(dim=-1) / (probe == index).sum(dim=-1) for index in range(3)]
    both = ((by_probe[0] == 1) & (by_probe[1] == 1)).float()
    conflict = tape.law[..., 0] != tape.law[..., 1]
    weight = tape.objective_weight
    odor = 2 * (success * weight * (probe < 3)).sum(dim=-1)
    marks = 2 * (success * weight * (probe >= 3)).sum(dim=-1)
    nan = float("nan")
    result = torch.stack(((odor + marks) / 2, odor, marks, *by_probe, both,
                          both.masked_fill(conflict, nan), both.masked_fill(~conflict, nan),
                          *(value.masked_fill(conflict, nan) for value in by_probe[:2]),
                          *(value.masked_fill(~conflict, nan) for value in by_probe[:2])), dim=-1)
    return result.mean(dim=2).numpy()


@torch.no_grad()
def evaluate(graph, rule, arm, tapes, wait, branch="native"):
    if branch == "donor" and rule is not None:
        arm = "local_independent"
    elif branch == "own" and rule is not None:
        arm = "local_active"
    intervention = "disabled" if branch == "disabled" else "active"
    return np.concatenate([_per_pair(train.rollout(graph, rule, tape, arm, wait, intervention=intervention), tape)
                           for tape in tapes], axis=1)


def estimate(values, boot_birth, boot_pair):
    """Resample independent training births and shared test pairs as crossed factors."""
    sampled = values[boot_birth[:, :, None], boot_pair[:, None, :]]
    draw = np.nanmean(sampled, axis=(1, 2))
    bounds = np.nanquantile(draw, [.025, .975])
    return {"mean": float(np.nanmean(values)), "ci95": bounds.tolist(),
            "by_birth": np.nanmean(values, axis=1).tolist()}


def run(path, checkpoints, births, pairs, seed, wait, families, bootstrap, batch, contrasts=()):
    torch.set_num_threads(1)
    rngs = [np.random.default_rng([train.DEVELOPMENT_DOMAIN, seed, i]) for i in (701, 702)]
    boot_birth = rngs[0].integers(len(births), size=(bootstrap, len(births)))
    boot_pair = rngs[1].integers(pairs, size=(bootstrap, pairs))
    banks = {family: [_shared_tape(births, seed, min(batch, pairs - block * batch), family, block, wait)
                      for block in range((pairs + batch - 1) // batch)] for family in families}
    path.mkdir(parents=True, exist_ok=True)
    packed, checkpoint_hashes = {}, {}
    for label, source, arm, step in checkpoints:
        filename = train.checkpoint_path(source, arm, step)
        checkpoint_hashes[label] = sha256(filename.read_bytes()).hexdigest()
        print(f"model={label} step={step}", flush=True)
        graph, rule = _batch_checkpoint(source, arm, births, step)
        branches = {"local_active": ("native", "disabled", "donor"),
                    "local_independent": ("native", "disabled", "own"),
                    "legacy_active": ("native", "disabled")}[arm]
        for family, tapes in banks.items():
            for branch in branches:
                packed[label, family, branch] = evaluate(graph, rule, arm, tapes, wait, branch)
    summary = {"births": births, "pairs_per_birth": pairs, "seed": seed, "max_wait": wait, "batch": batch,
               "development_domain": train.DEVELOPMENT_DOMAIN, "checkpoint_sha256": checkpoint_hashes,
               "layout_counts": {family: len(_layouts(family)) for family in families},
               "source_sha256": {Path(file).name: sha256(Path(file).read_bytes()).hexdigest()
                                 for file in (__file__, train.__file__, train.local.__file__, train.legacy.__file__)},
               "bootstrap_draws": bootstrap, "metrics": METRICS,
               "protocol": "Fresh antipodal worlds on one frozen graph per birth. The held-out layout family "
                           "requires a same-modality streak of four or five, which the trainer excludes. "
                           "Both A/B requires every scored A and B return correct. Worlds are crossed with "
                           "births. Donor traces use independent laws while scoring actual motor returns. "
                           "The independently trained controller uses donor laws on its native branch and "
                           "true physical tastes on its own branch.",
               "results": {}, "contrasts": {}}

    def summarize(values):
        return {metric: estimate(values[:, :, i], boot_birth, boot_pair) for i, metric in enumerate(METRICS)}

    for (label, family, branch), values in packed.items():
        name = f"{label}/{family}/{branch}"
        summary["results"][name] = summarize(values)
        if branch != "native":
            summary["contrasts"][f"{label}/{family}/native_minus_{branch}"] = (
                summarize(packed[label, family, "native"] - values))
    for family in families:
        reference = checkpoints[0][0]
        comparisons = [(label, reference) for label, *_ in checkpoints[1:]] + list(contrasts)
        for label, control in comparisons:
            choice, _, branch = label.partition("@")
            reference_label, _, other_branch = control.partition("@")
            summary["contrasts"][f"{family}/{label}_minus_{control}"] = summarize(
                packed[choice, family, branch or "native"]
                - packed[reference_label, family, other_branch or "native"])
    np.savez_compressed(path / "by-pair.npz", **{"/".join(key): value for key, value in packed.items()})
    (path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs=4, metavar=("LABEL", "RUN", "ARM", "STEP"),
                        action="append", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--births", type=int, nargs="+", default=train.BIRTHS)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=26911)
    parser.add_argument("--max-wait", type=int, default=8)
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=FAMILIES)
    parser.add_argument("--bootstrap", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--contrast", nargs=2, metavar=("MODEL", "CONTROL"), action="append", default=[])
    options = parser.parse_args()
    requested = [(label, Path(path), arm, int(step)) for label, path, arm, step in options.checkpoint]
    if (len({row[0] for row in requested}) != len(requested) or
            min(options.pairs, options.bootstrap, options.batch) < 2 or options.max_wait < 0):
        parser.error("Use distinct checkpoint labels, at least two pairs, draws and worlds per batch")
    run(options.out, requested, options.births, options.pairs, options.seed, options.max_wait,
        options.families, options.bootstrap, options.batch, options.contrast)
