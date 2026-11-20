"""Measure when one continued graph earns both physical returns within one short life."""

from argparse import ArgumentParser
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_development as dev, interleave_joint_continue as joint, interleave_train as train


DOMAIN = 2201167
METRICS = {"joint4": ("joint", "mark", "odor", "both"), "mark3": ("mark",),
           "context3": ("odor", "conflicting_odor", "agreeing_odor")}


def make_bank(births, config, seed, pairs, batch):
    """Cross the same independent evaluation worlds with every paired birth."""
    banks = {mode: [] for mode in METRICS}
    for start in range(0, pairs, batch):
        size = min(batch, pairs - start)
        input_config = train.Config(pairs=size, world_seed=seed, max_wait=config.max_wait)
        tape = train.make_tape((0,), input_config, start, domain=DOMAIN)
        for index, mode in enumerate(METRICS):
            short = joint.short_life(tape, mode, np.random.default_rng([DOMAIN, seed, start, index + 1]))
            replicated = train.Tape(*(value.expand(len(births), *value.shape[1:])
                                      for value in (getattr(short, field.name) for field in fields(train.Tape))))
            banks[mode].append(replicated)
    return banks


@torch.no_grad()
def evaluate(graph, rule, tape, mode, branch, wait):
    rolled = joint._short_rollout(graph, rule, tape, wait, intervention=branch)
    success, probe = rolled.resource > 0, tape.probe
    mark, odor = (success & (probe == 3)).any(-1).float(), (success & (probe >= 0) & (probe < 3)).any(-1).float()
    if mode == "joint4":
        values = ((mark + odor) / 2, mark, odor, mark * odor)
    elif mode == "mark3":
        values = (mark,)
    else:
        conflicting = tape.law[..., 0] != tape.law[..., 1]
        values = (odor, odor.masked_fill(~conflicting, float("nan")), odor.masked_fill(conflicting, float("nan")))
    return torch.stack(values, dim=-1).mean(dim=2).numpy()


def run(path, checkpoints, births, pairs, seed, batch, bootstrap, contrasts):
    torch.set_num_threads(1)
    config = train.Config()
    bank = make_bank(births, config, seed, pairs, batch)
    rng = np.random.default_rng([DOMAIN, seed, 701])
    boot_birth = rng.integers(len(births), size=(bootstrap, len(births)))
    boot_pair = rng.integers(pairs, size=(bootstrap, pairs))
    report = dict(births=births, pairs_per_birth=pairs, seed=seed, batch=batch, training_mode_domain=joint.DOMAIN,
                  evaluation_domain=DOMAIN, bootstrap=bootstrap, checkpoints={}, results={}, contrasts={},
                  sources={Path(f).name: sha256(Path(f).read_bytes()).hexdigest()
                           for f in (__file__, joint.__file__, dev.__file__, train.__file__)},
                  protocol="Same frozen 20-node starting graph for all modes per birth. Joint4 gives one real odor "
                           "experience and one real mark demonstration before their respective probes, randomly "
                           "interleaved. Both requires the two sampled decisions to be correct in the same life. "
                           "Mark3 gives two opposing mark demonstrations. Context3 introduces both visible A/B "
                           "cues for one shared odor pair in random order, then probes a randomized context. "
                           "A/B private laws are independent and conditional conflict/agree results are separate. "
                           "The disabled branch preserves contact timing and activity with graph writes disabled.")
    raw = {}

    def summarize(values, mode):
        return {metric: dev.estimate(values[..., i], boot_birth, boot_pair)
                for i, metric in enumerate(METRICS[mode])}

    for label, source, step in checkpoints:
        filename = train.checkpoint_path(source, joint.ARM, step)
        report["checkpoints"][label] = {"step": step, "sha256": sha256(filename.read_bytes()).hexdigest()}
        graph, rule = dev._batch_checkpoint(source, joint.ARM, births, step)
        for mode, tapes in bank.items():
            for branch in ("active", "disabled"):
                key = f"{label}/{mode}/{branch}"
                raw[key] = np.concatenate([evaluate(graph, rule, tape, mode, branch, config.max_wait)
                                           for tape in tapes], axis=1)
                report["results"][key] = summarize(raw[key], mode)
            report["contrasts"][f"{label}/{mode}/active_minus_disabled"] = summarize(
                raw[f"{label}/{mode}/active"] - raw[f"{label}/{mode}/disabled"], mode)
        joint_score = report["results"][f"{label}/joint4/active"]["both"]["mean"]
        print(f"model={label} step={step} both_joint4={joint_score:.3f}", flush=True)

    for left, right in contrasts:
        for mode in METRICS:
            report["contrasts"][f"{mode}/{left}_minus_{right}"] = summarize(
                raw[f"{left}/{mode}/active"] - raw[f"{right}/{mode}/active"], mode)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "by-pair.npz", **raw)
    (path / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs=3, metavar=("LABEL", "RUN", "STEP"), action="append", required=True)
    parser.add_argument("--contrast", nargs=2, metavar=("MODEL", "CONTROL"), action="append", default=[])
    parser.add_argument("--births", type=int, nargs="+", default=train.BIRTHS)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=26937)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=4000)
    options = parser.parse_args()
    checkpoints = [(label, Path(path), int(step)) for label, path, step in options.checkpoint]
    if (len({label for label, *_ in checkpoints}) != len(checkpoints)
            or min(options.pairs, options.batch, options.bootstrap) < 2):
        parser.error("Use distinct checkpoint labels and at least two pairs, draws and worlds per batch")
    run(options.out, checkpoints, options.births, options.pairs, options.seed, options.batch, options.bootstrap,
        options.contrast)
