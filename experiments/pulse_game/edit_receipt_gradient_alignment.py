"""Compare auxiliary and shared reward gradients on matched fresh histories."""

from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
from itertools import batched
import json
from pathlib import Path

import numpy as np
import torch

from . import edit_receipt_train as train
from . import self_param_writer as fw


DIAGNOSTIC_SEED = 829241
TERMS = ("common_reward", "local_effect", "future_benefit")
ROLLOUT_METRICS = tuple(name for name in train.METRICS if name != "grad_norm")
SUMMARY = train.ROOT / "docs/evidence/edit-receipt-2026-11-19/gradient-alignment.json"


def histories(config, births, sample, seed):
    domain = (seed, config.world_seed, sample)
    focal = int(np.random.default_rng([*domain, 1]).integers(config.contacts - config.horizon))
    tapes = [train.make_worlds([*domain, 0, birth], config.batch, config.contacts, noise=config.noise)
             for birth in births]
    return train.Tape.concatenate(tapes), focal, torch.arange(config.batch).repeat(len(tapes))


def stack_graphs(graphs):
    weights = torch.cat([graph.weights for graph in graphs]).requires_grad_()
    bias = torch.cat([graph.bias for graph in graphs]).requires_grad_()
    present = torch.cat([graph.present for graph in graphs])
    return fw.Graph(weights, bias, present)


def decompose(graph, tape, focal, shuffle, config):
    """Return the common gradient and the incremental auxiliary gradients."""
    common = replace(config, effect_weight=0., benefit_weight=0.)
    variants = (common, replace(common, effect_weight=config.effect_weight),
                replace(common, benefit_weight=config.benefit_weight))
    gradients, traces = [], []
    for variant in variants:
        loss, metrics = train.training_step(graph, tape, focal, shuffle, "causal", variant)
        weight_grad, bias_grad = torch.autograd.grad(loss, (graph.weights, graph.bias))
        gradients.append(torch.cat((weight_grad.flatten(1), bias_grad), dim=1).numpy())
        traces.append(metrics)
    match = np.arange(len(ROLLOUT_METRICS)) != ROLLOUT_METRICS.index("fork_used")
    if not all(np.array_equal(traces[0][:, match], trace[:, match]) for trace in traces[1:]):
        raise AssertionError("Changing loss weights changed the matched rollout")
    common, local, future = (g.astype(np.float64) for g in gradients)
    values = np.stack((common, local - common, future - common), axis=1)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite gradient in diagnostic batch")
    return values, traces[-1]


def average(values, include):
    return float(values[include].mean()) if include.any() else None


def describe(values, traces, config):
    norms = np.linalg.norm(values, axis=-1)
    valid = norms > 0
    by_term = {term: {"mean_norm": float(norms[..., index].mean()),
                      "median_norm": float(np.median(norms[..., index])),
                      "fraction_zero_gradient": float((~valid[..., index]).mean())}
               for index, term in enumerate(TERMS)}
    for index, term in enumerate(TERMS[1:], 1):
        dots = np.einsum("...k,...k->...", values[..., 0, :], values[..., index, :])
        informative = valid[..., 0] & valid[..., index]
        cosine = np.zeros_like(dots)
        np.divide(dots, norms[..., 0] * norms[..., index], out=cosine, where=informative)
        by_term[term].update({"informative_pairs": int(informative.sum()),
                              "mean_dot_with_common": float(dots.mean()),
                              "mean_cosine_with_common": average(cosine, informative),
                              "negative_dot_fraction_of_informative": average(dots < 0, informative)})

    def metric(name):
        return traces[..., ROLLOUT_METRICS.index(name)]

    usable = metric("fork_used") > 0
    credit_abs = "fork_taste_abs" if config.credit_target == "taste" else "fork_expected_taste_abs"
    signal = {
        "fraction_batches_without_attempted_edit": float((metric("attempted") == 0).mean()),
        "fraction_batches_without_usable_fork_labels": float((~usable).mean()),
        "fraction_usable_batches_with_all_fork_targets_zero": average(metric(credit_abs) == 0, usable),
        "mean_usable_fork_labels_per_batch": float(metric("fork_used").mean()),
    }
    return {"gradient": by_term, "signal": signal}


def run(path, summary, *, samples, seed, steps=None, births=None, raw=None):
    plan = json.loads((path / "run.json").read_text(encoding="utf-8"))
    config = train.Config(**plan["config"])
    births = plan["births"] if births is None else births
    steps = plan["checkpoints"] if steps is None else steps
    group_size = plan["fit_group_size"]
    sources = (Path(train.__file__), Path(train.native.__file__), Path(fw.__file__))
    source_hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in sources}
    if source_hashes != plan["sources_sha256"]:
        raise ValueError("Training and graph sources differ from the frozen run manifest")
    saved = [[train.load_checkpoint(path, "causal", birth, step) for birth in births] for step in steps]
    checkpoints = [train.checkpoint_path(path, "causal", birth, step) for step in steps for birth in births]
    inputs = [histories(config, births, sample, seed) for sample in range(samples)]
    gradient_blocks, trace_blocks, steps_summary = [], [], {}
    for step, graphs in zip(steps, saved, strict=True):
        all_values, all_traces = [], []
        for indices in batched(range(len(births)), group_size):
            graph = stack_graphs([graphs[index] for index in indices])
            start, stop = indices[0] * config.batch, (indices[-1] + 1) * config.batch
            window = slice(start, stop)
            group_values, group_traces = [], []
            for tape, focal, shuffle in inputs:
                tape = train.Tape(tape.x[window], tape.truth[window], tape.outcome[window], tape.draws[window])
                values, traces = decompose(graph, tape, focal, shuffle[window], config)
                group_values.append(values)
                group_traces.append(traces)
            all_values.append(np.stack(group_values, axis=1))
            all_traces.append(np.stack(group_traces, axis=1))
        values = np.concatenate(all_values)
        traces = np.concatenate(all_traces)
        gradient_blocks.append(values)
        trace_blocks.append(traces)
        records = describe(values, traces, config)
        records["births"] = {str(birth): describe(values[index], traces[index], config)
                             for index, birth in enumerate(births)}
        steps_summary[str(step)] = records
        print(f"Gradient alignment computed for step {step}, {len(births)} births × {samples} minibatches", flush=True)

    if raw is None:
        raw = path / "diagnostics/gradient-alignment.npz"
    train.write_arrays(raw, steps=steps, births=births, samples=np.arange(samples), seed=seed,
                       terms=TERMS, gradients=np.stack(gradient_blocks),
                       metrics=ROLLOUT_METRICS, rollout_metrics=np.stack(trace_blocks))
    results = {
        "interpretation": "Instantaneous pre-clipping gradient alignment at fixed causal checkpoints. A negative dot means an auxiliary descent direction increases the common sampled loss to first order. Training trajectory and Adam updates require separate measurements.",
        "gradient_composition": {
            "common_reward": "Actor score, writer score including acceptance, and differentiable motor reward path at their training weights.",
            "local_effect": "Gradient difference after adding the training-weight local edit-forecast loss.",
            "future_benefit": "Gradient difference after adding the training-weight paired future-benefit loss.",
            "weights": {"writer": config.writer_weight, "motor_path": config.motor_path_weight,
                        "local_effect": config.effect_weight, "future_benefit": config.benefit_weight},
        },
        "design": {"births": births, "steps": steps, "fresh_minibatches_per_birth": samples,
                   "batch_size": config.batch, "diagnostic_seed": seed,
                   "world_seed_fields": ["diagnostic_seed", "training_world_seed", "sample", 0, "birth"],
                   "matched": "Each birth and minibatch uses the same stimuli, taste noise, random draws and focal contact across checkpoints and losses.",
                   "cosine": "Cosines and negative-dot fractions are defined on pairs with two nonzero gradients. Mean dots include all pairs.",
                   "usable_fork_labels": "The training loss requires at least two eligible lane targets. All-zero fractions are computed among minibatches passing that mask."},
        "raw": {"file": raw.resolve().as_posix(), "sha256": sha256(raw.read_bytes()).hexdigest(),
                "axes": ["checkpoint", "birth", "minibatch", "gradient term", "weight coordinates then bias"]},
        "sources_sha256": {**source_hashes, Path(__file__).name: sha256(Path(__file__).read_bytes()).hexdigest()},
        "checkpoints_sha256": {p.relative_to(path).as_posix(): sha256(p.read_bytes()).hexdigest() for p in checkpoints},
        "by_step": steps_summary,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=train.DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DIAGNOSTIC_SEED)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--births", type=int, nargs="+")
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0 or args.seed == train.TRAIN_DOMAIN:
        parser.error("Use positive samples and a nonnegative seed distinct from training")
    torch.set_num_threads(1)
    run(args.run, args.summary, samples=args.samples, seed=args.seed, steps=args.steps, births=args.births, raw=args.raw)


if __name__ == "__main__":
    main()
