"""Fixed-budget structural plasticity on the shared context A-B-A task.

The three sparse arms have identical initial weights, masks, training examples,
teacher-residual weight updates, and per-output connection budgets.  Learned
rewiring drops small-magnitude active weights and grows inactive connections
using windowed pre-activity times teacher-residual correlations.  Random
rewiring replaces exactly the same number of edges at the same times.  A dense
fixed-topology readout is a separately labelled task-feasibility control.

This is externally supervised local learning on engineered signed features,
not autonomous learning from firing frequency or an LIF implementation.  Dense
arrays hold masked weights and candidate statistics; no sparse speed claim is
made.  Dormant weights stay exactly zero and are never secretly trained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.streaming.local_learning import derived_seed, make_task, sample_batch


ARMS = ("fixed", "learned", "random", "dense")
PHASES = (("A1", 0), ("B", 1), ("A2", 0))


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_hash(tensor):
    return hashlib.sha256(_bytes(tensor)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def nonzero_initial_weights(shape, rng, scale):
    signs = torch.randint(0, 2, shape, generator=rng).float().mul_(2).sub_(1)
    return signs * scale


@dataclass
class SparseReadout:
    """Only mask-selected weights are connections; all others remain zero."""

    weight: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if self.weight.ndim != 2 or self.mask.shape != self.weight.shape or self.mask.dtype != torch.bool:
            raise ValueError("weight and boolean mask must have shape [classes, features]")
        if self.weight.device.type != "cpu" or self.mask.device.type != "cpu":
            raise ValueError("This controlled experiment uses CPU tensors only")
        if bool(self.weight[~self.mask].ne(0).any()):
            raise ValueError("Dormant connections must have exactly zero weight")
        self.correlation = torch.zeros_like(self.weight)
        self.window_batches = 0

    def clone(self):
        copied = SparseReadout(self.weight.clone(), self.mask.clone())
        copied.correlation.copy_(self.correlation)
        copied.window_batches = self.window_batches
        return copied

    def probabilities(self, features):
        return (features @ self.weight.T).softmax(-1)

    @torch.no_grad()
    def update(self, features, residual, learning_rate):
        """Use only pre activity and externally provided per-output feedback.

        The residual is teacher_one_hot - predicted_probability.  For this
        single-layer softmax, this local delta rule also equals its supervised
        gradient update.  Every arm updates the same full candidate-statistic
        array, but only existing connections have weight updates.
        """
        if features.shape[0] == 0 or residual.shape != (features.shape[0], self.weight.shape[0]):
            raise ValueError("Features and residual must describe the same nonempty batch")
        correlation = residual.T @ features / features.shape[0]
        self.weight.add_(learning_rate * correlation * self.mask)
        self.correlation.add_(correlation)
        self.window_batches += 1

    @torch.no_grad()
    def rewire(self, strategy, count, rng, new_weight_scale=.01):
        """Replace count edges per output; removed edges cannot regrow now.

        Candidate correlations are accumulated over a fixed update window,
        then cleared for *all* arms.  There are no activity, age, homeostatic,
        success, or learned-importance thresholds. Ties are broken by the
        supplied RNG, rather than a privileged feature index.
        """
        if strategy not in {"fixed", "learned", "random"}:
            raise ValueError(strategy)
        if count < 0:
            raise ValueError("Replacement count must be non-negative")
        before = self.mask.clone()
        removed, added = [], []
        born_correlations = []
        window_batches = self.window_batches
        mean_correlation = self.correlation / max(1, window_batches)
        for output in range(self.weight.shape[0]):
            if strategy == "fixed" or count == 0:
                continue
            active = self.mask[output].nonzero().flatten()
            dormant = (~self.mask[output]).nonzero().flatten()
            if count > min(active.numel(), dormant.numel()):
                raise ValueError("Replacement count exceeds active or dormant connections")
            # Matching randperm shapes keep learned/random random consumption
            # equal even though only learned uses the data-dependent ranking.
            active = active[torch.randperm(active.numel(), generator=rng)]
            dormant = dormant[torch.randperm(dormant.numel(), generator=rng)]
            if strategy == "learned":
                active = active[torch.argsort(self.weight[output, active].abs(), stable=True)]
                score = mean_correlation[output, dormant].abs()
                dormant = dormant[torch.argsort(score, descending=True, stable=True)]
            dropped, born = active[:count], dormant[:count]
            self.mask[output, dropped] = False
            self.weight[output, dropped] = 0
            self.mask[output, born] = True
            self.weight[output, born] = nonzero_initial_weights((count,), rng, new_weight_scale)
            removed.extend([[output, int(index)] for index in dropped])
            added.extend([[output, int(index)] for index in born])
            born_correlations.extend(mean_correlation[output, born].tolist())
        deleted_actual = int((before & ~self.mask).sum())
        added_actual = int((~before & self.mask).sum())
        if deleted_actual != len(removed) or added_actual != len(added):
            raise RuntimeError("Reported structural changes disagree with the actual mask")
        if not torch.equal(before.sum(1), self.mask.sum(1)):
            raise RuntimeError("Rewiring changed a per-output connection budget")
        self.correlation.zero_()
        self.window_batches = 0
        return dict(strategy=strategy, window_batches=window_batches,
                    removed=deleted_actual, added=added_actual,
                    changed=deleted_actual + added_actual, removed_edges=removed, added_edges=added,
                    born_signed_window_correlation=born_correlations,
                    mask_before_sha256=tensor_hash(before), mask_after_sha256=tensor_hash(self.mask),
                    **self.connection_counts())

    def connection_counts(self):
        return dict(connection_slots=int(self.mask.sum()), actual_nonzero=int(self.weight.ne(0).sum()),
                    per_output_slots=self.mask.sum(1).tolist(),
                    dormant_nonzero=int(self.weight[~self.mask].ne(0).sum()))


def teacher_residual(probabilities, targets):
    """The sole external teaching signal; no connection-importance labels."""
    return torch.nn.functional.one_hot(targets, probabilities.shape[1]).to(probabilities.dtype) - probabilities


def initialize_models(features, classes, fan_in, seed, scale):
    if not 0 < fan_in <= features:
        raise ValueError("fan_in must lie in [1, feature width]")
    rng = generator(seed)
    full_weight = nonzero_initial_weights((classes, features), rng, scale)
    mask = torch.zeros((classes, features), dtype=torch.bool)
    for output in range(classes):
        mask[output, torch.randperm(features, generator=rng)[:fan_in]] = True
    initial = SparseReadout(full_weight * mask, mask)
    models = {arm: initial.clone() for arm in ARMS if arm != "dense"}
    models["dense"] = SparseReadout(full_weight.clone(), torch.ones_like(mask))
    return models


@torch.no_grad()
def evaluate(model, task, split):
    """Evaluate both contexts on the same x without updating any learner state."""
    result = {}
    for name, context in (("A", 0), ("B", 1)):
        features, targets = task.batch(split, context=context)
        predictions = model.probabilities(features).argmax(-1)
        correct = int(predictions.eq(targets).sum())
        result[name] = dict(correct=correct, count=targets.numel(), accuracy=correct / targets.numel())
    return result


def source_hashes():
    paths = ("experiments/streaming/structure.py", "experiments/streaming/local_learning.py",
             "experiments/streaming/test_structure.py")
    return {path: file_hash(ROOT / path) for path in paths}


def run_seed(seed, args, directory, manifest):
    begin = time.perf_counter()
    task_seed, sample_seed, init_seed, rewire_seed = (args.task_seed_base + seed,
        args.sample_seed_base + seed, args.init_seed_base + seed, args.rewire_seed_base + seed)
    task = make_task(task_seed, input_dim=args.input_dim, classes=args.classes,
                     n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
                     context_scale=args.context_scale, device="cpu")
    one_feature, _ = task.batch("train", indices=torch.tensor([0]), context=0)
    width = one_feature.shape[1]
    if args.replace_per_output > min(args.fan_in, width - args.fan_in):
        raise ValueError("Replacement count must fit both the active and inactive per-output budgets")
    models = initialize_models(width, args.classes, args.fan_in, init_seed, args.initial_scale)
    initial_masks = {arm: model.mask.clone() for arm, model in models.items()}
    timings = {arm: dict(update_seconds=0., rewire_seconds=0., evaluation_seconds=0.) for arm in ARMS}
    curves, events, stages, training = ({arm: [] for arm in ARMS} for _ in range(4))
    topology_rngs = {arm: generator(rewire_seed) for arm in ARMS}
    phase_rng_states = {}
    data_digest = hashlib.sha256()

    def record_stage(stage, step):
        for arm, model in models.items():
            evaluation_begin = time.perf_counter()
            validation = evaluate(model, task, "val")
            record = dict(stage=stage, global_step=step, validation=validation,
                          mask_sha256=tensor_hash(model.mask), weight_sha256=tensor_hash(model.weight),
                          final_mask_distance_from_initial=int(model.mask.ne(initial_masks[arm]).sum()),
                          **model.connection_counts())
            if not args.pilot:
                record["test"] = evaluate(model, task, "test")
            timings[arm]["evaluation_seconds"] += time.perf_counter() - evaluation_begin
            stages[arm].append(record)
        print(json.dumps(dict(event="structure_stage", seed=seed, stage=stage, step=step,
            validation={arm: {context: score["accuracy"] for context, score in stages[arm][-1]["validation"].items()}
                        for arm in ARMS})), flush=True)

    record_stage("initial", 0)
    global_step = 0
    for phase, context in PHASES:
        sample_rng = generator(derived_seed(sample_seed, f"stream/{phase}"))
        for phase_step in range(1, args.steps_per_phase + 1):
            global_step += 1
            # One actual sample is shared by all four arms, not merely a claim
            # that independently seeded sample generators should agree.
            features, targets = sample_batch(task, "train", context, args.batch_size, sample_rng)
            data_digest.update(_bytes(features))
            data_digest.update(_bytes(targets))
            for arm, model in models.items():
                update_begin = time.perf_counter()
                probabilities = model.probabilities(features)
                residual = teacher_residual(probabilities, targets)
                model.update(features, residual, args.lr)
                timings[arm]["update_seconds"] += time.perf_counter() - update_begin
                if global_step % args.eval_every == 0 or phase_step == 1:
                    training[arm].append(dict(phase=phase, phase_step=phase_step, global_step=global_step,
                        accuracy=probabilities.argmax(-1).eq(targets).float().mean().item(),
                        negative_log_likelihood=-probabilities.gather(1, targets[:, None]).clamp_min(1e-12).log().mean().item()))
                if global_step % args.rewire_every == 0:
                    rewire_begin = time.perf_counter()
                    event = model.rewire("fixed" if arm == "dense" else arm,
                                         args.replace_per_output, topology_rngs[arm], args.initial_scale)
                    timings[arm]["rewire_seconds"] += time.perf_counter() - rewire_begin
                    events[arm].append(dict(phase=phase, global_step=global_step, **event))
                counts = model.connection_counts()
                expected = args.classes * (width if arm == "dense" else args.fan_in)
                if counts["connection_slots"] != expected or counts["dormant_nonzero"]:
                    raise RuntimeError("An arm violated its fixed connection budget")
                if global_step % args.eval_every == 0:
                    evaluation_begin = time.perf_counter()
                    curves[arm].append(dict(phase=phase, phase_step=phase_step, global_step=global_step,
                        validation=evaluate(model, task, "val"), **counts))
                    timings[arm]["evaluation_seconds"] += time.perf_counter() - evaluation_begin
        phase_rng_states[phase] = tensor_hash(sample_rng.get_state())
        record_stage(phase, global_step)

    if not torch.equal(topology_rngs["learned"].get_state(), topology_rngs["random"].get_state()):
        raise RuntimeError("Learned and random rewiring consumed different topology random streams")
    common = dict(seed=seed, configuration=vars(args), environment=manifest["environment"],
        sources=manifest["sources"], experiment_fingerprint=manifest["fingerprint"],
        feature_width=width, task_seed=task_seed, sample_seed=sample_seed, init_seed=init_seed,
        rewire_seed=rewire_seed, training_stream_sha256=data_digest.hexdigest(),
        sample_rng_state_sha256=tensor_hash(sample_rng.get_state()),
        phase_sample_rng_state_sha256=phase_rng_states,
        train_examples=global_step * args.batch_size,
        total_update_steps=global_step, test_evaluated=not args.pilot,
        elapsed_seed_seconds=time.perf_counter() - begin)
    results = []
    for arm, model in models.items():
        checkpoint = directory / f"{arm}-seed{seed}-last.pt"
        torch.save(dict(weight=model.weight, mask=model.mask, candidate_correlation=model.correlation,
                        window_batches=model.window_batches, **common), checkpoint)
        result = dict(arm=arm, **common, stages=stages[arm], curves=curves[arm], training=training[arm],
            rewiring=events[arm], timings=timings[arm],
            connections_removed=sum(event["removed"] for event in events[arm]),
            connections_added=sum(event["added"] for event in events[arm]),
            connections_changed=sum(event["changed"] for event in events[arm]),
            changed_rewire_events=sum(event["changed"] > 0 for event in events[arm]),
            scheduled_rewire_events=len(events[arm]),
            final_mask_distance_from_initial=int(model.mask.ne(initial_masks[arm]).sum()),
            candidate_statistic_scalars=model.correlation.numel(),
            allocated_weight_scalars=model.weight.numel(),
            active_weight_parameters=int(model.mask.sum()),
            checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
            selected_checkpoint="fixed_last", **model.connection_counts())
        atomic_json(directory / f"{arm}-seed{seed}.json", result)
        results.append(result)
    print(json.dumps(dict(event="structure_complete", seed=seed,
        changed={result["arm"]: result["connections_changed"] for result in results},
        actual_nonzero={result["arm"]: result["actual_nonzero"] for result in results},
        seconds=time.perf_counter() - begin)), flush=True)
    return results


def aggregate(results, pilot):
    split = "validation" if pilot else "test"
    summary = {}
    fixed = {result["seed"]: result for result in results if result["arm"] == "fixed"}
    for arm in ARMS:
        rows = [result for result in results if result["arm"] == arm]
        if not rows:
            continue
        for stage_index, stage_name in enumerate(("initial", "A1", "B", "A2")):
            for context in ("A", "B"):
                values = [row["stages"][stage_index][split][context]["accuracy"] for row in rows]
                delta = [value - fixed[row["seed"]]["stages"][stage_index][split][context]["accuracy"]
                         for row, value in zip(rows, values)]
                summary[f"{arm}/{stage_name}/{context}"] = dict(mean=statistics.mean(values),
                    std=statistics.stdev(values) if len(values) > 1 else 0., values=values,
                    seed_ids=[row["seed"] for row in rows], mean_delta_vs_fixed=statistics.mean(delta),
                    paired_delta_vs_fixed=delta, split=split,
                    parameter_budget_matched=arm != "dense")
    return summary


def integers(text):
    return [int(item) for item in text.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/streaming/structure-v1")
    parser.add_argument("--seeds", type=integers, default=[0, 1, 2])
    parser.add_argument("--pilot", action="store_true", help="Never evaluate the test split")
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--context-scale", type=float, default=.5)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--steps-per-phase", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fan-in", type=int, default=16)
    parser.add_argument("--rewire-every", type=int, default=50)
    parser.add_argument("--replace-per-output", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.2)
    parser.add_argument("--initial-scale", type=float, default=.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--task-seed-base", type=int, default=0)
    parser.add_argument("--sample-seed-base", type=int, default=0)
    parser.add_argument("--init-seed-base", type=int, default=7300000)
    parser.add_argument("--rewire-seed-base", type=int, default=7400000)
    parser.add_argument("--selection-note", default="Predeclared configuration; no test-based selection")
    args = parser.parse_args(argv)
    for name in ("input_dim", "classes", "n_train", "n_val", "n_test", "steps_per_phase", "batch_size",
                 "fan_in", "rewire_every", "replace_per_output", "eval_every", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.classes < 2:
        parser.error("--classes must be at least 2")
    for name in ("context_scale", "lr", "initial_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain distinct non-negative integers")
    width = 2 * args.input_dim + 2
    if args.replace_per_output > min(args.fan_in, width - args.fan_in):
        parser.error("Replacement count must fit the active and dormant per-output connection budgets")
    for base in (args.task_seed_base, args.sample_seed_base, args.init_seed_base, args.rewire_seed_base):
        if any(not 0 <= base + seed < 2**63 for seed in args.seeds):
            parser.error("RNG seeds must lie in [0, 2**63)")
    directory = Path(args.output).expanduser()
    args.output = str((directory if directory.is_absolute() else ROOT / directory).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    directory = Path(args.output)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("Output is not an empty directory; choose a new output path")
    torch.set_num_threads(args.threads)
    manifest = dict(configuration=vars(args), sources=source_hashes(), arms=list(ARMS),
        environment=dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                         device="cpu", threads=torch.get_num_threads()),
        teaching_signal="teacher one-hot minus predicted probability; externally supervised local delta rule",
        structural_rule="fixed budget: smallest |weight| drops; largest absolute window pre*residual correlation grows",
        historical_reference=dict(
            budget="src/core.py grow_edges uses active-trace threshold and random births; prune uses weight and age",
            homeo="src/core.py structural_homeo changes E-E counts using rate targets and tolerance",
            new_experiment="Neither historical rule is executed; no rate or age thresholds are introduced"),
        limitations=["Engineered signed context-interaction features; no LIF/firing-rate autonomy claim",
                     "Teacher residual is external class-wise feedback, not an artificial edge-importance label",
                     "Dense is a larger-budget feasibility control, not evidence for rewiring superiority",
                     "Dense allocated arrays and candidate statistics; no event-sparse efficiency claim",
                     "A-B-A is a finite interference/relearning probe, not lifelong learning",
                     "Only the fixed last model is saved; validation curves never select a checkpoint"],
        test_evaluation_enabled=not args.pilot)
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "manifest.json", manifest)
    begin = time.perf_counter()
    print(json.dumps(dict(event="structure_start", output=str(directory), seeds=args.seeds,
                          pilot=args.pilot, fingerprint=manifest["fingerprint"])), flush=True)
    results = []
    for seed in args.seeds:
        results.extend(run_seed(seed, args, directory, manifest))
        atomic_json(directory / "summary.json", dict(configuration=vars(args),
            experiment_fingerprint=manifest["fingerprint"], summary=aggregate(results, args.pilot),
            completed_runs=len(results), elapsed_seconds=time.perf_counter() - begin))
    print(json.dumps(dict(event="structure_finished", seconds=time.perf_counter() - begin,
                          output=str(directory))), flush=True)
    return results


if __name__ == "__main__":
    main()
