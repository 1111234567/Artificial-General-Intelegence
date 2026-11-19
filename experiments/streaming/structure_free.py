"""Large local-learning probe of gradual retirement versus immediate edge swaps.

All models use engineered context features and an externally supplied teacher
residual, not autonomous firing-frequency learning. Existing edges receive the
same local delta update followed by L1 soft shrinkage. Gradual-free retires an
edge only when its weight reaches zero, and grows dormant local proposals at a
fixed window using the same connection cost. No birth requires an old edge to
die. Immediate-budget instead keeps the strongest K proposals per output.

A static control receives each output's peak free connection count from the
training trajectory and repeats the same stream from initialization. Its larger
pool is a conservative capacity control, not complete causal isolation. Dense
arrays/GEMMs are used; active edges and allocated arrays are reported separately.
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
from torch.nn import functional as F

from experiments.streaming.local_learning import derived_seed, make_task


PRIMARY = ("fixed_sparse", "immediate_budget", "gradual_free", "dense")
ALL_ARMS = PRIMARY + ("peak_static",)
PHASES = (("A1", 0), ("B", 1), ("A2", 0))
RECOVERY_OFFSETS = (1, 5, 10, 25)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def cpu_generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def initial_arrays(width, classes, fan_in, seed, scale):
    rng = cpu_generator(derived_seed(seed, "free/initial"))
    full_weight = scale * torch.randn((classes, width), generator=rng)
    mask = torch.zeros((classes, width), dtype=torch.bool)
    for output in range(classes):
        mask[output, torch.randperm(width, generator=rng)[:fan_in]] = True
    return full_weight, mask


def peak_capacity_mask(initial_mask, peak_per_output, seed):
    """Give a static pool all original edges plus random additional capacity."""
    mask = initial_mask.cpu().clone()
    rng = cpu_generator(derived_seed(seed, "free/peak_capacity"))
    for output, requested in enumerate(peak_per_output):
        required = max(int(requested), int(mask[output].sum()))
        dormant = (~mask[output]).nonzero().flatten()
        if required > mask.shape[1]:
            raise ValueError("Peak capacity exceeds the possible feature connections")
        extra = required - int(mask[output].sum())
        mask[output, dormant[torch.randperm(dormant.numel(), generator=rng)[:extra]]] = True
    return mask


@dataclass
class Population:
    """Fused, equally ordered dense kernels for several masked readouts."""

    names: tuple[str, ...]
    weight: torch.Tensor                 # [arms, classes, features]
    pool: torch.Tensor                   # potential currently eligible edges
    initial_fan_in: int

    def __post_init__(self):
        if self.weight.ndim != 3 or self.pool.shape != self.weight.shape or self.pool.dtype != torch.bool:
            raise ValueError("Weight and boolean pool must have shape [arms, classes, features]")
        if len(self.names) != self.weight.shape[0]:
            raise ValueError("Each weight array needs an arm name")
        if bool(self.weight[~self.pool].ne(0).any()):
            raise ValueError("Dormant weights must be zero")
        self.candidate = torch.zeros_like(self.weight)
        self.window_steps = 0
        shape = self.weight.shape[:2]
        self.pool_born = torch.zeros(shape, dtype=torch.long, device=self.weight.device)
        self.pool_retired = torch.zeros_like(self.pool_born)
        self.zero_exits = torch.zeros_like(self.pool_born)
        self.budget_nonzero_cuts = torch.zeros_like(self.pool_born)
        self.actual_born = torch.zeros_like(self.pool_born)
        self.actual_retired = torch.zeros_like(self.pool_born)
        actual = self.weight.ne(0).sum(-1)
        self.peak_per_output = actual.clone()
        self.peak_total = actual.sum(-1)
        self.survival = {}

    @property
    def device(self):
        return self.weight.device

    def probabilities(self, features, *, weight_override=None):
        weight = self.weight if weight_override is None else weight_override
        arms, classes, width = weight.shape
        logits = features @ weight.reshape(arms * classes, width).T
        logits = logits.reshape(features.shape[0], arms, classes).permute(1, 0, 2)
        return logits.softmax(-1)

    def _observe_actual_change(self, before):
        after = self.weight.ne(0)
        self.actual_born.add_((~before & after).sum(-1))
        self.actual_retired.add_((before & ~after).sum(-1))
        counts = after.sum(-1)
        self.peak_per_output = torch.maximum(self.peak_per_output, counts)
        self.peak_total = torch.maximum(self.peak_total, counts.sum(-1))
        for reference in self.survival.values():
            reference.logical_and_(after)

    @torch.no_grad()
    def local_update(self, features, residual, learning_rate, connection_cost):
        """Only local pre activity and external per-output residual enter here."""
        before_actual = self.weight.ne(0)
        gradient = residual.permute(0, 2, 1) @ features / features.shape[0]
        self.candidate.add_(gradient * ~self.pool)
        updated = F.softshrink(self.weight + learning_rate * gradient * self.pool,
                               lambd=learning_rate * connection_cost)
        self.weight.copy_(updated * self.pool)
        if "gradual_free" in self.names:
            index = self.names.index("gradual_free")
            zero = self.pool[index] & self.weight[index].eq(0)
            self.pool_retired[index].add_(zero.sum(-1))
            self.zero_exits[index].add_(zero.sum(-1))
            self.pool[index].logical_and_(~zero)
            # A retired edge begins a fresh dormant trace on the NEXT trial;
            # no active-era activity is reused to resurrect it immediately.
            self.candidate[index].masked_fill_(zero, 0)
        self.window_steps += 1
        self._observe_actual_change(before_actual)

    @torch.no_grad()
    def structural_window(self, learning_rate, connection_cost):
        """One lambda, in matching per-step/window proximal-update units."""
        if self.window_steps < 1:
            raise ValueError("A structural window needs activity observations")
        before_pool = self.pool.clone()
        before_actual = self.weight.ne(0)
        proposal = F.softshrink(learning_rate * self.candidate,
                               lambd=learning_rate * self.window_steps * connection_cost)
        proposal.masked_fill_(self.pool, 0)
        for index, name in enumerate(self.names):
            if name == "gradual_free":
                born = ~self.pool[index] & proposal[index].ne(0)
                self.weight[index][born] = proposal[index][born]
                self.pool[index].logical_or_(born)
                # There is deliberately no deletion in response to a birth.
            elif name == "immediate_budget":
                combined = self.weight[index] + proposal[index]
                selected = combined.abs().topk(self.initial_fan_in, dim=-1).indices
                new_pool = torch.zeros_like(self.pool[index])
                new_pool.scatter_(-1, selected, True)
                cuts = self.pool[index] & ~new_pool & self.weight[index].ne(0)
                self.budget_nonzero_cuts[index].add_(cuts.sum(-1))
                self.pool[index].copy_(new_pool)
                self.weight[index].copy_(combined * new_pool)
        self.pool_born.add_((~before_pool & self.pool).sum(-1))
        self.pool_retired.add_((before_pool & ~self.pool).sum(-1))
        self._observe_actual_change(before_actual)
        change = dict(window_steps=self.window_steps,
            pool_added=(~before_pool & self.pool).sum(-1).cpu().tolist(),
            pool_removed=(before_pool & ~self.pool).sum(-1).cpu().tolist(),
            nonzero_removed_at_window=(before_actual & self.weight.eq(0)).sum(-1).cpu().tolist(),
            actual_nonzero=self.weight.ne(0).sum(-1).cpu().tolist())
        self.candidate.zero_()
        self.window_steps = 0
        return change

    def remember_survival(self, name):
        self.survival[name] = self.weight.ne(0).clone()

    def counts(self):
        actual = self.weight.ne(0)
        rows = {}
        for index, name in enumerate(self.names):
            rows[name] = dict(pool_connections=int(self.pool[index].sum()),
                actual_nonzero=int(actual[index].sum()),
                per_output_nonzero=actual[index].sum(-1).cpu().tolist(),
                peak_nonzero=int(self.peak_total[index]),
                peak_per_output=self.peak_per_output[index].cpu().tolist(),
                pool_added=int(self.pool_born[index].sum()), pool_retired=int(self.pool_retired[index].sum()),
                zero_exits=int(self.zero_exits[index].sum()),
                nonzero_budget_cuts=int(self.budget_nonzero_cuts[index].sum()),
                actual_zero_to_nonzero=int(self.actual_born[index].sum()),
                actual_nonzero_to_zero=int(self.actual_retired[index].sum()),
                dormant_nonzero=int(self.weight[index][~self.pool[index]].ne(0).sum()),
                continuous_reference_survival={key: int(mask[index].sum()) for key, mask in self.survival.items()})
        return rows


def initialize_population(full_weight, mask, device):
    pools = torch.stack([mask, mask, mask, torch.ones_like(mask)]).to(device)
    weights = full_weight.to(device).unsqueeze(0).expand(len(PRIMARY), -1, -1).clone() * pools
    return Population(PRIMARY, weights, pools, int(mask[0].sum()))


def teacher_residual(probabilities, targets):
    teacher = F.one_hot(targets, probabilities.shape[-1]).to(probabilities.dtype)
    return teacher.unsqueeze(0) - probabilities


@torch.no_grad()
def evaluate(population, task, split, indices=None, return_probabilities=False, *, weight_override=None):
    result = {name: {} for name in population.names}
    probabilities = {}
    for context, label in ((0, "A"), (1, "B")):
        features, targets = task.batch(split, indices=indices, context=context)
        probability = population.probabilities(features, weight_override=weight_override)
        predicted = probability.argmax(-1)
        correct = predicted.eq(targets[None]).sum(-1).cpu().tolist()
        nll = (-probability.gather(-1, targets[None, :, None].expand(len(population.names), -1, 1)).clamp_min(1e-12).log().mean((1, 2))).cpu().tolist()
        for index, name in enumerate(population.names):
            result[name][label] = dict(correct=correct[index], count=targets.numel(),
                                       accuracy=correct[index] / targets.numel(), nll=nll[index])
        if return_probabilities:
            probabilities[label] = probability
    return (result, probabilities) if return_probabilities else result


def compare_probe(population, before, after):
    result = {name: {} for name in population.names}
    for label in ("A", "B"):
        p, q = before[label], after[label]
        flip = p.argmax(-1).ne(q.argmax(-1)).float().mean(-1).cpu().tolist()
        variation = (0.5 * (p - q).abs().sum(-1)).mean(-1).cpu().tolist()
        for index, name in enumerate(population.names):
            result[name][label] = dict(prediction_flip_rate=flip[index], probability_total_variation=variation[index])
    return result


def dataset_hashes(task, chunk_size=8192):
    result = {"teacher": tensor_hash(task.teacher)}
    for name, split in task.splits.items():
        digest = hashlib.sha256()
        for offset in range(0, len(split.x), chunk_size):
            data = split.x[offset:offset + chunk_size].detach().cpu().contiguous()
            digest.update(data.numpy().tobytes())
        result[name] = dict(x_sha256=digest.hexdigest(), labels_sha256=tensor_hash(split.labels_a))
    return result


def source_hashes():
    return {name: file_hash(Path(__file__).parent / name) for name in
            ("structure_free.py", "test_structure_free.py", "local_learning.py")}


def environment(args):
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  torch=str(torch.__version__), device=args.device, threads=torch.get_num_threads(),
                  cuda=torch.version.cuda, float32_matmul_precision=torch.get_float32_matmul_precision(),
                  tf32=torch.backends.cuda.matmul.allow_tf32)
    if torch.device(args.device).type == "cuda":
        props = torch.cuda.get_device_properties(args.device)
        result.update(gpu=props.name, total_gpu_bytes=props.total_memory,
                      capability=list(torch.cuda.get_device_capability(args.device)))
    return result


@torch.no_grad()
def run_population(population, task, args, seed, label):
    """Fixed-last A-B-A stream; probe measurements never select updates."""
    probe = torch.arange(min(args.probe_size, args.n_val), device=population.device)
    results = {name: dict(stages=[], validation_curve=[], connection_curve=[],
                          structural_events=[], training=[]) for name in population.names}
    phase_streams = {}
    synchronize(population.device)
    begin = time.perf_counter()
    timers = dict(joint_update_seconds=0., joint_structure_and_probe_seconds=0.,
                  joint_evaluation_seconds=0., joint_recovery_probe_seconds=0.)
    global_step = 0

    def record_counts(step, phase, position="after_local_update"):
        counts = population.counts()
        for name in population.names:
            results[name]["connection_curve"].append(dict(step=step, phase=phase, position=position, **counts[name]))
        return counts

    def record_stage(stage):
        synchronize(population.device)
        start = time.perf_counter()
        validation = evaluate(population, task, "val")
        test = None if args.pilot else evaluate(population, task, "test")
        counts = record_counts(global_step, stage, "stage")
        synchronize(population.device)
        timers["joint_evaluation_seconds"] += time.perf_counter() - start
        for name in population.names:
            point = dict(stage=stage, step=global_step, validation=validation[name], **counts[name])
            if test is not None:
                point["test"] = test[name]
            results[name]["stages"].append(point)
        print(json.dumps(dict(event="free_stage", population=label, seed=seed, stage=stage,
            step=global_step, val={name: {k: v["accuracy"] for k, v in validation[name].items()}
                                  for name in population.names},
            nonzero={name: counts[name]["actual_nonzero"] for name in population.names})), flush=True)

    record_stage("initial")
    zero_at_previous_window = population.zero_exits.clone()
    for phase, context in PHASES:
        rng = torch.Generator(device=population.device).manual_seed(derived_seed(seed, f"stream/{phase}"))
        last_event = None
        phase_first_batch_hash = None
        synchronize(population.device)
        block_begin = time.perf_counter()
        for phase_step in range(1, args.steps_per_phase + 1):
            global_step += 1
            indices = torch.randint(args.n_train, (args.batch_size,), generator=rng, device=population.device)
            if phase_step == 1:
                phase_first_batch_hash = tensor_hash(indices)
            features, targets = task.batch("train", indices=indices, context=context)
            probability = population.probabilities(features)
            residual = teacher_residual(probability, targets)
            population.local_update(features, residual, args.lr, args.connection_cost)
            if global_step % args.connection_every == 0:
                record_counts(global_step, phase)

            at_window = population.window_steps == args.window or phase_step == args.steps_per_phase
            if at_window:
                synchronize(population.device)
                timers["joint_update_seconds"] += time.perf_counter() - block_begin
                structure_begin = time.perf_counter()
                before_weight = population.weight.clone()
                before_metrics, before_probability = evaluate(population, task, "val", probe, True)
                window_change = population.structural_window(args.lr, args.connection_cost)
                # Read-only counterfactual: apply only removal of old weights,
                # with no new weights. This is never a learning/selection signal.
                cut_metrics, cut_probability = evaluate(population, task, "val", probe, True,
                                                        weight_override=before_weight * population.pool)
                cut_impact = compare_probe(population, before_probability, cut_probability)
                after_metrics, after_probability = evaluate(population, task, "val", probe, True)
                impact = compare_probe(population, before_probability, after_probability)
                zero_exits = (population.zero_exits - zero_at_previous_window).sum(-1).cpu().tolist()
                zero_at_previous_window.copy_(population.zero_exits)
                counts = population.counts()
                for index, name in enumerate(population.names):
                    contexts = {}
                    for key in ("A", "B"):
                        contexts[key] = dict(before=before_metrics[name][key], after=after_metrics[name][key],
                            accuracy_change_pp=100 * (after_metrics[name][key]["accuracy"] - before_metrics[name][key]["accuracy"]),
                            cut_only=cut_metrics[name][key],
                            cut_only_accuracy_change_pp=100 * (cut_metrics[name][key]["accuracy"] - before_metrics[name][key]["accuracy"]),
                            cut_only_prediction_flip_rate=cut_impact[name][key]["prediction_flip_rate"],
                            cut_only_probability_total_variation=cut_impact[name][key]["probability_total_variation"],
                            **impact[name][key])
                    event = dict(step=global_step, phase=phase, phase_step=phase_step,
                        current_context="A" if context == 0 else "B", other_context="B" if context == 0 else "A",
                        probe_split="validation", probe_size=len(probe), window_steps=window_change["window_steps"],
                        pool_added_at_window=sum(window_change["pool_added"][index]),
                        pool_removed_at_window=sum(window_change["pool_removed"][index]),
                        nonzero_removed_at_window=sum(window_change["nonzero_removed_at_window"][index]),
                        zero_exits_since_previous_window=zero_exits[index],
                        per_output_pool_added=window_change["pool_added"][index],
                        per_output_pool_removed=window_change["pool_removed"][index],
                        probe=contexts, recovery=[], **counts[name])
                    results[name]["structural_events"].append(event)
                    results[name]["connection_curve"].append(dict(step=global_step, phase=phase,
                        position="after_structure", **counts[name]))
                last_event = global_step
                synchronize(population.device)
                timers["joint_structure_and_probe_seconds"] += time.perf_counter() - structure_begin
                block_begin = time.perf_counter()
            elif last_event is not None and global_step - last_event in RECOVERY_OFFSETS:
                synchronize(population.device)
                timers["joint_update_seconds"] += time.perf_counter() - block_begin
                recovery_begin = time.perf_counter()
                recovery = evaluate(population, task, "val", probe)
                for name in population.names:
                    results[name]["structural_events"][-1]["recovery"].append(
                        dict(offset=global_step - last_event, step=global_step, probe=recovery[name]))
                synchronize(population.device)
                timers["joint_recovery_probe_seconds"] += time.perf_counter() - recovery_begin
                block_begin = time.perf_counter()

            if phase_step % args.eval_every == 0 or phase_step == args.steps_per_phase:
                synchronize(population.device)
                timers["joint_update_seconds"] += time.perf_counter() - block_begin
                evaluation_begin = time.perf_counter()
                validation = evaluate(population, task, "val")
                last_nll = (-probability.gather(-1, targets[None, :, None].expand(len(population.names), -1, 1)).clamp_min(1e-12).log().mean((1, 2))).cpu().tolist()
                for index, name in enumerate(population.names):
                    results[name]["validation_curve"].append(dict(step=global_step, phase=phase,
                        phase_step=phase_step, validation=validation[name]))
                    results[name]["training"].append(dict(step=global_step, phase=phase,
                        phase_step=phase_step, last_batch_nll=last_nll[index]))
                synchronize(population.device)
                timers["joint_evaluation_seconds"] += time.perf_counter() - evaluation_begin
                block_begin = time.perf_counter()
                print(json.dumps(dict(event="free_progress", population=label, seed=seed, phase=phase,
                    phase_step=phase_step, seconds=time.perf_counter() - begin,
                    current_val={name: validation[name]["A" if context == 0 else "B"]["accuracy"]
                                 for name in population.names})), flush=True)
        synchronize(population.device)
        timers["joint_update_seconds"] += time.perf_counter() - block_begin
        phase_streams[phase] = dict(first_batch_indices_sha256=phase_first_batch_hash,
                                    final_rng_sha256=tensor_hash(rng.get_state()))
        record_stage(phase)
        if phase in {"A1", "B"}:
            population.remember_survival(f"{phase}_terminal")
    synchronize(population.device)
    seconds = time.perf_counter() - begin
    for name in population.names:
        results[name].update(joint_population_names=list(population.names),
            joint_population_seconds=seconds, joint_population_timings=timers,
            phase_streams=phase_streams, final_counts=population.counts()[name],
            actual_training_examples=3 * args.steps_per_phase * args.batch_size,
            update_steps=3 * args.steps_per_phase, selected_checkpoint="fixed_last")
    return results


def run_seed(seed, args, directory, manifest):
    synchronize(args.device)
    begin = time.perf_counter()
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    task = make_task(seed, input_dim=args.input_dim, classes=args.classes,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        context_scale=args.context_scale, device=args.device)
    hashes = dataset_hashes(task)
    full_weight, initial_mask = initial_arrays(task.feature_dim, args.classes, args.fan_in,
                                               seed, args.initial_scale)
    population = initialize_population(full_weight, initial_mask, args.device)
    primary = run_population(population, task, args, seed, "four_fused_primary")
    free_index = PRIMARY.index("gradual_free")
    peak = population.peak_per_output[free_index].cpu().tolist()
    static_mask = peak_capacity_mask(initial_mask, peak, seed).to(args.device)
    static = Population(("peak_static",), (full_weight.to(args.device) * static_mask).unsqueeze(0),
                        static_mask.unsqueeze(0), args.fan_in)
    matched = run_population(static, task, args, seed, "posthoc_train_trajectory_peak_capacity")
    if primary["gradual_free"]["phase_streams"] != matched["peak_static"]["phase_streams"]:
        raise RuntimeError("Primary and peak-capacity control did not consume identical data streams")
    synchronize(args.device)
    seconds = time.perf_counter() - begin
    peak_bytes = torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type == "cuda" else None
    common = dict(seed=seed, configuration=vars(args), environment=manifest["environment"],
        sources=manifest["sources"], experiment_fingerprint=manifest["fingerprint"],
        dataset_hashes=hashes, feature_width=task.feature_dim,
        possible_weight_parameters=args.classes * task.feature_dim,
        candidate_statistic_scalars=args.classes * task.feature_dim,
        initial_sparse_pool=int(initial_mask.sum()),
        peak_static_per_output_from_training=peak,
        peak_static_capacity_source="gradual_free per-output maximum actual nonzero over training, never test labels",
        test_evaluated=not args.pilot, seed_total_seconds=seconds, joint_peak_gpu_bytes=peak_bytes)
    results = []
    for pop, values in ((population, primary), (static, matched)):
        for index, name in enumerate(pop.names):
            checkpoint = directory / f"{name}-seed{seed}-last.pt"
            torch.save(dict(weight=pop.weight[index], pool=pop.pool[index], candidate=pop.candidate[index],
                            window_steps=pop.window_steps, arm=name, **common), checkpoint)
            result = dict(arm=name, **common, **values[name], checkpoint=str(checkpoint),
                          checkpoint_sha256=file_hash(checkpoint))
            atomic_json(directory / f"{name}-seed{seed}.json", result)
            results.append(result)
    print(json.dumps(dict(event="free_seed_complete", seed=seed, seconds=seconds,
        peak_gpu_bytes=peak_bytes, final_nonzero={r["arm"]: r["final_counts"]["actual_nonzero"] for r in results},
        free_zero_exits=primary["gradual_free"]["final_counts"]["zero_exits"],
        immediate_nonzero_cuts=primary["immediate_budget"]["final_counts"]["nonzero_budget_cuts"])), flush=True)
    return results


def mean_std(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0., values=values)


def aggregate(results, pilot):
    split = "validation" if pilot else "test"
    summary = {}
    for arm in ALL_ARMS:
        rows = [result for result in results if result["arm"] == arm]
        if not rows:
            continue
        for index, stage in enumerate(("initial", "A1", "B", "A2")):
            for context in ("A", "B"):
                values = [row["stages"][index][split][context]["accuracy"] for row in rows]
                summary[f"{arm}/{stage}/{context}"] = dict(**mean_std(values), split=split)
        for metric in ("actual_nonzero", "peak_nonzero", "pool_connections", "pool_added", "pool_retired",
                       "zero_exits", "nonzero_budget_cuts", "actual_zero_to_nonzero", "actual_nonzero_to_zero"):
            summary[f"{arm}/connections/{metric}"] = mean_std([row["final_counts"][metric] for row in rows])
        for phase, _ in PHASES:
            for context in ("A", "B"):
                for metric in ("accuracy_change_pp", "prediction_flip_rate", "probability_total_variation",
                               "cut_only_accuracy_change_pp", "cut_only_prediction_flip_rate",
                               "cut_only_probability_total_variation"):
                    values = []
                    for row in rows:
                        events = [event for event in row["structural_events"] if event["phase"] == phase]
                        values.append(statistics.mean(event["probe"][context][metric] for event in events))
                    summary[f"{arm}/transition/{phase}/{context}/{metric}"] = mean_std(values)
    return summary


def integers(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/streaming/structure-free-v1")
    parser.add_argument("--seeds", type=integers, default=[0, 1, 2])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Mark a timing smoke, disable test evaluation")
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--context-scale", type=float, default=.5)
    parser.add_argument("--n-train", type=int, default=65536)
    parser.add_argument("--n-val", type=int, default=8192)
    parser.add_argument("--n-test", type=int, default=8192)
    parser.add_argument("--steps-per-phase", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--fan-in", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.2)
    parser.add_argument("--connection-cost", type=float, default=.001)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--connection-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--probe-size", type=int, default=1024)
    parser.add_argument("--initial-scale", type=float, default=.01)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--selection-note", default="Configuration fixed before test evaluation")
    args = parser.parse_args(argv)
    args.pilot = args.pilot or args.benchmark
    for name in ("input_dim", "classes", "n_train", "n_val", "n_test", "steps_per_phase", "batch_size",
                 "fan_in", "window", "connection_every", "eval_every", "probe_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.classes < 2 or args.fan_in >= 2 * args.input_dim + 2:
        parser.error("Need >=2 classes and a sparse initial pool smaller than the feature width")
    for name in ("context_scale", "lr", "connection_cost", "initial_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    directory = Path(args.output).expanduser()
    args.output = str((directory if directory.is_absolute() else ROOT / directory).resolve())
    args.device = str(torch.device(args.device))
    return args


def main(argv=None):
    args = parse_args(argv)
    directory = Path(args.output)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("Output is not empty; choose a new output path")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision(args.matmul_precision)
    manifest = dict(configuration=vars(args), environment=environment(args), sources=source_hashes(),
        possible_weight_parameters=args.classes * (2 * args.input_dim + 2),
        potential_scale_vs_small_198=args.classes * (2 * args.input_dim + 2) / 198,
        arms=list(ALL_ARMS), teacher_feedback="one_hot(target)-probability; externally supervised local delta",
        gradual_rule="L1 soft shrink every active update; zero weights retire; inactive window L1 proposal grows independently",
        immediate_rule="same local L1 rule, but keep per-output topK of old weights and dormant proposals at windows",
        capacity_control="static pool from per-output peak actual free edges, containing original sparse initialization",
        timing="primary arms fused in shared kernels; CUDA synchronization at timing blocks; peak-static is a separate pass",
        probe="same fixed validation subset before/after windows, cut-only counterfactual, offsets1,5,10,25; no gradient or selection",
        limitations=["Single linear readout and engineered context interactions, not autonomous representation learning",
                     "Rules are designed by us, not algorithms autonomously invented by the model",
                     "Teacher residual is external supervision; this is not solely firing-frequency learning",
                     "Dense allocated weights and candidate arrays; actual edges do not imply sparse runtime savings",
                     "Gradual/free and immediate-budget also differ in capacity; peak-static only partially controls this",
                     "Static peak capacity is derived from a training trajectory; it is not a predeclared fixed-size causal control",
                     "Only fixed-last models; validation-only pilot; no test-based hyperparameter selection"],
        test_evaluation_enabled=not args.pilot)
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "manifest.json", manifest)
    print(json.dumps(dict(event="free_start", output=str(directory), seeds=args.seeds,
        device=args.device, benchmark=args.benchmark, pilot=args.pilot, fingerprint=manifest["fingerprint"])), flush=True)
    begin = time.perf_counter()
    results = []
    for seed in args.seeds:
        results.extend(run_seed(seed, args, directory, manifest))
        atomic_json(directory / "summary.json", dict(configuration=vars(args),
            experiment_fingerprint=manifest["fingerprint"], summary=aggregate(results, args.pilot),
            completed_runs=len(results), elapsed_seconds=time.perf_counter() - begin))
    print(json.dumps(dict(event="free_finished", output=str(directory), seconds=time.perf_counter() - begin)), flush=True)
    return results


if __name__ == "__main__":
    main()
