"""Reproducible GPU/CPU experiments; all synthetic targets come from the world.

Run from any directory. Large artifacts stay under ignored data/persistent/.
The evaluator never trains and never supplies correctness to model.forward().
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from experiments.persistent.models import (MemoryLoop, CompressedGRU,
                                          selected_logits, parameter_count)
from experiments.persistent.tasks import make_batch


def integers(value):
    return [int(item) for item in value.split(",")]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def build_model(mode, args):
    if mode == "gru":
        return CompressedGRU(args.vocab, args.dim, args.gru_hidden)
    return MemoryLoop(args.vocab, args.dim,
                      "judge" if mode == "judge_aux" else mode,
                      judge_feedback=mode != "judge_aux")


def call_model(model, batch, *, steps=None, feedback_override=None):
    kwargs = {"steps": steps}
    if isinstance(model, CompressedGRU):
        kwargs["gap"] = batch.gap
    else:
        kwargs["feedback_override"] = feedback_override
    return model(batch.keys, batch.values, batch.query, batch.hops, **kwargs)


def source_hashes():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("models.py", "run.py", "tasks.py")}


def environment(device):
    result = dict(python=sys.version, platform=platform.platform(),
                  torch=torch.__version__, numpy=np.__version__, device=str(device),
                  cuda=torch.version.cuda, threads=torch.get_num_threads())
    if str(device).startswith("cuda"):
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_memory_bytes=props.total_memory,
                      capability=torch.cuda.get_device_capability(device))
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def fit_position_prior(model, args, seed):
    """A judge baseline using only (current step - requested step).

    Fit on fresh in-distribution validation worlds, never the test labels.
    Used for every test case, including long loops and corrupted memory.
    """
    generator = torch.Generator(device=args.device).manual_seed(seed)
    totals, positives = {}, {}
    for _ in range(4):
        batch = make_batch(args.eval_batch_size, 16, 32, args.max_hops,
                           args.vocab, args.device, generator)
        output = call_model(model, batch, steps=args.max_hops)
        correct = (output.logits.argmax(-1) == batch.target[:, None]).float()
        for step in range(args.max_hops):
            delta = step + 1 - batch.hops
            for offset in range(1-args.max_hops, args.max_hops):
                mask = delta == offset
                totals[offset] = totals.get(offset, 0) + mask.sum().item()
                positives[offset] = positives.get(offset, 0) + correct[:, step][mask].sum().item()
    return {offset: positives[offset] / totals[offset] for offset in totals if totals[offset]}


@torch.no_grad()
def evaluate(model, args, *, seed, cases=None):
    model.eval()
    position_prior = fit_position_prior(model, args, seed + 10000)
    if cases is None:
        cases = [
            ("small_gap0", 4, 0, 1), ("small_gap16", 4, 16, 1),
            ("small_gap256", 4, 256, 1), ("small_gap4096", 4, 4096, 1),
            ("one_hop", 16, 32, 1), ("two_hop", 16, 32, 2),
            ("four_hop", 16, 32, 4), ("delay256", 16, 256, 4),
            ("delay1024", 16, 1024, 4), ("delay4096", 16, 4096, 4),
            ("capacity32", 32, 32, 4), ("capacity64", 64, 32, 4),
            ("eight_hop", 16, 32, 8), ("corrupt_memory", 16, 32, 4),
        ]
        if isinstance(model, MemoryLoop) and model.mode == "judge":
            cases += [("feedback_zero", 16, 32, 4),
                      ("feedback_shuffle", 16, 32, 4)]
    records = []
    for name, count, gap, hops in cases:
        if count > args.vocab:
            continue
        # Same worlds across model modes, delays and feedback interventions.
        generator = torch.Generator(device=args.device).manual_seed(seed)
        correct = total = selected_correct = identity_correct = 0
        nonreturn_correct = nonreturn_count = 0
        confidence, outcomes = [], []
        strata = {i: [0.0, 0.0, 0, 0.0] for i in range(hops)}
        synchronize(args.device)
        begin = time.perf_counter()
        for batch_index in range(args.eval_batches):
            batch = make_batch(args.eval_batch_size, count, gap, hops, args.vocab,
                               args.device, generator, fixed_hops=hops)
            if name == "corrupt_memory":
                # Incorrect fact values, original query and target unchanged.
                batch.values = batch.values.roll(1, 0)
            output = call_model(model, batch, steps=hops,
                                feedback_override={"feedback_zero": "zero",
                                                   "feedback_shuffle": "shuffle"}.get(name))
            prediction = selected_logits(output, batch.hops).argmax(-1)
            correct += (prediction == batch.target).sum().item()
            identity_correct += (batch.query == batch.target).sum().item()
            no_return = ~(batch.path_targets == batch.query[:, None]).any(-1)
            nonreturn_correct += ((prediction == batch.target) & no_return).sum().item()
            nonreturn_count += no_return.sum().item()
            total += batch.target.numel()
            all_correct = (output.logits.argmax(-1) == batch.target[:, None]).float()
            probabilities = output.judgment.sigmoid()
            chosen = probabilities.argmax(-1)
            selected_correct += all_correct.gather(1, chosen[:, None]).sum().item()
            confidence.append(probabilities[:, -1].cpu())
            outcomes.append(all_correct[:, -1].cpu())
            for step in range(hops):
                strata[step][0] += ((probabilities[:, step] - all_correct[:, step]) ** 2).sum().item()
                strata[step][1] += all_correct[:, step].sum().item()
                strata[step][2] += batch.target.numel()
                prior = position_prior.get(step + 1 - hops, 0.0)
                strata[step][3] += ((prior - all_correct[:, step]) ** 2).sum().item()
        synchronize(args.device)
        p, y = torch.cat(confidence), torch.cat(outcomes)
        ece = 0.0
        bins = (p * 10).long().clamp(max=9)
        for index in range(10):
            mask = bins == index
            if mask.any():
                ece += mask.float().mean().item() * abs(p[mask].mean().item() - y[mask].mean().item())
        record = dict(case=name, n_records=count, gap=gap, hops=hops,
                      count=total, correct=correct, accuracy=correct / total,
                      identity_accuracy=identity_correct / total,
                      no_return_count=nonreturn_count,
                      no_return_accuracy=nonreturn_correct / nonreturn_count if nonreturn_count else None,
                      random_valid_key_accuracy=1/count,
                      judge_selected_accuracy=selected_correct / total,
                      final_judge_brier=((p-y)**2).mean().item(),
                      final_position_prior=position_prior[0],
                      final_position_brier=((position_prior[0]-y)**2).mean().item(),
                      final_judge_ece=ece, final_judge_mean=p.mean().item(),
                      seconds=time.perf_counter()-begin,
                      per_step={str(k+1): {"brier": v[0]/v[2], "accuracy": v[1]/v[2],
                                           "position_brier": v[3]/v[2]}
                                for k, v in strata.items()})
        records.append(record)
    return records


def train_one(mode, seed, args, directory, manifest):
    seed_all(seed)
    model = build_model(mode, args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    generator = torch.Generator(device=args.device).manual_seed(100000 + seed)
    chooser = random.Random(200000 + seed)
    curves = []
    synchronize(args.device)
    begin = time.perf_counter()
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(1, args.steps + 1):
        count = chooser.choice(args.train_records)
        gap = chooser.choice(args.train_gaps)
        hops = 1 if mode == "single" else args.max_hops
        batch = make_batch(args.batch_size, count, gap, hops, args.vocab,
                           args.device, generator)
        output = call_model(model, batch, steps=hops)
        prediction = selected_logits(output, batch.hops)
        answer_loss = F.cross_entropy(prediction, batch.target)
        loss = answer_loss
        judge_loss = loss.new_zeros(())
        if mode in {"judge", "judge_aux"}:
            # The self-judgment is grounded by independently generated truth.
            # No true feedback is given to the next forward computation step.
            judge_target = (output.logits.detach().argmax(-1) == batch.target[:, None]).float()
            judge_loss = F.binary_cross_entropy_with_logits(output.judgment, judge_target)
            loss = loss + args.judge_weight * judge_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            synchronize(args.device)
            point = dict(step=step, loss=loss.item(), answer_loss=answer_loss.item(),
                         judge_loss=judge_loss.item(), grad_norm=grad_norm.item(),
                         accuracy=(prediction.argmax(-1) == batch.target).float().mean().item(),
                         seconds=time.perf_counter()-begin)
            curves.append(point)
            print(json.dumps(dict(event="train", mode=mode, seed=seed, **point)), flush=True)
    synchronize(args.device)
    train_seconds = time.perf_counter() - begin
    peak = torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") else None
    checkpoint = directory / f"{mode}-seed{seed}.pt"
    torch.save(dict(model=model.state_dict(), mode=mode, seed=seed, args=vars(args)), checkpoint)
    evaluations = evaluate(model, args, seed=900000 + seed)
    result = dict(mode=mode, seed=seed, parameters=parameter_count(model),
                  train_seconds=train_seconds, peak_gpu_bytes=peak,
                  train_examples=args.steps * args.batch_size,
                  train_internal_steps=args.steps * args.batch_size * (1 if mode == "single" else args.max_hops),
                  curves=curves, evaluations=evaluations,
                  checkpoint=str(checkpoint), sources=manifest["sources"],
                  experiment_fingerprint=manifest["fingerprint"])
    atomic_json(directory / f"{mode}-seed{seed}.json", result)
    print(json.dumps(dict(event="complete", mode=mode, seed=seed,
                          scores={e["case"]: round(e["accuracy"], 4) for e in evaluations})), flush=True)
    return result


def aggregate(results):
    summary = {}
    for result in results:
        for evaluation in result["evaluations"]:
            key = f'{result["mode"]}/{evaluation["case"]}'
            summary.setdefault(key, []).append(evaluation["accuracy"])
    return {key: dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values)>1 else 0,
                      seeds=len(values), values=values) for key, values in summary.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/persistent/main")
    parser.add_argument("--modes", default="single,detach,joint,judge_aux,judge,gru")
    parser.add_argument("--seeds", type=integers, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--train-records", type=integers, default=[8, 16, 24])
    parser.add_argument("--train-gaps", type=integers, default=[0, 16, 32])
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--judge-weight", type=float, default=.2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="skip completed runs only if args and source hashes match")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    directory = Path(args.output)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(directory.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new path or use --resume")
    arguments = {k: v for k, v in vars(args).items() if k != "resume"}
    manifest = dict(arguments=arguments, environment=environment(args.device), sources=source_hashes(),
                    task="random permutation facts; final answer only; fresh worlds each batch",
                    limitations=["lossless fact writes are engineered, not learned", "NOOP does not overwrite archived facts",
                                 "GRU and archive differ in memory capacity; only detach/joint isolate backward connectivity",
                                 "single uses one-hop training; its training compute and task distribution differ",
                                 "judge auxiliary targets use task truth; this is not DeepSeek replication or self-reward RL",
                                 "no continual-learning or language capability claim"])
    manifest_path = directory / "manifest.json"
    manifest["fingerprint"] = hashlib.sha256(json.dumps(
        {k: manifest[k] for k in ("arguments", "environment", "sources")},
        sort_keys=True).encode()).hexdigest()
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError("Cannot resume with different arguments, source code or environment; use a new output directory")
    elif args.resume and any(directory.iterdir()):
        raise ValueError("Nonempty output directory has no manifest; choose a new output directory")
    atomic_json(manifest_path, manifest)
    print(json.dumps(dict(event="start", **manifest)), flush=True)
    results = []
    for seed in args.seeds:
        for mode in args.modes.split(","):
            path = directory / f"{mode}-seed{seed}.json"
            if args.resume and path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("experiment_fingerprint") != manifest["fingerprint"]:
                    raise ValueError(f"Stale or incompatible per-run result: {path}")
            else:
                result = train_one(mode, seed, args, directory, manifest)
            results.append(result)
            atomic_json(directory / "summary.json", aggregate(results))
    print(json.dumps(dict(event="finished", output=str(directory))), flush=True)


if __name__ == "__main__":
    main()
