"""Adapt a one-hop reader using real binary feedback on a harder graph task.

The three arms start from deep copies of one locally generated checkpoint:
frozen, REINFORCE with a leave-one-out reward baseline, and REINFORCE with a
shared-core critic/feedback head.  Targets enter adaptation only through the
environment's binary (sampled_action == target) reward.  There is no answer
cross-entropy, intermediate supervision, or substitution of predicted rewards
for real feedback.  This is a small task adaptation check, not a reproduction
of DeepSeek or a claim about lifelong learning.

Example (formal runs should use a fresh output directory):
    python experiments/persistent/adapt.py --output data/persistent/adapt-v1

Small CPU smoke:
    python experiments/persistent/adapt.py --device cpu --seeds 0 --steps 2 \
        --batch-size 4 --eval-batch-size 8 --eval-batches 1 --threads 1 \
        --curve-steps 1,2 --log-every 1 --output data/persistent/adapt-smoke
"""

from __future__ import annotations

import argparse
import copy
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

from experiments.persistent.models import MemoryLoop, parameter_count
from experiments.persistent.tasks import make_batch


ARMS = ("frozen", "reward", "critic")
CASES = (("old", 16, 1), ("new", 64, 4))
NEW_RECORDS, NEW_HOPS, VOCAB = 64, 4, 64


def integers(value):
    return [int(item) for item in value.split(",") if item.strip()]


def absolute_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(bytes(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    result = dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                  device=str(device), cuda=torch.version.cuda, threads=torch.get_num_threads(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32)
    if torch.device(device).type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, gpu_memory_bytes=properties.total_memory,
                      capability=list(torch.cuda.get_device_capability(device)))
    return result


def source_hashes():
    return {name: file_hash(Path(__file__).parent / name)
            for name in ("adapt.py", "models.py", "tasks.py")}


def load_reader(checkpoint, expected_seed):
    """Load only a user-selected, locally generated single-hop checkpoint."""
    # Explicitly permitted for this repository's own local checkpoint format,
    # which stores an argparse configuration together with the state dict.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"model", "args", "mode", "seed"} <= payload.keys():
        raise ValueError(f"Not a persistent experiment checkpoint: {checkpoint}")
    if payload["mode"] != "single" or int(payload["seed"]) != expected_seed:
        raise ValueError(f"Expected single-seed{expected_seed} checkpoint: {checkpoint}")
    configuration = payload["args"]
    configuration = vars(configuration) if isinstance(configuration, argparse.Namespace) else configuration
    if not isinstance(configuration, dict):
        raise ValueError("Checkpoint args must be a dictionary or argparse.Namespace")
    checkpoint_args = argparse.Namespace(**configuration)
    if checkpoint_args.vocab != VOCAB:
        raise ValueError(f"This adaptation experiment requires a {VOCAB}-item vocabulary")
    # 'single' describes the source training distribution, not an inference
    # step limit.  Joint and single have identical numerical reader forwards.
    reader = MemoryLoop(checkpoint_args.vocab, checkpoint_args.dim, mode="joint", judge_feedback=False)
    reader.load_state_dict(payload["model"], strict=True)
    reader.requires_grad_(True)
    return reader, dict(configuration)


def model_output(model, batch, steps):
    """The complete model-input boundary: neither gold tensor is passed."""
    return model(batch.keys, batch.values, batch.query, batch.hops, steps=steps)


def real_feedback(actions, target):
    """Environment truth is exposed to the learner only as a binary reward."""
    return (actions == target).to(dtype=torch.float32).detach()


def reinforce_objective(logits, score, actions, rewards, arm, entropy_weight, critic_weight):
    """An objective that cannot inspect answer targets or intermediate paths."""
    log_probabilities = logits.log_softmax(-1)
    probabilities = log_probabilities.exp()
    chosen_log_probability = log_probabilities.gather(1, actions[:, None]).squeeze(1)
    entropy = -(probabilities * log_probabilities).sum(-1).mean()
    if arm == "reward":
        if rewards.numel() < 2:
            raise ValueError("Leave-one-out REINFORCE requires batch_size >= 2")
        baseline = (rewards.sum() - rewards) / (rewards.numel() - 1)
        critic_loss = logits.new_zeros(())
    elif arm == "critic":
        baseline = score.sigmoid()
        critic_loss = F.binary_cross_entropy_with_logits(score, rewards)
    else:
        raise ValueError(f"No adaptation objective for arm {arm!r}")
    advantage = (rewards - baseline.detach()).detach()
    policy_loss = -(advantage * chosen_log_probability).mean()
    loss = policy_loss - entropy_weight * entropy + critic_weight * critic_loss
    metrics = dict(loss=loss.detach().item(), policy_loss=policy_loss.detach().item(),
                   critic_loss=critic_loss.detach().item(), entropy=entropy.detach().item(),
                   sampled_reward=rewards.mean().item(), baseline=baseline.detach().mean().item(),
                   advantage=advantage.mean().item())
    return loss, metrics


@torch.no_grad()
def evaluate(model, args, seed):
    """Greedy success on fresh worlds; this evaluation never updates weights."""
    previous_training = model.training
    model.eval()
    results = {}
    try:
        for name, count, hops in CASES:
            generator = torch.Generator(device=args.device).manual_seed(seed)
            correct = total = identity = no_return_correct = no_return_total = 0
            for _ in range(args.eval_batches):
                batch = make_batch(args.eval_batch_size, count, args.gap, hops, VOCAB,
                                   args.device, generator, fixed_hops=hops)
                output = model_output(model, batch, steps=hops)
                predicted = output.logits[:, -1].argmax(-1)
                success = predicted == batch.target
                no_return = ~(batch.path_targets == batch.query[:, None]).any(-1)
                correct += success.sum().item()
                identity += (batch.query == batch.target).sum().item()
                no_return_correct += (success & no_return).sum().item()
                no_return_total += no_return.sum().item()
                total += predicted.numel()
            results[name] = dict(n_records=count, hops=hops, gap=args.gap, seed=seed,
                                 count=total, correct=correct, accuracy=correct / total,
                                 identity_accuracy=identity / total,
                                 no_return_count=no_return_total,
                                 no_return_accuracy=no_return_correct / no_return_total if no_return_total else None)
    finally:
        model.train(previous_training)
    return results


def run_arm(model, arm, seed, args, directory, manifest, source, initial):
    synchronize(args.device)
    begin = time.perf_counter()
    optimizer = None if arm == "frozen" else torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    task_seed = args.train_seed_base + seed
    sampling_seed = args.sampling_seed_base + seed
    task_rng = torch.Generator(device=args.device).manual_seed(task_seed)
    sampling_rng = torch.Generator(device=args.device).manual_seed(sampling_seed)
    curves = [dict(step=0, evaluation=initial, split="validation")]
    training = []
    curve_steps = sorted({step for step in args.curve_steps if 0 < step <= args.steps} | {args.steps})
    model.train(arm != "frozen")
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    if arm == "frozen":
        # Reuse the exact same fixed-world scores: no optimization or changing
        # random worlds can produce an artificial frozen-arm learning curve.
        curves.extend(dict(step=step, evaluation=initial, split="validation") for step in curve_steps)
    else:
        for step in range(1, args.steps + 1):
            batch = make_batch(args.batch_size, NEW_RECORDS, args.gap, NEW_HOPS, VOCAB,
                               args.device, task_rng, fixed_hops=NEW_HOPS)
            output = model_output(model, batch, steps=NEW_HOPS)
            logits, score = output.logits[:, -1], output.judgment[:, -1]
            actions = torch.multinomial(logits.detach().softmax(-1), 1,
                                        generator=sampling_rng).squeeze(1)
            reward = real_feedback(actions, batch.target)
            loss, metrics = reinforce_objective(logits, score, actions, reward, arm,
                                                args.entropy_weight, args.critic_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                synchronize(args.device)
                point = dict(step=step, **metrics, grad_norm=grad_norm.item(),
                             seconds=time.perf_counter() - begin)
                training.append(point)
                print(json.dumps(dict(event="adapt_train", arm=arm, seed=seed, **point)), flush=True)
            if step in curve_steps:
                validation = evaluate(model, args, args.validation_seed_base + seed)
                curves.append(dict(step=step, evaluation=validation, split="validation"))
                print(json.dumps(dict(event="adapt_validation", arm=arm, seed=seed, step=step,
                                      accuracy={k: v["accuracy"] for k, v in validation.items()})), flush=True)
    synchronize(args.device)
    adaptation_seconds = time.perf_counter() - begin
    peak_gpu_bytes = (torch.cuda.max_memory_allocated(args.device)
                      if torch.device(args.device).type == "cuda" else None)
    final = evaluate(model, args, args.test_seed_base + seed)
    checkpoint = directory / f"{arm}-seed{seed}-last.pt"
    torch.save(dict(model=model.state_dict(), mode=model.mode, judge_feedback=model.judge_feedback,
                    arm=arm, seed=seed, args=vars(args), checkpoint_args=source["checkpoint_args"],
                    source=source, last_step=args.steps if optimizer is not None else 0,
                    experiment_fingerprint=manifest["fingerprint"]), checkpoint)
    synchronize(args.device)
    result = dict(arm=arm, seed=seed, configuration=vars(args), environment=manifest["environment"],
                  sources=manifest["sources"], experiment_fingerprint=manifest["fingerprint"],
                  source=source, parameters=parameter_count(model),
                  initial_state_sha256=source["initial_state_sha256"], final_state_sha256=state_hash(model),
                  reader_mode=model.mode, judge_feedback=model.judge_feedback,
                  initial=initial, initial_split="validation", curves=curves,
                  final=final, final_split="heldout_test", selected_checkpoint="last",
                  last_step=args.steps if optimizer is not None else 0,
                  train_examples=args.steps * args.batch_size if optimizer is not None else 0,
                  task_seed=task_seed, sampling_seed=sampling_seed,
                  task_rng_state_sha256=tensor_hash(task_rng.get_state()),
                  sampling_rng_state_sha256=tensor_hash(sampling_rng.get_state()),
                  training=training, adaptation_seconds=adaptation_seconds,
                  elapsed_seconds=time.perf_counter() - begin, peak_gpu_bytes=peak_gpu_bytes,
                  checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint))
    atomic_json(directory / f"{arm}-seed{seed}.json", result)
    print(json.dumps(dict(event="adapt_complete", arm=arm, seed=seed,
                          final={k: v["accuracy"] for k, v in final.items()})), flush=True)
    return result


def run_seed(seed, args, directory, manifest):
    source_checkpoint = next(item for item in manifest["checkpoints"] if item["seed"] == seed)
    reader, checkpoint_args = load_reader(source_checkpoint["path"], seed)
    initial_hash = state_hash(reader)
    models = {arm: copy.deepcopy(reader).to(args.device) for arm in ARMS}
    del reader
    for arm, model in models.items():
        model.mode = "judge" if arm == "critic" else "joint"
        model.judge_feedback = arm == "critic"
        model.requires_grad_(True)
        if state_hash(model) != initial_hash:
            raise RuntimeError("An adaptation arm did not start from identical checkpoint parameters")
    source = dict(**source_checkpoint, checkpoint_args=checkpoint_args,
                  initial_state_sha256=initial_hash, source_training_mode="single")
    # Complete and save all three variant-specific step-zero evaluations before
    # any variant updates a parameter. Critic feedback can change the initial
    # forward despite every initial parameter being exactly identical.
    initials = {arm: evaluate(model, args, args.validation_seed_base + seed)
                for arm, model in models.items()}
    atomic_json(directory / f"initial-seed{seed}.json",
                dict(seed=seed, source=source, split="validation", evaluations=initials,
                     initial_state_sha256=initial_hash,
                     note="Critic enables judge feedback at step 0; equal weights need not imply equal outputs.",
                     configuration=vars(args), environment=manifest["environment"],
                     sources=manifest["sources"], experiment_fingerprint=manifest["fingerprint"]))
    print(json.dumps(dict(event="adapt_initial", seed=seed,
                          accuracy={arm: {k: v["accuracy"] for k, v in values.items()}
                                    for arm, values in initials.items()})), flush=True)
    results = [run_arm(models[arm], arm, seed, args, directory, manifest, source, initials[arm])
               for arm in ARMS]
    trained = {result["arm"]: result for result in results if result["arm"] != "frozen"}
    for field in ("task_rng_state_sha256", "sampling_rng_state_sha256"):
        if trained["reward"][field] != trained["critic"][field]:
            raise RuntimeError(f"Reward and critic consumed different random streams: {field}")
    if results[0]["final_state_sha256"] != initial_hash:
        raise RuntimeError("Frozen model parameters changed")
    return results


def aggregate(results):
    summary = {}
    frozen = {result["seed"]: result for result in results if result["arm"] == "frozen"}
    for arm in ARMS:
        rows = [result for result in results if result["arm"] == arm]
        for case in ("old", "new"):
            values = [row["final"][case]["accuracy"] for row in rows]
            if not values:
                continue
            paired = [row["final"][case]["accuracy"] - frozen[row["seed"]]["final"][case]["accuracy"]
                      for row in rows]
            summary[f"{arm}/{case}"] = dict(mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0, seeds=len(values), values=values,
                seed_ids=[row["seed"] for row in rows], mean_delta_vs_frozen=statistics.mean(paired),
                paired_delta_vs_frozen=paired)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", default="data/persistent/main-v1")
    parser.add_argument("--checkpoint-pattern", default="single-seed{seed}.pt")
    parser.add_argument("--output", default="data/persistent/adapt-v1")
    parser.add_argument("--seeds", type=integers, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=.01)
    parser.add_argument("--critic-weight", type=float, default=.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gap", type=int, default=32)
    parser.add_argument("--curve-steps", type=integers, default=[100, 250, 500])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--train-seed-base", type=int, default=1900000)
    parser.add_argument("--sampling-seed-base", type=int, default=2900000)
    parser.add_argument("--validation-seed-base", type=int, default=3900000)
    parser.add_argument("--test-seed-base", type=int, default=4900000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    for name in ("steps", "eval_batch_size", "eval_batches", "log_every", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for a leave-one-out reward baseline")
    if args.gap < 0 or any(step < 0 for step in args.curve_steps):
        parser.error("gap and curve steps must be non-negative")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be a nonempty list of distinct non-negative integers")
    for name in ("lr", "grad_clip", "weight_decay", "entropy_weight", "critic_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in {"lr", "grad_clip"} and value == 0):
            parser.error(f"Invalid --{name.replace('_', '-')}")
    streams = [{getattr(args, name) + seed for seed in args.seeds}
               for name in ("train_seed_base", "sampling_seed_base", "validation_seed_base", "test_seed_base")]
    if any(seed < 0 or seed >= 2**63 for stream in streams for seed in stream):
        parser.error("Random seeds must lie in [0, 2**63)")
    if any(streams[i] & streams[j] for i in range(len(streams)) for j in range(i + 1, len(streams))):
        parser.error("Task, sampling, validation, and heldout test random streams must be disjoint")
    previous_test_seeds = {900000 + seed for seed in args.seeds}
    if previous_test_seeds & (streams[2] | streams[3]):
        parser.error("Adaptation evaluation seeds must differ from the main/pilot 900000 seeds")
    args.checkpoint_dir = str(absolute_path(args.checkpoint_dir))
    args.output = str(absolute_path(args.output))
    args.device = str(torch.device(args.device))
    return args


def main(argv=None):
    args = parse_args(argv)
    directory = Path(args.output)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("Output path is not an empty directory; choose a new output path")
    checkpoints = []
    for seed in args.seeds:
        checkpoint = (Path(args.checkpoint_dir) / args.checkpoint_pattern.format(seed=seed)).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing local source checkpoint: {checkpoint}")
        checkpoints.append(dict(seed=seed, path=str(checkpoint), sha256=file_hash(checkpoint),
                                bytes=checkpoint.stat().st_size))
    torch.set_num_threads(args.threads)
    directory.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    manifest = dict(configuration=vars(args), environment=environment(args.device), sources=source_hashes(),
        checkpoints=checkpoints, task=dict(n_records=NEW_RECORDS, fixed_hops=NEW_HOPS, vocab=VOCAB),
        arms=list(ARMS), reward="binary sampled_action == target; real environment feedback only",
        reward_baseline="leave-one-out batch mean; detached advantage",
        critic="final-step score predicts sampled binary reward; earlier scores feed the next internal step",
        checkpoint_selection="fixed last; validation curves never choose a checkpoint",
        limitations=["Small synthetic task adaptation, not DeepSeek reproduction or lifelong learning",
                     "Lossless record storage and writes are engineered",
                     "All arms start with identical weights; critic enables feedback and can differ at step 0",
                     "The critic score does not observe the sampled action; it estimates policy success probability",
                     "Shared-core BCE and score feedback change together in the critic arm"])
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(directory / "manifest.json", manifest)
    print(json.dumps(dict(event="adapt_start", output=str(directory), seeds=args.seeds,
                          steps=args.steps, fingerprint=manifest["fingerprint"])), flush=True)
    results = []
    for seed in args.seeds:
        results.extend(run_seed(seed, args, directory, manifest))
        atomic_json(directory / "summary.json", dict(configuration=vars(args),
            experiment_fingerprint=manifest["fingerprint"], summary=aggregate(results),
            completed_runs=len(results), elapsed_seconds=time.perf_counter() - begin))
    print(json.dumps(dict(event="adapt_finished", output=str(directory),
                          seconds=time.perf_counter() - begin)), flush=True)
    return results


if __name__ == "__main__":
    main()
