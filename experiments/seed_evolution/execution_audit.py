"""Apply the frozen large-run audit to a sealed execution amendment."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
import math
import os
from pathlib import Path
import re

import torch

from experiments.learned_binding import audit as binding_audit, common
from experiments.seed_evolution import audit, core, large_audit


require = audit.require
same_tree = audit.same_tree
_BASE_CANDIDATE = audit.audit_candidate
_BASE_REPORT = audit.audit_training_report
_BASE_REPLAY = audit.replay_candidate
_BASE_FINAL = audit.audit_final
_BASE_MODEL = core.model_from_genome
_BASE_CHECKPOINT_MODEL = audit.checkpoint_model
_BASE_UPDATE = audit.independent_ce_update
_AMENDMENT_NAME = "execution-amendment.json"
_CANDIDATE_ID = re.compile(r"(shared|evolution|random)_g(\d{2})_i(\d{2})")


@dataclass(frozen=True)
class Execution:
    start_generation: int
    hybrid_indices: tuple[int, ...]
    metadata: dict
    sha256: str

    def backend(self, cid):
        match = _CANDIDATE_ID.fullmatch(cid)
        require(match is not None, "invalid execution candidate identity")
        arm, generation, index = match.groups()
        generation, index = int(generation), int(index)
        require(generation < 32 and index < 64, "execution candidate identity is out of range")
        require((arm == "shared") == (generation == 0), "invalid shared candidate identity")
        if generation < self.start_generation:
            return "legacy_cpu"
        return "hybrid" if index in self.hybrid_indices else "cpu"

    def expected_metadata(self, backend):
        if backend == "legacy_cpu":
            return None
        return dict(self.metadata[backend], amendment_sha=self.sha256)


def _load_execution(root):
    from experiments.seed_evolution import execution_runtime

    amendment = execution_runtime.load_amendment(root)
    start = amendment["start_generation"]
    indices = amendment["hybrid_indices"]
    require(type(start) is int and 1 <= start <= 32, "invalid execution boundary")
    same_tree(indices, list(range(8)), "sealed hybrid indices")
    metadata = amendment["backend_metadata"]
    require(set(metadata) == {"cpu", "hybrid"}, "missing execution backend metadata")
    backend_sha = common.sha(Path(__file__).with_name("hybrid_backend.py"))
    implementation = dict(core=common.sha(core.__file__), backend=backend_sha)
    cache = amendment["cache_encoder"]
    require(type(cache) is bool, "invalid sealed encoder cache mode")
    if cache:
        implementation["encoder_cache"] = common.sha(Path(__file__).with_name("encoder_cache.py"))
    for backend, topology in (("cpu", "cpu"), ("hybrid", "cuda:0")):
        template = metadata[backend]
        same_tree(template["backend"], backend, "execution backend template identity")
        same_tree(template["cache"], cache, "execution backend template encoder cache")
        same_tree(template["implementation_sha256"], implementation, "execution backend implementation hashes")
        same_tree(template["device_topology"], dict(policy="cpu", optimizer="cpu", executor=topology),
                  "execution backend template device topology")
        same_tree((template["dtype"], template["amp"], template["tf32"]), ("float32", False, False),
                  "execution backend template precision")
        same_tree(template["backend_source_sha"], backend_sha, "sealed backend source hash")
        same_tree(template["lazy_streams"], amendment["lazy_training"], "sealed lazy stream mode")
        same_tree(template["lazy_source_sha"], common.sha(Path(__file__).with_name("lazy_training_streams.py")),
                  "sealed lazy stream source hash")
        require("amendment_sha" not in template, "execution template cannot bind its own manifest")
    execution = Execution(start, tuple(indices), metadata, common.sha(Path(root) / _AMENDMENT_NAME))
    _verify_prefix(root, amendment, execution)
    return execution


def _verify_prefix(root, amendment, execution):
    root = Path(root)
    plans = {Path(name.replace("\\", "/")): digest for name, digest in amendment["issued_plans"].items()}
    expected_plans = {Path(f"plans/generation_{generation:02d}.json")
                      for generation in range(execution.start_generation)}
    require(set(plans) == expected_plans, "issued plan inventory differs from execution boundary")
    for path, digest in plans.items():
        audit.checked_file(root, path, digest)
    for name, digest in amendment["existing_prefix_inventory"].items():
        path = Path(name.replace("\\", "/"))
        require(len(path.parts) == 4 and path.parts[0] == "search" and path.parts[1] in {"rep_0", "rep_1", "rep_2"}
                and path.name in {"candidate.json", "training.json", ".training.json.pending"},
                "unexpected execution prefix artifact: " + name)
        require(execution.backend(path.parent.name) == "legacy_cpu", "non-prefix artifact at cutover: " + name)
        if path.name != ".training.json.pending":
            audit.checked_file(root, path, digest)
            continue
        committed = path.with_name("training.json")
        if not (root / committed).is_file():
            try:
                audit.checked_file(root, path, digest)
                continue
            except (FileNotFoundError, binding_audit.AuditError):
                if (root / path).is_file() or not (root / committed).is_file():
                    raise
        audit.checked_file(root, committed, digest)


def _execution_for_job(root, execution):
    if execution is None:
        return _load_execution(root)
    require(common.sha(Path(root) / _AMENDMENT_NAME) == execution.sha256,
            "execution amendment changed before independent replay")
    return execution


@contextmanager
def _override(module, **replacements):
    with ExitStack() as stack:
        for name, replacement in replacements.items():
            stack.callback(setattr, module, name, getattr(module, name))
            setattr(module, name, replacement)
        yield


def audit_training_report(training, seed, updates, genome, executor, *, expected_execution):
    """Validate device provenance against the sealed allocation and compare CE settings."""
    if expected_execution is None:
        _BASE_REPORT(training, seed, updates, genome, executor)
        return
    settings = audit.HISTORICAL_CE
    encoder = {key.removeprefix("encoder."): value for key, value in genome.items()
               if key.startswith("encoder.")}
    device = "cpu+cuda_executor" if expected_execution["backend"] == "hybrid" else "cpu"
    expected = dict(source_seed=seed, steps=updates, device=device,
                    genome_hash=audit.genome_hash(genome),
                    executor_hash=binding_audit.parameter_hash(executor),
                    encoder_hash=binding_audit.parameter_hash(encoder),
                    stream_indices=[audit.historical_stream_at(step) for step in range(updates)],
                    initial_state=dict(h=0, z=0, memory=0), reset_every_stream=True,
                    optimizer="Adam", fresh_optimizer=True, learning_rate=settings["ce_lr"],
                    wait_weight=settings["wait_weight"], grad_clip_norm=5.,
                    trainable_parameters=2258, encoder_unchanged=True, executor_unchanged=True,
                    tf32=False, execution=expected_execution)
    same_tree(training["metadata"], expected, "official CE settings, schedule, and execution")
    log = training["log"]
    logged = [row["update"] for row in log]
    require(bool(logged) and logged == sorted(set(logged)) and logged[0] == 1
            and logged[-1] == updates, "incomplete or repeated training log steps")
    for row in log:
        require(row["stream"] == audit.historical_stream_at(row["update"] - 1),
                "logged CE stream differs")
        require(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]),
                "nonfinite CE metrics")


def _audit_candidate(root, record, config, source_executor, genomes, *, execution, counts):
    backend = execution.backend(record["id"])
    report = partial(audit_training_report, expected_execution=execution.expected_metadata(backend))
    with _override(audit, audit_training_report=report):
        schedule = _BASE_CANDIDATE(root, record, config, source_executor, genomes)
    counts[record["rep"], record["arm"], backend] += 1
    return schedule


def _attach_cuda_executor(model):
    device = torch.device("cuda:0")
    model.executor.to(device)

    def inputs(_executor, args):
        observations, hidden = args
        return observations.to(device), hidden.to(device)

    def outputs(_executor, _args, result):
        logits, next_hidden = result
        return logits.cpu(), next_hidden.cpu()

    model.executor.register_forward_pre_hook(inputs)
    model.executor.register_forward_hook(outputs)
    return model


@contextmanager
def _cuda_environment(execution):
    cuda = execution.metadata["hybrid"]["cuda"]
    workspace = cuda["cublas_workspace_config"]
    require(workspace in {":4096:8", ":16:8"}, "invalid CUDA replay workspace")
    require(os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "1", "CUDA replay requires strict FP32")
    variable = "CUBLAS_WORKSPACE_CONFIG"
    previous = os.environ.get(variable)
    require(previous is None or previous == workspace, "CUDA replay workspace differs from execution")
    with ExitStack() as stack:
        if previous is None:
            os.environ[variable] = workspace
            stack.callback(os.environ.pop, variable, None)
        require(torch.cuda.is_available() and torch.backends.cudnn.is_available(),
                "hybrid audit requires CUDA and cuDNN")
        properties = torch.cuda.get_device_properties(0)
        observed = dict(device_name=properties.name, compute_capability=[properties.major, properties.minor],
                        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                        cudnn_enabled=True, cudnn_benchmark=False, cudnn_deterministic=True,
                        deterministic_algorithms=True, deterministic_warn_only=False,
                        float32_matmul_precision="highest", cublas_workspace_config=workspace)
        same_tree(observed, cuda, "independent CUDA replay environment")
        same_tree(str(torch.__version__), execution.metadata["hybrid"]["torch_version"],
                  "independent CUDA replay torch version")
        yield


@contextmanager
def _hybrid_precision():
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        with (torch.backends.cudnn.flags(enabled=True, benchmark=False, benchmark_limit=None,
                                         deterministic=True, allow_tf32=False),
              torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False)):
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _hybrid_update(model, optimizer, trainable, stream):
    with _hybrid_precision():
        return _BASE_UPDATE(model, optimizer, trainable, stream)


def _hybrid_model(source_seed, genome):
    return _attach_cuda_executor(_BASE_MODEL(source_seed, genome))


def _hybrid_checkpoint_model(checkpoint, device="cpu"):
    require(device == "cpu", "hybrid replay requires CPU memory heads and Adam")
    return _attach_cuda_executor(_BASE_CHECKPOINT_MODEL(checkpoint, device))


def replay_candidate(job, *, execution=None):
    root, _, cid, _, _, retrain = job
    execution = _execution_for_job(root, execution)
    backend = execution.backend(cid)
    if backend == "hybrid" and retrain:
        with (_cuda_environment(execution), _override(core, model_from_genome=_hybrid_model),
              _override(audit, independent_ce_update=_hybrid_update)):
            result = _BASE_REPLAY(job)
    else:
        result = _BASE_REPLAY(job)
    return dict(result, execution_backend=backend)


def audit_final(job, *, execution=None):
    root, row, *_ = job
    execution = _execution_for_job(root, execution)
    backend = execution.backend(row["winner"]["id"])
    hooks = dict(audit_training_report=partial(
        audit_training_report, expected_execution=execution.expected_metadata(backend)))
    with ExitStack() as stack:
        if backend == "hybrid":
            stack.enter_context(_cuda_environment(execution))
            hooks.update(checkpoint_model=_hybrid_checkpoint_model, independent_ce_update=_hybrid_update)
        stack.enter_context(_override(audit, **hooks))
        result = _BASE_FINAL(job)
    return dict(result, execution_backend=backend)


class _ReplayPool:
    """Keep CUDA replays in one process and preserve frozen report ordering."""

    def __init__(self, max_workers, *, execution):
        self.max_workers = max_workers
        self.execution = execution

    def __enter__(self):
        with ExitStack() as stack:
            cuda_lane = self.execution.start_generation < 32 and self.max_workers > 1
            self.cpu = stack.enter_context(ProcessPoolExecutor(max_workers=self.max_workers - int(cuda_lane)))
            self.cuda = stack.enter_context(ProcessPoolExecutor(max_workers=1)) if cuda_lane else self.cpu
            self.stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def map(self, function, jobs):
        require(function in (replay_candidate, audit_final), "unknown execution replay job")
        pending = []
        for job in jobs:
            cid = job[2] if function is replay_candidate else job[1]["winner"]["id"]
            retrain = job[-1] if function is replay_candidate else True
            pool = self.cuda if retrain and self.execution.backend(cid) == "hybrid" else self.cpu
            pending.append(pool.submit(function, job, execution=self.execution))
        return (future.result() for future in pending)


def audit_all(root=large_audit.OUT, workers=9, *, metadata_only=False):
    """Check candidate allocation and replay on the actual training backend."""
    root = Path(root).resolve()
    execution = _load_execution(root)
    counts = Counter()
    candidate = partial(_audit_candidate, execution=execution, counts=counts)
    pool = partial(_ReplayPool, execution=execution)
    with _override(audit, audit_candidate=candidate, replay_candidate=replay_candidate,
                   audit_final=audit_final), _override(large_audit, ProcessPoolExecutor=pool):
        report = large_audit.audit_all(root, workers, metadata_only=metadata_only)

    prefix_count = sum(count for (_, _, backend), count in counts.items() if backend == "legacy_cpu")
    suffix_counts = {
        str(rep): {
            arm: {backend: counts[rep, arm, backend] for backend in ("cpu", "hybrid")}
            for arm in audit.ARMS
        }
        for rep in (0, 1, 2)
    }
    require(sum(counts.values()) == report["candidates_provenance_checked"], "execution ledger coverage differs")
    if not metadata_only:
        suffix_generations = report["target_generations"] - execution.start_generation
        expected_counts = dict(cpu=suffix_generations * (64 - len(execution.hybrid_indices)),
                               hybrid=suffix_generations * len(execution.hybrid_indices))
        for arms in suffix_counts.values():
            for arm in audit.ARMS:
                same_tree(arms[arm], expected_counts, "paired execution backend counts")
        if expected_counts["hybrid"]:
            require(sum(replay["execution_backend"] == "hybrid" and replay["independent_ce_updates"] > 0
                        for replay in report["development_replays"]) == 6,
                    "incomplete hybrid development training replay coverage")
    return dict(report, execution_amendment_sha256=execution.sha256,
                execution_audit_sha256=common.sha(__file__),
                execution_start_generation=execution.start_generation,
                legacy_cpu_candidates=prefix_count, suffix_backend_counts=suffix_counts)
