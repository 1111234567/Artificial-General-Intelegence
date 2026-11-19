"""Run the sealed trainer with optional CPU trace reuse and CUDA execution."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from threading import RLock, get_ident
from types import MethodType
from unittest.mock import patch

import torch

from experiments.learned_binding import common
from experiments.seed_evolution import core, encoder_cache


_TRAIN_LOCK = RLock()
_CUDA_DEVICE = torch.device("cuda:0")
_CUBLAS_CONFIG = "CUBLAS_WORKSPACE_CONFIG"
_DETERMINISTIC_WORKSPACES = (":4096:8", ":16:8")


def _cuda_executor_forward(executor, obs, hidden):
    # The trainer calls model.to("cpu") before training. Move the frozen
    # executor at its first use and keep both device copies differentiable.
    if executor.head.weight.device != _CUDA_DEVICE:
        executor.to(_CUDA_DEVICE)
    logits, next_hidden = type(executor).forward(
        executor, obs.to(_CUDA_DEVICE), hidden.to(_CUDA_DEVICE))
    return logits.to(obs.device), next_hidden.to(hidden.device)


@contextmanager
def _fp32():
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast("cpu", enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@contextmanager
def _deterministic_cuda():
    workspace = os.environ.get(_CUBLAS_CONFIG)
    if workspace is not None and workspace not in _DETERMINISTIC_WORKSPACES:
        raise ValueError(f"{_CUBLAS_CONFIG} must be one of {_DETERMINISTIC_WORKSPACES}")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise ValueError("Strict FP32 requires TORCH_ALLOW_TF32_CUBLAS_OVERRIDE to be disabled")
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with ExitStack() as stack:
        if workspace is None:
            workspace = _DETERMINISTIC_WORKSPACES[0]
            os.environ[_CUBLAS_CONFIG] = workspace
            stack.callback(os.environ.pop, _CUBLAS_CONFIG, None)
        if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
            raise RuntimeError("Hybrid training requires a CUDA device and cuDNN")
        torch.use_deterministic_algorithms(True, warn_only=False)
        stack.callback(torch.use_deterministic_algorithms, previous_deterministic,
                       warn_only=previous_warn_only)
        stack.enter_context(torch.backends.cudnn.flags(
            enabled=True, benchmark=False, benchmark_limit=None, deterministic=True, allow_tf32=False))
        stack.enter_context(torch.autocast("cuda", enabled=False))
        properties = torch.cuda.get_device_properties(_CUDA_DEVICE)
        yield dict(device_name=properties.name, compute_capability=[properties.major, properties.minor],
                   cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                   cudnn_enabled=True, cudnn_benchmark=False, cudnn_deterministic=True,
                   deterministic_algorithms=True, deterministic_warn_only=False,
                   float32_matmul_precision=torch.get_float32_matmul_precision(),
                   cublas_workspace_config=workspace)


def train(source_seed, genome, steps, checkpoints, backend="cpu", cache=True):
    """Run one candidate in a serial worker and return CPU checkpoints.

    Run candidates concurrently in separate worker processes. The CPU policy
    retains the original trainer's state and optimizer, including for hybrid CE.
    """
    if backend not in ("cpu", "hybrid"):
        raise ValueError("backend must be 'cpu' or 'hybrid'")
    if not isinstance(cache, bool):
        raise TypeError("cache must be a boolean")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("The sealed trainer requires the float32 default dtype")

    with _TRAIN_LOCK, ExitStack() as stack:
        stack.enter_context(_fp32())
        cuda = stack.enter_context(_deterministic_cuda()) if backend == "hybrid" else None
        cache_handle = None
        if cache or cuda is not None:
            model_from_genome = core.model_from_genome
            owner = get_ident()

            def make_model(seed, initial):
                nonlocal cache_handle
                model = model_from_genome(seed, initial)
                if get_ident() == owner:
                    if cache:
                        cache_handle = stack.enter_context(encoder_cache.attach_frozen_encoder_cache(model))
                    if cuda is not None:
                        if any(p.requires_grad for p in model.executor.parameters()):
                            raise ValueError("Hybrid training requires a frozen executor")
                        model.executor.forward = MethodType(_cuda_executor_forward, model.executor)
                return model

            stack.enter_context(patch.object(core, "model_from_genome", make_model))
        result = core.inner_train(source_seed, genome, steps=steps, checkpoints=checkpoints, device="cpu")
        result["cache_metrics"] = None if cache_handle is None else cache_handle.stats
    implementation = dict(core=common.sha(core.__file__), backend=common.sha(__file__))
    if cache:
        implementation["encoder_cache"] = common.sha(encoder_cache.__file__)
    result["metadata"]["execution"] = dict(
        backend=backend, cache=cache, device_topology=dict(
            policy="cpu", optimizer="cpu", executor=str(_CUDA_DEVICE) if cuda is not None else "cpu"),
        dtype="float32", amp=False, tf32=False, torch_version=str(torch.__version__),
        implementation_sha256=implementation, cuda=cuda)
    if cuda is not None:
        result["metadata"]["device"] = "cpu+cuda_executor"
    return result
