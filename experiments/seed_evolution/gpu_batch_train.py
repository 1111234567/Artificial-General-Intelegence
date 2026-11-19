"""Batch independent seed-evolution candidates under one frozen executor."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.learned_binding import common, pretrain
from experiments.learned_binding.model import (
    ENCODER_WIDTH, FEATURES_PER_JOB, KEY_DIM, N_JOBS, N_VALUES, soft_translate,
)
from experiments.seed_evolution import core, hybrid_backend


_HEADS = ("query", "key", "value", "gate")
_ENCODED_CACHE_BYTES = 256 << 20


def _models(source_seed, genomes, device, executor_dtype):
    models = [core.model_from_genome(source_seed, genome).train() for genome in genomes]
    executor = models[0].executor
    if executor_dtype != torch.float32:
        # Every checkpoint retains the original FP32 executor parameters.
        executor = deepcopy(executor)
    executor.to(device=device, dtype=executor_dtype)
    for model in models:
        model.encoder.to(device)
        for name in _HEADS:
            getattr(model, name).to(device)
        model.read_scale_raw = torch.nn.Parameter(model.read_scale_raw.detach().to(device))
    return models, executor


@torch.no_grad()
def _encode(models, observations):
    ticks, batch = observations.shape[:2]
    encoder_input = observations.reshape(ticks, batch * N_JOBS, FEATURES_PER_JOB)
    z = observations.new_zeros(1, batch * N_JOBS, ENCODER_WIDTH)
    return torch.stack([model.encoder(encoder_input, z)[0] for model in models]).reshape(
        len(models), ticks, batch, N_JOBS, ENCODER_WIDTH)


def _project(models, name, inputs):
    layers = [getattr(model, name) for model in models]
    weights = torch.stack([layer.weight for layer in layers]).transpose(1, 2)
    biases = torch.stack([layer.bias for layer in layers])[:, None, :]
    return torch.baddbmm(biases, inputs, weights)


def _read_sequence(models, encoded, observations):
    candidates, ticks, batch = encoded.shape[:3]
    jobs = observations.reshape(1, ticks, batch, N_JOBS, FEATURES_PER_JOB).expand(
        candidates, -1, -1, -1, -1)
    inputs = torch.cat((encoded, jobs), dim=-1).reshape(candidates, ticks * batch * N_JOBS, -1)

    def tick_order(tensor):
        return tensor.reshape(candidates, ticks, batch, N_JOBS, -1).transpose(0, 1).reshape(
            ticks, candidates * batch, N_JOBS, -1)

    queries = tick_order(F.normalize(_project(models, "query", inputs), dim=-1))
    keys = F.normalize(_project(models, "key", inputs), dim=-1)
    weighted = tick_order(_project(models, "gate", inputs).sigmoid() * keys)
    keys = tick_order(keys)
    values = tick_order(_project(models, "value", inputs).softmax(-1))
    scales = torch.stack([model.read_scale for model in models]).repeat_interleave(batch).reshape(-1, 1, 1)
    matrix = observations.new_zeros(candidates * batch, KEY_DIM, N_VALUES)
    keys_0, keys_1 = keys.unbind(2)
    weighted_0, weighted_1 = weighted.unbind(2)
    values_0, values_1 = values.unbind(2)
    frames = zip(queries.unbind(0), keys_0.unbind(0), keys_1.unbind(0), weighted_0.unbind(0),
                 weighted_1.unbind(0), values_0.unbind(0), values_1.unbind(0))
    reads = []
    for query, key_0, key_1, weighted_key_0, weighted_key_1, value_0, value_1 in frames:
        reads.append((scales * torch.bmm(query, matrix)).softmax(-1))
        prediction_0 = torch.bmm(key_0[:, None], matrix).squeeze(1)
        matrix = torch.baddbmm(matrix, weighted_key_0[:, :, None], (value_0 - prediction_0)[:, None])
        prediction_1 = torch.bmm(key_1[:, None], matrix).squeeze(1)
        matrix = torch.baddbmm(matrix, weighted_key_1[:, :, None], (value_1 - prediction_1)[:, None])
    return torch.stack(reads)


def _logits(models, executor, encoded, observations, mappings, executor_chunk):
    candidates, ticks, batch = encoded.shape[:3]
    reads = _read_sequence(models, encoded, observations)
    repeated = observations.unsqueeze(1).expand(-1, candidates, -1, -1).reshape(ticks, candidates * batch, -1)
    mapped = soft_translate(repeated, reads, mappings)

    def execute(chunk):
        executor_input = chunk.to(dtype=executor.head.weight.dtype).contiguous()
        initial_h = executor_input.new_zeros(executor_input.shape[1], executor.width)
        logits, _ = executor(executor_input, initial_h)
        return logits.float()

    if executor_chunk is None:
        logits = execute(mapped)
    else:
        # The executor hidden state has no cross-actor coupling. Retain the
        # entire time axis and recompute only its independent candidate lanes.
        pieces = mapped.split(executor_chunk * batch, dim=1)
        logits = torch.cat([checkpoint(execute, piece, use_reentrant=False, preserve_rng_state=False)
                            for piece in pieces], dim=1)
    return logits.reshape(ticks, candidates, batch, N_JOBS, -1).unbind(1)


def _snapshots(models, optimizers, source_seed, update):
    return [core._training_snapshot(model, optimizer, source_seed, update)
            for model, optimizer in zip(models, optimizers)]


def train_batch(source_seed, genomes, steps, *, device="cuda", dtype=None, executor_chunk=None,
                snapshot_steps=()):
    """Train same-source genomes with independent memory, loss, clipping and Adam.

    Returns CPU FP32 state dicts and final checkpoint trees in candidate order.
    ``snapshot_steps`` adds checkpoints at the requested completed updates,
    with step zero representing initialization. Each entry in ``snapshots``
    uses the same checkpoint format and candidate order as ``checkpoints``.
    ``dtype`` applies to the shared executor's computation and defaults to
    BF16 on CUDA and FP32 on CPU. ``executor_chunk``
    bounds recomputed executor groups by candidate count. Batched kernels
    change arithmetic order, so this is a benchmark backend.
    """
    source_seed = core._source_seed(source_seed)
    steps = core._integer("steps", steps, minimum=1)
    if steps > common.SETTINGS["ce_updates"]:
        raise ValueError("steps exceed the original CE budget")
    snapshot_steps = {core._integer("snapshot step", step) for step in snapshot_steps}
    if any(step > steps for step in snapshot_steps):
        raise ValueError("snapshot steps must lie within the requested training budget")
    genomes = tuple(genomes)
    if not genomes:
        raise ValueError("at least one genome is required")
    if executor_chunk is not None:
        executor_chunk = min(core._integer("executor_chunk", executor_chunk, 1), len(genomes))
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and device.index not in (None, 0)):
        raise ValueError("the benchmark accepts cpu or cuda:0")
    cuda = device.type == "cuda"
    if dtype is None:
        dtype = torch.bfloat16 if cuda else torch.float32
    if (dtype not in (torch.float32, torch.bfloat16) or (not cuda and dtype != torch.float32)
            or torch.get_default_dtype() != torch.float32):
        raise ValueError("use FP32, or BF16 executor compute on CUDA with FP32 model parameters")
    torch.set_num_threads(1)

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    losses = [[] for _ in genomes]
    grad_norms = [[] for _ in genomes]
    snapshots = {}
    capture_final = steps in snapshot_steps
    snapshot_steps.discard(steps)
    schedule = [core._stream_index(step) for step in range(steps)]
    start = perf_counter()
    cuda_flags = hybrid_backend._deterministic_cuda() if cuda else nullcontext()
    with hybrid_backend._fp32(), cuda_flags as cuda_environment:
        models, executor = _models(source_seed, genomes, device, dtype)
        parameters = [[p for p in model.parameters() if p.requires_grad] for model in models]
        optimizers = [torch.optim.Adam(group, lr=common.SETTINGS["ce_lr"]) for group in parameters]
        if 0 in snapshot_steps:
            snapshots[0] = _snapshots(models, optimizers, source_seed, 0)
        source_streams = core._training_streams(source_seed)
        streams = {index: tuple(tensor.to(device) for tensor in source_streams[index])
                   for index in sorted(set(schedule))}
        mappings = models[0].original_mappings.to(device)
        encoded_cache = {}
        cached_bytes = 0
        synchronize()
        prepared = perf_counter()
        for update, index in enumerate(schedule, 1):
            observations, actions = streams[index]
            encoded = encoded_cache.get(index)
            if encoded is None:
                encoded = _encode(models, observations)
                size = encoded.numel() * encoded.element_size()
                if size <= _ENCODED_CACHE_BYTES - cached_bytes:
                    encoded_cache[index] = encoded
                    cached_bytes += size

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            logits = _logits(models, executor, encoded, observations, mappings, executor_chunk)
            vector_loss = torch.stack([pretrain.loss_for(candidate, actions) for candidate in logits])
            reported_loss = vector_loss.detach().cpu()
            if not bool(reported_loss.isfinite().all()):
                raise ValueError(f"nonfinite candidate CE at update {update}")
            # Every loss uses the same denominator it would have in a solo
            # run. Summing independent losses preserves each input gradient.
            vector_loss.sum().backward()
            norms = [torch.nn.utils.clip_grad_norm_(group, 5., error_if_nonfinite=True) for group in parameters]
            for optimizer in optimizers:
                optimizer.step()
            for candidate_log, value in zip(losses, reported_loss.tolist()):
                candidate_log.append(value)
            for candidate_log, value in zip(grad_norms, torch.stack(norms).cpu().tolist()):
                candidate_log.append(value)
            if update in snapshot_steps:
                snapshots[update] = _snapshots(models, optimizers, source_seed, update)
        synchronize()
        trained = perf_counter()
        checkpoints = _snapshots(models, optimizers, source_seed, steps)
        if capture_final:
            snapshots[steps] = checkpoints
        synchronize()
        finished = perf_counter()
    return dict(states=[checkpoint["model"] for checkpoint in checkpoints], losses=losses,
                grad_norms=grad_norms, checkpoints=checkpoints, snapshots=snapshots,
                timing=dict(prepare_seconds=prepared - start, train_seconds=trained - prepared,
                            export_seconds=finished - trained, elapsed_seconds=finished - start),
                metadata=dict(source_seed=source_seed, steps=steps, candidates=len(genomes),
                              device=str(device), dtype="float32", executor_dtype=str(dtype).removeprefix("torch."),
                              executor_chunk=executor_chunk, stream_indices=schedule,
                              encoded_cache_streams=len(encoded_cache), encoded_cache_bytes=cached_bytes,
                              cuda=cuda_environment, tf32=False))
