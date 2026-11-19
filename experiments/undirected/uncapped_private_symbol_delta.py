"""Sample bounded parent-life weight changes for the private-symbol trial.

The sampling evaluator must use ``delta_rollout`` as its rollout, and must
have its own CUDA-graph cache. A training evaluator already captured with a
different rollout cannot be reused because rollout identity is not part of
the sealed backend's graph-cache key. The supplied schedule must create
independent private mappings per actor; production sampling uses 16 actors.

Only birth arcs which survive the whole life contribute. New arcs, retired
arcs, and retired-then-regrown arcs contribute zero. The denominator is every
sampled life, rather than only surviving lives, so a fragile change cannot
be amplified by conditioning on survival. Hidden state and topology are
never inherited by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import torch


DELTA_FIELD = "learned_delta"
MAX_BATCH_SIZE = 16


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _mean_surviving_delta(policy, born_weight, born_mask):
    """Reduce actor lives in FP32, preserving the genome's flat arc order.

    ``born_weight`` is the actual reset weight for actor zero, already cast
    through the policy storage dtype, with shape [population, 2, pairs].
    Reset gives every actor the same birth state. ``last_change == 0`` also
    excludes a birth arc that was retired and later admitted again.
    """
    surviving = (
        born_mask.unsqueeze(1)
        & policy.mask
        & policy.last_change.eq(0)
    )
    change = policy.w.to(dtype=torch.float32, copy=True).sub_(born_weight.unsqueeze(1))
    change.masked_fill_(~surviving, 0.)
    return change.mean(dim=1, dtype=torch.float32).flatten(1)


@torch.no_grad()
def delta_rollout(policy, tape, table):
    """Roll out one life and return only its mean surviving weight delta.

    Snapshot at the beginning, immediately after the evaluator resets the
    policy. Subtracting original CPU FP32 genes would mistake birth's FP16
    rounding for learning. Clone the actual reset weights instead. These
    snapshot and reduction operations are CUDA-graph capture compatible.
    """
    from .uncapped_private_symbol_world import task_rollout

    born_weight = policy.w[:, 0].to(dtype=torch.float32, copy=True)
    born_mask = policy.mask[:, 0].clone()
    task_rollout(policy, tape, table)
    return {DELTA_FIELD: _mean_surviving_delta(policy, born_weight, born_mask)}


@torch.no_grad()
def sample_deltas(evaluator, genomes: Mapping, *, seed, actors=16, visits=8,
                  batch_size=16, device="cuda"):
    """Return ``parent_id -> owned CPU FP32 flat-arc delta`` in small batches.

    ``evaluator`` follows ``RewardEvaluation.__call__`` and must be a
    dedicated evaluator configured with ``delta_rollout``. Its schedule is
    supplied by the caller. Use a seed independent of child evaluation and
    store that seed, actor count and parent IDs beside the returned deltas
    for replay. Every batch uses this same world seed, so parent comparisons
    share worlds and results do not depend on batch placement.

    At most 16 parents are evaluated simultaneously. The caller decides the
    capped parent subset and how frequently to sample. No random selection,
    mutation, inheritance scaling or checkpoint writes happen here.
    """
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(genomes, Mapping):
        raise TypeError("genomes must map parent IDs to birth genomes")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    actors = _positive_int("actors", actors)
    visits = _positive_int("visits", visits)
    batch_size = _positive_int("batch_size", batch_size)
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be at most {MAX_BATCH_SIZE}")
    device = torch.device(device)
    parent_ids = list(genomes)
    learned = {}
    for offset in range(0, len(parent_ids), batch_size):
        batch_ids = parent_ids[offset:offset + batch_size]
        batch = [genomes[parent_id] for parent_id in batch_ids]
        fields = evaluator(batch, seed=seed, actors=actors, device=device,
                           visits=visits, mode="full", capture=device.type == "cuda")
        if not isinstance(fields, Mapping) or DELTA_FIELD not in fields:
            raise ValueError("sampling evaluator must return a learned_delta tensor")
        delta = fields[DELTA_FIELD]
        expected = (len(batch), batch[0]["birth_w"].numel())
        if (not isinstance(delta, torch.Tensor) or delta.dtype != torch.float32
                or tuple(delta.shape) != expected):
            raise ValueError(f"expected FP32 learned_delta of shape {expected}")
        host = delta.detach().cpu()
        if not host.isfinite().all().item():
            raise ValueError("parent-life learned_delta contains nonfinite values")
        # Clone each row so the returned values do not retain batch outputs or
        # views into graph replay/packing buffers which later calls may reuse.
        for parent_id, row in zip(batch_ids, host.unbind(0)):
            learned[parent_id] = row.clone(memory_format=torch.contiguous_format)
        del fields, delta, host
    return learned
