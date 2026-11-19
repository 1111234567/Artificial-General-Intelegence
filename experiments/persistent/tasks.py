"""Random permutation-graph queries for the persistent-memory experiment.

Each example contains an unordered set of ``key -> value`` records.  Values
are a permutation of the example's distinct keys, so every query remains in
the graph for arbitrarily many hops.  Self-loops and longer cycles are part
of the sampling distribution; they are deliberately not filtered out.

``target`` and ``path_targets`` are labels for evaluation/analysis.  A model
must receive only records, query, hop count, and (for a sequence baseline)
the NOOP ``gap``.  In particular, path_targets are not intermediate training
supervision.  There is no shared table across examples or generated batches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral

import torch
from torch import Tensor


_TENSOR_FIELDS = ("keys", "values", "query", "hops", "target", "path_targets")


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _random_order(
    batch_size: int,
    width: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Independent row permutations, using only the requested torch RNG."""
    # Float64 ranks make accidental ties negligible even for large batches.
    return torch.rand(
        (batch_size, width),
        dtype=torch.float64,
        device=device,
        generator=generator,
    ).argsort(dim=1)


@dataclass
class Batch:
    """A batch of graphs and labels, with all tensor fields of dtype long.

    Shapes: keys/values ``[B, N]``; query/hops/target ``[B]``; and
    path_targets ``[B, H]``, where column ``h - 1`` is the endpoint after
    exactly ``h`` hops, including columns beyond the selected query's hops.
    ``gap`` is a batch-wide count of NOOP steps following the records.
    """

    keys: Tensor
    values: Tensor
    query: Tensor
    hops: Tensor
    target: Tensor
    path_targets: Tensor
    gap: int
    vocab: int

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Batch:
        """Return this batch's tensors on device, preserving metadata."""
        return replace(
            self,
            **{
                name: getattr(self, name).to(device, non_blocking=non_blocking)
                for name in _TENSOR_FIELDS
            },
        )

    def clone(self) -> Batch:
        """Copy every tensor, so diagnostic mutations cannot affect the source."""
        return replace(
            self, **{name: getattr(self, name).clone() for name in _TENSOR_FIELDS}
        )

    def with_gap(self, gap: int) -> Batch:
        """Change the NOOP count without resampling graphs or labels."""
        return replace(self, gap=_integer("gap", gap, 0))

    def shuffle_records(self, generator: torch.Generator | None = None) -> Batch:
        """Reorder complete key/value pairs independently in each example.

        The supplied RNG may live on a different device from the batch; in
        that case only the sampled ordering is transferred to the batch.
        Query and labels retain their values and are not re-generated.
        """
        sample_device = (
            torch.device(generator.device) if generator is not None else self.keys.device
        )
        order = _random_order(*self.keys.shape, sample_device, generator).to(self.keys.device)
        return replace(
            self,
            keys=self.keys.gather(1, order),
            values=self.values.gather(1, order),
        )


@torch.no_grad()
def make_batch(
    batch_size: int,
    n_records: int,
    gap: int,
    max_hops: int,
    vocab: int = 64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    fixed_hops: int | None = None,
) -> Batch:
    """Sample independent random graphs and exact multi-hop query labels.

    Every example samples N distinct vocabulary items, a random permutation
    of those items as values, an independent record ordering, and a uniform
    query among its keys.  Hops are uniform in ``1..max_hops`` unless fixed.
    N and gap are shared by the batch.  N must be at least two so the task
    cannot silently collapse to the single-record identity problem.

    Every random draw uses ``generator``.  With an explicit generator,
    sampling happens on its device before transfer to ``device``; a CPU
    generator therefore also works for CUDA output and does not touch the
    global torch RNG.  With None, sampling uses the output device's default
    torch RNG.  Gap is metadata and consumes no randomness.
    """
    batch_size = _integer("batch_size", batch_size, 1)
    n_records = _integer("n_records", n_records, 2)
    gap = _integer("gap", gap, 0)
    max_hops = _integer("max_hops", max_hops, 1)
    vocab = _integer("vocab", vocab, 2)
    if n_records > vocab:
        raise ValueError(f"n_records ({n_records}) must not exceed vocab ({vocab})")
    if fixed_hops is not None:
        fixed_hops = _integer("fixed_hops", fixed_hops, 1)
        if fixed_hops > max_hops:
            raise ValueError("fixed_hops must not exceed max_hops")

    output_device = torch.device(device)
    sample_device = (
        torch.device(generator.device) if generator is not None else output_device
    )
    keys = _random_order(batch_size, vocab, sample_device, generator)[:, :n_records]
    value_order = _random_order(batch_size, n_records, sample_device, generator)
    values = keys.gather(1, value_order)
    record_order = _random_order(batch_size, n_records, sample_device, generator)
    keys = keys.gather(1, record_order)
    values = values.gather(1, record_order)

    query_index = torch.randint(
        n_records, (batch_size, 1), device=sample_device, generator=generator
    )
    query = keys.gather(1, query_index).squeeze(1)
    if fixed_hops is None:
        hops = torch.randint(
            1, max_hops + 1, (batch_size,), device=sample_device, generator=generator
        )
    else:
        hops = torch.full((batch_size,), fixed_hops, dtype=torch.long, device=sample_device)

    successors = torch.full(
        (batch_size, vocab), -1, dtype=torch.long, device=sample_device
    )
    successors.scatter_(1, keys, values)
    current = query
    path = []
    for _ in range(max_hops):
        current = successors.gather(1, current.unsqueeze(1)).squeeze(1)
        path.append(current)
    path_targets = torch.stack(path, dim=1)
    target = path_targets.gather(1, (hops - 1).unsqueeze(1)).squeeze(1)

    return Batch(keys, values, query, hops, target, path_targets, gap, vocab).to(output_device)


def exact_oracle(batch: Batch) -> Tensor:
    """Follow Python dictionaries and return endpoints on the query device.

    This deliberately independent, CPU reference is for semantic checks,
    not a learned model or a training operation.  It never reads target,
    path_targets, gap, or record order.  A malformed graph raises ValueError
    instead of silently inventing an endpoint.
    """
    if batch.keys.ndim != 2 or batch.values.shape != batch.keys.shape:
        raise ValueError("keys and values must both have shape [B, N]")
    batch_size = batch.keys.shape[0]
    if batch.query.shape != (batch_size,) or batch.hops.shape != (batch_size,):
        raise ValueError("query and hops must both have shape [B]")
    if any(
        tensor.dtype != torch.long
        for tensor in (batch.keys, batch.values, batch.query, batch.hops)
    ):
        raise ValueError("records, query, and hops must have dtype torch.long")

    rows = zip(
        batch.keys.detach().cpu().tolist(),
        batch.values.detach().cpu().tolist(),
        batch.query.detach().cpu().tolist(),
        batch.hops.detach().cpu().tolist(),
        strict=True,
    )
    endpoints = []
    for row, (keys, values, query, hops) in enumerate(rows):
        memory = dict(zip(keys, values, strict=True))
        if len(memory) != len(keys):
            raise ValueError(f"example {row} contains duplicate keys")
        if hops < 1:
            raise ValueError(f"example {row} has a non-positive hop count")
        current = query
        for _ in range(hops):
            if current not in memory:
                raise ValueError(f"example {row} has no record for key {current}")
            current = memory[current]
        endpoints.append(current)
    return torch.tensor(endpoints, dtype=torch.long, device=batch.query.device)
