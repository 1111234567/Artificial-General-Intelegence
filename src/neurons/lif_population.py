"""Population evaluation for the existing 64-cell adaptive spiking graph.

The graph accepts the seven input ports of the attractor experiments. The
second compartment receives those same events through a separate projection.
This is a fixed input and readout format for the two pulse-game models.
"""

import operator
import sys

import numpy as np
import torch

from experiments.pulse_game import attractor_search as _base
from experiments.pulse_game import dendrite_search as _dendrite

Schedule = _base.Schedule
BASE_GENE_RANGES = _base.RANGES
DENDRITE_GENE_RANGES = _dendrite.RANGES

__all__ = ["Schedule", "BASE_GENE_RANGES", "DENDRITE_GENE_RANGES", "evaluate_population"]

_DENDRITE_MODES = ("basal", "add", "gain", "both")
_BYTE_LIMITS = {
    "cue": (0, 3), "ctx": (0, 1), "issue": (0, 255), "query": (0, 255),
    "length": (1, 255), "raw_at": (0, 255), "distractor": (0, 7),
    "at": (0, 255), "goal": (0, 3),
}


def _schedule_for_kernel(schedule):
    goal = np.asarray(schedule.goal)
    if goal.ndim != 2 or not goal.size:
        raise ValueError("schedule.goal must have nonempty (lives, trials) shape")
    shape = goal.shape
    seeds = tuple(schedule.seeds)
    if len(seeds) != shape[0]:
        raise ValueError("schedule.seeds must contain one ID per life")

    fields = {}
    for name, (lower, upper) in _BYTE_LIMITS.items():
        array = np.array(getattr(schedule, name), order="C", copy=True)
        if array.shape != shape or array.dtype.kind not in "iu":
            raise ValueError(f"schedule.{name} must be an integer array with shape {shape}")
        if (array < lower).any() or (array > upper).any():
            raise ValueError(f"schedule.{name} must be in [{lower}, {upper}]")
        fields[name] = array.astype(np.uint8, copy=False)

    length = fields["length"]
    for name in ("issue", "query", "raw_at", "at"):
        ticks = fields[name]
        outside = ticks >= length
        if name in ("raw_at", "at"):
            outside &= ticks != 255
        if outside.any():
            raise ValueError(f"schedule.{name} has a tick outside its trial")

    tie = np.array(schedule.tie, order="C", copy=True)
    if tie.shape != (*shape, 4) or tie.dtype.kind not in "fiu":
        raise ValueError(f"schedule.tie must be a real array with shape {(*shape, 4)}")
    if not np.isfinite(tie).all() or (tie < 0).any() or (tie > 1).any():
        raise ValueError("schedule.tie must contain finite values in [0, 1]")
    return Schedule(**fields, tie=tie.astype(np.float32, copy=False), seeds=seeds)


def _cuda_device(device, mode, pop, schedule):
    if mode != "base" and device.index not in (None, 0):
        raise ValueError("the second-compartment CUDA evaluator requires cuda:0")
    lives, trials = schedule.goal.shape
    # CUDA grid.y contains population size. The kernels use signed int indices
    # for feature and pulse offsets, and signed int counters for pulses.
    int_limit = np.iinfo(np.int32).max
    if (pop > 65535 or pop * lives * max(trials * _base.K, _base.N) > int_limit
            or (schedule.length.sum(axis=1, dtype=np.int64) > int_limit).any()):
        raise ValueError("population or schedule exceeds the CUDA evaluator's index range")

    if sys.platform != "win32" or not torch.cuda.is_available():
        raise RuntimeError("CUDA population evaluation requires the Windows PyTorch NVRTC backend")
    device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
    if device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device} is unavailable")
    if mode != "base" and device.index != 0:
        raise ValueError("the second-compartment CUDA evaluator requires cuda:0")
    return device


def _cpu_batch(vectors, net_seed, schedule, mode, cut, feedback, features):
    shape = (len(vectors), *schedule.goal.shape)
    answers = np.empty(shape, np.uint8)
    spike_fraction = np.empty(len(vectors), np.float64)
    z = np.empty((*shape, _base.K), np.float32) if features else None
    if mode != "base":
        from experiments.pulse_game.dendrite_audit import reference

    for i, vector in enumerate(vectors):
        if mode == "base":
            activity, spike_fraction[i] = _base.cpu_features(vector, net_seed, schedule, cut=cut)
        else:
            activity, spike_fraction[i] = reference(vector, net_seed, schedule, mode, cut=cut)
        answers[i] = _base.cpu_answers(vector[:len(BASE_GENE_RANGES)], activity, schedule,
                                       feedback=feedback)
        if z is not None:
            z[i] = activity
    return answers, spike_fraction, z


def evaluate_population(vectors, net_seed, schedule, *, mode="base", device="cpu",
                        cut=False, feedback=True, features=False, batch_size=32):
    """Evaluate normalized genomes against caller-provided continuous lives.

    ``vectors`` has shape ``(population, genes)`` with values in [0, 1] and gene
    order given by the exported ranges. Use ``base`` or one of the second-
    compartment modes ``basal``, ``add``, ``gain`` and ``both``.

    Schedule fields are integer arrays of shape ``(lives, trials)``. Each trial
    delivers context ``ctx`` (0–1) at tick ``issue``. Cue ``cue`` (0–3) arrives
    at ``raw_at``, or at ``issue`` if ``raw_at`` is 255. The ``distractor``
    (0–7) arrives at tick ``at``. Use 255 for ``at`` to skip the distractor.
    ``query`` captures the features within ``length`` ticks. The four-action
    readout uses normalized ``tie`` jitter shaped ``(lives, trials, 4)`` and
    correctness against ``goal`` (0–3) for reward. ``seeds`` supplies one ID per
    life. Each life carries its state across all trials in the supplied order.

    Returns NumPy ``answers`` with shape ``(population, lives, trials)``,
    per-genome ``fitness`` (proportion correct) and ``spike_fraction`` (spikes
    divided by the number of cell-ticks). Request ``features=True`` to include
    ``z`` of shape ``(population, lives, trials, 16)``. The CPU backend uses the
    original reference. The supplied CUDA backend for the second compartment
    supports only ``cuda:0``. Each GPU launch evaluates at most ``batch_size``
    candidates, preserving order and limiting temporary device storage.
    """
    if mode != "base" and mode not in _DENDRITE_MODES:
        raise ValueError(f"unknown population evaluation mode {mode!r}")
    ranges = BASE_GENE_RANGES if mode == "base" else DENDRITE_GENE_RANGES
    vectors = np.array(vectors, order="C", copy=True)
    if vectors.ndim != 2 or vectors.shape[1] != len(ranges) or not vectors.shape[0]:
        raise ValueError(f"vectors must have nonempty (population, {len(ranges)}) shape")
    if (vectors.dtype.kind not in "fiu" or not np.isfinite(vectors).all()
            or (vectors < 0).any() or (vectors > 1).any()):
        raise ValueError("gene values must be finite real numbers in [0, 1]")
    net_seed = operator.index(net_seed)
    if net_seed < 0:
        raise ValueError("net_seed must be a nonnegative integer")
    batch_size = operator.index(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    schedule = _schedule_for_kernel(schedule)
    dev = torch.device(device)
    if dev.type == "cpu":
        answers, spike_fraction, z = _cpu_batch(vectors, net_seed, schedule, mode, cut, feedback, features)
    elif dev.type == "cuda":
        dev = _cuda_device(dev, mode, min(batch_size, len(vectors)), schedule)
        shape = (len(vectors), *schedule.goal.shape)
        answers = np.empty(shape, np.uint8)
        spike_fraction = np.empty(len(vectors), np.float64)
        z = np.empty((*shape, _base.K), np.float32) if features else None
        with torch.cuda.device(dev):
            for start in range(0, len(vectors), batch_size):
                lanes = slice(start, start + batch_size)
                if mode == "base":
                    raw = _base.gpu_batch(vectors[lanes], net_seed, schedule, cut=cut,
                                          feedback=feedback, features=features, device=dev)
                    batch_answers, rate, activity = raw["answers"], raw["spike_fraction"], raw.get("z")
                else:
                    batch_answers, rate, activity = _dendrite.run(
                        vectors[lanes], net_seed, schedule, mode=mode, cut=cut,
                        feedback=feedback, features=features)
                answers[lanes], spike_fraction[lanes] = batch_answers, rate
                if z is not None:
                    z[lanes] = activity
    else:
        raise ValueError(f"expected a CPU or CUDA device, got {dev}")

    result = {"answers": answers, "fitness": (answers == schedule.goal).mean(axis=(1, 2)),
              "spike_fraction": spike_fraction}
    if z is not None:
        result["z"] = z
    return result
