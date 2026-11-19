"""Continue directed evolution with coactivity computed inside its CUDA updates."""

from __future__ import annotations

import argparse
import gzip
import json
import sys

import torch

from . import uncapped_directed_continuation_run as run
from . import uncapped_directed_reward as backend
from .uncapped_directed_fused_accumulation_probe import FusedAccumulationEvaluation, _backend_accumulation
from .uncapped_directed_reward_cuda_prepared import _FusedPreparedPolicy


class PreparedEvaluation(FusedAccumulationEvaluation):
    def __call__(self, *args, **kwargs):
        if torch.device(kwargs["device"]).type != "cuda":
            return super().__call__(*args, **kwargs)
        original = backend._DirectedRewardPolicy
        backend._DirectedRewardPolicy = _FusedPreparedPolicy
        try:
            with _backend_accumulation(kwargs["device"], self._fp16_accumulation):
                return backend.RewardEvaluation.__call__(self, *args, **kwargs)
        finally:
            backend._DirectedRewardPolicy = original


def _evaluate(schedule, **settings):
    evaluator = PreparedEvaluation if schedule is run.original.training_schedule else FusedAccumulationEvaluation
    return evaluator(schedule, **settings)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--execution-batch-size", type=int, choices=(16, 32), default=32)
    settings, rest = parser.parse_known_args()
    original = run.FusedAccumulationEvaluation
    previous_batches = run.runner.batches
    previous_argv = sys.argv

    def batches(folder, entries, config, previous, pool, depth):
        size = settings.execution_batch_size
        committed = sorted((folder / "batches").glob("*.json.gz"))
        if committed:
            # Finish the current journal with its original batch boundaries.
            size = len(json.loads(gzip.decompress(committed[0].read_bytes())))
        yield from previous_batches(folder, entries, {**config, "batch_size": size}, previous, pool, depth)

    run.FusedAccumulationEvaluation = _evaluate
    run.runner.batches = batches
    sys.argv = [sys.argv[0], *rest]
    try:
        print("Training fuses endpoint-product preparation and local reward writes.", flush=True)
        print(f"Execution batches use {settings.execution_batch_size} candidates after the current generation journal.", flush=True)
        run.main()
    finally:
        run.FusedAccumulationEvaluation = original
        run.runner.batches = previous_batches
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
