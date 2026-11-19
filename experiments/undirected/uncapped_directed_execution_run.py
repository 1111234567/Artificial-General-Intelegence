"""Combine the verified CUDA policy and environment execution optimizations."""

from __future__ import annotations

import torch

from . import uncapped_directed_continuation_run as run
from . import uncapped_directed_reward as backend
from . import uncapped_directed_prepared_run as prepared
from .uncapped_directed_packed import PackedCaptureMixin, unpack_scores_packed
from .uncapped_directed_world_fused import rollout_with_table


class _PackedCapture(PackedCaptureMixin, backend._RewardCapture):
    pass


class OptimizedEvaluation(prepared.PreparedEvaluation):
    def __init__(self, schedule, *, fp16_accumulation=False, fuse_world=True, pack_results=True):
        super().__init__(schedule, fp16_accumulation=fp16_accumulation)
        self.fuse_world = fuse_world
        self.pack_results = pack_results

    def __call__(self, *args, **kwargs):
        if torch.device(kwargs["device"]).type != "cuda":
            return super().__call__(*args, **kwargs)
        old_world = backend._rollout_with_preloaded_receipts
        old_capture = backend._RewardCapture
        if self.fuse_world:
            backend._rollout_with_preloaded_receipts = rollout_with_table
        if self.pack_results:
            backend._RewardCapture = _PackedCapture
        try:
            return super().__call__(*args, **kwargs)
        finally:
            backend._rollout_with_preloaded_receipts = old_world
            backend._RewardCapture = old_capture


def _evaluation(schedule, **settings):
    if schedule is run.original.training_schedule:
        return OptimizedEvaluation(schedule, **settings)
    return prepared.FusedAccumulationEvaluation(schedule, **settings)


def main():
    original_evaluation = prepared._evaluate
    original_unpack = run.runner.base.unpack_scores
    prepared._evaluate = _evaluation

    def unpack(fields, size):
        return unpack_scores_packed(fields, size, unpack_scores=original_unpack)

    run.runner.base.unpack_scores = unpack
    try:
        print("Training fuses environment transitions and groups result transfers by dtype.", flush=True)
        prepared.main()
    finally:
        prepared._evaluate = original_evaluation
        run.runner.base.unpack_scores = original_unpack


if __name__ == "__main__":
    main()
