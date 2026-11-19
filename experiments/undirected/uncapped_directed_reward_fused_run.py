"""Run the directed experiment with fused CUDA reward and eligibility updates."""

from __future__ import annotations

import torch

from . import uncapped_directed_reward as backend
from . import uncapped_directed_reward_run as run
from .uncapped_directed_reward_cuda_fused import _FusedRewardPolicy
from .uncapped_directed_reward_fast_run import FIRST_RECEIPT, REQUEST_TICKS, _FastTrainingEvaluation


class _FusedTrainingPolicy(_FusedRewardPolicy):
    def _receipt_possible(self):
        return self.step % REQUEST_TICKS >= FIRST_RECEIPT


class _FusedTrainingEvaluation(_FastTrainingEvaluation):
    def __call__(self, *args, **kwargs):
        if torch.device(kwargs["device"]).type != "cuda":
            return super().__call__(*args, **kwargs)
        # Audit evaluators run separately on the reference policy.
        original = backend._DirectedRewardPolicy
        backend._DirectedRewardPolicy = _FusedTrainingPolicy
        try:
            return backend.RewardEvaluation.__call__(self, *args, **kwargs)
        finally:
            backend._DirectedRewardPolicy = original


def _evaluation(schedule):
    if schedule is run.training_schedule:
        return _FusedTrainingEvaluation(schedule)
    return backend.RewardEvaluation(schedule)


def main():
    original = run.RewardEvaluation
    run.RewardEvaluation = _evaluation
    try:
        print("Training fuses CUDA eligibility and reward updates and writes only changed matrix entries.", flush=True)
        run.main()
    finally:
        run.RewardEvaluation = original


if __name__ == "__main__":
    main()
