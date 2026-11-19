"""Resume the directed reward run with statically empty writes skipped."""

from __future__ import annotations

from functools import partial

import torch

from . import uncapped_directed_reward as backend
from . import uncapped_directed_reward_run as run
from .evolved_plasticity import GATE_INTERVAL
from .evolved_plasticity_world import READY
from .uncapped_island_crossover_run import block_ticks


REQUEST_TICKS = block_ticks(run.TIMING)
FIRST_RECEIPT = run.TIMING["min_wait"] + 1


class _SkipEmptyReceiptTicks(backend._DirectedRewardPolicy):
    def _change_weights(self, coactivity, event):
        if self.step % REQUEST_TICKS >= FIRST_RECEIPT:
            return super()._change_weights(coactivity, event)
        self.eligibility.lerp_(coactivity, .1).masked_fill_(~self.mask, 0.)

    def _update_matrix(self):
        if self.step % REQUEST_TICKS >= FIRST_RECEIPT or self.step % GATE_INTERVAL == 0:
            super()._update_matrix()


def _checked_schedule(schedule, **settings):
    tape = schedule(**settings)
    if tape.public.size(1) != REQUEST_TICKS:
        raise ValueError("Unexpected request length in training tape")
    ready = tape.public[..., 0, READY].flatten(0, 1).bool().any(-1)
    next_position = (torch.arange(len(ready), device=ready.device) + 1) % REQUEST_TICKS
    if (ready & next_position.lt(FIRST_RECEIPT)).any():
        raise ValueError("Training tape permits a receipt in a skipped write tick")
    return tape


class _FastTrainingEvaluation(backend.RewardEvaluation):
    def __init__(self, schedule):
        super().__init__(partial(_checked_schedule, schedule))

    def __call__(self, *args, **kwargs):
        # The runner evaluates on its main thread. Prefetch only builds genomes.
        original = backend._DirectedRewardPolicy
        backend._DirectedRewardPolicy = _SkipEmptyReceiptTicks
        try:
            return super().__call__(*args, **kwargs)
        finally:
            backend._DirectedRewardPolicy = original


def _evaluation(schedule):
    if schedule is run.training_schedule:
        return _FastTrainingEvaluation(schedule)
    return backend.RewardEvaluation(schedule)


def main():
    original = run.RewardEvaluation
    run.RewardEvaluation = _evaluation
    try:
        print("Training skips guaranteed-empty receipt writes after tape validation.", flush=True)
        run.main()
    finally:
        run.RewardEvaluation = original


if __name__ == "__main__":
    main()
