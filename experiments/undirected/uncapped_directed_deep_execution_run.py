"""Use fused node and support operations in the directed CUDA experiment."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import torch

from . import uncapped_directed_execution_run as execution
from . import uncapped_directed_prepared_run as prepared
from .uncapped_directed_node_fused import _FusedNodePolicy
from .uncapped_directed_support_fused import _FusedSupportPolicy


class _CombinedPolicy(_FusedNodePolicy, _FusedSupportPolicy):
    pass


class DeepEvaluation(execution.OptimizedEvaluation):
    def __init__(self, schedule, *, policy_type=_CombinedPolicy, **settings):
        @lru_cache(maxsize=1)
        def cached_schedule(**world):
            tape = schedule(**world)
            return replace(tape, public=tape.public.to(torch.float16))

        super().__init__(cached_schedule, **settings)
        self.policy_type = policy_type

    def __call__(self, *args, **kwargs):
        original = prepared._FusedPreparedPolicy
        prepared._FusedPreparedPolicy = self.policy_type
        try:
            return super().__call__(*args, **kwargs)
        finally:
            prepared._FusedPreparedPolicy = original


def main():
    original = execution.OptimizedEvaluation
    execution.OptimizedEvaluation = DeepEvaluation
    try:
        print("Training fuses node transitions and support updates.", flush=True)
        execution.main()
    finally:
        execution.OptimizedEvaluation = original


if __name__ == "__main__":
    main()
