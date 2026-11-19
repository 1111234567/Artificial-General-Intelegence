"""Execute directed plasticity on compact active-pair work lists."""

from __future__ import annotations

from . import uncapped_directed_deep_execution_run as deep
from . import uncapped_directed_execution_run as execution
from . import uncapped_directed_reward as backend
from .uncapped_directed_cached_transfer import CachedTapeCaptureMixin
from .uncapped_directed_node_fused import _FusedNodePolicy
from .uncapped_directed_reward_cuda_compact import _CompactRewardPolicy
from .uncapped_directed_support_scattered import _ScatteredSupportPolicy


class _CompactPolicy(_CompactRewardPolicy, deep._CombinedPolicy):
    pass


class _ScatterPolicy(_FusedNodePolicy, _ScatteredSupportPolicy):
    pass


class _CompactScatteredPolicy(_CompactRewardPolicy, _FusedNodePolicy, _ScatteredSupportPolicy):
    pass


class _CachedCapture(CachedTapeCaptureMixin, backend._RewardCapture):
    pass


class CompactEvaluation(deep.DeepEvaluation):
    def __init__(self, schedule, *, policy_type=_CompactPolicy, cache_transfers=False, **settings):
        super().__init__(schedule, policy_type=policy_type, **settings)
        self.cache_transfers = cache_transfers

    def __call__(self, *args, **kwargs):
        original = execution._PackedCapture
        if self.cache_transfers:
            execution._PackedCapture = _CachedCapture
        try:
            return super().__call__(*args, **kwargs)
        finally:
            execution._PackedCapture = original


def main():
    original = deep.DeepEvaluation
    deep.DeepEvaluation = CompactEvaluation
    try:
        print("Training uses compact active-pair work and cached immutable world tapes.", flush=True)
        deep.main()
    finally:
        deep.DeepEvaluation = original


if __name__ == "__main__":
    main()
