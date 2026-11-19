"""Continue the private-symbol search with fused support and encoded pairs."""

from . import uncapped_private_symbol_fast_run as fast
from .uncapped_directed_reward_encoded_pairs import _EncodedPairMixin
from .uncapped_directed_support_tick import _SupportTickPolicy


class _FusedPolicy(_EncodedPairMixin, _SupportTickPolicy, fast._TunedPolicy):
    pass


class FusedEvaluation(fast.TunedEvaluation):
    def __init__(self, schedule, **options):
        super().__init__(schedule, **options)
        if self.reference is None:
            self.policy_type = _FusedPolicy


def main():
    original = fast.TunedEvaluation
    fast.TunedEvaluation = FusedEvaluation
    try:
        print("Private-symbol training uses fused support ticks and encoded active pairs.", flush=True)
        fast.main()
    finally:
        fast.TunedEvaluation = original


if __name__ == "__main__":
    main()
