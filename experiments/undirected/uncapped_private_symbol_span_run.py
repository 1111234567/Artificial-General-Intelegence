"""Continue the main private-symbol search with the verified span backend."""

from . import uncapped_private_symbol_fast_run as fast
from .uncapped_directed_support_rebuild import _SupportRebuildMixin
from .uncapped_private_symbol_span import SpanEvaluation, SpanPolicy


class _MainSpanPolicy(_SupportRebuildMixin, SpanPolicy):
    _support_cooldown_threads = 256
    _span_threads = 256
    _span_matrix_stride = 104
    _span_lookup_tanh = True
    _span_shared_deltas = False


class MainSpanEvaluation(SpanEvaluation):
    span_policy = _MainSpanPolicy


def main():
    original = fast.TunedEvaluation
    fast.TunedEvaluation = MainSpanEvaluation
    try:
        print("Private-symbol training replays ordinary ticks in on-chip spans and defers eligibility writes.",
              flush=True)
        fast.main()
    finally:
        fast.TunedEvaluation = original


if __name__ == "__main__":
    main()
