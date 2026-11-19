"""Run the private-symbol search with fixed-size parental replay and lean transfers."""

from __future__ import annotations

from functools import lru_cache

import torch

from . import uncapped_directed_packed as packed
from . import uncapped_private_symbol_run as run
from . import uncapped_private_symbol_world as world
from .uncapped_directed_support_selective import _SelectivePolicy
from .uncapped_directed_reward_sparse_reduction import _SparseDiagnosticPolicy
from .uncapped_private_symbol_delta import DELTA_FIELD, _mean_surviving_delta, delta_rollout, sample_deltas


class _TunedPolicy(_SparseDiagnosticPolicy, _SelectivePolicy):
    pass


class TunedEvaluation(run.PrivateEvaluation):
    def __init__(self, schedule, *, rollout=world.task_rollout, policy_type=None,
                 selective_support=True, cache_transfers=True, lean_delta=True):
        if rollout is delta_rollout and lean_delta:
            rollout = lean_delta_rollout
        super().__init__(schedule, rollout=rollout, policy_type=policy_type)
        if self.reference is None:
            if selective_support:
                self.policy_type = _TunedPolicy
            self.schedule = lru_cache(maxsize=1)(self.schedule)
            self.cache_transfers = cache_transfers


@torch.no_grad()
def lean_delta_rollout(policy, tape, table):
    born_weight = policy.w[:, 0].to(dtype=torch.float32, copy=True)
    born_mask = policy.mask[:, 0].clone()
    world.rollout_with_table(table)(policy, tape, policy.population)
    return {DELTA_FIELD: _mean_surviving_delta(policy, born_weight, born_mask)}


def padded_sample_deltas(evaluator, genomes, *, batch_size=16, **settings):
    """Keep the CUDA Graph shape stable when fewer parents fill the last batch."""
    device = torch.device(settings["device"])
    stable_padding = (batch_size == settings.get("actors", 16) == 16 and device.type == "cuda"
                      and str(torch.__version__) == "2.11.0+cu130"
                      and torch.cuda.get_device_capability(device) == (8, 9))

    def evaluate_batch(batch, **world_settings):
        count = len(batch)
        # The one-parent GEMM can choose a different summation path.
        if stable_padding and 2 <= count < batch_size:
            result = evaluator(batch + [batch[-1]] * (batch_size - count), **world_settings)
            return {DELTA_FIELD: result[DELTA_FIELD][:count]}
        return evaluator(batch, **world_settings)

    return sample_deltas(evaluate_batch, genomes, batch_size=batch_size, **settings)


def unpack_scores_lists(fields, size, unpack_scores=packed._unpack_scores):
    """Transfer each dtype once and slice its scalar columns as Python lists."""
    if (not isinstance(fields, packed.PackedResults)
            or len(fields) != len(fields._layout)
            or any(field.shape != (size,) or fields.get(field.name) is not fields._original_fields[field.name]
                   for field in fields._layout)):
        return packed_reference_unpack(fields, size, unpack_scores=unpack_scores)
    columns = [buffer.detach().cpu().tolist() for buffer in fields.packed_buffers]
    host = {field.name: columns[field.group][field.offset:field.offset + field.size]
            for field in fields._layout}
    return unpack_scores(host, size)


packed_reference_unpack = packed.unpack_scores_packed


def main():
    original_eval = run.PrivateEvaluation
    original_deltas = run.sample_deltas
    original_unpack = packed.unpack_scores_packed
    run.PrivateEvaluation = TunedEvaluation
    run.sample_deltas = padded_sample_deltas
    packed.unpack_scores_packed = unpack_scores_lists
    try:
        print("Private-symbol search uses selective support writes, event-sparse diagnostics, immutable tape reuse "
              "and fixed-size parent replay where numerically equivalent.", flush=True)
        run.main()
    finally:
        run.PrivateEvaluation = original_eval
        run.sample_deltas = original_deltas
        packed.unpack_scores_packed = original_unpack


if __name__ == "__main__":
    main()
