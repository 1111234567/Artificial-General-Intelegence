"""Continue the private-symbol search with fused reward, support and transitions."""

import torch

from . import uncapped_private_symbol_fast_run as fast
from . import uncapped_private_symbol_world as world
from . import uncapped_private_symbol_world_cuda as cuda_world
from .evolved_plasticity import GATE_INTERVAL
from .uncapped_directed_matvec_tensor import _TensorTransitionMixin
from .uncapped_directed_reward_prepare import _RewardPrepareMixin
from .uncapped_directed_support_tile import _SupportTileMixin
from .uncapped_private_symbol_delta import delta_rollout
from .uncapped_private_symbol_fused_run import FusedEvaluation, _FusedPolicy


class KernelPolicy(_TensorTransitionMixin, _RewardPrepareMixin, _SupportTileMixin, _FusedPolicy):
    _fuse_reward_diagnostic = True

    def act(self, obs):
        if not (self._tensor_transition and self._reward_prepare and self._tiled_support) or not self.step:
            return super().act(obs)
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=self.h.dtype).contiguous()
        support_tick = self.step % GATE_INTERVAL == 0
        local = self._prepare_and_change_weights(obs, save_local=support_tick)
        if support_tick:
            self._update_tiled_support(local, obs)
        actions = self._tensor_step(obs)
        self.step += 1
        return actions


class KernelEvaluation(FusedEvaluation):
    def __init__(self, schedule, *, rollout=world.task_rollout, **options):
        if schedule is world.training_schedule and not options.get("policy_type"):
            if rollout is world.task_rollout:
                rollout = cuda_world.packed_task_rollout
            elif rollout is delta_rollout:
                rollout = cuda_world.packed_delta_rollout
        super().__init__(schedule, rollout=rollout, **options)
        if self.reference is None:
            self.policy_type = KernelPolicy


def main():
    original = fast.TunedEvaluation
    fast.TunedEvaluation = KernelEvaluation
    try:
        print("Private-symbol training fuses reward preparation, support and Tensor Core node transitions.", flush=True)
        fast.main()
    finally:
        fast.TunedEvaluation = original


if __name__ == "__main__":
    main()
