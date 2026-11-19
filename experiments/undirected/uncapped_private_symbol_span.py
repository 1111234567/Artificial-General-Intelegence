"""Replay ordinary private-symbol ticks with the recurrent matrix on chip."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from ._cuda_nvrtc import compile_kernels
from . import uncapped_directed_world_fused as fused
from . import uncapped_private_symbol_world as world
from . import uncapped_private_symbol_world_cuda as cuda_world
from .evolved_plasticity import GATE_INTERVAL, JOBS, WEIGHT_LIMIT
from .uncapped_answer_feedback import _answer_metrics
from .uncapped_directed_reward_encoded_pairs import _encoded_source
from .uncapped_directed_reward_fast_run import FIRST_RECEIPT, REQUEST_TICKS
from .uncapped_directed_reward_sparse_reduction import _replace_once
from .uncapped_directed_support_table import _table_support
from .uncapped_private_symbol_delta import DELTA_FIELD, _mean_surviving_delta, delta_rollout as reference_delta_rollout
from .uncapped_private_symbol_kernel_run import KernelEvaluation, KernelPolicy
from .uncapped_reward_precision import _first_finish_metrics


@lru_cache(maxsize=12)
def _kernel(device, defer_eligibility, matrix_stride, lookup_tanh, shared_deltas, fused_rows=False):
    folder = Path(__file__).parent
    names = ("uncapped_directed_reward_persistent.cu", "uncapped_private_symbol_world_device.cu")
    source = ("#define NODE_INPUTS 56\n#define NODE_ACK 22\n#define NODE_REJ 23\n"
              f"#define SPAN_LAZY {int(defer_eligibility)}\n#define SPAN_MATRIX_STRIDE {matrix_stride}\n"
              f"#define SPAN_TABLE {int(lookup_tanh)}\n#define SPAN_SHARED_DELTAS {int(shared_deltas)}\n"
              f"#define SPAN_FUSED_ROWS {int(fused_rows)}\n") + _encoded_source()
    reward = (folder / "uncapped_directed_reward_prepare.cu").read_text(encoding="utf-8")
    reward = _replace_once(reward, "u * prepare_nodes + v", "u * SPAN_MATRIX_STRIDE + v")
    reward = _replace_once(reward, "v * prepare_nodes + u", "v * SPAN_MATRIX_STRIDE + u")
    source += "\n" + reward
    source += "\n" + "\n".join((folder / name).read_text(encoding="utf-8") for name in names)
    matvec = (folder / "uncapped_directed_matvec_tensor.cu").read_text(encoding="utf-8")
    source += "\n" + matvec[matvec.index("__device__"):matvec.index('extern "C"')]
    source += "\n" + Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels(source, ("directed_private_span",), device=device)["directed_private_span"]


class SpanPolicy(KernelPolicy):
    _defer_eligibility = True
    _span_matrix_stride = 96
    _span_lookup_tanh = False
    _span_shared_deltas = True
    _span_threads = 192
    _span_fused_rows = False

    def __init__(self, birth, actors, device, mode):
        super().__init__(birth, actors, device, mode)
        self.span_kernel = (_kernel(device, self._defer_eligibility, self._span_matrix_stride,
                                    self._span_lookup_tanh, self._span_shared_deltas, self._span_fused_rows)
                            if self._tensor_transition and self._encoded_build else None)
        self.span_tanh_table = ctypes.c_void_p()
        if self.span_kernel and self._span_lookup_tanh:
            _, self.span_tanh_table, ready = _table_support(self.device)
            torch.cuda.current_stream(self.device).wait_event(ready)


class _SpanDispatch:
    def __init__(self, policy, tape, table, trace):
        self.policy = policy
        self.tape = tape
        self.table = table
        self.population = policy.population
        self.requests, self.ticks, self.actors = tape.stage.shape
        self.batch = self.population * self.actors
        self.trace = trace.view(-1, self.batch) if trace is not None else None
        self.step = -1
        self.actions = torch.empty((self.population, self.actors, JOBS), device=policy.device, dtype=torch.long)
        self.kernels = cuda_world._kernels(policy.device)
        self.observation_grid = ((self.batch * 32 + fused._THREADS - 1) // fused._THREADS,)
        self.world_buffers = None

    def observe(self, grid, block, *arguments):
        self.step += 1
        phase = self.step % GATE_INTERVAL
        if phase == 0:
            self.kernels["private_packed_observation"](self.observation_grid, block, *arguments)
        elif phase == 1:
            self.span(min(self.step + GATE_INTERVAL - 1, self.requests * self.ticks))

    def act(self, obs):
        if self.step % GATE_INTERVAL == 0:
            return self.policy.act(obs)
        return self.actions

    def record(self, grid, block, *arguments):
        if self.step % GATE_INTERVAL != 0:
            return
        if self.world_buffers is None:
            self.world_buffers = arguments[3:7]
        trace = ctypes.c_void_p() if self.trace is None else self.trace[self.step]
        self.kernels["directed_world_action"](grid, block, *arguments, trace)

    def span(self, stop):
        policy = self.policy
        if policy.step != self.step:
            raise RuntimeError("span start disagrees with the support clock")
        policy.span_kernel(
            (self.batch,), (policy._span_threads,), self.tape.public, self.tape.stage, self.tape.target,
            self.table, policy.span_tanh_table,
            *self.world_buffers, self.trace if self.trace is not None else ctypes.c_void_p(),
            policy.bias, policy.matrix, policy.h, policy.activity, policy.node_trace,
            policy.tried, policy.pending, policy.reward, policy.reward_events, policy.reward_rate,
            policy._active_pairs, policy._active_pair_counts, policy.mask, policy.eligibility,
            policy.w, policy.proposal, policy.attempted_writes, policy.weight_updates, policy.weight_change,
            self.batch, self.actors, self.requests, self.ticks, self.step, stop, FIRST_RECEIPT, WEIGHT_LIMIT,
        )
        policy.step = stop


def _rollout(policy, tape, table, trace):
    if policy.span_kernel is None or tape.stage.size(1) != REQUEST_TICKS or not tape.public.is_contiguous():
        return cuda_world._rollout_cuda(policy, tape, table, trace=trace, packed_observation=True)
    dispatch = _SpanDispatch(policy, tape, table, trace)
    kernels = {"directed_world_observation": dispatch.observe, "directed_world_action": dispatch.record}
    return fused._rollout_cuda(dispatch, tape, policy.population, table, kernels)


@torch.no_grad()
def observed_rollout(policy, tape, table):
    if table.device.type != "cuda":
        return world.observed_rollout(policy, tape, table)
    requests, ticks, actors = tape.stage.shape
    trace = torch.empty((requests, ticks, policy.population, actors), device=table.device, dtype=torch.long)
    scores = _rollout(policy, tape, table, trace)
    normalized = world.normalize_actions(trace)
    return {**policy.summary(), **scores, **_answer_metrics(normalized, tape),
            **_first_finish_metrics(normalized, tape), **world._symbol_metrics(normalized, tape)}, trace


def task_rollout(policy, tape, table):
    return observed_rollout(policy, tape, table)[0]


@torch.no_grad()
def delta_rollout(policy, tape, table):
    if table.device.type != "cuda":
        return cuda_world.packed_delta_rollout(policy, tape, table)
    born_weight = policy.w[:, 0].to(dtype=torch.float32, copy=True)
    born_mask = policy.mask[:, 0].clone()
    _rollout(policy, tape, table, None)
    return {DELTA_FIELD: _mean_surviving_delta(policy, born_weight, born_mask)}


class SpanEvaluation(KernelEvaluation):
    span_policy = SpanPolicy

    def __init__(self, schedule, *, rollout=world.task_rollout, **options):
        if schedule is world.training_schedule and not options.get("policy_type"):
            if rollout is world.task_rollout:
                rollout = task_rollout
            elif rollout is reference_delta_rollout:
                rollout = delta_rollout
        super().__init__(schedule, rollout=rollout, **options)
        if self.reference is None:
            self.policy_type = self.span_policy
