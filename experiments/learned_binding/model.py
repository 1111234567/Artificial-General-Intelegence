"""A learned public-observation binder around the frozen native GRU policy.

Each job has a 32-scalar recurrent state and shares one 16-by-4 delta memory.
No cue identities, receipt parsing, success masks, or private world fields are
provided to this controller.  Both jobs write on every tick, in job order, with
learned gates.  Only the executor's known, fixed context permutations are used
by the differentiable command-payload interface.

Development-stage revision, before the formal comparison was frozen: initialize
the encoder's learned update-gate bias to +4 in total, and expose the current
28 public features directly to all memory heads alongside the 32 encoded
features.  Neither change adds persistent state or a hand-written receipt rule.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from experiments.learn_remember.gru_train import GRUAgent
from experiments.lifelong.world import FAMILY_MAPPINGS


FEATURES_PER_JOB = 28
N_JOBS = 2
ENCODER_WIDTH = 32
HEAD_INPUT_WIDTH = ENCODER_WIDTH + FEATURES_PER_JOB
KEY_DIM = 16
N_VALUES = 4
MEMORY_SCALARS = KEY_DIM * N_VALUES
EXECUTOR_CONFIG = dict(width=192, input_size=56, n_jobs=2, n_actions=11)


def _mapping_tensor(original_mappings):
    values = torch.as_tensor(original_mappings)
    if values.shape != (4, 4) or values.dtype == torch.bool:
        raise ValueError("original_mappings must be four permutations of 0..3")
    integers = values.to(dtype=torch.long)
    if (not torch.equal(values, integers)
            or not torch.equal(integers.sort(-1).values,
                               torch.arange(4, device=integers.device).expand(4, 4))):
        raise ValueError("original_mappings must be four permutations of 0..3")
    return integers.clone()


def soft_translate(obs, read_probs, original_mappings=FAMILY_MAPPINGS):
    """Map target probabilities to canonical raw probabilities on commands.

    Accept ``[B,56]`` or ``[T,B,56]`` public observations, with corresponding
    ``[...,2,4]`` probabilities.  For a public one-hot context ``c``, the raw
    probability is ``raw[r] = read_probs[original_mappings[c,r]]``.  Context is
    combined linearly, without argmax or a detached probability path.  Every
    non-payload feature, and all non-command payloads, are retained exactly.
    """
    if obs.ndim not in (2, 3) or obs.shape[-1] != 56:
        raise ValueError("obs must have shape [batch,56] or [time,batch,56]")
    if read_probs.shape != (*obs.shape[:-1], N_JOBS, N_VALUES):
        raise ValueError("read_probs must have shape [...,2,4] matching obs")
    if not obs.is_floating_point() or not read_probs.is_floating_point():
        raise TypeError("obs and read_probs must be floating point")
    if obs.device != read_probs.device or obs.dtype != read_probs.dtype:
        raise ValueError("obs and read_probs must have the same device and dtype")
    # The registered model buffer is already validated at construction.  This
    # helper also accepts the public constant directly for standalone audits.
    mappings = (original_mappings if isinstance(original_mappings, torch.Tensor)
                else _mapping_tensor(original_mappings))
    if mappings.shape != (4, 4) or mappings.dtype != torch.long:
        raise ValueError("original_mappings must be a [4,4] long tensor")
    mappings = mappings.to(device=obs.device)
    public = obs.reshape(*obs.shape[:-1], N_JOBS, FEATURES_PER_JOB)
    command = (public[..., 0] > .5) | (public[..., 1] > .5)
    # [...,context,raw], followed by the original public context mixture.
    raw_probs = (read_probs[..., mappings] * public[..., 24:28, None]).sum(-2)
    payload = torch.where(command[..., None], raw_probs, public[..., 5:9])
    return torch.cat((public[..., :5], payload, public[..., 9:]), -1).reshape_as(obs)


def detach_state(state):
    """Detach exactly the three explicit recurrent-state tensors for TBPTT."""
    if set(state) != {"h", "z", "memory"}:
        raise ValueError("state must contain h, z, and memory only")
    return {name: value.detach() for name, value in state.items()}


class NativeMemoryPolicy(nn.Module):
    """Learn all binding and write timing from the unchanged public stream.

    The complete sequence first passes through a shared job encoder, then a
    causal delta-memory scan, then the original executor's fused GRU.  Executor
    weights are frozen, while its input Jacobian remains in the autograd graph.
    No additional action head is introduced.
    """

    def __init__(self, executor=None, executor_config=None,
                 original_mappings=FAMILY_MAPPINGS):
        super().__init__()
        if executor is not None and executor_config is not None:
            raise ValueError("pass an executor or executor_config, not both")
        if executor is None:
            config = EXECUTOR_CONFIG if executor_config is None else executor_config
            executor = GRUAgent(**config)
        if not isinstance(executor, GRUAgent) or executor.config() != EXECUTOR_CONFIG:
            raise ValueError("executor must be the original 192/56/2/11 GRUAgent")
        self.executor = executor
        self.executor.requires_grad_(False)
        # Clear pre-existing training gradients without changing any weights.
        for parameter in self.executor.parameters():
            parameter.grad = None
        self.encoder = nn.GRU(FEATURES_PER_JOB, ENCODER_WIDTH)
        # PyTorch GRU gate order is reset, update, new.  This is a learned
        # retention prior, not a fixed clock, TTL, or hard state-reset rule.
        nn.init.constant_(self.encoder.bias_ih_l0[ENCODER_WIDTH:2 * ENCODER_WIDTH], 2.)
        nn.init.constant_(self.encoder.bias_hh_l0[ENCODER_WIDTH:2 * ENCODER_WIDTH], 2.)
        self.query = nn.Linear(HEAD_INPUT_WIDTH, KEY_DIM)
        self.key = nn.Linear(HEAD_INPUT_WIDTH, KEY_DIM)
        self.value = nn.Linear(HEAD_INPUT_WIDTH, N_VALUES)
        self.gate = nn.Linear(HEAD_INPUT_WIDTH, 1)
        for projection in (self.query, self.key, self.value, self.gate):
            nn.init.normal_(projection.weight, mean=0., std=.02)
            nn.init.zeros_(projection.bias)
        nn.init.constant_(self.gate.bias, -5.)
        self.read_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(5.))))
        self.register_buffer("original_mappings", _mapping_tensor(original_mappings))
        # A supplied executor can already reside on CUDA or use float64.
        reference = self.executor.head.weight
        self.to(device=reference.device, dtype=reference.dtype)

    @property
    def read_scale(self):
        return F.softplus(self.read_scale_raw)

    def initial_state(self, batch_size, device=None):
        if (isinstance(batch_size, bool) or not isinstance(batch_size, Integral)
                or batch_size < 0):
            raise ValueError("batch_size must be a nonnegative integer")
        reference = self.encoder.weight_ih_l0
        options = dict(device=reference.device if device is None else device,
                       dtype=reference.dtype)
        return dict(h=torch.zeros(batch_size, 192, **options),
                    z=torch.zeros(batch_size, N_JOBS, ENCODER_WIDTH, **options),
                    memory=torch.zeros(batch_size, MEMORY_SCALARS, **options))

    detach_state = staticmethod(detach_state)

    def _check_inputs(self, obs, state, write_enabled, read_enabled):
        if obs.ndim != 3 or obs.shape[-1] != 56 or obs.shape[0] == 0:
            raise ValueError("obs must have nonempty shape [time,batch,56]")
        if not obs.is_floating_point():
            raise TypeError("obs must be floating point")
        if not isinstance(write_enabled, bool) or not isinstance(read_enabled, bool):
            raise TypeError("write_enabled and read_enabled must be global booleans")
        if set(state) != {"h", "z", "memory"}:
            raise ValueError("state must contain h, z, and memory only")
        batch = obs.shape[1]
        shapes = dict(h=(batch, 192), z=(batch, N_JOBS, ENCODER_WIDTH),
                      memory=(batch, MEMORY_SCALARS))
        reference = self.encoder.weight_ih_l0
        for name, shape in shapes.items():
            value = state[name]
            if value.shape != shape:
                raise ValueError(f"state {name} must have shape {shape}")
            if value.device != obs.device or value.dtype != obs.dtype:
                raise ValueError("state and obs must have the same device and dtype")
        if obs.device != reference.device or obs.dtype != reference.dtype:
            raise ValueError("model and obs must have the same device and dtype")

    def forward(self, obs, state, *, write_enabled=True, read_enabled=True,
                return_diagnostics=False):
        self._check_inputs(obs, state, write_enabled, read_enabled)
        ticks, batch = obs.shape[:2]
        encoded, next_z = self.encoder(
            obs.reshape(ticks, batch * N_JOBS, FEATURES_PER_JOB),
            state["z"].reshape(1, batch * N_JOBS, ENCODER_WIDTH))
        encoded = encoded.reshape(ticks, batch, N_JOBS, ENCODER_WIDTH)
        head_input = torch.cat((encoded, obs.reshape(
            ticks, batch, N_JOBS, FEATURES_PER_JOB)), dim=-1)
        q = F.normalize(self.query(head_input), dim=-1)
        k = F.normalize(self.key(head_input), dim=-1)
        value = self.value(head_input).softmax(-1)
        gate = self.gate(head_input).sigmoid()
        matrix = state["memory"].reshape(batch, KEY_DIM, N_VALUES)
        read_scale = self.read_scale
        reads = []
        if write_enabled:
            # Unbind once, rather than differentiable indexing at every tick:
            # its backward stacks gradients once instead of repeatedly making
            # a full-sequence zero tensor.  Weighting keys outside the scan and
            # baddbmm also reduce tiny device kernels without reducing dtype.
            keys_0, keys_1 = k.unbind(2)
            weighted_0, weighted_1 = (gate * k).unbind(2)
            values_0, values_1 = value.unbind(2)
            frames = zip(q.unbind(0), keys_0.unbind(0), keys_1.unbind(0),
                         weighted_0.unbind(0), weighted_1.unbind(0),
                         values_0.unbind(0), values_1.unbind(0))
            for query, key_0, key_1, weighted_0, weighted_1, value_0, value_1 in frames:
                prediction_0 = torch.bmm(key_0[:, None], matrix).squeeze(1)
                matrix = torch.baddbmm(matrix, weighted_0[:, :, None],
                                      (value_0 - prediction_0)[:, None])
                prediction_1 = torch.bmm(key_1[:, None], matrix).squeeze(1)
                matrix = torch.baddbmm(matrix, weighted_1[:, :, None],
                                      (value_1 - prediction_1)[:, None])
                if read_enabled:
                    reads.append((read_scale * torch.bmm(query, matrix)).softmax(-1))
        elif read_enabled:
            # The matrix is constant in this ablation; batch all query reads.
            read_logits = read_scale * torch.matmul(q, matrix[None])
            reads = [read_logits.softmax(-1)]
        if read_enabled:
            read_probs = torch.stack(reads) if write_enabled else reads[0]
        else:
            # The same soft interface applies to every ablation arm.
            read_probs = obs.new_full((ticks, batch, N_JOBS, N_VALUES), .25)
        mapped_obs = soft_translate(obs, read_probs, self.original_mappings)
        logits, next_h = self.executor(mapped_obs, state["h"])
        memory = (matrix.reshape(batch, MEMORY_SCALARS) if write_enabled
                  else state["memory"])
        next_state = dict(h=next_h,
                          z=next_z.reshape(batch, N_JOBS, ENCODER_WIDTH),
                          memory=memory)
        if not return_diagnostics:
            return logits, next_state
        diagnostics = dict(q=q, k=k, value=value, gate=gate,
                           read_probs=read_probs, mapped_obs=mapped_obs, z=encoded)
        return logits, next_state, diagnostics

    def step(self, obs, state, *, write_enabled=True, read_enabled=True,
             return_diagnostics=False):
        if obs.ndim != 2 or obs.shape[-1] != 56:
            raise ValueError("step obs must have shape [batch,56]")
        result = self.forward(obs[None], state, write_enabled=write_enabled,
                              read_enabled=read_enabled,
                              return_diagnostics=return_diagnostics)
        if return_diagnostics:
            logits, state, diagnostics = result
            return logits[0], state, {name: value[0]
                                     for name, value in diagnostics.items()}
        logits, state = result
        return logits[0], state

    def config(self):
        return dict(executor_config=self.executor.config(),
                    original_mappings=self.original_mappings.cpu().tolist())

    def parameter_report(self):
        by_name = {name: parameter.numel()
                   for name, parameter in self.named_parameters()}
        executor_count = sum(p.numel() for p in self.executor.parameters())
        return dict(kind="learned_public_binding_delta_native_executor",
                    total_parameters=sum(by_name.values()),
                    trainable_parameters=sum(p.numel() for p in self.parameters()
                                             if p.requires_grad),
                    executor_parameters=executor_count,
                    encoder_parameters=sum(p.numel() for p in self.encoder.parameters()),
                    projection_parameters=sum(p.numel() for module in
                        (self.query, self.key, self.value, self.gate)
                        for p in module.parameters()),
                    read_scale_parameters=1, state_scalars=192 + 64 + 64,
                    executor_state_scalars=192, encoder_state_scalars=64,
                    memory_state_scalars=64, parameters_by_name=by_name,
                    head_input_features=HEAD_INPUT_WIDTH,
                    initial_encoder_update_bias_total=4.,
                    architecture_revision="development_before_formal_freeze",
                    public_inputs_only=True, learned_write_gates=True,
                    write_order=[0, 1],
                    read_scale=float(self.read_scale.detach().cpu()))
