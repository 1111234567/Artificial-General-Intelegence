"""Record native IID inputs around one answer and its distractor requests."""

from dataclasses import replace

import torch

from . import uncapped_private_symbol_world as world
from .evolved_plasticity_world import (
    CONTEXT, FIRST_FINISH, INPUT_SIZE, NEW, N_SYMBOLS, RAW,
)


_DISTRACTORS = 8


def _write_new_keys(frame, raw, context):
    port = frame[..., 0, :]
    new = port[..., NEW].ne(0).unsqueeze(-1)
    raw = torch.as_tensor(raw, device=frame.device)
    context = torch.as_tensor(context, device=frame.device)
    symbols = torch.arange(N_SYMBOLS, device=frame.device)
    contexts = torch.arange(CONTEXT.stop - CONTEXT.start, device=frame.device)
    port[..., RAW].copy_(new & raw.unsqueeze(-1).eq(symbols))
    port[..., CONTEXT].copy_(new & context.unsqueeze(-1).eq(contexts))


def probe_frame(frame, raw, context):
    """Copy one public frame and set its primary-port keys on the NEW pulse."""
    frame = frame.clone()
    _write_new_keys(frame, raw, context)
    return frame


def set_keys(tape, idx, raw, context):
    """Return a copy with new public keys and matching private targets at idx."""
    shape = tape.raw[idx].shape
    raw = torch.as_tensor(raw, dtype=tape.raw.dtype, device=tape.raw.device).expand(shape)
    context = torch.as_tensor(context, dtype=tape.context.dtype, device=tape.context.device).expand(shape)
    public = tape.public.clone()
    targets = tape.target.clone()
    raw_keys = tape.raw.clone()
    context_keys = tape.context.clone()
    raw_keys[idx] = raw
    context_keys[idx] = context
    actors = torch.arange(shape[-1], device=tape.maps.device)
    targets[idx] = FIRST_FINISH + tape.maps[actors, context, raw]
    _write_new_keys(public[idx], raw.unsqueeze(-2), context.unsqueeze(-2))
    return replace(tape, public=public, target=targets, raw=raw_keys, context=context_keys)


class _NativeTrace:
    def __init__(self, policy, tape, source, before_source):
        self.policy = policy
        self.before_source = before_source
        ticks, actors = tape.stage.shape[1:]
        self.ticks = ticks
        self.offset = -source * ticks
        shape = (_DISTRACTORS + 1, ticks, policy.population, actors)
        self.inputs = tape.public.new_empty((*shape, INPUT_SIZE))
        self.actions = torch.empty(shape, device=tape.public.device, dtype=torch.long)

    def act(self, obs):
        if self.offset == 0:
            self.before_source()
        if self.offset >= 0:
            request, tick = divmod(self.offset, self.ticks)
            self.inputs[request, tick].copy_(obs)
        action = self.policy.act(obs)
        if self.offset >= 0:
            self.actions[request, tick].copy_(action[..., 0])
        self.offset += 1
        return action


@torch.no_grad()
def native_inputs(policy, tape, table, source, before_source):
    """Run through eight distractors, recording native observations and actions."""
    end = source + 1 + _DISTRACTORS
    if source < 0 or end > tape.raw.size(0):
        raise ValueError("source must leave eight distractor requests in the tape")
    prefix = replace(tape, **{
        field: getattr(tape, field)[:end] for field in ("public", "stage", "target", "raw", "context")
    })
    distractors = slice(source + 1, end)
    raw = (prefix.raw[source] + 2 + prefix.raw[distractors].remainder(2)) % N_SYMBOLS
    prefix = set_keys(prefix, distractors, raw, prefix.context[distractors])
    trace = _NativeTrace(policy, prefix, source, before_source)
    world.rollout_with_table(table)(trace, prefix, policy.population)
    return trace.inputs, trace.actions
