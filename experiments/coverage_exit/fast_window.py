"""Opt-in FP64 rollout for the frozen undirected models.

Each call spans one constant set of parameters. The matrix is local to the
call, including every probe trial and every recorded historical parameter set.
The live world, optimizer, history reservoir and credit horizon stay under the
native trainer's control. Install again when restoring a native checkpoint.
"""
import math
from types import MethodType

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from experiments.undirected.credit import weighted_loss
from experiments.undirected.model import PortGraphModel, UndirectedModel


class WindowBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observations, hidden, elapsed, collect_logits, check_finite):
        model = self.model
        if len(observations) != len(elapsed) or hidden.shape != (*observations.shape[1:-1], model.width):
            raise ValueError("incompatible trajectory shapes")
        matrix = model.recurrent_matrix()
        outputs, check_states, check_logits = [], [], []
        for obs, dt in zip(observations, elapsed):
            if not math.isfinite(dt) or dt < 0:
                raise ValueError("invalid time increment")
            steps = max(1, math.ceil(dt / model.max_step))
            delta = dt / steps
            for _ in range(steps):
                first = -hidden + F.linear(hidden.tanh(), matrix)
                predictor = hidden + delta * first
                second = -predictor + F.linear(predictor.tanh(), matrix)
                hidden = hidden + .5 * delta * (first + second)
            hidden = hidden + model.observation_pulse(obs)
            if collect_logits or check_finite:
                output = model.read_state(hidden).reshape(*hidden.shape[:-1], model.n_jobs, model.n_actions)
                if collect_logits:
                    outputs.append(output)
                if check_finite:
                    check_states.append(hidden.detach())
                    check_logits.append(output.detach())
        if check_finite and check_states:
            if not torch.isfinite(torch.stack(check_states)).all():
                raise FloatingPointError("nonfinite replayed hidden state")
            if not torch.isfinite(torch.stack(check_logits)).all():
                raise FloatingPointError("nonfinite replayed logits")
        return (torch.stack(outputs) if outputs else None), hidden


def _recorded_replay(engine, records, *, start_h=None):
    block = WindowBlock(engine.model)
    hidden = records[0]["start_h"].to(engine.device) if start_h is None else start_h
    max_error = 0.
    for i, record in enumerate(records):
        max_error = max(max_error, engine._same_hidden(hidden, record["start_h"], "block start_h"))
        params = {"model." + k: v for k, v in engine._shifted_parameters(record).items()}
        buffers = {"model." + k: v.to(engine.device) for k, v in record["buffers"].items()}
        outputs, hidden = functional_call(block, (params, buffers),
            (record["observations"].to(engine.device), hidden, record["elapsed"], i == len(records)-1, False),
            strict=True)
        max_error = max(max_error, engine._same_hidden(hidden, record["end_h"], "block end_h"))
    return outputs, hidden, max_error


def _auxiliary_replay(trainer, record, warmup, *, differentiable=True):
    hidden = record["start_h"].to(trainer.device).detach().clone()
    block = WindowBlock(trainer.model)
    observations = record["observations"].to(trainer.device)
    if not torch.isfinite(observations).all():
        raise ValueError("nonfinite retained observation data")
    elapsed = record["elapsed"]
    if warmup:
        with torch.no_grad():
            _, hidden = block(observations[:warmup], hidden, elapsed[:warmup], False, True)
    with torch.set_grad_enabled(differentiable):
        logits, _ = block(observations[warmup:], hidden, elapsed[warmup:], True, True)
        loss = weighted_loss(logits, record["targets"][warmup:].to(trainer.device), trainer.config.wait_weight)[0]
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite replayed loss")
    return loss


def install_window(trainer):
    """Attach the fast recorded and auxiliary rollouts at an optimizer boundary."""
    if type(trainer.model) not in (UndirectedModel, PortGraphModel) or trainer.credit.model is not trainer.model:
        raise ValueError("requires a native undirected model shared with its credit engine")
    if not trainer._at_boundary or next(trainer.model.parameters()).dtype != torch.float64:
        raise ValueError("install at an FP64 optimizer boundary")
    trainer.credit._replay = MethodType(_recorded_replay, trainer.credit)
    trainer._replay_loss = MethodType(_auxiliary_replay, trainer)
    return trainer
