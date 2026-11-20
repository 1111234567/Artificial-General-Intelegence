"""Let the primary population try one motor pulse inside its replica before the real choice."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from . import fork_lesson_world as world


PREPARATION_TICKS = 3
FORECAST_TICK = 3
AFTER_FORECAST_TICKS = 2
TRIAL_TICKS = FORECAST_TICK + AFTER_FORECAST_TICKS
QUERY_EXPLORATION = 0.5
ACTION_EXPLORATION = 0.3
MOTOR_PORTS = slice(10, 12)


@dataclass
class Trial:
    pulse: torch.Tensor
    feedback: torch.Tensor
    forecast: torch.Tensor
    forecast_state: world.State
    final_state: world.State


@dataclass
class Rollout:
    query: torch.Tensor
    driven_query: torch.Tensor
    query_right: torch.Tensor
    action: torch.Tensor
    action_right: torch.Tensor
    forecast: torch.Tensor
    actual_payoff: torch.Tensor
    motor_logit: torch.Tensor
    query_logit: torch.Tensor
    query_score: torch.Tensor
    action_score: torch.Tensor
    immediate: torch.Tensor
    trial: Trial


class WhatIfGraph(nn.Module):
    """Trainable interfaces around the same lifetime recurrence and plasticity as the whole-body fit."""

    def __init__(self, body):
        super().__init__()
        self.body = body.requires_grad_(False)
        # A matched pair of axons targets the replica's homologous motor neurons.
        self.send = nn.Parameter(torch.tensor(((1.2, -0.8), (-0.8, 1.2))))
        self.relay = nn.Parameter(torch.zeros(world.REGION, 2))
        self.query_readout = nn.Linear(world.REGION, 1, bias=False)
        self.motor_readout = nn.Linear(world.REGION, 1, bias=False)
        self.forecast_base = nn.Linear(world.REGION, 1, bias=True)
        self.forecast_residual = nn.Sequential(nn.Linear(world.REGION, 16), nn.Tanh(), nn.Linear(16, 1))
        with torch.no_grad():
            self.query_readout.weight.zero_()
            self.motor_readout.weight.zero_()
            self.forecast_base.weight.zero_()
            self.forecast_base.weight[:, 12:14] = 0.5
            self.forecast_base.bias.zero_()
            self.forecast_residual[-1].weight.zero_()
            self.forecast_residual[-1].bias.zero_()

    def forecast(self, state):
        primary = state.h[:, :world.REGION]
        return (self.forecast_base(primary) + self.forecast_residual(primary)).squeeze(1).tanh()

    def query_logit(self, state):
        return self.body.logit(state) + self.query_readout(state.h[:, :world.REGION]).squeeze(1)

    def motor_logit(self, state):
        return self.body.logit(state) + self.motor_readout(state.h[:, :world.REGION]).squeeze(1)


def start_trial(model, episode):
    state = world.cue(model.body, world.support(model.body, episode), episode)
    prepared, _, _ = world.choice_window(model.body, state, ticks=PREPARATION_TICKS)
    return prepared


def probability(logit, exploration):
    return (1.0 - exploration) * logit.sigmoid() + exploration / 2.0


def pick(logit, draws, exploration, forced=None, *, greedy=False):
    right = probability(logit, exploration)
    if forced is None:
        side = (logit > 0 if greedy else draws < right.detach()).long()
    else:
        side = torch.as_tensor(forced, dtype=torch.long, device=logit.device).expand_as(logit)
        if ((side < 0) | (side > 1)).any():
            raise ValueError("A trial or real side must be zero or one")
    score = torch.where(side.bool(), right.log(), torch.log1p(-right))
    return side, right, score


def bridge_current(weights, fast, activity):
    return F.linear(activity, weights) + torch.bmm(fast, activity[:, :, None]).squeeze(2)


def perform_trial(model, prepared, query, *, driven=None, blank=False, feedback_override=None,
                  cut_return=False):
    """Only the pulse at the replica's motor ports depends on the physically driven side."""
    state = prepared
    n = world.REGION
    if feedback_override is not None and feedback_override.shape != (TRIAL_TICKS, len(state.h), n):
        raise ValueError(f"Expected returning currents with shape {(TRIAL_TICKS, len(state.h), n)}")
    pulse_port = model.send[query if driven is None else driven].clamp(-3.0, 3.0)
    pulse = pulse_port.new_zeros((len(pulse_port), n))
    if not blank:
        pulse[:, MOTOR_PORTS] = pulse_port
    ordinary_back = model.body.weight[:, n:] * model.body.mask[:n, n:]
    ordinary_out = model.body.return_weight * model.body.mask[n:, :n]
    zero = state.h.new_zeros((len(state.h), world.PORTS))
    feedback = []
    for tick in range(TRIAL_TICKS):
        primary, replica = state.h.split(n, dim=1)
        outbound = None
        if tick == 0:
            outbound = bridge_current(ordinary_out, state.fast[:, n:, :n], primary) + pulse
        if feedback_override is not None:
            back = feedback_override[tick]
        elif cut_return:
            back = torch.zeros_like(primary)
        else:
            back = bridge_current(ordinary_back, state.fast[:, :n, n:], replica)
            if tick:
                back = back + F.linear(replica[:, MOTOR_PORTS], model.relay)
        state, delivered = model.body.tick(state, zero, outbound_override=outbound, feedback_override=back)
        feedback.append(delivered)
        if tick == FORECAST_TICK - 1:
            forecast_state = state
    forecast = model.forecast(forecast_state)
    return Trial(pulse, torch.stack(feedback), forecast, forecast_state, state)


def rollout(model, episode, trial_draw, action_draw, *, forced_query=None, forced_action=None,
            drive_query=None, shuffle_trial=False, blank_trial=False, greedy=False,
            feedback_override=None, cut_return=False, prepared=None):
    state = start_trial(model, episode) if prepared is None else prepared
    query_logit = model.query_logit(state)
    query, query_right, query_score = pick(query_logit, trial_draw, QUERY_EXPLORATION, forced_query, greedy=greedy)
    if drive_query is not None and shuffle_trial:
        raise ValueError("Choose one way to replace the requested trial pulse")
    driven = (query.roll(-1) if shuffle_trial else query) if drive_query is None else torch.as_tensor(
        drive_query, dtype=torch.long, device=query.device).expand_as(query)
    trial = perform_trial(model, state, query, driven=driven, blank=blank_trial,
                          feedback_override=feedback_override, cut_return=cut_return)
    motor_logit = model.motor_logit(trial.final_state)
    action, action_right, action_score = pick(motor_logit, action_draw, ACTION_EXPLORATION, forced_action, greedy=greedy)
    immediate = torch.zeros_like(motor_logit)
    choice = world.Choice(motor_logit, action, immediate, action_score)
    with torch.no_grad():
        _, outcome = world.commit(model.body, trial.final_state, episode, choice)
    return Rollout(query, driven, query_right, action, action_right, trial.forecast, outcome.payoff, motor_logit,
                   query_logit, query_score, action_score, immediate, trial)
