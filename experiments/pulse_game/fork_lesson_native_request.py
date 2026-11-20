"""Ask one internal motor question and read one native primary neuron."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from . import fork_lesson_whatif as protocol


WORLD = protocol.world
READOUT = 14


class NativeRequest(WORLD.ForkGraph):
    def __init__(self, birth):
        super().__init__(birth)
        self.trial_gain = nn.Parameter(torch.full((2,), 1.2))

    def probe_current(self, query):
        chosen = F.one_hot(query, num_classes=2).bool()
        return torch.where(chosen, self.trial_gain.clamp(0., 3.), -0.8)


@dataclass
class Trial:
    final: WORLD.State
    forecast: torch.Tensor
    pulse: torch.Tensor
    first_primary: torch.Tensor
    returns: torch.Tensor


@dataclass
class Selection:
    side: torch.Tensor
    probability: torch.Tensor
    loglikelihood: torch.Tensor
    logit: torch.Tensor


@dataclass
class Rollout:
    requested: Selection
    action: Selection
    driven: torch.Tensor
    trial: Trial
    paid: torch.Tensor


def prepare(model, scene):
    state = WORLD.cue(model, WORLD.support(model, scene), scene)
    return WORLD.choice_window(model, state, ticks=protocol.PREPARATION_TICKS)[0]


def select(model, state, draws, exploration, forced=None):
    logit = model.logit(state)
    side, prob, score = protocol.pick(logit, draws, exploration, forced)
    return Selection(side, prob, score, logit)


def try_side(model, prepared, query, *, mode="natural", feedback=None):
    region = WORLD.REGION
    pulse = model.probe_current(query)
    if mode == "blank":
        pulse = torch.zeros_like(pulse)
    state = prepared
    zero = state.h.new_zeros((len(state.h), WORLD.PORTS))
    returns = []
    for tick in range(protocol.TRIAL_TICKS):
        outgoing = None
        if tick == 0:
            outward = protocol.bridge_current(model.return_weight * model.mask[region:, :region],
                                               state.fast[:, region:, :region], state.h[:, :region])
            outgoing = outward + F.pad(pulse, (10, region - 12))
        state, back = model.tick(state, zero, outbound_override=outgoing,
                                 feedback_override=None if feedback is None else feedback[tick],
                                 disable_feedback=mode in ("cut_return", "isolate_return"),
                                 adoption_off=mode == "isolate_return")
        returns.append(back)
        if tick == 0:
            first_primary = state.h[:, :region]
        if tick == protocol.FORECAST_TICK - 1:
            forecast = state.h[:, READOUT]
    return Trial(state, forecast, pulse, first_primary, torch.stack(returns))


def settle(model, state, scene, decision):
    choice = WORLD.Choice(decision.logit, decision.side, torch.zeros_like(decision.logit), decision.loglikelihood)
    with torch.no_grad():
        _, outcome = WORLD.commit(model, state, scene, choice)
    return outcome.payoff


def roll(model, scene, query_draw, action_draw, *, shuffle=False):
    state = prepare(model, scene)
    query = select(model, state, query_draw, protocol.QUERY_EXPLORATION)
    driven = query.side.roll(-1) if shuffle else query.side
    trial = try_side(model, state, driven)
    action = select(model, trial.final, action_draw, protocol.ACTION_EXPLORATION)
    return Rollout(query, action, driven, trial, settle(model, trial.final, scene, action))
