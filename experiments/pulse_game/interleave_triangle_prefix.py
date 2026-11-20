"""Share pre-taste prefixes in the fully crossed triangle program."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import interleave_triangle_train as tri
from . import self_param_writer as writer


net = tri.native
MODES = tri.CONTROLS
PUBLIC = ("left", "right", "order", "study_flip", "study_u", "probe_u", "query", "query_flip", "query_u",
          "donor_cycle", "wait")


@dataclass(frozen=True)
class PrefixTape:
    public: dict
    law: torch.Tensor
    canonical: tuple
    mode: str


def prepare(tape, mode="active"):
    """Validate shared geometry and construct canonical-to-revelation indices."""
    if mode not in MODES:
        raise ValueError(f"Unknown triangle mode {mode!r}")
    law = tape["law"]
    if law.ndim != 2 or law.shape[1] != 3 or not len(law) or len(law) % 8:
        raise ValueError("Expected adjacent sets of eight private assignments")
    n = len(law) // 8
    labels = law.new_tensor(tri.LABELS).expand(n, -1, -1)
    if not torch.equal(law.reshape(n, 8, 3), labels):
        raise ValueError("The eight canonical law rows must be in the archival order")
    public = {}
    for key in PUBLIC:
        groups = tape[key].unflatten(0, (n, 8))
        first = groups[:, 0]
        if not torch.equal(groups, first[:, None].expand_as(groups)):
            raise ValueError(f"The public {key} field differs between private assignments")
        public[key] = first
    order = public["order"]
    axis = torch.arange(3, device=law.device)
    if order.shape != (n, 3) or not torch.equal(order.sort(1).values, axis.expand(n, 3)):
        raise ValueError("Every cue must be studied once")
    revealed = ((order + public["donor_cycle"][:, None]) % 3 if mode == "shuffled_objects" else order)
    bits = (torch.arange(8, device=law.device)[None, :, None] >> (2 - revealed[:, None])) & 1
    chronological = (bits * axis.new_tensor([4, 2, 1])).sum(2)
    world_index = torch.arange(n, device=law.device)[:, None]
    canonical = tuple(((world_index << t) + (chronological >> (3 - t))).flatten() for t in range(4))
    return PrefixTape(public, law, canonical, mode)


def _repeat(tensor, lanes):
    return tensor.repeat_interleave(lanes, dim=0) if lanes > 1 else tensor


def _split(state):
    graph = state.graph
    branch = net.Graph(*(_repeat(getattr(graph, name), 2) for name in ("weights", "bias", "present")))
    return net.State(branch, *(_repeat(getattr(state, name), 2) for name in ("hidden", "sender_trace", "receiver_trace")))


def _menus(public, item, flip, lanes):
    left, right = tri.binding.present(public, item, flip)
    return _repeat(left, lanes), _repeat(right, lanes)


def _study(graph, root_rule, packed, *, wait, trace):
    pub, law, canonical, mode = packed.public, packed.law, packed.canonical, packed.mode
    worlds = len(law) // (8 * len(graph.bias))
    if worlds < 1 or len(law) != worlds * len(graph.bias) * 8:
        raise ValueError("Each birth must carry the same number of worlds")
    state = net.start(graph, worlds)
    rule = root_rule.expand(worlds)
    study = state.hidden.new_zeros(len(state.hidden))
    edit = study
    for t in range(3):
        width = 1 << t
        item, flip = pub["order"][:, t], pub["study_flip"][:, t]
        left, right = _menus(pub, item, flip, width)
        draw = pub["study_u"][:, t]
        action = net.act(rule, state, left, right, _repeat(draw[:, 0], width))
        # The contact stimulus is still public. Split only at the hidden taste.
        contact = net._advance(rule, action.state, writer.sensory(action.state.graph, left, right, action.side),
                               net.CONTACT_TICKS)
        contact = _split(contact)
        left, right, side = _repeat(left, 2), _repeat(right, 2), _repeat(action.side, 2)
        law_bit = side.new_tensor([-1., 1.]).repeat(len(side) // 2)
        taste = side * _repeat(flip, width * 2) * law_bit
        if mode == "false_sign":
            taste = -taste
        rule = root_rule.expand(worlds * width * 2)
        post = net._advance(rule, contact, writer.sensory(contact.graph, left, right, side, taste), net.TASTE_TICKS)
        delta = post.hidden - contact.hidden
        feedback = torch.stack((contact.hidden, post.hidden, delta, torch.ones_like(delta)), dim=2)
        experience = net.Experience(post, side, taste, feedback)
        update = net.commit(rule, experience, _repeat(draw[:, 1:], width * 2),
                            intervention="sham" if mode == "sham" else "active")
        state = update.state
        if wait is None:
            state = net.wait(rule, state, _repeat(pub["wait"][:, t], width * 2), max_ticks=8)
        else:
            state = net.wait(rule, state, wait)
        study = _repeat(study + action.motor_log_prob, 2)
        edit = _repeat(edit, 2) + update.edit_log_prob
        if trace is not None:
            parent, children = canonical[t:t + 2]
            physical_side = action.side[parent]
            actual = physical_side * _repeat(flip, 8) * law.gather(1, _repeat(item, 8)[:, None]).squeeze(1)
            trace.append(dict(logit=action.motor_logit[parent].detach().clone(), side=physical_side.detach(),
                              earned=actual.detach(), taste=taste[children].detach(),
                              birth=update.receipt.birth[children].detach(), retired=update.receipt.retired[children].detach()))
    return state, rule, study, edit


def training_step(graph, rule, tape, worlds, *, trace=None):
    """Native objective with shared-prefix autograd. Floating reductions may round differently."""
    packed = prepare(tape)
    if len(packed.law) != len(graph.bias) * worlds * 8 or worlds < 2:
        raise ValueError("Training requires at least two crossed worlds per birth")
    state, rule, study, edit = _study(graph, rule, packed, wait=None, trace=trace)
    pub, flat = packed.public, packed.canonical[-1]
    item, flip = pub["query"], pub["query_flip"]
    left, right = _menus(pub, item, flip, 8)
    terminal = net.act(rule, state, left, right, _repeat(pub["query_u"], 8))
    side = terminal.side[flat]
    earned = (side * _repeat(flip, 8) * packed.law.gather(1, _repeat(item, 8)[:, None]).squeeze(1)).detach()
    if trace is not None:
        trace.append(dict(logit=terminal.motor_logit[flat].detach(), side=side.detach(), earned=earned))
    earned = earned.reshape(-1, worlds, 8)
    mean = earned.mean(2, keepdim=True)
    advantage = earned - (mean.sum(1, keepdim=True) - mean) / (worlds - 1)
    loss = [tri.old_fit.score_loss(score[flat].reshape_as(earned), advantage)
            for score in (terminal.motor_log_prob, study, edit)]
    with torch.no_grad():
        correct = (earned > 0).float()
        metrics = torch.stack((correct.mean((1, 2)), correct[..., [0, 7]].mean((1, 2)),
                               correct[..., 1:7].mean((1, 2)), *loss), dim=1)
    return sum(loss).sum(), metrics


def optimizer_step(graph, rule, optimizer, tape, worlds, *, allow_approximate=False, trace=None):
    """Opt in to reordered FP32 gradient accumulation. Keep the original FP64 fallback."""
    if not allow_approximate:
        raise ValueError("Use allow_approximate=True to accept floating gradient reduction drift")
    params = (graph.weights, graph.bias, *rule.parameters())
    for recovery in (False, True):
        optimizer.zero_grad(set_to_none=True)
        if recovery:
            model = net.Graph(nn.Parameter(graph.weights.detach().double()), nn.Parameter(graph.bias.detach().double()),
                              graph.present)
            coefficients = deepcopy(rule).double()
        else:
            model, coefficients = graph, rule
        group = (model.weights, model.bias, *coefficients.parameters())
        loss, metrics = training_step(model, coefficients, tape, worlds, trace=trace)
        if loss.isfinite():
            loss.backward()
            norm = tri.revisit._gradients(group, 2.0)
            if norm is not None:
                if recovery:
                    for parameter, source in zip(params, group, strict=True):
                        parameter.grad = None if source.grad is None else source.grad.to(parameter)
                optimizer.step()
                return torch.cat((metrics.double(), norm[:, None], torch.full_like(norm[:, None], recovery)), dim=1)
    raise FloatingPointError("Non-finite gradient after same-world float64 replay")


@torch.no_grad()
def trials(graph, rule, tape, mode="active", *, trace=None):
    """Read all three probes twice from the final state, with the archival four-tick waits."""
    packed = prepare(tape, mode)
    state, rule, _, _ = _study(graph, rule, packed, wait=4, trace=trace)
    pub, law, leaf = packed.public, packed.law, packed.canonical[-1]
    n = len(law) // 8
    logits, selected = [], []
    for obj in range(3):
        item = torch.full((n,), obj, device=law.device)
        actions = []
        for orientation in (1., -1.):
            left, right = _menus(pub, item, law.new_full((n,), orientation), 8)
            draw = _repeat(pub["probe_u"][:, obj], 8)
            actions.append(net.act(rule, state, left, right, draw if orientation > 0 else 1 - draw))
        logits.append(torch.stack((actions[0].motor_logit[leaf], actions[1].motor_logit[leaf]), dim=1))
        selected.append(torch.stack((actions[0].side[leaf], -actions[1].side[leaf]), dim=1))
    delivered = law
    if mode == "false_sign":
        delivered = -law
    elif mode == "shuffled_objects":
        other = (torch.arange(3, device=law.device) + _repeat(pub["donor_cycle"], 8)[:, None]) % 3
        delivered = law.gather(1, other)
    return dict(logit=torch.stack(logits, dim=1).numpy(), selected=torch.stack(selected, dim=1).numpy().astype(np.int8),
                observed=law.numpy(), delivered=delivered.numpy())
