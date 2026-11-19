"""Every-tick directed plasticity for the November 19 private-symbol trial.

The arc order is the sealed directed order: the first half is v -> u for
u < v, then u -> v. Each life owns its weights, node biases and arc support.
The CUDA path and the Torch reference use the same inherited coefficients.
"""

from functools import lru_cache
from pathlib import Path
import sys

import torch

from .evolved_plasticity import ACTIONS, EDGE_COOLDOWN, GATE_INTERVAL, INPUTS, JOBS, RULE_FEATURES, WEIGHT_LIMIT
from .evolved_plasticity_world import FEATURES_PER_SLOT, FIRST_FINISH, N_ACTIONS, NEW, READY
from .uncapped_directed_reward import _DirectedRewardPolicy
from .uncapped_plastic_graph import BIRTH_THRESHOLD, DEATH_THRESHOLD, GATE_RATE as LEGACY_GATE_RATE
from .uncapped_reward_precision import _ACK, _REJ, _matmul_precision


PLASTIC_FIELDS = (
    "weight_pre", "weight_post", "weight_coactivity", "weight_decay", "weight_ack", "weight_rej",
    "bias_activity", "bias_trace", "bias_decay", "bias_ack", "bias_rej",
    "gate_bias", "gate_coactivity", "gate_trace", "gate_weight", "gate_ack", "gate_rej",
    "mod_activity", "mod_trace",
)
PLASTIC_SIZE = len(PLASTIC_FIELDS)
_PLASTIC_START = (
    0., 0., .04, .10, .015, -.015,
    -.015, 0., .10, .01, -.01,
    0., .03, 0., 0., .015, -.015,
    .05, .05,
)
_COUNTS = ("edge_count", "edge_ticks", "births", "deaths", "reward_events", "public_ack_events",
           "public_rej_events", "attempted_writes", "changed_writes", "weight_updates", "bias_updates")
_TOTALS = ("weight_change", "bias_change")
_CONSTANTS = {
    "INPUTS": INPUTS, "JOBS": JOBS, "ACTIONS": ACTIONS, "NEW": NEW, "READY": READY,
    "ACK": _ACK, "REJ": _REJ, "PORT_SIZE": FEATURES_PER_SLOT,
    "FIRST_FINISH": FIRST_FINISH, "N_ACTIONS": N_ACTIONS, "COOLDOWN": EDGE_COOLDOWN,
    "WEIGHT_LIMIT": WEIGHT_LIMIT, "BIAS_LIMIT": .12,
    "TRACE_RATE": .05, "ELIGIBILITY_RATE": .10, "GATE_RATE": 1. - (1. - LEGACY_GATE_RATE) ** (1. / GATE_INTERVAL),
    "RAMP_RATE": 1. / 32., "RULE_SIZE": len(RULE_FEATURES), "SEED_RANGE": .08, "SEED_OFFSET": .01,
    "BIRTH_THRESHOLD": BIRTH_THRESHOLD, "DEATH_THRESHOLD": DEATH_THRESHOLD,
    "WEIGHT_RATE": .002, "WEIGHT_RECEIPT_RATE": .02, "BIAS_RATE": .001,
    "BIAS_RECEIPT_RATE": .01, "SPARE_RECEIPT_RATE": .25, "MOD_SCALE": 2.,
}
_NODE_THREADS = 128
_ARC_THREADS = 256


def default_plastic():
    """Return a new CPU FP32 chromosome with small initial local updates."""
    return torch.tensor(_PLASTIC_START, dtype=torch.float32, device="cpu")


@lru_cache(maxsize=None)
def _cuda_kernels(device, tensor_transition):
    from ._cuda_nvrtc import compile_kernels

    def literal(value):
        return f"{value!r}f" if isinstance(value, float) else str(value)

    constants = {**_CONSTANTS, **{field.upper(): i for i, field in enumerate(PLASTIC_FIELDS)},
                 **{f"COUNT_{name.upper()}": i for i, name in enumerate(_COUNTS)},
                 **{f"TOTAL_{name.upper()}": i for i, name in enumerate(_TOTALS)},
                 "PLASTIC_SIZE": PLASTIC_SIZE, "ARC_THREADS": _ARC_THREADS,
                 "TENSOR_TRANSITION": int(tensor_transition)}
    definitions = "\n".join(f"#define {name} {literal(value)}" for name, value in constants.items())
    matvec = ""
    if tensor_transition:
        matvec = Path(__file__).with_name("uncapped_directed_matvec_tensor.cu").read_text(encoding="utf-8")
        matvec = matvec[:matvec.index('extern "C"')]
    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return compile_kernels("\n".join((definitions, matvec, source)),
                           ("local_rule_prepare", "local_rule_arcs", "local_rule_finish"), device=device)


def make_policy(birth, actors, device, mode, *, backend="auto"):
    """Select the sealed original rule or one of the two every-tick rules."""
    device = torch.device(device)
    if mode == "original":
        sealed_birth = {name: value.to(device) for name, value in birth.items() if name != "plastic"}
        return _DirectedRewardPolicy(sealed_birth, actors, device, "full")
    return LocalRulePolicy(birth, actors, device, mode, backend=backend)


class LocalRulePolicy:
    """Local traces, learned biases and hysteretic support with per-tick updates.

    ``local`` and ``modulated`` share all state and equations. In ``modulated``,
    the receiving node's instantaneous activity and activity trace modulate
    its incoming weight and gate evidence, and its own bias drive. With
    bounded inherited coefficients the gain can vary from about -0.93 to 2.93.
    Both arms retain the inherited support projection and signed birth seed.
    """

    @torch.no_grad()
    def __init__(self, birth, actors, device, mode, *, backend="auto"):
        if mode not in ("local", "modulated"):
            raise ValueError("expected 'local' or 'modulated'")
        if backend not in ("auto", "torch", "cuda"):
            raise ValueError("expected backend 'auto', 'torch' or 'cuda'")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda") or (backend == "cuda" and device.type != "cuda"):
            raise ValueError("local plasticity requires a CPU or CUDA device")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if actors < 1 or birth["node_bias"].ndim != 2 or birth["node_bias"].size(0) < 1:
            raise ValueError("local plasticity requires a positive population and actor count")
        self.mode = mode
        self.device = device
        self.actors = actors
        self.population, self.nodes = birth["node_bias"].shape
        if self.nodes < INPUTS + JOBS * ACTIONS:
            raise ValueError("graph must contain all fixed input and action ports")
        self._batch_shape = (self.population, self.actors)
        self._batch = self.population * self.actors
        self.input_shape = (*self._batch_shape, INPUTS)
        self.u, self.v = torch.triu_indices(self.nodes, self.nodes, offset=1, device=device)
        self.pair_count = self.u.numel()
        self._dst = torch.cat((self.u, self.v))
        self._src = torch.cat((self.v, self.u))
        self.arc_indices = self._dst * self.nodes + self._src
        self._cuda_indices = self.arc_indices.int()
        ports = (*self._batch_shape, self.nodes)
        arcs = (*self._batch_shape, 2, self.pair_count)

        self.clock = torch.empty((), device=device, dtype=torch.int32)
        self.initial_edges = torch.empty(self.population, device=device, dtype=torch.long)
        self.initial_w = torch.empty((self.population, 1, 2, self.pair_count), device=device, dtype=torch.float16)
        self.initial_bias = torch.empty((self.population, 1, self.nodes), device=device, dtype=torch.float16)
        self.plastic = torch.empty((self.population, PLASTIC_SIZE), device=device, dtype=torch.float32)
        self.legacy_rule = torch.empty((self.population, 2, len(RULE_FEATURES)), device=device, dtype=torch.float32)
        self.reward_rate = torch.empty(self.population, device=device, dtype=torch.float32)
        self._default = default_plastic().to(device).expand_as(self.plastic)
        self.h = torch.empty(ports, device=device, dtype=torch.float16)
        self.activity = torch.empty_like(self.h)
        self.bias = torch.empty_like(self.h, dtype=torch.float32)
        self.node_trace = torch.empty_like(self.bias)
        self._local_activity = torch.empty_like(self.bias)
        self._gain = torch.empty_like(self.bias)
        self._receipts = torch.empty((*self._batch_shape, 5), device=device, dtype=torch.float32)
        self.pending = torch.empty(self._batch_shape, device=device, dtype=torch.bool)
        self.tried = torch.empty_like(self.pending)
        self.mask = torch.empty(arcs, device=device, dtype=torch.bool)
        self.last_change = torch.empty(arcs, device=device, dtype=torch.int32)
        self.w = torch.empty(arcs, device=device, dtype=torch.float16)
        self.matrix = torch.empty((*ports, self.nodes), device=device, dtype=torch.float16)
        self.weight_state = torch.empty_like(self.w, dtype=torch.float32)
        self.g = torch.empty_like(self.weight_state)
        self.strength = torch.empty_like(self.w)
        self.eligibility = torch.empty_like(self.w)
        self._counts = torch.empty((len(_COUNTS), *self._batch_shape), device=device, dtype=torch.long)
        self._totals = torch.empty((len(_TOTALS), *self._batch_shape), device=device, dtype=torch.float32)
        for name, view in (*zip(_COUNTS, self._counts.unbind()), *zip(_TOTALS, self._totals.unbind())):
            setattr(self, name, view)
        self._tensor_transition = False
        self._cuda = None
        if device.type == "cuda" and (backend == "cuda" or (backend == "auto" and sys.platform == "win32")):
            self._tensor_transition = self.nodes == 96 and torch.cuda.get_device_capability(device)[0] >= 8
            self._cuda = _cuda_kernels(device, self._tensor_transition)
        self.reset(birth)

    @torch.no_grad()
    def reset(self, birth):
        """Copy birth tensors and reset every actor, including replay's device clock."""
        plastic = birth.get("plastic", self._default)
        if plastic.shape != self.plastic.shape or plastic.dtype != torch.float32:
            raise ValueError(f"expected FP32 plastic with shape {tuple(self.plastic.shape)}")
        self.plastic.copy_(plastic).clamp_(-1., 1.)
        self.legacy_rule.copy_(birth["rule"])
        self.reward_rate.copy_(birth["reward_rate"])
        birth_mask = birth["birth_mask"].view_as(self.initial_w).to(self.device)
        self.initial_w.copy_(birth["birth_w"].view_as(self.initial_w)).masked_fill_(~birth_mask, 0.)
        self.initial_bias.copy_(birth["node_bias"].unsqueeze(1))
        self.w.copy_(self.initial_w)
        self.weight_state.copy_(self.initial_w)
        self.bias.copy_(self.initial_bias)
        self.mask.copy_(birth_mask)
        self.strength.copy_(birth_mask)
        self.initial_edges.copy_(birth_mask.sum((1, 2, 3)))
        for state in (self.clock, self.h, self.activity, self.node_trace, self._local_activity, self._receipts,
                      self.pending, self.tried, self.last_change, self.matrix, self.g, self.eligibility,
                      self._counts, self._totals):
            state.zero_()
        self._gain.fill_(1.)
        self.edge_count.copy_(self.initial_edges.unsqueeze(1))
        self._update_matrix()

    def state_tensors(self):
        """Return the live tensors needed to snapshot and restore a tick boundary."""
        names = ("clock", "initial_edges", "initial_w", "initial_bias", "plastic", "legacy_rule", "reward_rate",
                 "h", "activity", "bias", "node_trace", "pending", "tried", "mask", "last_change", "w",
                 "matrix", "weight_state", "g", "strength", "eligibility", "_counts", "_totals")
        return {name: getattr(self, name) for name in names}

    def _update_matrix(self):
        self.matrix.flatten(-2).index_copy_(-1, self.arc_indices, self.w.flatten(-2))

    @torch.no_grad()
    def act(self, obs):
        if tuple(obs.shape) != self.input_shape:
            raise ValueError(f"expected observation shape {self.input_shape}")
        obs = obs.to(device=self.device, dtype=torch.float16).contiguous()
        if self._cuda is not None:
            return self._act_cuda(obs)
        started = self.clock.ne(0)
        self._prepare_reference(obs, started)
        self._arcs_reference(started)
        return self._finish_reference(obs)

    def _prepare_reference(self, obs, started):
        c = _CONSTANTS
        a = self._local_activity
        z = self.node_trace
        a.copy_((4. * self.h.float()).tanh())
        z.add_((a - z) * (started * c["TRACE_RATE"]))
        p = {name: self.plastic[:, i, None, None] for i, name in enumerate(PLASTIC_FIELDS)}
        if self.mode == "modulated":
            self._gain.copy_(1. + c["MOD_SCALE"] * (p["mod_activity"] * a + p["mod_trace"] * z).tanh())
        primary_ack = obs[..., _ACK].float()
        primary_rej = obs[..., _REJ].float()
        ack = (primary_ack + c["SPARE_RECEIPT_RATE"] * obs[..., FEATURES_PER_SLOT + _ACK].float()) * started
        rej = (primary_rej + c["SPARE_RECEIPT_RATE"] * obs[..., FEATURES_PER_SLOT + _REJ].float()) * started
        reward = (primary_ack - primary_rej) * (self.pending & started)
        ports = obs.reshape(*self._batch_shape, JOBS, FEATURES_PER_SLOT).float()
        begin = ports[..., 12:14].sum(-1)
        finish = ports[..., 14:22].sum(-1)
        legacy_signals = torch.stack(tuple((kind * ports[..., status]).mean(-1)
                                           for status in (_ACK, _REJ) for kind in (begin, finish)), dim=-1)
        legacy = self.legacy_rule[:, None, :, 5:]
        first, second, third, fourth = (legacy_signals[..., None, :] * legacy).unbind(-1)
        birth_receipt, gate_receipt = ((first + second + third + fourth) * started).unbind(-1)
        self._receipts.copy_(torch.stack((ack, rej, reward, birth_receipt, gate_receipt), dim=-1))
        self.reward_events.add_(reward.ne(0))
        self.public_ack_events.add_(ack.ne(0))
        self.public_rej_events.add_(rej.ne(0))

        bias_old = self.bias.half()
        bias_base = self.initial_bias.float()
        local = c["BIAS_RATE"] * (p["bias_activity"] * a + p["bias_trace"] * z)
        local += c["BIAS_RECEIPT_RATE"] * (p["bias_ack"] * ack[..., None] + p["bias_rej"] * rej[..., None]) * z
        bias_delta = self._gain * local - c["BIAS_RATE"] * p["bias_decay"] * (self.bias - bias_base)
        updated_bias = (self.bias + bias_delta).clamp(bias_base - c["BIAS_LIMIT"], bias_base + c["BIAS_LIMIT"])
        self.bias.copy_(torch.where(started, updated_bias, self.bias))
        bias_new = self.bias.half()
        self.bias_updates.add_(bias_new.ne(bias_old).sum(-1))
        self.bias_change.add_((bias_new.float() - bias_old.float()).abs().sum(-1))

    def _arcs_reference(self, started):
        c = _CONSTANTS
        p = {name: self.plastic[:, i, None, None] for i, name in enumerate(PLASTIC_FIELDS)}
        ack, rej, reward, birth_receipt, gate_receipt = (v[..., None] for v in self._receipts.unbind(-1))
        a = self._local_activity
        z = self.node_trace
        pre = z.index_select(-1, self._src)
        source_activity = a.index_select(-1, self._src)
        post = a.index_select(-1, self._dst)
        gain = self._gain.index_select(-1, self._dst)
        coactivity = pre * post
        trace_product = pre * z.index_select(-1, self._dst)
        eligible_product = source_activity * post

        g = self.g.flatten(-2)
        w = self.weight_state.flatten(-2)
        live = self.mask.flatten(-2)
        e = self.eligibility.flatten(-2)
        old_strength = self.strength.flatten(-2).float()
        last = self.last_change.flatten(-2)
        visible = self.w.flatten(-2)
        anchor = self.initial_w.flatten(-2).float()
        eligible = e.float() + c["ELIGIBILITY_RATE"] * (eligible_product - e.float())
        eligible = torch.where(live & started, eligible.half().float(), e.float())

        endpoint_difference = (source_activity - post).abs()
        legacy_support = self._legacy_drive(1, w, eligible_product, trace_product, endpoint_difference, gate_receipt)
        support = (legacy_support.tanh() + p["gate_bias"]
                   + p["gate_coactivity"] * coactivity + p["gate_trace"] * trace_product
                   + p["gate_weight"] * (w.abs() / c["WEIGHT_LIMIT"]))
        support += (p["gate_ack"] * ack + p["gate_rej"] * rej) * coactivity
        target = (gain * support).clamp(-1., 1.)
        g.add_((target - g) * (started * c["GATE_RATE"]))

        ready = started & (self.clock - last).ge(c["COOLDOWN"])
        admit = ~live & ready & g.ge(c["BIRTH_THRESHOLD"])
        fade = live & ready & g.le(c["DEATH_THRESHOLD"])
        ramp = c["RAMP_RATE"]
        strength = (old_strength + torch.where(fade, -ramp, ramp)).clamp(0., 1.)
        strength = torch.where((live | admit) & started, strength, old_strength)
        retire = fade & strength.eq(0.)

        local = c["WEIGHT_RATE"] * (p["weight_pre"] * pre + p["weight_post"] * post
                                      + p["weight_coactivity"] * coactivity)
        rest = c["WEIGHT_RATE"] * p["weight_decay"] * (w - anchor)
        receipt = (self.reward_rate[:, None, None] * reward
                   + c["WEIGHT_RECEIPT_RATE"] * (p["weight_ack"] * ack + p["weight_rej"] * rej)) * eligible
        no_receipt = (w + (gain * local - rest)).clamp(-c["WEIGHT_LIMIT"], c["WEIGHT_LIMIT"])
        new_weight = (w + (gain * (local + receipt) - rest)).clamp(-c["WEIGHT_LIMIT"], c["WEIGHT_LIMIT"])
        update = live & started & ~retire
        intrinsic = (new_weight * old_strength).half()
        without_receipt = (no_receipt * old_strength).half()
        weight_changes = update & intrinsic.ne(visible)
        self.attempted_writes.add_((update & receipt.ne(0.)).sum(-1))
        self.changed_writes.add_((update & intrinsic.ne(without_receipt)).sum(-1))
        self.weight_updates.add_(weight_changes.sum(-1))
        self.weight_change.add_((intrinsic.float() - visible.float()).abs().masked_fill(~weight_changes, 0.).sum(-1))

        birth_drive = self._legacy_drive(0, w, eligible_product, trace_product, endpoint_difference, birth_receipt).tanh()
        seed = c["SEED_RANGE"] * birth_drive + torch.where(birth_drive.ge(0.), c["SEED_OFFSET"], -c["SEED_OFFSET"])
        w.copy_(torch.where(admit, seed.half().float(), torch.where(update, new_weight, w)).masked_fill(retire, 0.))
        e.copy_(eligible.masked_fill(retire | admit, 0.))
        live.logical_xor_(admit | retire)
        last.copy_(torch.where(admit | retire, self.clock, last))
        self.strength.flatten(-2).copy_(strength)
        visible.copy_(w * strength)
        self._update_matrix()

        admitted, retired = admit.sum(-1), retire.sum(-1)
        self.births.add_(admitted)
        self.deaths.add_(retired)
        self.edge_count.add_(admitted).sub_(retired)
        self.edge_ticks.add_(self.edge_count)

    def _legacy_drive(self, row, weight, coactivity, trace, endpoint_difference, receipt):
        bias, prior_weight, prior_coactivity, prior_trace, prior_endpoint = (
            coeff[:, None, None] for coeff in self.legacy_rule[:, row, :5].unbind(-1)
        )
        return (bias + prior_weight * (weight / WEIGHT_LIMIT) + prior_coactivity * coactivity
                + (prior_trace + receipt) * trace + prior_endpoint * endpoint_difference)

    def _matvec(self):
        matrix = self.matrix.flatten(0, 1)
        activity = self.activity.flatten(0, 1).unsqueeze(-1)
        if self.device.type == "cpu":
            return torch.bmm(matrix.float(), activity.float()).half().reshape_as(self.h)
        with _matmul_precision(torch.float16, False):
            return torch.bmm(matrix, activity).reshape_as(self.h)

    def _finish_reference(self, obs):
        current = self._matvec()
        self.h.mul_(.75).add_(current + self.bias.half(), alpha=.25)
        self.h[..., :INPUTS].add_(obs, alpha=.25)
        action_ports = self.h[..., INPUTS:INPUTS + JOBS * ACTIONS]
        actions = action_ports.reshape(*self._batch_shape, JOBS, ACTIONS).argmax(-1)
        primary = actions[..., 0]
        self.tried.logical_and_(~obs[..., NEW].bool())
        self.pending.copy_(obs[..., READY].bool() & ~self.tried
                           & primary.ge(FIRST_FINISH) & primary.lt(N_ACTIONS))
        self.tried.logical_or_(self.pending)
        torch.tanh(self.h, out=self.activity)
        self.clock.add_(1)
        return actions

    def _act_cuda(self, obs):
        grid = (self._batch,)
        self._cuda["local_rule_prepare"](
            grid, (_NODE_THREADS,), self.clock, obs, self.h, self.pending, self.plastic, self.legacy_rule, self.initial_bias,
            self.bias, self.node_trace, self._local_activity, self._gain, self._receipts, self._counts,
            self._totals, self.actors, self.nodes, int(self.mode == "modulated"),
        )
        self._cuda["local_rule_arcs"](
            grid, (_ARC_THREADS,), self.clock, self._cuda_indices, self._local_activity, self.node_trace,
            self._gain, self._receipts, self.plastic, self.legacy_rule, self.reward_rate, self.initial_w, self.weight_state,
            self.g, self.eligibility, self.strength, self.mask, self.last_change, self.w, self.matrix,
            self._counts, self._totals, self.actors, self.nodes, 2 * self.pair_count,
        )
        current = self.matrix if self._tensor_transition else self._matvec()
        actions = torch.empty((*self._batch_shape, JOBS), device=obs.device, dtype=torch.long)
        self._cuda["local_rule_finish"](
            grid, (192 if self._tensor_transition else _NODE_THREADS,), current, self.bias, obs,
            self.h, self.activity, self.tried, self.pending, actions, self.clock, self.nodes,
        )
        return actions

    @torch.no_grad()
    def summary(self):
        """Return per-genome diagnostics on the device without synchronizing the life clock."""
        totals = self.edge_count.sum(-1)
        reciprocated = (self.mask[..., 0, :] & self.mask[..., 1, :]).sum((-2, -1)) * 2
        average_edges = self.edge_ticks.float().mean(-1) / self.clock.clamp_min(1)
        return {
            "initial_edges": self.initial_edges.clone(),
            "final_edges": self.edge_count.float().mean(-1),
            "mean_edges": torch.where(self.clock.ne(0), average_edges, self.initial_edges.float()),
            "final_effective_edges": self.strength.sum((-2, -1), dtype=torch.float32).mean(-1),
            "final_reciprocity": reciprocated / totals.clamp_min(1),
            **{key: getattr(self, key).sum(-1) for key in (*_COUNTS[2:], *_TOTALS)},
            "final_gate_abs": self.g.abs().mean((1, 2, 3)),
            "final_weight_delta": (self.w.float() - self.initial_w.float()).abs().mean((1, 2, 3)),
            "final_bias_delta": (self.bias - self.initial_bias.float()).abs().mean((1, 2)),
        }
