"""One continuous graph, with shared or independently directed pair weights.

The recurrent state has no assigned banks, layers, clocks, or memory slots.
An undirected pair has one orthonormal coordinate s: W_uv=W_vu=s/sqrt(2).
The untied control adds one skew coordinate a: W_uv=(s+a)/sqrt(2),
W_vu=(s-a)/sqrt(2).  Its newborn skew coordinates are exactly zero.

This module represents the smooth *potential* graph.  Eligibility has exactly
one owner, the runner's structural operator.  Before a real forward boundary,
that operator must project inactive coordinates to zero and validate them.
The forward does not multiply by a mask: derivatives at a zero inactive pair
are ordinary counterfactual sensitivities of this potential graph, not a
straight-through derivative of a masked computation.  Probes may explicitly
evaluate temporary candidate graphs without changing the real eligibility.

Zero self-couplings are a configuration of this experiment, not a necessary
property of undirected graphs.  Dense matrix multiplication is used even when
many potential coordinates are zero.
"""
from __future__ import annotations

import hashlib
import math
import random

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("undirected", "untied")
READOUTS = ("tanh", "linear")
_INV_SQRT_TWO = 1. / math.sqrt(2.)
GRAPH_SAMPLER = "permuted-ring-regular-double-edge-switch-v1"


def _integer(name, value, *, lower=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return value


def _seed(name, value):
    value = _integer(name, value)
    if value >= 2 ** 63:
        raise ValueError(f"{name} must be less than 2**63")
    return value


def _connected(neighbors):
    reached, pending = {0}, [0]
    while pending:
        node = pending.pop()
        for other in neighbors[node]:
            if other not in reached:
                reached.add(other)
                pending.append(other)
    return len(reached) == len(neighbors)


def connected_regular_mask(width=64, degree=16, seed=0, *, switches_per_edge=20):
    """Return a reproducible connected regular simple graph in pair order.

    A randomly relabeled regular ring starts a fixed-budget switch chain.
    Self-couplings and duplicate edges are rejected; each accepted switch
    preserves every degree.  Degree-two switches additionally reject a cycle
    split; other degrees check connectivity at the end, with at most eight
    deterministic retries from fresh relabeled rings.  This sampler is
    explicitly not claimed to be uniform over connected regular graphs.
    Only a private Python RNG is consumed.
    """
    _integer("width", width, lower=2)
    _integer("degree", degree, lower=1)
    _seed("seed", seed)
    _integer("switches_per_edge", switches_per_edge)
    if degree >= width or (width * degree) % 2 or (degree == 1 and width != 2):
        raise ValueError("a connected regular graph requires degree < width, even width*degree, and feasible connectivity")
    rng = random.Random(seed)
    for _ in range(8):
        labels = list(range(width))
        rng.shuffle(labels)
        edges = set()
        for node in range(width):
            for offset in range(1, degree // 2 + 1):
                edges.add(tuple(sorted((labels[node], labels[(node + offset) % width]))))
        if degree % 2:
            for node in range(width // 2):
                edges.add(tuple(sorted((labels[node], labels[node + width // 2]))))
        ordered = sorted(edges)
        cycle_neighbors = None
        if degree == 2:
            cycle_neighbors = [set() for _ in range(width)]
            for a, b in edges:
                cycle_neighbors[a].add(b)
                cycle_neighbors[b].add(a)
        if len(ordered) > 1:
            for _ in range(switches_per_edge * len(ordered)):
                first, second = rng.sample(range(len(ordered)), 2)
                old_first, old_second = ordered[first], ordered[second]
                a, b = old_first if rng.getrandbits(1) else old_first[::-1]
                c, d = old_second if rng.getrandbits(1) else old_second[::-1]
                if len({a, b, c, d}) != 4:
                    continue
                new_first = tuple(sorted((a, d)))
                new_second = tuple(sorted((c, b)))
                if new_first in edges or new_second in edges:
                    continue
                if cycle_neighbors is not None:
                    for x, y in (old_first, old_second):
                        cycle_neighbors[x].remove(y)
                        cycle_neighbors[y].remove(x)
                    for x, y in (new_first, new_second):
                        cycle_neighbors[x].add(y)
                        cycle_neighbors[y].add(x)
                    if not _connected(cycle_neighbors):
                        for x, y in (new_first, new_second):
                            cycle_neighbors[x].remove(y)
                            cycle_neighbors[y].remove(x)
                        for x, y in (old_first, old_second):
                            cycle_neighbors[x].add(y)
                            cycle_neighbors[y].add(x)
                        continue
                edges.remove(old_first)
                edges.remove(old_second)
                edges.add(new_first)
                edges.add(new_second)
                ordered[first], ordered[second] = new_first, new_second
        neighbors = [set() for _ in range(width)]
        for a, b in edges:
            neighbors[a].add(b)
            neighbors[b].add(a)
        if _connected(neighbors):
            mask = torch.zeros(width * (width - 1) // 2, dtype=torch.bool)
            coordinates = [a * (2 * width - a - 1) // 2 + b - a - 1 for a, b in sorted(edges)]
            mask[coordinates] = True
            return mask
    raise RuntimeError("connected regular graph sampler exhausted its deterministic retries")


def _tensor_digest(value):
    value = value.detach().cpu().contiguous()
    prefix = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(prefix + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def contextual_input_scale(n_jobs=2):
    """Public observation units for 28-feature contextual jobs, without labels."""
    _integer("n_jobs", n_jobs, lower=1)
    result = [1.] * (28 * n_jobs)
    for job in range(n_jobs):
        result[job * 28 + 9] = 1. / 16.
        result[job * 28 + 10] = 1. / 2.5
    return result


def _input_scale(value, input_size):
    if value is None:
        value = [1.] * input_size
    try:
        result = torch.as_tensor(value, dtype=torch.float64).detach().cpu().clone()
    except (TypeError, ValueError) as error:
        raise ValueError("input_scale must contain one positive finite scale per input") from error
    if result.shape != (input_size,) or not bool(torch.isfinite(result).all()) or bool(result.le(0).any()):
        raise ValueError("input_scale must contain one positive finite scale per input")
    return result


def _ports(input_size, output_size, width, input_ports, output_ports):
    inputs = tuple(range(input_size)) if input_ports is None else tuple(input_ports)
    outputs = tuple(range(input_size, input_size + output_size)) if output_ports is None else tuple(output_ports)
    if len(inputs) != input_size or len(outputs) != output_size:
        raise ValueError("fixed port counts must match the public input and output dimensions")
    for node in (*inputs, *outputs):
        _integer("port node", node)
        if node >= width:
            raise ValueError("port nodes must fit inside the recurrent graph")
    if len(set((*inputs, *outputs))) != len(inputs) + len(outputs):
        raise ValueError("fixed input and output ports must be distinct")
    return inputs, outputs


class UndirectedModel(nn.Module):
    """Undirected internal flow with trainable dense input/readout interfaces.

    The interfaces are intentionally distinguished from a fully undirected
    graph with fixed external ports.  ``readout='linear'`` reads raw state
    through the trainable head.  Initialization preserves the global RNG.
    """

    def __init__(self, input_size=112, n_jobs=4, n_actions=11, width=64,
                 max_step=.25, mode="undirected", input_scale=None,
                 readout="tanh", initialization_seed=None):
        super().__init__()
        self._initialize_graph(input_size, n_jobs, n_actions, width, max_step, mode)
        if readout not in READOUTS:
            raise ValueError("unknown readout activation")
        if initialization_seed is not None:
            _seed("initialization_seed", initialization_seed)
        self.readout, self.initialization_seed = readout, initialization_seed
        # A runner may seed/fork the enclosing construction, or select a
        # private seed here.  Neither mode advances the ambient CPU RNG.
        with torch.random.fork_rng(devices=[]):
            if initialization_seed is not None:
                torch.manual_seed(initialization_seed)
            self.encoder = nn.Linear(input_size, width, bias=False)
            self.head = nn.Linear(width, n_jobs * n_actions)
        self.register_buffer("input_scale", _input_scale(input_scale, input_size))

    def _initialize_graph(self, input_size, n_jobs, n_actions, width, max_step, mode):
        for name, value in (("input_size", input_size), ("n_jobs", n_jobs),
                            ("n_actions", n_actions), ("width", width)):
            _integer(name, value, lower=1)
        if isinstance(max_step, bool) or not isinstance(max_step, (int, float)) or not math.isfinite(max_step) or max_step <= 0:
            raise ValueError("max_step must be positive and finite")
        if mode not in MODES:
            raise ValueError("unknown graph parameterization")
        self.input_size, self.n_jobs, self.n_actions = input_size, n_jobs, n_actions
        self.width, self.max_step, self.mode = width, float(max_step), mode
        self.register_buffer("pair_index", torch.triu_indices(width, width, offset=1))
        self.edge_shared = nn.Parameter(torch.zeros(self.pair_count))
        if mode == "untied":
            self.edge_skew = nn.Parameter(torch.zeros(self.pair_count))

    @property
    def pair_count(self):
        return self.width * (self.width - 1) // 2

    def recurrent_matrix(self):
        """Expand independent coordinates; no parameter or diagonal is hidden."""
        upper = lower = self.edge_shared * _INV_SQRT_TWO
        if self.mode == "untied":
            upper = (self.edge_shared + self.edge_skew) * _INV_SQRT_TWO
            lower = (self.edge_shared - self.edge_skew) * _INV_SQRT_TWO
        u, v = self.pair_index.unbind(0)
        indices = torch.cat((u * self.width + v, v * self.width + u))
        values = torch.cat((upper, lower))
        flat = self.edge_shared.new_zeros(self.width * self.width)
        return flat.scatter(0, indices, values).reshape(self.width, self.width)

    def project_matrix_gradient(self, gradient):
        """Pull a matrix sensitivity back to the actual orthonormal coordinates."""
        if gradient.ndim < 2 or gradient.shape[-2:] != (self.width, self.width):
            raise ValueError("invalid recurrent matrix gradient shape")
        u, v = self.pair_index.unbind(0)
        upper, lower = gradient[..., u, v], gradient[..., v, u]
        result = {"edge_shared": (upper + lower) * _INV_SQRT_TWO}
        if self.mode == "untied":
            result["edge_skew"] = (upper - lower) * _INV_SQRT_TWO
        return result

    def vector_field(self, state):
        if state.ndim < 1 or state.shape[-1] != self.width:
            raise ValueError("invalid hidden-state shape")
        return -state + F.linear(state.tanh(), self.recurrent_matrix())

    def flow(self, state, dt, max_step=None):
        limit = self.max_step if max_step is None else float(max_step)
        elapsed = float(dt)
        if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(limit) or limit <= 0:
            raise ValueError("invalid time increment or solver step")
        n = max(1, math.ceil(elapsed / limit))
        delta = elapsed / n
        for _ in range(n):
            first = self.vector_field(state)
            second = self.vector_field(state + delta * first)
            state = state + .5 * delta * (first + second)
        return state

    def step(self, obs, state, dt, max_step=None):
        if obs.ndim < 1 or obs.shape[-1] != self.input_size:
            raise ValueError("invalid observation shape")
        if state.shape != (*obs.shape[:-1], self.width):
            raise ValueError("invalid hidden-state shape")
        state = self.flow(state, dt, max_step) + self.observation_pulse(obs)
        logits = self.read_state(state).reshape(*state.shape[:-1], self.n_jobs, self.n_actions)
        return logits, state

    def normalized_observation(self, obs):
        """Apply only the fixed public feature units, using no world labels."""
        return obs.to(dtype=self.edge_shared.dtype) * self.input_scale.to(dtype=self.edge_shared.dtype)

    def observation_pulse(self, obs):
        return self.encoder(self.normalized_observation(obs))

    def read_state(self, state):
        return self.head(state if self.readout == "linear" else state.tanh())

    def dynamic_parameter_names(self):
        edges = ("edge_shared", "edge_skew") if self.mode == "untied" else ("edge_shared",)
        return (*edges, "encoder.weight")

    @torch.no_grad()
    def initialize_port_matched_interfaces_(self, pulse_gain=.25, input_ports=None,
                                            output_ports=None, input_scale=None):
        """Give trainable interfaces the exact fixed-port newborn function.

        This does not freeze either interface.  Paired graph/weight seeds,
        zero skew coordinates, equal hidden states and these initial selector
        values make core and port models execute the same newborn function.
        """
        if isinstance(pulse_gain, bool) or not isinstance(pulse_gain, (int, float)) or not math.isfinite(pulse_gain) or pulse_gain <= 0:
            raise ValueError("pulse_gain must be positive and finite")
        inputs, outputs = _ports(self.input_size, self.n_jobs * self.n_actions,
                                  self.width, input_ports, output_ports)
        if input_scale is None:
            if self.input_size != 28 * self.n_jobs:
                raise ValueError("non-contextual toy dimensions require an explicit input_scale")
            input_scale = contextual_input_scale(self.n_jobs)
        self.input_scale.copy_(_input_scale(input_scale, self.input_size).to(self.input_scale))
        self.encoder.weight.zero_()
        self.head.weight.zero_()
        self.head.bias.zero_()
        columns = torch.arange(self.input_size, device=self.encoder.weight.device)
        self.encoder.weight[torch.tensor(inputs, device=columns.device), columns] = float(pulse_gain)
        rows = torch.arange(self.n_jobs * self.n_actions, device=self.head.weight.device)
        self.head.weight[rows, torch.tensor(outputs, device=rows.device)] = 1.
        self.readout = "linear"
        return {"interface_birth": "fixed_port_matched_trainable_selectors", "pulse_gain": float(pulse_gain),
                "input_ports": list(inputs), "output_ports": list(outputs),
                "input_scale": self.input_scale.detach().cpu().tolist(), "readout": self.readout}

    def forward(self, obs, state, dt):
        return self.step(obs, state, dt)

    def initial_state(self, batch_size, device=None):
        _integer("batch_size", batch_size, lower=1)
        return self.edge_shared.new_zeros(batch_size, self.width,
                                          device=device or self.edge_shared.device)

    def _eligibility(self, mask):
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (self.pair_count,):
            raise ValueError("eligibility must be one boolean per unordered pair")
        return mask.to(self.edge_shared.device)

    @torch.no_grad()
    def project_inactive_(self, mask):
        """Operator-owned projection at real boundaries; no mask is retained."""
        inactive = ~self._eligibility(mask)
        self.edge_shared.masked_fill_(inactive, 0.)
        if self.mode == "untied":
            self.edge_skew.masked_fill_(inactive, 0.)
        return self

    @torch.no_grad()
    def validate_eligibility(self, mask):
        mask = self._eligibility(mask)
        parameters = [self.edge_shared]
        if self.mode == "untied":
            parameters.append(self.edge_skew)
        if any(not bool(torch.isfinite(value).all()) or bool(value[~mask].ne(0).any()) for value in parameters):
            raise ValueError("real graph boundary contains nonfinite or inactive nonzero coordinates")

    @torch.no_grad()
    def initialize_birth(self, mask_seed=0, weight_seed=0, degree=16, gain=1.2):
        """Initialize one canonical shared-coordinate birth, independently of mode.

        Encoder/head initialization is unchanged.  Pair weights use a private
        CPU generator and one global Frobenius normalization; per-row scaling
        would break symmetry.  Both modes draw exactly the same shared vector.
        """
        _seed("mask_seed", mask_seed)
        _seed("weight_seed", weight_seed)
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise ValueError("birth gain must be positive and finite")
        mask = connected_regular_mask(self.width, degree, mask_seed)
        generator = torch.Generator(device="cpu").manual_seed(weight_seed)
        shared = torch.randn(self.pair_count, generator=generator, dtype=torch.float64)
        shared.masked_fill_(~mask, 0.)
        shared.mul_(float(gain) * math.sqrt(self.width) / float(shared.norm()))
        self.edge_shared.copy_(shared.to(self.edge_shared))
        if self.mode == "untied":
            self.edge_skew.zero_()
        matrix = self.recurrent_matrix().cpu().double()
        eigenvalues = torch.linalg.eigvalsh(matrix)
        metadata = {"source": "from_scratch", "graph_sampler": GRAPH_SAMPLER,
            "graph_sampler_uniform": False, "mask_seed": mask_seed, "weight_seed": weight_seed,
            "width": self.width, "degree": degree, "gain": float(gain),
            "initial_pair_count": int(mask.sum()), "initial_oriented_couplings": int(mask.sum()) * 2,
            "potential_pair_count": self.pair_count, "self_couplings": False,
            "initial_mask_hash": _tensor_digest(mask),
            "initial_shared_hash": _tensor_digest(self.edge_shared),
            "initial_recurrent_matrix_hash": _tensor_digest(self.recurrent_matrix()),
            "initial_frobenius_norm": float(matrix.norm()),
            "initial_operator_norm": float(eigenvalues.abs().max()),
            "initial_min_eigenvalue": float(eigenvalues.min()),
            "initial_max_eigenvalue": float(eigenvalues.max()),
            "initial_mean_row_squared_norm": float(matrix.square().sum(1).mean()),
            "coordinate_convention": "orthonormal shared/skew pair coordinates",
            "normalization": "one global shared-coordinate L2 scale gives Frobenius(W)=gain*sqrt(width)",
            "eligibility_owner": "structural operator; forward evaluates the unmasked smooth potential graph"}
        mask = mask.to(self.edge_shared.device)
        self.validate_eligibility(mask)
        return mask, metadata

    def config(self):
        return {**{name: getattr(self, name) for name in (
            "input_size", "n_jobs", "n_actions", "width", "max_step", "mode", "readout", "initialization_seed")},
            "input_scale": self.input_scale.detach().cpu().tolist()}

    def parameter_report(self):
        recurrent = self.edge_shared.numel() * (2 if self.mode == "untied" else 1)
        return {"kind": "learned", "mode": self.mode, "input_size": self.input_size,
            "n_jobs": self.n_jobs, "n_actions": self.n_actions, "state_scalars": self.width,
            "persistent_state_scalars": self.width, "potential_pairs": self.pair_count,
            "recurrent_parameters": recurrent,
            "total_parameters": sum(value.numel() for value in self.parameters()),
            "trainable_parameters": sum(value.numel() for value in self.parameters() if value.requires_grad),
            "bank_count": None, "assigned_node_roles": None, "self_couplings": False,
            "interface": "learned_dense", "graph_scope": "recurrent state graph; input and readout interfaces are trainable maps",
            "input_interface_parameters": self.width * self.input_size,
            "readout_interface_parameters": self.n_jobs * self.n_actions * (self.width + 1),
            "readout": self.readout, "input_scale": self.input_scale.detach().cpu().tolist(),
            "dynamic_parameter_names": list(self.dynamic_parameter_names()),
            "recurrent_mac_per_field": self.width ** 2,
            "field_evaluations_per_heun_substep": 2, "tanh_scalars_per_field": self.width,
            "max_solver_step": self.max_step, "time_constant": 1.,
            "coordinate_convention": "orthonormal shared/skew pair coordinates",
            "topology": "arbitrary unordered pairs; no predefined banks or layers",
            "eligibility_owner": "structural operator; no forward mask",
            "optimizer_note": "Shared gradients sum both matrix directions divided by sqrt(2); Adam is not learning-rate-equivalent to independent directions.",
            "formula": "dh/dt=-h+W tanh(h); W diagonal=0; shared/skew pair expansion"}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        name = prefix + "pair_index"
        if name in state_dict:
            candidate = state_dict[name]
            canonical = torch.triu_indices(self.width, self.width, offset=1)
            if (not isinstance(candidate, torch.Tensor) or candidate.dtype != torch.long
                    or candidate.shape != canonical.shape or not torch.equal(candidate.cpu(), canonical)):
                error_msgs.append(f"{name} must be the canonical zero-diagonal unordered-pair index")
        for buffer_name in ("input_scale",):
            name = prefix + buffer_name
            if name in state_dict:
                candidate, expected = state_dict[name], getattr(self, buffer_name)
                if (not isinstance(candidate, torch.Tensor) or candidate.shape != expected.shape
                        or not torch.equal(candidate.cpu(), expected.detach().cpu())):
                    error_msgs.append(f"{name} must match its fixed model configuration")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class PortGraphModel(UndirectedModel):
    """One graph whose input and output ports are ordinary evolving nodes.

    Fixed external pulses enter distinct input nodes; raw states at output
    nodes are logits.  All nodes, including ports, evolve in the same graph
    and may couple to every other node.  Ports are never clamped or reset.
    There is no trainable encoder, readout, bias, or separate output layer.
    """

    def __init__(self, input_size=56, n_jobs=2, n_actions=11, width=80,
                 max_step=.25, mode="undirected", input_scale=None,
                 pulse_gain=.25, input_ports=None, output_ports=None):
        nn.Module.__init__(self)
        self._initialize_graph(input_size, n_jobs, n_actions, width, max_step, mode)
        if isinstance(pulse_gain, bool) or not isinstance(pulse_gain, (int, float)) or not math.isfinite(pulse_gain) or pulse_gain <= 0:
            raise ValueError("pulse_gain must be positive and finite")
        inputs, outputs = _ports(input_size, n_jobs * n_actions, width, input_ports, output_ports)
        if input_scale is None:
            if input_size != 28 * n_jobs:
                raise ValueError("non-contextual toy dimensions require an explicit input_scale")
            input_scale = contextual_input_scale(n_jobs)
        self.pulse_gain, self.readout = float(pulse_gain), "linear"
        self.register_buffer("input_scale", _input_scale(input_scale, input_size))
        self.register_buffer("input_ports", torch.tensor(inputs, dtype=torch.long))
        self.register_buffer("output_ports", torch.tensor(outputs, dtype=torch.long))

    def observation_pulse(self, obs):
        values = self.normalized_observation(obs) * self.pulse_gain
        indices = self.input_ports.expand(*values.shape[:-1], self.input_size)
        return values.new_zeros(*values.shape[:-1], self.width).scatter(-1, indices, values)

    def read_state(self, state):
        return state.index_select(-1, self.output_ports)

    def dynamic_parameter_names(self):
        return ("edge_shared", "edge_skew") if self.mode == "untied" else ("edge_shared",)

    def initialize_port_matched_interfaces_(self, **kwargs):
        raise ValueError("fixed ports already define this model's interfaces")

    def config(self):
        return {**{name: getattr(self, name) for name in (
            "input_size", "n_jobs", "n_actions", "width", "max_step", "mode", "pulse_gain")},
            "input_scale": self.input_scale.detach().cpu().tolist(),
            "input_ports": self.input_ports.detach().cpu().tolist(),
            "output_ports": self.output_ports.detach().cpu().tolist()}

    def parameter_report(self):
        report = super().parameter_report()
        inputs, outputs = self.input_ports.detach().cpu().tolist(), self.output_ports.detach().cpu().tolist()
        report.update(interface="fixed_graph_ports", graph_scope="all state nodes, including input and output ports",
            input_interface_parameters=0, readout_interface_parameters=0,
            pulse_gain=self.pulse_gain, input_ports=inputs, output_ports=outputs,
            assigned_node_roles={"external_input_ports": inputs, "external_output_ports": outputs,
                                 "internal_functional_roles": None},
            non_port_state_nodes=self.width - self.input_size - self.n_jobs * self.n_actions,
            port_state_rule="ordinary evolving recurrent nodes; no clamping or resets; raw output-node states are logits")
        return report

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for buffer_name in ("input_ports", "output_ports"):
            name = prefix + buffer_name
            if name in state_dict:
                candidate, expected = state_dict[name], getattr(self, buffer_name)
                if (not isinstance(candidate, torch.Tensor) or candidate.dtype != torch.long
                        or candidate.shape != expected.shape or not torch.equal(candidate.cpu(), expected.cpu())):
                    error_msgs.append(f"{name} must match its fixed model configuration")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
