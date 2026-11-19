"""Keep each life resident on one CUDA block between request boundaries."""

import ctypes
from functools import lru_cache
import operator
from pathlib import Path
import sys

import torch

from . import local_rule_evaluator
from . import local_rule_policy as local
from ._cuda_nvrtc import compile_kernels
from .evolved_plasticity_world import _receipt_table


@lru_cache(maxsize=None)
def _kernel(device):
    directory = Path(__file__).parent
    constants = {
        **local._CONSTANTS,
        **{field.upper(): i for i, field in enumerate(local.PLASTIC_FIELDS)},
        **{f"COUNT_{name.upper()}": i for i, name in enumerate(local._COUNTS)},
        **{f"TOTAL_{name.upper()}": i for i, name in enumerate(local._TOTALS)},
        "PLASTIC_SIZE": local.PLASTIC_SIZE,
        "NODE_THREADS": local._NODE_THREADS,
        "ARC_THREADS": local._ARC_THREADS,
        "COUNT_FIELDS": len(local._COUNTS),
        "TOTAL_FIELDS": len(local._TOTALS),
        "TENSOR_TRANSITION": 1,
    }
    defines = []
    for name, value in constants.items():
        literal = f"{value!r}f" if isinstance(value, float) else str(value)
        defines.append(f"#define {name} {literal}")
    tensor = (directory / "uncapped_directed_matvec_tensor.cu").read_text(encoding="utf-8")
    helpers = (directory / "local_rule_policy.cu").read_text(encoding="utf-8")
    world = (directory / "uncapped_private_symbol_world_device.cu").read_text(encoding="utf-8")
    life = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    source = "\n".join((*defines, tensor[:tensor.index('extern "C"')],
                        helpers[:helpers.index('extern "C"')], world, life))
    return compile_kernels(source, "fast_local_life", device=device)["fast_local_life"]


class FastLocalLife:
    """Run a complete tape in contiguous request spans with sealed state layouts.

    ``trace`` contains every primary action in ``[request,tick,population,actor]``
    order. ``capture_world`` also retains the spare actions and six per-request
    outcome flags. Final world receipts, flags and counts are always available.
    Each call resets and updates the supplied policy in place.
    """

    def __init__(self, policy, requests_per_launch=8, *, capture_world=False):
        if not isinstance(policy, local.LocalRulePolicy) or not policy._tensor_transition:
            raise ValueError("fast local life requires a 96-node CUDA Tensor Core local rule policy")
        self.requests_per_launch = operator.index(requests_per_launch)
        if self.requests_per_launch < 1:
            raise ValueError("requests_per_launch must be positive")
        self.policy = policy
        self.capture_world = capture_world
        self.kernel = _kernel(policy.device)
        self.table = _receipt_table(policy.device).half().contiguous()
        batch = policy._batch_shape
        self.world_counts = torch.empty((8, *batch), device=policy.device, dtype=torch.int32)
        self.world_flags = torch.empty((6, *batch), device=policy.device, dtype=torch.bool)
        self.world_receipts = torch.empty((*batch, 2), device=policy.device, dtype=torch.long)

    @torch.no_grad()
    def run(self, birth, tape, pairs):
        p = self.policy
        requests, ticks, actors = tape.stage.shape
        if requests < 1 or ticks < 1 or requests * ticks >= 1 << 31 or actors != p.actors:
            raise ValueError("expected a nonempty tape with a valid tick count and actor count")
        for tensor, shape, dtype in (
            (tape.public, (requests, ticks, actors, local.JOBS, local.FEATURES_PER_SLOT), torch.float16),
            (tape.stage, (requests, ticks, actors), torch.uint8),
            (tape.target, (requests, actors), torch.long),
            (pairs, (requests, actors), torch.long),
        ):
            if (tensor.shape != shape or tensor.dtype != dtype or tensor.device != p.device
                    or not tensor.is_contiguous()):
                raise ValueError(f"expected contiguous tape data with shape {shape}, dtype {dtype} on {p.device}")

        with torch.cuda.device(p.device):
            p.reset(birth)
            self.world_counts.zero_()
            self.world_flags.zero_()
            self.world_receipts.zero_()
            self.trace = torch.empty((requests, ticks, *p._batch_shape), device=p.device, dtype=torch.long)
            self.spare_trace = torch.empty_like(self.trace) if self.capture_world else None
            self.outcomes = (torch.empty((6, requests, *p._batch_shape), device=p.device, dtype=torch.bool)
                             if self.capture_world else None)
            spare = ctypes.c_void_p() if self.spare_trace is None else self.spare_trace
            outcomes = ctypes.c_void_p() if self.outcomes is None else self.outcomes
            for start in range(0, requests, self.requests_per_launch):
                stop = min(requests, start + self.requests_per_launch)
                self.kernel(
                    (p._batch,), (local._ARC_THREADS,), tape.public, tape.stage, tape.target, self.table,
                    self.trace, spare, outcomes, self.world_counts, self.world_flags, self.world_receipts, p.clock,
                    p._cuda_indices, p.plastic, p.legacy_rule, p.reward_rate, p.initial_bias, p.initial_w,
                    p.h, p.activity, p.bias, p.node_trace, p._local_activity, p._gain, p._receipts, p.tried, p.pending,
                    p.weight_state, p.g, p.eligibility, p.strength, p.mask, p.last_change, p.w, p.matrix, p._counts,
                    p._totals, actors, requests, ticks, start, stop, int(p.mode == "modulated"),
                )
            counts = self.world_counts.sum(-1)
            return {**p.summary(), "idle": counts[0] / counts[1].clamp_min(1),
                    **local_rule_evaluator.trace_statistics(self.trace, tape, pairs)}


class Evaluation(local_rule_evaluator.Evaluation):
    """Evaluator with the original interface and the resident CUDA path for both local arms."""

    def __init__(self, device, split, *, requests_per_launch=8):
        super().__init__(device, split)
        self.requests_per_launch = requests_per_launch

    @torch.no_grad()
    def __call__(self, genomes, *, arm, seed, actors=8, visits=16):
        if (sys.platform != "win32" or self.device.type != "cuda" or arm not in ("local", "modulated") or not genomes
                or genomes[0]["node_bias"].numel() != 96
                or torch.cuda.get_device_capability(self.device)[0] < 8):
            return super().__call__(genomes, arm=arm, seed=seed, actors=actors, visits=visits)

        with torch.cuda.device(self.device):
            key = seed, actors, visits
            if self.schedule_key != key:
                cpu = local_rule_evaluator.world.tape_on_cpu(seed, actors, visits, self.training)
                self.tape = local_rule_evaluator._convert(cpu, self.device)
                self.pairs = local_rule_evaluator.world.partners(cpu.raw, cpu.context).to(self.device)
                self.schedule_key = key
            shape = arm, len(genomes), actors, self.tape.stage.size(1), 96
            if shape not in self.replays:
                policy = local_rule_evaluator.make_policy(genomes, actors, self.device, arm)
                self.replays[shape] = FastLocalLife(policy, self.requests_per_launch)
            birth = local_rule_evaluator.stack_birth(genomes, self.device, arm)
            metrics = self.replays[shape].run(birth, self.tape, self.pairs)
        columns = {key: value.detach().cpu().tolist() for key, value in metrics.items()}
        rows = []
        for i in range(len(genomes)):
            row = {key: values[i] for key, values in columns.items()}
            row["symbol_late_mean"] = (row["symbol_late_a"] + row["symbol_late_b"]) / 2
            rows.append(local_rule_evaluator.score(row, self.tape.raw.numel()))
        return rows
