"""Compare per-tick FP16 state storage to the sealed local-rule CUDA rollout."""

from argparse import ArgumentParser
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import local_rule_policy as rule
from . import local_rule_world as world
from . import uncapped_private_symbol_world as private
from .evolved_plasticity import ACTIONS, INPUTS, JOBS
from .evolved_plasticity_world import _receipt_table
from .local_rule_evaluator import _convert, score, stack_birth, trace_statistics


STORAGE_EDITS = {
    "weight_state": (
        ("float* __restrict__ weight,", "half_bits* __restrict__ weight,", 1),
        ("live ? weight[i] :", "live ? from_half(weight[i]) :", 1),
        ("weight[i] = new_weight;", "weight[i] = to_half(new_weight);", 1),
    ),
    "g": (
        ("float* __restrict__ gate,", "half_bits* __restrict__ gate,", 1),
        ("previous = gate[i];", "previous = from_half(gate[i]);", 1),
        ("gate[i] = next_gate;", "gate[i] = to_half(next_gate);", 1),
    ),
    "bias": (
        ("float* __restrict__ bias,", "half_bits* __restrict__ bias,", 2),
        ("old = bias[i];", "old = from_half(bias[i]);", 1),
        ("bias[i] = updated;", "bias[i] = to_half(updated);", 1),
        ("from_half(to_half(bias[i]))", "from_half(bias[i])", 1),
    ),
    "node_trace": (
        ("float* __restrict__ node_trace,", "half_bits* __restrict__ node_trace,", 2),
        ("trace_old = node_trace[i];", "trace_old = from_half(node_trace[i]);", 1),
        ("node_trace[i] = trace;", "node_trace[i] = to_half(trace);", 1),
        ("node_trace[n + src]", "from_half(node_trace[n + src])", 1),
        ("node_trace[n + dest]", "from_half(node_trace[n + dest])", 1),
    ),
}
TRACKED = (*STORAGE_EDITS, "w", "mask", "last_change", "strength", "h", "pending", "tried")
VARIANTS = {"weight_only": ("weight_state",), "all_persistent": tuple(STORAGE_EDITS)}
SCORES = (
    "selection_score", "strict", "completion", "first_finish_correct", "symbol_late_a",
    "symbol_late_b", "pairs_both_strict", "eligible_pairs", "post_answer_actions", "idle",
    "births", "deaths", "final_edges", "weight_updates", "bias_updates", "changed_writes",
)


@lru_cache(maxsize=None)
def storage_kernels(device, use_tensor, fields):
    from . import _cuda_nvrtc

    compile_kernels = _cuda_nvrtc.compile_kernels

    def with_half_storage(source, names, **kwargs):
        for field in fields:
            for before, after, count in STORAGE_EDITS[field]:
                if source.count(before) != count:
                    raise ValueError(f"CUDA source changed near {before!r}")
                source = source.replace(before, after)
        return compile_kernels(source, names, **kwargs)

    with patch.object(_cuda_nvrtc, "compile_kernels", with_half_storage):
        return rule._cuda_kernels.__wrapped__(device, use_tensor)


class ProbeReplay:
    """The evaluator's request graph, retaining both action ports for comparison."""

    def __init__(self, birth, mode, tape, fields=()):
        device = tape.public.device
        population = len(birth["node_bias"])
        _, ticks, actors = tape.stage.shape
        batch = population * actors
        self.policy = rule.LocalRulePolicy(birth, actors, device, mode, backend="cuda")
        if fields:
            self.policy._cuda = storage_kernels(device, self.policy._tensor_transition, fields)
            for field in fields:
                setattr(self.policy, field, getattr(self.policy, field).half())

        self.frame = torch.empty_like(tape.public[0])
        self.stage = torch.empty_like(tape.stage[0])
        self.target = torch.empty_like(tape.target[0])
        self.receipts = torch.zeros((population, actors, JOBS), dtype=torch.long, device=device)
        self.flags = torch.zeros((6, population, actors), dtype=torch.bool, device=device)
        self.counts = torch.zeros((8, population, actors), dtype=torch.int32, device=device)
        self.outcomes = torch.empty((6, 1, population, actors), dtype=torch.bool, device=device)
        self.table = _receipt_table(device).half()
        self.obs = torch.empty(self.policy.input_shape, dtype=torch.float16, device=device)
        self.actions = torch.empty((ticks, population, actors, JOBS), dtype=torch.long, device=device)
        self.ports = torch.empty((*self.actions.shape, ACTIONS), dtype=torch.float16, device=device)
        self.kernels = private._kernels(device)
        self.ticks, self.actors, self.batch = ticks, actors, batch
        self.load_request(tape, 0)
        self.request()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.request()

    def load_request(self, tape, request):
        self.frame.copy_(tape.public[request])
        self.stage.copy_(tape.stage[request])
        self.target.copy_(tape.target[request])

    def reset(self, birth):
        self.policy.reset(birth)
        self.receipts.zero_()
        self.flags.zero_()
        self.counts.zero_()

    def tick(self, tick):
        batch = self.batch
        self.kernels["directed_world_observation"](
            ((batch * INPUTS + 127) // 128,), (128,), self.frame[tick], self.table,
            self.receipts, self.flags, self.obs, self.actors, batch, int(tick == 0),
        )
        action = self.policy.act(self.obs)
        self.actions[tick].copy_(action)
        self.ports[tick].copy_(self.policy.h[..., INPUTS:INPUTS + JOBS * ACTIONS].unflatten(-1, (JOBS, ACTIONS)))
        self.kernels["directed_world_action"](
            ((batch + 127) // 128,), (128,), action, self.stage[tick], self.target,
            self.counts, self.receipts, self.flags, self.outcomes, batch, self.actors, 1, 0,
            int(tick == 0), int(tick == self.ticks - 1), *action.stride(),
        )

    def request(self):
        for tick in range(self.ticks):
            self.tick(tick)

    def run(self, birth, tape, pairs):
        self.reset(birth)
        actions = self.actions.new_empty((len(tape.raw), *self.actions.shape))
        ports = self.ports.new_empty((len(tape.raw), *self.ports.shape))
        for request in range(len(tape.raw)):
            self.load_request(tape, request)
            self.graph.replay()
            actions[request].copy_(self.actions)
            ports[request].copy_(self.ports)
        counts = self.counts.sum(-1)
        metrics = {**self.policy.summary(), **trace_statistics(actions[..., 0], tape, pairs),
                   "idle": counts[0] / counts[1].clamp_min(1)}
        columns = {field: values.detach().cpu().tolist() for field, values in metrics.items()}
        rows = [score({field: values[i] for field, values in columns.items()}, tape.stage.shape[0] * self.actors)
                for i in range(self.policy.population)]
        return actions.cpu(), ports.cpu(), [{field: row[field] for field in SCORES} for row in rows]


def state_differences(base, variant, birth, tape):
    """Find each first state difference during the first request with normal feedback."""
    for replay in (base, variant):
        replay.reset(birth)
        replay.load_request(tape, 0)
    found = {}
    previous = {field: getattr(base.policy, field).cpu() for field in TRACKED}
    for tick in range(base.ticks):
        for replay in (base, variant):
            replay.tick(tick)
        for field in TRACKED:
            if field in found:
                continue
            a = getattr(base.policy, field).cpu()
            b = getattr(variant.policy, field).cpu().to(a.dtype)
            indices = a.ne(b).nonzero()
            if indices.numel():
                index = tuple(indices[0].tolist())
                item = {"request": 0, "tick": tick, "changed": len(indices), "elements": a.numel(),
                        "index": index, "previous_shared": previous[field][index].item(),
                        "baseline": a[index].item(), "variant": b[index].item()}
                if field in STORAGE_EDITS:
                    item["baseline_rounded_to_half"] = a[index].half().float().item()
                found[field] = item
            else:
                previous[field] = a
    return found


def action_difference(base_actions, variant_actions, base_ports, variant_ports, ids, tape):
    changed = base_actions.ne(variant_actions)
    first = None
    locations = changed.nonzero()
    if locations.numel():
        request, tick, genome, actor, port = locations[0].tolist()
        first = {"request": request, "tick": tick, "life_tick": request * tape.stage.size(1) + tick,
                 "genome": ids[genome], "actor": actor, "port": port,
                 "baseline_action": base_actions[request, tick, genome, actor, port].item(),
                 "variant_action": variant_actions[request, tick, genome, actor, port].item(),
                 "stage": tape.stage[request, tick, actor].item(), "target": tape.target[request, actor].item(),
                 "baseline_action_ports": base_ports[request, tick, genome, actor, port].tolist(),
                 "variant_action_ports": variant_ports[request, tick, genome, actor, port].tolist()}
    return {"first": first, "primary_by_genome": changed[..., 0].sum((0, 1, 3)).tolist(),
            "spare_by_genome": changed[..., 1].sum((0, 1, 3)).tolist(),
            "sha256": hashlib.sha256(variant_actions.numpy().tobytes()).hexdigest()}


@torch.no_grad()
def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", type=Path, default=Path("data/undirected/2026-11-19-local-rule-v1"))
    parser.add_argument("--generation", type=int, default=64)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    sources = ("local_rule_policy.cu", "local_rule_policy.py", "local_rule_evaluator.py",
               "uncapped_directed_matvec_tensor.cu", "uncapped_private_symbol_world.py")
    evidence = {"schema": 1, "generation": args.generation, "source": str(args.run), "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__, "actors": 8, "visits": 16, "variants": VARIANTS, "arms": {},
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}}
    for mode in ("local", "modulated"):
        folder = args.run / "arms" / mode / "generations" / f"{args.generation:04d}"
        seed = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["development_seed"]
        checkpoint = folder / "elites.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)["champions"]
        ids = [entry["row"]["id"] for entry in saved]
        tape_cpu = world.tape_on_cpu(seed, evidence["actors"], evidence["visits"], True)
        tape = _convert(tape_cpu, device)
        pairs = world.partners(tape_cpu.raw, tape_cpu.context).to(device)
        birth = stack_birth([entry["genome"] for entry in saved], device, mode)
        base = ProbeReplay(birth, mode, tape)
        base_actions, base_ports, base_rows = base.run(birth, tape, pairs)
        archived = [{key: entry["row"]["metrics"][key] for key in SCORES} for entry in saved]
        replayed_fields = [key for key in SCORES if any(a[key] != b[key] for a, b in zip(archived, base_rows))]
        rows = {"seed": seed, "requests_per_actor": len(tape.raw), "ticks_per_request": tape.stage.size(1),
                "tensor_transition": base.policy._tensor_transition, "genomes": ids, "archived": archived,
                "baseline": base_rows, "archive_fields_different": replayed_fields,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "actions_sha256": hashlib.sha256(base_actions.numpy().tobytes()).hexdigest(), "variants": {}}
        for name, fields in VARIANTS.items():
            variant = ProbeReplay(birth, mode, tape, fields)
            actions, ports, metrics = variant.run(birth, tape, pairs)
            rows["variants"][name] = {"actions": action_difference(base_actions, actions, base_ports, ports, ids, tape_cpu),
                                      "first_request_state": state_differences(base, variant, birth, tape),
                                      "metrics": metrics}
            print(mode, name, rows["variants"][name]["actions"]["first"], flush=True)
        evidence["arms"][mode] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
