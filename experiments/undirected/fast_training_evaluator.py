"""Optional CUDA execution backends for the sealed local-rule experiment."""

from collections import defaultdict

import torch

from . import local_rule_evaluator as reference
from . import local_rule_world as world
from .evolved_plasticity_world import _receipt_table
from .uncapped_private_symbol_world_cuda import _kernels
from .uncapped_reward_precision import _matmul_precision


class RequestChunks:
    """Replay several complete requests together, retaining all cross-request state."""

    def __init__(self, policy, tape, requests):
        self.policy = policy
        self.requests = requests
        ticks, actors = tape.stage.shape[1:]
        pop = policy.population
        batch = pop * actors
        dev = tape.public.device
        self.frames = tape.public[:requests].clone()
        self.stages = tape.stage[:requests].clone()
        self.targets = tape.target[:requests].clone()
        self.receipts = torch.zeros((pop, actors, 2), dtype=torch.long, device=dev)
        self.flags = torch.zeros((6, pop, actors), dtype=torch.bool, device=dev)
        self.counts = torch.zeros((8, pop, actors), dtype=torch.int32, device=dev)
        self.outcomes = torch.empty((6, requests, pop, actors), dtype=torch.bool, device=dev)
        self.block_trace = torch.empty((requests, ticks, pop, actors), dtype=torch.long, device=dev)
        self.trace = self.block_trace.new_empty((len(tape.raw), ticks, pop, actors))
        self.table = _receipt_table(dev).half()
        self.obs = self.table.new_empty((pop, actors, 56))
        kernels = _kernels(dev)

        def run_block():
            for req in range(requests):
                for tick in range(ticks):
                    kernels["private_packed_observation"](
                        ((batch * 32 + 127) // 128,), (128,), self.frames[req, tick], self.table,
                        self.receipts, self.flags, self.obs, actors, batch, int(tick == 0))
                    action = policy.act(self.obs)
                    kernels["directed_world_action"](
                        ((batch + 127) // 128,), (128,), action, self.stages[req, tick], self.targets[req],
                        self.counts, self.receipts, self.flags, self.outcomes, batch, actors, requests, req,
                        int(tick == 0), int(tick == ticks - 1), *action.stride(), self.block_trace[req, tick])

        run_block()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            run_block()

    def run(self, birth, tape, pairs):
        self.policy.reset(birth)
        self.receipts.zero_()
        self.flags.zero_()
        self.counts.zero_()
        for idx in range(0, len(tape.raw), self.requests):
            block = slice(idx, idx + self.requests)
            self.frames.copy_(tape.public[block])
            self.stages.copy_(tape.stage[block])
            self.targets.copy_(tape.target[block])
            self.graph.replay()
            self.trace[block].copy_(self.block_trace)
        counts = self.counts.sum(-1)
        return {**self.policy.summary(), "idle": counts[0] / counts[1].clamp_min(1),
                **reference.trace_statistics(self.trace, tape, pairs)}


class FastEvaluation(reference.Evaluation):
    def __init__(self, device, split, *, backend, requests=4, tiles=1):
        super().__init__(device, split)
        if backend not in ("chunks", "arcs", "compact", "life") or requests <= 0:
            raise ValueError("Invalid local training execution configuration")
        self.backend, self.requests, self.tiles = backend, requests, tiles

    @torch.no_grad()
    def __call__(self, genomes, *, arm, seed, actors=8, visits=16):
        if (not self.training or self.device.type != "cuda" or arm not in ("local", "modulated")
                or not genomes or visits * 4 % self.requests):
            return super().__call__(genomes, arm=arm, seed=seed, actors=actors, visits=visits)
        key = seed, actors, visits
        if self.schedule_key != key:
            cpu = world.tape_on_cpu(seed, actors, visits, self.training)
            self.tape = reference._convert(cpu, self.device)
            self.pairs = world.partners(cpu.raw, cpu.context).to(self.device)
            self.schedule_key = key
        shape = arm, len(genomes), actors, self.tape.stage.shape[:2], genomes[0]["node_bias"].numel()
        with _matmul_precision(torch.float16, False):
            if shape not in self.replays:
                policy = reference.make_policy(genomes, actors, self.device, arm)
                if self.backend == "life":
                    from .fast_local_life import FastLocalLife
                    replay = FastLocalLife(policy, requests_per_launch=self.requests)
                else:
                    if self.backend == "arcs":
                        from .fast_local_arcs import install
                        install(policy, tiles=self.tiles)
                    elif self.backend == "compact":
                        from .fast_local_arcs_compact import install
                        install(policy)
                    replay = RequestChunks(policy, self.tape, self.requests)
                self.replays[shape] = replay
            metrics = self.replays[shape].run(reference.stack_birth(genomes, self.device, arm), self.tape, self.pairs)
        groups = defaultdict(list)
        for k, v in metrics.items():
            groups[v.dtype].append(k)
        columns = {}
        for keys in groups.values():
            columns.update(zip(keys, torch.stack([metrics[k] for k in keys]).cpu().tolist(), strict=True))
        rows = []
        for i in range(len(genomes)):
            row = {k: v[i] for k, v in columns.items()}
            row["symbol_late_mean"] = (row["symbol_late_a"] + row["symbol_late_b"]) / 2
            rows.append(reference.score(row, len(self.tape.raw) * actors))
        return rows
