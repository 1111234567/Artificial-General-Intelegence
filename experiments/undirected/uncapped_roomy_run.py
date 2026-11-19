"""Evolve variable birth graphs with a configurable per-request decision budget."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

import torch

from experiments.undirected import plasticity_roomy_world as world
from experiments.undirected import uncapped_bf16_backend as backend
from experiments.undirected import uncapped_birthmask_run as birthmask
from experiments.undirected import uncapped_plasticity_run as runner
from experiments.undirected import variable_birth_genome as birth_genome
from experiments.undirected.evolved_plasticity_world import N_SYMBOLS, _receipt_table
from experiments.undirected.plasticity_continuous_world import rollout_continuous
from experiments.undirected.uncapped_plastic_graph import UncappedGraphBatch


OUT = runner.OUT.with_name("2026-11-16-plasticity-roomy-v1")


class EvaluateRoomy:
    """Reuse the native evaluator and keep per-shape captures within one timing run."""
    def __init__(self, timing):
        self.schedule = partial(world.make_roomy_schedule, **timing)
        self.captures = {}

    @torch.no_grad()
    def __call__(self, genomes, *, seed, actors, device, visits):
        if device == "cpu":
            tape = self.schedule(seed=seed, actors=actors, device=device, visits=visits)
            graph = UncappedGraphBatch(genomes, actors=actors, device=device, seed=seed)
            metrics = rollout_continuous(graph, tape, len(genomes))
            return {**graph.summary(), **metrics}

        device = torch.device("cuda", torch.cuda.current_device())
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("The roomy CUDA evaluator requires native BF16 support")
        nodes, birth = backend._stack_genomes(genomes)
        tape = self.schedule(seed=seed, actors=actors, device="cpu", visits=visits)
        tape = replace(tape, public=tape.public.bfloat16())
        if len(genomes) < 32:
            graph = backend._BF16Policy({key: value.to(device) for key, value in birth.items()}, actors, device)
            return backend._rollout(graph, backend._device_tape(tape, device), _receipt_table(device).bfloat16())

        key = (device, len(genomes), nodes, tape.public.shape)
        captured = self.captures.get(key)
        if captured is None:
            captured = backend._BF16Capture(birth, tape, actors, device)
            self.captures[key] = captured
        return captured.evaluate(birth, tape, torch.cuda.current_stream(device))


def protocol(args, timing):
    result = birthmask.protocol(args)
    ticks_per_request = timing["max_wait"] + timing["finish_window"] + 2
    result["timing"] = {
        **timing, "ticks_per_request": ticks_per_request,
        "life_ticks": ticks_per_request * args.visits * N_SYMBOLS,
        "wait_hint": "relative value 1–3",
    }
    source = Path(__file__).resolve().parent
    for name in ("uncapped_roomy_run.py", "plasticity_roomy_world.py"):
        result["sources"][name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    return result


def main(argv=None):
    timing_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    timing_parser.add_argument("--min-wait", type=int, default=1, help="Earliest window start after NEW (default: %(default)s).")
    timing_parser.add_argument("--max-wait", type=int, default=3, help="Latest window start after NEW (default: %(default)s).")
    timing_parser.add_argument("--finish-window", type=int, default=8, help="Number of legal submission ticks (default: %(default)s).")
    chosen, rest = timing_parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if min(chosen.min_wait, chosen.finish_window) < 1 or chosen.max_wait < chosen.min_wait:
        timing_parser.error("Require 1 <= min-wait <= max-wait and finish-window >= 1.")
    if any(flag in rest for flag in ("-h", "--help")):
        timing_parser.print_help()
    timing = vars(chosen)
    overrides = {
        "OUT": OUT, "protocol": partial(protocol, timing=timing), "candidate": birth_genome.candidate,
        "evaluate_continuous": EvaluateRoomy(timing),
        "evaluate_baseline": partial(world.evaluate_baseline, **timing),
    }
    originals = {name: getattr(runner, name) for name in overrides}
    prior_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        for name, value in overrides.items():
            setattr(runner, name, value)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        runner.main(rest)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prior_reduction
        for name, value in originals.items():
            setattr(runner, name, value)


if __name__ == "__main__":
    main()
