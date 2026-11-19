"""Run the continuous experiment with an evolvable birth adjacency mask."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from experiments.undirected import uncapped_plasticity_run as runner
from experiments.undirected import variable_birth_genome as birth_genome
from experiments.undirected.uncapped_bf16_run import evaluate as evaluate_bf16


OUT = runner.OUT.with_name("2026-11-16-plasticity-variable-birth-v1")
reference_protocol = runner.protocol
evaluate_fp32 = runner.evaluate_continuous


def protocol(args):
    config = reference_protocol(args)
    config.pop("birth_edges")
    config["schema"] = 2
    config["birth_graph"] = {
        "representation": "one heritable mask bit for every unordered node pair",
        "founder_prior_mean_degree": birth_genome.FOUNDER_MEAN_DEGREE,
        "founder_prior_expected_edges": config["nodes"] * birth_genome.FOUNDER_MEAN_DEGREE / 2,
        "degree_or_edge_count_constraint": None,
    }
    config["precision"] = "bf16" if args.device == "cuda" else "float32 highest"
    source = Path(__file__).resolve().parent
    for name in ("uncapped_birthmask_run.py", "variable_birth_genome.py",
                 "uncapped_bf16_run.py", "uncapped_bf16_backend.py", "uncapped_cuda_backend.py"):
        config["sources"][name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    return config


def evaluate(genomes, *, device, **settings):
    scorer = evaluate_bf16 if device == "cuda" else evaluate_fp32
    return scorer(genomes, device=device, **settings)


def main(argv=None):
    # Reuse the continuous runner's selection and checkpoint code. The separate
    # launcher leaves the source hashes and semantics of archived runs intact.
    overrides = {
        "OUT": OUT, "protocol": protocol, "candidate": birth_genome.candidate,
        "evaluate_continuous": evaluate,
    }
    originals = {name: getattr(runner, name) for name in overrides}
    prior_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        for name, value in overrides.items():
            setattr(runner, name, value)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        runner.main(argv)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prior_reduction
        for name, value in originals.items():
            setattr(runner, name, value)


if __name__ == "__main__":
    main()
