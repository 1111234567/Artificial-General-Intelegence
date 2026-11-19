"""Select variable birth graphs for correct, signal-triggered first answers."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import statistics
import sys

import torch

from experiments.undirected import uncapped_answer_feedback as answers
from experiments.undirected import uncapped_plasticity_run as runner
from experiments.undirected import variable_birth_genome as birth_genome
from experiments.undirected.plasticity_roomy_world import make_roomy_schedule
from experiments.undirected.uncapped_roomy_run import protocol as time_protocol


OUT = runner.OUT.with_name("2026-11-16-plasticity-ready-signal-v1")
TIMING = {"min_wait": 1, "max_wait": 8, "finish_window": 16}
DELAYED = {"min_wait": 20, "max_wait": 27, "finish_window": 16}
reference_audit = runner.audit
reference_score_fields = runner.score_fields


def schedule(*, seed, actors, device, visits, timing=TIMING, pad_to=None):
    tape = make_roomy_schedule(seed=seed, actors=actors, device=device, visits=visits, **timing)
    # READY supplies the temporal signal. Channel 9 discloses no schedule.
    tape.public[..., 9].zero_()
    if pad_to is None or pad_to == tape.public.shape[1]:
        return tape
    requests, ticks, *shape = tape.public.shape
    if pad_to < ticks:
        raise ValueError("The matched probe length must cover the whole request.")
    public = torch.cat((tape.public, tape.public.new_zeros(requests, pad_to - ticks, *shape)), dim=1)
    stage = torch.cat((tape.stage, tape.stage.new_full((requests, pad_to - ticks, actors), 3)), dim=1)
    return replace(tape, public=public, stage=stage)


def score_fields(row, max_pairs, penalty):
    result = reference_score_fields(row, max_pairs)
    result["selection_score"] = row["strict"] - penalty * row["wrong_answer_rate"]
    return result


def rank(row):
    return (
        row["selection_score"], row["strict_both_return"], row["strict_return"], row["strict"],
        -row["premature_request_rate"], row.get("sparsity_bonus", -1.0), row["ready_coverage"],
    )


def protocol(args, penalty):
    result = time_protocol(args, TIMING)
    result["timing"]["wait_hint"] = "absent"
    result["selection_score"] = {
        "strict_credit": 1.0, "first_wrong_legal_finish_debit": penalty, "normalization": "all requests",
    }
    result["ordering"] = [
        "selection_score", "strict_both_return", "strict_return", "strict",
        "fewer pre-READY FINISH requests", "reward for fewer average edges", "ready_coverage",
    ]
    result["ready_shift_audit"] = {
        "timings": [TIMING, DELAYED], "matched_request_ticks": DELAYED["max_wait"] + DELAYED["finish_window"] + 2,
        "phase": "held-out diagnosis only",
    }
    source = Path(__file__).resolve().parent
    for name in ("uncapped_ready_signal_run.py", "uncapped_answer_feedback.py"):
        result["sources"][name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    return result


def audit(config, out, generation, checkpoint, publish, *, probes):
    result = reference_audit(config, out, generation, checkpoint, publish)
    path = out / "audits" / f"{generation:04d}-ready-shift.json"
    if path.is_file():
        return result
    genome = checkpoint["genomes"][result["candidate"]]
    tracked = ("strict", "ready_coverage", "first_finish_correct", "wrong_answer_rate",
               "premature_request_rate", "version_accuracy", "symbol_accuracy", "strict_return")
    rows = []
    for seed in result["seeds"]:
        row = {"seed": seed}
        for name, evaluate in probes.items():
            fields = evaluate([genome], seed=seed, actors=config["audit_actors"],
                              device=config["device"], visits=config["visits"])
            row[name] = {metric: float(fields[metric][0].item()) for metric in tracked}
        rows.append(row)
    means = {name: {key: statistics.fmean(row[name][key] for row in rows) for key in tracked}
             for name in probes}
    runner._write(path, {
        "candidate": result["candidate"], "generation": generation, "seeds": result["seeds"],
        "settings": config["ready_shift_audit"], "means": means, "paired": rows,
    })
    print(f"READY shift {generation}: early attempts/request "
          f"{means['matched_regular']['premature_request_rate']:.4f} -> "
          f"{means['delayed_ready']['premature_request_rate']:.4f}", flush=True)
    return result


def main(argv=None):
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--wrong-penalty", type=float, default=.05,
                       help="Cost per request with a wrong first in-window FINISH (default: %(default)s).")
    chosen, rest = extra.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not 0 <= chosen.wrong_penalty <= 1:
        extra.error("Require a finite --wrong-penalty in [0, 1].")
    if any(flag in rest for flag in ("-h", "--help")):
        extra.print_help()
    evaluate = answers.AnswerEvaluation(schedule)
    block_ticks = DELAYED["max_wait"] + DELAYED["finish_window"] + 2
    probes = {
        "matched_regular": answers.AnswerEvaluation(partial(schedule, pad_to=block_ticks)),
        "delayed_ready": answers.AnswerEvaluation(partial(schedule, timing=DELAYED, pad_to=block_ticks)),
    }

    def baseline(**settings):
        result = evaluate.evaluate_baseline(**settings)
        result["selection_score"] = result["strict"] - chosen.wrong_penalty * result["wrong_answer_rate"]
        return result

    overrides = {
        "OUT": OUT, "protocol": partial(protocol, penalty=chosen.wrong_penalty),
        "candidate": birth_genome.candidate, "evaluate_continuous": evaluate, "evaluate_baseline": baseline,
        "score_fields": partial(score_fields, penalty=chosen.wrong_penalty),
        "rank": rank, "audit": partial(audit, probes=probes),
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
