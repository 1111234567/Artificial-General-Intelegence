"""Run the four-island READY experiment with crossover and archived birth genomes."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from pathlib import Path
import sys

import torch

from experiments.undirected import uncapped_answer_feedback as answers
from experiments.undirected import uncapped_archived_founders as founders
from experiments.undirected import uncapped_island_recombination as breeding
from experiments.undirected import uncapped_plasticity_run as runner
from experiments.undirected import uncapped_ready_signal_run as signal
from experiments.undirected.evolved_plasticity_world import N_SYMBOLS


OUT = runner.OUT.with_name("2026-11-16-plasticity-island-crossover-v1")
ARCHIVE = runner.OUT
TIMING = {"min_wait": 8, "max_wait": 12, "finish_window": 4}
DELAYED = {"min_wait": 20, "max_wait": 24, "finish_window": 4}


def block_ticks(timing):
    return timing["max_wait"] + timing["finish_window"] + 2


def main(argv=None):
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--population", type=int, default=4096,
                       help="Candidates per generation (default: %(default)s).")
    extra.add_argument("--islands", type=int, default=4, help="Population groups (default: %(default)s).")
    extra.add_argument("--warm-start", type=int, help="Archived founders (default: population / 16).")
    extra.add_argument("--archive", type=Path, default=ARCHIVE, help="Completed source experiment.")
    extra.add_argument("--wrong-penalty", type=float, default=.05,
                       help="Debit per request with a wrong first in-window FINISH (default: %(default)s).")
    args, rest = extra.parse_known_args(sys.argv[1:] if argv is None else argv)
    warm_count = args.population // 16 if args.warm_start is None else args.warm_start
    if not 0 <= args.wrong_penalty <= 1 or not 0 <= warm_count <= args.population:
        extra.error("Require 0 <= wrong-penalty <= 1 and 0 <= warm-start <= population.")
    if any(flag in rest for flag in ("-h", "--help")):
        extra.print_help()

    historical = []
    genomes = {}
    probes = {
        "matched_regular": answers.AnswerEvaluation(partial(
            signal.schedule, timing=TIMING, pad_to=block_ticks(DELAYED))),
        "delayed_ready": answers.AnswerEvaluation(partial(
            signal.schedule, timing=DELAYED, pad_to=block_ticks(DELAYED))),
    }
    evaluate = answers.AnswerEvaluation(partial(signal.schedule, timing=TIMING))

    def protocol(run_args):
        descriptors, bank, record = founders.load_founders(args.archive.resolve(), warm_count, run_args.seed)
        historical[:] = descriptors
        genomes.clear()
        genomes.update(bank)
        result = signal.protocol(run_args, penalty=args.wrong_penalty)
        result["timing"].update(**TIMING, ticks_per_request=block_ticks(TIMING),
                                life_ticks=block_ticks(TIMING) * run_args.visits * N_SYMBOLS)
        result["ready_shift_audit"].update(timings=[TIMING, DELAYED], matched_request_ticks=block_ticks(DELAYED))
        result["warm_start"] = {"count": warm_count, "sources": record, "selected": descriptors}
        result["breeding"] = {
            "per_1024_island_candidates": {
                kind: slots * 1024 // sum(breeding.SLOT_COUNTS.values())
                for kind, slots in breeding.SLOT_COUNTS.items()
            },
            "donor_pair_fraction": breeding.DONOR_FRACTION,
            "rule_donor_fraction_within_each_crossover_kind": breeding.RULE_DONOR_FRACTION,
            "rule_rows": list(breeding.RULE_ROW_NAMES),
            "inheritance": "choose the edge mask and weight together; copy one whole donor rule row on selected crosses",
        }
        source = Path(__file__).resolve().parent
        for name in ("uncapped_island_crossover_run.py", "uncapped_island_recombination.py",
                     "uncapped_archived_founders.py"):
            result["sources"][name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        return result

    def baseline(**settings):
        result = evaluate.evaluate_baseline(**settings)
        result["selection_score"] = result["strict"] - args.wrong_penalty * result["wrong_answer_rate"]
        return result

    overrides = {
        "OUT": OUT, "protocol": protocol,
        "plan": partial(breeding.plan, warm_founders=historical),
        "candidate": partial(breeding.candidate, warm_genomes=genomes),
        "evaluate_continuous": evaluate, "evaluate_baseline": baseline,
        "score_fields": partial(signal.score_fields, penalty=args.wrong_penalty),
        "rank": signal.rank, "audit": partial(signal.audit, probes=probes),
    }
    originals = {name: getattr(runner, name) for name in overrides}
    prior_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        for name, value in overrides.items():
            setattr(runner, name, value)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        runner.main(["--population", str(args.population), "--islands", str(args.islands), *rest])
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prior_reduction
        for name, value in originals.items():
            setattr(runner, name, value)


if __name__ == "__main__":
    main()
