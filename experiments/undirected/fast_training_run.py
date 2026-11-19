"""Launch/resume the local-rule protocol using a separately recorded execution backend."""

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

from experiments.seed_evolution.large_runner import _write
from . import fast_training_host, local_rule_evaluator, local_rule_run as run
from .fast_training_evaluator import FastEvaluation


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--implementation", choices=("arcs", "chunks", "compact", "life"), default="arcs")
    parser.add_argument("--arc-tiles", type=int, default=1)
    parser.add_argument("--request-chunk", type=int, default=4)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--append-run-logs", action="store_true")
    args, other = parser.parse_known_args(argv)
    locate = argparse.ArgumentParser(add_help=False)
    locate.add_argument("--out", type=Path, default=run.OUT)
    output = locate.parse_known_args(other)[0].out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_evaluation = local_rule_evaluator.Evaluation
    original_train = run.train

    def evaluation(device, split):
        if split != "training" or str(device) == "cpu":
            return original_evaluation(device, split)
        return FastEvaluation(device, split, backend=args.implementation,
                              requests=args.request_chunk, tiles=args.arc_tiles)

    def train(out, config, warm, **kwargs):
        previous_path = out / "progress.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
        src = Path(__file__).resolve().parent
        names = ["fast_training_run.py", "fast_training_host.py", "fast_training_evaluator.py"]
        if args.implementation == "compact":
            names += ["fast_local_arcs.py", "fast_local_arcs.cu",
                      "fast_local_arcs_compact.py", "fast_local_arcs_compact.cu"]
        elif args.implementation != "chunks":
            names += [f"fast_local_{args.implementation}.{ext}" for ext in ("py", "cu")]
        now = datetime.now(timezone.utc)
        meta = {"started_utc": now.isoformat(), "backend": args.implementation, "arc_tiles": args.arc_tiles,
                "requests_per_launch": args.request_chunk, "base_protocol_sha256": file_hash(out / "protocol.json"),
                "prior_progress": {k: previous.get(k) for k in ("generation", "current_arm", "completed_candidates")},
                "execution_sources": {name: file_hash(src / name) for name in names}}
        if args.verification:
            meta["verification"] = {"file": str(args.verification.resolve()), "sha256": file_hash(args.verification)}
        _write(out / "execution_versions" / (now.strftime("%Y%m%dT%H%M%S%fZ") + ".json"), meta)
        print("Using", args.implementation, "CUDA implementation on eligible training batches;", meta["prior_progress"], flush=True)
        return original_train(out, config, warm, **kwargs)

    with ExitStack() as stack:
        if args.append_run_logs:
            for name, redirect in (("stdout", redirect_stdout), ("stderr", redirect_stderr)):
                log = stack.enter_context((output / f"execution.{name}.log").open("a", encoding="utf-8", buffering=1))
                stack.enter_context(redirect(log))
        stack.enter_context(fast_training_host.installed())
        stack.enter_context(patch.object(local_rule_evaluator, "Evaluation", evaluation))
        stack.enter_context(patch.object(run, "train", train))
        try:
            return run.main(other)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
