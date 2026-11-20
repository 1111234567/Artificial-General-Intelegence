"""Run the fixed six-birth confirmation of the terminal query experiment."""

from argparse import ArgumentParser
import json
from pathlib import Path
from shutil import copyfile

from . import interleave_revisit_train as revisit
from .local_trace_eval import digest


BIRTHS = tuple(range(811, 817))
TRAIN_SEED = 431984
CONFIRMATION_SEED = 9124431
WORLDS = 192
DATA = revisit.ROOT / "data/interleaved-learning-20261120"
EVIDENCE = revisit.ROOT / "docs/evidence/interleaved-learning-2026-11-20"


def confirm(path, warm, evidence, *, workers=2, threads=1):
    """Open the confirmation tape only after training both arms on all six births."""
    if not 1 <= workers <= 2 or threads < 1:
        raise ValueError("Use one or two workers and a positive per-worker thread count")
    config = revisit.Config(world_seed=TRAIN_SEED)
    revisit.SEEDS = {"confirmation": CONFIRMATION_SEED}
    contract = {
        "births": list(BIRTHS), "worlds_per_birth": WORLDS, "confirmation_seed": CONFIRMATION_SEED,
        "training_seed": TRAIN_SEED, "updates_per_arm": config.updates, "arms": list(revisit.ARMS),
        "training_evaluation": "none", "frozen_evaluation_stages": [0, config.updates],
        "source_sha256": {Path(name).name: digest(Path(name)) for name in (__file__, revisit.__file__)},
        "warm_manifest_sha256": digest(warm / "run.json"),
        "warm_checkpoint_sha256": {str(birth): digest(revisit._checkpoint(warm, "active", birth, 8000))
                                   for birth in BIRTHS},
    }
    declaration = path / "confirmation-contract.json"
    if declaration.exists():
        if json.loads(declaration.read_text(encoding="utf8")) != contract:
            raise ValueError(f"Confirmation protocol differs at {declaration}")
    else:
        revisit._write_json(declaration, contract)
    for birth in BIRTHS:
        source = revisit._checkpoint(warm, "active", birth, 8000)
        archived = path / "warm" / source.relative_to(warm)
        archived.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, archived)
        if digest(archived) != contract["warm_checkpoint_sha256"][str(birth)]:
            raise ValueError(f"Warm checkpoint changed while archiving {source}")
    revisit.run(path, warm, config, BIRTHS, workers=workers, group_size=3, threads=threads, evaluate=False)
    expected = [revisit._checkpoint(path, arm, birth, config.updates) for arm in revisit.ARMS for birth in BIRTHS]
    if not all(file.is_file() for file in expected):
        raise RuntimeError("The confirmation requires all six final checkpoints for each arm")
    for stage in (0, config.updates):
        name = f"retention-objective-confirmation-{stage:06d}.json"
        existing = path / "evaluation" / name
        if existing.exists():
            report = json.loads(existing.read_text(encoding="utf8"))
            if (report["births"], report["seed"], report["stage"]) != (list(BIRTHS), CONFIRMATION_SEED, stage):
                raise ValueError(f"Unexpected frozen evaluation identity at {existing}")
            for group in ("raw_sha256", "checkpoint_sha256"):
                for file, expected_hash in report[group].items():
                    if digest(path / file) != expected_hash:
                        raise ValueError(f"Confirmation archive changed at {file}")
            revisit._write_json(evidence / name, report)
        else:
            revisit.evaluate_stage(path, evidence, BIRTHS, stage, split="confirmation", worlds=WORLDS)
    archived_files = sorted(file for file in path.rglob("*")
                            if file.is_file() and file.suffix in {".npz", ".pt", ".json"}
                            and file.name != "confirmation-sha256.json")
    archive = {"files": {file.relative_to(path).as_posix(): digest(file) for file in archived_files}}
    revisit._write_json(path / "confirmation-sha256.json", archive)
    revisit._write_json(evidence / "retention-objective-confirmation-sha256.json", archive)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA / "retention-objective-confirmation")
    parser.add_argument("--warm", type=Path, default=DATA / "replication-births")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    confirm(args.out, args.warm, args.evidence, workers=args.workers, threads=args.threads)
