"""Run the unchanged order protocol with independent births and physical holdout roots."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
import time

import numpy as np
import torch

from . import order_analysis as analysis, order_study as study


DEFAULT = study.DEFAULT.with_name("order-replication")
REPLICATION_BIRTHS = 64
REPLICATION_ROOTS = tuple(range(93549140, 93549332))
BIRTH_SEED = 637207183
DRIVER = Path(__file__).resolve()


def register(out, original):
    if (out / "registered.json").exists():
        raise RuntimeError("Replication is already registered")
    assert not (set(REPLICATION_ROOTS) & (set(study.TEST) | set(study.TRAIN) | set(study.DEV)))
    rng = np.random.default_rng(BIRTH_SEED)
    births = rng.choice(np.arange(1_000_000, 9_000_000), REPLICATION_BIRTHS, replace=False).tolist()
    assert not (set(births) & (set(study.BIRTHS) | set(study.DEV_BIRTHS)))
    old = study.read(original / "registered.json")
    recipe = original / "recipe.json"
    manifest = study.read(original / "frozen.json")
    if manifest["source_sha256"] != study.source_hashes() or manifest["recipe_sha256"] != study.digest(recipe):
        raise RuntimeError("The original source or recipe changed after freezing")
    original_analysis = study.read(original / "analysis-registration.json")
    if original_analysis["analysis_sha256"] != study.digest(Path(analysis.__file__)):
        raise RuntimeError("The original analysis changed after registration")
    out.mkdir(parents=True, exist_ok=True)
    copyfile(recipe, out / "recipe.json")
    study.write(out / "registered.json", {
        **old,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "post_primary_replication": True,
        "stopping": "All 64 new birth pairs on all 192 new roots. No interim scoring or tuning.",
        "development_selection": "None. Run the exact original development-selected recipe independently from scratch.",
        "unchanged_training": "Exact original training-world pool, optimizer, and training budget. New birth and mini-batch sampling seeds.",
        "confirm_births": births,
        "holdout_roots": REPLICATION_ROOTS,
        "birth_generation_seed": BIRTH_SEED,
        "original_registered_sha256": study.digest(original / "registered.json"),
        "original_summary_sha256": study.digest(original / "summary.json"),
        "recipe_sha256": study.digest(recipe),
        "analysis_sha256": original_analysis["analysis_sha256"],
        "source_sha256_at_registration": study.source_hashes(),
        "replication_driver_sha256": study.digest(DRIVER),
    })


def configure_replication(out):
    """Bind the unchanged original runner to the new preregistered seeds in this process."""
    reg = study.read(out / "registered.json")
    if (reg["replication_driver_sha256"] != study.digest(DRIVER)
            or reg["analysis_sha256"] != study.digest(Path(analysis.__file__))
            or reg["source_sha256_at_registration"] != study.source_hashes()
            or reg["recipe_sha256"] != study.digest(out / "recipe.json")):
        raise RuntimeError("A preregistered recipe or source file changed")
    if len(reg["confirm_births"]) != REPLICATION_BIRTHS or tuple(reg["holdout_roots"]) != REPLICATION_ROOTS:
        raise RuntimeError("Preregistered replication cohort does not match")
    study.BIRTHS = tuple(reg["confirm_births"])
    study.TEST = tuple(reg["holdout_roots"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "train", "freeze", "confirm", "summarize"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--original", type=Path, default=study.DEFAULT)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "register":
        register(args.out, args.original)
        return
    configure_replication(args.out)
    if args.stage == "train":
        if not (0 <= args.worker < args.workers):
            parser.error("Worker index must be between 0 and workers - 1")
        # Reuse the same immutable, deterministic explorer pools across this worker's births.
        training = study.pools()
        study.pools = lambda: training
        for birth in study.BIRTHS[args.worker::args.workers]:
            study.train_birth(args.out, birth)
    elif args.stage == "freeze":
        study.frozen(args.out)
    elif args.stage == "confirm":
        study.confirm(args.out)
    elif args.stage == "summarize":
        analysis.summarize(args.out, None)


if __name__ == "__main__":
    main()
