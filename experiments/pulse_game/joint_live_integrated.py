"""Confirm the same factorial at a common rate selected by the integrated joint arm."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from unittest.mock import patch

from . import joint_live_study as study
from . import live_module_edits_study as original
from . import ring_logic_study as io


OUT = io.REPO / "data/pulse-game/joint-live-integrated-20261119"
DEV_GROUPS = study.DEV_GROUPS
DEV_ROOTS = study.DEV_ROOTS
TEST_GROUPS = study.TEST_GROUPS
TEST_ROOTS = tuple(range(69024000, 69024016))


def code_hash():
    return study.code_hash() | {Path(__file__).name: io.sha(Path(__file__))}


def register(out):
    path = out / "registered.json"
    if path.exists():
        raise RuntimeError("Already registered")
    if set(TEST_ROOTS) & set(DEV_ROOTS + study.TEST_ROOTS + original.DEV_ROOTS + original.TEST_ROOTS):
        raise RuntimeError("The confirmation roots must be new")
    study.require_registered(study.OUT)
    baseline = io.read(study.OUT / "registered.json")
    baseline.update(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source_sha256=code_hash(), test_roots=TEST_ROOTS,
                    selection="Select active rate on the joint dev arm with both weight writes and editing enabled. Score is pooled last20 reversal gain minus twice any stable last20 loss, compared to neither on dev. Report no-write alternative score 0 separately. Common selected rate in all weight-plastic arms. Never use the first factorial's confirmation results for selection. Confirmation worlds are newly registered; fitted groups are reused.",
                    first_study_registration_sha256=io.sha(study.OUT / "registered.json"))
    io.write(path, baseline)


def require_registered(out):
    if io.read(out / "registered.json")["source_sha256"] != code_hash():
        raise RuntimeError("Registered code changed")


def develop(out):
    require_registered(out)
    if (out / "recipe.json").exists():
        raise RuntimeError("Development already sealed")
    head = original.heads(DEV_GROUPS)
    sets = {case: study.tapes(DEV_ROOTS, head[2], case) for case in ("stable", "reversal")}

    def accuracy(case, rate):
        data = sets[case]
        physical = study.run(head, DEV_ROOTS, data, study.SCHEDULE, "joint" if rate else "neither", rate)["physical"]
        return float(study.windows(physical, data[-2])["last20"].mean())

    baseline = {case: accuracy(case, 0.) for case in sets}

    def evaluate(rate):
        fitted = {case: accuracy(case, rate) for case in sets}
        delta = {case: fitted[case] - baseline[case] for case in sets}
        result = dict(rate=rate, accuracy=fitted, delta=delta, score=delta["reversal"] - 2 * max(0, -delta["stable"]))
        print("joint development", result, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        candidates = list(pool.map(evaluate, study.RATES))
    best = max(candidates, key=lambda row: row["score"])
    io.write(out / "development.json", dict(checkpoints=head[3], rows=head[2], baseline=baseline, candidates=candidates))
    io.write(out / "recipe.json", dict(rate=best["rate"], score=best["score"], deploy="joint" if best["score"] > 0 else "no_writes",
        registered_sha256=io.sha(out / "registered.json"), development_sha256=io.sha(out / "development.json")))


def sealed_recipe(out):
    require_registered(out)
    recipe = io.read(out / "recipe.json")
    for stage in ("registered", "development"):
        if recipe[f"{stage}_sha256"] != io.sha(out / f"{stage}.json"):
            raise RuntimeError("Development seal mismatch")
    return recipe


def confirm(out):
    # Reuse the prior sealed confirmation runner with a new world partition and seal.
    with patch.multiple(study, TEST_ROOTS=TEST_ROOTS, sealed_recipe=sealed_recipe):
        study.confirm(out)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "develop", "confirm"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out)


if __name__ == "__main__":
    main()
