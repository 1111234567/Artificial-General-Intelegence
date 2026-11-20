"""Independently recompute actual resource receipts from archived visible events and motor choices."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from . import interleave_revisit_train as archive
from . import local_trace_eval as prior
from . import noon_review_repeated as life
from . import noon_review_world as world


def check_episode(raw, tape):
    candidate = tape["candidates"]
    lanes = len(candidate)
    rows = np.arange(lanes)[:, None]
    selected = (raw["selected_side"] > 0).astype(int)
    reviewed = np.take_along_axis(candidate, selected[..., None], axis=2).squeeze(2)
    np.testing.assert_array_equal(raw["selected_object"], reviewed)

    law = tape["law"]
    sign = tape["ordinary_orientation"]
    motor = raw["motor"]
    learned = tape["seen_item"]
    round_id = np.arange(life.ROUNDS)
    later_position = life.WARM + 2 * round_id
    review = law[rows, reviewed] * tape["review_orientation"] * motor[:, :, 0]
    seen = law[rows, learned[:, :-1]] * sign[:, later_position] * motor[:, :, 1]
    novel = law[:, life.WARM + round_id] * sign[:, later_position + 1] * motor[:, :, 2]
    final = law[np.arange(lanes), learned[:, -1]] * sign[:, -1] * raw["final_side"]
    np.testing.assert_array_equal(raw["earned"], np.stack((review, seen, novel), axis=2))
    np.testing.assert_array_equal(raw["final"], final)
    assert (np.abs(raw["warm"]) == 1).all()
    assert np.isfinite(raw["preview_motor"]).all()
    signed = raw["warm"].sum(1) + review.sum(1) + seen.sum(1) + novel.sum(1) + final
    np.testing.assert_array_equal(life.resources(raw)["total"], signed)
    return len(signed)


def verify(fit_stage, worlds):
    summaries, certificates = {}, {}
    specs = [("main", 0), ("replicate", 0)]
    if fit_stage:
        specs.append(("trained", fit_stage))
    verified_lives, verified_raw, receipts = 0, 0, 0
    for split, stage in specs:
        paired = world.EVIDENCE / f"noon-review-repeated-{split}-{stage:04d}-summary-n{worlds}.json"
        summaries[paired.name] = prior.digest(paired)
        for cohort, births in world.COHORTS.items():
            name = world.EVIDENCE / f"noon-review-repeated-{split}-{stage:04d}-{cohort}-n{worlds}.json"
            evidence = json.loads(name.read_text(encoding="utf8"))
            certificates[name.name] = prior.digest(name)
            paths = evidence["raw_sha256"]
            if any(prior.digest(life.DATA / file) != sha for file, sha in paths.items()):
                raise ValueError("A result no longer matches the frozen report hashes")
            verified_raw += len(paths)
            folder = life.DATA / "evaluation" / split / f"{stage:04d}" / cohort
            with np.load(folder / f"tape-n{worlds}.npz", allow_pickle=False) as recorded:
                tape = {key: recorded[key] for key in recorded.files}
            np.testing.assert_array_equal(tape["law"][::2], -tape["law"][1::2])
            assert (tape["candidates"].max(2) < life.WARM + np.arange(life.ROUNDS)).all()
            assert (tape["candidates"][:, :, 0] != tape["candidates"][:, :, 1]).all()
            policies = (*life.POLICIES, *(life.FIT_ARMS if stage else ()))
            for birth in births:
                reference = None
                for policy in policies:
                    file = folder / f"{policy}-{birth}-n{worlds}.npz"
                    with np.load(file, allow_pickle=False) as raw:
                        count = check_episode(raw, tape)
                        verified_lives += count
                        receipts += count * (life.ROUNDS * 3 + 1)
                        if reference is None:
                            reference = raw["warm"]
                        else:
                            np.testing.assert_array_equal(raw["warm"], reference)

    origin = torch.load(life.SOURCE, weights_only=True, map_location="cpu")
    assert origin["step"] == 4000 and sum(value.numel() for value in origin["weights"].values()) == 17
    assert origin["manifest"] == prior.digest(life.SOURCE.parent / "run.json")
    parameters = {"single_live_route": prior.digest(life.SOURCE)}
    if fit_stage:
        manifest = life.DATA / "fit" / "run.json"
        data = json.loads(manifest.read_text(encoding="utf8"))
        assert data["original_readout_sha256"] == parameters["single_live_route"]
        assert data["plan_sha256"] == prior.digest(life.PLAN)
        for file, expected in data["source_sha256"].items():
            assert prior.digest(Path(life.__file__).parent / file) == expected
        parameters["fit_manifest"] = prior.digest(manifest)
        for arm in life.FIT_ARMS:
            checkpoints = life.DATA / "fit" / arm / f"step-{fit_stage}.pt"
            state = torch.load(checkpoints, weights_only=True, map_location="cpu")
            assert state["arm"] == arm and state["step"] == fit_stage
            assert state["trace"].shape[0] == fit_stage and state["trace"].isfinite().all()
            assert state["manifest_sha256"] == parameters["fit_manifest"]
            assert sum(value.numel() for value in state["weights"].values()) == 17
            parameters[arm] = prior.digest(checkpoints)
    suffix = "archive-verification" if fit_stage else "frozen-archive-verification"
    archive._write_json(world.EVIDENCE / f"noon-review-repeated-{suffix}.json",
                        dict(raw_files_checked=verified_raw, real_lives_checked=verified_lives,
                             receipts_recomputed_from_motors=receipts, warm_inputs_match=True, reports=certificates,
                             summary_sha256=summaries, parameters_sha256=parameters))
    print(f"Verified {verified_raw} raw files, {verified_lives} whole lives, {receipts} recomputed tastes", flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, default=1600)
    p.add_argument("--worlds", type=int, default=384)
    a = p.parse_args()
    verify(a.stage, a.worlds)


if __name__ == "__main__":
    main()
