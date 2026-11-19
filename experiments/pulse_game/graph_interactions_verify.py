"""Independently recompute nonlinear-task metrics and physical evidence from the archives."""

from argparse import ArgumentParser
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import graph_interactions as model, graph_interactions_study as study
from .unified_live_graph_study import audit_edits, digest, read, write


def verify(out):
    registration = study.require(out)
    summary = read(out / "summary.json")
    assert summary["registered_sha256"] == digest(out / "registered.json")
    assert summary["development_sha256"] == digest(out / "development.json")
    seeds = summary["configurations"]
    arms = (*seeds, "tanh_sham", "square_sham", *study.CONTROLS)
    prefixes = {}
    physical_world = {}
    validation = {}
    rng = np.random.default_rng(registration["bootstrap_seed"])

    for mode in study.MODES:
        recorded = summary["outcomes"][mode]
        zipped = out / f"{mode}-raw.zip"
        assert digest(zipped) == recorded["archive_sha256"]
        with ZipFile(zipped) as archive:
            assert archive.namelist() == [f"{mode}.npz"]
            contents = archive.read(f"{mode}.npz")
        assert hashlib.sha256(contents).hexdigest() == recorded["npz_sha256"]
        with np.load(BytesIO(contents), allow_pickle=False) as data:
            roots = len(study.TEST_WORLDS)
            total = roots * len(study.TEST_BIRTHS)
            np.testing.assert_array_equal(data["worlds"], np.tile(study.TEST_WORLDS, len(study.TEST_BIRTHS)))
            np.testing.assert_array_equal(data["births"], np.repeat(study.TEST_BIRTHS, roots))
            novel = data["raw_odors"].reshape(roots, study.TRIALS * 2, -1)
            assert all(len(np.unique(life, axis=0)) == len(life) for life in novel)
            np.testing.assert_allclose(np.linalg.norm(data["private_weights"], axis=-1), 1, atol=1e-7)
            coefficients = np.einsum(
                "btik,bk->bti", model.interactions(data["raw_odors"] * 2. - 1.), data["private_weights"]
            )
            if mode == "reverse":
                coefficients = np.where(
                    np.arange(study.TRIALS)[None, :, None] >= data["change"][:roots, None, None],
                    -coefficients, coefficients,
                )
            expected = 1. / (1. + np.exp(-model.TASTE_GAIN * coefficients))
            np.testing.assert_array_equal(expected, data["probability"])
            np.testing.assert_array_equal(data["private_best"], np.tile(expected.argmax(axis=-1), (len(study.TEST_BIRTHS), 1)))
            private_world = ("raw_odors", "private_weights", "worlds", "births", "change")
            if mode == "stable":
                physical_world = {key: data[key] for key in private_world}
            else:
                for key, values in physical_world.items():
                    np.testing.assert_array_equal(data[key], values)
            active_prefix = np.arange(study.TRIALS)[None, :] < data["change"][:, None]
            stats = {}
            for arm in arms:
                payload = {key.split("/", 1)[1]: data[key] for key in data.files if key.startswith(f"{arm}/")}
                shown, contact = payload["shown"], payload["contact"]
                hit = contact >= 0
                acted = contact.clip(min=0)[..., None]
                latent = data["probability"][np.arange(total) % roots]
                realized = data["realized"][np.arange(total) % roots]
                np.testing.assert_array_equal(payload["correct"], hit & (contact == data["private_best"]))
                np.testing.assert_array_equal(payload["expected"], hit * np.take_along_axis(latent, acted, axis=-1)[..., 0])
                np.testing.assert_array_equal(payload["taste"], hit * np.take_along_axis(realized, acted, axis=-1)[..., 0])
                assert not shown[:, -study.TAIL:].any()
                if not arm.endswith("_sham"):
                    np.testing.assert_array_equal(shown[:, :-study.TAIL], payload["taste"][:, :-study.TAIL])
                if "edits" in payload:
                    audit_edits(payload["initial_mask"], payload["initial_weight"], payload["edits"],
                                payload["final_mask"], payload["final_weight"])
                    assert np.all(payload["edits"][shown == 0] == -1)
                    assert not np.any(payload["writes"][shown == 0])
                prefix_fields = ("correct", "expected", "taste", "contact", "contact_tick", "shown")
                if "edits" in payload:
                    prefix_fields += ("edits", "writes")
                if mode == "stable":
                    prefixes[arm] = {key: payload[key][active_prefix] for key in prefix_fields}
                else:
                    for key, values in prefixes[arm].items():
                        np.testing.assert_array_equal(payload[key][active_prefix], values)
                stats[arm] = study.measure(payload | {"change": data["change"]})
                assert recorded["means"][arm] == {key: float(values.mean()) for key, values in stats[arm].items()}
            for a, b in study.PRIMARY:
                expected = study.contrast(stats[a]["silent_tail"] - stats[b]["silent_tail"], rng)
                assert expected == recorded["paired"][f"{a}_minus_{b}"]
            ties = int((data["probability"][..., 0] == data["probability"][..., 1]).sum())
            validation[mode] = dict(conditioned_utility_ties=ties, distinct_foods_per_world=novel.shape[1],
                                    raw_rechecked=True, sha256=recorded["npz_sha256"])
    write(out / "verified-replay.json", dict(
        source_sha256=digest(Path(__file__)), summary_sha256=digest(out / "summary.json"),
        validation=validation, equal_prefix_verified=True,
    ))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    verify(parser.parse_args().out)


if __name__ == "__main__":
    main()
