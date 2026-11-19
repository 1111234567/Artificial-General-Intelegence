"""Additive fixed-feature ridge audit, separate from controller learning.

The train-only normalizer is shared by the direct and chunked calculations.
Only after fixing it do chunks contribute [1,z]^T[1,z] and [1,z]^T onehot(y).
This is a two-pass algebraic audit of cached features, not an online controller,
autonomous consolidation, adaptive-feature replay equivalence or FORCE.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.continuity.run import atomic_json
from experiments.credit.probe import classification_metrics

FORMAT_VERSION = "credit-fixed-feature-statistics-v1"
FORCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC2756108/"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_training(cls, x):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("nonempty finite training features required")
        scale = x.std(0)
        return cls(x.mean(0), np.where(scale > 1e-8, scale, 1.0))

    def design(self, x):
        x = np.asarray(x, np.float64)
        return np.column_stack((np.ones(len(x)), (x - self.mean) / self.scale))


class AdditiveRidge:
    def __init__(self, features, classes):
        if features < 1 or classes < 2:
            raise ValueError("positive feature count and at least two classes required")
        self.features, self.classes, self.n = features, classes, 0
        self.xtx = np.zeros((features + 1, features + 1), np.float64)
        self.xty = np.zeros((features + 1, classes), np.float64)

    def add(self, design, labels):
        a, labels = np.asarray(design, np.float64), np.asarray(labels, np.int64)
        if a.shape != (len(labels), self.features + 1) or not np.isfinite(a).all():
            raise ValueError("invalid finite augmented design")
        if np.any((labels < 0) | (labels >= self.classes)):
            raise ValueError("invalid target class")
        if not np.all(a[:, 0] == 1):
            raise ValueError("first design column must be the intercept")
        self.xtx += a.T @ a
        self.xty += a.T @ np.eye(self.classes, dtype=np.float64)[labels]
        self.n += len(labels)

    def equations(self, alpha):
        if not self.n or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("positive fixed alpha and nonempty statistics required")
        matrix = self.xtx / self.n
        matrix[np.arange(1, self.features + 1), np.arange(1, self.features + 1)] += alpha
        return matrix, self.xty / self.n

    def solve(self, alpha):
        return np.linalg.solve(*self.equations(alpha))


def actor_safe_blocks(columns, selected, *, block_rows=64):
    """Chronological per actor/condition/job; keep a request's samples together."""
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("block_rows must be positive")
    indices = sorted(np.flatnonzero(selected), key=lambda i: (
        str(columns["condition"][i]), int(columns["actor"][i]), int(columns["job"][i]), int(columns["tick"][i])))
    group_key = lambda i: (str(columns["condition"][i]), int(columns["actor"][i]), int(columns["job"][i]))
    for _, group in itertools.groupby(indices, group_key):
        pending = []
        for _, request in itertools.groupby(group, lambda i: int(columns["request"][i])):
            request = list(request)
            if pending and len(pending) + len(request) > block_rows:
                yield np.asarray(pending, np.int64)
                pending = []
            pending.extend(request)
        if pending:
            yield np.asarray(pending, np.int64)


def _relative(error, reference):
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(error) / denominator) if denominator else float(np.linalg.norm(error))


def _prediction_check(a, direct, additive, labels):
    scores_direct, scores_additive = a @ direct, a @ additive
    predictions_direct, predictions_additive = scores_direct.argmax(-1), scores_additive.argmax(-1)
    error = float(np.max(np.abs(scores_direct - scores_additive))) if len(a) else 0.0
    sorted_scores = np.sort(scores_direct, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    disagreement = predictions_direct != predictions_additive
    return dict(
        n=len(a), max_abs_score_difference=error,
        prediction_agreement=float((~disagreement).mean()) if len(a) else None,
        disagreements=int(disagreement.sum()),
        disagreements_outside_numerical_tie_bound=int((disagreement & (margin > 2 * error + 1e-12)).sum()),
        direct=classification_metrics(labels, predictions_direct, direct.shape[1]),
        additive=classification_metrics(labels, predictions_additive, direct.shape[1]),
    )


def audit_arrays(features, labels, columns, train, heldout, *, alpha=.01, block_rows=64, classes=8):
    """One job's fixed readout. No heldout sample contributes to any statistic."""
    x, y = np.asarray(features), np.asarray(labels, np.int64)
    train, heldout = np.asarray(train, bool), np.asarray(heldout, bool)
    if np.any(train & heldout):
        raise ValueError("training and heldout samples overlap")
    if not train.any():
        return dict(status="undefined_no_training_samples", train_samples=0, heldout_samples=int(heldout.sum()))
    normalizer = Normalizer.from_training(x[train])
    direct = AdditiveRidge(x.shape[1], classes)
    direct.add(normalizer.design(x[train]), y[train])
    additive = AdditiveRidge(x.shape[1], classes)
    digest = hashlib.sha256()
    n_blocks, max_block = 0, 0
    for indices in actor_safe_blocks(columns, train, block_rows=block_rows):
        if not np.all(train[indices]) or np.any(heldout[indices]):
            raise AssertionError("heldout leakage into sufficient statistics")
        identities = {(int(columns["actor"][i]), str(columns["condition"][i]), int(columns["job"][i])) for i in indices}
        if len(identities) != 1 or np.any(np.diff(columns["tick"][indices]) < 0):
            raise AssertionError("stream block crossed actor/domain/job or reversed time")
        additive.add(normalizer.design(x[indices]), y[indices])
        digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
        n_blocks, max_block = n_blocks + 1, max(max_block, len(indices))
    if direct.n != additive.n:
        raise AssertionError("streamed samples lost or repeated")
    beta_direct, beta_additive = direct.solve(alpha), additive.solve(alpha)
    matrix, rhs = additive.equations(alpha)
    check = _prediction_check(normalizer.design(x[heldout]), beta_direct, beta_additive, y[heldout])
    residual = _relative(matrix @ beta_additive - rhs, rhs)
    parameter_relative = _relative(beta_direct - beta_additive, beta_direct)
    passed = (parameter_relative < 1e-8 and residual < 1e-10
              and check["disagreements_outside_numerical_tie_bound"] == 0)
    return dict(
        status="passed_float64_equivalence" if passed else "failed_numerical_equivalence",
        train_samples=direct.n, heldout_samples=int(heldout.sum()), feature_dimensions=x.shape[1], classes=classes,
        alpha=alpha, intercept_penalized=False, normalizer="shared train-only mean/std; same threshold1e-8 as frozen probe",
        blocks=n_blocks, max_block_rows=max_block, block_order_sha256=digest.hexdigest(),
        xtx_max_abs_difference=float(np.abs(direct.xtx - additive.xtx).max()),
        xty_max_abs_difference=float(np.abs(direct.xty - additive.xty).max()),
        parameter_max_abs_difference=float(np.abs(beta_direct - beta_additive).max()),
        parameter_relative_l2_difference=parameter_relative,
        additive_normal_equation_relative_residual=residual,
        heldout=check,
        memory=dict(
            additive_xtx_bytes=additive.xtx.nbytes, additive_xty_bytes=additive.xty.nbytes,
            normalizer_bytes=normalizer.mean.nbytes + normalizer.scale.nbytes, count_bytes=8,
            retained_statistic_state_bytes=additive.xtx.nbytes + additive.xty.nbytes + normalizer.mean.nbytes + normalizer.scale.nbytes + 8,
            coefficients_bytes=beta_additive.nbytes,
            cached_training_feature_bytes=x[train].nbytes, cached_training_label_bytes=y[train].nbytes,
            direct_augmented_float64_design_bytes=direct.n * (x.shape[1] + 1) * 8,
            direct_onehot_float64_target_bytes=direct.n * classes * 8,
            peak_stream_design_bytes=max_block * (x.shape[1] + 1) * 8,
            excludes="Python metadata, source NPZ storage and the already cached heldout features; this is not an end-to-end memory-free learner",
        ),
    )


def target_association_counterexample(alpha=.01):
    x = np.asarray([[-1.], [1.]])
    normalizer = Normalizer.from_training(x)
    design = normalizer.design(x)
    first, second = AdditiveRidge(1, 2), AdditiveRidge(1, 2)
    first.add(design, [0, 1])
    second.add(design, [1, 0])
    beta_a, beta_b = first.solve(alpha), second.solve(alpha)
    return dict(
        same_xtx=bool(np.array_equal(first.xtx, second.xtx)), same_label_marginals=True,
        xty_max_abs_difference=float(np.abs(first.xty - second.xty).max()),
        coefficient_max_abs_difference=float(np.abs(beta_a - beta_b).max()),
        predictions_a=(design @ beta_a).argmax(-1).tolist(), predictions_b=(design @ beta_b).argmax(-1).tolist(),
        interpretation="Feature correlations and class frequencies alone do not determine the supervised readout; input-target association is also needed. This is not an impossibility theorem for other local objectives.",
    )


def audit_probe(probe, *, alpha=.01, block_rows=64):
    probe = Path(probe).resolve()
    source_hash = file_sha256(__file__)
    probe_hash = file_sha256(probe)
    document = json.loads(probe.read_text(encoding="utf-8"))
    manifest = document.get("collection", document)
    if manifest.get("kind") != "frozen_state_collection":
        raise ValueError("probe JSON must contain a frozen collection manifest")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be fixed, finite and positive")
    parts, metadata = [], []
    for record in manifest["records"]:
        path = Path(record["dataset"])
        if file_sha256(path) != record["dataset_sha256"]:
            raise ValueError("cached feature checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            part = dict(x=data["view__tanh_h"].copy())
            for name in ("joint", "actor", "job", "request", "condition", "tick"):
                part[name] = data[f"column__{name}"].copy()
        parts.append(part)
        metadata.append(dict(condition=record["condition"], path=str(path.resolve()), sha256=record["dataset_sha256"]))
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    del parts
    partition = {int(actor): int(split) for actor, split in manifest["actor_partition"].items()}
    split = np.asarray([partition[int(actor)] for actor in data["actor"]])
    known = data["joint"] >= 0
    records = []
    for job in sorted(set(map(int, data["job"]))):
        job_mask = data["job"] == job
        train, test = known & job_mask & (split == 0), known & job_mask & (split == 2)
        result = audit_arrays(data["x"], data["joint"], data, train, test, alpha=alpha, block_rows=block_rows)
        result.update(job=job, train_actors=sorted(set(map(int, data["actor"][train]))),
                      heldout_actors=sorted(set(map(int, data["actor"][test]))))
        records.append(result)
    if probe_hash != file_sha256(probe) or source_hash != file_sha256(__file__):
        raise AssertionError("input probe or statistics source changed during audit")
    if any(file_sha256(item["path"]) != item["sha256"] for item in metadata):
        raise AssertionError("cached probe features changed during audit")
    return dict(
        format=FORMAT_VERSION, kind="algebraic_fixed_feature_audit", probe=str(probe), probe_sha256=probe_hash,
        source_sha256=source_hash, collector_source_sha256=manifest["collector_source_sha256"],
        checkpoint=manifest["checkpoint"], checkpoint_sha256=manifest["checkpoint_sha256"],
        training_seed=manifest["training_seed"], training_arm=manifest["training_arm"],
        alpha=alpha, feature="exact_float32_tanh_h", target="joint8", conditions=[item["condition"] for item in metadata],
        validation_samples_unused=int((known & (split == 1)).sum()), datasets=metadata, records=records,
        passed=all(record["status"] == "passed_float64_equivalence" for record in records),
        counterexample=target_association_counterexample(alpha),
        force_reference=dict(url=FORCE_URL, title="Generating Coherent Patterns of Activity from Chaotic Neural Networks",
            context="FORCE uses error-driven recursive least squares with feedback that changes recurrent dynamics. Here features and world trajectories are cached and fixed, and readout predictions never feed back; this is not a FORCE replication."),
        limitations=[
            "A shared train-only normalizer is fixed before additive standardized statistics are accumulated; this is a two-pass audit, not one-pass normalization.",
            "All cached conditions, including8--16, enter the pooled diagnostic training mixture; no probe transfer or autonomous control claim is made.",
            "Exact algebraic equality permits floating-point summation differences; score ties can change argmax despite negligible coefficient error.",
            "Sufficient statistics cease to represent the new design matrix if the recurrent representation changes. This does not establish adaptive-body replay equivalence.",
            "The target cross-statistic uses supervised associations. Activity-only correlation is not a substitute for it for this objective.",
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--block-rows", type=int, default=64)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_probe(args.probe, alpha=args.alpha, block_rows=args.block_rows)
    atomic_json(args.output, result)
    print(json.dumps(dict(output=str(args.output.resolve()), passed=result["passed"], jobs=len(result["records"]))))


if __name__ == "__main__":
    main()
