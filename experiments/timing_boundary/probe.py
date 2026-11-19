"""Frozen, offline timing/content readouts. No recurrent parameter is trained.

CLI order: declare; fit --seed 0/1/2; freeze; evaluate --seed 0/1/2.
Only teacher train data fit any transform or readout. Validation is diagnostic;
teacher test and own-policy test are inaccessible to this program before freeze.
The exported NPZ readouts can be reproduced by ``numpy_predict`` without sklearn.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import warnings

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/timing-boundary/probe-v1"
ARMS = ("midpoint", "first", "interior", "random")
SEEDS = (0, 1, 2)
BOUNDARY = (-1, 0)
TIMING_FEATURES = ("h", "obs", "logits", "h_shuffled", "h_mlp64")
CONTENT_FEATURES = ("h", "obs")
REMAINING_FEATURES = ("h", "obs")
META_KEYS = ("pair_id", "offset", "job", "actor", "rid", "tick", "family", "raw",
             "conflict", "target", "duration", "remaining_ticks", "legal")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def source_path(seed, split, arm, mode="teacher"):
    from experiments.timing_boundary.data import sample_path
    return sample_path(seed, split, arm, mode=mode)


def input_pins(path):
    path = Path(path)
    meta = path.with_suffix(".json")
    pins = {relative(path): sha(path)}
    if meta.exists():
        pins[relative(meta)] = sha(meta)
    return pins


def check_pins(pins):
    for path, expected in pins.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"source changed: {path}")


def declare():
    """Declare all methods before reading feature arrays or any results."""
    from experiments.timing_boundary import data
    data_protocol = data.OUT / "protocol.json"
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), version=1,
        script_sha256=sha(__file__), data_protocol_path=relative(data_protocol),
        data_protocol_sha256=sha(data_protocol), seeds=list(SEEDS), arms=list(ARMS),
        question="Can fixed frozen recurrent state decode the consecutive last-illegal / first-legal boundary independently of FINISH content?",
        data="New teacher trajectories only for fitting; disjoint actor train/validation/test. Each model receives identical teacher histories. Own-policy test uses the same frozen teacher-fitted probes, with no own-policy fitting. Random192 is an untrained generic reservoir on teacher histories only.",
        units="One paired request; endpoints offset -1 (last actual illegal) and 0 (first actual legal), exactly consecutive ticks, are mandatory. All metrics retain request and row denominators.",
        representation="h is shared 192-dimensional POST-current-observation hidden state used by original head. obs is the complete 56-dimensional current observation. logits feature is target-slot 11 original logits in native action order 0..10. Fit each job/slot separately, pool A/B and raw codes.",
        timing=dict(training_rows=list(BOUNDARY), target="actual legal bool", features=list(TIMING_FEATURES),
                    logistic=dict(C=1.0, penalty="L2", l1_ratio=0.0, solver="lbfgs", max_iter=2000, tol=1e-8,
                                  class_weight=None),
                    standardization="mean and population standard deviation fitted on teacher TRAIN rows only; zero scale becomes 1",
                    threshold=0.5, threshold_rule="p>=0.5 predicts legal", rank="score(first legal)>score(last illegal); exact ties count 0.5; strict wins and ties also reported",
                    shuffled="h-only logistic with TRAIN labels swapped within exactly floor(P/2) pairs per slot; RNG 928000+100*modelseed+slot, same flips across arms. Real labels retained for all evaluation.",
                    nonlinear=dict(kind="MLPClassifier", hidden_layer_sizes=[64], activation="tanh", solver="adam", alpha=1e-4,
                                   batch_size=256, learning_rate_init=1e-3, max_iter=400, early_stopping=False,
                                   n_iter_no_change=401, tol=0.0, shuffle=True, random_state="910000+100*modelseed+slot", budget="fixed 400 epochs, no sweep")),
        content=dict(rows=list(BOUNDARY), target="target action minus 3, eight FINISH classes 0..7", features=list(CONTENT_FEATURES),
                     model="same fixed L2 multinomial logistic; independent from timing; no timing label in content fit", baseline="original conditional argmax(logits[target slot,3:11])+3"),
        remaining=dict(rows="all retained available offsets -8,-4,-2,-1,0,1,2", target="(ready_time-now)/0.25 in ticks; first legal may be slightly negative", features=list(REMAINING_FEATURES),
                       model="Ridge(alpha=1,fit_intercept=True,solver=cholesky), train-only standardization", boundary="predict legal iff predicted remaining_ticks<=0; report sign row and both-correct as well as boundary rank predicted_remaining(last)>predicted_remaining(first)"),
        evaluation="Validation only diagnostic; no feature/hyperparameter/threshold/epoch/model selection. Freeze every coefficient/scaler artifact for all seeds/arms before any test or own test scoring. Family A/B, raw, conflict/shared, slot, duration [8,12) / [12,16] strata; modelseed-level summaries. Probe predictions never become world actions. Free-policy outcomes are separate data artifacts.",
        uncertainty="Linear failure does not establish absence of information. Fixed tiny nonlinear h-only timing probe is a limited capacity control. Random GRU tests generic reservoir decodability. No significance claim from treating rows as independent model seeds.",
    )
    write_json(OUT / "protocol.json", protocol)
    return protocol


def verify_protocol():
    protocol = read(OUT / "protocol.json")
    if protocol["script_sha256"] != sha(__file__):
        raise ValueError("probe script changed after declaration")
    if protocol["data_protocol_sha256"] != sha(ROOT / protocol["data_protocol_path"]):
        raise ValueError("data protocol changed after declaration")
    return protocol


def load_samples(path):
    with np.load(path, allow_pickle=False) as archive:
        rows = {key: archive[key] for key in ("h", "obs", "logits", *META_KEYS)}
    n = len(rows["pair_id"])
    if rows["h"].shape != (n, 192) or rows["obs"].shape != (n, 56) or rows["logits"].shape != (n, 2, 11):
        raise ValueError("unexpected feature shape")
    if any(len(value) != n for value in rows.values()):
        raise ValueError("row length mismatch")
    if not np.isfinite(rows["h"]).all() or not np.isfinite(rows["obs"]).all() or not np.isfinite(rows["logits"]).all():
        raise ValueError("nonfinite features")
    if not set(np.unique(rows["job"])).issubset({0, 1}):
        raise ValueError("job outside two slots")
    boundary_indices(rows)  # also proves the mandatory paired endpoints
    return rows


def boundary_indices(rows):
    selected = np.flatnonzero(np.isin(rows["offset"], BOUNDARY))
    last, first = pair_indices(rows, selected)
    all_pairs = np.unique(rows["pair_id"])
    if len(last) != len(all_pairs):
        raise ValueError("not every retained request has a boundary pair")
    if not np.all(rows["tick"][first] == rows["tick"][last] + 1):
        raise ValueError("boundary observations must be consecutive actual ticks")
    if rows["legal"][last].any() or not rows["legal"][first].all():
        raise ValueError("boundary requires actual illegal / legal labels")
    for key in ("job", "actor", "rid", "family", "raw", "conflict", "target", "duration"):
        if not np.array_equal(rows[key][last], rows[key][first]):
            raise ValueError(f"pair metadata mismatch: {key}")
    return selected


def pair_indices(rows, selected=None):
    """Return source indices ordered by pair ID, never by incidental row order."""
    if selected is None:
        selected = np.arange(len(rows["pair_id"]))
    selected = np.asarray(selected, dtype=np.int64)
    last = selected[np.asarray(rows["offset"])[selected] == -1]
    first = selected[np.asarray(rows["offset"])[selected] == 0]
    last = last[np.argsort(np.asarray(rows["pair_id"])[last], kind="stable")]
    first = first[np.argsort(np.asarray(rows["pair_id"])[first], kind="stable")]
    ids = np.asarray(rows["pair_id"])
    if len(last) != len(first) or not np.array_equal(ids[last], ids[first]) or len(np.unique(ids[last])) != len(last):
        raise ValueError("each selected pair needs exactly one -1 and one 0 row")
    return last, first


def features(rows, name, indices):
    base = "h" if name in ("h_shuffled", "h_mlp64") else name
    if base == "logits":
        return np.asarray(rows[base][indices, rows["job"][indices]], dtype=np.float64)
    return np.asarray(rows[base][indices], dtype=np.float64)


def pairflip_targets(rows, indices, seed, slot):
    indices = np.asarray(indices)
    pairs = np.unique(rows["pair_id"][indices])
    rng = np.random.default_rng(928000 + 100 * seed + slot)
    flipped = np.sort(rng.choice(pairs, size=len(pairs) // 2, replace=False))
    y = rows["legal"][indices].astype(np.int64)
    y = np.logical_xor(y, np.isin(rows["pair_id"][indices], flipped)).astype(np.int64)
    return y, flipped


def fit_readout(x, y, task, nonlinear=False, random_state=0):
    """The sole fitting entry point: no validation or test argument exists."""
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    if task == "remaining":
        estimator = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
    elif nonlinear:
        estimator = MLPClassifier(hidden_layer_sizes=(64,), activation="tanh", solver="adam", alpha=1e-4,
                                  batch_size=256, learning_rate_init=1e-3, max_iter=400, early_stopping=False,
                                  n_iter_no_change=401, tol=0.0, shuffle=True, random_state=random_state)
    else:
        estimator = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=2000,
                                       tol=1e-8, fit_intercept=True, class_weight=None)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit(z, y)
    arrays = dict(mean=scaler.mean_, scale=scaler.scale_, train_targets=np.asarray(y))
    if nonlinear:
        arrays.update(w0=estimator.coefs_[0], b0=estimator.intercepts_[0],
                      w1=estimator.coefs_[1], b1=estimator.intercepts_[1], classes=estimator.classes_,
                      loss_curve=np.asarray(estimator.loss_curve_))
    else:
        arrays.update(coef=np.asarray(estimator.coef_), intercept=np.asarray(estimator.intercept_))
        if task != "remaining":
            arrays["classes"] = estimator.classes_
    iterations = getattr(estimator, "n_iter_", None)
    if iterations is not None:
        iterations = np.asarray(iterations).tolist()
    info = dict(sklearn_version=sklearn.__version__, estimator=type(estimator).__name__,
                task=task, nonlinear=nonlinear, train_rows=len(y), feature_count=x.shape[1],
                n_iter=iterations, convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in captured),
                warnings=[str(w.message) for w in captured], training_score=float(estimator.score(z, y)),
                params=estimator.get_params(), feature_order="native source column order")
    return arrays, info


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    exp = np.exp(value[~positive])
    result[~positive] = exp / (1 + exp)
    return result


def numpy_predict(model, x, task):
    """Reproduce exported sklearn scores using only NumPy and saved arrays."""
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    if "w0" in model:
        decision = np.tanh(z @ model["w0"] + model["b0"]) @ model["w1"] + model["b1"]
    else:
        decision = z @ np.asarray(model["coef"]).T + model["intercept"]
    if task == "remaining":
        return np.asarray(decision).reshape(-1)
    classes = np.asarray(model["classes"], dtype=np.int64)
    if np.asarray(decision).ndim == 1 or np.asarray(decision).shape[1] == 1:
        p = sigmoid(np.asarray(decision).reshape(-1))
        native = np.column_stack((1 - p, p))
    else:
        shifted = decision - np.max(decision, axis=1, keepdims=True)
        native = np.exp(shifted)
        native /= native.sum(axis=1, keepdims=True)
    if task == "timing":
        return native[:, int(np.flatnonzero(classes == 1)[0])]
    result = np.zeros((len(x), 8), dtype=np.float64)
    result[:, classes] = native
    return result


def model_stem(seed, arm, task, feature, slot):
    return OUT / f"seed_{seed:03d}" / arm / "models" / f"{task}__{feature}__slot{slot}"


def task_features(task):
    return {"timing": TIMING_FEATURES, "content": CONTENT_FEATURES, "remaining": REMAINING_FEATURES}[task]


def tasks():
    return ("timing", "content", "remaining")


def fit_seed(seed):
    verify_protocol()
    if (OUT / "freeze.json").exists():
        raise ValueError("all models already frozen")
    records = []
    for arm in ARMS:
        path = source_path(seed, "train", arm)
        rows = load_samples(path)
        pins = input_pins(path)
        boundary = boundary_indices(rows)
        for task in tasks():
            selected = np.arange(len(rows["pair_id"])) if task == "remaining" else boundary
            for feature in task_features(task):
                for slot in (0, 1):
                    indices = selected[rows["job"][selected] == slot]
                    if task == "timing":
                        y = rows["legal"][indices].astype(np.int64)
                    elif task == "content":
                        y = rows["target"][indices].astype(np.int64) - 3
                    else:
                        y = rows["remaining_ticks"][indices].astype(np.float64)
                    flips = np.empty(0, dtype=np.int64)
                    if feature == "h_shuffled":
                        y, flips = pairflip_targets(rows, indices, seed, slot)
                    arrays, info = fit_readout(features(rows, feature, indices), y, task,
                                              nonlinear=feature == "h_mlp64", random_state=910000 + 100 * seed + slot)
                    arrays.update(train_row_indices=indices, flipped_train_pair_ids=flips)
                    stem = model_stem(seed, arm, task, feature, slot)
                    stem.parent.mkdir(parents=True, exist_ok=True)
                    if stem.with_suffix(".npz").exists():
                        raise FileExistsError(stem)
                    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
                    info.update(seed=seed, arm=arm, feature=feature, slot=slot, source_pins=pins,
                                protocol_sha256=sha(OUT / "protocol.json"), artifact_sha256=sha(stem.with_suffix(".npz")))
                    write_json(stem.with_suffix(".json"), info)
                    records.append(dict(task=task, feature=feature, slot=slot, arm=arm,
                                        convergence_warning=info["convergence_warning"], n_iter=info["n_iter"],
                                        artifact=relative(stem.with_suffix(".npz"))))
                    print(json.dumps(dict(event="fit", seed=seed, **records[-1])), flush=True)
        del rows
    write_json(OUT / f"seed_{seed:03d}" / "fit.json", dict(seed=seed, records=records))


def freeze():
    """Pin all readouts and inputs before allowing held-out evaluation."""
    protocol = verify_protocol()
    pins = {relative(OUT / "protocol.json"): sha(OUT / "protocol.json")}
    for seed in protocol["seeds"]:
        fit_file = OUT / f"seed_{seed:03d}" / "fit.json"
        pins[relative(fit_file)] = sha(fit_file)
        for arm in ARMS:
            for task in tasks():
                for feature in task_features(task):
                    for slot in (0, 1):
                        stem = model_stem(seed, arm, task, feature, slot)
                        info = read(stem.with_suffix(".json"))
                        check_pins(info["source_pins"])
                        if info["artifact_sha256"] != sha(stem.with_suffix(".npz")):
                            raise ValueError("coefficient hash mismatch")
                        for suffix in (".npz", ".json"):
                            artifact = stem.with_suffix(suffix)
                            pins[relative(artifact)] = sha(artifact)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), pins=pins,
                  policy="No validation/test/own-test result has been read by this pipeline; no further fitting or artifact mutation permitted. Held-out evaluation is diagnostic only.")
    write_json(OUT / "freeze.json", result)
    return result


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def timing_metrics(rows, probability):
    """Pair-aware boundary counts; order-independent and with explicit ties."""
    last, first = pair_indices(rows)
    legal = rows["legal"].astype(bool)
    pred = np.asarray(probability) >= 0.5
    correct = pred == legal
    both = correct[last] & correct[first]
    delta = np.asarray(probability)[first] - np.asarray(probability)[last]
    wins, ties = int((delta > 0).sum()), int((delta == 0).sum())
    n, pairs = len(legal), len(last)
    return dict(rows=n, pairs=pairs, correct_rows=int(correct.sum()), row_accuracy=fraction(correct.sum(), n),
                both_correct=int(both.sum()), both_accuracy=fraction(both.sum(), pairs),
                rank_wins=wins, rank_ties=ties, rank_strict_accuracy=fraction(wins, pairs),
                rank_accuracy=fraction(wins + 0.5 * ties, pairs),
                false_early=int(pred[last].sum()), last_illegal_rows=pairs, false_early_rate=fraction(pred[last].sum(), pairs),
                missed_first_legal=int((~pred[first]).sum()), first_legal_rows=pairs,
                missed_first_legal_rate=fraction((~pred[first]).sum(), pairs))


def content_metrics(rows, probability):
    correct = np.asarray(probability).argmax(axis=1) == rows["target"] - 3
    last, first = pair_indices(rows)
    return dict(rows=len(correct), pairs=len(last), correct_rows=int(correct.sum()),
                row_accuracy=fraction(correct.sum(), len(correct)), both_correct=int((correct[last] & correct[first]).sum()),
                both_accuracy=fraction((correct[last] & correct[first]).sum(), len(last)),
                last_correct=int(correct[last].sum()), first_correct=int(correct[first].sum()),
                first_accuracy=fraction(correct[first].sum(), len(first)),
                target_class_counts=np.bincount(rows["target"].astype(int) - 3, minlength=8).tolist())


def remaining_metrics(rows, prediction):
    prediction = np.asarray(prediction)
    error = np.abs(prediction - rows["remaining_ticks"])
    selected = np.flatnonzero(np.isin(rows["offset"], BOUNDARY))
    boundary_rows = {key: value[selected] for key, value in rows.items()}
    last, first = pair_indices(boundary_rows)
    # A first legal observation can occur fractionally after ready. Zero is legal.
    boundary_prediction = prediction[selected]
    signs = boundary_prediction <= 0
    correct = signs == boundary_rows["legal"].astype(bool)
    delta = boundary_prediction[last] - boundary_prediction[first]
    wins, ties = int((delta > 0).sum()), int((delta == 0).sum())
    by_offset = {str(int(offset)): dict(rows=int((rows["offset"] == offset).sum()),
                 mae_ticks=float(error[rows["offset"] == offset].mean())) for offset in np.unique(rows["offset"])}
    return dict(rows=len(error), mae_ticks=float(error.mean()) if len(error) else None, by_offset=by_offset,
                boundary_rows=len(selected), boundary_pairs=len(last), boundary_correct_rows=int(correct.sum()),
                boundary_sign_accuracy=fraction(correct.sum(), len(correct)),
                boundary_both_correct=int((correct[last] & correct[first]).sum()),
                boundary_both_accuracy=fraction((correct[last] & correct[first]).sum(), len(last)),
                boundary_rank_wins=wins, boundary_rank_ties=ties, boundary_rank_strict_accuracy=fraction(wins, len(last)),
                boundary_rank_accuracy=fraction(wins + 0.5 * ties, len(last)),
                boundary_false_early=int(signs[last].sum()), boundary_missed_first_legal=int((~signs[first]).sum()))


def strata(rows):
    n = len(rows["pair_id"])
    yield "all", np.ones(n, dtype=bool)
    for slot in (0, 1):
        yield f"slot_{slot}", rows["job"] == slot
    for family in (0, 1):
        name = "AB"[family]
        yield f"family_{name}", rows["family"] == family
        for conflict in (False, True):
            yield f"family_{name}_{'conflict' if conflict else 'shared'}", (rows["family"] == family) & (rows["conflict"] == conflict)
    for conflict in (False, True):
        yield "conflict" if conflict else "shared", rows["conflict"] == conflict
    for raw in np.unique(rows["raw"]):
        yield f"raw_{raw}", rows["raw"] == raw
    yield "duration_8_12", (rows["duration"] >= 8) & (rows["duration"] < 12)
    yield "duration_12_16", (rows["duration"] >= 12) & (rows["duration"] <= 16)


def summarize(task, rows, prediction):
    scorer = {"timing": timing_metrics, "content": content_metrics, "remaining": remaining_metrics}[task]
    result = {}
    for label, mask in strata(rows):
        if mask.any():
            subset = {key: value[mask] for key, value in rows.items()}
            result[label] = scorer(subset, prediction[mask])
    return result


def evaluate_dataset(seed, arm, split, mode):
    path = source_path(seed, split, arm, mode)
    rows = load_samples(path)
    pins = input_pins(path)
    records = []
    for task in tasks():
        indices = np.arange(len(rows["pair_id"])) if task == "remaining" else boundary_indices(rows)
        selected = {key: rows[key][indices] for key in META_KEYS}
        arrays = dict(row_indices=indices, **selected)
        metrics = {}
        for feature in task_features(task):
            prediction = np.empty((len(indices), 8), dtype=np.float64) if task == "content" else np.empty(len(indices), dtype=np.float64)
            model_pins = {}
            for slot in (0, 1):
                positions = np.flatnonzero(rows["job"][indices] == slot)
                stem = model_stem(seed, arm, task, feature, slot)
                model_path = stem.with_suffix(".npz")
                with np.load(model_path, allow_pickle=False) as archive:
                    model = dict(archive)
                prediction[positions] = numpy_predict(model, features(rows, feature, indices[positions]), task)
                model_pins[relative(model_path)] = sha(model_path)
            arrays[feature] = prediction
            metrics[feature] = summarize(task, selected, prediction)
            records.append(dict(task=task, feature=feature, source_pins=pins, model_pins=model_pins))
        if task == "content":
            native = rows["logits"][indices, rows["job"][indices], 3:11]
            arrays["original_conditional_choice"] = native.argmax(axis=1)
            original = np.eye(8)[native.argmax(axis=1)]
            metrics["original_conditional_argmax"] = summarize(task, selected, original)
        folder = OUT / f"seed_{seed:03d}" / arm / "predictions"
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"{mode}__{split}__{task}"
        if stem.with_suffix(".npz").exists():
            raise FileExistsError(stem)
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        write_json(stem.with_suffix(".json"), dict(seed=seed, arm=arm, split=split, mode=mode, task=task,
                   source_pins=pins, predictions_sha256=sha(stem.with_suffix(".npz")),
                   freeze_sha256=sha(OUT / "freeze.json"), metrics=metrics,
                   interpretation="offline readout predictions only; no actions or completion outcomes generated"))
    return records


def evaluate_seed(seed):
    verify_protocol()
    frozen = read(OUT / "freeze.json")
    check_pins(frozen["pins"])
    records = []
    for arm in ARMS:
        for split in ("validation", "test"):
            records += evaluate_dataset(seed, arm, split, "teacher")
        if arm != "random":
            records += evaluate_dataset(seed, arm, "test", "own")
        print(json.dumps(dict(event="evaluated", seed=seed, arm=arm)), flush=True)
    write_json(OUT / f"seed_{seed:03d}" / "evaluation.json", dict(seed=seed, records=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "fit", "freeze", "evaluate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command in ("fit", "evaluate") and args.seed is None:
        parser.error("--seed is required for fit/evaluate")
    if args.command == "declare":
        declare()
    elif args.command == "fit":
        fit_seed(args.seed)
    elif args.command == "freeze":
        freeze()
    else:
        evaluate_seed(args.seed)


if __name__ == "__main__":
    main()
