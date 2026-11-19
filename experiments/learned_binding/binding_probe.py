"""Post-hoc linear recovery of answered context/raw at successful receipts.

Analysis heads are fitted once with fixed ridge lambda=1, never installed in a
policy.  Train: the original 16 training streams.  Test: already-used actual
teacher-support observations, not a new independent validation dataset.  The
legacy raw-stream GRU feature is a representation reference, not the native
policy's actual translated-input hidden state.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.associative_state.bridge import PublicReceiptBinder
from experiments.learned_binding.common import (
    ROOT, OUT, SEEDS, load_native, new_native, native_checkpoint, parameter_hash, sha,
)


FEATURES = ("public28", "trained_z32_public28", "untrained_z32_public28", "legacy_raw_h192_public28")
RIDGE = 1.
N_CLASSES = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None:
        require(digest == expected, f"artifact SHA changed: {path}")
    return dict(path=str(path.relative_to(ROOT)).replace("\\", "/"), sha256=digest)


def contexts(seed):
    return SEEDS[seed]["history_cue"], SEEDS[seed]["B_cue"]


def receipt_rows(obs, context_pair):
    """Bind the preceding FINISH before processing any simultaneous new cue.

    ``PublicReceiptBinder.consume`` emits the answered key before replacing its
    cache with this tick's NEW/REPLACE.  A simultaneous cue in the answered slot
    is subsequently excluded, so current context/raw cannot reveal the label.
    """
    binder = PublicReceiptBinder(obs.shape[1], n_jobs=2, contexts=context_pair)
    coordinates, labels = [], []
    counts = dict(public_successful_finishes=0, binder_matched_receipts=0,
                  excluded_same_slot_command=0, selected_receipts=0)
    for tick, frame in enumerate(obs):
        public = frame.reshape(frame.shape[0], 2, 28)
        previous_finish = public[..., 14:22].sum(-1) > .5
        successful = previous_finish & (public[..., 22] > .5) & (public[..., 23] < .5)
        command = (public[..., 0] > .5) | (public[..., 1] > .5)
        answered = binder.consume(frame)
        counts["public_successful_finishes"] += int(successful.sum())
        for job, (key, _value, matched) in enumerate(answered):
            counts["binder_matched_receipts"] += int(matched.sum())
            counts["excluded_same_slot_command"] += int((matched & command[:, job]).sum())
            lanes = (matched & ~command[:, job]).nonzero(as_tuple=True)[0]
            if lanes.numel():
                coordinates.append(torch.stack((torch.full_like(lanes, tick), lanes,
                                                torch.full_like(lanes, job)), 1))
                labels.append(key[lanes])
    require(bool(coordinates), "no eligible successful FINISH receipts in stream")
    coordinates, labels = torch.cat(coordinates), torch.cat(labels)
    counts["selected_receipts"] = len(labels)
    require(bool(((labels >= 0) & (labels < 8)).all()), "analysis key outside 0..7")
    return coordinates, labels.numpy(), counts


@torch.no_grad()
def extract(obs, trained, untrained, context_pair, expected_final_z=None):
    """Select receipt rows first; retain only selected feature vectors afterward."""
    require(obs.dtype == torch.float32 and obs.ndim == 3 and obs.shape[-1] == 56, "FP32 public56 input required")
    indices, labels, counts = receipt_rows(obs, context_pair)
    tick, lane, job = indices.unbind(1)
    public = obs.reshape(len(obs), obs.shape[1], 2, 28)[tick, lane, job]
    result = {"public28": public.numpy().copy()}
    batch = obs.shape[1]
    encoder_input = obs.reshape(len(obs), batch * 2, 28)
    final_error = None
    for name, model in (("trained_z32_public28", trained), ("untrained_z32_public28", untrained)):
        z, final = model.encoder(encoder_input, obs.new_zeros(1, batch * 2, 32))
        if name == "trained_z32_public28" and expected_final_z is not None:
            actual_final = final.reshape(batch, 2, 32)
            final_error = float((actual_final - expected_final_z).abs().max())
            torch.testing.assert_close(actual_final, expected_final_z, rtol=2e-5, atol=2e-6)
        selected = z.reshape(len(obs), batch, 2, 32)[tick, lane, job]
        result[name] = torch.cat((selected, public), 1).numpy().copy()
        del z, final, selected
    # This independent read-only reference consumes ORIGINAL RAW observations.
    # It is intentionally not called the model's actual executor hidden state.
    raw_h, _ = trained.executor.core(obs, obs.new_zeros(1, batch, 192))
    result["legacy_raw_h192_public28"] = torch.cat((raw_h[tick, lane], public), 1).numpy().copy()
    require(all(len(values) == len(labels) for values in result.values()), "feature/label alignment")
    return result, labels, counts, final_error


def fit_ridge(features, labels):
    x = np.asarray(features, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0, ddof=0)
    scale = np.where(scale < 1e-12, 1., scale)
    standardized = (x - mean) / scale
    design = np.column_stack((standardized, np.ones(len(x))))
    target = np.eye(N_CLASSES, dtype=np.float64)[labels]
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE
    penalty[-1, -1] = 0.
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    require(np.isfinite(weights).all(), "nonfinite ridge solution")
    return mean, scale, weights


def predict(features, fitted):
    mean, scale, weights = fitted
    x = (np.asarray(features, dtype=np.float64) - mean) / scale
    return (np.column_stack((x, np.ones(len(x)))) @ weights).argmax(1)


def classification(labels, predictions):
    confusion = np.bincount(N_CLASSES * labels + predictions,
                            minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    per_class = []
    for key in range(N_CLASSES):
        count, correct = int(confusion[key].sum()), int(confusion[key, key])
        per_class.append(dict(key=key, context_index=key // 4, raw=key % 4,
                              count=count, correct=correct,
                              accuracy=correct / count if count else None))
    return dict(n=len(labels), accuracy=float((labels == predictions).mean()),
                confusion_true_rows_predicted_columns=confusion.tolist(), per_class=per_class)


def run_seed(seed):
    started = time.perf_counter()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    trained, untrained = load_native(seed, "cpu").eval(), new_native(seed, "cpu").eval()
    original_hashes = dict(trained=parameter_hash(trained), untrained=parameter_hash(untrained),
                           legacy_executor=parameter_hash(trained.executor))
    freeze = json.loads((OUT / "data-freeze.json").read_text(encoding="utf-8"))["streams"]
    training_freeze = json.loads((OUT / "training-freeze.json").read_text(encoding="utf-8"))["checkpoints"]
    checkpoint = native_checkpoint(seed)
    checkpoint_ref = reference(checkpoint, training_freeze[str(checkpoint.relative_to(ROOT))])
    train_features, train_labels, streams = {name: [] for name in FEATURES}, [], []
    for index in range(16):
        row = next(r for r in freeze if r["seed"] == seed and r["stream"] == index)
        path = ROOT / row["path"]
        source = reference(path, row["sha256"])
        data = torch.load(path, map_location="cpu", weights_only=False)
        features, labels, counts, _ = extract(data["obs"], trained, untrained, contexts(seed))
        train_labels.append(labels)
        for name in FEATURES:
            train_features[name].append(features[name])
        streams.append(dict(stream=index, source=source, selection=counts,
                            class_counts=np.bincount(labels, minlength=8).tolist()))
        del data, features, labels
    train_labels = np.concatenate(train_labels)
    folder = OUT / f"seed_{seed:03d}/behavior/learned"
    prefix = folder / "teacher_support"
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    require(report["teacher"] is True, "test observations must be actual teacher support")
    report_ref = reference(prefix.with_suffix(".json"))
    array_ref = reference(prefix.with_suffix(".npz"), report["arrays_sha256"])
    state_ref = reference(prefix.with_suffix(".pt"), report["states_sha256"])
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        test_obs = torch.from_numpy(archive["obs"].copy())
    states = torch.load(prefix.with_suffix(".pt"), map_location="cpu", weights_only=False)
    require(states["start"]["stream_id"] == 100, "expected the already-used stream100 support")
    require(int(states["start"]["state"]["z"].count_nonzero()) == 0, "teacher encoder must start at zero")
    for name, value in trained.state_dict().items():
        require(torch.equal(value, states["start"]["model"][name]), "teacher endpoint parameters differ: " + name)
    test_features, test_labels, test_counts, final_error = extract(
        test_obs, trained, untrained, contexts(seed), states["final"]["state"]["z"])
    heads = {}
    majority = int(np.bincount(train_labels, minlength=8).argmax())
    for name in FEATURES:
        train = np.concatenate(train_features[name])
        fitted = fit_ridge(train, train_labels)
        heads[name] = dict(dimensions=train.shape[1],
            train=classification(train_labels, predict(train, fitted)),
            test=classification(test_labels, predict(test_features[name], fitted)))
        del train, fitted
    final_hashes = dict(trained=parameter_hash(trained), untrained=parameter_hash(untrained),
                        legacy_executor=parameter_hash(trained.executor))
    require(original_hashes == final_hashes, "policy parameters changed during analysis")
    require(all(p.grad is None for model in (trained, untrained) for p in model.parameters()), "policy gradients were created")
    return dict(seed=seed, contexts=list(contexts(seed)), source_checkpoint=checkpoint_ref,
        model_parameter_hashes=original_hashes, policy_parameters_unchanged=True,
        policy_gradients_absent=True, train_streams=streams, features=heads,
        teacher_test_sources=dict(report=report_ref, arrays=array_ref, states=state_ref),
        teacher_test_selection=test_counts,
        teacher_encoder_fused_vs_actual_step_final_z_maxabs=final_error,
        train_majority_key=majority, train_majority_baseline_test_accuracy=float((test_labels == majority).mean()),
        seconds=time.perf_counter() - started)


def self_test():
    torch.set_num_threads(1)
    obs = torch.zeros(4, 1, 2, 28)
    # Job0 cue=context0/raw2, then a successful FINISH.  The simultaneous new
    # cue is deliberately a different raw/context and must not label receipt.
    obs[0, 0, 0, 0] = obs[0, 0, 0, 3] = obs[0, 0, 0, 7] = obs[0, 0, 0, 24] = 1.
    obs[1, 0, 0, 0] = obs[1, 0, 0, 3] = obs[1, 0, 0, 6] = obs[1, 0, 0, 27] = 1.
    obs[1, 0, 0, 11 + 5] = obs[1, 0, 0, 22] = 1.
    obs[2, 0, 0, 11 + 4] = obs[2, 0, 0, 22] = 1.
    indices, labels, counts = receipt_rows(obs.reshape(4, 1, 56), (0, 3))
    require(labels.tolist() == [5], "receipt-before-new identity/filter")
    require(indices.tolist() == [[2, 0, 0]] and counts["excluded_same_slot_command"] == 1, "same-slot exclusion")
    rng = np.random.default_rng(1637)
    labels = np.tile(np.arange(8), 20)
    x = np.column_stack((np.eye(8)[labels] + rng.normal(0, .01, (len(labels), 8)), np.ones(len(labels))))
    fit = fit_ridge(x, labels)
    require(float((predict(x, fit) == labels).mean()) == 1., "fixed ridge classifier")
    require(classification(labels, predict(x, fit))["confusion_true_rows_predicted_columns"] == (np.eye(8, dtype=int) * 20).tolist(), "confusion counts")
    return dict(self_test="passed", receipt_before_new=True, same_slot_command_excluded=True,
                fixed_ridge_lambda=RIDGE, policy_artifacts_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    output = OUT / "diagnostics/binding-probe.json"
    require(not output.exists(), f"diagnostic already exists: {output}")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for completed in as_completed([pool.submit(run_seed, seed) for seed in range(3)]):
            row = completed.result()
            rows.append(row)
            print(json.dumps(dict(event="binding_probe_seed", seed=row["seed"],
                test_accuracy={name: item["test"]["accuracy"] for name, item in row["features"].items()},
                seconds=row["seconds"])), flush=True)
    rows.sort(key=lambda r: r["seed"])
    result = dict(format="learned-binding-posthoc-receipt-linear-probe-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), posthoc=True,
        scope="Representation diagnosis only; no additional policy training or policy evaluation run.",
        independent_new_validation=False,
        test_disclosure="The same stream100 teacher-support observations were already used in formal teacher-supported scoring. These fitted analysis heads are a post-hoc diagnosis, not new independent validation.",
        labels="Legacy PublicReceiptBinder reconstructs answered context-index*4+raw solely from public history, processing receipt before this tick's cue. Labels enter only the temporary analysis heads and are never written back to policy.",
        selection="Successful matched FINISH receipt rows, excluding NEW/REPLACE in that same slot on the current tick; both jobs remain represented.",
        chance_uniform8=.125,
        ridge=dict(lambda_value=RIDGE, dtype="float64", targets="8-way one-hot", classifier="argmax of eight linear scores",
            standardization="Training-only column mean and population std; std<1e-12 replaced by 1.",
            bias="unpenalized", objective="sum squared residuals + lambda * squared weights (excluding bias)",
            hyperparameter_selection=False, checkpoint_selection=False, train_streams_per_seed=16),
        feature_definitions=dict(public28="Current public features of the answered job only.",
            trained_z32_public28="Actual trained public-only encoder state after consuming the receipt, concatenated with that job's current public28.",
            untrained_z32_public28="Encoder from fixed new_native(seed) initialization, frozen, driven by the same public history, plus public28.",
            legacy_raw_h192_public28="Frozen original executor.core driven by unmodified raw56, zero initial h, plus the answered job public28. Extra representation reference; NOT the main policy's actual translated-executor state."),
        per_seed=rows,
        equal_seed_mean={name: {split + "_accuracy": sum(row["features"][name][split]["accuracy"] for row in rows) / 3
                               for split in ("train", "test")} for name in FEATURES},
        no_policy_writeback=True, sources={name: reference(ROOT / name) for name in (
            "experiments/learned_binding/binding_probe.py", "experiments/learned_binding/model.py",
            "experiments/learned_binding/common.py", "experiments/associative_state/bridge.py",
            "data/learned-binding/v1/protocol.json", "data/learned-binding/v1/data-freeze.json",
            "data/learned-binding/v1/training-freeze.json")})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size, sha256=sha(output),
                         equal_seed_mean=result["equal_seed_mean"])), flush=True)


if __name__ == "__main__":
    main()
