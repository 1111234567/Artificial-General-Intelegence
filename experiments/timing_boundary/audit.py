"""Independent stream, recurrent-state, and fitted-probe artifact audit.

No producer cohort, fitting, prediction, or metric functions are imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/timing-boundary/audit-v1"
DATA = ROOT / "data/timing-boundary/v1"
PROBES = ROOT / "data/timing-boundary/probe-v1"
PLAN = OUT / "protocol.json"
ARMS = ("midpoint", "first", "interior", "random")
SPLITS = ("train", "validation", "test")
OFFSETS = (-8, -4, -2, -1, 0, 1, 2)
WARM = 512
EPS = 1e-6


def require(value, label):
    if not value:
        raise AssertionError(label)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def exact(actual, expected, label):
    np.testing.assert_array_equal(actual, expected, err_msg=label)


def close(actual, expected, label, atol=1e-9, rtol=1e-8):
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), label + ": nonfinite")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def receipts(facts):
    """Apply native action acceptance equations directly to pre-action facts."""
    action = facts["actions"]
    version = np.where((action == 1) | (action == 2), action - 1, (action - 3) // 4)
    begin = (action == 1) | (action == 2)
    finish = (action >= 3) & (action < 11)
    same = version == facts["version"]
    accepted_begin = begin & facts["active"] & same & ~facts["started"]
    accepted_finish = (finish & facts["active"] & same & facts["started"]
                       & (((action - 3) % 4) == facts["payload"])
                       & (facts["now"][:, None, None] >= facts["ready"] - EPS)
                       & (facts["now"][:, None, None] <= facts["deadline"] + EPS))
    accepted = accepted_begin | accepted_finish
    rejected = (action != 0) & ~accepted
    return accepted.astype(np.int64) - rejected.astype(np.int64), accepted_begin, accepted_finish


def boundary_metrics(pair_ids, offsets, targets, scores, predictions=None):
    """Rebuild binary endpoint metrics, including complete-pair denominator."""
    pair_ids, offsets, targets, scores = map(np.asarray, (pair_ids, offsets, targets, scores))
    require(len(pair_ids) == len(offsets) == len(targets) == len(scores), "metric row lengths")
    require(set(np.unique(offsets)).issubset({-1, 0}), "timing offsets must be endpoints")
    exact(targets, (offsets == 0).astype(np.int64), "real timing labels")
    predictions = (scores >= 0).astype(np.int64) if predictions is None else np.asarray(predictions)
    correct = predictions == targets
    groups = defaultdict(dict)
    for index, (identity, offset) in enumerate(zip(pair_ids, offsets)):
        require(int(offset) not in groups[int(identity)], "duplicate endpoint in metric pair")
        groups[int(identity)][int(offset)] = index
    require(all(set(group) == {-1, 0} for group in groups.values()), "incomplete timing metric pair")
    both = ranking_twice = ties = ranking_wins = 0
    for group in groups.values():
        left, right = group[-1], group[0]
        both += int(correct[left] and correct[right])
        ranking_twice += 2 * int(scores[right] > scores[left]) + int(scores[right] == scores[left])
        ranking_wins += int(scores[right] > scores[left])
        ties += int(scores[right] == scores[left])
    rows, pairs = len(pair_ids), len(groups)
    return dict(rows=rows, correct=int(correct.sum()), accuracy=float(correct.mean()) if rows else None,
                pairs=pairs, both_correct=both, both_accuracy=both / pairs if pairs else None,
                ranking_wins=ranking_wins, ranking_strict_accuracy=ranking_wins / pairs if pairs else None,
                ranking_credit_twice=ranking_twice, ranking_accuracy=ranking_twice / (2 * pairs) if pairs else None,
                ranking_ties=ties)


def normalized_features(features, fitted):
    features = np.asarray(features, dtype=np.float64)
    mean, scale = fitted["mean"], fitted["scale"]
    require(features.shape[1:] == mean.shape == scale.shape, "normalizer dimensions")
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), "normalizer validity")
    return (features - mean) / scale


def affine_prediction(features, fitted):
    """Independent saved coefficient replay; columns are outputs."""
    values = normalized_features(features, fitted)
    if "w0" in fitted:
        return np.tanh(values @ fitted["w0"] + fitted["b0"]) @ fitted["w1"] + fitted["b1"]
    coefficient = fitted["coef"]
    return values @ (coefficient.T if coefficient.ndim == 2 else coefficient) + fitted["intercept"]


def predict(features, fitted, task):
    decision = affine_prediction(features, fitted)
    require(np.isfinite(decision).all(), "nonfinite coefficient prediction")
    if task == "remaining":
        return decision.reshape(-1)
    if decision.ndim == 1 or decision.shape[1] == 1:
        # Stable exp formulation differs from the fitting program's sigmoid.
        decision = decision.reshape(-1)
        p = np.exp(-np.logaddexp(0., -decision))
        probability = np.column_stack((1 - p, p))
    else:
        shifted = decision - decision.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probability = exponent / exponent.sum(axis=1, keepdims=True)
    classes = fitted["classes"].astype(int)
    if task == "timing":
        require(set(classes) == {0, 1}, "timing classes")
        return probability[:, np.flatnonzero(classes == 1)[0]]
    result = np.zeros((len(features), 8), dtype=np.float64)
    result[:, classes] = probability
    return result


def pair_positions(rows):
    groups = defaultdict(dict)
    for index, (identity, offset) in enumerate(zip(rows["pair_id"], rows["offset"])):
        if int(offset) in (-1, 0):
            require(int(offset) not in groups[int(identity)], "duplicate metric pair")
            groups[int(identity)][int(offset)] = index
    require(all(set(group) == {-1, 0} for group in groups.values()), "incomplete metric pair")
    keys = sorted(groups)
    return np.array([groups[key][-1] for key in keys], dtype=int), np.array([groups[key][0] for key in keys], dtype=int)


def metric_strata(rows):
    yield "all", np.ones(len(rows["job"]), dtype=bool)
    for slot in (0, 1):
        yield f"slot_{slot}", rows["job"] == slot
    for family, label in enumerate("AB"):
        yield f"family_{label}", rows["family"] == family
        for conflict, name in ((False, "shared"), (True, "conflict")):
            yield f"family_{label}_{name}", (rows["family"] == family) & (rows["conflict"] == conflict)
    for conflict, name in ((False, "shared"), (True, "conflict")):
        yield name, rows["conflict"] == conflict
    for raw in np.unique(rows["raw"]):
        yield f"raw_{raw}", rows["raw"] == raw
    yield "duration_8_12", (rows["duration"] >= 8) & (rows["duration"] < 12)
    yield "duration_12_16", (rows["duration"] >= 12) & (rows["duration"] <= 16)


def rate(n, d):
    return n / d if d else None


def metrics(task, rows, prediction):
    """Independent arithmetic producing the frozen exported metric names."""
    last, first = pair_positions(rows)
    count, pairs = len(rows["job"]), len(last)
    if task == "timing":
        predicted = prediction >= .5
        correct = predicted == rows["legal"].astype(bool)
        delta = prediction[first] - prediction[last]
        wins, ties = int((delta > 0).sum()), int((delta == 0).sum())
        return dict(rows=count, pairs=pairs, correct_rows=int(correct.sum()), row_accuracy=rate(int(correct.sum()), count),
                    both_correct=int((correct[last] & correct[first]).sum()), both_accuracy=rate(int((correct[last] & correct[first]).sum()), pairs),
                    rank_wins=wins, rank_ties=ties, rank_accuracy=rate(wins + .5 * ties, pairs), rank_strict_accuracy=rate(wins, pairs),
                    false_early=int(predicted[last].sum()), last_illegal_rows=pairs, false_early_rate=rate(int(predicted[last].sum()), pairs),
                    missed_first_legal=int((~predicted[first]).sum()), first_legal_rows=pairs, missed_first_legal_rate=rate(int((~predicted[first]).sum()), pairs))
    if task == "content":
        correct = prediction.argmax(axis=1) == rows["target"] - 3
        return dict(rows=count, pairs=pairs, correct_rows=int(correct.sum()), row_accuracy=rate(int(correct.sum()), count),
                    both_correct=int((correct[last] & correct[first]).sum()), both_accuracy=rate(int((correct[last] & correct[first]).sum()), pairs),
                    last_correct=int(correct[last].sum()), first_correct=int(correct[first].sum()), first_accuracy=rate(int(correct[first].sum()), pairs),
                    target_class_counts=np.bincount(rows["target"].astype(int) - 3, minlength=8).tolist())
    error = prediction - rows["remaining_ticks"]
    signs = prediction <= 0
    correct = signs == rows["legal"].astype(bool)
    delta = prediction[last] - prediction[first]
    wins, ties = int((delta > 0).sum()), int((delta == 0).sum())
    result = dict(rows=count, mae_ticks=float(np.abs(error).mean()) if count else None,
                  boundary_rows=2 * pairs, boundary_pairs=pairs, boundary_correct_rows=int(correct[last].sum() + correct[first].sum()),
                  boundary_sign_accuracy=rate(int(correct[last].sum() + correct[first].sum()), 2 * pairs),
                  boundary_both_correct=int((correct[last] & correct[first]).sum()), boundary_both_accuracy=rate(int((correct[last] & correct[first]).sum()), pairs),
                  boundary_rank_wins=wins, boundary_rank_ties=ties, boundary_rank_accuracy=rate(wins + .5 * ties, pairs),
                  boundary_rank_strict_accuracy=rate(wins, pairs), boundary_false_early=int(signs[last].sum()), boundary_missed_first_legal=int((~signs[first]).sum()))
    result["by_offset"] = {str(offset): dict(rows=int((rows["offset"] == offset).sum()),
                             mae_ticks=float(np.abs(error[rows["offset"] == offset]).mean()) if np.any(rows["offset"] == offset) else None)
                           for offset in OFFSETS}
    return result


def compare_metrics(saved, rebuilt, label):
    """Every saved value must match; independent extra diagnostics are allowed."""
    if isinstance(saved, dict):
        require(isinstance(rebuilt, dict), label + ": metric type")
        for key, value in saved.items():
            require(key in rebuilt, label + ": unaudited metric " + key)
            compare_metrics(value, rebuilt[key], label + "/" + key)
    elif isinstance(saved, list):
        require(saved == rebuilt, label + ": metric list")
    elif saved is None:
        require(rebuilt is None, label + ": undefined metric")
    elif isinstance(saved, (int, bool)):
        require(saved == rebuilt, label + ": integer metric")
    elif isinstance(saved, float):
        close(np.asarray(saved), np.asarray(rebuilt), label)
    else:
        require(saved == rebuilt, label + ": metric value")


def feature_matrix(rows, name):
    if name in ("h", "h_shuffled", "h_mlp64"):
        return rows["h"].astype(np.float64)
    if name == "obs":
        return rows["obs"].astype(np.float64)
    if name in ("logits", "targetslot_logits11"):
        return rows["logits"][np.arange(len(rows["job"])), rows["job"]].astype(np.float64)
    raise AssertionError("unknown declared feature " + name)


def rebuild_rows(facts, seed):
    """Rebuild the selected cohort from raw events, not saved sample metadata.

    Mature events are matched to their original command before reading retained
    current fields, because a maturity pulse may coincide with a new command.
    """
    from experiments.experience_transfer.train_transfer import SEEDS

    specification = SEEDS[int(seed)]
    expected, cohort, counts = {}, [], Counter()
    origins = {}
    command = (facts["event_kind"] == 1) | (facts["event_kind"] == 2)
    for tick, lane, job in zip(*np.where(command)):
        identity = (int(lane), int(job), int(facts["event_id"][tick, lane, job]))
        require(identity not in origins, "duplicate command identity")
        origins[identity] = int(tick)
    maturity = {}
    for tick, lane, job in zip(*np.where(facts["mature"])):
        identity = (int(lane), int(job), int(facts["mature_id"][tick, lane, job]))
        require(identity in origins and identity not in maturity, "maturity without unique command")
        origin = origins[identity]
        require(origin < tick, "maturity precedes issue")
        require(bool(facts["event_eligible"][origin, lane, job]), "ineligible request matured")
        require(facts["now"][tick] > facts["deadline"][origin, lane, job] + EPS, "premature maturity")
        close(np.asarray(facts["mature_issued"][tick, lane, job]), np.asarray(facts["now"][origin]), "mature issued")
        close(np.asarray(facts["mature_duration"][tick, lane, job]), np.asarray(facts["duration"][origin, lane, job]), "mature duration")
        exact(facts["mature_family"][tick, lane, job], facts["family"][origin, lane, job], "mature family")
        maturity[identity] = int(tick)
    for (lane, job, rid), origin in sorted(origins.items()):
        if origin < WARM:
            continue
        counts["postwarm_commands"] += 1
        eligible = bool(facts["event_eligible"][origin, lane, job])
        counts["eligible"] += int(eligible)
        if not eligible:
            counts["ineligible"] += 1
            continue
        if (lane, job, rid) not in maturity:
            counts["eligible_immature"] += 1
            continue
        counts["matured"] += 1
        family_cue = int(facts["family"][origin, lane, job])
        require(family_cue in (specification["history_cue"], specification["B_cue"]), "unexpected family")
        family = int(family_cue == specification["B_cue"])
        raw = int(facts["raw"][origin, lane, job])
        target = 3 + 4 * int(facts["version"][origin, lane, job]) + specification["B" if family else "A"][raw]
        require(target == 3 + 4 * facts["version"][origin, lane, job] + facts["payload"][origin, lane, job], "private mapped target")
        same = facts["rid"][:, lane, job] == rid
        started = same & facts["active"][:, lane, job] & facts["started"][:, lane, job]
        process = np.flatnonzero(started & (facts["now"] < facts["ready"][:, lane, job] - EPS))
        legal = np.flatnonzero(started & (facts["now"] >= facts["ready"][:, lane, job] - EPS)
                               & (facts["now"] <= facts["deadline"][:, lane, job] + EPS))
        base = dict(actor=int(facts["actor_ids"][lane]), job=job, rid=rid, origin=origin,
                    mature=maturity[(lane, job, rid)], family=family, raw=raw,
                    conflict=int(raw in specification["conflict_raw"]), target=int(target))
        reason = "no_processing" if not len(process) else "no_legal" if not len(legal) else "nonconsecutive" if int(process[-1]) + 1 != int(legal[0]) else None
        if reason:
            counts[reason] += 1
            cohort.append(dict(**base, included=False, exclusion=reason))
            continue
        first, previous = int(legal[0]), int(process[-1])
        require(first > origin and previous >= origin, "invalid boundary origin")
        for name in ("rid", "version", "payload", "family", "raw", "issued_time", "duration", "ready", "deadline"):
            exact(facts[name][previous, lane, job], facts[name][first, lane, job], "boundary identity " + name)
        require(bool(facts["eligible"][first, lane, job]), "boundary current eligibility")
        counts["pairs"] += 1
        cohort.append(dict(**base, included=True, first_legal=first, last_pre_ready=previous))
        for offset in OFFSETS:
            tick = first + offset
            if tick < origin or tick >= len(facts["now"]) or not started[tick]:
                continue
            key = (base["actor"], job, rid, offset)
            require(key not in expected, "duplicate selected sample")
            expected[key] = dict(**base, offset=offset, tick=tick, lane=lane,
                                 duration=float(facts["duration"][tick, lane, job]),
                                 remaining_ticks=float((facts["ready"][tick, lane, job] - facts["now"][tick]) / .25),
                                 legal=int(facts["now"][tick] >= facts["ready"][tick, lane, job] - EPS))
    require(counts["matured"] == counts["pairs"] + counts["no_processing"] + counts["no_legal"] + counts["nonconsecutive"], "mature cohort partition")
    return expected, cohort, dict(counts)


def verify_sample_rows(rows, facts, seed):
    expected, cohort, counts = rebuild_rows(facts, seed)
    require(len(rows["h"]) == len(expected), "selected row count")
    seen, pair_keys = set(), {}
    for index in range(len(rows["h"])):
        key = tuple(int(rows[name][index]) for name in ("actor", "job", "rid", "offset"))
        require(key in expected and key not in seen, "saved sample identity differs")
        seen.add(key)
        reference = expected[key]
        for name in ("actor", "job", "rid", "offset", "tick", "family", "raw", "conflict", "target", "legal"):
            exact(rows[name][index], reference[name], "selected metadata " + name)
        for name in ("duration", "remaining_ticks"):
            close(rows[name][index], np.asarray(reference[name]), "selected metadata " + name, atol=1e-6)
        exact(rows["obs"][index], facts["obs"][reference["tick"], reference["lane"]], "sample public observation")
        identity = key[:3]
        pair = int(rows["pair_id"][index])
        require(pair_keys.setdefault(identity, pair) == pair, "pair_id changes across offsets")
    require(len(set(pair_keys.values())) == len(pair_keys), "pair_id collision")
    require(len(pair_keys) == counts["pairs"], "pair count mismatch")
    require(np.isfinite(rows["h"]).all() and np.isfinite(rows["logits"]).all(), "nonfinite saved model output")
    return dict(rows=len(expected), **counts), cohort


NATIVE_FIELDS = {
    "rid": "request_id", "active": "active", "started": "started", "ready": "ready_time",
    "deadline": "deadline", "issued_time": "issued_time", "version": "version", "payload": "payload",
    "raw": "raw_payload", "family": "current_family", "eligible": "current_eligible", "duration": "duration",
    "had_error": "had_error", "event_kind": "event_kind", "event_id": "event_request_id",
    "event_eligible": "event_eligible", "event_family": "event_family", "event_raw": "raw_event_payload",
    "event_version": "event_version", "event_duration": "event_duration", "event_window": "event_window",
    "mature": "matured_mask", "mature_id": "matured_request_id", "mature_issued": "matured_issued_at",
    "mature_duration": "matured_duration", "mature_family": "matured_family",
    "mature_done": "matured_completed", "mature_strict": "matured_clean_completed",
}


def verify_native_stream(facts, metadata, data_plan):
    import torch
    from experiments.lifelong.world import ContextWorld
    from experiments.experience_transfer.train_transfer import mapping_scope

    exact(facts["tick"], np.arange(WARM + 4096), "full unbroken native tick range")
    exact(facts["now"], .25 * facts["tick"], "physical clock")
    exact(facts["actor_ids"], data_plan["actors"][metadata["split"]], "declared actors")
    receipt, _, accepted_finish = receipts(facts)
    exact(receipt, facts["receipts"], "independent arithmetic receipts")
    public = facts["obs"].reshape(len(facts["now"]), len(facts["actor_ids"]), 2, 28)
    expected_action = np.eye(11)[facts["actions"][:-1]]
    expected_action *= (facts["actions"][:-1] != 0)[..., None]
    exact(public[1:, ..., 11:22], expected_action, "public preceding action")
    exact(public[1:, ..., 22], receipt[:-1] == 1, "public preceding accepted")
    exact(public[1:, ..., 23], receipt[:-1] == -1, "public preceding rejected")
    require(not public[0, ..., 11:24].any(), "origin has receipt history")
    with mapping_scope(data_plan["mapping"][str(metadata["seed"]) ]), torch.inference_mode():
        world = ContextWorld(**metadata["world_config"])
        for tick in range(len(facts["now"])):
            exact(world.observe().numpy(), facts["obs"][tick], "native public observation")
            require(world.tick == int(facts["tick"][tick]) and world.now == float(facts["now"][tick]), "native clock")
            exact(world.oracle("midpoint").numpy(), facts["teacher"][tick], "native oracle actions")
            for field, attribute in NATIVE_FIELDS.items():
                exact(getattr(world, attribute).numpy(), facts[field][tick], "native private state " + field)
            if metadata["mode"] == "teacher":
                exact(facts["actions"][tick], facts["teacher"][tick], "teacher executed action")
            world.act(torch.from_numpy(facts["actions"][tick]))
            exact(world.last_result.numpy(), facts["receipts"][tick], "native executed receipt")
        require(world.event_hash == metadata["events_sha256"], "native external stream hash")
    native_requests = metadata["requests"]
    require(len(native_requests) == int(np.isin(facts["event_kind"], (1, 2)).sum()), "full command inventory")
    ordinary = strict = mature = 0
    for request in native_requests:
        lane, job, rid, issued = map(int, (request[k] for k in ("lane", "job", "rid", "issued")))
        require(request["actor"] == int(facts["actor_ids"][lane]), "request actor")
        exact(facts["event_id"][issued, lane, job], rid, "request event origin")
        require(request["eligible"] == bool(facts["event_eligible"][issued, lane, job]), "request eligibility")
        matches = np.flatnonzero(facts["mature"][:, lane, job] & (facts["mature_id"][:, lane, job] == rid))
        require(bool(len(matches)) == request["mature"], "request maturation presence")
        require(len(matches) <= 1, "request duplicate maturity")
        require(request["mature_tick"] == (int(matches[0]) if len(matches) else None), "request maturity tick")
        if len(matches):
            mt = int(matches[0])
            require(mt == np.flatnonzero(facts["now"] > request["deadline"] + EPS)[0], "first possible maturity tick")
            identity = facts["rid"][:, lane, job] == rid
            finished = np.flatnonzero(identity & accepted_finish[:, lane, job])
            done = len(finished) > 0
            clean = done and not np.any((receipt[issued:int(finished[0]) + 1, lane, job] == -1)
                                       & facts["active"][issued:int(finished[0]) + 1, lane, job])
            require(request["ordinary"] == done and request["strict"] == bool(clean), "independent ordinary/strict completion")
            if issued >= WARM and request["eligible"]:
                mature += 1; ordinary += int(done); strict += int(clean)
    idle = ~facts["active"][WARM:]
    require(metadata["idle_opportunities"] == int(idle.sum()), "actual idle denominator")
    require(metadata["idle_actions"] == int((idle & (facts["actions"][WARM:] != 0)).sum()), "actual idle actions")
    require(metadata["counts"].get("matured_eligible_afterwarm", 0) == mature, "free mature count")
    require(metadata["counts"].get("ordinary", 0) == ordinary and metadata["counts"].get("strict", 0) == strict, "free completion counts")
    return dict(ticks=len(facts["now"]), requests=len(native_requests), matured=mature,
                ordinary=ordinary, strict=strict, idle_opportunities=int(idle.sum()), idle_actions=metadata["idle_actions"])


def model_state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def verify_recurrent(rows, facts, model_info, mode):
    import torch
    from experiments.learn_remember.gru_train import GRUAgent

    if "path" in model_info:
        checkpoint = ROOT / model_info["path"]
        require(sha(checkpoint) == model_info["checkpoint_sha256"], "original checkpoint binding")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = GRUAgent(**saved["model_config"])
        model.load_state_dict(saved["model"])
    else:
        with torch.random.fork_rng():
            torch.manual_seed(model_info["initialization_seed"])
            model = GRUAgent(width=192)
    model.eval().requires_grad_(False)
    before = model_state_sha(model)
    require(before == model_info["state_sha256"], "frozen GRU parameter hash")
    actor_lane = {int(actor): lane for lane, actor in enumerate(facts["actor_ids"])}
    lane = np.array([actor_lane[int(actor)] for actor in rows["actor"]])
    maximum_h_error = maximum_logit_error = 0.
    vectorized_action_disagreements = 0
    h = model.initial_state(len(facts["actor_ids"]))
    with torch.inference_mode():
        # The producer uses one model.step per tick. This path independently
        # invokes the complete sequence GRU and carries its final state onward.
        for begin in range(0, len(facts["now"]), 128):
            stop = min(begin + 128, len(facts["now"]))
            hidden, final = model.core(torch.from_numpy(facts["obs"][begin:stop]), h[None])
            logits = model.head(hidden).reshape(stop - begin, len(facts["actor_ids"]), 2, 11)
            if mode == "own":
                vectorized_action_disagreements += int((logits.argmax(-1).numpy() != facts["actions"][begin:stop]).sum())
            indices = np.flatnonzero((rows["tick"] >= begin) & (rows["tick"] < stop))
            selected_hidden = hidden[rows["tick"][indices] - begin, lane[indices]].numpy()
            selected_logits = logits[rows["tick"][indices] - begin, lane[indices]].numpy()
            require(np.isfinite(selected_hidden).all() and np.isfinite(selected_logits).all(), "finite vectorized replay")
            if len(indices):
                maximum_h_error = max(maximum_h_error, float(np.max(np.abs(rows["h"][indices] - selected_hidden))))
                maximum_logit_error = max(maximum_logit_error, float(np.max(np.abs(rows["logits"][indices] - selected_logits))))
            h = final[0]
        # Chunk and single-tick GEMM paths can accumulate different float32
        # rounding. A separate complete single-tick replay therefore provides
        # the identity check without loosening any tolerance after inspection.
        h = model.initial_state(len(facts["actor_ids"]))
        selected_at = defaultdict(list)
        for index, tick in enumerate(rows["tick"]):
            selected_at[int(tick)].append(index)
        for tick, observation in enumerate(facts["obs"]):
            values, next_h = model.core(torch.from_numpy(observation)[None], h[None])
            logits = model.head(values).reshape(1, len(facts["actor_ids"]), 2, 11)[0]
            h = next_h[0]
            if mode == "own":
                exact(logits.argmax(-1).numpy(), facts["actions"][tick], "own exact policy actions")
            indices = selected_at.get(tick, [])
            if indices:
                exact(rows["h"][indices], h[lane[indices]].numpy(), "exact independent tickwise hidden")
                exact(rows["logits"][indices], logits[lane[indices]].numpy(), "exact independent tickwise logits")
    independent_head = rows["h"].astype(np.float64) @ model.head.weight.numpy().astype(np.float64).T + model.head.bias.numpy()
    independent_head = independent_head.reshape(-1, 2, 11)
    close(rows["logits"], independent_head, "independent NumPy original head", 5e-6, 1e-5)
    require(model_state_sha(model) == before, "GRU parameter mutation")
    if "path" in model_info:
        require(sha(checkpoint) == model_info["checkpoint_sha256"], "checkpoint file mutation")
    return dict(rows=len(rows["h"]), max_hidden_abs_error=maximum_h_error,
                max_logit_abs_error=maximum_logit_error, vectorized_action_disagreements=vectorized_action_disagreements,
                tickwise_hidden_and_logits_bit_exact=True, model_parameters_unchanged=True)


def data_worker(task):
    """Bounded process worker, independent of producer APIs."""
    import torch
    torch.set_num_threads(1)
    seed, split, mode, own_arm = task
    plan = read(DATA / "protocol.json")
    base = DATA / mode / split / f"seed_{seed:03d}"
    fp = base / ("facts.npz" if mode == "teacher" else own_arm + "_facts.npz")
    before = sha(fp)
    facts, metadata = load(fp), read(fp.with_suffix(".json"))
    metadata_before = sha(fp.with_suffix(".json"))
    require(metadata["facts_sha256"] == before, "raw facts binding")
    require(metadata["protocol_sha256"] == sha(DATA / "protocol.json"), "raw data protocol binding")
    native = verify_native_stream(facts, metadata, plan)
    if mode == "own":
        teacher = load(DATA / "teacher/test" / f"seed_{seed:03d}/facts.npz")
        for name in ("actor_ids", "event_kind", "event_id", "event_raw", "event_family", "event_version", "event_duration", "event_window", "event_eligible"):
            exact(facts[name], teacher[name], "own/teacher common external stream " + name)
    arms = ARMS if mode == "teacher" else (own_arm,)
    output = []
    reference_keys = None
    for arm in arms:
        path = base / (arm + ".npz")
        sample_before = sha(path)
        rows = load(path)
        row_meta = read(path.with_suffix(".json"))
        require(row_meta["facts_sha256"] == before and row_meta["sample_sha256"] == sample_before, "sample/raw binding")
        require(row_meta["facts_metadata_sha256"] == metadata_before, "sample/raw metadata binding")
        require(row_meta["model"] == plan["models"][f"{seed}/{arm}"], "sample/frozen model binding")
        counts, cohort = verify_sample_rows(rows, facts, seed)
        require(row_meta["pairs"] == counts["pairs"], "sample pair metadata")
        require(metadata["counts"].get("included_pairs", 0) == counts["pairs"], "request pair count")
        require(metadata["counts"].get("matured_eligible_afterwarm", 0) == counts["matured"], "mature request denominator")
        include_saved = {(int(r["actor"]), int(r["job"]), int(r["rid"])) for r in metadata["requests"] if r["included"]}
        include_audit = {(r["actor"], r["job"], r["rid"]) for r in cohort if r["included"]}
        require(include_saved == include_audit, "request-level included identities")
        keys = {tuple(int(rows[name][i]) for name in ("actor", "job", "rid", "offset")) for i in range(len(rows["h"]))}
        if reference_keys is None:
            reference_keys = keys
        else:
            require(keys == reference_keys, "teacher models share exact cohort")
        recurrent = verify_recurrent(rows, facts, plan["models"][f"{seed}/{arm}"], mode)
        require(sha(path) == sample_before, "saved feature mutation")
        output.append(dict(seed=seed, split=split, mode=mode, arm=arm, counts=counts, recurrent=recurrent,
                           sample=str(path.relative_to(ROOT)), sample_sha256=sample_before))
    require(sha(fp) == before, "raw stream mutation")
    require(sha(fp.with_suffix(".json")) == metadata_before, "raw metadata mutation")
    return dict(seed=seed, split=split, mode=mode, arm=own_arm, facts_sha256=before, native=native, models=output)


def safe_inventory(paths):
    return {str(Path(path).resolve()): sha(path) for path in sorted(set(map(Path, paths))) if Path(path).is_file()}


def declare():
    require(not PLAN.exists(), "audit protocol already exists")
    require((DATA / "protocol.json").exists(), "data must be declared first")
    require((PROBES / "protocol.json").exists(), "probes must be declared first")
    sources = [Path(__file__), Path(__file__).with_name("audit-plan.md"), Path(__file__).with_name("test_audit.py"),
               Path(__file__).with_name("analysis-plan.md"),
               Path(__file__).with_name("data.py"), Path(__file__).with_name("probe.py"),
               DATA / "protocol.json", PROBES / "protocol.json"]
    write(PLAN, dict(format="timing-boundary-independent-audit-v1",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source_bindings=safe_inventory(sources),
                    tolerance=dict(recurrent_atol=5e-6, recurrent_rtol=1e-5, probe_atol=1e-9, probe_rtol=1e-8),
                    declaration="audit-plan.md recorded before probe scoring; no fitting or metric producer imports"))
    return read(PLAN)


def checked():
    plan = read(PLAN)
    for path, expected in plan["source_bindings"].items():
        require(sha(path) == expected, "changed frozen audit/source input: " + path)
    return plan


def verify_pin_map(pins):
    for path, expected in pins.items():
        candidate = Path(path)
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        require(sha(candidate) == expected, "changed artifact/source binding " + str(candidate))


def probe_worker(task):
    seed, arm = task
    features_by_task = dict(timing=("h", "obs", "logits", "h_shuffled", "h_mlp64"),
                            content=("h", "obs"), remaining=("h", "obs"))
    train_path = DATA / "teacher/train" / f"seed_{seed:03d}" / (arm + ".npz")
    train = load(train_path)
    hashes = {str(train_path): sha(train_path)}
    fitted = {}
    model_count = 0
    for task_name, features in features_by_task.items():
        selected = np.arange(len(train["job"])) if task_name == "remaining" else np.flatnonzero(np.isin(train["offset"], (-1, 0)))
        for feature in features:
            x = feature_matrix(train, feature)
            for slot in (0, 1):
                path = PROBES / f"seed_{seed:03d}" / arm / "models" / f"{task_name}__{feature}__slot{slot}.npz"
                hashes[str(path)] = sha(path)
                model = load(path)
                info = read(path.with_suffix(".json"))
                verify_pin_map(info["source_pins"])
                require(info["artifact_sha256"] == hashes[str(path)], "coefficient metadata binding")
                indices = selected[train["job"][selected] == slot]
                exact(model["train_row_indices"], indices, "train-only selected rows")
                expected_y = (train["remaining_ticks"][indices] if task_name == "remaining" else
                              train["target"][indices] - 3 if task_name == "content" else train["legal"][indices].astype(int))
                pairs = np.unique(train["pair_id"][indices])
                if feature == "h_shuffled":
                    generator = np.random.default_rng(928000 + 100 * seed + slot)
                    flipped = np.sort(generator.choice(pairs, size=len(pairs) // 2, replace=False))
                    expected_y = np.logical_xor(expected_y, np.isin(train["pair_id"][indices], flipped)).astype(int)
                else:
                    flipped = np.array([], dtype=np.int64)
                exact(model["flipped_train_pair_ids"], flipped, "independent deterministic pair-label flip")
                exact(model["train_targets"], expected_y, "fitted targets")
                reference = x[indices]
                mean, scale = reference.mean(axis=0), reference.std(axis=0)
                scale[scale == 0] = 1.
                close(model["mean"], mean, "training-only normalizer mean")
                close(model["scale"], scale, "training-only normalizer scale")
                require(all(np.isfinite(value).all() for value in model.values()), "nonfinite fitted artifact")
                fitted[(task_name, feature, slot)] = model
                model_count += 1
    datasets = [("teacher", "validation"), ("teacher", "test")]
    if arm != "random":
        datasets.append(("own", "test"))
    verified, metrics_output = [], []
    for mode, split in datasets:
        source_path = DATA / mode / split / f"seed_{seed:03d}" / (arm + ".npz")
        source = load(source_path)
        hashes[str(source_path)] = sha(source_path)
        for task_name, features in features_by_task.items():
            path = PROBES / f"seed_{seed:03d}" / arm / "predictions" / f"{mode}__{split}__{task_name}.npz"
            hashes[str(path)] = sha(path)
            saved, metadata = load(path), read(path.with_suffix(".json"))
            require(metadata["predictions_sha256"] == hashes[str(path)], "saved predictions binding")
            require(metadata["freeze_sha256"] == sha(PROBES / "freeze.json"), "evaluation uses frozen coefficients")
            verify_pin_map(metadata["source_pins"])
            indices = np.arange(len(source["job"])) if task_name == "remaining" else np.flatnonzero(np.isin(source["offset"], (-1, 0)))
            exact(saved["row_indices"], indices, "evaluation source selection")
            for key in ("pair_id", "offset", "job", "actor", "rid", "tick", "family", "raw", "conflict", "target", "duration", "remaining_ticks", "legal"):
                exact(saved[key], source[key][indices], "prediction metadata " + key)
            for feature in features:
                prediction = np.empty((len(indices), 8) if task_name == "content" else len(indices), dtype=np.float64)
                x = feature_matrix(source, feature)
                for slot in (0, 1):
                    positions = np.flatnonzero(source["job"][indices] == slot)
                    prediction[positions] = predict(x[indices[positions]], fitted[(task_name, feature, slot)], task_name)
                close(saved[feature], prediction, "independent coefficient prediction " + feature)
            if task_name == "content":
                choices = source["logits"][indices, source["job"][indices], 3:11].argmax(axis=1)
                exact(saved["original_conditional_choice"], choices, "original conditional-eight head")
            for feature, stratified in metadata["metrics"].items():
                prediction = np.eye(8)[saved["original_conditional_choice"]] if feature == "original_conditional_argmax" else saved[feature]
                rebuilt = {}
                for stratum, mask in metric_strata(saved):
                    if mask.any():
                        subset = {key: saved[key][mask] for key in ("pair_id", "offset", "job", "actor", "rid", "tick", "family", "raw", "conflict", "target", "duration", "remaining_ticks", "legal")}
                        rebuilt[stratum] = metrics(task_name, subset, prediction[mask])
                require(set(stratified) == set(rebuilt), "complete declared metric strata")
                compare_metrics(stratified, rebuilt, "independent metrics")
                metrics_output.append(dict(seed=seed, arm=arm, mode=mode, split=split, task=task_name, feature=feature,
                                           metrics=rebuilt["all"]))
            verified.append(dict(mode=mode, split=split, task=task_name, rows=len(indices), sha256=hashes[str(path)]))
    verify_pin_map(hashes)
    return dict(seed=seed, arm=arm, coefficient_artifacts=model_count, verified_predictions=verified, pooled_metrics=metrics_output)


def audit(workers=6):
    from concurrent.futures import ProcessPoolExecutor

    checked()
    data_plan = read(DATA / "protocol.json")
    verify_pin_map(data_plan["source_files"])
    for specification in data_plan["models"].values():
        if "path" in specification:
            require(sha(ROOT / specification["path"]) == specification["checkpoint_sha256"], "original model hash")
    actor_sets = [set(data_plan["actors"][split]) for split in SPLITS]
    require(all(len(group) == 32 for group in actor_sets), "32 actors per split")
    require(all(actor_sets[i].isdisjoint(actor_sets[j]) for i in range(3) for j in range(i)), "actor split leakage")
    frozen = read(PROBES / "freeze.json")
    verify_pin_map(frozen["pins"])
    data_tasks = [(seed, split, "teacher", None) for seed in range(3) for split in SPLITS]
    data_tasks += [(seed, "test", "own", arm) for seed in range(3) for arm in ARMS if arm != "random"]
    source_initial = safe_inventory([DATA / "protocol.json", PROBES / "protocol.json", PROBES / "freeze.json"])
    data_rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(data_worker, data_tasks):
            data_rows.append(result)
            write(OUT / "partial-data.json", dict(complete=False, streams=len(data_rows), data=data_rows))
            print(json.dumps(dict(event="audited_data", seed=result["seed"], split=result["split"], mode=result["mode"], arm=result["arm"])), flush=True)
    write(OUT / "data-audit.json", dict(complete=True, streams=len(data_rows), data=data_rows))
    probe_rows = []
    with ProcessPoolExecutor(max_workers=min(workers, 3)) as executor:
        for result in executor.map(probe_worker, [(seed, arm) for seed in range(3) for arm in ARMS]):
            probe_rows.append(result)
            print(json.dumps(dict(event="audited_probes", seed=result["seed"], arm=result["arm"])), flush=True)
    require(sum(len(row["models"]) for row in data_rows) == 45, "all45 model streams")
    require(sum(row["coefficient_artifacts"] for row in probe_rows) == 216, "all216 probe coefficient artifacts")
    require(sum(len(row["verified_predictions"]) for row in probe_rows) == 99, "all99 prediction artifacts")
    verify_pin_map(frozen["pins"])
    verify_pin_map(source_initial)
    checked()
    result = dict(passed=True, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(PLAN), native_streams=18, recurrent_model_streams=45,
                  probe_coefficient_artifacts=216, probe_prediction_artifacts=99,
                  actor_splits_disjoint=True, full_tickwise_hidden_logits_exact=True,
                  original_models_and_raw_unchanged=True, coefficients_independently_replayed=True,
                  native_free_behavior_separate=True, data=data_rows, probes=probe_rows,
                  frozen_probe_sha256=sha(PROBES / "freeze.json"))
    write(OUT / "audit.json", result)
    return {key: value for key, value in result.items() if key not in ("data", "probes")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    args = parser.parse_args()
    result = declare() if args.command == "declare" else audit(args.workers)
    print(json.dumps(result, ensure_ascii=False), flush=True)
