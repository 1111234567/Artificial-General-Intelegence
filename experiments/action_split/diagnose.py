"""Bounded post-result, read-only mode/content failure diagnosis.

Final weights are replayed from an independent zero state. In particular the
training-stream replay is post-hoc inference, not the original online h trace.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch

from experiments.action_split import data, evaluate, train
from experiments.action_split.model import SplitGRUAgent
from experiments.continuity.run import state_digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/action-split/diagnostic-v1"
PLAN = OUT / "protocol.json"
FIELDS = ("obs", "actions", "active", "started", "now", "ready", "deadline", "duration",
          "version", "payload", "rid", "eligible")
GROUPS = ("all", "target_WAIT", "target_BEGIN", "target_FINISH", "idle", "active_unstarted",
          "processing", "legal", "active_other", "active_eligible", "active_ineligible")


def declare():
    sources = [Path(__file__), Path(data.__file__), Path(evaluate.__file__), Path(train.__file__),
        ROOT / "experiments/action_split/model.py", ROOT / "experiments/learn_remember/gru_train.py",
        ROOT / "experiments/lifelong/world.py", ROOT / "experiments/continuity/world.py",
        ROOT / "experiments/experience_transfer/train_transfer.py"]
    inputs = [train.OUT / "globalfreeze.json", train.PLAN, data.PLAN,
        evaluate.OUT / "protocol.json", evaluate.OUT / "freeze.json", evaluate.OUT / "completion.json"]
    for seed in range(3):
        inputs.extend([data.stream_path(seed), data.stream_path(seed).with_suffix(".json")])
        for arm in train.ARMS:
            inputs.extend([train.checkpoint_path(seed, arm, 256),
                evaluate.stem(seed, arm + "256", "long").with_suffix(".npz"),
                evaluate.stem(seed, arm + "256", "long").with_suffix(".json")])
    plan = dict(format="action-split-readonly-diagnostic-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        known_results_before_declaration="Primary free-action long: frozen15.77% strict, full0%, original65.50%; all4038 full matured requests began/reached legal but had no legal FINISH. This is a bounded post-result diagnosis, not a confirmatory comparison.",
        seeds=[0, 1, 2], arms=list(train.ARMS), snapshots=[256], updates=0,
        histories=dict(training_teacher="Original training public teacher stream, independent h0 from tick0; report ticks512..33279. Fixed final weights throughout: posthoc replay, not actual training-online h.",
            heldout_teacher="Fresh native earliest-teacher execution of evaluate.config(seed,'long'), same heldout actors/exogenous seed as free evaluation;512warm+4096report. Shared physical stream for both final models.",
            own_long="Each final model's archived ownlong public observations, h0 and exact CPU per-tick step; verify every native action and final h against archive."),
        computation=dict(teacher_device="cuda", teacher_block=128, own_device="cpu", own_block=1,
            dtype="float32", deterministic=True, TF32=False, numerical_h_continuous=True,
            private_facts="Only targets/statistical groups; model input remains public56 observations. No gradients, optimizer, thresholds, or model selection."),
        targets="Causal earliest oracle reconstructed from active/started, now, duration, ready, deadline, version, payload. BEGIN requires now+duration<=deadline; FINISH requires active+started+ready<=now<=deadline. Compare archived teacher actions/targets exactly.",
        metrics="Fixed groups all, targetWAIT/BEGIN/FINISH, idle, unstarted, processing, legal, otheractive, eligible/ineligibleactive.3x3modeconfusion, modeprob means, FINISHmode rank1..3,8way correctcontent rank/accuracy, mode/contentjoint, exact native actions. Save all legal-row coordinates/logits/probs/targets, compact exact group sums, all seed reports.",
        weight_accounting="Training raw WAIT/BEGIN/FINISH counts weighted .25/1/1; global shares and mean of actual256 block-normalized shares, bias-only optimum implied by each. Actual fixed-weight replay CE components kept separate from exposure shares; no claim of reconstructing perclass online training losses.",
        source_sha256={str(p): data.sha(p) for p in sources}, input_sha256={str(p): data.sha(p) for p in inputs})
    plan["protocol_hash"] = state_digest(plan)
    data.write_new(PLAN, plan)
    print(json.dumps(dict(event="diagnostic_declared", protocol_hash=plan["protocol_hash"])), flush=True)


def checked():
    plan = data.read(PLAN)
    assert state_digest({k: v for k, v in plan.items() if k != "protocol_hash"}) == plan["protocol_hash"]
    for key in ("source_sha256", "input_sha256"):
        for path, digest in plan[key].items():
            assert data.sha(path) == digest, path
    return plan


def load(path, fields=FIELDS):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in fields}


def labels(facts):
    active, started = facts["active"], facts["started"]
    now = facts["now"][:, None, None]
    begin = active & ~started & (now + facts["duration"] <= facts["deadline"] + 1e-6)
    legal = active & started & (now >= facts["ready"] - 1e-6) & (now <= facts["deadline"] + 1e-6)
    processing = active & started & (now < facts["ready"] - 1e-6)
    mode = np.where(begin, 1, np.where(legal, 2, 0)).astype(np.int64)
    native = np.where(begin, 1 + facts["version"], np.where(legal, 3 + 4 * facts["version"] + facts["payload"], 0))
    return dict(mode=mode, native=native, begin=facts["version"], finish=4 * facts["version"] + facts["payload"],
        legal=legal, processing=processing)


def masks(facts, target):
    active = facts["active"]
    return dict(all=np.ones_like(active), target_WAIT=target["mode"] == 0,
        target_BEGIN=target["mode"] == 1, target_FINISH=target["mode"] == 2,
        idle=~active, active_unstarted=active & ~facts["started"], processing=target["processing"],
        legal=target["legal"], active_other=active & facts["started"] & ~target["processing"] & ~target["legal"],
        active_eligible=active & facts["eligible"], active_ineligible=active & ~facts["eligible"])


def blank():
    return dict(rows=0, confusion=np.zeros((3, 3), dtype=np.int64), mode_prob_sum=np.zeros(3, dtype=np.float64),
        finish_mode_rank=np.zeros(3, dtype=np.int64), mode_FINISH_argmax=0, exact_native_actions=0,
        content_rows=0, content_correct=0, content_rank=np.zeros(8, dtype=np.int64),
        mode_content_joint=np.zeros((2, 2), dtype=np.int64), finish_and_correct_content=0,
        begin_rows=0, begin_content_correct=0)


def json_tree(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_tree(item) for item in value]
    return value


def add(group, mask, facts, target, logits, probability):
    mode = logits[..., :3].argmax(-1)
    finish_choice = logits[..., 5:].argmax(-1)
    begin_choice = logits[..., 3:5].argmax(-1)
    native = np.where(mode == 1, 1 + begin_choice, np.where(mode == 2, 3 + finish_choice, 0))
    group["rows"] += int(mask.sum())
    group["confusion"] += np.bincount((3 * target["mode"][mask] + mode[mask]), minlength=9).reshape(3, 3)
    group["mode_prob_sum"] += probability[mask].sum(axis=0, dtype=np.float64)
    rank = 1 + (logits[..., 0] >= logits[..., 2]) + (logits[..., 1] >= logits[..., 2])
    group["finish_mode_rank"] += np.bincount(rank[mask].astype(np.int64), minlength=4)[1:4]
    group["mode_FINISH_argmax"] += int((mask & (mode == 2)).sum())
    group["exact_native_actions"] += int((mask & (native == target["native"])).sum())
    active = mask & facts["active"]
    correct = finish_choice == target["finish"]
    mode_correct = mode == target["mode"]
    group["content_rows"] += int(active.sum())
    group["content_correct"] += int((active & correct).sum())
    group["mode_content_joint"] += np.bincount((2 * mode_correct[active].astype(np.int64) + correct[active]), minlength=4).reshape(2, 2)
    group["finish_and_correct_content"] += int((active & (mode == 2) & correct).sum())
    if active.any():
        scores = logits[..., 5:][active]
        true = target["finish"][active]
        true_scores = scores[np.arange(len(true)), true]
        content_rank = 1 + (scores > true_scores[:, None]).sum(-1) + ((scores == true_scores[:, None]) & (np.arange(8)[None] < true[:, None])).sum(-1)
        group["content_rank"] += np.bincount(content_rank, minlength=9)[1:9]
    begin = mask & (target["mode"] == 1)
    group["begin_rows"] += int(begin.sum())
    group["begin_content_correct"] += int((begin & (begin_choice == target["begin"])).sum())


def weight_accounting(seed):
    meta = data.read(data.stream_path(seed).with_suffix(".json"))
    blocks = meta["block_counts"]
    classes = ("mode_WAIT", "mode_BEGIN", "mode_FINISH")
    counts = np.array([meta["counts_training"][key] for key in classes], dtype=np.float64)
    weighted = counts * [.25, 1., 1.]
    shares = weighted / weighted.sum()
    block_shares = np.array([[row[key] for key in classes] for row in blocks]) * [.25, 1., 1.]
    block_shares /= block_shares.sum(axis=1, keepdims=True)
    mean_share = block_shares.mean(0)
    return dict(classes=list(classes), raw_counts=counts.astype(np.int64).tolist(),
        weights=[.25, 1., 1.], weighted_mass=weighted.tolist(), global_weighted_mass_share=shares.tolist(),
        mean_of256_actual_block_normalized_mass_share=mean_share.tolist(),
        bias_only_optimal_mode_probability_global=shares.tolist(),
        bias_only_optimal_mode_probability_actual_equal_block_objective=mean_share.tolist(),
        class_mass_is_not_measured_online_class_loss=True)


@torch.inference_mode()
def replay(seed, arm, kind, facts, plan):
    cp_path = train.checkpoint_path(seed, arm, 256)
    before = data.sha(cp_path)
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    device = "cpu" if kind == "own_long" else "cuda"
    block = 1 if kind == "own_long" else 128
    model = SplitGRUAgent(**cp["model_config"]).to(device)
    model.load_state_dict(cp["model"])
    model.eval().requires_grad_(False)
    initial_weights = state_digest(model.state_dict())
    h = model.initial_state(32, device=device)
    aggregate = {key: blank() for key in GROUPS}
    legal_rows = {key: [] for key in ("tick", "lane", "slot", "rid", "mode_logits", "mode_probability",
        "mode_target", "finish_logits", "finish_target", "version", "payload", "eligible")}
    target_all = labels(facts)
    if kind != "own_long":
        np.testing.assert_array_equal(target_all["native"], facts["actions"])
    mismatches = 0
    weighted_class_ce = np.zeros(3, dtype=np.float64)
    weighted_class_mass = np.zeros(3, dtype=np.float64)
    block_normalized_class_ce = np.zeros(3, dtype=np.float64)
    loss_blocks = 0
    for start in range(0, len(facts["obs"]), block):
        stop = min(start + block, len(facts["obs"]))
        scores, h = model(torch.from_numpy(facts["obs"][start:stop]).to(device), h)
        h = h.detach()
        if kind == "own_long":
            native = model.choose(scores).cpu().numpy()
            mismatches += int((native != facts["actions"][start:stop]).sum())
        if stop <= evaluate.WARM:
            continue
        report_start = max(start, evaluate.WARM)
        offset = report_start - start
        scores = scores[offset:]
        logits = scores.cpu().numpy()
        probabilities = scores[..., :3].softmax(-1).cpu().numpy()
        current = {key: value[report_start:stop] for key, value in facts.items() if key != "final_h"}
        target = {key: value[report_start:stop] for key, value in target_all.items()}
        group_masks = masks(current, target)
        for key, mask in group_masks.items():
            add(aggregate[key], mask, current, target, logits, probabilities)
        actual = target["mode"]
        logp = scores[..., :3].log_softmax(-1).cpu().numpy()
        component = np.zeros(3, dtype=np.float64)
        mass = np.zeros(3, dtype=np.float64)
        for class_id, weight in enumerate((.25, 1., 1.)):
            selected = actual == class_id
            component[class_id] = float((-logp[..., class_id][selected]).sum(dtype=np.float64)) * weight
            mass[class_id] = float(selected.sum()) * weight
        weighted_class_ce += component
        weighted_class_mass += mass
        block_normalized_class_ce += component / mass.sum()
        loss_blocks += 1
        legal = target["legal"]
        ti, lane, slot = np.where(legal)
        legal_rows["tick"].append(ti + report_start)
        legal_rows["lane"].append(lane); legal_rows["slot"].append(slot)
        for key in ("rid", "version", "payload", "eligible"):
            legal_rows[key].append(current[key][legal])
        legal_rows["mode_logits"].append(logits[..., :3][legal])
        legal_rows["mode_probability"].append(probabilities[legal])
        legal_rows["mode_target"].append(actual[legal])
        legal_rows["finish_logits"].append(logits[..., 5:][legal])
        legal_rows["finish_target"].append(target["finish"][legal])
    if kind == "own_long":
        assert mismatches == 0, (seed, arm, mismatches)
        np.testing.assert_array_equal(h.cpu().numpy(), facts["final_h"])
    assert data.sha(cp_path) == before
    assert initial_weights == state_digest(model.state_dict())
    for group in aggregate.values():
        group["mode_probability_mean"] = group["mode_prob_sum"] / max(1, group["rows"])
        group["content_accuracy"] = group["content_correct"] / max(1, group["content_rows"])
        group["exact_native_action_accuracy"] = group["exact_native_actions"] / max(1, group["rows"])
    folder = OUT / f"seed_{seed:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / (arm + "_" + kind)
    arrays = {key: np.concatenate(parts) for key, parts in legal_rows.items()}
    with stem.with_suffix(".legal.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result = dict(seed=seed, arm=arm, history=kind, report_start=evaluate.WARM,
        report_stop=len(facts["obs"]), inference_device=device, inference_block=block,
        protocol_hash=plan["protocol_hash"], checkpoint_path=str(cp_path), checkpoint_sha256=before,
        model_unchanged=True, initial_h="zero at independent origin", numerical_h="continuous, no resets",
        all_teacher_targets_reconstructed=kind != "own_long", own_native_action_mismatches=mismatches if kind == "own_long" else None,
        own_final_h_bit_exact=kind == "own_long", groups=json_tree(aggregate),
        fixed_replay_mode_CE=dict(class_order=["WAIT", "BEGIN", "FINISH"],
            weighted_class_numerator=weighted_class_ce.tolist(), weighted_class_mass=weighted_class_mass.tolist(),
            global_normalized_class_loss=(weighted_class_ce / weighted_class_mass.sum()).tolist(),
            mean_block_normalized_class_loss=(block_normalized_class_ce / loss_blocks).tolist(),
            block_count=loss_blocks, note="Fixed final-weight replay loss. Not the original online training loss; block1 own means perrow normalized."),
        legal_rows_path=str(stem.with_suffix(".legal.npz")), legal_rows_sha256=data.sha(stem.with_suffix(".legal.npz")))
    data.write_new(stem.with_suffix(".json"), result)
    print(json.dumps(dict(event="diagnosed", seed=seed, arm=arm, history=kind,
        legal=result["groups"]["legal"])), flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def run():
    plan = checked()
    train.deterministic(923000)
    results = []
    accounting = {}
    for seed in range(3):
        facts = load(data.stream_path(seed))
        target = labels(facts)
        with np.load(data.stream_path(seed), allow_pickle=False) as archive:
            np.testing.assert_array_equal(target["mode"], archive["mode_target"])
            np.testing.assert_array_equal(target["finish"][facts["active"]], archive["finish_target"][facts["active"]])
        for arm in train.ARMS:
            results.append(replay(seed, arm, "training_teacher", facts, plan))
        del facts
        accounting[str(seed)] = weight_accounting(seed)
        teacher, context = data.collect_native(seed, total=evaluate.WARM + evaluate.STEPS,
            world_config=evaluate.config(seed, "long"))
        audit = data.verify_facts(teacher)
        folder = OUT / f"seed_{seed:03d}"
        teacher_path = folder / "heldout_teacher.npz"
        with teacher_path.open("xb") as stream:
            np.savez_compressed(stream, **teacher)
        data.write_new(teacher_path.with_suffix(".json"), dict(seed=seed, world_config=evaluate.config(seed, "long"),
            facts_sha256=data.sha(teacher_path), external_hash=context["events_sha256"], audit=audit,
            independent_earliest_teacher=True, common_to_both_models=True))
        for arm in train.ARMS:
            results.append(replay(seed, arm, "heldout_teacher", teacher, plan))
        del teacher
        for arm in train.ARMS:
            own = load(evaluate.stem(seed, arm + "256", "long").with_suffix(".npz"), (*FIELDS, "final_h"))
            results.append(replay(seed, arm, "own_long", own, plan))
            del own
    checked()
    result = dict(format="action-split-readonly-diagnostic-results-v1", protocol_hash=plan["protocol_hash"],
        no_updates=True, training_weight_accounting=accounting, reports=results,
        limitations=["Post-result diagnosis with the existing fixed models, not a new confirmatory experiment.",
            "Training teacher replay uses final weights from h0, not the original changing online training state.",
            "Teacher versus own differences combine action-history and state-distribution changes; no causal attribution from this comparison alone.",
            "Class exposure mass and bias-only optimum do not prove that class imbalance alone caused failure."])
    data.write_new(OUT / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    args = parser.parse_args()
    declare() if args.command == "declare" else run()
