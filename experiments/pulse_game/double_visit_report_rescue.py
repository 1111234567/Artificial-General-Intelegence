"""Replace only the native tick-five reports with natural seen-map activity templates."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import double_visit_eval as ev
from . import double_visit_eval_report as statistics
from . import double_visit_fit as fit
from . import double_visit_world as world
from . import interleave_revisit_train as storage
from . import reroute_eval as paired


PLAN = ev.EVIDENCE / "rescue-plan.md"
MODES = ("self", "endpoint_true", "endpoint_opposite", "value_true", "value_opposite", "both_true", "both_opposite")
QUANTILE = .75
BOOT = 734590723


def weighted_sample(scores, weights, allowed):
    indices = np.flatnonzero(allowed)
    if not len(indices):
        return None
    sorted_indices = indices[np.argsort(scores[indices], kind="stable")]
    cumulative = weights[sorted_indices].cumsum()
    pick = np.searchsorted(cumulative, QUANTILE * cumulative[-1])
    selected = int(sorted_indices[pick])
    return {"episode": selected // 2, "door": selected % 2, "eligible_branches": len(indices),
            "natural_probability_mass": float(cumulative[-1]), "template_score": float(scores[selected])}


def calibrate(data):
    included = np.repeat(~data["heldout"], 2)
    q = data["natural/query_p"].astype(np.float64)
    weights = np.column_stack((1. - q, q)).ravel()
    endpoints = data["physical/endpoint"].ravel()
    payouts = data["physical/paid"].ravel()
    readout = data["natural/exit_read"].reshape(-1, 3) / world.ENDPOINT_GAIN
    value = data["natural/value_read"].ravel()
    last_two = np.sort(readout, axis=1)[:, -2:]
    margin = last_two[:, 1] - last_two[:, 0]
    endpoint_templates = {}
    for goal in range(3):
        mask = included & (endpoints == goal) & (readout.argmax(1) == goal) & (margin > 0)
        selected = weighted_sample(margin, weights, mask)
        if selected is not None:
            selected["activity"] = readout[2 * selected["episode"] + selected["door"]].tolist()
            endpoint_templates[goal] = selected
    value_templates = {}
    for sign in (-1, 1):
        mask = included & (payouts == sign) & (value * sign > 0.)
        selected = weighted_sample(np.abs(value), weights, mask)
        if selected is None:
            raise AssertionError("No correctly signed natural calibration report")
        selected["activity"] = float(value[2 * selected["episode"] + selected["door"]])
        value_templates[sign] = selected
    return endpoint_templates, value_templates


def continue_report(model, trial, endpoint, value):
    index = world.FORECAST_TICK - 1
    original = trial.states[index]
    activity = original.h.clone()
    activity[:, world.ENDPOINT_READOUT] = endpoint
    activity[:, world.READOUT] = value
    kept = torch.ones(activity.shape[1], dtype=torch.bool)
    kept[world.ENDPOINT_READOUT] = False
    kept[world.READOUT] = False
    torch.testing.assert_close(activity[:, kept], original.h[:, kept], atol=0, rtol=0)
    state = replace(original, h=activity)
    if any(getattr(state, name) is not getattr(original, name) for name in ("trace", "fast")):
        raise AssertionError("Report replacement touched the retained trace or local fast weights")
    for _ in range(world.FORECAST_TICK, world.TRIAL_TICKS):
        state, _ = model.tick(state, world.senses(state))
    return state, activity[:, world.ENDPOINT_READOUT], activity[:, world.READOUT]


def branch_values(p_right, early, truth):
    early_probability = np.where(early == 1, p_right, 1. - p_right).reshape(-1, 2)
    margin = early_probability[:, 0] - early_probability[:, 1]
    return {"both": margin.clip(min=0.), "neither": (-margin).clip(min=0.),
            "positive_optimal": early_probability[:, 0], "negative_optimal": 1. - early_probability[:, 1],
            "early_door_p_positive": early_probability[:, 0], "early_door_p_negative": early_probability[:, 1],
            "early_door_p_margin": margin,
            "mean_pay": (truth * np.column_stack((1. - p_right, p_right))).sum(1).reshape(-1, 2).mean(1)}


def scores_for_tape(p_right, early, truth, tape):
    choice = (tape < p_right[:, None]).astype(np.int8)
    paid = np.take_along_axis(truth, choice, 1)
    optimal = (paid == truth.max(1, keepdims=True)).reshape(-1, 2, tape.shape[1])
    early_choice = (choice == early[:, None]).reshape(-1, 2, tape.shape[1]).mean(2)
    return {"both": optimal.all(1).mean(1), "neither": (~optimal).all(1).mean(1),
            "positive_optimal": optimal[:, 0].mean(1), "negative_optimal": optimal[:, 1].mean(1),
            "early_door_p_positive": early_choice[:, 0], "early_door_p_negative": early_choice[:, 1],
            "early_door_p_margin": early_choice[:, 0] - early_choice[:, 1],
            "mean_pay": paid.reshape(-1, 2, tape.shape[1]).mean((1, 2))}


def effect_norm(end, val, source_end, source_val):
    terminal_delta = (end - source_end).double().norm(dim=1).numpy()
    value_delta = (val - source_val).double().abs().numpy()
    return {"endpoint_l2_mean": float(terminal_delta.mean()), "endpoint_l2_max": float(terminal_delta.max()),
            "value_abs_mean": float(value_delta.mean()), "value_abs_max": float(value_delta.max())}


@torch.inference_mode()
def collect(condition, birth, full, selected, tapes, provenance):
    model, _, checkpoint_hash = fit.load_final(birth, condition)
    model.eval().requires_grad_(False)
    archive = ev.data_path(condition, birth)
    with np.load(archive, allow_pickle=False) as saved:
        if str(saved["checkpoint"]) != checkpoint_hash or list(saved["source_sha256"]) != list(ev.fingerprints().values()):
            raise AssertionError("Rescue requires the same frozen body and sealed main evaluation")
        end_templates, value_templates = calibrate(saved)
        natural_probabilities = saved["natural/action_p"][selected]
    scene = full[selected]
    reference = world.oracle(scene)
    early = (scene.door_rooms[:, 1] == scene.inspected_rooms[:, 0]).long()
    truth = reference.payoffs.numpy()
    right_endpoint = reference.endpoints.gather(1, early[:, None]).squeeze(1).numpy()
    right_value = reference.payoffs.gather(1, early[:, None]).squeeze(1).numpy()
    np.testing.assert_array_equal(right_value.reshape(-1, 2), np.tile((1., -1.), (len(selected) // 2, 1)))
    np.testing.assert_array_equal(reference.payoffs.gather(1, (1 - early)[:, None]), 0.)
    wrong_endpoint = right_endpoint.reshape(-1, 2)[:, ::-1].ravel()
    prepared = world.prepare(model, scene)
    trial = world.try_side(model, prepared, early)
    base_end, base_val = trial.endpoint_activity, trial.forecast
    zero = torch.zeros_like(base_val)
    truth_physical = []
    for side in (torch.zeros_like(early), torch.ones_like(early)):
        motion = world.select(model, trial.final, zero, 0., forced=side)
        contact = world.settle(model, trial.final, scene, motion)
        ev.check_settle(reference, motion, contact)
        truth_physical.append(contact.paid.numpy())
    np.testing.assert_array_equal(np.column_stack(truth_physical), truth)
    read_sign = (base_val.numpy() * right_value > 0).reshape(-1, 2).all(1)
    raw = dict(critical_index=selected, early_door=early.numpy(), exact_paid=truth, endpoint=right_endpoint,
               payout=right_value, both_self_value_signs_correct=read_sign, template_support=sorted(end_templates),
               checkpoint=checkpoint_hash, archived_eval_sha256=ev.digest(archive),
               source_names=list(provenance), source_sha256=list(provenance.values()))
    interventions = {"self": (base_end, base_val)}
    good_value = torch.tensor([value_templates[x]["activity"] for x in right_value], dtype=base_val.dtype)
    bad_value = torch.tensor([value_templates[-x]["activity"] for x in right_value], dtype=base_val.dtype)
    interventions.update(value_true=(base_end, good_value), value_opposite=(base_end, bad_value))
    if set(end_templates) == {0, 1, 2}:
        good_end, bad_end = (torch.tensor([end_templates[x]["activity"] for x in ids], dtype=base_end.dtype)
                             for ids in (right_endpoint, wrong_endpoint))
        interventions.update(endpoint_true=(good_end, base_val), endpoint_opposite=(bad_end, base_val),
                             both_true=(good_end, good_value), both_opposite=(bad_end, bad_value))
    outcomes, norms, validity = {}, {}, {}
    for key in MODES:
        if key not in interventions:
            continue
        state, read_end, read_val = continue_report(model, trial, *interventions[key])
        is_true = key.endswith("_true")
        is_endpoint = key.split("_", 1)[0] in ("endpoint", "both")
        is_value = key.split("_", 1)[0] in ("value", "both")
        if is_endpoint:
            np.testing.assert_array_equal(read_end.argmax(1), right_endpoint if is_true else wrong_endpoint)
        else:
            torch.testing.assert_close(read_end, base_end, rtol=0, atol=0)
        if is_value:
            np.testing.assert_array_equal(read_val.sign(), right_value if is_true else -right_value)
        else:
            torch.testing.assert_close(read_val, base_val, rtol=0, atol=0)
        if key == "self" and not paired.equal_state(state, trial.final):
            raise AssertionError("Self-replacement changed the original final state")
        p = world.select(model, state, zero, 0.).probability.numpy()
        if key == "self":
            np.testing.assert_allclose(p, natural_probabilities[np.arange(len(scene)), early], rtol=0, atol=3e-6)
        raw.update({f"{key}/p_right": p, f"{key}/endpoint_activity": read_end.numpy(), f"{key}/value": read_val.numpy()})
        scores = {"exact": branch_values(p.astype(np.float64), early.numpy(), truth)}
        for seed, tape in zip(ev.SEEDS, tapes, strict=True):
            motion = world.select(model, state, torch.from_numpy(tape[:, 0].copy()), 0.)
            contact = world.settle(model, state, scene, motion)
            ev.check_settle(reference, motion, contact)
            raw[f"{key}/{seed}/door"] = motion.side.numpy()
            raw[f"{key}/{seed}/room"] = contact.room.numpy()
            raw[f"{key}/{seed}/endpoint"] = contact.endpoint.numpy()
            raw[f"{key}/{seed}/paid"] = contact.paid.numpy()
            scores[str(seed)] = scores_for_tape(p, early.numpy(), truth, tape)
        outcomes[key] = scores
        norms[key] = effect_norm(read_end, read_val, base_end, base_val)
        validity[key] = {"endpoint_accuracy_at_tick5": float((read_end.argmax(1).numpy() == right_endpoint).mean()),
                         "value_sign_accuracy_at_tick5": float((read_val.numpy() * right_value > 0.).mean())}
    file = ev.DATA / f"rescue-{condition}-{birth}.npz"
    storage._write_raw(file, **raw)
    controls = dict(calibration={"endpoint": end_templates, "value": value_templates}, replacement_size=norms,
                    report_verification=validity, archive_sha256=ev.digest(file),
                    archived_eval_sha256=ev.digest(archive), checkpoint_sha256=checkpoint_hash,
                    both_self_signs_count=int(read_sign.sum()))
    return outcomes, read_sign, controls


def estimate(scores, births):
    resamples = np.random.default_rng(BOOT).integers(len(births), size=(8192, len(births)))
    return statistics.birth_statistics(np.asarray(scores), resamples) | {"births": births}


def aggregate(outcomes):
    group = {}
    for name in MODES:
        subset = sorted(birth for birth, (entry, _) in outcomes.items() if name in entry)
        group[name] = {key: {metric: estimate([outcomes[b][0][name][key][metric].mean() for b in subset], subset)
                             for metric in outcomes[subset[0]][0][name][key]}
                       for key in statistics.MOTOR_NAMES}
    contrasts = {}
    for report in ("endpoint", "value", "both"):
        a = f"{report}_true"
        subset = sorted(birth for birth, (entry, _) in outcomes.items() if a in entry)
        for b in (f"{report}_opposite", "self"):
            contrasts[f"{a}_minus_{b}"] = {
                key: {metric: estimate([(outcomes[k][0][a][key][metric] - outcomes[k][0][b][key][metric]).mean()
                                        for k in subset], subset) for metric in outcomes[subset[0]][0][a][key]}
                for key in statistics.MOTOR_NAMES}
    conditional = {}
    for label, invert in (("both_signs_correct", False), ("one_or_no_sign_correct", True)):
        subsets = {birth: (~mask if invert else mask) for birth, (_, mask) in outcomes.items()}
        eligible = sorted(birth for birth, mask in subsets.items() if mask.any())
        conditional[label] = {"pair_counts_by_birth": {birth: int(mask.sum()) for birth, mask in subsets.items()},
                              "both": {key: estimate([outcomes[b][0]["self"][key]["both"][subsets[b]].mean()
                                                     for b in eligible], eligible) for key in statistics.MOTOR_NAMES}}
    return dict(rates=group, contrasts=contrasts, conditional=conditional)


def write_report(report):
    rows = ["# 第五拍原生报告替换", "", "[对照方案](rescue-plan.md) · [逐出生模板、完整数据和哈希](rescue-results.json)", ""]
    rows.append("只替换第五拍报告活动。身体从同一前史试向首查房的门，这扇门的真实价值在两份生命中分别为 +1、−1，另一门始终为 0。每个出生有 144 对未见旧图任务。")
    rows.extend(["", "## 原生马达的双对率", "",
                 "| 教学和替换 | 出生数 | 精确双对率 | 抽签一 | 抽签二 |", "|---|---:|---:|---:|---:|"])
    names = {"self": "自身回填", "endpoint_true": "正确终点", "endpoint_opposite": "相反终点",
             "value_true": "正确收益", "value_opposite": "反号收益", "both_true": "两者正确", "both_opposite": "两者相反"}
    for condition, title in (("identity", "终点身份教学"), ("baseline", "原教学")):
        data = report["results"][condition]
        for mode, caption in names.items():
            score = data["rates"][mode]
            cells = [statistics.as_rate(score[key]["both"], ci=False) for key in statistics.MOTOR_NAMES]
            rows.append(f"| {title}，{caption} | {len(score['exact']['both']['births'])} | " + " | ".join(cells) + " |")
    rows.extend(["", "原教学生命 735 在偶半图中没有终点 0 的正确自然报告。涉及终点的原教学对照只统计其余五个出生，收益对照使用六个出生。",
                 "", "| 终点身份教学的正确模板减相反模板 | 精确双对率差 | 抽签一 | 抽签二 | 试探门的价值反应差 |",
                 "|---|---:|---:|---:|---:|"])
    for label, caption in (("endpoint", "终点"), ("value", "收益"), ("both", "两者")):
        data = report["results"]["identity"]["contrasts"][f"{label}_true_minus_{label}_opposite"]
        cells = [statistics.as_rate(data[key]["both"]) for key in statistics.MOTOR_NAMES]
        cells.append(statistics.as_rate(data["exact"]["early_door_p_margin"]))
        rows.append(f"| {caption} | " + " | ".join(cells) + " |")
    rows.extend(["", "上表均为百分点，括号给出按出生配对重采样的 95% 区间。试探门的价值反应是正值世界和负值世界的试探门选择概率之差。",
                 "", "| 终点身份教学的正确模板减自身回填 | 精确双对率差 | 精确正值选优差 | 精确负值选优差 |",
                 "|---|---:|---:|---:|"])
    for label, caption in (("endpoint", "终点"), ("value", "收益"), ("both", "两者")):
        data = report["results"]["identity"]["contrasts"][f"{label}_true_minus_self"]["exact"]
        cells = [statistics.as_rate(data[key]) for key in ("both", "positive_optimal", "negative_optimal")]
        rows.append(f"| {caption} | " + " | ".join(cells) + " |")
    rows.extend(["", "## 替换幅度和未经替换的正确报告", "",
                 "自身回填逐位复现原最终状态。所有可用正确终点模板在第五拍分类正确，相反模板分类为对侧终点。收益模板逐条达到预定正负号。沿用完全相同的正式马达，逐项核对每套首次抽签的物理走门、房间、出口和到账。", ""])
    for condition, caption in (("identity", "终点身份教学"), ("baseline", "原教学")):
        group = report["results"][condition]["conditional"]["both_signs_correct"]
        values = [statistics.as_rate(group["both"][key], ci=False) for key in statistics.MOTOR_NAMES]
        counts = list(group["pair_counts_by_birth"].values())
        rows.append(f"- {caption}在未替换时，每出生两份报告收益符号都正确的对数为 {counts}，双对率依精确、抽签一、抽签二为 " + "%、".join(values) + "%。")
    controls = [report["controls"][f"identity/{birth}"] for birth in ev.BIRTHS]
    for label, caption in (("endpoint_true", "正确终点"), ("value_true", "正确收益"), ("both_true", "两者正确")):
        changes = [sample["replacement_size"][label] for sample in controls]
        end_mean = np.mean([sample["endpoint_l2_mean"] for sample in changes])
        val_mean = np.mean([sample["value_abs_mean"] for sample in changes])
        rows.append(f"- 身份教学的{caption}模板，替换动作产生的终点三节点活动变化平均范数为 {end_mean:.3f}，收益活动平均绝对变化为 {val_mean:.3f}。")
    value_activities = [source["activity"] for row in controls for source in row["calibration"]["value"].values()]
    rows.append(f"- 身份教学的原生收益模板实际活动跨出生为 {min(value_activities):.3f} 到 {max(value_activities):.3f}。每个模板保留原样自然供体，具体向量见 JSON。")
    rows.extend(["", "本轮正确收益相对反号收益的双对率小幅上升。正值与负值时的平均试探门反应差没有相应增加。正确收益相对自身回填时，两份生命的平均选优率均下降。终点身份模板没有显示正确身份相对相反身份的稳定双对优势。该对照的结论限于第五拍移植的报告节点。整段试探中分布在其他节点、其他时间的信号仍由原状态承载。", ""])
    (ev.EVIDENCE / "rescue-results.md").write_text("\n".join(rows), encoding="utf8")


def main():
    torch.set_num_threads(1)
    e, _, _, heldout, _, pairs, group_id = ev.layout()
    critical = pairs["first_only_changes"]
    selected = critical[heldout[critical].all(1)].ravel()
    tapes = [ev.random_tape(seed, group_id)[selected, :, 1] for seed in ev.SEEDS]
    provenance = ev.fingerprints() | {PLAN.name: ev.digest(PLAN), Path(__file__).name: ev.digest(__file__)}
    all_controls, results = {}, {}
    for condition in ev.CONDITIONS:
        outcomes = {}
        for birth in ev.BIRTHS:
            scores, signs, controls = collect(condition, birth, e, selected, tapes, provenance)
            outcomes[birth] = scores, signs
            all_controls[f"{condition}/{birth}"] = controls
        results[condition] = aggregate(outcomes)
    report = dict(pairs_per_birth=len(selected) // 2, sources=provenance, controls=all_controls, results=results,
                  calibration_quantile=QUANTILE, paired_bootstrap_seed=BOOT,
                  seeds=ev.SEEDS, draws_per_tape=ev.DRAWS)
    storage._write_json(ev.EVIDENCE / "rescue-results.json", report)
    write_report(report)
    print("Frozen tick-five report rescue complete", flush=True)


if __name__ == "__main__":
    main()
