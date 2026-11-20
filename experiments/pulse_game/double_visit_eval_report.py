"""Integrate exact policy probabilities and summarize two independent motor tapes."""

import numpy as np

from . import double_visit_eval as ev
from . import double_visit_fit as fit
from . import double_visit_world as world
from . import interleave_revisit_train as storage


BOOT = 734590703
MOTOR_NAMES = ("exact", *(str(seed) for seed in ev.SEEDS))
CONTRASTS = {
    "teaching": ("identity/natural", "baseline/natural"),
    "identity_trial": ("identity/natural", "identity/blank"),
    "omit_first": ("identity/natural", "identity/omit_first"),
    "return_mismatch": ("identity/natural", "identity/wrong_return"),
    "baseline_trial": ("baseline/natural", "baseline/blank"),
}


def by_row(values, col):
    return values[np.arange(len(values))[:, None], col]


def metrics_for_tape(q, p, settled, tape):
    query = (tape[..., 0] < q[:, None]).astype(np.intp)
    p_final = by_row(p, query)
    side = (tape[..., 1] < p_final).astype(np.intp)
    independent_side = (tape[..., 2] < p_final).astype(np.intp)
    paid = by_row(settled, side)
    return query, side, independent_side, paid


def integrate_pair(q, p, wanted, indices):
    """Integrate one shared query flag and either one or two native action draws."""
    two_q = q[indices].astype(np.float64)
    two_p = p[indices].astype(np.float64)
    two_wanted = wanted[indices]
    n = len(indices)
    cuts = np.column_stack((np.zeros(n), np.sort(two_q, axis=1), np.ones(n)))
    accum = np.zeros((n, 5), dtype=np.float64)
    for low, high in zip(cuts[:, :-1].T, cuts[:, 1:].T, strict=True):
        query = ((high + low)[:, None] * .5 < two_q).astype(np.intp)
        prob = np.take_along_axis(two_p, query[..., None], 2)[..., 0]
        correct_lo = np.where(two_wanted, 0., prob)
        correct_hi = np.where(two_wanted, prob, 1.)
        probability = correct_hi - correct_lo
        both = (correct_hi.min(1) - correct_lo.max(1)).clip(min=0.)
        neither = np.where(two_wanted, 1., prob).min(1) - np.where(two_wanted, prob, 0.).max(1)
        neither = neither.clip(min=0.)
        values = np.column_stack((both, neither, probability.prod(1), probability,))
        accum += (high - low)[:, None] * values
    return pair_metrics(*accum.T)


def pair_metrics(both, neither, independent, first, second):
    return dict(both=both, neither=neither, both_independent_action=independent,
                first=first, second=second, one=1. - both - neither, direction=both - neither,
                same_correctness=both + neither)


def sampled_pair(wanted, indices, action, indep_action):
    left, right = indices.T
    first = action[left] == wanted[left, None]
    second = action[right] == wanted[right, None]
    other = indep_action[right] == wanted[right, None]
    return pair_metrics((first & second).mean(1), (~first & ~second).mean(1), (first & other).mean(1),
                        first.mean(1), second.mean(1))


def native_readouts(data, prefix, q, settled):
    value = data[prefix + "value_read"]
    scalar = data[prefix + "room_read"]
    endpoints = data[prefix + "exit_read"].argmax(2)
    room = data["room"]
    exit_correct = endpoints == data["endpoint"]
    room_correct = np.where(scalar == 0., .5, (scalar > 0.) == room)
    value_error = (value - settled) ** 2
    rank = (value[:, 1] - value[:, 0]) * np.sign(settled[:, 1] - settled[:, 0])
    weights = np.column_stack((1. - q, q))
    readout = {}
    for label, branch_weights in (("probed", weights), ("other_branch", weights[:, ::-1]),
                                  ("both_branches", np.full_like(weights, .5))):
        for metric, values in (("room_accuracy", room_correct), ("exit_accuracy", exit_correct),
                               ("value_mse", value_error)):
            readout[f"{label}/{metric}"] = (branch_weights * values).sum(1)
    readout["rank"] = np.where(settled[:, 0] == settled[:, 1], np.nan, (rank > 0) + .5 * (rank == 0))
    return readout


def assess(data, mode, row_masks, pair_masks, pairs, tapes):
    prefix = mode + "/"
    q, p = data[prefix + "query_p"].astype(np.float64), data[prefix + "action_p"].astype(np.float64)
    settled = data["physical/paid"]
    wanted = settled.argmax(1)
    exact_right = p[:, 0] * (1. - q) + p[:, 1] * q
    maximum = settled.max(1)
    tied = settled[:, 1] == settled[:, 0]
    row_scores = {"exact": {"optimal": np.where(tied, 1., np.where(wanted, exact_right, 1. - exact_right)),
                            "door_right": exact_right, "pay": settled[:, 0] * (1. - exact_right) + settled[:, 1] * exact_right}}
    pair_scores = {label: {"exact": integrate_pair(q, p, wanted, ids)} for label, ids in pairs.items()}
    for i, seed in enumerate(ev.SEEDS):
        _, side, independent, paid = metrics_for_tape(q, p, settled, tapes[i])
        for attribute, from_table in (("door", side[:, 0]), ("paid", paid[:, 0]),
                                      ("room", by_row(data["physical/room"], side)[:, 0]),
                                      ("endpoint", by_row(data["physical/endpoint"], side)[:, 0])):
            np.testing.assert_array_equal(data[prefix + "tape_contact/" + attribute][:, i], from_table)
        row_scores[str(seed)] = {"optimal": (paid == maximum[:, None]).mean(1),
                                 "door_right": side.mean(1), "pay": paid.mean(1)}
        for label, ids in pairs.items():
            pair_scores[label][str(seed)] = sampled_pair(wanted, ids, side, independent)
    if mode == "omit_first":
        left, right = pairs["first_only_changes"].T
        np.testing.assert_array_equal(q[left], q[right])
        np.testing.assert_array_equal(p[left], p[right])
        for sampled in pair_scores["first_only_changes"].values():
            np.testing.assert_array_equal(sampled["both"], 0.)
            np.testing.assert_array_equal(sampled["neither"], 0.)

    def mean_by_panel(scores, masks):
        return {split: {f"{source}/{metric}": float(value[mask].mean())
                        for source, values in scores.items() for metric, value in values.items()}
                for split, mask in masks.items() if mask.any()}

    result = {"choice": mean_by_panel(row_scores, row_masks)}
    result["pairs"] = {name: mean_by_panel(pair_scores[name], pair_masks[name]) for name in pairs}
    readout = native_readouts(data, prefix, q, settled)
    result["readouts"] = {stratum: {metric: float(value[mask & np.isfinite(value)].mean())
                                   for metric, value in readout.items() if (mask & np.isfinite(value)).any()}
                          for stratum, mask in row_masks.items() if mask.any()}
    return result


def build_strata(e, case, heldout, pairs):
    touched = e.inspected_rooms.numpy()
    seen = e.observed_endpoints.numpy()
    two_rewrites = ((touched[:, 0] == touched[:, 1]) & (seen[:, 0] != seen[:, 1])
                    & (seen[:, 0] != e.prior_endpoints.numpy()[np.arange(len(case)), touched[:, 0]]))
    same_room = (touched[:, 0] == touched[:, 1])
    tied = case == world.TIE
    tie_value = world.oracle(e).payoffs.numpy()[:, 0]
    splits = {"unseen": heldout, "seen": ~heldout}
    cases = {name: case == i for i, name in enumerate(world.CASE_NAMES)}
    cases.update(all=np.ones_like(heldout), same_room=same_room, different_rooms=~same_room,
                 same_room_two_rewrites=two_rewrites, same_room_two_rewrites_strict=two_rewrites & ~tied,
                 tie_minus=tied & (tie_value == -1), tie_zero=tied & (tie_value == 0),
                 tie_plus=tied & (tie_value == 1))
    for name in ("flip", "stay", "unchanged", "tie"):
        for room_type in ("same_room", "different_rooms"):
            cases[f"{room_type}/{name}"] = cases[room_type] & cases[name]
    row_masks = {f"{split}/{event}": used & subset for split, used in splits.items() for event, subset in cases.items()}
    pair_masks = {name: {split: mask[indices].all(1) for split, mask in splits.items()} for name, indices in pairs.items()}
    pair_masks["room_relabel"]["cross_split"] = (heldout[pairs["room_relabel"]].sum(1) == 1)
    return row_masks, pair_masks


def flatten(tree, stem=""):
    result = {}
    for key, value in tree.items():
        full = f"{stem}/{key}" if stem else key
        if isinstance(value, dict):
            result.update(flatten(value, full))
        else:
            result[full] = value
    return result


def birth_statistics(entries, indices):
    return {"mean": float(entries.mean()), "ci95": np.quantile(entries[indices].mean(1), (.025, .975)).tolist(),
            "birth_means": entries.tolist()}


def check_data(data, e, case, map_id, heldout, donor, group, condition, birth):
    sources = ev.fingerprints()
    if (list(data["source_names"]) != list(sources) or list(data["source_sha256"]) != list(sources.values())
            or int(data["birth"]) != birth or str(data["condition"]) != condition
            or str(data["checkpoint"]) != ev.digest(fit.destination(birth, condition))
            or int(data["draws"]) != ev.DRAWS):
        raise AssertionError(f"Unsealed frozen result: {condition} {birth}")
    for key, value in (("seeds", ev.SEEDS), ("case", case), ("map_id", map_id), ("heldout", heldout),
                       ("donor", donor), ("group_id", group), ("door_rooms", e.door_rooms),
                       ("prior_endpoints", e.prior_endpoints), ("visit_order", e.visit_order),
                       ("inspected_rooms", e.inspected_rooms), ("observed_endpoints", e.observed_endpoints),
                       ("goals", e.goals)):
        np.testing.assert_array_equal(data[key], value)
    truth = world.oracle(e)
    for name, actual in (("room", truth.rooms), ("endpoint", truth.endpoints), ("paid", truth.payoffs)):
        np.testing.assert_array_equal(data["physical/" + name], actual)
    np.testing.assert_array_equal(data["physical/door"], np.tile((0, 1), (len(e), 1)))


def summarize():
    e, case, map_id, heldout, donor, pairs, group = ev.layout()
    row_masks, pair_masks = build_strata(e, case, heldout, pairs)
    tapes = [ev.random_tape(seed, group) for seed in ev.SEEDS]
    series, checkpoints, archives, audits = {}, {}, {}, {}
    for condition in ev.CONDITIONS:
        for birth in ev.BIRTHS:
            source = ev.data_path(condition, birth)
            with np.load(source, allow_pickle=False) as raw:
                data = dict(raw)
            check_data(data, e, case, map_id, heldout, donor, group, condition, birth)
            archives[source.name] = ev.digest(source)
            checkpoints[f"{condition}/{birth}"] = str(data["checkpoint"])
            audits[f"{condition}/{birth}"] = {"norm_error_max": float(data["norm_error"]), "fallback": int(data["fallback"])}
            for mode in ev.MODES:
                group_key = f"{condition}/{mode}"
                result = flatten(assess(data, mode, row_masks, pair_masks, pairs, tapes))
                for name, value in result.items():
                    series.setdefault(group_key, {}).setdefault(name, []).append(value)

    indices = np.random.default_rng(BOOT).integers(len(ev.BIRTHS), size=(8192, len(ev.BIRTHS)))
    series = {key: {name: np.array(values) for name, values in rows.items()} for key, rows in series.items()}
    results = {key: {name: birth_statistics(values, indices) for name, values in rows.items()}
               for key, rows in series.items()}
    contrasts = {label: {name: birth_statistics(series[a][name] - series[b][name], indices) for name in series[a]}
                 for label, (a, b) in CONTRASTS.items()}
    def gates(series):
        return {"teaching_both": [contrasts["teaching"][f"pairs/first_only_changes/unseen/{sample}/both"]["ci95"][0] > 0
                                  for sample in MOTOR_NAMES],
                "natural_trial_both": [contrasts["identity_trial"][f"pairs/first_only_changes/unseen/{sample}/both"]["ci95"][0] > 0
                                       for sample in MOTOR_NAMES],
                "direction": [series["identity/natural"][f"pairs/first_only_changes/unseen/{sample}/direction"]["ci95"][0] > 0
                              for sample in MOTOR_NAMES],
                "stay_noninferiority": [contrasts["identity_trial"][f"choice/unseen/stay/{sample}/optimal"]["ci95"][0] > -.05
                                         for sample in MOTOR_NAMES]}

    report = dict(protocol=ev.fingerprints(), raw_sha256=archives, checkpoint_sha256=checkpoints,
                  controls=audits, births=ev.BIRTHS, old_maps_supported=world.OLD_MAPS,
                  train_map_ids=world.EVEN_MAP_IDS,
                  heldout_map_ids=sorted(set(map_id[heldout].tolist())), doors_per_old_room_map=2,
                  draws_per_tape=ev.DRAWS, tape_seeds=ev.SEEDS, motor_names=MOTOR_NAMES,
                  tape_groups=int(group.max()) + 1, tape_bootstrap_seed=BOOT,
                  row_counts={name: int(mask.sum()) for name, mask in row_masks.items()},
                  pair_counts={name: {split: int(mask.sum()) for split, mask in masks.items()}
                               for name, masks in pair_masks.items()},
                  results=results, contrasts=contrasts, gates=gates(results))
    storage._write_json(ev.EVIDENCE / "eval-results.json", report)
    write_markdown(report)
    for name in ("baseline/natural", "identity/natural", "identity/blank", "identity/omit_first", "identity/wrong_return"):
        key = "pairs/first_only_changes/unseen/exact/both"
        print(name, as_rate(results[name][key]), flush=True)
    print(report["gates"], flush=True)


def as_rate(item, *, ci=True, multiplier=100):
    value = item["mean"] * multiplier
    if not ci:
        return f"{value:.2f}"
    low, high = (x * multiplier for x in item["ci95"])
    return f"{value:.2f} [{low:.2f}, {high:.2f}]"


def write_markdown(report):
    values, contrasts = report["results"], report["contrasts"]
    group_names = {"baseline/natural": "原教学，自然", "identity/natural": "追加身份教学，自然",
                   "identity/blank": "追加身份教学，同体空试", "identity/omit_first": "追加身份教学，等时省略第一笔",
                   "identity/wrong_return": "追加身份教学，等范数错配回流", "baseline/blank": "原教学，同体空试"}
    text = ["# 双复查冻结评估", "", "[冻结协议](eval-plan.md) · [全部数值、逐出生结果和哈希](eval-results.json)",
            "", "## 两份生命都选对", ""]

    def pair_table(which, split, metric):
        n = report["pair_counts"][which][split]
        text.extend([f"每出生 {n} 对完整任务，按六个出生重采样。数值为百分比，方括号是 95% 区间。", "",
                     "| 条件 | 精确 | 抽签一 | 抽签二 |", "|---|---:|---:|---:|"])
        for g, title in group_names.items():
            cells = [as_rate(values[g][f"pairs/{which}/{split}/{sample}/{metric}"]) for sample in MOTOR_NAMES]
            text.append(f"| {title} | " + " | ".join(cells) + " |")
        text.append("")

    pair_table("first_only_changes", "unseen", "both")
    text.extend(["| 严格留出主端点的精确策略 | 双对 | 双错 | 第一笔指向正值时选优 | 第一笔指向负值时选优 |",
                 "|---|---:|---:|---:|---:|"])
    for g, title in group_names.items():
        row = [as_rate(values[g][f"pairs/first_only_changes/unseen/exact/{metric}"], ci=False)
               for metric in ("both", "neither", "first", "second")]
        text.append(f"| {title} | " + " | ".join(row) + " |")
    text.append("")
    text.extend(["| 严格留出主端点 | 精确 | 抽签一 | 抽签二 |", "|---|---:|---:|---:|"])
    for label, caption in (("teaching", "追加教学减原教学"), ("identity_trial", "同体自然减空试"),
                           ("return_mismatch", "同体自然减错配回流"), ("baseline_trial", "原教学自然减空试")):
        key = "pairs/first_only_changes/unseen/{}/both"
        cells = [as_rate(contrasts[label][key.format(sample)]) for sample in MOTOR_NAMES]
        text.append(f"| {caption}，百分点 | " + " | ".join(cells) + " |")
    key = "pairs/first_only_changes/unseen/{}/direction"
    text.append("| 追加教学自然状况的双对减双错，百分点 | " + " | ".join(
        as_rate(values["identity/natural"][key.format(sample)]) for sample in MOTOR_NAMES) + " |")
    text.append("")
    text.extend(["每对两份生命的正式马达各自独立抽签时，双对率如下。试探抽签依然共享。", ""])
    pair_table("first_only_changes", "unseen", "both_independent_action")
    text.extend(["## 保留、覆盖、时间反转与交换目标", "",
                 "以下选择均列精确期望。抽签结果和逐出生数值见 JSON。", "",
                 "| 严格留出场景 | 每出生场景数 | 原教学，自然 | 身份教学，自然 | 身份教学，空试 |",
                 "|---|---:|---:|---:|---:|"])
    for stratum, caption in (("flip", "最终应换门"), ("stay", "最终应保留旧优门"), ("unchanged", "最终图未改"),
                              ("same_room_two_rewrites_strict", "同房实际改写两次且有严格优门"),
                              ("different_rooms/flip", "分房复查后应换门"),
                              ("tie_minus", "两门同归，到账 -1"), ("tie_zero", "两门同归，到账 0"),
                              ("tie_plus", "两门同归，到账 +1")):
        key = "pay" if stratum.startswith("tie_") else "optimal"
        sample = f"choice/unseen/{stratum}/exact/{key}"
        cells = [as_rate(values[g][sample], ci=False, multiplier=1 if key == "pay" else 100)
                 for g in list(group_names)[:3]]
        text.append(f"| {caption} | {report['row_counts'][f'unseen/{stratum}']} | " + " | ".join(cells) + " |")
    text.extend(["", "| 配对对照 | 每出生对数 | 原教学，自然 | 身份教学，自然 | 身份教学，空试 |",
                 "|---|---:|---:|---:|---:|"])
    for which, split, title in (("last_only_changes", "unseen", "仅最后一笔不同，最佳门相反"),
                                ("goal_swap", "unseen", "沿用历史，只交换目标的两个终点价值"),
                                ("overwritten_first", "unseen", "同房两次复查，首笔被覆盖"),
                                ("room_relabel", "cross_split", "换房间编号但保留物理路线，跨划分")):
        key = f"pairs/{which}/{split}/exact/both"
        cells = [as_rate(values[g][key], ci=False) for g in list(group_names)[:3]]
        text.append(f"| {title} | {report['pair_counts'][which][split]} | " + " | ".join(cells) + " |")
    relabel_key = "pairs/room_relabel/cross_split/exact/same_correctness"
    relabel_agreement = [as_rate(values[g][relabel_key], ci=False) for g in list(group_names)[:3]]
    text.extend(["", "只重新编号房间时，两份生命选择同一扇门的概率依次为 " + "%、".join(relabel_agreement) + "%。", "",
                 "## 原生预测读数", "",
                 "以严格留出的全部生命报告房间与终点准确率（%）、数值均方误差。两门排名只用有严格优门的任务。未试门读数来自独立运行的另一条内部试探支路。原教学的终点读出未接受直接终点身份监督。", "",
                 "| 教学和分支 | 房间准确率 | 终点准确率 | 数值均方误差 |", "|---|---:|---:|---:|"])
    for group in ("baseline/natural", "identity/natural", "identity/blank", "identity/wrong_return"):
        for sub, caption in (("probed", "自选试门"), ("other_branch", "未试门")):
            base = f"readouts/unseen/all/{sub}/"
            cells = [as_rate(values[group][base + metric], ci=False, multiplier=scale)
                     for metric, scale in (("room_accuracy", 100), ("exit_accuracy", 100), ("value_mse", 1))]
            text.append(f"| {group_names[group]}，{caption} | " + " | ".join(cells) + " |")
    text.extend(["", "两门数值排序准确率（%）为 " + "、".join(
        f"{group_names[g]} {as_rate(values[g]['readouts/unseen/all/rank'], ci=False)}"
         for g in ("baseline/natural", "identity/natural", "identity/blank", "identity/wrong_return")) + "。", "",
        "## 分母、物理回执与判据", ""])
    total = report["row_counts"]["unseen/all"]
    n = report["pair_counts"]["first_only_changes"]["unseen"]
    text.append(f"三种未见旧房出口图配两种旧门接法，共六张完整未见旧图、{total} 个任务和 {n} 对主端点。"
                f"各自复用于六个独立出生、两个教学臂。每套抽签在六个平衡组内各重复 {ev.DRAWS} 次，组内共用随机带。"
                "每种模式在每个任务和抽签带上的首次抽签均推进物理状态，并与两门独立走过的固定结算逐项核对。"
                "其余抽签用同一原生概率产生正式马达指令，再从已实走并核对的两种结算中取得所选门的回执。")
    outcomes = report["gates"]
    for key, caption in (("teaching_both", "教学的双对率差"), ("natural_trial_both", "同体自然相对空试的双对率差"),
                         ("direction", "自然双对率减双错率"), ("stay_noninferiority", "同体保留原优门的非劣条件")):
        status = "通过" if all(outcomes[key]) else "未通过"
        members = "、".join("通过" if passed else "未通过" for passed in outcomes[key])
        text.append(f"\n- {caption}{status}。精确期望、抽签一、抽签二依次为{members}。")
    max_norm = max(row["norm_error_max"] for row in report["controls"].values())
    zero_donor = sum(row["fallback"] for row in report["controls"].values())
    text.extend(["", f"错配回流逐拍增量范数最大偏差为 {max_norm:.2g}，采用固定轴 {zero_donor} 次。"
                       "错配作用于主区收到的整组回流增量。自然回流在主端点未显示稳定优势，整组替换无法单独辨别房间、终点或数值的因果贡献。", ""])
    (ev.EVIDENCE / "eval-results.md").write_text("\n".join(text), encoding="utf-8")
