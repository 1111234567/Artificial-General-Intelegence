"""Validate the frozen reroute ledgers and report paired behavioral evidence."""

import numpy as np

from . import interleave_revisit_train as storage
from . import reroute_eval as run
from . import reroute_eval_stats as stats
from . import reroute_world as world


MODES = (*run.PROBE_MODES, *run.SHORTCUTS)
SHARED = (
    "case", "root", "truth", "prior_truth", "truth_room", "truth_endpoint", "goals",
    "door_rooms", "prior_endpoints", "inspected_room", "observed_endpoint", "query_draw", "action_draw",
)


def check_source(raw, path, sources, checkpoint, kind, birth):
    if (dict(zip(raw["source_names"], raw["source_sha256"], strict=True)) != sources
            or raw["checkpoint"].item() != checkpoint or raw["kind"].item() != kind
            or raw["birth"].item() != birth):
        raise AssertionError(f"Ledger provenance differs from the frozen inputs at {path}")


def independent_case_masks(raw, roots):
    pay, old = raw["truth"], raw["prior_truth"]
    ref_room = raw["door_rooms"]
    old_endpoint = np.take_along_axis(raw["prior_endpoints"], ref_room, axis=1)
    is_inspected = ref_room == raw["inspected_room"][:, None]
    ref_end = np.where(is_inspected, raw["observed_endpoint"][:, None], old_endpoint)
    np.testing.assert_array_equal(ref_end, raw["truth_endpoint"])
    np.testing.assert_array_equal(ref_room, raw["truth_room"])
    np.testing.assert_array_equal(pay, np.take_along_axis(raw["goals"], ref_end, axis=1))
    np.testing.assert_array_equal(old, np.take_along_axis(raw["goals"], old_endpoint, axis=1))
    changed = np.any(old_endpoint != ref_end, axis=1)
    tie = pay[:, 0] == pay[:, 1]
    prior_tie = old[:, 0] == old[:, 1]
    if prior_tie.any() or not np.array_equal(tie, ref_end[:, 0] == ref_end[:, 1]):
        raise AssertionError("The panel must begin with strict optima and ties must have a common endpoint")
    switch = old.argmax(1) != pay.argmax(1)
    masks = {
        "flip": changed & ~tie & switch,
        "stay": changed & ~tie & ~switch,
        "unchanged": ~changed,
        "tie": tie,
    }
    old_side, new_side = old.argmax(axis=1), pay.argmax(axis=1)
    inspected_side = is_inspected.argmax(axis=1)
    root_roles, role_counts = np.unique(raw["prior_endpoints"][::run.REPEATS], axis=0, return_counts=True)
    counts = {"root_endpoint_role_orders": {
        str(tuple(int(x) for x in order)): int(count) for order, count in zip(root_roles, role_counts)
    }}
    for index, (key, mask) in enumerate(masks.items()):
        np.testing.assert_array_equal(mask, raw["case"] == index)
        per_root = np.bincount(raw["root"][mask])
        np.testing.assert_array_equal(per_root, np.full(roots, (16, 32, 48, 48)[index]))
        np.testing.assert_array_equal(np.bincount(old_side[mask]), [mask.sum() // 2] * 2)
        np.testing.assert_array_equal(np.bincount(inspected_side[mask]), [mask.sum() // 2] * 2)
        # Reversing the two goal preferences preserves membership in this stratum.
        ordered, frequencies = np.unique(pay[mask], axis=0, return_counts=True)
        mixture = {tuple(int(x) for x in key): int(value) for key, value in zip(ordered, frequencies)}
        if any(mixture.get(tuple(-y for y in pair)) != count for pair, count in mixture.items()):
            raise AssertionError("The signed outcomes are not balanced in a test stratum")
        counts[key] = dict(
            per_root=int(per_root[0]), cases_per_birth=int(mask.sum()),
            old_optimal_sides=np.bincount(old_side[mask]).tolist(),
            inspected_physical_sides=np.bincount(inspected_side[mask]).tolist(),
            terminal_payoff_pairs={str(k): v for k, v in mixture.items()},
        )
        if key != "tie":
            np.testing.assert_array_equal(np.bincount(new_side[mask]), [mask.sum() // 2] * 2)
            counts[key]["current_optimal_sides"] = np.bincount(new_side[mask]).tolist()
    for value in (-1, 0, 1):
        masks[f"tie_{value}"] = tie & (pay[:, 0] == value)
    return masks, counts


def motor_metrics(raw, mode, masks):
    common = np.ones_like(masks["tie"])
    strict = ~masks["tie"]
    side = raw[f"{mode}/action"]
    paid = raw[f"{mode}/pay"]
    row = np.arange(len(side))
    if not np.isin(side, (0, 1)).all():
        raise AssertionError("An action must commit one physical door")
    for field, reference in (("room", "truth_room"), ("endpoint", "truth_endpoint"), ("pay", "truth")):
        np.testing.assert_array_equal(raw[f"{mode}/{field}"], raw[reference][row, side])
    if mode in run.PROBE_MODES:
        np.testing.assert_array_equal(side, raw["action_draw"] < raw[f"{mode}/action_p"])
        if mode != "last_only":
            np.testing.assert_array_equal(raw[f"{mode}/requested"], raw["query_draw"] < raw["query_p"])
    if mode == "old_best":
        np.testing.assert_array_equal(side, raw["prior_truth"].argmax(axis=1))

    best = paid == raw["truth"].max(axis=1)
    metrics = {
        "strict_optimal": (best, strict),
        "all_pay": (paid, common),
        "all_positive": (paid > 0, common),
        "all_negative": (paid < 0, common),
    }
    for label, mask in masks.items():
        metrics[label + "_optimal"] = (best, mask)
        metrics[label + "_pay"] = (paid, mask)
        metrics[label + "_right"] = (side == 1, mask)
    return metrics


def forecast_metrics(raw, mode, masks):
    target = raw["truth"]
    forecast, middle = (raw[f"{mode}/{name}"] for name in ("forecast", "mid"))
    room = 2 * raw["truth_room"] - 1
    strict = ~masks["tie"]
    common = np.ones_like(strict)
    if any(a.shape != target.shape or not np.isfinite(a).all() for a in (forecast, middle)):
        raise AssertionError("Bad frozen prediction trajectory shape or activity")
    true_diff = target[:, 1] - target[:, 0]
    signed_gap = (forecast[:, 1] - forecast[:, 0]) * np.sign(true_diff)
    rank = (signed_gap > 0) + 0.5 * (signed_gap == 0)
    observed = middle * room
    room_score = (observed > 0) + 0.5 * (observed == 0)
    request = raw[f"{mode}/requested"]
    row = np.arange(len(request))
    action = raw[f"{mode}/action"]
    unseen = 1 - action
    error = (forecast - target) ** 2
    result = {
        "forecast_rank": (rank, strict),
        "forecast_margin": (signed_gap, strict),
        "forecast_mse": (error.mean(axis=1), common),
        "forecast_mse_zero": ((target ** 2).mean(axis=1), common),
        "unwalked_forecast_mse": (error[row, unseen], common),
        "queried_forecast_mse": (error[row, request], common),
        "mid_accuracy": (room_score.mean(axis=1), common),
        "queried_mid_accuracy": (room_score[row, request], common),
        "mid_mse": (((middle - room) ** 2).mean(axis=1), common),
        "mid_mse_zero": ((room ** 2).mean(axis=1), common),
    }
    for case in ("flip", "stay", "unchanged"):
        result[f"forecast_rank_{case}"] = (rank, masks[case])
    return result


def compare_sources():
    contrasts = [
        ("next_room/natural", "terminal/natural"),
        ("next_room/natural", "misbound/natural"),
        ("terminal/natural", "misbound/natural"),
    ]
    for kind in run.fit.KINDS:
        contrasts.extend((f"{kind}/natural", f"{kind}/{mode}") for mode in MODES if mode != "natural")
    return contrasts


def aggregate_world(seed, roots, sources):
    shared = None
    summaries, fingerprints = {}, {}
    fallback = 0
    max_norm_error = 0.
    for kind in run.fit.KINDS:
        for birth in run.fit.BIRTHS:
            checkpoint = run.digest(run.fit.destination(kind, birth))
            calibration = run.cal_for(kind, birth)
            cal_digest = run.digest(calibration)
            with np.load(calibration, allow_pickle=False) as raw:
                check_source(raw, calibration, sources, checkpoint, kind, birth)
                if (int(raw["seed"]), int(raw["roots"])) != run.CALIBRATION:
                    raise AssertionError("Unexpected template calibration world")
            path = run.path_for(kind, birth, seed)
            name = f"{kind}/{birth}"
            fingerprints[name] = dict(ledger=run.digest(path), checkpoint=checkpoint, calibration=cal_digest)
            with np.load(path, allow_pickle=False) as archive:
                check_source(archive, path, sources, checkpoint, kind, birth)
                raw = {key: archive[key] for key in archive.files}
            if ((int(raw["seed"]), int(raw["roots"])) != (seed, roots)
                    or raw["calibration_sha256"].item() != cal_digest):
                raise AssertionError("A ledger used the wrong world or side template")
            fallback += int(raw["fallback"])
            max_norm_error = max(max_norm_error, float(raw["norm_error"]))
            if raw["post_change_door_walks_before_decision"].any():
                raise AssertionError("Unexpected external access between local observation and decision")
            if shared is None:
                shared = {name: raw[name] for name in SHARED}
                np.testing.assert_array_equal(raw["root"], np.repeat(np.arange(roots), run.REPEATS))
                masks, case_counts = independent_case_masks(raw, roots)
            else:
                for column in SHARED:
                    np.testing.assert_array_equal(raw[column], shared[column])

            for mode in MODES:
                group = summaries.setdefault(f"{kind}/{mode}", {})
                metrics = motor_metrics(raw, mode, masks)
                if mode in run.PROBE_MODES:
                    metrics.update(forecast_metrics(raw, mode, masks))
                for endpoint, (values, mask) in metrics.items():
                    entries, expected = group.setdefault(endpoint, ([], mask))
                    if not np.array_equal(mask, expected):
                        raise AssertionError("Paired metric masks changed across frozen births")
                    entries.append(values)
    bank = {
        name: {key: stats.measurement(np.stack(rows), mask, shared["root"])
               for key, (rows, mask) in endpoints.items()}
        for name, endpoints in summaries.items()
    }
    report = stats.statistics(bank, compare_sources(), seed)
    return dict(**report, roots=roots, case_counts=case_counts, raw_sha256=fingerprints,
                max_incremental_norm_error=max_norm_error, fixed_axis_fallback_ticks=fallback,
                postchange_door_retraversals=0)


def percentage(rate):
    return f'{100 * rate["mean"]:.2f}%'


def interval(rate, scale=100, places=2):
    mean, (lo, hi) = rate["mean"], rate["ci95"]
    return f"{scale * mean:+.{places}f} [{scale * lo:+.{places}f}, {scale * hi:+.{places}f}]"


def passes(condition):
    return "通过" if condition else "未通过"


def write_readable(report):
    worlds = report["worlds"]
    gates = report["gates"]
    seeds = tuple(worlds)
    lines = [
        "# 局部改道冻结行为结果",
        "",
        f"六个独立出生 731–736，留出世界一、二分别使用种子 {seeds[0]}、{seeds[1]}。地图与行动抽签独立于训练。两门到两房有两种映射，原房到不同终点有六种映射。根地图按有限标签置换，各根遍历其余因素，故 128 个随机根只覆盖这类地图，训练与留出在同一支持域。",
        "区间按同出生同根地图配对，交叉重采样 8192 次。每世界翻转 12,288 次，真实改道但优门保留 24,576 次，无改道 36,864 次，同归 36,864 次。每项以首次真实走门结算为准。",
        "",
        f"额外教下一房相对只教终局的总体教学动作门槛 **{passes(gates['teaching_action']['pass_both'])}**。",
    ]
    for kind in run.fit.KINDS:
        prediction = passes(gates["conditional_prediction"][kind]["pass_both"])
        action = passes(gates["actual_trial_gain"][kind]["pass_both"])
        lines.append(f"- `{kind}` 的条件预测门槛{prediction}，同身体试探的动作收益门槛{action}。")
    lines.extend([
        "",
        "## 真实首次选优",
        "",
        "| 教学 / 模式 | 翻转一 | 翻转二 | 保留一 | 保留二 | 无改道一 | 无改道二 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for kind in run.fit.KINDS:
        for mode in MODES:
            label = f"{kind}/{mode}"
            cells = [percentage(worlds[seed]["rates"][label][endpoint])
                     for endpoint in ("flip_optimal", "stay_optimal", "unchanged_optimal") for seed in seeds]
            lines.append("| " + " | ".join((f"`{label}`", *cells)) + " |")
    lines.extend([
        "",
        "同归的负、零、正各 12,288 次。所有教学及干预两门均按各档得到相同结算，实际最优比例均为 100%。同归行保留在整体收益和预测误差分母。",
        "",
        "## 配对动作差与区间（百分点）",
        "",
        "| 实际动作差 | 翻转一 | 翻转二 | 保留一 | 保留二 |",
        "|---|---:|---:|---:|---:|",
    ])
    for one, two in compare_sources():
        label = f"{one} minus {two}"
        cells = [interval(worlds[seed]["contrasts"][label][endpoint])
                 for endpoint in ("flip_optimal", "stay_optimal") for seed in seeds]
        lines.append("| " + " | ".join((f"`{one}` − `{two}`", *cells)) + " |")
    lines.extend([
        "",
        "## 强制内部试左、试右的预测",
        "",
        "终局排序只计严格优门的列，分差相同时给半分。终局 MSE 包含同归，以真实收益为目标。下一房符号正确率以两路实际经过的房标计算；概率恰平取半分。",
        "",
        "| 教学 / 模式 | 排序一 | 排序二 | 终局 MSE 一 | 终局 MSE 二 | 下一房一 | 下一房二 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for kind in run.fit.KINDS:
        for mode in run.PROBE_MODES:
            label = f"{kind}/{mode}"
            rates = [worlds[seed]["rates"][label] for seed in seeds]
            cells = [*(percentage(rate["forecast_rank"]) for rate in rates),
                     *(f'{rate["forecast_mse"]["mean"]:.4f}' for rate in rates),
                     *(percentage(rate["mid_accuracy"]) for rate in rates)]
            lines.append("| " + " | ".join((f"`{label}`", *cells)) + " |")
    lines.extend([
        "",
        "## 条件预测的配对差",
        "",
        "| 比较 | 排序一（百分点） | 排序二（百分点） | MSE 一 | MSE 二 | 下一房一（百分点） | 下一房二（百分点） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for one, two in compare_sources():
        if two.split("/")[-1] not in run.PROBE_MODES:
            continue
        label = f"{one} minus {two}"
        cells = [interval(worlds[seed]["contrasts"][label][endpoint], scale, places)
                 for endpoint, scale, places in (("forecast_rank", 100, 2), ("forecast_mse", 1, 4),
                                                 ("mid_accuracy", 100, 2)) for seed in seeds]
        lines.append("| " + " | ".join((f"`{one}` − `{two}`", *cells)) + " |")
    lines.extend([
        "",
        "身体在最后一次局部观察时没有经过任何起点门。其后，原生决策窗口只推进目标输入、准备和内部空转，至正式选择才开始实际走门。",
        "冻结评估核对了真实房间、终点与收益。原回流重注入和零侧别模板在每种模型与世界的首批逐位重现原轨迹。左、右真实回流的错误供体保留接收者逐拍增量范数。",
        "",
        "[冻结方案](eval-plan.md) · [完整分母、六出生、区间与哈希](eval-results.json)",
    ])
    path = run.EVIDENCE / "eval-results.md"
    storage._atomic(path, lambda temporary: temporary.write_text("\n".join(lines) + "\n", encoding="utf8"))


def summarize():
    sources = run.fingerprints()
    worlds = {str(seed): aggregate_world(seed, roots, sources) for seed, roots in run.TESTS}
    report = {"births": run.fit.BIRTHS, "calibration": run.CALIBRATION, "bootstrap_draws": stats.BOOTSTRAP_DRAWS,
              "sources_sha256": sources, "worlds": worlds, "gates": stats.evaluate_gates(worlds, run.fit.KINDS)}
    storage._write_json(run.EVIDENCE / "eval-results.json", report)
    write_readable(report)
    print(run.EVIDENCE / "eval-results.md", flush=True)
