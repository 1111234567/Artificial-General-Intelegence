"""Audit actual contact and summarize common-draw state swaps by paired birth."""

import numpy as np
import torch

from . import double_visit_eval_report as behavior
from . import double_visit_fit as fit
from . import double_visit_trace_swap as experiment
from . import double_visit_world as world


def verify(data, condition, birth):
    panel, pairs, maps, heldout, tape = experiment.layout()
    payload = world.oracle(panel[pairs.flatten()])
    best = payload.payoffs.argmax(1).reshape(-1, 2).numpy()
    sources = {"condition": condition, "birth": birth,
               "checkpoint_sha256": fit.digest(fit.destination(birth, condition)),
               "plan_sha256": fit.digest(experiment.PLAN), "source_sha256": fit.digest(experiment.__file__)}
    for key, value in sources.items():
        np.testing.assert_equal(data[key].item(), value)
    for key, value in {"pairs": pairs, "old_map": maps, "heldout": heldout, "tape": tape,
                       "wanted": best, "stages": experiment.STAGES, "modes": experiment.MODES}.items():
        np.testing.assert_array_equal(data[key], value)
    query, action = data["query_p"], data["action_p"]
    want = (len(experiment.STAGES), len(experiment.MODES), len(pairs), 2)
    np.testing.assert_equal(query.shape, want)
    np.testing.assert_equal(action.shape, (*want, 2))
    observed_query = (tape[:, 0].numpy()[:, None] < query).astype(np.intp)
    np.testing.assert_array_equal(data["trial_side"], observed_query)
    p_action = np.take_along_axis(action, observed_query[..., None], axis=-1)[..., 0]
    observed_action = (tape[:, 1].numpy()[:, None] < p_action).astype(np.intp)
    np.testing.assert_array_equal(data["real_door"], observed_action)
    for kind, truth in (("room", payload.rooms), ("endpoint", payload.endpoints), ("paid", payload.payoffs)):
        table = np.broadcast_to(truth.numpy().reshape(len(pairs), 2, 2), action.shape)
        observed = np.take_along_axis(table, observed_action[..., None], axis=-1)[..., 0]
        np.testing.assert_array_equal(data[f"real_{kind}"], observed)
    for key in ("query_p", "action_p", "trial_side", "real_door", "real_room", "real_endpoint", "real_paid"):
        value = data[key]
        np.testing.assert_array_equal(value[0, 0], value[1, 0])
        np.testing.assert_array_equal(value[:, 0], value[:, 1])
        if key in ("query_p", "action_p", "trial_side", "real_door"):
            np.testing.assert_array_equal(value[:, 4], np.flip(value[:, 0], axis=2))
    return best, heldout.numpy()


def conditioned(data, best, mask):
    n = len(best)
    pairs = np.arange(2 * n).reshape(n, 2)
    record = {}
    for stage_id, stage in enumerate(experiment.STAGES):
        record[stage] = {}
        for mode_id, mode in enumerate(experiment.MODES):
            q = data["query_p"][stage_id, mode_id].reshape(-1)
            p = data["action_p"][stage_id, mode_id].reshape(-1, 2)
            exact = behavior.integrate_pair(q, p, best.flatten(), pairs)
            hits = data["real_door"][stage_id, mode_id] == best
            actual_both = hits.all(1)
            actual_neither = (~hits).all(1)
            record[stage][mode] = {
                "both_correct": float(exact["both"][mask].mean()),
                "both_wrong": float(exact["neither"][mask].mean()),
                "direction_correct_minus_wrong": float(exact["direction"][mask].mean()),
                "a_correct": float(exact["first"][mask].mean()), "b_correct": float(exact["second"][mask].mean()),
                "sample_both_correct": float(actual_both[mask].mean()),
                "sample_both_wrong": float(actual_neither[mask].mean()),
                "sample_mean_paid": float(data["real_paid"][stage_id, mode_id][mask].mean()),
            }
        start = record[stage]["natural"]
        for row in record[stage].values():
            row["both_correct_change"] = row["both_correct"] - start["both_correct"]
            row["a_action_shift_toward_donor"] = start["a_correct"] - row["a_correct"]
            row["b_action_shift_toward_donor"] = start["b_correct"] - row["b_correct"]
    return record


def stage_contrast(record):
    before, after = (record[stage] for stage in experiment.STAGES)
    fields = {
        "both_correct_loss": ("both_correct_change", -1.),
        "a_action_shift_toward_donor": ("a_action_shift_toward_donor", 1.),
        "b_action_shift_toward_donor": ("b_action_shift_toward_donor", 1.),
    }
    return {key: direction * ((after["fast_candidate"][metric] - after["activity_trace"][metric])
                              - (before["fast_candidate"][metric] - before["activity_trace"][metric]))
            for key, (metric, direction) in fields.items()}


def summarize():
    rng = np.random.default_rng(experiment.BOOT_SEED)
    indices = rng.integers(len(fit.BIRTHS), size=(8192, len(fit.BIRTHS)))

    def stats(values):
        values = np.asarray(values)
        return {"mean": float(values.mean()), "interval95": np.quantile(values[indices].mean(1), (.025, .975)).tolist()}

    report = {"plan_sha256": fit.digest(experiment.PLAN), "run_source_sha256": fit.digest(experiment.__file__),
              "report_source_sha256": fit.digest(__file__), "integrator_source_sha256": fit.digest(behavior.__file__),
              "fit_protocol_sha256": fit.digest(fit.PROTOCOL), "births": fit.BIRTHS,
              "pairs_per_birth_per_split": 144, "sample_seed": experiment.SAMPLE_SEED,
              "bootstrap_seed": experiment.BOOT_SEED, "bootstrap_samples": len(indices),
              "audit": {"self_return_bitwise": True, "complete_donor_state_replay_bitwise": True,
                        "physical_receiver_routes_verified": True}, "conditions": {}}
    for condition in fit.CONDITIONS:
        cohorts = {}
        for birth in fit.BIRTHS:
            file = experiment.path(condition, birth)
            with np.load(file, allow_pickle=False) as data:
                best, heldout = verify(data, condition, birth)
                per_map = {}
                for map_id in np.unique(data["old_map"][heldout]):
                    mask = data["old_map"] == map_id
                    np.testing.assert_equal(mask.sum(), 48)
                    per_map[str(map_id)] = stage_contrast(conditioned(data, best, mask))
                average = {key: float(np.mean([row[key] for row in per_map.values()]))
                           for key in next(iter(per_map.values()))}
                heldout_summary = conditioned(data, best, heldout)
                for key, value in stage_contrast(heldout_summary).items():
                    np.testing.assert_allclose(value, average[key], rtol=0., atol=1e-15)
                cohorts[str(birth)] = {"file": file.relative_to(fit.ROOT).as_posix(), "sha256": fit.digest(file),
                                       "heldout": heldout_summary, "heldout_stage_contrast": average,
                                       "heldout_stage_contrast_each_old_map": per_map,
                                       "seen": conditioned(data, best, ~heldout)}
        metrics = tuple(cohorts[str(fit.BIRTHS[0])]["heldout"]["after_first"]["natural"])
        results = {split: {stage: {mode: {metric: stats([
            record[split][stage][mode][metric] for record in cohorts.values()
        ]) for metric in metrics} for mode in experiment.MODES} for stage in experiment.STAGES}
                   for split in ("heldout", "seen")}
        interactions = {}
        keys = cohorts[str(fit.BIRTHS[0])]["heldout_stage_contrast"]
        for key in keys:
            values = [r["heldout_stage_contrast"][key] for r in cohorts.values()]
            interactions[key] = {**stats(values), "sign_by_birth": dict(zip(map(str, fit.BIRTHS),
                                                                         np.sign(values).astype(int).tolist(), strict=True))}
        report["conditions"][condition] = {"cohorts": cohorts, "heldout_stage_contrast": interactions, **results}
    fit.prior_fit.storage._write_json(fit.EVIDENCE / "trace-swap-results.json", report)
    return report


def write_markdown(report):
    def points(stat, *, interval=True):
        mean, (lo, hi) = stat["mean"], stat["interval95"]
        return f"{mean * 100:.2f} [{lo * 100:.2f}, {hi * 100:.2f}]" if interval else f"{mean * 100:.2f}"

    lines = ["# 第一笔复查状态的因果交换", "",
             "六个冻结出生每组各 144 个奇置换旧图关键对。A 应选早复查房对应门，B 应选晚复查房对应门。每对仅首笔看见的终点不同。之后两份生命的感官输入逐位相同。表内为共同抽签的精确概率及六个配对出生自助法 95% 区间，单位均为百分点。A、B 正值表示受体更常选供体的最佳门。 [附加诊断方案](trace-swap-plan.md)", ""]
    mode_names = {"natural": "自然", "activity_trace": "换活动和迹", "fast_candidate": "换快权重和写入提议",
                  "all": "全换"}
    stage_names = {"after_first": "第 18 拍", "after_second": "第 24 拍"}
    for name, label in (("identity", "加入终点身份教学"), ("baseline", "只有下一房和收益教学")):
        lines += [f"## {label}", "", "| 交换时点 | 干预 | 配对双对率 | 双对率相对自然 | A 行动转向供体 | B 行动转向供体 |",
                  "|:--|:--|--:|--:|--:|--:|"]
        for stage, display in stage_names.items():
            for mode, word in mode_names.items():
                if mode == "natural" and stage == "after_second":
                    continue
                item = report["conditions"][name]["heldout"][stage][mode]
                cells = [points(item[key]) for key in ("both_correct", "both_correct_change",
                                                       "a_action_shift_toward_donor", "b_action_shift_toward_donor")]
                lines.append("| " + " | ".join([display, word, *cells]) + " |")
        lines.append("")
    lines += ["## 时点与状态分量的配对复核", "",
              "每个出生先在每张奇置换旧图的相同 48 对生命上计算换快权重和写入提议的双对率损失减去换活动和迹的双对率损失，再取第 24 拍相对第 18 拍的变化。三张旧图等权，区间只按六出生重采样。这项复算在状态交换之后追加。", "",
              "| 条件 | 快状态相对活动和迹的双对损失增量 | 六出生 731–736 的符号 |", "|:--|--:|:--|"]
    for name, label in (("identity", "加入终点身份教学"), ("baseline", "只有下一房和收益教学")):
        item = report["conditions"][name]["heldout_stage_contrast"]["both_correct_loss"]
        sign = "、".join({-1: "−", 0: "0", 1: "+"}[value] for value in item["sign_by_birth"].values())
        lines.append(f"| {label} | {points(item)} | {sign} |")
    lines.append("")
    lines += [
        "所有回注自身和全换的后续状态与马达概率均通过逐位对照。全换复现供体的选门过程，受体仍在自己的物理世界结算。账本将每个对、条件、交换时点和分量的原生试探、真实选门、房间、终点与报酬逐一复算。", "",
        "两组单换可能拼出自然身体很少访问的状态。交换得到的动作变化包含所移入分量、保留分量及后续递归的共同作用。迹随活动一起换，写入提议随快权重一起换，各自的贡献仍待分辨。这个时间差限于两次复查之间的动态变化，长期巩固需要另外验证。按出生给出完整方向性、双错率、实际抽签和偶置换旧图的结果。 [完整结果与原始账本散列](trace-swap-results.json)", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m experiments.pulse_game.double_visit_trace_swap baseline",
        ".\\.venv\\Scripts\\python.exe -m experiments.pulse_game.double_visit_trace_swap identity",
        ".\\.venv\\Scripts\\python.exe -m experiments.pulse_game.double_visit_trace_swap_report", "```", "",
    ]
    fit.atomic(fit.EVIDENCE / "trace-swap-results.md", lambda path: path.write_text("\n".join(lines), encoding="utf8"))


if __name__ == "__main__":
    torch.set_num_threads(1)
    write_markdown(summarize())
