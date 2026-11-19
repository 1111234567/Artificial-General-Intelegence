"""Summarize frozen lifetime and unified physical outcomes by paired world."""

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parents[2] / "data" / "pulse-game"
BOOTSTRAP_DRAWS = 6000
BOOTSTRAP_SEED = 202611190715
SELECTED = "selected_full_feedback"
MASKED = "selected_no_new_B_evidence"
LONG_CASES = {
    "stable": (2304, "stable", .05, "clustered", "smooth"),
    "reverse_return": (2304, "reverse_return", .05, "clustered", "smooth"),
    "noise_stable": (2304, "stable", .19, "clustered", "smooth"),
    "noise_reverse_return": (2304, "reverse_return", .19, "clustered", "smooth"),
    "similar_habitats": (2304, "reverse_return", .08, "iid", "nearby"),
    "long_life": (9216, "reverse_return", .07, "clustered", "smooth"),
}
PHASES = ("pre", "post")
VISITS = ("1_1", "3_4", "9_16")
MOTION_SIDES = ("B_initial_side", "B_final_side", "A_return_side", "A_first_side", "B_first_side")


def read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=20)
def bootstrap_indices(n):
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, n]))
    return rng.integers(n, size=(BOOTSTRAP_DRAWS, n), dtype=np.int32)


def interval(draws):
    return np.quantile(draws, [.025, .975]).tolist()


@dataclass
class WorldMetric:
    roots: tuple
    models: dict
    opportunities: np.ndarray | None = None

    def averaged(self):
        rows = np.stack(list(self.models.values()))
        if rows.ndim != 2 or rows.shape[1] != len(self.roots) or rows.shape[0] != 3:
            raise ValueError("Expected 3 fixed models applied to all roots")
        return rows.mean(axis=0)

    def summary(self):
        value = self.averaged()
        ok = np.isfinite(value)
        a = value[ok]
        if len(a) == 0:
            return {"n_worlds": 0, "mean": None, "ci95": None}
        ix = bootstrap_indices(len(a))
        row = {"n_worlds": len(a), "mean": float(a.mean()), "ci95": interval(a[ix].mean(axis=1)),
               "fixed_model_means": {key: float(values[ok].mean()) for key, values in self.models.items()}}
        if self.opportunities is not None:
            w = self.opportunities[ok]
            hits = a * w
            row.update(requests_per_model=int(w.sum()), request_weighted_mean=float(hits.sum() / w.sum()),
                       request_weighted_ci95=interval(hits[ix].sum(axis=1) / w[ix].sum(axis=1)))
        return row


def paired(left, right):
    if left.roots != right.roots:
        raise ValueError("Paired contrasts must use the same world roots in the same order")
    x, y = left.averaged(), right.averaged()
    valid = np.isfinite(x) & np.isfinite(y)
    delta = (x - y)[valid]
    if not len(delta):
        return {"n_worlds": 0, "difference": None, "ci95": None}
    ix = bootstrap_indices(len(delta))
    return {"n_worlds": len(delta), "difference": float(delta.mean()), "ci95": interval(delta[ix].mean(axis=1))}


def roots_equal(packets, field):
    values = [tuple(row[field]) for row in packets.values()]
    if len(set(values)) != 1 or len(set(values[0])) != len(values[0]):
        raise ValueError("The physical world roots are not in a common, duplicate-free order")
    return values[0]


def long_lives(sealed, selection):
    manifest, manifest_hash = read(sealed / "complete.json")
    names = tuple(manifest["parents"])
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("Expect three fixed long-life models")
    files = {str(sealed / "complete.json"): manifest_hash}
    winners = {}
    for name in names:
        path = selection / (name + ".json")
        record, files[str(path)] = read(path)
        index = record["global_validation_winner"]
        matches = [arm for arm, info in record["settings"].items() if info["index"] == index]
        if len(matches) != 1:
            raise ValueError(f"Cannot unambiguously resolve the development winner for {name}")
        winners[name] = {"arm": matches[0], "candidate_index": index, "parameters": record["candidates"][index]}

    all_scores = {}
    scenes = {}
    signatures = {}
    for case, (requests, mode, noise, order, appearance) in LONG_CASES.items():
        packs = {}
        for name in names:
            path = sealed / f"{name}-{case}.json"
            packs[name], files[str(path)] = read(path)
        roots = roots_equal(packs, "root_seeds")
        ref = packs[names[0]]
        if any(p["requests"] != requests or p["ticks"] != requests * 16 or p["boundaries"] != ref["boundaries"] for p in packs.values()):
            raise ValueError(f"Different shape or switching chronology in {case}")
        signatures[case] = (roots, ref["boundaries"])
        arms = tuple(ref["rows"]) + (SELECTED,)
        available = set(ref["rows"]["rigid"])
        scene = {"requests_per_world": requests, "blocks_per_world": ref["blocks"], "ticks_per_world": ref["ticks"],
                 "rule_schedule": mode, "non_probe_taste_flip_probability": noise, "habitat_order": order,
                 "habitat_sensation": appearance, "physical_roots": roots, "arms": {}}
        for arm in arms:
            region_table = {}
            for region in sorted(available):
                source = {name: packs[name]["rows"][winners[name]["arm"] if arm == SELECTED else arm][region]
                          for name in names}
                if arm == SELECTED:
                    for name in names:
                        p = packs[name]
                        selected = np.asarray(winners[name]["parameters"], np.float32)
                        if not np.array_equal(selected, p["parameters"][winners[name]["arm"]]):
                            raise ValueError(f"Selected settings do not match the sealed arm for {name}")
                        if MASKED in p["parameters"] and not np.array_equal(selected, p["parameters"][MASKED]):
                            raise ValueError(f"The masked feedback arm is not parameter paired for {name}")
                counts = np.asarray(source[names[0]]["count"], dtype=np.int64)
                if len(counts) != len(roots) or any(not np.array_equal(s["count"], counts) for s in source.values()):
                    raise ValueError(f"Probe opportunities differ among models in {case}, {region}")
                if not np.array_equal(counts, ref["rows"]["rigid"][region]["count"]):
                    raise ValueError(f"Probe opportunities differ among arms in {case}, {region}")
                region_table[region] = {}
                for phase in PHASES:
                    hits = {k: np.asarray(v[phase], dtype=np.float64) for k, v in source.items()}
                    if any(np.any((v < 0) | (v > counts)) for v in hits.values()):
                        raise ValueError(f"Hits outside opportunity counts in {case}, {region}, {phase}")
                    values = {k: np.divide(v, counts, out=np.full(counts.shape, np.nan), where=counts > 0)
                              for k, v in hits.items()}
                    observation = WorldMetric(roots, values, counts)
                    all_scores[(case, arm, region, phase)] = observation
                    region_table[region][phase] = observation.summary()
            scene["arms"][arm] = region_table
        scenes[case] = scene
    for high, low in (("noise_stable", "stable"), ("noise_reverse_return", "reverse_return")):
        if signatures[high] != signatures[low]:
            raise ValueError("The low- and high-noise schedules do not pair")
        for region in scenes[high]["arms"][SELECTED]:
            a = all_scores[(high, SELECTED, region, "pre")].opportunities
            b = all_scores[(low, SELECTED, region, "pre")].opportunities
            if not np.array_equal(a, b):
                raise ValueError(f"The high- and low-noise opportunities do not pair for {region}")
    if signatures["long_life"][0] != signatures["reverse_return"][0][:64]:
        raise ValueError("Unexpected roots in the 9,216-request subset")

    effects = {}
    for case, scene in scenes.items():
        def difference(arm, other):
            return {k: {p: paired(all_scores[(case, arm, k, p)], all_scores[(case, other, k, p)]) for p in PHASES}
                    for k in scene["arms"][arm]}

        effects[case] = {"selected_minus_rigid": difference(SELECTED, "rigid")}
        if MASKED in scene["arms"]:
            effects[case]["full_feedback_minus_hidden_B"] = difference(SELECTED, MASKED)
        effects[case]["selected_post_minus_pre"] = {
            k: paired(all_scores[(case, SELECTED, k, "post")], all_scores[(case, SELECTED, k, "pre")])
            for k in scene["arms"][SELECTED]}
    noise_effects = {}
    for high, low in (("noise_stable", "stable"), ("noise_reverse_return", "reverse_return")):
        noise_effects[high] = {arm: {
            k: {phase: paired(all_scores[(high, arm, k, phase)], all_scores[(low, arm, k, phase)])
                for phase in PHASES}
            for k in scenes[high]["arms"][arm]} for arm in (SELECTED, "rigid")}
    return {"frozen_models": names, "development_selected_arm_by_model": winners, "scenes": scenes,
            "within_scene_paired": effects, "high_minus_low_noise_paired": noise_effects,
            "hidden_B_feedback": {
                "starts_at": "first change boundary",
                "covered": ["pre-teaching anchor taste", "paired teaching tastes", "post-teaching anchor taste",
                            "all B visits after the original rule is restored"],
                "A_feedback": "unmodified", "test_probes": "silent in both arms",
                "interpretation": "combined B-feedback ablation; does not isolate paired-relation teaching",
            }}, files


def unified(sealed):
    complete, manifest_hash = read(sealed / "complete.json")
    files = {str(sealed / "complete.json"): manifest_hash}
    modes = {}
    vectors = {}
    first_probe = {}
    roots = tuple(complete["roots"])
    for mode in ("full", "motor"):
        packs = {}
        for path in sorted(sealed.glob(f"*--{mode}-*.json")):
            d, files[str(path)] = read(path)
            key = d["settings"]["model"]
            if key in packs:
                raise ValueError("Duplicate model record")
            packs[key] = d
        if len(packs) != 3 or roots_equal(packs, "seeds") != roots:
            raise ValueError("Wrong unified cohort")
        names = tuple(packs)
        reference = packs[names[0]]
        if any(p["order"] != ["++", "+-", "-+", "--"] or p["settings"]["mode"] != mode for p in packs.values()):
            raise ValueError("Modes or hidden family order do not match")
        by_case = {}
        for case, arms in reference["cases"].items():
            codes = {"overall": (0, 1), "family_plus": (0,), "family_minus": (1,)} if case == "shared_transfer_game" else {
                "overall": (0, 1, 2, 3), "shared": (0, 3), "conflict": (1, 2)}
            arm_table = {}
            for arm, samples in arms.items():
                group_table = {k: {} for k in codes}
                for metric in samples:
                    packed = {key: np.asarray(packs[key]["cases"][case][arm][metric], dtype=np.float64) for key in names}
                    if metric in MOTION_SIDES:
                        if metric == "B_initial_side":
                            for key, a in packed.items():
                                if not (np.array_equal(a[0], a[1]) and np.array_equal(a[2], a[3])):
                                    raise ValueError(f"Unseen B relation changed the recorded first motion for {key}")
                            first_probe[(mode, case, arm)] = True
                        packed = {k: v >= 0 for k, v in packed.items()}
                        metric = metric + "_contact"
                    for group, inds in codes.items():
                        per_root = {key: samples[list(inds)].mean(axis=0) for key, samples in packed.items()}
                        metric_worlds = WorldMetric(roots, per_root)
                        vectors[(mode, case, arm, group, metric)] = metric_worlds
                        group_table[group][metric] = metric_worlds.summary()
                arm_table[arm] = group_table
            by_case[case] = {"arms": arm_table, "within_case_paired": {}}
            for arm in arm_table:
                if arm == "selected":
                    continue
                by_case[case]["within_case_paired"][f"selected_minus_{arm}"] = {
                    group: {key: paired(vectors[(mode, case, "selected", group, key)], vectors[(mode, case, arm, group, key)])
                            for key in arm_table[arm][group]} for group in codes}
            if "shared" in codes:
                by_case[case]["shared_minus_conflict"] = {
                    arm: {metric: paired(vectors[(mode, case, arm, "shared", metric)], vectors[(mode, case, arm, "conflict", metric)])
                          for metric in arm_table[arm]["shared"]} for arm in arm_table}
        modes[mode] = {"three_fixed_models": names, "scenes": by_case, "cpu_gpu_parity": {k: v["parity"] for k, v in packs.items()}}
    if modes["full"]["three_fixed_models"] != modes["motor"]["three_fixed_models"]:
        raise ValueError("The motor and full genotypes do not pair by training identity")
    compact = {}
    for case, value in modes["full"]["scenes"].items():
        compact[case] = {side: {
            metric: paired(vectors[("full", case, "selected", side, metric)], vectors[("motor", case, "selected", side, metric)])
            for metric in value["arms"]["selected"][side]} for side in value["arms"]["selected"]}
    return {"physical_roots": roots, "counterfactual_family_order": ["++", "+-", "-+", "--"],
            "confirmation_prior_shared": .50, "training_prior_shared": .77,
            "first_B_probe": "one B anchor request (visible taste if contacted) precedes it; no B relation pair offered",
            "first_B_paired_side_equal": all(first_probe.values()), "modes": modes,
            "selected_full_minus_motor": compact}, files


def percent(a, digits=1):
    return f'{a["mean"] * 100:.{digits}f}%'


def pp(a, digits=1):
    lo, hi = [x * 100 for x in a["ci95"]]
    return f'{a["difference"] * 100:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'


def report_text(data):
    long = data["long_life"]
    new = data["unified_physical"]["modes"]["full"]["scenes"]
    standard = new["standard"]["arms"]["selected"]
    big = new["long_competing_run"]["arms"]
    short = long["scenes"]["reverse_return"]["arms"]
    extended = long["scenes"]["long_life"]["arms"]
    noise = long["high_minus_low_noise_paired"]["noise_reverse_return"][SELECTED]
    stable_noise = long["high_minus_low_noise_paired"]["noise_stable"][SELECTED]
    release_cost = long["within_scene_paired"]["noise_stable"]["selected_minus_rigid"]["stable_A"]["post"]
    end_a = new["long_competing_run"]["within_case_paired"]["selected_minus_parent"]["conflict"]["return_A"]
    robust = new["noisy"]["arms"]["selected"]["conflict"]
    lines = [
        "# 长生命规则改变与统一网络",
        "",
        "每个世界先平均三个冻结模型，再以世界根配对自举 6,000 次。两个长生命协议分别有 192 与 64 个世界，统一场景有 512 个。超时计入分母。下表每格为访问内配对教学前→后的静默探题正确率。",
        "",
        "| 每生请求 | B 规则阶段 | 首次访问 | 第 3–4 次 | 第 9–16 次 |",
        "|---:|---|---:|---:|---:|",
    ]
    for scenes, label in [(short, "2,304"), (extended, "9,216")]:
        for phase, display in [("change", "翻转"), ("return", "恢复")]:
            cells = ["→".join(percent(scenes[SELECTED][f"{phase}_{stage}"][p]) for p in PHASES) for stage in VISITS]
            lines.append("| " + " | ".join((label, display, *cells)) + " |")
    def stable_pair(scenes):
        return "→".join(percent(scenes[SELECTED]["stable_A"][phase]) for phase in PHASES)

    lines.extend([
        "",
        (f'翻转后第 9–16 次 B，常规 RLS 更新组的教学后正确率分别为 '
         f'{percent(short["rigid"]["change_9_16"]["post"])} 和 {percent(extended["rigid"]["change_9_16"]["post"])}。'
         f'从首次翻转起隐藏所有 B 回执，同窗口分别为 {percent(short[MASKED]["change_9_16"]["post"])} 和 '
         f'{percent(extended[MASKED]["change_9_16"]["post"])}。这个控制同时遮锚味与关系教学，恢复原规则后也继续遮 B。'
         '全反馈采用各模型在开发阶段选定的释放机制，两份全局释放，一份按回执意外程度调制全局释放。'),
        "",
        (f'全反馈下，翻转后仍稳定的 A 教学前→后为 {stable_pair(short)} 和 {stable_pair(extended)}。'
         f'在 2,304 请求中，回执噪声由 5% 增至 19%，翻转后 A 教学后变化 {pp(noise["stable_A"]["post"])} 个百分点，'
         f'B 第 9–16 次教学后变化 {pp(noise["change_9_16"]["post"])}。'
         f'全程稳定时，A 教学后的噪声代价为 {pp(stable_noise["stable_A"]["post"])}，'
         f'同一高噪声下，选定释放机制相对常规 RLS 的差为 {pp(release_cost)} 个百分点。'
         '方括号内是 95% 区间。9,216 请求用 7% 噪声。'),
        "",
        ('统一网络训练时 A/B 同规则的先验为 77%，确认时同规则和冲突各半。'
         '首个 B 探题前有一道锚题，接触可尝到锚味，尚未提供 B 关系配对。'
         f'同规则与冲突的首 B 分别为 {percent(standard["shared"]["B_after_0"])} 和 '
         f'{percent(standard["conflict"]["B_after_0"])}。提供 12 对后冲突组为 '
         f'{percent(standard["conflict"]["B_after_12"])}。长 B 条件提供 48 对，冲突组末 B 为 '
         f'{percent(big["selected"]["conflict"]["B_after_48"])}，回 A 为 {percent(big["selected"]["conflict"]["return_A"])}，'
         f'相对亲本差 {pp(end_a)} 个百分点，两题均正确 {percent(big["selected"]["conflict"]["both"])}。'
         f'长 B 同时改变噪声和配对跨度。高噪声、强气味扰动组合下，冲突组末 B 为 {percent(robust["B_after_16"])}。'),
        "",
        "[完整区间、分母、各臂与校验和](lifetime-review-statistics.json) · [中文图](lifetime-review-figure.png)",
        "",
    ])
    return "\n".join(lines)


def draw_figure(data, path, font_path=None):
    import os
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    if font_path is None:
        windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates = (windows / "msyh.ttc", windows / "simhei.ttf",
                      Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
        font_path = next((p for p in candidates if p.is_file()), None)
    if font_path is None or not font_path.is_file():
        raise ValueError("A Chinese font is required for the scientific figure; pass --font")
    font_manager.fontManager.addfont(str(font_path))
    name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": name, "axes.unicode_minus": False, "font.size": 11.5,
                         "axes.edgecolor": "#a8b1bd", "axes.labelcolor": "#303d4e",
                         "xtick.color": "#405269", "ytick.color": "#405269"})
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.0))
    fig.subplots_adjust(left=.148, right=.962, bottom=.14, top=.81, wspace=.29, hspace=.55)
    fig.set_facecolor("#fcfdff")
    ink = "#18273d"
    short_color, long_color = "#147f75", "#4458aa"
    gray, orange = "#6d7381", "#bb5739"
    fig.text(.148, .94, "长生命中的规则修正与回访", fontsize=23, weight="bold", color=ink)
    fig.text(.148, .894, "冻结策略在真实运动中的选择。长生命用 192 或 64 个世界，统一网络用 512 个世界。",
             fontsize=11.5, color="#435367")
    fig.text(.148, .862, "先在每个世界平均三个固定模型。误差线为世界自举 95% 区间，全部机会包含超时。",
             fontsize=11.5, color="#435367")

    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.7, pad=5)

    def curve(ax, x, stats, label, color, marker, linestyle="-", fill=True):
        value = np.array([row["mean"] for row in stats]) * 100
        bounds = np.array([row["ci95"] for row in stats]).T * 100
        err = np.maximum(0.0, [value - bounds[0], bounds[1] - value])
        ax.errorbar(x, value, yerr=err, marker=marker, color=color, label=label, lw=1.9, ls=linestyle,
                    capsize=3.3, elinewidth=1.0, markersize=5.1, markeredgewidth=1.2,
                    markerfacecolor=color if fill else "white", zorder=3)

    long = data["long_life"]["scenes"]
    short = long["reverse_return"]["arms"]
    extended = long["long_life"]["arms"]
    x = np.arange(3)
    for phase, title, ax in (
        ("change", "A　B 规则翻转之后", axes[0, 0]),
        ("return", "B　B 恢复原规则之后", axes[0, 1]),
    ):
        ax.set_title(title, loc="left", color=ink, fontsize=14, weight="bold", pad=10)
        ax.axhline(50, color="#a1a9b7", lw=.8, ls=":", zorder=1)
        ax.grid(axis="y", color="#e8edf3", linewidth=.7)
        for scenes, color, offset, size, marker in (
            (short, short_color, -.044, "2,304", "o"), (extended, long_color, .044, "9,216", "s"),
        ):
            for period, label, linestyle, filled in (("pre", "教学前", "--", False), ("post", "教学后", "-", True)):
                rows = [scenes[SELECTED][f"{phase}_{stage}"][period] for stage in VISITS]
                curve(ax, x + offset, rows, f"{size} · {label}", color, marker, linestyle, filled)
        hidden = [extended[MASKED][f"{phase}_{stage}"]["post"] for stage in VISITS]
        curve(ax, x, hidden, "9,216 · 隐去 B 回执", gray, "D", ":", False)
        ax.set(ylim=(0, 104), xlim=(-.20, 2.22), ylabel="选择正确率（%）",
               xlabel="从本阶段起的 B 访问次数")
        ax.set_xticks(x, labels=["第 1 次", "第 3–4 次", "第 9–16 次"])
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0, 0].legend(loc="lower right", ncol=2, fontsize=9, frameon=False, columnspacing=1, handlelength=1.9)
    axes[0, 1].text(.99, .07, "灰线从首翻起遮所有 B 回执，含锚味。", transform=axes[0, 1].transAxes,
                    color=gray, fontsize=9.9, ha="right")

    ax = axes[1, 0]
    ax.set_title("C　回执噪声从 5% 增至 19%", loc="left", color=ink, fontsize=14, weight="bold", pad=10)
    high = data["long_life"]["high_minus_low_noise_paired"]
    items = [
        ("稳定 A · 全程不翻转", "noise_stable", "stable_A"),
        ("稳定 A · B 发生翻转", "noise_reverse_return", "stable_A"),
        ("B 翻转后第 1 次", "noise_reverse_return", "change_1_1"),
        ("B 翻转后第 9–16 次", "noise_reverse_return", "change_9_16"),
        ("B 恢复后第 9–16 次", "noise_reverse_return", "return_9_16"),
    ]
    ax.axvline(0, color="#8f99a8", lw=.9)
    ax.grid(axis="x", color="#e8edf3", linewidth=.7)
    for phase, name, shift, color, style in (
        ("pre", "教学前", -.12, short_color, "o"), ("post", "教学后", .12, long_color, "s"),
    ):
        stat = [high[case][SELECTED][metric][phase] for _, case, metric in items]
        loc = np.array([p["difference"] for p in stat]) * 100
        ci = np.array([p["ci95"] for p in stat]).T * 100
        ax.errorbar(loc, np.arange(len(stat)) + shift, xerr=np.maximum(0.0, [loc - ci[0], ci[1] - loc]),
                    fmt=style, color=color, markersize=4.8, capsize=3, elinewidth=1.4, label=name, zorder=3)
    ax.set_yticks(np.arange(len(items)), labels=[k[0] for k in items])
    ax.set_ylim(len(items) - .56, -.44)
    ax.set_xlabel("高噪声 − 低噪声（百分点）")
    ax.legend(loc="lower right", frameon=False, fontsize=9.4)

    ax = axes[1, 1]
    ax.set_title("D　统一网络的共享、修正与回访", loc="left", color=ink, fontsize=14, weight="bold", pad=10)
    cases = data["unified_physical"]["modes"]["full"]["scenes"]
    for scene, group, final, marker, color, label in (
        ("standard", "shared", 12, "o", short_color, "同规则 · B 教学 12 对"),
        ("standard", "conflict", 12, "s", long_color, "冲突 · B 教学 12 对"),
        ("long_competing_run", "conflict", 48, "D", orange, "冲突 · B 教学 48 对"),
    ):
        chosen = cases[scene]["arms"]["selected"][group]
        curve(ax, x, [chosen[k] for k in ("B_after_0", f"B_after_{final}", "return_A")], label, color, marker)
    ax.set(ylim=(0, 104), xlim=(-.16, 2.22), ylabel="选择正确率（%）")
    ax.set_xticks(x, labels=["首个 B 探题", "末次 B 探题", "回访 A"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(50, color="#a1a9b7", lw=.8, ls=":", zorder=1)
    ax.grid(axis="y", color="#e8edf3", linewidth=.7)
    ax.legend(loc="lower right", frameon=False, fontsize=9.2)
    chosen = cases["long_competing_run"]["arms"]["selected"]["conflict"]["return_A"]
    ax.annotate(percent(chosen), (2, chosen["mean"] * 100), xytext=(-29, 11), textcoords="offset points",
                fontsize=10, color=orange, weight="bold")
    fig.text(.148, .070, "长生命探题均静默，只改变 B 规则。统一场景首个 B 探题前有一道锚题，接触可尝到锚味，尚无 B 关系配对。",
             fontsize=10.5, color="#435367")
    fig.text(.148, .043, "训练偏向 A/B 共享（77%），统一确认中共享与冲突各半。长 B 协议也改变了噪声和配对跨度。",
             fontsize=10.5, color="#435367")
    fig.savefig(path, dpi=175, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, default=DATA)
    cli.add_argument("--output-folder", type=Path)
    cli.add_argument("--font", type=Path)
    args = cli.parse_args()
    root = args.data_root / "research-2026-11-19"
    out = args.output_folder or root
    out.mkdir(parents=True, exist_ok=True)
    long, first_paths = long_lives(root / "long-life-sealed", args.data_root / "long-life-plasticity-v1")
    physical, second_paths = unified(root / "unified-physical-sealed")
    data = {"estimator": {"model_average": "three fixed models within each world root",
                          "main_rate": "equal weight per world with an eligible request",
                          "additional_request_rate": "each eligible request equally weighted",
                          "paired_differences": "equal-root; same physical root in both arms",
                          "bootstrap_world_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                          "inference_scope": "world sampling for the fixed trained checkpoints"},
            "long_life": long, "unified_physical": physical, "source_sha256": {**first_paths, **second_paths}}
    stat = out / "lifetime-review-statistics.json"
    report = out / "lifetime-review-report.md"
    figure = out / "lifetime-review-figure.png"
    draw_figure(data, figure, args.font)
    stat.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report.write_text(report_text(data), encoding="utf-8")
    print(report)
    print(stat)
    print(figure)


if __name__ == "__main__":
    main()
