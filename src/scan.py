# agi/scan.py — 参数扫描（工具，不是实验）。
#
# 2026-10-24：把两个旧的扫描脚本（scan_base / calib5）统一成「声明一批 Config + 跑」。
# **不重新实现本体**——它们本来就是「用本体跑 N 步、只看 rate」。
#
# 与 experiments.py 的区别：
#   experiments.py  声明「机制实验」，跑完登记到 results.jsonl
#   scan.py         生成「参数网格」，只为看趋势，不登记

import numpy as np

from .config import Config, NetSpec, Plasticity, SignalSpec, TaskSpec, ReadoutSpec, EnvSpec

# 扫描用的本体底座：无 E-E（homeo 形态）/ 无结构可塑性 / 只有背景脉冲
_SCAN_NET = NetSpec(
    init_kind="homeo", theta_mode="fire_adapt",
    tau_adapt=100.0, d_theta=0.20, th_min=1.0, th_max=3.0,
    d_theta_out=0.0, theta_out_adapt=False, t_ref_out=20,
    use_std=True, tau_rec=100.0, u_std=0.30, tau_rate=500.0,
    n_in_fanout=50,
)
_SCAN_TASK = TaskSpec(kind="none", n_cue=4, enc_steps=20, enc_gap=150,
                      cue_after=10 ** 9, bg_rate=0.015, in_spike=3.0)
_SCAN_PLAST = Plasticity(weight=False, topology=False, bias=False,
                         trigger="step", t_s=50, struct_kind="homeo",
                         r_target=0.008, tol=0.20, p_del=0.30, p_new=0.30)


def grid_homeo(steps=4000):
    """scan_base 的网格：无 E-E 时，什么 (W_I, D_IE, D_EI) 让 rate 落在 0.2~0.5%？

    旧脚本用的是 w_i 直接值（-2.5 / -1.5 / -1.0 / -0.6），
    框架用 `w_i = -w_e * w_i_ratio`（w_e = 0.5）→ ratio = |w_i| / 0.5。
    """
    out = []
    combos = []
    for w_i in (-2.5, -1.5, -1.0, -0.6):
        for d_ie in (0.2, 0.1, 0.05):
            combos.append((w_i, d_ie, 0.2))
    for w_i in (-1.5, -1.0):
        combos.append((w_i, 0.1, 0.1))
    for (w_i, d_ie, d_ei) in combos:
        label = "W_I=%.1f D_IE=%.2f D_EI=%.2f" % (w_i, d_ie, d_ei)
        cfg = Config(
            name="scan_%.1f_%.2f_%.2f" % (w_i, d_ie, d_ei), note=label,
            flow="homeo", steps=steps,
            net=NetSpec(**{**{f: getattr(_SCAN_NET, f)
                              for f in _SCAN_NET.__dataclass_fields__},
                           "w_i_ratio": abs(w_i) / 0.5,
                           "d_ie": d_ie, "d_ei": d_ei, "d_ii": 0.0}),
            task=_SCAN_TASK, plast=_SCAN_PLAST,
            signal=SignalSpec(source="none"), n_ep=0,
        )
        out.append((label, cfg))
    return out


def grid_calib(steps=8000):
    """calib5 的网格：加不应期后，(w_E, 抑制比, T_REF) 怎么标定到 2%？

    与 homeo 网格的差别：结构是**纯随机**（不是无 E-E），且有权重噪声 0.15。
    """
    out = []
    for w_e in (0.15, 0.25, 0.35, 0.50):
        for ratio in (1.0, 2.0, 3.0, 5.0):
            for t_ref in (50, 20):
                label = "w_E=%.2f 抑制比=%.1f T_REF=%d" % (w_e, ratio, t_ref)
                cfg = Config(
                    name="calib_%.2f_%.1f_%d" % (w_e, ratio, t_ref), note=label,
                    flow="homeo", steps=steps,
                    net=NetSpec(
                        init_kind="random", theta_mode="static",
                        e0_init=15000, budget=10 ** 9,   # 不淘汰
                        w_e=w_e, w_i_ratio=ratio, init_w_noise=0.15,
                        tau_adapt=100.0, d_theta=0.20, th_min=1.0, th_max=3.0,
                        d_theta_out=0.0, theta_out_adapt=False,
                        t_ref=t_ref, t_ref_out=t_ref,
                        use_std=False, tau_rate=500.0,
                        block_input_edges=False, protect_out_edges=False,
                    ),
                    task=TaskSpec(kind="none", n_cue=4, enc_steps=20, enc_gap=150,
                                  cue_after=10 ** 9, bg_rate=0.002, in_spike=1.0,
                                  bg_all=True),          # calib5 给全体 N 个

                    plast=Plasticity(weight=False, topology=False, bias=False,
                                     trigger="step", t_s=50, struct_kind="homeo"),
                    signal=SignalSpec(source="none"), n_ep=0,
                )
                out.append((label, cfg))
    return out


def summarize(rows):
    """rows: [(label, [result, ...]), ...] → 打表用的行。"""
    out = []
    for label, rs in rows:
        out.append(dict(
            label=label,
            rate_e=float(np.mean([r["rate_tail_mean"] for r in rs])) * 100,
            rate_i=float(np.mean([r["rate_i"] for r in rs])) * 100,
            n_active=float(np.mean([r["n_active"] for r in rs])),
            std=float(np.mean([r["rate_tail_std"] for r in rs])),
            n=len(rs),
        ))
    return out
