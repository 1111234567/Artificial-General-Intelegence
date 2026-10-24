# agi/experiments.py — 实验定义。
#
# **这里是唯一一个「声明实验」的地方。** 新增实验 = 在这里加一个 Config。
# 不在 test/ 下另写脚本——那正是之前 21 个脚本各实现一遍本体的原因。
#
# 每个实验都必须说清：这份配置在哪些维度上与基准不同（对照组的差集）。

from .config import (Config, Plasticity, SignalSpec, TaskSpec, EnvSpec,
                     ReadoutSpec, NetSpec)

BASE = Config(
    name="base",
    note="全开：动作级 m + 三闸 + 三层可塑性 + 真实环境",
    n_ep=200,
)


# ---------------- 可塑性层（三层各自独立）----------------
# 2026-10-24：这三组就是为了回答「哪一层在起作用」。之前 frozen 名不副实导致结论反了。

PLAST_NONE = BASE.with_(name="plast_none", note="什么都不学（真基线）",
                        plast=Plasticity(weight=False, topology=False, bias=False))
PLAST_BIAS = BASE.with_(name="plast_bias", note="只学 bias（§14.6 作用点二）",
                        plast=Plasticity(weight=False, topology=False, bias=True))
PLAST_W = BASE.with_(name="plast_w", note="只学权重",
                     plast=Plasticity(weight=True, topology=False, bias=False))
PLAST_TOPO = BASE.with_(name="plast_topo", note="只学拓扑",
                        plast=Plasticity(weight=False, topology=True, bias=False))
PLAST_W_BIAS = BASE.with_(name="plast_w_bias", note="bias + 权重（无拓扑）",
                          plast=Plasticity(weight=True, topology=False, bias=True))


# ---------------- 信号源（学习方式本身）----------------

SIG_NONE = BASE.with_(name="sig_none", note="m 恒 0（比 plast_none 更彻底：连偏置都不动）",
                      signal=SignalSpec(source="none"),
                      plast=Plasticity(weight=False, topology=False, bias=False))
SIG_RANDOM = BASE.with_(name="sig_random", note="m 随机（检验「有 m 就行吗」）",
                        signal=SignalSpec(source="random"))
SIG_EPISODE = BASE.with_(name="sig_episode", note="m 只在 episode 末（延迟 = 整个任务）",
                         signal=SignalSpec(source="episode"))
SIG_SCALAR = BASE.with_(name="sig_scalar", note="m 是标量（不逐神经元分配）",
                        signal=SignalSpec(source="action", shape="scalar"))
SIG_NORPE = BASE.with_(name="sig_norpe", note="m 不用 RPE（直接 r）",
                       signal=SignalSpec(source="action", glob="raw"))
SIG_NOGATE = BASE.with_(name="sig_nogate", note="不用三闸（m = 正确性）",
                        signal=SignalSpec(source="action", gate=False))


# ---------------- 隔离 ----------------
NONISO = BASE.with_(name="no_isolate", note="写入不做局部化（§9.2 隔离 = 0）",
                    plast=Plasticity(weight=True, topology=True, bias=True, isolate=False))


# ---------------- 任务 ----------------
# 2026-10-24：任务不可完成（6 个动作 vs 差距 40）曾使全部读数无意义。
# 前置检查现在会自动拦这种配置——下面两个都因此调整过：
#   task_sampled：obs 范围 80 -> 40，使最大差距=40 能被 48 个动作盖住
#   readout_argmax：argmax 只能管「输出数 = 动作数」的情形，所以 n_out=2
TASK_SAMPLED = BASE.with_(
    name="task_sampled", note="初态/目标随机采样（样本几乎不重复）",
    task=TaskSpec(kind="sampled", obs_lo=30, obs_hi=70))


# ---------------- 读数 ----------------
ARGMAX = BASE.with_(
    name="readout_argmax", note="argmax 读数（永远选一个，无沉默）",
    readout=ReadoutSpec(kind="argmax", n_out=2, out_idx=(500, 501),
                        allow_noop=False))


# ---------------- 对齐（不是实验）----------------
# 2026-10-24：只为回答「框架为什么差 5 倍」。每个开关对应旧 grow_agent(discrete) 的一处行为：
#   actions=(1,-1,10,-10)   旧 ACTION=[1,-1,10,-10,0]（第 5 个从不被索引到）
#   tie_break=argmax_v      旧：cand[argmax(v[OUT_IDX][cand])]
#   topo_gate=False         旧：每个 ep>0 都跑拓扑（不看 m_ep）
#   bridge=0.15             旧：correct = max(succ, 0.15×moved)
#   silence_update_w=True   旧：沉默时 upd_weights(mv, 0.0)
#   steady_gate_ep=True     旧：m = m_gate(...) × steady_gate(ema_rate)
#   u_buf_rows=65536        旧：u_buf = rng.random((65536, N_IN))
# 若 aligned 能回到 37%，说明差集完整；再逐项关掉量贡献。
# （上述开关已全部内联到下面的 ga_* 系列；ALIGNED 本身已由 ga_discrete 取代。）


# ---------------- 从旧 grow_agent 移植（2026-10-24）----------------
# 每个 Config 对应 grow_agent.py 的一个 mode，目标是**行为逐项一致**（用 scripts/align_layers.py seq 验证）。
# 旧代码的 mode 解析很容易让人误解（`discrete_w` 含 bias、`no_gate` 只学 bias），
# 所以下面每个 name 后面标的是**代码解析出的真实三层开关**（b/w/topo），不是 mode 名字。
_GA = dict(
    env=EnvSpec(actions=(1, -1, 10, -10)),
    readout=ReadoutSpec(kind="first_spike", n_out=4,
                        out_idx=(500, 501, 502, 503),
                        allow_noop=True, tie_break="argmax_v"),
    signal=SignalSpec(source="action", shape="per_neuron", glob="rpe", gate=True,
                      bridge=0.15, silence_update_w=True, steady_gate_ep=True),
    u_buf_rows=65536,
    reset_readout_per_ep=False,
)
_DISCRETE_TASK = TaskSpec(kind="discrete")
_SAMPLED_TASK = TaskSpec(kind="sampled", obs_lo=10, obs_hi=90, k_ep_task=5)
_RECALL_TASK = TaskSpec(kind="recall", obs_lo=10, obs_hi=90, k_ep_task=5)

GA_DISCRETE = Config(name="ga_discrete", note="grow_agent mode=discrete（b/w/topo 全开）",
                     plast=Plasticity(topo_gate=False, topo_skip_first=True), **_GA)
GA_NONE = Config(name="ga_none", note="grow_agent discrete_none（0/0/0，discrete 任务）",
                 plast=Plasticity(weight=False, topology=False, bias=False,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_BIAS = Config(name="ga_bias", note="grow_agent discrete_bias（1/0/0）",
                 plast=Plasticity(weight=False, topology=False, bias=True,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_W = Config(name="ga_w", note="grow_agent discrete_w（1/1/0：名字叫 w，实际含 bias）",
              plast=Plasticity(weight=True, topology=False, bias=True,
                               topo_gate=False, topo_skip_first=True), **_GA)
GA_TOPO = Config(name="ga_topo", note="grow_agent discrete_topo（1/0/1）",
                 plast=Plasticity(weight=False, topology=True, bias=True,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_FROZEN = Config(name="ga_frozen", note="grow_agent frozen（0/0/0，**sampled 任务**）",
                   task=_SAMPLED_TASK,
                   plast=Plasticity(weight=False, topology=False, bias=False,
                                    topo_gate=False, topo_skip_first=True), **_GA)
GA_BASE = Config(name="ga_base", note="grow_agent base（1/0/0，sampled 任务）",
                 task=_SAMPLED_TASK,
                 plast=Plasticity(weight=False, topology=False, bias=True,
                                  topo_gate=False, topo_skip_first=True), **_GA)
GA_RECALL = Config(name="ga_recall", note="grow_agent recall（1/0/0，目标不可观测）",
                   task=_RECALL_TASK,
                   plast=Plasticity(weight=False, topology=False, bias=True,
                                    topo_gate=False, topo_skip_first=True), **_GA)
_GA_NO_READOUT = {k: v for k, v in _GA.items() if k != "readout"}
GA_RANDOM = Config(name="ga_random",
                   note="grow_agent mode=random——不建网、不学，只随机选动作（下界基线）",
                   readout=ReadoutSpec(kind="random", n_out=4,
                                       out_idx=(500, 501, 502, 503),
                                       allow_noop=False),
                   **_GA_NO_READOUT)


# ---------------- 从旧 grow_goal 移植（2026-10-24）----------------
# 形态差异全部靠 Config 字段表达：flow=goal / env.kind=math_2d / readout.kind=sign_2d /
# plast.trigger=step / signal.source=value。本体（core.py）与原子操作与 agent 流程共用。
_GOAL = dict(
    flow="goal",
    net=NetSpec(side_inh=0.0, t_ref_out=20),   # grow_goal 无侧抑制、无输出专用不应期
    env=EnvSpec(kind="math_2d"),
    task=TaskSpec(kind="sampled", t_act=20, t_ep=120, t_gap=0),
    readout=ReadoutSpec(kind="sign_2d", n_out=2, out_idx=(500, 501),
                        a_tau=20.0, base_tau=100.0),
    plast=Plasticity(weight=True, topology=True, bias=False,
                     trigger="step", t_s=20, topo_thresh=0.5),
    signal=SignalSpec(source="value"),
)

GA_GOAL_A = Config(name="ga_goal_A", note="grow_goal A（主体：成败当 m，每 20 步更新）",
                   n_ep=200, **_GOAL)
GA_GOAL_FROZEN = Config(name="ga_goal_frozen",
                        note="grow_goal A_frozen（不强化、不生长；淘汰仍执行）",
                        n_ep=200,
                        **dict(_GOAL, plast=Plasticity(
                            weight=False, topology=False, bias=False,
                            trigger="step", t_s=20, topo_thresh=0.5)))
GA_GOAL_FIXED = Config(name="ga_goal_fixed",
                       note="grow_goal A_fixed（目标固定为 (0.5,-0.5)，排除「恒定动作碰巧命中」）",
                       n_ep=300,
                       **dict(_GOAL, task=TaskSpec(kind="sampled", t_act=20, t_ep=120,
                                                   t_gap=0, goal_mode="fixed")))
GA_GOAL_PROBE = Config(name="ga_goal_probe",
                       note="grow_goal A_probe（诊断：从状态线性读出 (g-x)）",
                       n_ep=150,
                       **dict(_GOAL, readout=ReadoutSpec(
                           kind="sign_2d", n_out=2, out_idx=(500, 501),
                           a_tau=20.0, base_tau=100.0, probe=True)))
GA_GOAL_T_SEQ = Config(name="ga_goal_t_seq",
                       note="grow_goal T_seq（先 task A 右上 再 task B 左下）",
                       n_ep=400,
                       **dict(_GOAL, task=TaskSpec(kind="sampled", t_act=20, t_ep=120,
                                                   t_gap=0, goal_mode="two_phase",
                                                   phase="seq")))
GA_GOAL_T_FRESH = Config(name="ga_goal_t_fresh",
                         note="grow_goal T_fresh（直接跑 task B）",
                         n_ep=400,
                         **dict(_GOAL, task=TaskSpec(kind="sampled", t_act=20, t_ep=120,
                                                     t_gap=0, goal_mode="two_phase",
                                                     phase="fresh")))


# ---------------- 从旧 grow_world 移植（2026-10-24）----------------
# 形态：单条连续时间轴（无 episode）+ 2D 振荡器世界 + 连续 tanh 动作。
# m = 稳态门 × (0.2 + 0.8×动作方差代理)；el 不减 θ；非对称权重更新。
# 生长**不过滤输入区**、有全局随机边、淘汰**不保护**接口边——这三条都与 grow_agent 相反。
_WORLD = dict(
    flow="world",
    steps=50000,
    net=NetSpec(side_inh=0.0, t_ref_out=20, w_i_ratio=5.0,
                block_input_edges=False, protect_out_edges=False,
                p_global=4, t_global=2000),
    env=EnvSpec(kind="oscillator", period_d=20.0, period_d_slow=500.0,
                tau_w_f=50.0, tau_w_s=500.0, amp_d=0.10, amp_d_slow=0.10,
                k_phase=0.05, w_noise=0.01),
    task=TaskSpec(kind="none", t_ep=1, t_act=1, t_gap=0),
    readout=ReadoutSpec(kind="tanh_2d", n_out=2, out_idx=(500, 501),
                        a_tau=20.0, r_base=0.02, r_scale=0.03),
    plast=Plasticity(weight=True, topology=True, bias=False,
                     trigger="step", t_s=20, topo_thresh=0.5,
                     tau_trace=20.0, trace_decay=0.95),
    signal=SignalSpec(source="agency", gate=True, base=0.2, agency_scale=0.2),
)

GA_WORLD_CLOSED = Config(name="ga_world_closed",
                         note="grow_world closed（闭环：动作推世界，世界回输入）", **_WORLD)
GA_WORLD_OPEN = Config(name="ga_world_open",
                       note="grow_world open（断开：动作不推世界）",
                       **dict(_WORLD, signal=SignalSpec(source="agency", gate=False,
                                                        base=0.2, agency_scale=0.2)))
GA_WORLD_RANDACT = Config(name="ga_world_randact",
                          note="grow_world randact（动作随机化）",
                          **dict(_WORLD, readout=ReadoutSpec(
                              kind="randact", n_out=2, out_idx=(500, 501),
                              a_tau=20.0, r_base=0.02, r_scale=0.03)))
GA_WORLD_NOISE = Config(name="ga_world_noise_in",
                        note="grow_world noise_in（输入白噪声，无世界结构）",
                        **dict(_WORLD, signal=SignalSpec(source="noise", gate=True,
                                                          base=0.2, agency_scale=0.2)))
GA_WORLD_FROZEN = Config(name="ga_world_frozen",
                         note="grow_world frozen（结构冻结：可塑性整块关闭）",
                         **dict(_WORLD, plast=Plasticity(
                             weight=False, topology=False, bias=False,
                             trigger="step", t_s=20, topo_thresh=0.5,
                             tau_trace=20.0, trace_decay=0.95)))


ALL = [
    BASE,
    PLAST_NONE, PLAST_BIAS, PLAST_W, PLAST_TOPO, PLAST_W_BIAS,
    SIG_NONE, SIG_RANDOM, SIG_EPISODE, SIG_SCALAR, SIG_NORPE, SIG_NOGATE,
    NONISO, TASK_SAMPLED, ARGMAX,
    GA_DISCRETE, GA_NONE, GA_BIAS, GA_W, GA_TOPO, GA_FROZEN, GA_BASE, GA_RECALL,
    GA_RANDOM,
    GA_GOAL_A, GA_GOAL_FROZEN, GA_GOAL_FIXED, GA_GOAL_PROBE,
    GA_GOAL_T_SEQ, GA_GOAL_T_FRESH,
    GA_WORLD_CLOSED, GA_WORLD_OPEN, GA_WORLD_RANDACT, GA_WORLD_NOISE, GA_WORLD_FROZEN,
]

BY_NAME = {c.name: c for c in ALL}
