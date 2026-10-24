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


ALL = [
    BASE,
    PLAST_NONE, PLAST_BIAS, PLAST_W, PLAST_TOPO, PLAST_W_BIAS,
    SIG_NONE, SIG_RANDOM, SIG_EPISODE, SIG_SCALAR, SIG_NORPE, SIG_NOGATE,
    NONISO, TASK_SAMPLED, ARGMAX,
]

BY_NAME = {c.name: c for c in ALL}
