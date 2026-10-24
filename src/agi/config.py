# agi/config.py — 实验的完整声明
#
# 设计原则（2026-10-24，起因：test/ 下 21 个脚本各实现一遍本体，每个丢点漏点）：
#
#   1. **一个实验 = 一个 Config 对象**。没有任何隐式默认——每一样都必须写出来。
#      看到一份 Config，就知道这个实验开了什么、关了什么。
#   2. **开关是枚举/元组，不是字符串魔法**。`Plasticity` 每个目标独立开关。
#   3. **可序列化**。Config -> dict 是稳定的，用来当结果登记的主键（去重 / 查重跑）。
#   4. **不可变**。写错了就重写一个，不要在半路上改参数。
#
# 为什么不用命令行参数：命令行能表达「开什么」，但表达不了「关了什么」。
# 而今天的教训正是「我不知道哪个版本到底有什么」——那是关项丢掉的典型后果。

import hashlib
import json
from dataclasses import dataclass, field, asdict, replace
from typing import Tuple, Optional


# ---------------------------------------------------------------- 本体

@dataclass(frozen=True)
class NetSpec:
    """网络本体（LIF + 稀疏连接）。默认值全部来自恒定标定（calib5 / §11.4.0）。"""
    N: int = 1000
    e_ratio: float = 0.8
    n_in: int = 100
    tau_v: float = 20.0
    v_th: float = 1.0
    t_ref: int = 20
    t_ref_out: int = 4          # 输出神经元单独的不应期（§16.10 缺陷 6）
    w_e: float = 0.50
    w_i_ratio: float = 5.0
    w_max_e: float = 1.0
    w_new_e: float = 0.15
    budget: int = 15000
    e0_init: int = 30000        # 过度生产 -> 修剪
    max_active: int = 48
    min_age: int = 25
    p_new: float = 0.004
    side_inh: float = 1.5


# ---------------------------------------------------------------- 环境

@dataclass(frozen=True)
class EnvSpec:
    """环境。`kind` 必须显式：本机真实文件是唯一实装项。"""
    kind: str = "local_files"       # local_files = 真实读写 data/env/...
    root: str = "data/env"
    v_max: int = 100
    actions: Tuple[int, ...] = (1, -1)   # 2026-10-24：±10 无用（linear 基线 W 恒 0）


# ---------------------------------------------------------------- 任务

@dataclass(frozen=True)
class TaskSpec:
    """任务。`kind` 决定状态怎么来。

    discrete: 固定的（初态, 目标）对循环 —— 每个状态重复 n_ep/len(cases) 次
    sampled : 初态与目标随机采样 —— 样本几乎不重复
    """
    kind: str = "discrete"
    cases: Tuple[Tuple[int, int], ...] = ((20, 60), (40, 70), (60, 20), (70, 30))
    obs_lo: int = 10
    obs_hi: int = 90
    t_ep: int = 240
    t_act: int = 5              # 动作窗口（必须满足 t_ep/t_act * step >= 最大差距）
    t_gap: int = 200            # episode 间隙（逼退隐状态）

    def check_feasible(self, steps_per_action: int) -> Tuple[bool, str]:
        """**前置检查**：最多能走多远 vs 目标差距。

        2026-10-24 的教训：t_ep=120 / t_act=20 = 6 个动作，而差距 40 -> 任务不可完成，
        而我用 5 轮「机制修补」去解释一个算术问题。任何实验先过这一关。
        """
        n_act = self.t_ep // self.t_act
        reach = n_act * steps_per_action
        if self.kind == "discrete":
            gap = max(abs(g - s) for s, g in self.cases)
        else:
            gap = self.obs_hi - self.obs_lo
        if reach < gap * 1.2:
            return False, "最多移动 %d < 差距 %d × 1.2（需要 %d）" % (reach, gap, int(gap * 1.2))
        return True, "最多移动 %d >= 差距 %d × 1.2" % (reach, gap)


# ---------------------------------------------------------------- 可塑性

@dataclass(frozen=True)
class Plasticity:
    """**三层独立开关**。每层单独声明，不能靠名字猜。

    2026-10-24 的教训：`frozen` 只关了权重+拓扑，bias 还在学
    -> 它实际是「只学 bias」，而真正的「什么都不学」根本不存在
    -> 「frozen 更好」被读成「不学更好」，结论完全相反。
    """
    weight: bool = True         # 连接权重（Δw = η·m·trace[pre]·(trace[post]−θ)）
    topology: bool = True       # 新建 / 淘汰边
    bias: bool = True           # 输出神经元的节点偏置（§14.6 作用点二）
    eta_w: float = 0.02
    eta_bias: float = 0.05
    bias_min: float = -0.3
    bias_max: float = 2.0
    tau_trace: float = 100.0    # 迹必须跨过「动作 → 反馈」的延迟
    tau_theta: float = 200.0    # BCM 滑动阈值
    isolate: bool = True        # 只写入本 episode 活跃过的边（§9.2 隔离）
    isolate_thresh: float = 0.3


# ---------------------------------------------------------------- 信号 m

@dataclass(frozen=True)
class SignalSpec:
    """`m` 的来源与形态——这是「学习方式」本身，必须显式。

    source:
        none       m ≡ 0（真基线）
        random     m 随机（检验「有 m 就行吗」）
        action     每动作：朝目标 +1 / 背目标 −1（延迟 0，§5.4 环境事实）
        episode    每 episode：成功 +1 / 失败 0（延迟 = episode 长）
    shape:
        scalar        m 是 1 个标量，广播到所有边
        per_neuron    m 是 N 维（接口边用逐输出值，其余用全局）
    glob:
        raw         直接 r
        rpe         r − 基线（可正可负）
    gate:
        三闸（§7.1）：admit × correct × consistent；关掉就是 1
    """
    source: str = "action"
    shape: str = "per_neuron"
    glob: str = "rpe"
    gate: bool = True
    bridge: float = 0.0          # 桥接强度（0 = 不用）
    silence_hint: float = 0.2    # 沉默时给「朝目标侧」的偏置增量（0 = 不提示）


# ---------------------------------------------------------------- 读数

@dataclass(frozen=True)
class ReadoutSpec:
    """输出怎么变成动作。"""
    kind: str = "first_spike"   # first_spike（谁先放选谁；都不放=不说） / argmax
    n_out: int = 4
    out_idx: Tuple[int, ...] = (500, 501, 502, 503)
    allow_noop: bool = True      # 「不说」是不是一个合法动作
    ensemble: str = "action"     # argmax 是 action；first_spike 不必


# ---------------------------------------------------------------- 完整配置

@dataclass(frozen=True)
class Config:
    name: str = "unnamed"
    note: str = ""
    net: NetSpec = field(default_factory=NetSpec)
    env: EnvSpec = field(default_factory=EnvSpec)
    task: TaskSpec = field(default_factory=TaskSpec)
    plast: Plasticity = field(default_factory=Plasticity)
    signal: SignalSpec = field(default_factory=SignalSpec)
    readout: ReadoutSpec = field(default_factory=ReadoutSpec)
    n_ep: int = 200
    seed: int = 0

    # ---- 序列化 / 指纹 ----
    def to_dict(self) -> dict:
        d = asdict(self)
        # tuple -> list，保证 json 可写
        def conv(x):
            if isinstance(x, tuple):
                return [conv(i) for i in x]
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            return x
        return conv(d)

    def fingerprint(self) -> str:
        """配置指纹：**只含影响行为的字段**（不含 name / note / seed / n_ep）。

        用途：查「这个配置跑过吗」。名字可以变，指纹不变。
        """
        d = self.to_dict()
        for k in ("name", "note", "seed", "n_ep"):
            d.pop(k, None)
        s = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

    def validate(self) -> list:
        """硬检查。任何一条不过就不该跑。"""
        errs = []
        step = max(abs(a) for a in self.env.actions)
        ok, msg = self.task.check_feasible(step)
        if not ok:
            errs.append("任务不可完成：" + msg)
        if self.signal.source not in ("none", "random", "action", "episode"):
            errs.append("未知 signal.source: %s" % self.signal.source)
        if self.signal.shape not in ("scalar", "per_neuron"):
            errs.append("未知 signal.shape: %s" % self.signal.shape)
        if self.readout.n_out != len(self.readout.out_idx):
            errs.append("n_out 与 out_idx 长度不一致")
        if len(self.env.actions) < self.readout.n_out and not self.readout.allow_noop:
            errs.append("动作数少于输出数且不允许 noop -> 有些输出永远无法执行")
        return errs

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)
