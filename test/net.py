# net.py - 网络共享参数与基元（供 test/ 下的新脚本 import）
#
# 为什么要这个文件：2026-10-24 清点发现——`test/` 下 27 个脚本里，“同一组网络参数”
# 被各抄了 10~25 份：584 个常数定义、161 个唯一名字、56 个名字重复 3 次以上
# （V_TH 27 次 / N 25 次 / DECAY_V 25 次 / P_GLOBAL 16 次）。改一个网络参数要改 25 处。
#
# **本文件冻结**：改它会改变所有 import 它的脚本的行为，历史实验将不可复现。
# 要改网络参数 → 新建 net2.py 并在新脚本里用；旧脚本一律不动（它们是实验记录）。
#
# 来源标注（`design.md` §0.3 第 5 条：「不发明新参数——每个常量要么是需求给的，要么有文献出处」）：
#   [标定] §11.4.0 手扫标定（calib 系列）——有实测数字
#   [约定] 归一化约定（阈值=1，其余相对它）
#   [派生] 由其它常数算出——**不应作为可调旋钮**
#   [需求] 由设计/任务定义
#   [拍]   无依据，取一个能跑的值
#
# 台账（逐个常数的来源与“为什么不能删”）：见同目录 `CONSTANTS.md`。

import numpy as np

# ---- 规模 ----
N = 1000                 # [拍] §11.4 给的范围 100~2000，取中间
E_RATIO = 0.8            # [需求] E:I = 4:1（§11.4.0：E/I 平衡是稳态的前提）
N_E = int(round(N * E_RATIO))    # [派生]
N_IN = 100               # [拍] 输入神经元数

# ---- 膜与不应期 ----
TAU_V = 20.0             # [标定] 膜时间常数（calib5）
DECAY_V = float(np.exp(-1.0 / TAU_V))   # [派生]
V_TH = 1.0               # [约定] 阈值归一化
T_REF = 20               # [标定] 不应期（缺了放电率无上界，§11.4.0）

# ---- 权重 ----
W_E = 0.50               # [标定] rate 1.99%
W_I_RATIO = 5.0          # [标定] 尾巴波动 0.94%
W_MAX_E = 1.0            # [约定] 兴奋上界
W_MAX_I = W_E * W_I_RATIO        # [派生] 旧代码写死 2.5
W_NEW_E = 0.15           # [拍] 新建边的固定小权重（不继承本类均值）
W_NEW_I = W_NEW_E * W_I_RATIO    # [派生] 旧代码写死 0.75

# ---- 结构 ----
BUDGET = 15000           # [拍] 硬预算（生长与裁剪是同一选择的两面）
E0_INIT = 2 * BUDGET     # [派生] “过度生产 -> 修剪”，旧代码写死 30000
T_S = 20                 # [拍] 结构更新周期
TRACE_DECAY = 0.95       # [拍] 资格迹衰减（τ≈20 步）
ETA_POT = 0.02           # [拍] 学习率
P_NEW = 0.004            # [拍] 活跃区新建边概率
P_GLOBAL = 4             # [拍] 全局随机边：每次更新加 4 条（**未验证**：与 P_NEW 功能重叠，从未单独测过）
T_GLOBAL = 2000          # [拍] 全局随机边的周期
MAX_ACTIVE = 48          # [拍] 参与生长候选的活跃神经元上限
MIN_AGE = 25             # [拍] 年龄保护（新建后 25 次更新内不参与淘汰）

# ---- 稳态门（m 的第一层）----
RATE_LO = 0.005          # [拍] 低于此率可塑性按比例压下去
RATE_HI = 0.05           # [拍] 高于此率同理


def _sort(pre, post, w, age):
    key = post.astype(np.int64) * N + pre
    o = np.argsort(key, kind='stable')
    key, pre, post, w, age = key[o], pre[o], post[o], w[o], age[o]
    if key.size > 1:
        u = np.empty(key.size, bool); u[0] = True
        np.not_equal(key[1:], key[:-1], out=u[1:])
        if not u.all():
            key, pre, post, w, age = key[u], pre[u], post[u], w[u], age[u]
    return (np.ascontiguousarray(pre), np.ascontiguousarray(post),
            np.ascontiguousarray(w), np.ascontiguousarray(key),
            np.ascontiguousarray(age))


def find(key, q):
    if key.size == 0:
        return np.zeros(q.size, bool), np.zeros(q.size, np.int64)
    i = np.searchsorted(key, q)
    np.clip(i, 0, key.size - 1, out=i)
    return key[i] == q, i


def steady_gate(r):
    """m 的第一层：活动率离开 [RATE_LO, RATE_HI] 时把可塑性按比例压下去。"""
    if r < RATE_LO:
        return max(0.0, r / RATE_LO)
    if r > RATE_HI:
        return max(0.0, RATE_HI / r)
    return 1.0


def cheb(x, g):
    """切比雪夫距离（2 维）。"""
    return float(max(abs(x[0] - g[0]), abs(x[1] - g[1])))
