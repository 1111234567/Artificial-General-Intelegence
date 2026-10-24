# agi/signal.py — `m` 的来源与形态。**这是「学习方式」本身**。
#
# 2026-10-24 的教训：「你把它学习的方式固定了」——对。
# 我实现的只有一种：三因子 + 标量/逐输出 m + 低频更新。
# 而文献里至少四类（标量 m / per-neuron m（TESS）/ per-connection gain（ChessFly）/ 不学靠结构（果蝇））。
# 所以这里把 source / shape / glob / gate 全部显式化，并把每个实现标清楚依据。

import numpy as np


def steady_gate(rate, lo=0.005, hi=0.05):
    """活动率离开区间时把可塑性压下去。第一层（不是三闸之一）。"""
    if rate < lo:
        return max(0.0, rate / lo)
    if rate > hi:
        return max(0.0, hi / rate)
    return 1.0


def three_gates(admit, correct, consistent):
    """§7.1 三闸：准入 × 裁决（= 正确 × 一致）。三个 0~1 标量，可手推。"""
    return float(admit) * float(correct) * float(consistent)


class Signal(object):
    """`m` 的产生器。分为「动作级」与「任务级」两个出口（design.md §5.4）。"""

    def __init__(self, spec, n_out):
        self.spec = spec
        self.n_out = n_out
        self.base = 0.0          # RPE 基线
        self._last = (None, None)

    # ---------- 动作级（延迟 0）----------
    def on_action(self, a_idx, moved, want):
        """返回 (m_vec, m_glob)。

        moved = 实际位移（带符号）；want = sign(goal - state)。
        """
        # noop（a_idx < 0）：什么都没做 -> 不给任何 m。
        # 2026-10-24 实测：不修这一条时，noop 分支会返回 m_glob = -base（非零），
        # 而 runner 的条件是 `mv is not None or mg is not None` -> 照样更新，
        # 于是「什么都不做」的那个窗口也在削弱全部边（实浏 noop 时 100% 非零，均值 -0.29）。
        # 旧 grow_agent 不存在这条分支（a_idx 永远 0..3），所以这是框架独有的缺陷。
        if a_idx < 0:
            self._last = (None, None)
            return None, None

        mv = np.zeros(self.n_out, np.float32)
        r = 0.0
        if a_idx >= 0 and want != 0 and moved != 0:
            if np.sign(moved) == want:
                mv[a_idx] = 1.0
                r = 1.0
            else:
                mv[a_idx] = -1.0
                r = -1.0

        src = self.spec.source
        if src in ("none", "episode"):
            self._last = (None, None)
            return None, None
        if src == "random":
            out = (np.random.rand(self.n_out).astype(np.float32), float(np.random.randn()))
            self._last = out
            return out

        # src == "action"
        if self.spec.glob == "rpe":
            mg = r - self.base
            self.base = 0.9 * self.base + 0.1 * r
        else:
            mg = r
        if self.spec.shape == "scalar":
            mv = None
        self._last = (mv, mg)
        return mv, mg

    def on_silence(self, want):
        """沉默时的弱信号（grow_talk 的镜像：提高「正确目标」的兴奋性）。

        2026-10-24 实测：没有这一条时，agent 学会「少说话」——
        因为不动作不会错，而「不说」是个免费的安全选项（实测 none 组只有 5.9%，
        比 random 的 20.6% 还差）。
        """
        k = self.spec.silence_hint
        if k <= 0 or want == 0:
            return None
        mv = np.zeros(self.n_out, np.float32)
        if want > 0:
            mv[0] = k
            if self.n_out > 2:
                mv[2] = k
        else:
            mv[1] = k
            if self.n_out > 3:
                mv[3] = k
        return mv

    # ---------- 任务级（延迟 = episode）----------
    def on_episode(self, succ, progress, consistent, io_ok):
        """返回 (m_ep, m_vec_action, m_glob_action)。

        `m_ep` 给**拓扑（结构）**用；后两个给 `source=episode` 时补发权重用。
        """
        src = self.spec.source
        if src == "none":
            return 0.0, None, None
        if src == "random":
            return float(np.random.rand()), None, None
        if src == "episode":
            if not succ:
                return 0.0, None, None
            mv, mg = self._last
            return 1.0, mv, mg

        admit = 1.0 if io_ok else 0.0
        correct = 1.0 if succ else 0.0
        if self.spec.bridge > 0 and not succ:
            correct = max(correct, self.spec.bridge * min(progress, 1.0))
        if self.spec.gate:
            return three_gates(admit, correct, consistent), None, None
        return correct, None, None
