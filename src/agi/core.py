# agi/core.py — 本体（LIF 稀疏网络）。**唯一实现**。
#
# 2026-10-24 的教训：test/ 下 21 个脚本各实现了一遍本体（584 个常数定义 / 161 个唯一名字 /
# 56 个名字重复 3 次以上）。改一个参数要改 10~25 处，抄的过程中已经漂了。
# 本文件是唯一一份；其他一切（新旧实验）都应当向它靠。

import numpy as np


def sort_edges(pre, post, w, age, N):
    """按 (post, pre) 排序去重，返回 (pre, post, w, key, age)。key = post*N + pre。"""
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


def find_edges(key, q):
    if key.size == 0:
        return np.zeros(q.size, bool), np.zeros(q.size, np.int64)
    i = np.searchsorted(key, q)
    np.clip(i, 0, key.size - 1, out=i)
    return key[i] == q, i


class Network:
    """LIF 稀疏网络。状态全在实例里，没有模块级可变状态。"""

    def __init__(self, spec, rng, out_idx):
        self._spec = spec
        self.rng = rng
        self.N = spec.N
        self.N_E = int(round(spec.N * spec.e_ratio))
        self.N_IN = spec.n_in
        self.out_idx = np.asarray(out_idx, np.int64)

        self.tau_trace = 100.0    # 由 Plasticity 覆盖
        self.trace_decay = 0.0    # >0 时直接用（grow_world 的 0.95）
        self.tau_theta = 200.0

        # 结构：过度生产 -> 硬预算修剪
        pre = rng.integers(0, self.N, spec.e0_init).astype(np.int32)
        post = rng.integers(0, self.N, spec.e0_init).astype(np.int32)
        ok = (pre != post)
        if spec.block_input_edges:
            ok = ok & (post >= self.N_IN)      # 输入区不接受反向边（grow_agent）
        pre, post = pre[ok], post[ok]
        w = np.where(pre >= self.N_E, -spec.w_e * spec.w_i_ratio, spec.w_e).astype(np.float32)
        age = np.full(pre.size, 100, np.int16)
        self.pre, self.post, self.w, self.key, self.age = sort_edges(
            pre, post, w, age, self.N)

        # 状态
        self.v = np.zeros(self.N, np.float32)
        self.state = np.zeros(self.N, np.float32)
        self.ref = np.zeros(self.N, np.int32)
        self.trace = np.zeros(self.N, np.float32)
        self.theta = np.zeros(self.N, np.float32)
        self.bias = np.zeros(self.N, np.float32)
        self.ema_rate = 0.02

    # ---- 派生量 ----
    @property
    def decay_v(self):
        return float(np.exp(-1.0 / self._spec.tau_v))

    @property
    def w_max_i(self):
        return self._spec.w_e * self._spec.w_i_ratio

    @property
    def w_new_i(self):
        return self._spec.w_new_e * self._spec.w_i_ratio

    # ---- 单步 ----
    def step(self, xin, apply_side_inh=True):
        sp = self._spec
        I = np.bincount(self.post, weights=self.w * self.state[self.pre],
                        minlength=self.N).astype(np.float32)
        if apply_side_inh:
            ro = self.state[self.out_idx]
            if ro.any():
                I[self.out_idx] -= sp.side_inh * ro
        self.v = self.v * self.decay_v + I + xin + self.bias
        fire = (self.v > sp.v_th) & (self.ref <= 0)
        self.state = fire.astype(np.float32)
        self.v = np.where(self.state > 0, 0.0, self.v)
        self.ref -= 1
        np.maximum(self.ref, 0, out=self.ref)
        self.ref[self.state > 0] = sp.t_ref
        of = self.state[self.out_idx] > 0
        if of.any():
            self.ref[self.out_idx[of]] = sp.t_ref_out

        nsp = int(self.state.sum())
        self.ema_rate = 0.99 * self.ema_rate + 0.01 * (nsp / self.N)
        self.trace *= (self.trace_decay if self.trace_decay > 0
                       else float(np.exp(-1.0 / self.tau_trace)))
        self.trace += self.state
        self.theta *= (1.0 - 1.0 / self.tau_theta)
        self.theta += self.trace / self.tau_theta
        return nsp

    # ---- 可塑性：权重 ----
    def apply_weight(self, m_vec, m_glob, eta, isolate_mask=None, norm="after_m"):
        """Δw = η · m · trace[pre] · (trace[post] − θ)。

        接口边（突触后是输出）用 m_vec 逐神经元；其余用 m_glob。
        m_vec / m_glob 都可为 None（= 0）。

        norm：归一化的位置（两个旧实现不一样，必须显式）：
          after_m  —— grow_agent：e = el·m，再按 max|e| 归一（m 被约掉）
          before_m —— grow_goal：el 先归一，再乘 m（m 的大小保留）
        """
        if not self.key.size:
            return
        el = self.trace[self.pre] * (self.trace[self.post] - self.theta[self.post])
        if isolate_mask is not None:
            el = np.where(isolate_mask, el, 0.0)

        if norm == "before_m":
            if m_glob is None or abs(m_glob) < 1e-12:
                return
            ex = float(np.abs(el).max())
            if ex <= 1e-9:
                return
            dw = (eta * float(m_glob) * (el / ex)).astype(np.float32)
            cap = np.where(self.w > 0, self._spec.w_max_e, self.w_max_i)
            self.w = self.w + dw * (cap - np.abs(self.w))
            np.clip(self.w, -self.w_max_i, self._spec.w_max_e, out=self.w)
            return

        if m_vec is not None:
            mfull = np.zeros(self.N, np.float32)
            mfull[self.out_idx] = m_vec
            ist_o = np.isin(self.post, self.out_idx)
            sel = np.where(ist_o, mfull[self.post.astype(np.int64)],
                           np.float32(0.0 if m_glob is None else m_glob)).astype(np.float32)
        else:
            if m_glob is None or abs(m_glob) < 1e-12:
                return
            sel = np.full(self.key.size, np.float32(m_glob), np.float32)
        e = el * sel
        emax = float(np.abs(e).max())
        if emax <= 1e-9:
            return
        e /= emax
        dw = (eta * e).astype(np.float32)
        cap = np.where(self.w > 0, self._spec.w_max_e, self.w_max_i)
        self.w = self.w + dw * (cap - np.abs(self.w))
        np.clip(self.w, -self.w_max_i, self._spec.w_max_e, out=self.w)

    # ---- 可塑性：偏置 ----
    def apply_bias(self, m_vec, eta, lo, hi):
        if m_vec is None:
            return
        b = self.bias[self.out_idx] + eta * m_vec
        self.bias[self.out_idx] = np.clip(b, lo, hi)

    # ---- 可塑性：拓扑 ----
    def grow_edges(self, thresh, p_new, max_active):
        """只在活跃区新建边（不动现有边、不淘汰）。返回出生数。"""
        sp = self._spec
        self.last_born = 0
        act = np.nonzero(self.trace > thresh)[0]
        if act.size > max_active:
            act = self.rng.choice(act, max_active, replace=False)
        if act.size < 2:
            return 0
        na = act.size
        q_pre = np.repeat(act, na)
        q_post = np.tile(act, na)
        okk = (q_pre != q_post)
        if sp.block_input_edges:
            okk = okk & (q_post >= self.N_IN)
        q_pre, q_post = q_pre[okk], q_post[okk]
        qk = q_post.astype(np.int64) * self.N + q_pre
        found, _ = find_edges(self.key, qk)
        npw, nqw = q_pre[~found], q_post[~found]
        if not npw.size:
            return 0
        pick = self.rng.random(npw.size) < p_new
        if not pick.any():
            return 0
        npw, nqw = npw[pick], nqw[pick]
        nw = np.where(npw < self.N_E, sp.w_new_e, -self.w_new_i).astype(np.float32)
        self._grow(npw, nqw, nw)
        self.last_born = int(npw.size)
        return self.last_born

    def prune(self, budget, min_age):
        """年龄 +1，超预算时按「在本类上界中的相对强度」淘汰。返回删除数。"""
        sp = self._spec
        self.age += 1
        if self.key.size <= budget:
            return 0
        dd = self.key.size - budget
        scale = np.abs(self.w) / np.where(self.pre >= self.N_E, self.w_max_i, sp.w_max_e)
        if sp.protect_out_edges:
            ist_out = np.isin(self.post, self.out_idx)
            cand = (self.age >= min_age) & (~ist_out)
            if not cand.any():
                cand = self.age >= min_age
        else:
            cand = self.age >= min_age
        if not cand.any():
            return 0
        idx_c = np.nonzero(cand)[0]
        dd = min(dd, idx_c.size)
        if dd <= 0:
            return 0
        sub = idx_c[np.argpartition(scale[idx_c], dd - 1)[:dd]]
        keep = np.ones(self.key.size, bool)
        keep[sub] = False
        self.pre, self.post, self.w, self.key, self.age = (
            self.pre[keep], self.post[keep], self.w[keep],
            self.key[keep], self.age[keep])
        return int(dd)

    def apply_topology(self, thresh, budget, p_new, max_active, min_age, grow=True):
        """生长 + 硬预算淘汰（grow_agent / grow_goal 的两合一入口）。"""
        self.last_born = self.grow_edges(thresh, p_new, max_active) if grow else 0
        return self.prune(budget, min_age)

    def _grow(self, npw, nqw, nw):
        self.pre, self.post, self.w, self.key, self.age = sort_edges(
            np.concatenate([self.pre, npw.astype(np.int32)]),
            np.concatenate([self.post, nqw.astype(np.int32)]),
            np.concatenate([self.w, nw]),
            np.concatenate([self.age, np.zeros(npw.size, np.int16)]),
            self.N)

    # ---- 可塑性：非对称权重更新（grow_world 形态）----
    def apply_weight_asym(self, m, eta, use_theta=False):
        """grow_world 的形态：el = trace[pre]·trace[post]（可选减 θ），**非对称**更新。

        与 apply_weight 的三处不同（旧脚本就是这么写的，不是我的选择）：
          ① 无 BCM 时 el 恒 ≥ 0（只增强不削弱）；
          ② dw 先归一化再乘 m（m 的大小保留）；
          ③ 兴奋边向 +w_max_e 靠近，抑制边向 −w_max_i 靠近。
        """
        if not self.key.size or abs(m) < 1e-12:
            return 0.0
        if use_theta:
            el = self.trace[self.pre] * (self.trace[self.post] - self.theta[self.post])
        else:
            el = self.trace[self.pre] * self.trace[self.post]
        emax = float(np.abs(el).max()) if use_theta else float(el.max())
        if emax <= 1e-9:
            return 0.0
        el = el / emax
        dw = (eta * float(m) * el).astype(np.float32)
        pos = self.w > 0
        self.w[pos] = np.minimum(self.w[pos] + dw[pos] * (self._spec.w_max_e - self.w[pos]),
                                 self._spec.w_max_e)
        neg = ~pos
        self.w[neg] = np.maximum(self.w[neg] - dw[neg] * (self.w_max_i + self.w[neg]),
                                 -self.w_max_i)
        return float(el.mean())

    # ---- 拓扑：全局随机边（grow_world 形态）----
    def grow_global(self, n):
        sp = self._spec
        gi = self.rng.integers(0, self.N, n)
        gj = self.rng.integers(0, self.N, n)
        okk = gi != gj
        gi, gj = gi[okk], gj[okk]
        if gi.size:
            nw = np.where(gi >= self.N_E, -self.w_new_i, sp.w_new_e).astype(np.float32)
            self._grow(gi, gj, nw)
            self.last_born = int(gi.size)
            return int(gi.size)
        self.last_born = 0
        return 0

    def in_degree(self):
        return np.bincount(self.post, minlength=self.N)
