# grow_snn3.py - 稀疏实现（性能优先，单套有序数组）
#
# 修正记录：
#   v1/v2：用稠密 N×N（1e6 格）存 E≈8000 条的边 -> 什么操作都碰 1e6 格，慢百倍
#   v3-初：维护了「原始数组 + 排序视图」两套 -> 改排序视图写不回原始数组（实用 bug）
#   v3-本：只有一套数组，始终按 key 排序。改 w 就是直接改。
#
# 复杂度：一切都只碰 E 长度。加边时才 O(E log E) 重排（每 T_S 步一次）。
# 稳态用全局增益（O(1)），不缩放 E 个权重。

import sys
import time
import numpy as np

N           = 1000
TAU         = 20.0
DECAY_V     = float(np.exp(-1.0 / TAU))
V_TH        = 1.0

W_INIT      = 0.10
ETA_POT     = 0.05
DECAY_W     = 0.995
W_MAX       = 1.0
THETA_ABS   = 0.012

T_S         = 20
T_G         = 2000
P_GLOBAL    = 6

TARGET_RATE = 0.02
K_HOME      = 0.25
OUT_RATE    = 0.004
MAX_ACTIVE  = 64


class Net:
    __slots__ = ('pre', 'post', 'w', 'key')

    def __init__(self, E0, rng):
        pre = rng.integers(0, N, E0).astype(np.int32)
        post = rng.integers(0, N, E0).astype(np.int32)
        ok = pre != post
        pre, post = pre[ok], post[ok]
        w = (W_INIT * (1.0 + 0.15 * rng.standard_normal(len(pre)))).astype(np.float32)
        np.clip(w, 0.001, None, out=w)
        self._set(pre, post, w)

    def _set(self, pre, post, w):
        key = post.astype(np.int64) * N + pre
        order = np.argsort(key, kind='stable')
        self.pre = np.ascontiguousarray(pre[order])
        self.post = np.ascontiguousarray(post[order])
        self.w = np.ascontiguousarray(w[order], dtype=np.float32)
        self.key = np.ascontiguousarray(key[order])

    def size(self):
        return self.key.size

    def find(self, qkey):
        """返回 (found_mask, idx)。idx 已 clip。"""
        if self.key.size == 0:
            return np.zeros(qkey.size, bool), np.zeros(qkey.size, np.int64)
        idx = np.searchsorted(self.key, qkey)
        np.clip(idx, 0, self.key.size - 1, out=idx)
        return self.key[idx] == qkey, idx

    def add(self, pre, post, wv):
        pre = np.concatenate([self.pre, pre.astype(np.int32)])
        post = np.concatenate([self.post, post.astype(np.int32)])
        w = np.concatenate([self.w, np.full(pre.size - self.pre.size, wv, np.float32)])
        self._set(pre, post, w)

    def prune(self, thresh):
        m = self.w >= thresh
        if m.all():
            return 0
        n = self.size() - int(m.sum())
        self.pre = self.pre[m]
        self.post = self.post[m]
        self.w = self.w[m]
        self.key = self.key[m]
        return n


def run(mode, steps, seed=0, E0=8000, log_every=10000):
    rng = np.random.default_rng(seed)
    net = Net(E0, rng)

    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    gain = 1.0
    use_global = mode in ('B', 'C')
    use_anneal = mode == 'C'

    t0 = time.perf_counter()
    sp_acc = 0
    prev_key = None

    for t in range(steps):
        pre, post, w = net.pre, net.post, net.w
        I = np.bincount(post, weights=w * s[pre], minlength=N).astype(np.float32)
        x = (rng.random(N) < OUT_RATE).astype(np.float32)
        v = v * DECAY_V + gain * I + x
        s = (v > V_TH).astype(np.float32)
        v = np.where(s > 0, 0.0, v)
        sp_acc += int(s.sum())

        if t > 0 and t % T_S == 0:
            frac = t / steps
            temp = (1.0 - 0.93 * frac) if use_anneal else 1.0

            # 稳态：全局增益（O(1)）
            rate = sp_acc / (T_S * N)
            sp_acc = 0
            if rate > 1e-9:
                gain *= (TARGET_RATE / rate) ** K_HOME
            else:
                gain *= 1.3
            gain = float(np.clip(gain, 1e-3, 1e3))

            # 衰减 + 剪除
            net.w *= (1.0 - (1.0 - DECAY_W) * temp)
            net.prune(THETA_ABS)

            # 局部：共激活
            active = np.nonzero(s > 0)[0]
            if active.size > MAX_ACTIVE:
                active = rng.choice(active, MAX_ACTIVE, replace=False)
            if active.size >= 2:
                na = active.size
                q_pre = np.repeat(active, na)
                q_post = np.tile(active, na)
                ok = q_pre != q_post
                q_pre, q_post = q_pre[ok], q_post[ok]
                qkey = q_post.astype(np.int64) * N + q_pre
                found, idx = net.find(qkey)

                if found.any():
                    fi = idx[found]
                    net.w[fi] = np.minimum(
                        net.w[fi] + ETA_POT * temp * (W_MAX - net.w[fi]), W_MAX)

                np_, nq_ = q_pre[~found], q_post[~found]
                if np_.size:
                    pick = rng.random(np_.size) < (0.15 * temp)
                    if pick.any():
                        net.add(np_[pick], nq_[pick], W_INIT)

            # 全局随机新建
            if use_global and t % T_G == 0:
                gi = rng.integers(0, N, P_GLOBAL)
                gj = rng.integers(0, N, P_GLOBAL)
                ok2 = gi != gj
                net.add(gi[ok2], gj[ok2], W_INIT)

        if log_every and t > 0 and t % log_every == 0:
            E = net.size()
            if prev_key is None:
                tov = float('nan')
            else:
                inter = np.intersect1d(prev_key, net.key, assume_unique=True).size
                union = prev_key.size + net.key.size - inter
                tov = 1.0 - (inter / union if union else 1.0)
            prev_key = net.key.copy()
            deg = np.bincount(net.post, minlength=N)
            el = time.perf_counter() - t0
            print('  t=%7d  E=%6d  tov=%.3f  rate=%5.2f%%  gain=%7.3f  deg(m/mx/nz)=%.1f/%d/%d  [%.1fs %.0f sps]' % (
                t, E, tov, s.sum() / N * 100, gain, deg.mean(), deg.max(),
                int((deg > 0).sum()), el, (t + 1) / el), flush=True)

    return net, time.perf_counter() - t0


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    mode = sys.argv[2] if len(sys.argv) > 2 else 'C'
    print('=' * 108)
    print('grow_snn3  稀疏  N=%d  steps=%d  mode=%s  目标率=%.0f%%' % (N, steps, mode, TARGET_RATE * 100))
    print('=' * 108)
    net, el = run(mode, steps)
    print('\n完成 %.1fs (%.0f 步/秒)  E=%d' % (el, steps / el, net.size()))
